"""Drive controller adoption, scheduling and evaluation through real state files."""
import copy
import importlib.util
import os
from pathlib import Path
import pytest

from mri_vla_jepa.io import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("unfrozen_lifecycle", ROOT / "scripts/run_unfrozen_ablation.py")
controller = importlib.util.module_from_spec(SPEC)
# The script imports its neighbouring frozen-runtime helpers.
import sys
sys.path.insert(0, str(ROOT / "scripts"))
SPEC.loader.exec_module(controller)


class Child:
    def __init__(self, pid):
        self.pid, self.returncode = pid, 0

    def poll(self):
        return self.returncode


class Harness:
    def __init__(self, tmp_path, monkeypatch, variants=("dropout", "dropout", "ema")):
        self.output = tmp_path / "suite"
        self.output.mkdir()
        self.commands, self.reports = [], []
        self.processes, self.sleeps = {}, 0
        self.state = {"gpu": {"uuid": "fake-gpu", "budget_mib": 1000}, "profiles": {},
                      "limits": {}, "old_suite": None, "jobs": []}
        for i, variant in enumerate(variants):
            name = f"{variant}_seed{i}"
            self.state["jobs"].append({"name": name, "variant": variant, "seed": i,
                "status": "queued", "config": str(tmp_path / f"{name}.yaml"),
                "manifest": str(tmp_path / "manifest.json"), "output": str(self.output / "runs" / name)})
            self.state["profiles"][variant] = {"reservation_mib": 100}
            self.state["limits"][variant] = 2
        monkeypatch.setattr(controller, "GPU_LOCK_ROOT", tmp_path / "gpu_locks")
        monkeypatch.setattr(controller, "validate_identity", lambda state: None)
        monkeypatch.setattr(controller, "old_gpu_busy", lambda *args: False)
        monkeypatch.setattr(controller, "preflight", lambda *args: None)
        monkeypatch.setattr(controller, "storage_probe", lambda *args: None)
        monkeypatch.setattr(controller, "gpu_memory_used", lambda: {"fake-gpu": 0})
        monkeypatch.setattr(controller, "process_memory", lambda: {})
        monkeypatch.setattr(controller, "process_info", lambda pid: (
            {"pid": pid, "start_ticks": "controller", "command": ["controller"]}
            if pid == os.getpid() else self.processes.get(pid)))
        monkeypatch.setattr(controller, "live_processes", lambda: list(self.processes.values()))
        monkeypatch.setattr(controller, "spawn", self.spawn)
        monkeypatch.setattr(controller, "write_suite_reports", self.report)
        monkeypatch.setattr(controller.time, "sleep", self.sleep)

    def report(self, output, jobs, *, suite_complete):
        self.reports.append((copy.deepcopy(jobs), suite_complete))

    def sleep(self, seconds):
        self.sleeps += 1
        assert self.sleeps < 20, "Controller stopped making progress"

    def training_done(self, job):
        run = Path(job["output"])
        run.mkdir(parents=True, exist_ok=True)
        write_json(run / "train_report.json", {"completed": True, "best_epoch": 3})
        (run / "best.pt").write_bytes(b"best")
        (run / "last.pt").write_bytes(b"last")

    def spawn(self, command, log, gpu):
        self.commands.append(list(command))
        if "train-vla-jepa" in command:
            job = next(j for j in self.state["jobs"] if j["config"] in command)
            self.training_done(job)
        else:
            name = command[command.index("--evaluate-job") + 1]
            job = next(j for j in self.state["jobs"] if j["name"] == name)
            run = Path(job["output"])
            # The evaluation helper is separately integration-tested for receipts.
            write_json(run / "evaluation_done.json", {"completed": True})
            (run / "best.pt").unlink()
            (run / "last.pt").unlink()
        pid = 1000 + len(self.commands)
        self.processes[pid] = {"pid": pid, "start_ticks": str(pid), "command": command}
        return Child(pid)

    def run(self):
        write_json(self.output / "matrix.json", self.state)
        result = controller.worker(self.output)
        self.state = read_json(self.output / "matrix.json")
        return result


def test_complete_lifecycle_prioritizes_evaluation_and_releases_weights(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch)
    assert harness.run() == 0
    assert harness.state["status"] == "completed"
    assert all(job["status"] == "completed" for job in harness.state["jobs"])
    assert len(harness.commands) == 6
    # Mixed variants fill the measured budget together, regardless of stale limit=2.
    ema_training = next(i for i, c in enumerate(harness.commands) if "train-vla-jepa" in c and "ema_seed2" in " ".join(c))
    assert ema_training == 2
    assert all("--evaluate-job" in command for command in harness.commands[3:])
    assert len(harness.reports[0][0]) == 3
    assert not list((harness.output / "runs").rglob("*.pt"))
    assert harness.reports[-1][1] is True
    assert all(not complete for _, complete in harness.reports[:-1])


def test_restart_adopts_live_worker_before_device_busy_check(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch, variants=("dropout",))
    job = harness.state["jobs"][0]
    job.update(status="running", pid=99, start_ticks="original")
    harness.processes[99] = {"pid": 99, "start_ticks": "original", "command": [job["config"], job["name"]]}
    def forbidden(*args):
        raise AssertionError("A surviving training worker must be adopted before GPU waiting/profiling")
    monkeypatch.setattr(controller, "old_gpu_busy", forbidden)
    monkeypatch.setattr(controller, "preflight", forbidden)
    def finish_original(seconds):
        harness.sleep(seconds)
        harness.training_done(job)
        harness.processes.pop(99, None)
    monkeypatch.setattr(controller.time, "sleep", finish_original)
    assert harness.run() == 0
    assert len(harness.commands) == 1
    assert "--evaluate-job" in harness.commands[0]


def test_restart_recovers_spawn_before_journal_commit_without_duplicate_training(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch, variants=("dropout",))
    job = harness.state["jobs"][0]
    command = ["python", str(controller.ROOT / "train.py"), "train-vla-jepa",
               "--config", job["config"], "--manifest", job["manifest"], "--output", job["output"]]
    harness.processes[99] = {"pid": 99, "start_ticks": "uncommitted", "command": command}
    def forbidden(*args):
        raise AssertionError("An unjournaled worker must be recovered before waiting or profiling")
    monkeypatch.setattr(controller, "old_gpu_busy", forbidden)
    monkeypatch.setattr(controller, "preflight", forbidden)
    def finish_original(seconds):
        harness.sleep(seconds)
        if 99 in harness.processes:
            saved = read_json(harness.output / "matrix.json")["jobs"][0]
            assert saved["pid"] == 99 and saved["start_ticks"] == "uncommitted"
            harness.training_done(job)
            harness.processes.pop(99)
    monkeypatch.setattr(controller.time, "sleep", finish_original)
    assert harness.run() == 0
    assert len(harness.commands) == 1 and "--evaluate-job" in harness.commands[0]
    assert harness.state["worker_recovery"]["recovered_jobs"] == [job["name"]]


@pytest.mark.parametrize("live", [False, True])
def test_controller_failure_restores_only_when_no_workers_survive(tmp_path, monkeypatch, live):
    harness = Harness(tmp_path, monkeypatch, variants=("dropout",))
    state = harness.state
    state["old_suite"] = str(tmp_path / "old")
    journal = harness.output / "old_gpu_allocation.json"
    journal.write_text("{}")
    restored = []
    monkeypatch.setattr(controller, "set_old_gpus", lambda *a, **k: restored.append(k.get("restoring")))
    if live:
        state["jobs"][0].update(pid=99, start_ticks="active")
        harness.processes[99] = {"pid": 99, "start_ticks": "active"}
    with pytest.raises(RuntimeError, match="profile failed"):
        with controller.allocation_lock(tmp_path / "allocation.lock", harness.output, state, journal):
            raise RuntimeError("profile failed")
    assert state["status"] == "controller_failed"
    assert "profile failed" in read_json(harness.output / "matrix.json")["controller_error"]
    assert restored == ([] if live else [True])
