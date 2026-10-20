"""CPU lifecycle checks for profiling recovery and evaluated-checkpoint cleanup."""
import copy
import importlib.util
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest
import torch

from mri_vla_jepa.io import read_json, write_json
from mri_vla_jepa.train_config import from_dict
from mri_vla_jepa.training import _scores


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("unfrozen_runner_test", ROOT / "scripts/run_unfrozen_ablation.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class Child:
    def __init__(self, pid=101, returncode=None):
        self.pid, self.returncode, self.terminated = pid, returncode, False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated, self.returncode = True, -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def test_checked_profile_is_recorded_and_stopped_on_monitor_error(tmp_path, monkeypatch):
    child = Child()
    command = [sys.executable, str(runner.ROOT / "scripts/profile_capacity_phase.py"),
               "--output", str(tmp_path / "result.json")]
    monkeypatch.setattr(runner, "spawn", lambda *args: child)
    monkeypatch.setattr(runner, "process_info", lambda pid: {
        "pid": pid, "start_ticks": "start", "command": command})
    snapshots = []
    original = runner.durable_json

    def record(path, value):
        snapshots.append(copy.deepcopy(value))
        original(path, value)

    monkeypatch.setattr(runner, "durable_json", record)
    monkeypatch.setattr(runner, "gpu_memory_used", lambda: {"GPU-A": 200})
    with pytest.raises(RuntimeError, match="memory/time budget"):
        runner.checked_child(command, tmp_path / "worker.log", {"uuid": "GPU-A", "budget_mib": 100},
                             state={}, output=tmp_path, stage="preflight")
    assert child.terminated
    identity = snapshots[0]["active_preflight"]["children"][0]
    assert identity["pid"] == 101 and identity["start_ticks"] == "start"
    assert identity["command"] == command
    assert "active_preflight" not in snapshots[-1]


def test_orphan_recovery_checks_every_profile_pid_and_preserves_other_processes(tmp_path, monkeypatch):
    command = [sys.executable, str(runner.ROOT / "scripts/profile_capacity_phase.py"), "--output", "a.json"]
    records = [{"pid": pid, "start_ticks": str(pid), "command": command} for pid in (101, 102)]
    processes = {record["pid"]: dict(record) for record in records}
    processes[999] = {"pid": 999, "start_ticks": "999", "command": ["train-vla-jepa"]}
    monkeypatch.setattr(runner, "process_info", lambda pid: processes.get(pid))
    killed = []

    def kill(pid, signum):
        assert signum == signal.SIGTERM
        killed.append(pid)
        processes.pop(pid)

    monkeypatch.setattr(runner.os, "kill", kill)
    state = {"active_preflight": {"children": records}}
    runner.recover_preflight_children(state, tmp_path)
    assert killed == [101, 102] and 999 in processes
    assert "active_preflight" not in state


def test_orphan_identity_mismatch_is_not_signalled(tmp_path, monkeypatch):
    state = {"active_preflight": {"children": [{"pid": 101, "start_ticks": "a",
                                              "command": ["wrong"]}]}}
    monkeypatch.setattr(runner, "process_info", lambda pid: {
        "pid": pid, "start_ticks": "a", "command": ["training-worker"]})
    monkeypatch.setattr(runner.os, "kill", lambda *args: pytest.fail("Unexpected process signal"))
    with pytest.raises(RuntimeError, match="identity mismatch"):
        runner.recover_preflight_children(state, tmp_path)


def test_empty_preflight_spawn_journal_can_resume(tmp_path):
    state = {"active_preflight": {"children": []}}
    runner.recover_preflight_children(state, tmp_path)
    assert "active_preflight" not in state


def preflight_state(tmp_path):
    job = {"seed": 17, "variant": "warmup_cosine", "config": str(tmp_path / "config.yaml"),
           "config_digest": "cfg", "manifest": str(tmp_path / "manifest.json"),
           "manifest_sha256": "manifest"}
    profile = {"status": "passed", "source_identity": "source", "config_digest": "cfg",
               "manifest_sha256": "manifest", "reservation_mib": 100}
    return {"gpu": {"uuid": "GPU-A", "budget_mib": 1000}, "jobs": [job],
            "profiles": {"warmup_cosine": profile}, "limits": {}, "source_identity": "source"}


def test_cached_profiles_migrate_to_memory_admission_without_worker_count_limits(tmp_path, monkeypatch):
    state = preflight_state(tmp_path)
    state["limits"] = {"warmup_cosine": 2}
    monkeypatch.setattr(runner, "spawn", lambda *args: pytest.fail("Valid memory profiles do not need pair benchmarks"))
    runner.preflight(state, tmp_path)
    assert "limits" not in state
    assert state["admission_policy"]["task_count_cap"] is None
    assert state["admission_policy"]["batch_size"] == 16


def test_preflight_uses_largest_lifecycle_peak_and_preserves_batch16(tmp_path, monkeypatch):
    state = preflight_state(tmp_path)
    state["profiles"] = {}
    state["gpu"]["budget_mib"] = 10000
    def measure(command, log, gpu, **kwargs):
        target = Path(command[command.index("--output") + 1])
        write_json(target, {"status": "passed", "config": state["jobs"][0]["config"],
            "batch_size": 16, "peak_reserved_mib": 3000, "training_peak_reserved_mib": 2000,
            "inference_peak_reserved_mib": 3000, "diagnostic_peak_reserved_mib": 4000})
        return 3500
    monkeypatch.setattr(runner, "checked_child", measure)
    runner.preflight(state, tmp_path)
    assert state["profiles"]["warmup_cosine"]["reservation_mib"] == 4512
    assert state["admission_policy"]["mixed_variants"]


def admission_state(reservations=(2000, 2500, 1500, 1800)):
    gpu = {"uuid": "GPU-A", "budget_mib": 9000, "name": "same-model", "total_mib": 10000}
    jobs = [{"name": f"variant{i}", "variant": f"variant{i}", "pid": 100 + i, "gpu_uuid": "GPU-A"}
            for i in range(len(reservations))]
    return {"gpu": gpu, "jobs": jobs, "profiles": {
        job["variant"]: {"reservation_mib": amount} for job, amount in zip(jobs, reservations)}}


def test_host_pid_mismatch_still_admits_mixed_workers_above_two_until_memory_full():
    state, active = admission_state(), {}
    # NVML host PIDs intentionally differ from the container's worker PIDs.
    used, observations = {"GPU-A": 100}, {}
    for index, job in enumerate(state["jobs"]):
        assert runner.choose_gpu(state, job, active, [state["gpu"]], used, observations) == state["gpu"]
        active[job["name"]] = None
        observations[90000 + index] = {"gpu_uuid": "GPU-A", "mib": 1200}
        used["GPU-A"] += 1200
    assert len(active) == 4
    extra = {"name": "extra", "variant": "extra", "pid": 999}
    state["profiles"]["extra"] = {"reservation_mib": 1500}
    state["jobs"].append(extra)
    assert runner.choose_gpu(state, extra, active, [state["gpu"]], used, observations) is None
    measured = state["gpu_memory_state"]["GPU-A"]
    assert measured["pid_mapping_available"] is False and measured["foreign_mib"] == 100


def test_actual_device_pressure_and_measured_foreign_memory_block_admission():
    state = admission_state((2000, 2000))
    first, second = state["jobs"]
    active = {first["name"]: None}
    assert runner.choose_gpu(state, second, active, [state["gpu"]], {"GPU-A": 8000}, {}) is None
    observations = {first["pid"]: {"gpu_uuid": "GPU-A", "mib": 1500}}
    assert runner.choose_gpu(state, second, active, [state["gpu"]], {"GPU-A": 8500}, observations) is None
    assert state["gpu_memory_state"]["GPU-A"]["pid_mapping_available"] is True
    assert state["gpu_memory_state"]["GPU-A"]["foreign_mib"] == 7000


def test_runtime_reservation_expands_and_exclusive_oom_retry_preserves_batch():
    state = admission_state((2000, 2000))
    first, second = state["jobs"]
    first["exclusive_retry"] = True
    active = {first["name"]: None}
    observed = {first["pid"]: {"gpu_uuid": "GPU-A", "mib": 3000}}
    assert runner.choose_gpu(state, second, active, [state["gpu"]], {"GPU-A": 3000}, observed) is None
    assert state["profiles"][first["variant"]]["reservation_mib"] == 3512


def test_gpu0_joins_only_after_all_old_jobs_naturally_finish(tmp_path, monkeypatch):
    old = tmp_path / "old"
    old.mkdir()
    state = admission_state()
    second = {**state["gpu"], "uuid": "GPU-B"}
    state.update(gpus=[state["gpu"], second], deferred_gpu_uuids=["GPU-B"], old_suite=str(old))
    monkeypatch.setattr(runner, "same_process", lambda record: None)
    write_json(old / "matrix.json", {"jobs": [{"status": "evaluation_pending"}, {"status": "paused"}]})
    assert runner.eligible_gpus(state) == [state["gpu"]]
    write_json(old / "matrix.json", {"jobs": [{"status": "completed"}, {"status": "paused"}]})
    assert runner.eligible_gpus(state) == state["gpus"]
    assert runner.gpu_profiles(state, second) is state["profiles"]


def test_gpu_choice_uses_both_budgets_and_measures_different_hardware_separately():
    state = admission_state((7000, 2000))
    second = {**state["gpu"], "uuid": "GPU-B"}
    state["gpus"] = [state["gpu"], second]
    first, next_job = state["jobs"]
    active = {first["name"]: None}
    chosen = runner.choose_gpu(state, next_job, active, state["gpus"], {"GPU-A": 6500, "GPU-B": 0}, {})
    assert chosen["uuid"] == "GPU-B"
    second["name"] = "different-model"
    assert runner.gpu_profiles(state, second) == {}


def evaluation_setup(tmp_path, monkeypatch, *, direct=False):
    output, run = tmp_path / "suite", tmp_path / "suite/run"
    run.mkdir(parents=True)
    codec = tmp_path / "codec.pt"
    codec.write_bytes(b"protected pretrained codec")
    for kind in ("best", "last"):
        (run / f"{kind}.pt").write_bytes(kind.encode())
    manifest = output / "manifest.json"
    patients = [{"patient_key": name, "split": split, "target": {"pcr": 1},
                 "visits": [{"available_at": 0}, {"available_at": 1}, None, None]}
                for name, split in (("Val-A", "val"), ("Test-A", "test"))]
    write_json(manifest, {"patients": patients})
    report = {"completed": True, "completed_epochs": 10, "best_epoch": 7, "best_selection_score": 1.}
    write_json(run / "train_report.json", report)
    write_json(run / "diagnostics.json", [{"epoch": 5}, {"epoch": 10}])
    job = {"name": "variant_seed17", "output": str(run), "config": str(output / "config.yaml"),
           "config_digest": "cfg", "manifest": str(manifest)}
    write_json(output / "matrix.json", {"codec": str(codec), "jobs": [job]})
    configuration = {"training_unit": "epochs", "accumulation": 1, "diagnostics_every_epochs": 5,
                     "device": "cpu", "precision": "fp32"}
    if direct:
        configuration.update(landmarks="t0", jepa_weight=0., rollout_weight=0., flow_weight=0.,
                             reconstruction_weight=0., variance_weight=0.,
                             model={"enable_world_model": False, "pcr_readout": "direct", "enable_flow": False})
    cfg = from_dict(configuration)
    monkeypatch.setattr(runner, "validate_identity", lambda state: None)
    monkeypatch.setattr(runner, "load_config", lambda path: cfg)
    monkeypatch.setattr(runner, "load_queue", lambda: SimpleNamespace(validate_completed_run=lambda *args: report))
    monkeypatch.setattr(runner, "_source_snapshot", lambda: {"source": "fixed"})
    monkeypatch.setattr(runner, "load_trained", lambda checkpoint, *args: (object(), cfg, {
        "source_snapshot": {"source": "fixed"}, "normalization": {},
        "epoch_state": {"completed_epochs": 7 if Path(checkpoint).name == "best.pt" else 10}}))
    monkeypatch.setattr(runner, "RawMRIStore", lambda *args: object())
    monkeypatch.setattr(runner.gc, "collect", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    calls = []

    def evaluate(checkpoint, manifest_path, target, *, split, **kwargs):
        calls.append((Path(checkpoint).name, split))
        stages = (0,) if direct else (0, 1)
        rows = [{"patient_key": patient["patient_key"], "landmark": stage, "pcr": 1,
                 "pcr_probability": .8} for patient in patients if patient["split"] == split for stage in stages]
        result = {"split": split, "landmarks": cfg.landmarks, "rows": rows,
                  **_scores(rows, cfg.landmarks, selection_metric="f1")}
        write_json(target, result)
        return result

    monkeypatch.setattr(runner, "evaluate", evaluate)
    monkeypatch.setattr(runner, "evaluate_world", lambda model, store, cfg, *, split, batch_size: {
        "split": split, "patients": 1, "legal_prefixes": 2})
    return output, run, codec, calls


def test_evaluation_receipt_diagnostics_cleanup_and_weightless_resume(tmp_path, monkeypatch):
    output, run, codec, calls = evaluation_setup(tmp_path, monkeypatch)
    runner.evaluate_job(output, "variant_seed17")
    assert calls == [("best.pt", "val"), ("best.pt", "test"), ("last.pt", "val")]
    assert not (run / "best.pt").exists() and not (run / "last.pt").exists() and codec.exists()
    assert read_json(run / "evaluation_done.json")["completed"]
    receipt = read_json(run / "evaluation_complete.json")
    for name in ("best_val", "best_test", "last_val", "world_best_val", "world_best_test", "world_last_val"):
        artifact = read_json(receipt["outputs"][name]["path"])
        kind = "last" if "last" in name else "best"
        assert artifact["checkpoint_epoch"] == (10 if kind == "last" else 7)
        assert artifact["checkpoint_kind"] == kind
        assert receipt["outputs"][name]["sha256"]
    diagnostics = read_json(receipt["outputs"]["diagnostics"]["path"])
    assert diagnostics["reports"] == [{"epoch": 5}, {"epoch": 10}]
    (run / "evaluation_done.json").unlink()
    monkeypatch.setattr(runner, "load_queue", lambda: pytest.fail("Weights must not be reopened"))
    runner.evaluate_job(output, "variant_seed17")
    assert read_json(run / "evaluation_done.json")["completed"] and len(calls) == 3


def test_missing_scheduled_diagnostics_prevents_weight_deletion(tmp_path, monkeypatch):
    output, run, _, calls = evaluation_setup(tmp_path, monkeypatch)
    write_json(run / "diagnostics.json", [{"epoch": 5}])
    with pytest.raises(ValueError, match="diagnostics do not cover"):
        runner.evaluate_job(output, "variant_seed17")
    assert (run / "best.pt").exists() and (run / "last.pt").exists() and not calls


def test_evaluation_failure_never_deletes_checkpoints(tmp_path, monkeypatch):
    output, run, _, _ = evaluation_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "evaluate_world", lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("Evaluation failed")))
    with pytest.raises(RuntimeError, match="Evaluation failed"):
        runner.evaluate_job(output, "variant_seed17")
    assert (run / "best.pt").exists() and (run / "last.pt").exists()
    assert not (run / "evaluation_complete.json").exists()


def test_direct_world_not_applicable_is_receipted_and_allows_checkpoint_cleanup(tmp_path, monkeypatch):
    output, run, codec, calls = evaluation_setup(tmp_path, monkeypatch, direct=True)
    monkeypatch.setattr(runner, "load_trained", lambda *args: pytest.fail("Direct model has no world branch to load"))
    monkeypatch.setattr(runner, "evaluate_world", lambda *args, **kwargs: pytest.fail("No world diagnostics for direct model"))
    runner.evaluate_job(output, "variant_seed17")
    receipt = read_json(run / "evaluation_complete.json")
    for label in ("world_best_val", "world_best_test", "world_last_val"):
        artifact = read_json(receipt["outputs"][label]["path"])
        assert artifact["status"] == "not_applicable" and artifact["world_head_trained"] is False
        assert artifact["reason"] and artifact["landmarks"] == "t0"
        assert receipt["outputs"][label]["applicable"] is False
    assert len(calls) == 3 and codec.exists()
    assert not (run / "best.pt").exists() and not (run / "last.pt").exists()
