"""Exercise the matrix controller through real state files and fake workers."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from mri_vla_jepa.io import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("matrix_controller_test", ROOT / "scripts/run_capacity_phase_matrix.py")
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


class Worker:
    def __init__(self, pid, returncode=0):
        self.pid, self.returncode = pid, returncode

    def poll(self):
        return self.returncode


class Harness:
    def __init__(self, tmp_path, monkeypatch, *, oom=None, bad_f1=False):
        self.output = tmp_path / "matrix"
        self.output.mkdir()
        self.commands, self.validations, self.aggregates = [], [], []
        self.processes, self.oom, self.bad_f1 = {}, oom, bad_f1
        self.failures, self.sleeps = 0, 0
        self.failure_text, self.failure_code = "CUDA out of memory\n", 1
        self.report = {"completed": True, "best_selection_score": .6,
                       "best_validation_f1": .6, "best_validation_nll": .7,
                       "best_epoch": 3, "completed_epochs": 4, "stop_reason": "max_epochs"}
        self.job = {"name": "small_all_teacher_forcing_seed17", "size": "small", "phase": "all",
                    "mode": "teacher_forcing", "seed": 17, "config": str(tmp_path / "config.yaml"),
                    "config_digest": "config", "output": str(self.output / "run"),
                    "status": "queued", "retries": 0}
        self.state = {"source_identity": "source", "runtime_hashes": {"a": "b"},
                      "manifest": str(tmp_path / "manifest.json"), "manifest_sha256": "digest",
                      "codec_sha256": "digest", "data_signature": "data", "jobs": [self.job],
                      "profiles": {controller.profile_key(self.job): {"reservation_mib": 100}},
                      "limits": {controller.benchmark_key(self.job): 2},
                      "gpus": [{"index": 0, "uuid": "GPU-A", "budget_mib": 1000},
                               {"index": 1, "uuid": "GPU-B", "budget_mib": 1000}]}
        self.original_aggregate = controller.aggregate
        monkeypatch.setattr(controller, "load_queue", lambda: SimpleNamespace(
            process_info=self.process_info, validate_completed_run=self.validate))
        monkeypatch.setattr(controller, "load_config", lambda path: SimpleNamespace(
            digest="config", encoder_checkpoint=tmp_path / "codec.pt"))
        monkeypatch.setattr(controller, "RawMRIStore", lambda path: SimpleNamespace(signature=lambda: "data"))
        monkeypatch.setattr(controller, "source_identity", lambda: "source")
        monkeypatch.setattr(controller, "runtime_hashes", lambda path: {"a": "b"})
        monkeypatch.setattr(controller, "digest", lambda path: "digest")
        monkeypatch.setattr(controller, "process_memory", lambda: {})
        monkeypatch.setattr(controller, "gpu_memory_used", lambda: {"GPU-A": 100, "GPU-B": 100})
        monkeypatch.setattr(controller, "spawn", self.spawn)
        monkeypatch.setattr(controller, "aggregate", lambda output, state: self.aggregates.append(copy.deepcopy(state)))
        monkeypatch.setattr(controller.time, "sleep", self.sleep)

    def sleep(self, seconds):
        self.sleeps += 1
        if self.sleeps > 100:
            pytest.fail("Controller stopped making lifecycle progress")

    def save(self):
        write_json(self.output / "matrix.json", self.state)

    def run(self):
        self.save()
        result = controller.work(self.output)
        self.state = read_json(self.output / "matrix.json")
        self.job = self.state["jobs"][0]
        return result

    def process_info(self, pid):
        return self.processes.get(pid)

    def validate(self, job, manifest):
        self.validations.append(job["name"])
        return read_json(Path(job["output"]) / "train_report.json")

    def complete_training(self, run):
        run.mkdir(exist_ok=True)
        write_json(run / "train_report.json", self.report)
        for name in ("best.pt", "last.pt"):
            (run / name).touch()

    def complete_evaluation(self, target):
        if target.name.startswith("world_"):
            result = {"mode": "world", "valid": True}
        else:
            result = {"selection_score": .5 if self.bad_f1 and target.name == "evaluation_best.json" else .6,
                      "metrics": {"f1": .6, "nll": .7},
                      "per_landmark": {"T0": {"f1": .5, "nll": .8}},
                      "patient_equal_metrics": {"f1": .6}}
        write_json(target, result)

    def spawn(self, command, log, gpu):
        pid = 1000 + len(self.commands)
        self.commands.append((list(command), gpu["uuid"]))
        self.processes[pid] = {"start_ticks": str(pid), "command": list(command)}
        training = "train-vla-jepa" in command
        kind = "training" if training else (
            "world" if "--world-checkpoint" in command else "pcr")
        log.parent.mkdir(exist_ok=True)
        if self.oom == kind and self.failures == 0:
            self.failures += 1
            with log.open("a") as handle:
                handle.write(self.failure_text)
            # A recoverable optimizer checkpoint may already have been written.
            if training:
                (log.parent / "last.pt").touch()
            return Worker(pid, self.failure_code)
        with log.open("a") as handle:
            handle.write("worker complete\n")
        output = Path(command[command.index("--output") + 1])
        if training:
            self.complete_training(output)
        else:
            self.complete_evaluation(output)
        return Worker(pid)


def test_training_then_all_four_evaluations_and_completion(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch)
    assert harness.run() == 0
    assert harness.state["status"] == harness.job["status"] == "completed"
    assert harness.job["evaluated"] == ["best", "last", "world_best", "world_last"]
    assert harness.validations == [harness.job["name"]]
    commands = [command for command, gpu in harness.commands]
    assert len(commands) == 5 and "train-vla-jepa" in commands[0]
    assert [Path(c[c.index("--output") + 1]).name for c in commands[1:]] == [
        "evaluation_best.json", "evaluation_last.json", "world_best.json", "world_last.json"]
    assert all(snapshot["jobs"][0]["status"] != "completed" for snapshot in harness.aggregates[:-1]
               if set(snapshot["jobs"][0].get("evaluated", [])) != {"best", "last", "world_best", "world_last"})
    assert all(not gpu["running"] for gpu in harness.state["gpus_live"])


def test_recomputed_best_f1_must_match_before_completion(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch, bad_f1=True)
    harness.save()
    with pytest.raises(ValueError, match="Recomputed best F1 differs"):
        controller.work(harness.output)
    persisted = read_json(harness.output / "matrix.json")
    assert persisted["status"] != "completed"
    assert persisted["jobs"][0]["status"] != "completed"
    assert len(harness.commands) == 2


@pytest.mark.parametrize("adopted_status", ["running", "evaluating"])
def test_live_pid_is_adopted_and_evaluation_resumes_without_duplicate_training(tmp_path, monkeypatch, adopted_status):
    harness = Harness(tmp_path, monkeypatch)
    run = Path(harness.job["output"])
    run.mkdir()
    harness.job.update(status=adopted_status, pid=77, start_ticks="old", gpu_uuid="GPU-A",
                       gpu_index=0, reservation_mib=100, benchmark_key=controller.benchmark_key(harness.job),
                       attempt_started=controller.time.time())
    if adopted_status == "evaluating":
        harness.complete_training(run)
        harness.job.update(report=harness.report, evaluation_kind="best")
    checked = []
    original = harness.process_info

    def process_info(pid):
        if pid != 77:
            return original(pid)
        checked.append(pid)
        if len(checked) < 3:
            return {"start_ticks": "old", "command": ["python", harness.job["config"], str(run)]}
        if adopted_status == "running":
            harness.complete_training(run)
        else:
            harness.complete_evaluation(run / "evaluation_best.json")
        return None

    monkeypatch.setattr(harness, "process_info", process_info)
    assert harness.run() == 0
    assert len(checked) == 3
    assert all("train-vla-jepa" not in command for command, gpu in harness.commands)
    assert len(harness.commands) == (4 if adopted_status == "running" else 3)
    assert harness.job["status"] == "completed"


def test_dead_or_reused_training_pid_resumes_saved_optimizer_checkpoint(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch)
    run = Path(harness.job["output"])
    run.mkdir()
    (run / "last.pt").touch()
    harness.job.update(status="running", pid=77, start_ticks="old")
    harness.processes[77] = {"start_ticks": "reused", "command": ["another", "process"]}
    assert harness.run() == 0
    assert "--resume" in harness.commands[0][0]
    assert harness.job["status"] == "completed"


def test_live_pid_command_identity_mismatch_is_rejected(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch)
    harness.job.update(status="running", pid=77, start_ticks="old")
    harness.processes[77] = {"start_ticks": "old", "command": ["unrelated"]}
    harness.save()
    with pytest.raises(RuntimeError, match="PID identity mismatch"):
        controller.work(harness.output)
    assert not harness.commands


@pytest.mark.parametrize("kind", ["training", "pcr", "world"])
def test_first_oom_gets_single_worker_retry_and_finishes(tmp_path, monkeypatch, kind):
    harness = Harness(tmp_path, monkeypatch, oom=kind)
    assert harness.run() == 0
    assert harness.job["retries"] == 1 and harness.job["status"] == "completed"
    assert harness.state["gpu_concurrency_caps"]["GPU-A"] == 1
    assert len(harness.commands) == 6
    if kind == "training":
        assert "--resume" in harness.commands[1][0]


def test_oom_retry_cannot_move_onto_busy_gpu_or_accept_additional_workers(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch)
    job = {**harness.job, "force_exclusive_retry": True}
    worker = {"reservation_mib": 100, "benchmark_key": controller.benchmark_key(job)}
    active = {"a": {**worker, "gpu_uuid": "GPU-A"}, "b": {**worker, "gpu_uuid": "GPU-B"}}
    choose = lambda candidate: controller.choose_gpu(candidate, active, harness.state["gpus"],
        harness.state["profiles"], harness.state["limits"], {"GPU-A": 1})
    assert choose(job) is None
    del active["b"]
    assert choose(job)["uuid"] == "GPU-B"
    # Once the retry starts there, subsequent jobs must wait for its completion.
    active["b"] = {**worker, "gpu_uuid": "GPU-B", "force_exclusive_retry": True}
    assert choose(harness.job) is None


@pytest.mark.parametrize("kind", ["training", "pcr", "world"])
def test_repeated_oom_while_alone_fails_without_infinite_retries(tmp_path, monkeypatch, kind):
    harness = Harness(tmp_path, monkeypatch, oom=kind)
    spawn = harness.spawn

    def repeated_oom(command, log, gpu):
        harness.failures = 0
        return spawn(command, log, gpu)

    monkeypatch.setattr(controller, "spawn", repeated_oom)
    assert harness.run() == 1
    assert harness.job["retries"] == 1
    assert harness.job["status"] == ("failed" if kind == "training" else "evaluation_failed")
    assert harness.state["status"] == "failed"
    assert len(harness.commands) == ({"training": 2, "pcr": 3, "world": 5}[kind])


@pytest.mark.parametrize("kind", ["training", "pcr", "world"])
def test_interrupted_worker_retries_saved_stage_and_finishes(tmp_path, monkeypatch, kind):
    harness = Harness(tmp_path, monkeypatch, oom=kind)
    harness.failure_text, harness.failure_code = "worker interrupted by SIGKILL\n", -9
    assert harness.run() == 0
    assert harness.job["retries"] == 1 and harness.job["status"] == "completed"
    assert len(harness.job["retry_events"]) == 1
    assert not harness.state["gpu_concurrency_caps"]
    if kind == "training":
        assert "--resume" in harness.commands[1][0]
    else:
        assert sum("train-vla-jepa" in command for command, gpu in harness.commands) == 1
    assert harness.job["evaluated"] == ["best", "last", "world_best", "world_last"]


@pytest.mark.parametrize("kind", ["training", "pcr", "world"])
def test_transient_failures_have_bounded_retries_including_controller_resume(tmp_path, monkeypatch, kind):
    harness = Harness(tmp_path, monkeypatch, oom=kind)
    harness.failure_text = "OSError: [Errno 5] Input/output error\n"
    spawn = harness.spawn

    def repeated_failure(command, log, gpu):
        harness.failures = 0
        return spawn(command, log, gpu)

    monkeypatch.setattr(controller, "spawn", repeated_failure)
    assert harness.run() == 1
    assert harness.job["retries"] == controller.MAX_WORKER_RETRIES
    assert harness.job["status"] == ("failed" if kind == "training" else "evaluation_failed")
    attempts = len(harness.commands)
    harness.sleeps = 0
    assert harness.run() == 1
    assert len(harness.commands) == attempts


@pytest.mark.parametrize("status", ["failed", "evaluation_failed"])
def test_resume_requeues_legacy_transient_failure_from_saved_stage(tmp_path, monkeypatch, status):
    harness = Harness(tmp_path, monkeypatch)
    run = Path(harness.job["output"])
    run.mkdir()
    harness.job.update(status=status, pid=None, retries=1)
    if status == "failed":
        (run / "last.pt").touch()
        log = run / "training.log"
    else:
        harness.complete_training(run)
        harness.job.update(report=harness.report, evaluation_kind="last", evaluated=["best"])
        log = run / "evaluation.log"
    # Only the failed attempt is relevant; an old deterministic error must not
    # prevent recovery of the current interrupted attempt.
    old = "ValueError: old attempt\n"
    log.write_text(old + "OSError: transient storage interruption\n")
    harness.job["log_offset"] = len(old.encode())
    assert harness.run() == 0
    assert harness.job["retries"] == 2 and harness.job["status"] == "completed"
    if status == "failed":
        assert "--resume" in harness.commands[0][0]
    else:
        first = harness.commands[0][0]
        assert Path(first[first.index("--output") + 1]).name == "evaluation_last.json"
        assert all("train-vla-jepa" not in command for command, gpu in harness.commands)


@pytest.mark.parametrize("message", [
    "ValueError: Resume config/data identity differs from the saved run\n",
    "RuntimeError: Error(s) in loading state_dict for RawVLAJEPA\n",
    "FloatingPointError: Non-finite VLA-JEPA loss\n",
])
def test_deterministic_worker_failure_surfaces_without_retry(tmp_path, monkeypatch, message):
    harness = Harness(tmp_path, monkeypatch, oom="training")
    harness.failure_text = message
    assert harness.run() == 1
    assert harness.job["status"] == "failed" and harness.job["retries"] == 0
    assert harness.job["retryable_failure"] is False
    assert len(harness.commands) == 1
    assert harness.run() == 1
    assert len(harness.commands) == 1


def test_one_deterministic_failure_does_not_stop_unrelated_jobs(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch, oom="training")
    harness.failure_text = "ValueError: Resume implementation identity differs from the saved run\n"
    other = {**harness.job, "name": "unrelated_seed43", "seed": 43,
             "output": str(harness.output / "other")}
    harness.state["jobs"].append(other)
    assert harness.run() == 1
    assert [job["status"] for job in harness.state["jobs"]] == ["failed", "completed"]


def test_distinct_heavy_workloads_seed_cards_then_fill_measured_homogeneous_capacity(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch)
    jobs = []
    for mode in ("teacher_forcing", "hybrid"):
        for seed in (17, 43):
            jobs.append({**harness.job, "name": f"large_{mode}_{seed}", "size": "large",
                         "mode": mode, "seed": seed})
    profiles = {controller.profile_key(job): {"reservation_mib": 300} for job in jobs}
    limits = {controller.benchmark_key(job): 2 for job in jobs}
    active, assignments = {}, []
    for job in controller.launch_candidates(jobs):
        gpu = controller.choose_gpu(job, active, harness.state["gpus"], profiles, limits, {})
        assert gpu is not None
        assignments.append((job["mode"], job["seed"], gpu["uuid"]))
        active[job["name"]] = {"gpu_uuid": gpu["uuid"], "reservation_mib": 300,
                               "benchmark_key": controller.benchmark_key(job)}
    assert assignments == [("teacher_forcing", 17, "GPU-A"), ("hybrid", 17, "GPU-B"),
                           ("teacher_forcing", 43, "GPU-A"), ("hybrid", 43, "GPU-B")]


def test_matrix_summary_completed_only_after_all_48_evaluated(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch)
    jobs = []
    for size in controller.SIZES:
        for phase in controller.PHASES:
            for mode in controller.MODES:
                for seed in controller.SEEDS:
                    job = {**harness.job, "name": f"{size}_{phase}_{mode}_{seed}", "size": size,
                           "phase": phase, "mode": mode, "seed": seed, "status": "completed"}
                    run = harness.output / job["name"]
                    run.mkdir()
                    job["output"] = str(run)
                    write_json(run / "train_report.json", harness.report)
                    harness.complete_evaluation(run / "evaluation_best.json")
                    harness.state["profiles"][controller.profile_key(job)] = {"reservation_mib": 100}
                    jobs.append(job)
    harness.state["jobs"] = jobs
    summary = harness.original_aggregate(harness.output, harness.state)
    assert summary["completed"] and summary["completed_runs"] == 48
    assert all(group["complete_seed_pair"] for group in summary["groups"])
    jobs[-1]["status"] = "evaluating"
    summary = harness.original_aggregate(harness.output, harness.state)
    assert not summary["completed"] and summary["completed_runs"] == 47
