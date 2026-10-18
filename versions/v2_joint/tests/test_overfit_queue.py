"""Immutable plans, conservative admission, exact ownership and restart safety."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("overfit_queue", Path(__file__).resolve().parents[1] /
                                           "scripts" / "run_overfit_suite.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


@pytest.fixture
def plan(tmp_path):
    records = []
    for number in range(2):
        config = tmp_path / f"config{number}.yaml"
        config.write_text(f"train:\n  seed: {number}\n")
        manifest = tmp_path / f"manifest{number}.json"
        manifest.write_text("{}")
        records.append({"name": f"run{number}", "config": config.name, "manifest": manifest.name,
                        "output": f"result{number}", "reservation_mib": 10240})
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"description": "CPU test", "runs": records}))
    return path


def suite(plan, output=None):
    return queue.prepare_suite(queue.PROJECT, output or plan.parent / "suite", plan)


def artifacts(output, state="completed", selection=0.5):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    queue.atomic_json(output / "completion.json", {"state": state, "updates": 0, "best_update": 0})
    queue.atomic_json(output / "metrics_best.json", {"selection": selection,
                                                   "selection_name": "patient_equal_pcr_nll"})
    for name in ("best.pt", "last.pt"):
        (output / name).write_bytes(b"test fixture only")


def recorded_run(plan):
    item = suite(plan)["runs"][0]
    item.update(state="running", attempts=[{"attempt": 1, "resume": False, "launch_complete": True}])
    return item


def args(plan):
    return SimpleNamespace(plan=str(plan), output=str(plan.parent / "suite"),
                           max_concurrent=3, gpu_memory_budget_mib=31232, poll_seconds=3)


def test_plan_preserves_distinct_manifests_config_text_and_identity(plan):
    status = suite(plan)
    assert len({item["manifest"] for item in status["runs"]}) == 2
    assert all(Path(item["output"]).is_absolute() for item in status["runs"])
    for item in status["runs"]:
        assert Path(item["config"]).read_bytes() == Path(item["source_config"]).read_bytes()
    assert suite(plan) == status
    assert "manifest_files" in status["protocol"]
    assert "sha" not in json.dumps(status).lower()


def test_optional_research_metadata_survives_and_is_identity_locked(plan):
    data = json.loads(plan.read_text())
    metadata = {"algorithm": "patient_equal_detach_consistency", "profile": "dynamic", "seed": 17}
    data["runs"][0].update(metadata)
    plan.write_text(json.dumps(data))
    status = suite(plan)
    assert {field: status["runs"][0][field] for field in metadata} == metadata
    assert suite(plan) == status
    data["runs"][0]["seed"] = 43
    plan.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identity changed"):
        suite(plan)


def test_unknown_run_metadata_rejected(plan):
    data = json.loads(plan.read_text())
    data["runs"][0]["typo_algorithm"] = "combined"
    plan.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Each run requires"):
        queue.load_plan(plan)


@pytest.mark.parametrize("change", ["config", "manifest", "plan"])
def test_changes_to_locked_inputs_rejected(plan, change):
    status = suite(plan)
    if change == "config":
        Path(status["runs"][0]["source_config"]).write_text("train:\n  seed: 999\n")
    elif change == "manifest":
        Path(status["runs"][0]["manifest"]).write_text('{"modified": true}')
    else:
        data = json.loads(plan.read_text())
        data["runs"][0]["reservation_mib"] += 1
        plan.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="changed"):
        suite(plan)


def test_existing_untracked_output_is_not_overwritten(plan):
    output = plan.parent / "result1"
    output.mkdir()
    (output / "last.pt").write_bytes(b"existing")
    with pytest.raises(ValueError, match="already exists"):
        suite(plan)
    assert not (plan.parent / "suite" / "status.json").exists()


@pytest.mark.parametrize("change", ["duplicate_name", "unsafe_name", "reservation", "overlap", "missing_manifest"])
def test_malformed_plan_rejected_before_run_launch(plan, change):
    data = json.loads(plan.read_text())
    runs = data["runs"]
    if change == "duplicate_name":
        runs[1]["name"] = runs[0]["name"]
    elif change == "unsafe_name":
        runs[0]["name"] = "../unsafe"
    elif change == "reservation":
        runs[0]["reservation_mib"] = 0
    elif change == "overlap":
        runs[1]["output"] = runs[0]["output"] + "/nested"
    else:
        runs[0]["manifest"] = "missing.json"
    plan.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        queue.load_plan(plan)


def test_reserved_memory_prevents_oversubscription_before_cuda_allocates():
    item = {"reservation_mib": 10240}
    snapshot = {"total_mib": 32607, "used_mib": 0, "free_mib": 32607, "processes": {}}
    assert queue.can_launch(item, [item, item], snapshot)
    assert queue.reserved_usage([item, item, item], snapshot) == 30720
    assert not queue.can_launch(item, [item, item, item], snapshot)
    assert not queue.can_launch({"reservation_mib": 32000}, [], snapshot)


def test_external_usage_and_actual_peak_reduce_admission():
    running = [{"reservation_mib": 10240, "trainer_pid": 11},
               {"reservation_mib": 8192, "trainer_pid": 12}]
    snapshot = {"total_mib": 32607, "used_mib": 22000, "free_mib": 10607,
                "processes": {11: 12000, 12: 8000, 99: 2000}}
    assert queue.reserved_usage(running, snapshot) == 22192
    assert not queue.can_launch({"reservation_mib": 10240}, running, snapshot)
    assert queue.can_launch({"reservation_mib": 8192}, running, snapshot)
    snapshot["free_mib"] = 7000
    assert not queue.can_launch({"reservation_mib": 8192}, running, snapshot)


def test_exact_recorded_process_birth_and_command_are_required():
    info = queue.process_info(os.getpid())
    assert queue.find_process(info["cmdline"], info["cwd"], info["pid"], info["start_ticks"]) == info
    assert queue.find_process(info["cmdline"], info["cwd"], info["pid"], "old birth") is None
    assert queue.find_process(info["cmdline"], info["cwd"], info["pid"], None) is None
    assert queue.find_process(info["cmdline"] + ["--resume"], info["cwd"], info["pid"], info["start_ticks"]) is None


@pytest.mark.parametrize("state,selection,expected", [("completed", 0.5, True), ("running", 0.5, False),
                                                      ("completed", None, False), ("completed", float("nan"), False)])
def test_completion_contract_supports_best_at_update_zero(tmp_path, state, selection, expected):
    artifacts(tmp_path, state, selection)
    assert queue.terminal_training(tmp_path) is expected
    (tmp_path / "best.pt").unlink()
    assert not queue.terminal_training(tmp_path)


def test_exit_zero_and_terminal_artifacts_both_required(plan, monkeypatch):
    monkeypatch.setattr(queue, "find_process", lambda *a, **k: None)
    item = recorded_run(plan)
    artifacts(item["output"])
    _, exit_path = queue.attempt_paths(item, item["attempts"][-1])
    queue.atomic_json(exit_path, {"exit_code": 0, "ended_at": "test"})
    queue.update_run(item, {})
    assert item["state"] == "completed"
    item = recorded_run(plan)
    queue.atomic_json(exit_path, {"exit_code": 2, "ended_at": "test"})
    queue.update_run(item, {}, restarting=True)
    assert item["state"] == "failed"


def test_zero_exit_without_completion_does_not_resume(plan, monkeypatch):
    monkeypatch.setattr(queue, "find_process", lambda *a, **k: None)
    item = recorded_run(plan)
    artifacts(item["output"], state="running")
    _, exit_path = queue.attempt_paths(item, item["attempts"][-1])
    queue.atomic_json(exit_path, {"exit_code": 0, "ended_at": "test"})
    queue.update_run(item, {}, restarting=True)
    assert item["state"] == "failed" and not item["resume_next"]


def test_interrupted_checkpoint_recovers_once_only_without_live_process(plan, monkeypatch):
    monkeypatch.setattr(queue, "find_process", lambda *a, **k: None)
    item = recorded_run(plan)
    artifacts(item["output"], state="running")
    queue.update_run(item, {}, restarting=True)
    assert item["state"] == "queued" and item["resume_next"]
    queue.update_run(item, {})
    assert item["state"] == "queued" and item["resume_next"]
    item["attempts"].append({"attempt": 2, "resume": True, "launch_complete": True})
    item["state"] = "running"
    queue.update_run(item, {}, restarting=True)
    assert item["state"] == "failed"
    queue.update_run(item, {}, restarting=True)
    assert item["state"] == "failed"


def test_missing_exit_with_terminal_checkpoint_is_not_retrained(plan, monkeypatch):
    monkeypatch.setattr(queue, "find_process", lambda *a, **k: None)
    item = recorded_run(plan)
    artifacts(item["output"])
    queue.update_run(item, {}, restarting=True)
    assert item["state"] == "failed" and not item["resume_next"]


def test_dead_run_outside_restart_is_failed_and_never_silently_retried(plan, monkeypatch):
    monkeypatch.setattr(queue, "find_process", lambda *a, **k: None)
    item = recorded_run(plan)
    artifacts(item["output"], state="running")
    queue.update_run(item, {})
    assert item["state"] == "failed"
    queue.update_run(item, {}, restarting=True)
    assert item["state"] == "failed" and len(item["attempts"]) == 1


def test_launch_intent_without_process_or_checkpoint_can_requeue(plan, monkeypatch):
    monkeypatch.setattr(queue, "find_process", lambda *a, **k: None)
    item = recorded_run(plan)
    item["attempts"][0].pop("launch_complete")
    queue.update_run(item, {}, restarting=True)
    assert item["state"] == "queued" and item["attempts"] == []


def test_exact_existing_worker_and_trainer_are_adopted_without_resume(plan, monkeypatch):
    item = recorded_run(plan)
    monkeypatch.setattr(queue, "find_process", lambda cmd, *a: {"pid": 123 if "--worker" in cmd else 456,
                                                               "start_ticks": "birth"})
    queue.update_run(item, {}, restarting=True)
    assert item["state"] == "running" and item["trainer_pid"] == 456
    assert len(item["attempts"]) == 1 and not item["resume_next"]


def test_controller_lock_blocks_duplicate_and_detach_reports_existing(plan, capsys):
    output = plan.parent / "suite"
    suite(plan)
    with (output / ".controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another controller"):
            queue.controller(args(plan))
        assert queue.detach(args(plan)) == 0
        assert json.loads(capsys.readouterr().out)["state"] == "already_running"


def test_detach_passes_held_lock_to_child_and_does_not_run_gpu(plan, monkeypatch, capsys):
    recorded = {}
    def fake_popen(command, **kwargs):
        recorded.update(command=command, **kwargs)
        with (plan.parent / "suite" / ".controller.lock").open("a") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return SimpleNamespace(pid=123)
    monkeypatch.setattr(queue.subprocess, "Popen", fake_popen)
    assert queue.detach(args(plan)) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "started"
    assert recorded["start_new_session"] and recorded["pass_fds"]
    assert "--controller-lock-fd" in recorded["command"]
    assert recorded["env"]["PYTHONDONTWRITEBYTECODE"] == "1"


def test_detach_never_restarts_terminal_suite(plan, monkeypatch, capsys):
    status = suite(plan)
    for run in status["runs"]:
        run["state"] = "failed"
    queue.atomic_json(plan.parent / "suite" / "status.json", status)
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    assert queue.detach(args(plan)) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "already_terminal"


def test_worker_records_exit_and_uses_detached_unchanged_training_command(plan, monkeypatch):
    item = recorded_run(plan)
    monkeypatch.setattr(queue, "find_process", lambda *a, **k: None)
    monkeypatch.setattr(queue, "process_info", lambda pid: {"start_ticks": "birth"})
    calls = []
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=123, wait=lambda: 0)
    monkeypatch.setattr(queue.subprocess, "Popen", popen)
    worker_args = SimpleNamespace(config=item["config"], manifest=item["manifest"], output=item["output"],
                                  attempt=1, resume=False)
    assert queue.worker_main(worker_args) == 0
    command, kwargs = calls[0]
    assert kwargs["start_new_session"] and "--set" not in command and "--init" not in command
    worker, exit_path = queue.attempt_paths(item, item["attempts"][0])
    assert queue.read_json(worker)["trainer_start_ticks"] == "birth"
    assert queue.read_json(exit_path)["exit_code"] == 0


def test_controller_finishes_simulated_jobs_without_epoch_checks(plan, monkeypatch):
    snapshot = {"total_mib": 32607, "used_mib": 0, "free_mib": 32607, "processes": {}}
    monkeypatch.setattr(queue, "gpu_snapshot", lambda: snapshot)
    monkeypatch.setattr(queue.time, "sleep", lambda _: None)
    def launch(run, save):
        run["state"] = "completed"
        save()
        return SimpleNamespace(poll=lambda: 0)
    monkeypatch.setattr(queue, "launch_run", launch)
    assert queue.controller(args(plan)) == 0
    status = queue.read_json(plan.parent / "suite" / "status.json")
    assert status["controller"]["state"] == "completed"
    assert all(run["state"] == "completed" for run in status["runs"])
    line = json.loads((plan.parent / "suite" / "gpu_monitor.jsonl").read_text().splitlines()[0])
    assert line["running_count"] == 0 and line["used_mib"] == 0
