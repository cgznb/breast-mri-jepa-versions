"""Queue admission, restart ownership, and actual training completion contracts."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml


spec = importlib.util.spec_from_file_location("joint_queue", Path(__file__).resolve().parents[1] /
                                         "scripts" / "run_joint_suite.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def run(profile, pid=None):
    return {"profile": profile, "trainer_pid": pid}


def test_three_jobs_reserved_before_cuda_allocates():
    snapshot = {"total_mib": 32607, "used_mib": 0, "processes": {}}
    running = [run("dynamic"), run("t0")]
    assert queue.can_launch(run("dynamic"), running, snapshot, 29696, 3)
    running.append(run("dynamic"))
    assert queue.reserved_usage(running, snapshot) == 28672
    assert not queue.can_launch(run("t0"), running, snapshot, 29696, 3)


def test_external_memory_and_observed_peaks_reduce_admission():
    snapshot = {"total_mib": 32607, "used_mib": 20500,
                "processes": {11: 11000, 12: 8000, 99: 1500}}
    running = [run("dynamic", 11), run("t0", 12)]
    assert queue.reserved_usage(running, snapshot) == 20692
    assert not queue.can_launch(run("dynamic"), running, snapshot, 29696, 3)
    assert queue.can_launch(run("t0"), running, snapshot, 29696, 3)
    assert not queue.can_launch(run("t0"), [], {"total_mib": 7000, "used_mib": 0, "processes": {}}, 29696, 3)


def test_custom_reservation_is_applied_to_running_and_candidate():
    snapshot = {"total_mib": 32607, "used_mib": 0, "processes": {}}
    candidate = {"profile": "dynamic", "reservation_mib": 14000}
    running = [{"profile": "t0", "reservation_mib": 16000}]
    assert not queue.can_launch(candidate, running, snapshot, 29696, 3)


def test_process_match_requires_full_command_directory_and_birth():
    info = queue.process_info(os.getpid())
    assert info is not None
    assert queue.matches_process(info, info["cmdline"], info["cwd"], info["start_ticks"])
    assert not queue.matches_process(info, info["cmdline"] + ["--resume"], info["cwd"])
    assert not queue.matches_process(info, info["cmdline"], "/tmp", info["start_ticks"])
    assert not queue.matches_process(info, info["cmdline"], info["cwd"], "old_birth")


def templates(tmp_path):
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    for seed in (17, 43, 71):
        for profile in ("t0", "dynamic"):
            data = {"train": {"phase": "joint", "profile": profile, "seed": seed,
                              "epochs": 60, "patience": 15, "batch_size": 16,
                              "accumulation": 1, "prefix_sampling": "all_prefixes" if profile == "dynamic"
                              else "balanced_cycle"}}
            name = f"joint_{profile}{'_all_prefixes' if profile == 'dynamic' else ''}_batch16_seed{seed}.yaml"
            (project / "configs" / name).write_text(yaml.safe_dump(data))
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    return project, manifest


def test_six_run_plan_preserves_configs_and_long_job_priority(tmp_path):
    project, manifest = templates(tmp_path)
    output = tmp_path / "suite"
    status = queue.prepare_suite(project, output, manifest, [17, 43, 71])
    assert [item["name"] for item in status["runs"]][:3] == [
        "joint_dynamic_seed17", "joint_t0_seed17", "joint_dynamic_seed43"]
    assert len({item["output"] for item in status["runs"]}) == 6
    for item in status["runs"]:
        assert Path(item["config"]).read_bytes() == Path(item["source_config"]).read_bytes()
    queue.atomic_json(output / "status.json", status)
    assert queue.prepare_suite(project, output, manifest, [17, 43, 71]) == status
    with pytest.raises(ValueError, match="identity differs"):
        queue.prepare_suite(project, output, manifest, [17, 43])
    first = Path(status["runs"][0]["source_config"])
    data = yaml.safe_load(first.read_text())
    data["train"]["epochs"] = 30
    first.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="configuration changed"):
        queue.prepare_suite(project, output, manifest, [17, 43, 71])


def test_missing_or_wrong_config_does_not_create_status(tmp_path):
    project, manifest = templates(tmp_path)
    source = project / "configs" / "joint_dynamic_all_prefixes_batch16_seed17.yaml"
    data = yaml.safe_load(source.read_text())
    data["train"]["prefix_sampling"] = "random"
    source.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="Unexpected joint"):
        queue.prepare_suite(project, tmp_path / "suite", manifest, [17, 43, 71])
    assert not (tmp_path / "suite" / "status.json").exists()


def artifacts(output, epoch=60, bad=0):
    output.mkdir(parents=True, exist_ok=True)
    config = {"train": {"epochs": 60, "patience": 15}}
    queue.atomic_json(output / "config.json", config)
    queue.atomic_json(output / "metrics_best.json", {"selection": .5, "selection_name": "patient_equal_pcr_nll"})
    (output / "best.pt").write_bytes(b"present")
    (output / "predictions_val.csv").write_text("landmark,p_fused\n")
    torch.save({"schema": "response_jepa_checkpoint_local_v1", "config": config,
                "epoch": epoch, "bad_epochs": bad, "best_score": .5}, output / "last.pt")


@pytest.mark.parametrize("epoch,bad,expected", [(60, 0, True), (20, 15, True), (1, 0, False), (59, 14, False)])
def test_completion_requires_terminal_epoch_or_early_stop(tmp_path, epoch, bad, expected):
    artifacts(tmp_path, epoch, bad)
    assert queue.complete_artifacts(tmp_path)
    assert queue.terminal_training(tmp_path) is expected


def test_invalid_metrics_or_missing_predictions_are_incomplete(tmp_path):
    artifacts(tmp_path)
    queue.atomic_json(tmp_path / "metrics_best.json", {"selection": .7, "selection_name": "patient_equal_pcr_nll"})
    assert not queue.terminal_training(tmp_path)
    (tmp_path / "predictions_val.csv").unlink()
    assert not queue.complete_artifacts(tmp_path)


def recorded_run(tmp_path, attempts=1):
    return {"name": "joint_t0_seed17", "profile": "t0", "state": "running",
            "config": str(tmp_path / "config.yaml"), "output": str(tmp_path / "run"),
            "attempts": [{"attempt": index + 1, "resume": bool(index), "launch_complete": True}
                         for index in range(attempts)]}


def test_dead_run_resume_only_once_and_only_on_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "find_process", lambda *args, **kwargs: None)
    item = recorded_run(tmp_path)
    artifacts(Path(item["output"]), epoch=1)
    queue.update_run(item, tmp_path / "manifest", {}, restarting=False)
    assert item["state"] == "failed"
    queue.update_run(item, tmp_path / "manifest", {}, restarting=True)
    assert item["state"] == "queued" and item["resume_next"]
    item = recorded_run(tmp_path, attempts=2)
    queue.update_run(item, tmp_path / "manifest", {}, restarting=True)
    assert item["state"] == "failed"
    (Path(item["output"]) / "last.pt").unlink()
    item = recorded_run(tmp_path)
    queue.update_run(item, tmp_path / "manifest", {}, restarting=True)
    assert item["state"] == "failed"


def test_exit_zero_smoke_not_completed_then_real_terminal_is_completed(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "find_process", lambda *args, **kwargs: None)
    item = recorded_run(tmp_path)
    output = Path(item["output"])
    artifacts(output, epoch=1)
    _, exit_path = queue.attempt_paths(item, item["attempts"][-1])
    queue.atomic_json(exit_path, {"exit_code": 0, "ended_at": "test"})
    queue.update_run(item, tmp_path / "manifest", {})
    assert item["state"] == "failed"
    artifacts(output, epoch=60)
    queue.update_run(item, tmp_path / "manifest", {})
    assert item["state"] == "completed"


def test_adopt_exact_existing_worker_and_trainer_without_relaunch(tmp_path, monkeypatch):
    item = recorded_run(tmp_path)
    def find(command, cwd, *args):
        return {"pid": 123 if "--worker" in command else 456, "start_ticks": "birth"}
    monkeypatch.setattr(queue, "find_process", find)
    queue.update_run(item, tmp_path / "manifest", {}, restarting=True)
    assert item["state"] == "running"
    assert item["trainer_pid"] == 456 and len(item["attempts"]) == 1


def test_launch_intent_without_process_can_requeue_safely(tmp_path, monkeypatch):
    item = recorded_run(tmp_path)
    item["attempts"][0].pop("launch_complete")
    monkeypatch.setattr(queue, "find_process", lambda *args, **kwargs: None)
    queue.update_run(item, tmp_path / "manifest", {}, restarting=True)
    assert item["state"] == "queued" and item["attempts"] == []


def test_controller_lock_prevents_duplicate_launches(tmp_path):
    import fcntl
    with (tmp_path / ".controller.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another controller"):
            queue.controller(SimpleNamespace(output=str(tmp_path)))


def test_trainer_environment_and_resume_have_no_train_overrides():
    env = queue.training_environment()
    assert env["PYTHONPATH"] == str(queue.PROJECT / "src")
    assert env["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert all(env[key] == "4" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
    command = queue.trainer_command("config", "manifest", "output", True)
    assert command[-1] == "--resume" and "--set" not in command and "--init" not in command
