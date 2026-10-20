"""CPU-only admission, launch isolation, durable recovery and filesystem checks."""
import errno
import importlib.util
import os
from pathlib import Path

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).parents[1] / "scripts/run_t0_fivefold.py"
    spec = importlib.util.spec_from_file_location("t0_cv_runner_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inventory(count=10):
    return "\n".join(f"{i}, GPU-{i:04d}, Test GPU, 81920, 0" for i in range(count))


def test_gpu_allocation_respects_mask_order_and_count(runner):
    query = lambda arguments: _inventory()
    gpus = runner.discover_gpus(8, {"CUDA_VISIBLE_DEVICES": "8,2,0,1,3,4,5,6"}, query)
    assert [g["index"] for g in gpus] == [8, 2, 0, 1, 3, 4, 5, 6]
    assert [g["logical_index"] for g in gpus] == list(range(8))
    by_uuid = runner.discover_gpus(2, {"CUDA_VISIBLE_DEVICES": "GPU-0003,GPU-0001"}, query)
    assert [g["index"] for g in by_uuid] == [3, 1]
    with pytest.raises(ValueError, match="exactly 8"):
        runner.discover_gpus(8, {"CUDA_VISIBLE_DEVICES": "0,1"}, query)
    with pytest.raises(ValueError, match="distinct"):
        runner.discover_gpus(2, {"CUDA_VISIBLE_DEVICES": "0,0"}, query)
    with pytest.raises(ValueError, match="exactly"):
        runner.discover_gpus(8, {"NVIDIA_VISIBLE_DEVICES": "void"}, query)


def test_ddp_launch_is_rejected_before_controller_creation(runner):
    runner.reject_distributed_launch({})
    runner.reject_distributed_launch({"WORLD_SIZE": "1", "RANK": "0"})
    for env in ({"WORLD_SIZE": "8"}, {"LOCAL_WORLD_SIZE": "8"}, {"LOCAL_RANK": "1"}):
        with pytest.raises(RuntimeError, match="custom|one custom"):
            runner.reject_distributed_launch(env)


def _admission(runner):
    gpu = {"uuid": "GPU-a", "name": "Test GPU", "total_mib": 81920, "logical_index": 0}
    profile = {"reservation_mib": 18000, "variant": "forecast"}
    return gpu, {runner.profile_key(gpu, "forecast"): profile}


def test_peak_reservations_allow_multiple_tasks_without_count_cap(runner):
    gpu, profiles = _admission(runner)
    workers = []
    for index in range(4):
        assert runner.choose_gpu([gpu], profiles, workers, {gpu["uuid"]: 0}, variant="forecast") == gpu
        workers.append({"gpu_uuid": gpu["uuid"], "variant": "forecast", "pid": index + 10})
    assert runner.choose_gpu([gpu], profiles, workers, {gpu["uuid"]: 0}, variant="forecast") is None


def test_unknown_host_pids_do_not_erase_foreign_memory(runner):
    gpu, profiles = _admission(runner)
    active = [{"gpu_uuid": gpu["uuid"], "variant": "forecast", "pid": 10}]
    used = {gpu["uuid"]: 50000}
    assert runner.choose_gpu([gpu], profiles, active, used, {}, variant="forecast") is None
    assert runner.choose_gpu([gpu], profiles, active, used,
        {10: {"gpu_uuid": gpu["uuid"], "mib": 45000}}, variant="forecast") == gpu
    active[0]["exclusive_retry"] = True
    assert runner.choose_gpu([gpu], profiles, active, {}, variant="forecast") is None


def test_oom_exclusive_retry_admits_idle_gpu_with_nonzero_baseline(runner):
    gpu, profiles = _admission(runner)
    profile = profiles[runner.profile_key(gpu, "forecast")]
    profile["exclusive_only"] = True
    assert runner.choose_gpu([gpu], profiles, [], {gpu["uuid"]: 512},
                             variant="forecast", exclusive=True) == gpu
    active = [{"gpu_uuid": gpu["uuid"], "variant": "forecast", "pid": 10}]
    assert runner.choose_gpu([gpu], profiles, active, {}, variant="forecast") is None
    profile.pop("exclusive_only")
    assert runner.choose_gpu([gpu], profiles, active, {}, variant="forecast", exclusive=True) is None


def test_worker_environment_uses_frozen_code_and_one_gpu(runner, monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "/wrong/editable/source")
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("WORLD_SIZE", "8")
    env = runner.worker_environment(tmp_path, {"uuid": "GPU-allocated"})
    assert env["PYTHONPATH"] == str(tmp_path / "src")
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-allocated"
    assert "RANK" not in env and "WORLD_SIZE" not in env
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_controller_lock_excludes_duplicate_and_releases(runner, tmp_path):
    output = tmp_path / "projects/xts0030/public/cxx/example"
    with runner.controller_lock(output):
        with pytest.raises(RuntimeError, match="active controller"):
            with runner.controller_lock(output):
                pass
    with runner.controller_lock(output):
        pass


def test_missing_projects_parent_errors_without_creating_fake_mount(runner, tmp_path):
    parent = tmp_path / "missing_projects/public"
    spec = {"paths": {"output_root": str(parent / "cxx")}}
    with pytest.raises(FileNotFoundError, match="Mount the projects storage"):
        runner.validate_output_parent(spec)
    assert not parent.exists()


def test_pid_reuse_is_not_treated_as_a_surviving_worker(runner):
    info = runner.process_info(os.getpid())
    assert runner.same_process(info)
    assert runner.same_process({**info, "start_ticks": str(int(info["start_ticks"]) + 1)}) is None
    assert runner.same_process({"pid": 999999999, "start_ticks": "1"}) is None


def test_freeze_resumes_pre_matrix_rename_and_rejects_mutation(runner, monkeypatch, tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    (source / "src/pkg").mkdir(parents=True)
    (source / "scripts").mkdir()
    (source / "src/pkg/__init__.py").write_text("VALUE = 1\n")
    (source / "scripts/profile.py").write_text("pass\n")
    (source / "train.py").write_text("pass\n")
    output.mkdir()
    monkeypatch.setattr(runner, "ROOT", source)
    runtime = runner.freeze_runtime(output)
    assert runner.freeze_runtime(output) == runtime
    state = {"runtime": str(runtime), "runtime_hashes": runner.runtime_hashes(runtime)}
    runner.validate_runtime(state)
    (runtime / "src/pkg/__init__.py").write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="runtime changed"):
        runner.validate_runtime(state)
    with pytest.raises(FileExistsError, match="differs"):
        runner.freeze_runtime(output)


def test_storage_probe_surfaces_quota_and_removes_probe(runner, monkeypatch, tmp_path):
    def quota(*args):
        raise OSError(errno.EDQUOT, "Disk quota exceeded")
    monkeypatch.setattr(runner.os, "posix_fallocate", quota)
    with pytest.raises(OSError) as raised:
        runner.storage_probe(tmp_path, 4 * 2**20)
    assert raised.value.errno == errno.EDQUOT
    assert not list(tmp_path.glob(".write_probe.*"))


def test_checkpoint_budget_covers_active_and_atomic_growth(runner, tmp_path):
    state = {"spec": {"resources": {"storage_margin_factor": 1.25}},
             "profiles": {"key": {"variant": "forecast", "checkpoint_bytes": 1000}}}
    job = {"variant": "forecast", "output": str(tmp_path / "run")}
    assert runner._checkpoint_required(state, [], job) == 3750 + 256 * 2**20
    Path(job["output"]).mkdir()
    (Path(job["output"]) / "best.pt").write_bytes(b"0" * 1000)
    assert runner._checkpoint_required(state, [], job) == 2500 + 256 * 2**20
    with pytest.raises(ValueError, match="checkpoint size"):
        runner._checkpoint_required({**state, "profiles": {}}, [], job)


def test_host_admission_respects_cpu_and_eval_memory_reservation(runner, monkeypatch):
    state = {"spec": {"resources": {"threads_per_task": 4, "loader_workers": 2, "host_memory_fraction": .85}},
             "profiles": {"profile": {"variant": "forecast", "host_peak_mib": 100,
                                          "host_reservation_mib": 12000}}}
    monkeypatch.setattr(runner, "host_resources", lambda: {"cpus": 12, "total_bytes": 64000 * 2**20,
                                                          "available_bytes": 60000 * 2**20})
    job = {"variant": "forecast"}
    assert runner._host_admits(state, [], job)
    assert not runner._host_admits(state, [job, job], job)
    monkeypatch.setattr(runner, "host_resources", lambda: {"cpus": 96, "total_bytes": 64000 * 2**20,
                                                          "available_bytes": 8000 * 2**20})
    assert not runner._host_admits(state, [], job)


def test_host_admission_reserves_active_workers_future_growth(runner, monkeypatch):
    state = {"spec": {"resources": {"threads_per_task": 4, "loader_workers": 2, "host_memory_fraction": .85}},
             "profiles": {"p": {"variant": "forecast", "host_reservation_mib": 24000}}}
    monkeypatch.setattr(runner, "host_resources", lambda: {"cpus": 96, "total_bytes": 128000 * 2**20,
                                                          "available_bytes": 40000 * 2**20})
    monkeypatch.setattr(runner, "process_tree_rss", lambda pid: 8000 * 2**20)
    job = {"variant": "forecast", "pid": 1}
    assert not runner._host_admits(state, [job] * 3, job)
    assert runner.process_tree_rss(1) > 0


def test_single_worker_exceeding_host_budget_fails_explicitly(runner, monkeypatch):
    state = {"spec": {"resources": {"host_memory_fraction": .8}},
             "profiles": {"p": {"variant": "forecast", "host_reservation_mib": 14000}}}
    monkeypatch.setattr(runner, "host_resources", lambda: {"total_bytes": 16000 * 2**20})
    with pytest.raises(RuntimeError, match="request more RAM"):
        runner.validate_host_capacity(state)


def test_atomic_state_never_serializes_nan(runner, tmp_path):
    path = tmp_path / "state.json"
    runner.durable_json(path, {"status": "running"})
    with pytest.raises(ValueError):
        runner.durable_json(path, {"value": float("nan")})
    assert runner.read_json(path) == {"status": "running"}
    assert not list(tmp_path.glob("*.tmp"))


def test_existing_suite_reexecutes_frozen_controller(runner, monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    (runtime / "scripts").mkdir(parents=True)
    (runtime / "scripts/run_t0_fivefold.py").write_text("pass\n")
    state = {"runtime": str(runtime), "runtime_hashes": runner.runtime_hashes(runtime)}
    captured = {}

    class Executed(Exception):
        pass

    def execute(binary, arguments, env):
        captured.update(binary=binary, arguments=arguments, env=env)
        raise Executed

    monkeypatch.setattr(runner.os, "execve", execute)
    with pytest.raises(Executed):
        runner.enter_frozen_controller(state, ["--config", "a config.yaml", "--report-only"])
    assert captured["arguments"][1:] == [str(runtime / "scripts/run_t0_fivefold.py"),
                                         "--config", "a config.yaml", "--report-only"]
    assert captured["env"]["PYTHONPATH"] == str(runtime / "src")
    (runtime / "scripts/run_t0_fivefold.py").write_text("changed\n")
    with pytest.raises(ValueError, match="runtime changed"):
        runner.enter_frozen_controller(state, [])


def _prepared_state(runner, monkeypatch, tmp_path, *, completed=False):
    import mri_vla_jepa.t0_cv_suite as suite
    output = tmp_path / "suite"
    output.mkdir()
    spec = {"output_dir": str(output), "paths": {"output_root": str(tmp_path)}}
    state = {"spec": spec, "status": "completed" if completed else "running",
             "jobs": [{"name": "one", "status": "completed" if completed else "running"}]}
    runner.durable_json(output / "matrix.json", state)
    monkeypatch.setattr(suite, "load_suite_config", lambda path: spec)
    monkeypatch.setattr(suite, "validate_inputs", lambda value: None)
    monkeypatch.setattr(runner, "enter_frozen_controller", lambda *args: None)
    monkeypatch.setattr(runner, "validate_runtime", lambda *args: None)
    return output, state, suite


def test_report_only_works_while_training_controller_holds_lock(runner, monkeypatch, tmp_path):
    output, state, suite = _prepared_state(runner, monkeypatch, tmp_path)
    reports = []
    monkeypatch.setattr(suite, "write_reports", lambda *args: reports.append(args))
    with runner.controller_lock(output):
        assert runner.main(["--config", "ignored.yaml", "--report-only"]) == 0
    assert len(reports) == 1 and reports[0][1] == state


def test_completed_resubmission_needs_no_gpu_and_does_not_train(runner, monkeypatch, tmp_path):
    output, state, suite = _prepared_state(runner, monkeypatch, tmp_path, completed=True)
    verified, reports = [], []
    monkeypatch.setattr(runner, "evaluation_complete", lambda job, out: verified.append(job) or True)
    monkeypatch.setattr(suite, "write_reports", lambda *args: reports.append(args))
    monkeypatch.setattr(runner, "discover_gpus", lambda **kwargs: pytest.fail("Completed suite requested GPUs"))
    monkeypatch.setattr(runner, "spawn_worker", lambda *args: pytest.fail("Completed suite launched a worker"))
    assert runner.main(["--config", "ignored.yaml"]) == 0
    assert len(verified) == 1 and len(reports) == 1


def test_storage_quota_blocks_new_training_without_stopping_existing_work(runner, monkeypatch, tmp_path):
    import mri_vla_jepa.t0_cv_suite as suite
    gpu, profiles = _admission(runner)
    profiles[next(iter(profiles))]["checkpoint_bytes"] = 1000
    job = {"name": "one", "variant": "forecast", "fold_id": 1, "status": "queued",
           "output": str(tmp_path / "one")}
    state = {"spec": {"resources": {"memory_fraction": .9, "reservation_margin_mib": 768,
                                    "poll_seconds": 1, "storage_margin_factor": 1.25}},
             "gpus": [gpu], "profiles": profiles, "jobs": [job]}
    stopped = [False]
    monkeypatch.setattr(runner, "recover_workers", lambda *args: None)
    monkeypatch.setattr(runner, "evaluation_complete", lambda *args: False)
    monkeypatch.setattr(runner, "training_complete", lambda *args: False)
    monkeypatch.setattr(runner, "gpu_observations", lambda: ({gpu["uuid"]: 0}, {}))
    monkeypatch.setattr(runner, "_host_admits", lambda *args: True)
    monkeypatch.setattr(runner, "validate_runtime", lambda *args: None)
    monkeypatch.setattr(suite, "validate_inputs", lambda *args: None)
    monkeypatch.setattr(runner, "spawn_worker", lambda *args: pytest.fail("Quota failure launched a worker"))

    def quota(*args):
        raise OSError(errno.EDQUOT, "Disk quota exceeded")

    monkeypatch.setattr(runner, "storage_probe", quota)
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: stopped.__setitem__(0, True))
    assert runner.run_controller(state, tmp_path, stopped) == 130
    assert state["storage_blocked"] is True
    assert job["status"] == "queued"
    assert (tmp_path / "matrix.json").is_file()


def test_adopted_worker_exit_receipt_and_attempt_log_are_checked(runner, tmp_path):
    log = tmp_path / "train.log"
    log.write_text("Old traceback (most recent call last)\n")
    offset = log.stat().st_size
    job = {"name": "one", "attempt_id": "id123", "pid": 11, "phase": "train",
           "attempt_status": str(tmp_path / "attempts/id123.json"), "log": str(log), "log_start": offset}
    assert runner.recovered_exit_code(job, tmp_path) is None
    with log.open("a") as stream:
        stream.write("RuntimeError: CUDA out of memory\n")
    assert runner.recovered_exit_code(job, tmp_path) == 1
    receipt = {"attempt_id": "id123", "pid": 11, "phase": "train", "job": "one", "exit_code": 0}
    runner.durable_json(job["attempt_status"], receipt)
    assert runner.recovered_exit_code(job, tmp_path) == 0
    runner.durable_json(job["attempt_status"], {**receipt, "pid": 12})
    with pytest.raises(ValueError, match="identity differs"):
        runner.recovered_exit_code(job, tmp_path)


def test_second_oom_of_adopted_worker_stops_instead_of_requeueing(runner, monkeypatch, tmp_path):
    gpu, profiles = _admission(runner)
    log = tmp_path / "train.log"
    log.write_text("RuntimeError: CUDA out of memory\n")
    job = {"name": "one", "variant": "forecast", "status": "running", "phase": "train",
           "pid": 999999999, "gpu_uuid": gpu["uuid"], "log": str(log), "oom_retries": 1,
           "output": str(tmp_path / "one")}
    state = {"spec": {"resources": {"memory_fraction": .9}},
             "gpus": [gpu], "profiles": profiles, "jobs": [job]}
    monkeypatch.setattr(runner, "recover_workers", lambda *args: None)
    monkeypatch.setattr(runner, "evaluation_complete", lambda *args: False)
    monkeypatch.setattr(runner, "training_complete", lambda *args: False)
    monkeypatch.setattr(runner, "spawn_worker", lambda *args: pytest.fail("Repeated OOM relaunched"))
    with pytest.raises(RuntimeError, match="Worker failed; checkpoints preserved"):
        runner.run_controller(state, tmp_path, [False])
    assert job["status"] == "failed" and job["oom_retries"] == 1
