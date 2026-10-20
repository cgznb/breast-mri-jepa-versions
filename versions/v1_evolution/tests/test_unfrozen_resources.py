"""Controller handoff and admission use fake workers, never real GPU processes."""
import errno
import importlib.util
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

from mri_vla_jepa.io import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    name = "unfrozen_resource_tests"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/run_unfrozen_ablation.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_process_identity_matches_start_ticks_and_rejects_reused_pid(runner):
    info = runner.process_info(os.getpid())
    assert info["pid"] == os.getpid() and info["command"]
    assert runner.same_process(info) == info
    assert runner.same_process({"pid": os.getpid(), "start_ticks": "reused-pid"}) is None
    assert runner.process_info(None) is None
    assert runner.process_info(10**12) is None


def test_stop_controller_checks_command_and_signals_only_its_pid(runner, monkeypatch):
    info = {"pid": 4001, "start_ticks": "900", "command": ["python", "/old/controller.py"]}
    monkeypatch.setattr(runner, "same_process", lambda record: info)
    signals = []
    monkeypatch.setattr(runner.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(RuntimeError, match="command"):
        runner._stop_controller(info, "/different/controller.py")
    assert not signals
    runner._stop_controller(info, "/old/controller.py")
    assert signals == [(4001, signal.SIGTERM)]


def test_stopped_controller_receives_cont_after_term_without_signalling_workers(runner, monkeypatch):
    info = {"pid": 4001, "state": "T", "start_ticks": "900", "command": ["python", "/old/controller.py"]}
    monkeypatch.setattr(runner, "same_process", lambda record: info)
    signals = []
    monkeypatch.setattr(runner.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    runner._stop_controller(info, "/old/controller.py")
    assert signals == [(4001, signal.SIGTERM), (4001, signal.SIGCONT)]


def test_storage_probe_fsyncs_and_removes_successful_probe(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner.os, "statvfs", lambda path: SimpleNamespace(f_bavail=10**9, f_frsize=4096))
    calls = []
    monkeypatch.setattr(runner.os, "fsync", lambda descriptor: calls.append(descriptor))
    runner.storage_probe(tmp_path, 2**30)
    assert len(calls) == 1
    assert not (tmp_path / ".checkpoint_write_probe").exists()


def test_storage_probe_stops_before_write_without_headroom(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner.os, "statvfs", lambda path: SimpleNamespace(f_bavail=1, f_frsize=4096))
    with pytest.raises(OSError) as caught:
        runner.storage_probe(tmp_path, 8192)
    assert caught.value.errno == errno.ENOSPC
    assert list(tmp_path.iterdir()) == []


def test_storage_probe_preserves_quota_error_and_cleans_partial_file(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner.os, "statvfs", lambda path: SimpleNamespace(f_bavail=10**9, f_frsize=4096))
    def quota_error(descriptor):
        raise OSError(errno.EDQUOT, "Disk quota exceeded")
    monkeypatch.setattr(runner.os, "fsync", quota_error)
    with pytest.raises(OSError) as caught:
        runner.storage_probe(tmp_path, 8192)
    assert caught.value.errno == errno.EDQUOT
    assert not (tmp_path / ".checkpoint_write_probe").exists()


@pytest.mark.parametrize("status,busy", [("running", True), ("evaluating", True),
    ("evaluation_pending", False), ("queued", False), ("paused", False),
    ("completed", False), ("failed", False)])
def test_old_gpu_busy_reserves_unfinished_allocation_only(runner, tmp_path, monkeypatch, status, busy):
    write_json(tmp_path / "matrix.json", {"jobs": [{"status": status, "gpu_uuid": "GPU-test"}]})
    monkeypatch.setattr(runner, "gpu_memory_used", lambda: {"GPU-test": 4})
    monkeypatch.setattr(runner, "same_process", lambda record: {"pid": 7001})
    assert runner.old_gpu_busy(tmp_path, "GPU-test") is busy
    assert not runner.old_gpu_busy(tmp_path, "GPU-other")
    monkeypatch.setattr(runner, "same_process", lambda record: None)
    assert not runner.old_gpu_busy(tmp_path, "GPU-test")


def test_old_gpu_busy_detects_foreign_gpu_workload_without_old_matrix(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "gpu_memory_used", lambda: {"GPU-test": 2048})
    assert runner.old_gpu_busy(tmp_path, "GPU-test")
    monkeypatch.setattr(runner, "gpu_memory_used", lambda: {"GPU-test": 4})
    assert not runner.old_gpu_busy(tmp_path, "GPU-test")


@pytest.fixture
def old_suite(runner, tmp_path, monkeypatch):
    root = tmp_path / "old"
    (root / "runtime").mkdir(parents=True)
    gpu0, gpu1 = {"uuid": "GPU-0"}, {"uuid": "GPU-1"}
    live = {"name": "allowed_live", "status": "running", "pid": 4033,
            "start_ticks": "33", "output": str(root / "allowed_live"), "gpu_uuid": "GPU-1"}
    held = {"name": "held", "status": "paused", "pid": None, "start_ticks": None,
            "output": str(root / "held"), "gpu_uuid": None, "paused_from_status": "queued"}
    matrix = {"gpus": [gpu0, gpu1], "controller_pid": 4011, "jobs": [live, held]}
    policy = {"python": "python", "supervisor_pid": 4022, "supervisor_start_ticks": "22",
        "training_allowed": ["allowed_live"], "evaluation_allowed": ["allowed_live"],
        "held_jobs": ["held"], "original_workers": {"allowed_live": {"pid": 4033, "start_ticks": "33"}}}
    write_json(root / "matrix.json", matrix)
    write_json(root / "drain_policy.json", policy)
    registry = {
        4011: {"pid": 4011, "start_ticks": "11", "command": ["python", str(root / "runtime/scripts/run_capacity_phase_matrix.py")]},
        4022: {"pid": 4022, "start_ticks": "22", "command": ["python", str(root / "drain_control.py")]},
        4033: {"pid": 4033, "start_ticks": "33", "command": ["python", "train.py", "allowed_live"]},
    }
    signals, starts, locks = [], [], {}
    def hold(pid, name):
        handle = (root / name).open("a")
        runner.fcntl.flock(handle, runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB)
        locks[pid] = handle
    hold(4011, "controller.lock")
    hold(4022, "drain_supervisor.lock")
    def forget(pid):
        registry.pop(pid, None)
        if pid in locks:
            locks.pop(pid).close()
    monkeypatch.setattr(runner, "process_info", lambda pid: registry.get(pid))
    def fake_kill(pid, sig):
        assert pid != 4033, "Handoff must never signal an original training worker"
        assert sig == signal.SIGTERM
        signals.append((pid, sig))
        forget(pid)
    monkeypatch.setattr(runner.os, "kill", fake_kill)
    def fake_spawn(command, **kwargs):
        pid = 4044 + 2 * len(starts)
        starts.append(command)
        registry[pid] = {"pid": pid, "start_ticks": str(pid), "command": command}
        # Emulate the supervisor publishing its own identity and controller.
        updated = read_json(root / "drain_policy.json")
        updated.update(supervisor_pid=pid, supervisor_start_ticks=str(pid),
                       controller_pid=pid + 1, controller_start_ticks=str(pid + 1))
        write_json(root / "drain_policy.json", updated)
        state = read_json(root / "matrix.json")
        state.update(controller_pid=pid + 1, controller_start_ticks=str(pid + 1))
        write_json(root / "matrix.json", state)
        registry[pid + 1] = {"pid": pid + 1, "start_ticks": str(pid + 1),
                            "command": ["python", str(root / "runtime/scripts/run_capacity_phase_matrix.py")]}
        hold(pid, "drain_supervisor.lock")
        hold(pid + 1, "controller.lock")
        return SimpleNamespace(pid=pid, poll=lambda: None)
    monkeypatch.setattr(runner.subprocess, "Popen", fake_spawn)
    yield SimpleNamespace(root=root, journal=tmp_path / "handoff.json", gpu0=gpu0, gpu1=gpu1,
        live=live, held=held, matrix=matrix, policy=policy, registry=registry, starts=starts,
        signals=signals, forget=forget, locks=locks)
    for handle in locks.values():
        handle.close()


def test_handoff_and_restore_keep_original_workers_and_drain_holds(runner, old_suite):
    old = old_suite
    runner.set_old_gpus(old.root, [old.gpu0], old.journal)
    state, policy = read_json(old.root / "matrix.json"), read_json(old.root / "drain_policy.json")
    assert state["gpus"] == [old.gpu0]
    assert state["jobs"] == [old.live, old.held]
    assert policy["held_jobs"] == old.policy["held_jobs"]
    assert policy["training_allowed"] == old.policy["training_allowed"]
    assert policy["evaluation_allowed"] == old.policy["evaluation_allowed"]
    assert runner.same_process(old.live)
    assert old.signals == [(4022, signal.SIGTERM), (4011, signal.SIGTERM)]
    assert len(old.starts) == 1 and "--supervise" in old.starts[0]
    runner.set_old_gpus(old.root, [old.gpu0], old.journal)
    assert len(old.starts) == 1
    runner.set_old_gpus(old.root, [], old.journal, restoring=True)
    restored = read_json(old.root / "matrix.json")
    assert restored["gpus"] == [old.gpu0, old.gpu1]
    assert restored["jobs"] == [old.live, old.held]
    assert runner.same_process(old.live)
    assert len(old.starts) == 2 and all("--supervise" in command for command in old.starts)


def test_interrupted_handoff_recovers_dead_controller_even_when_gpu_target_matches(runner, old_suite):
    old = old_suite
    state = read_json(old.root / "matrix.json")
    state.update(gpus=[old.gpu0], controller_pid=None)
    write_json(old.root / "matrix.json", state)
    old.forget(4011)
    old.forget(4022)
    write_json(old.journal, {"old_suite": str(old.root), "original_gpus": [old.gpu0, old.gpu1],
        "original_workers": {old.live["name"]: {"pid": 4033, "start_ticks": "33"}}, "status": "handoff"})
    runner.set_old_gpus(old.root, [old.gpu0], old.journal)
    assert len(old.starts) == 1, "A matching GPU list must not hide a dead controller after interrupted handoff"
    assert runner.same_process(old.live)
    assert read_json(old.root / "matrix.json")["jobs"] == [old.live, old.held]


def test_already_paused_old_suite_does_not_restart_held_experiments(runner, old_suite):
    old = old_suite
    state = read_json(old.root / "matrix.json")
    state.update(status="paused", controller_pid=None)
    state["jobs"][0].update(status="completed", pid=None)
    old.forget(4011)
    old.forget(4022)
    old.forget(4033)
    write_json(old.root / "matrix.json", state)
    runner.set_old_gpus(old.root, [old.gpu0], old.journal)
    assert not old.starts and not old.signals
    assert read_json(old.root / "matrix.json")["jobs"][1] == old.held


def test_replacement_must_be_ready_before_handoff_journal_is_committed(runner, old_suite, monkeypatch):
    old = old_suite
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs:
                        SimpleNamespace(pid=4999, poll=lambda: 1))
    with pytest.raises(RuntimeError, match="before confirmed handoff"):
        runner.set_old_gpus(old.root, [old.gpu0], old.journal)
    assert read_json(old.journal)["status"] == "handoff"
    assert runner.same_process(old.live)
    assert read_json(old.root / "matrix.json")["jobs"][1] == old.held


def test_handoff_refuses_existing_drain_hold_violation(runner, old_suite):
    old = old_suite
    state = read_json(old.root / "matrix.json")
    state["jobs"][1]["status"] = "queued"
    write_json(old.root / "matrix.json", state)
    with pytest.raises(RuntimeError, match="drain hold changed"):
        runner.set_old_gpus(old.root, [old.gpu0], old.journal)
    assert not old.starts and not old.signals


def test_handoff_never_automatically_resumes_a_paused_drain(runner, old_suite):
    old = old_suite
    policy = read_json(old.root / "drain_policy.json")
    policy["status"] = "paused"
    write_json(old.root / "drain_policy.json", policy)
    with pytest.raises(RuntimeError, match="refusing automatic resume"):
        runner.set_old_gpus(old.root, [old.gpu0], old.journal)
    assert not old.starts and not old.signals


def test_handoff_refuses_controller_pid_with_changed_start_ticks(runner, old_suite):
    old = old_suite
    state = read_json(old.root / "matrix.json")
    state["controller_start_ticks"] = "not-the-original-process"
    write_json(old.root / "matrix.json", state)
    with pytest.raises(RuntimeError, match="controller identity changed"):
        runner.set_old_gpus(old.root, [old.gpu0], old.journal)
    assert not old.starts and not old.signals


@pytest.fixture
def orphan_suite(runner, tmp_path, monkeypatch):
    output = tmp_path / "new_suite"
    output.mkdir()
    runtime = output / "runtime"
    job = {"name": "attention_seed17", "status": "queued", "pid": None, "start_ticks": None,
           "config": str(output / "configs/attention_seed17.yaml"),
           "manifest": str(output / "manifests/seed17.json"),
           "output": str(output / "runs/attention_seed17")}
    state = {"runtime": str(runtime), "gpu": {"uuid": "GPU-1"}, "jobs": [job]}
    processes = {}
    monkeypatch.setattr(runner, "process_info", lambda pid: processes.get(pid))
    monkeypatch.setattr(runner, "live_processes", lambda: list(processes.values()))
    monkeypatch.setattr(runner.os, "kill", lambda *args: pytest.fail("Recovery must never signal workers"))
    train = ["python", str(runtime / "train.py"), "train-vla-jepa", "--config", job["config"],
             "--manifest", job["manifest"], "--output", job["output"]]
    evaluate = ["python", str(runtime / "scripts/run_unfrozen_ablation.py"),
                "--output", str(output), "--evaluate-job", job["name"]]
    return SimpleNamespace(output=output, runtime=runtime, job=job, state=state,
                           processes=processes, train=train, evaluate=evaluate)


@pytest.mark.parametrize("kind", ["train", "resume", "evaluate"])
def test_recovers_unjournaled_worker_and_persists_identity_before_adoption(runner, orphan_suite, kind):
    suite = orphan_suite
    command = suite.evaluate if kind == "evaluate" else suite.train + (["--resume"] if kind == "resume" else [])
    suite.processes[9001] = {"pid": 9001, "start_ticks": "901", "command": command}
    # A reused recorded PID must not conceal a genuine unjournaled worker.
    suite.job.update(pid=9001, start_ticks="former-process")
    recovered = runner.recover_unregistered_workers(suite.state, suite.output)
    saved = read_json(suite.output / "matrix.json")
    assert saved["jobs"][0]["status"] == ("evaluating" if kind == "evaluate" else "running")
    assert saved["jobs"][0]["pid"] == 9001 and saved["jobs"][0]["start_ticks"] == "901"
    assert saved["jobs"][0]["gpu_uuid"] == "GPU-1"
    assert saved["worker_recovery"]["recovered_jobs"] == [suite.job["name"]]
    assert recovered[0]["pid"] == 9001


@pytest.mark.parametrize("mutation", ["runtime", "config", "manifest", "output", "duplicate_flag",
    "unknown_flag", "duplicate_resume", "subcommand", "eval_output", "eval_name", "eval_worker"])
def test_orphan_discovery_rejects_other_suites_and_ambiguous_commands(runner, orphan_suite, mutation):
    suite = orphan_suite
    command = list(suite.evaluate if mutation.startswith("eval_") else suite.train)
    if mutation == "runtime":
        command[1] = str(suite.runtime.parent / "other_runtime/train.py")
    elif mutation in {"config", "manifest", "output"}:
        index = command.index("--" + mutation) + 1
        command[index] += "_other"
    elif mutation == "duplicate_flag":
        command += ["--output", suite.job["output"]]
    elif mutation == "unknown_flag":
        command += ["--worker"]
    elif mutation == "duplicate_resume":
        command += ["--resume", "--resume"]
    elif mutation == "subcommand":
        command[2] = "some-other-training"
    elif mutation == "eval_output":
        command[command.index("--output") + 1] += "_other"
    elif mutation == "eval_name":
        command[command.index("--evaluate-job") + 1] += "_other"
    elif mutation == "eval_worker":
        command += ["--worker"]
    suite.processes[9001] = {"pid": 9001, "start_ticks": "901", "command": command}
    assert runner.recover_unregistered_workers(suite.state, suite.output) == []
    assert suite.job["status"] == "queued" and suite.job["pid"] is None
    assert not (suite.output / "matrix.json").exists()


@pytest.mark.parametrize("registered", [False, True])
def test_duplicate_live_job_workers_block_dispatch_without_killing_either(runner, orphan_suite, registered):
    suite = orphan_suite
    suite.processes[9001] = {"pid": 9001, "start_ticks": "901", "command": suite.train}
    suite.processes[9002] = {"pid": 9002, "start_ticks": "902", "command": suite.evaluate}
    if registered:
        suite.job.update(status="running", pid=9001, start_ticks="901")
        # Already journaled identities do not depend on discovery command parsing.
        suite.processes[9001]["command"] = [suite.job["config"], suite.job["name"]]
    with pytest.raises(RuntimeError, match="Multiple live workers"):
        runner.recover_unregistered_workers(suite.state, suite.output)
    recovery = read_json(suite.output / "matrix.json")["worker_recovery"]
    assert recovery["duplicate_jobs"] == [suite.job["name"]]
    assert {record["pid"] for record in recovery["workers"]} == {9001, 9002}
    assert len(runner.recover_unregistered_workers(suite.state, suite.output, reject_duplicates=False)) == 2


def test_allocation_error_does_not_restore_gpu_with_unjournaled_worker(runner, orphan_suite, monkeypatch):
    suite = orphan_suite
    suite.state["old_suite"] = str(suite.output.parent / "old_suite")
    journal = suite.output / "old_gpu_allocation.json"
    journal.write_text("{}")
    suite.processes[9001] = {"pid": 9001, "start_ticks": "901", "command": suite.train}
    monkeypatch.setattr(runner, "set_old_gpus", lambda *a, **k: pytest.fail("A live orphan still owns this GPU"))
    with pytest.raises(RuntimeError, match="after spawn"):
        with runner.allocation_lock(suite.output / "allocation.lock", suite.output, suite.state, journal):
            raise RuntimeError("after spawn")
    assert read_json(suite.output / "matrix.json")["jobs"][0]["pid"] == 9001
    assert suite.state["status"] == "controller_failed"


def test_failed_orphan_scan_conservatively_preserves_gpu_and_original_error(runner, orphan_suite, monkeypatch):
    suite = orphan_suite
    suite.state["old_suite"] = str(suite.output.parent / "old_suite")
    journal = suite.output / "old_gpu_allocation.json"
    journal.write_text("{}")
    monkeypatch.setattr(runner, "set_old_gpus", lambda *a, **k: pytest.fail("An incomplete scan cannot release the GPU"))
    def failed_scan():
        raise OSError("process scan unavailable")
    monkeypatch.setattr(runner, "live_processes", failed_scan)
    with pytest.raises(RuntimeError, match="original error"):
        with runner.allocation_lock(suite.output / "allocation.lock", suite.output, suite.state, journal):
            raise RuntimeError("original error")
    assert "process scan unavailable" in read_json(suite.output / "matrix.json")["worker_recovery_error"]


def test_secondary_gpu_orphan_uses_container_environment_when_nvml_has_host_pids(runner, orphan_suite, monkeypatch):
    suite = orphan_suite
    suite.state["gpus"] = [suite.state["gpu"], {"uuid": "GPU-0"}]
    suite.processes[9001] = {"pid": 9001, "start_ticks": "901", "command": suite.train}
    monkeypatch.setattr(runner, "process_memory", lambda: {225388: {"gpu_uuid": "GPU-0", "mib": 1000}})
    monkeypatch.setattr(runner, "process_cuda_selector", lambda pid: "GPU-0" if pid == 9001 else None)
    runner.recover_unregistered_workers(suite.state, suite.output)
    assert suite.job["gpu_uuid"] == "GPU-0"
    assert read_json(suite.output / "matrix.json")["jobs"][0]["gpu_uuid"] == "GPU-0"


def test_multiple_gpu_locks_release_on_controller_failure(runner, orphan_suite, monkeypatch, tmp_path):
    suite = orphan_suite
    suite.state["gpus"] = [suite.state["gpu"], {"uuid": "GPU-0"}]
    lock_root = tmp_path / "locks"
    lock_root.mkdir()
    monkeypatch.setattr(runner, "GPU_LOCK_ROOT", lock_root)
    primary, secondary = lock_root / "GPU-1.lock", lock_root / "GPU-0.lock"
    with pytest.raises(RuntimeError, match="multi-gpu failure"):
        with runner.allocation_lock(primary, suite.output, suite.state, suite.output / "journal.json"):
            assert runner._lock_held(primary) and runner._lock_held(secondary)
            assert set(suite.state["claimed_gpu_uuids"]) == {"GPU-1", "GPU-0"}
            raise RuntimeError("multi-gpu failure")
    assert not runner._lock_held(primary) and not runner._lock_held(secondary)
    assert suite.state["status"] == "controller_failed"


def test_optional_gpu_locked_elsewhere_does_not_block_primary(runner, orphan_suite, monkeypatch, tmp_path):
    suite = orphan_suite
    suite.state["gpus"] = [suite.state["gpu"], {"uuid": "GPU-0"}]
    lock_root = tmp_path / "locks"
    lock_root.mkdir()
    monkeypatch.setattr(runner, "GPU_LOCK_ROOT", lock_root)
    primary, secondary = lock_root / "GPU-1.lock", lock_root / "GPU-0.lock"
    with secondary.open("a") as other:
        runner.fcntl.flock(other, runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB)
        with runner.allocation_lock(primary, suite.output, suite.state, suite.output / "journal.json"):
            assert runner.eligible_gpus(suite.state) == [suite.state["gpu"]]
            assert suite.state["claimed_gpu_uuids"] == ["GPU-1"]
        assert not runner._lock_held(primary) and runner._lock_held(secondary)
