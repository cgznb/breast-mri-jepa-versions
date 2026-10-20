#!/usr/bin/env python3
"""Frozen VQ ablations with measured multi-GPU admission and evaluated-weight cleanup."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import replace
from datetime import datetime
import errno
import fcntl
import gc
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from mri_vla_jepa.ablation_suite import (
    prepare_suite, write_suite_reports, complete_evaluation_receipt,
    delete_evaluated_checkpoints,
)
from mri_vla_jepa.data import RawMRIStore
from mri_vla_jepa.diagnostics import evaluate_world
from mri_vla_jepa.io import digest, read_json, stable_hash, write_json
from mri_vla_jepa.train_config import load_config
from mri_vla_jepa.training import _source_snapshot, evaluate, load_trained
from run_capacity_phase_matrix import (
    ROOT, gpu_inventory, gpu_memory_used, process_memory, freeze_runtime, runtime_hashes, load_queue,
)

GPU_LOCK_ROOT = Path("/personal_fast_storage/.vla_gpu_allocations")


def suite_gpus(state):
    return state.get("gpus") or [state["gpu"]]


def gpu_profiles(state, gpu):
    """Memory measurements transfer between identical accelerator models."""
    primary = state["gpu"]
    if (gpu["uuid"] == primary["uuid"] or
            (gpu.get("name") and gpu.get("name") == primary.get("name")
             and gpu.get("total_mib") == primary.get("total_mib"))):
        return state.setdefault("profiles", {})
    return state.setdefault("gpu_profiles", {}).setdefault(gpu["uuid"], {})


def old_suite_drained(old_root):
    if not old_root or not (Path(old_root) / "matrix.json").exists():
        return True
    old = read_json(Path(old_root) / "matrix.json")
    return all(job["status"] in {"completed", "failed", "paused"} and not same_process(job)
               for job in old["jobs"])


def eligible_gpus(state):
    deferred = set(state.get("deferred_gpu_uuids", []))
    drained = old_suite_drained(state.get("old_suite")) if deferred else True
    state["deferred_gpu_status"] = "released_after_old_suite_drained" if drained else "waiting_for_old_suite"
    claimed = set(state.get("claimed_gpu_uuids", [gpu["uuid"] for gpu in suite_gpus(state)]))
    return [gpu for gpu in suite_gpus(state) if gpu["uuid"] in claimed
            and (gpu["uuid"] not in deferred or drained)]


def process_cuda_selector(pid):
    try:
        values = (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
        return next(value.split(b"=", 1)[1].decode() for value in values
                    if value.startswith(b"CUDA_VISIBLE_DEVICES="))
    except (FileNotFoundError, PermissionError, StopIteration):
        return None


def gpu_worker_uuid(info, state, observations):
    """Recover a child's GPU even before CUDA has created an NVML allocation."""
    gpus = suite_gpus(state)
    if len(gpus) == 1:
        return gpus[0]["uuid"]
    measured = observations.get(info["pid"], {}).get("gpu_uuid")
    if measured in {gpu["uuid"] for gpu in gpus}:
        return measured
    selector = process_cuda_selector(info["pid"])
    for gpu in gpus:
        if selector in {gpu["uuid"], str(gpu.get("index"))}:
            return gpu["uuid"]
    raise RuntimeError("Cannot identify the GPU of an unjournaled frozen-runtime worker")


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def durable_json(path, value):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + ".controller.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o600)
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def process_info(pid):
    if not pid:
        return None
    try:
        root = Path("/proc") / str(pid)
        fields = (root / "stat").read_text().split(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "state": fields[0], "start_ticks": fields[19],
                "command": [p.decode() for p in (root / "cmdline").read_bytes().split(b"\0") if p]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(record):
    info = process_info(record.get("pid"))
    return info if info and str(info["start_ticks"]) == str(record.get("start_ticks")) else None


def live_processes():
    """Read process identities without relying on a controller's child registry."""
    for entry in Path("/proc").iterdir():
        if entry.name.isdecimal():
            try:
                info = process_info(int(entry.name))
            except PermissionError:
                continue
            if info is not None:
                yield info


def _worker_command_phase(command, job, runtime, output):
    """Recognize only commands emitted for this frozen runtime and exact job."""
    if len(command) < 2:
        return None
    if command[1] == str(runtime / "train.py"):
        if len(command) < 3 or command[2] != "train-vla-jepa":
            return None
        arguments, allowed, phase = command[3:], {"--config", "--manifest", "--output"}, "running"
        expected = {"--config": str(job["config"]), "--manifest": str(job["manifest"]),
                    "--output": str(Path(job["output"]))}
        resume_allowed = True
    elif command[1] == str(runtime / "scripts/run_unfrozen_ablation.py"):
        arguments, allowed, phase = command[2:], {"--output", "--evaluate-job"}, "evaluating"
        expected = {"--output": str(output), "--evaluate-job": job["name"]}
        resume_allowed = False
    else:
        return None
    values, resumed, index = {}, False, 0
    while index < len(arguments):
        flag = arguments[index]
        if flag == "--resume" and resume_allowed and not resumed:
            resumed = True
            index += 1
        elif flag in allowed and flag not in values and index + 1 < len(arguments):
            values[flag] = arguments[index + 1]
            index += 2
        else:
            return None
    return phase if values == expected else None


def recover_unregistered_workers(state, output, *, reject_duplicates=True):
    """Persist workers lost between spawn and journal commit before adopting them.

    Existing PID/start identities remain valid independently of command discovery.
    Extra matches are never signalled; duplicate live workers prevent dispatch.
    """
    output = Path(output).resolve()
    runtime = Path(state.get("runtime", ROOT)).resolve()
    candidates = {job["name"]: {} for job in state["jobs"]}
    observations = None
    for job in state["jobs"]:
        info = same_process(job)
        if info:
            candidates[job["name"]][(info["pid"], str(info["start_ticks"]))] = {
                **info, "job": job["name"], "phase": job["status"], "discovered": False}
    for info in live_processes():
        for job in state["jobs"]:
            phase = _worker_command_phase(info.get("command", []), job, runtime, output)
            if phase:
                key = (info["pid"], str(info["start_ticks"]))
                previous = candidates[job["name"]].get(key)
                if previous and job.get("gpu_uuid"):
                    gpu_uuid = job["gpu_uuid"]
                else:
                    if observations is None:
                        observations = process_memory() if len(suite_gpus(state)) > 1 else {}
                    gpu_uuid = gpu_worker_uuid(info, state, observations)
                candidates[job["name"]][key] = {
                    **info, "job": job["name"], "phase": phase, "gpu_uuid": gpu_uuid,
                    "discovered": previous is None or previous["phase"] != phase}
    duplicates, recovered = [], []
    for job in state["jobs"]:
        workers = list(candidates[job["name"]].values())
        if len(workers) > 1:
            duplicates.append(job["name"])
        elif workers and workers[0]["discovered"]:
            info = workers[0]
            job.update(pid=info["pid"], start_ticks=info["start_ticks"], status=info["phase"],
                       gpu_uuid=info["gpu_uuid"], recovered_at=now(),
                       recovery_reason="Live frozen-runtime worker found before journal commit")
            recovered.append(job["name"])
    records = [record for workers in candidates.values() for record in workers.values()]
    if recovered or duplicates:
        state["worker_recovery"] = {"checked_at": now(), "recovered_jobs": recovered,
            "duplicate_jobs": duplicates, "workers": records}
        durable_json(output / "matrix.json", state)
    if duplicates and reject_duplicates:
        raise RuntimeError("Multiple live workers for the same experiment; refusing dispatch: " + ", ".join(duplicates))
    return records


def spawn(command, log_path, gpu=None):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"}
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    with log_path.open("a", buffering=1) as stream:
        return subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)


def storage_probe(output, required_bytes):
    """Check pool headroom and actual personal write/fsync; EDQUOT is not df space."""
    stat = os.statvfs(output)
    if stat.f_bavail * stat.f_frsize < required_bytes:
        raise OSError(errno.ENOSPC, "Insufficient headroom for active checkpoints and atomic copies")
    probe = Path(output) / ".checkpoint_write_probe"
    try:
        with probe.open("wb") as stream:
            stream.write(b"\0" * 2**20)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        probe.unlink(missing_ok=True)


def _stop_controller(record, expected):
    info = same_process(record)
    if info:
        if str(expected) not in info["command"]:
            raise RuntimeError("Old controller PID command does not match its recorded suite")
        # Signal only the controller PID, never its process group or training workers.
        os.kill(info["pid"], signal.SIGTERM)
        if info.get("state") == "T":
            os.kill(info["pid"], signal.SIGCONT)


def _wait_stopped(records, timeout=60):
    deadline = time.monotonic() + timeout
    while any(same_process(record) for record in records):
        if time.monotonic() > deadline:
            raise TimeoutError("Controller handoff did not release; workers were left untouched")
        time.sleep(.5)


def _lock_held(path):
    with Path(path).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def _old_scheduler(old_root, state, drain):
    """Accept only the recorded frozen controller and optional drain wrapper."""
    controller = process_info(state.get("controller_pid"))
    if controller:
        expected = str(old_root / "runtime/scripts/run_capacity_phase_matrix.py")
        ticks = state.get("controller_start_ticks")
        if expected not in controller["command"] or (ticks and str(ticks) != controller["start_ticks"]):
            raise RuntimeError("Old controller identity changed")
    supervisor = process_info(drain.get("supervisor_pid")) if drain else None
    if supervisor and (supervisor["start_ticks"] != str(drain.get("supervisor_start_ticks"))
                       or str(old_root / "drain_control.py") not in supervisor["command"]):
        raise RuntimeError("Drain supervisor identity changed")
    return controller, supervisor


def _wait_old_scheduler(old_root, target, child, *, drained, timeout=60):
    """Do not commit handoff until the replacement owns its controller lock."""
    deadline = time.monotonic() + timeout
    while True:
        state = read_json(old_root / "matrix.json")
        drain = read_json(old_root / "drain_policy.json") if drained else None
        controller, supervisor = _old_scheduler(old_root, state, drain)
        same_gpus = [g["uuid"] for g in state["gpus"]] == [g["uuid"] for g in target]
        if drained:
            identity_matches = (supervisor and supervisor["pid"] == child.pid and controller
                and drain.get("controller_pid") == controller["pid"]
                and str(drain.get("controller_start_ticks")) == controller["start_ticks"]
                and _lock_held(old_root / "drain_supervisor.lock"))
        else:
            identity_matches = controller and controller["pid"] == child.pid
        if same_gpus and identity_matches and _lock_held(old_root / "controller.lock"):
            return {"controller": controller, "supervisor": supervisor, "terminal": False}
        pending = any(j["status"] in {"running", "evaluating", "queued", "evaluation_pending"}
                      for j in state["jobs"])
        if same_gpus and not pending and (not drain or drain.get("status") == "paused"):
            return {"controller": controller, "supervisor": supervisor, "terminal": True}
        if child.poll() is not None:
            raise RuntimeError("Replacement old controller exited before confirmed handoff; see gpu_handoff.log")
        if time.monotonic() >= deadline:
            raise TimeoutError("Replacement old controller did not acquire its recorded lock and identity")
        time.sleep(.2)


def set_old_gpus(old_root, gpus, journal_path, *, restoring=False):
    """Hand off only the old controller; respect an existing user drain policy."""
    old_root = Path(old_root)
    if not (old_root / "matrix.json").exists():
        return
    state = read_json(old_root / "matrix.json")
    journal = read_json(journal_path) if Path(journal_path).exists() else {
        "old_suite": str(old_root), "original_gpus": state["gpus"], "created_at": now(),
        "original_workers": {j["name"]: {"pid": j.get("pid"), "start_ticks": j.get("start_ticks")}
                             for j in state["jobs"] if same_process(j)},
    }
    if Path(journal["old_suite"]).resolve() != old_root.resolve():
        raise ValueError("GPU allocation journal belongs to a different old suite")
    target = journal["original_gpus"] if restoring else gpus
    drain_path = old_root / "drain_policy.json"
    drain = read_json(drain_path) if drain_path.exists() else None
    controller, supervisor = _old_scheduler(old_root, state, drain)
    pending = any(j["status"] in {"running", "evaluating", "queued", "evaluation_pending"}
                  for j in state["jobs"])
    if drain:
        held = set(drain.get("held_jobs", []))
        if any(j["name"] in held and j["status"] != "paused" for j in state["jobs"]):
            raise RuntimeError("Existing drain hold changed; refusing automatic experiment release")
        if drain.get("status") == "paused" and pending:
            raise RuntimeError("Paused drain has unfinished jobs; refusing automatic resume")
    healthy = bool(controller and (not drain or supervisor)
                   and _lock_held(old_root / "controller.lock")
                   and (not drain or _lock_held(old_root / "drain_supervisor.lock")))
    same_gpus = [g["uuid"] for g in state["gpus"]] == [g["uuid"] for g in target]
    if same_gpus and (not pending or healthy):
        journal.update(status="restored" if restoring else "restricted", updated_at=now())
        durable_json(journal_path, journal)
        return
    if not pending and supervisor:
        # Let an already drained wrapper commit its paused state and reports;
        # terminating it here could strand a completed drain as "draining".
        _wait_stopped([supervisor] + ([controller] if controller else []))
        state = read_json(old_root / "matrix.json")
        drain = read_json(drain_path)
        controller, supervisor = _old_scheduler(old_root, state, drain)
    journal.update(status="restoring" if restoring else "handoff", updated_at=now())
    durable_json(journal_path, journal)
    records = []
    # Stop the wrapper first so it cannot write an obsolete terminal snapshot.
    if supervisor:
        expected = str(old_root / "drain_control.py")
        _stop_controller(supervisor, expected)
        records.append(supervisor)
    if drain and drain.get("controller_pid") != state.get("controller_pid"):
        other = same_process({"pid": drain.get("controller_pid"),
                              "start_ticks": drain.get("controller_start_ticks")})
        if other:
            _stop_controller(other, old_root / "runtime/scripts/run_capacity_phase_matrix.py")
            records.append(other)
    if controller:
        _stop_controller(controller, old_root / "runtime/scripts/run_capacity_phase_matrix.py")
        records.append(controller)
    _wait_stopped(records)
    with (old_root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read_json(old_root / "matrix.json")
        state.update(gpus=target, controller_pid=None, controller_start_ticks=None, updated_at=now())
        durable_json(old_root / "matrix.json", state)
    pending = any(j["status"] in {"running", "evaluating", "queued", "evaluation_pending"}
                  for j in state["jobs"])
    if pending:
        if drain:
            command = [drain["python"], str(old_root / "drain_control.py"),
                       "--supervise", "--output", str(old_root)]
        else:
            command = [sys.executable, str(old_root / "runtime/scripts/run_capacity_phase_matrix.py"),
                       "--worker", "--output", str(old_root)]
        env = {**os.environ, "PYTHONPATH": str(old_root / "runtime/src"), "PYTHONUNBUFFERED": "1"}
        with (old_root / "gpu_handoff.log").open("a", buffering=1) as stream:
            child = subprocess.Popen(command, cwd=old_root / "runtime", env=env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        journal["replacement_controller"] = process_info(child.pid)
        durable_json(journal_path, journal)
        journal["replacement_ready"] = _wait_old_scheduler(old_root, target, child, drained=bool(drain))
    latest = {j["name"]: j for j in state["jobs"]}
    journal["workers_after_handoff"] = {
        name: {"same_identity_alive": bool(same_process(identity)),
               "completed_naturally": bool((Path(latest[name]["output"]) / "train_report.json").exists()
                 and read_json(Path(latest[name]["output"]) / "train_report.json").get("completed"))}
        for name, identity in journal["original_workers"].items()}
    journal.update(status="restored" if restoring else "restricted", updated_at=now())
    durable_json(journal_path, journal)


def old_gpu_busy(old_root, uuid):
    path = Path(old_root) / "matrix.json"
    if path.exists():
        state = read_json(path)
        # Queued work can migrate to the retained GPUs. Its previous GPU UUID
        # is historical allocation, not a live reservation after handoff.
        if any(j.get("gpu_uuid") == uuid and j["status"] in {"running", "evaluating"}
               and same_process(j) for j in state["jobs"]):
            return True
    return gpu_memory_used().get(uuid, 0) > 1024


def validate_identity(state):
    if state["source_identity"] != stable_hash(_source_snapshot()) or state["runtime_hashes"] != runtime_hashes(ROOT):
        raise ValueError("Frozen suite runtime changed")
    if digest(state["codec"]) != state["codec_sha256"]:
        raise ValueError("Original pretrained codec changed")
    signatures = {}
    for job in state["jobs"]:
        cfg = load_config(job["config"])
        if cfg.digest != job["config_digest"]:
            raise ValueError("Frozen experiment config changed")
        if Path(cfg.encoder_checkpoint).resolve() != Path(state["codec"]).resolve():
            raise ValueError("Experiment codec differs from the frozen pretrained codec")
        if digest(job["manifest"]) != job["manifest_sha256"]:
            raise ValueError("Frozen patient split changed")
        if job["seed"] not in signatures:
            signatures[job["seed"]] = (job["manifest_sha256"], stable_hash(RawMRIStore(
                job["manifest"], cfg.image_shape, cfg.allow_synthetic, image_cache=cfg.image_cache).signature()))
        if signatures[job["seed"]][0] != job["manifest_sha256"]:
            raise ValueError("Same-seed experiments must use the same frozen patient split")
        if signatures[job["seed"]][1] != state["data_signatures"][str(job["seed"])]:
            raise ValueError("MRI data identity changed")


def _profile_record(child, command, target, log):
    info = process_info(child.pid)
    if info is None and child.poll() is None:
        raise RuntimeError("Cannot establish preflight process identity")
    return {"pid": child.pid, "start_ticks": info["start_ticks"] if info else None,
            "command": list(command), "target": str(target), "log": str(log)}


def recover_preflight_children(state, output):
    """Reclaim recorded orphan profiles before remeasurement, without touching training."""
    record = state.get("active_preflight")
    if not record:
        return
    records = record.get("children", [record] if record.get("pid") else [])
    if not records and "children" not in record:
        raise ValueError("Active preflight lacks recorded child process identities")
    for child_record in records:
        info = same_process(child_record)
        if info is None:
            continue
        command = child_record.get("command", [])
        if (not command or info["command"] != command
                or str(ROOT / "scripts/profile_capacity_phase.py") not in command):
            raise RuntimeError("Recorded preflight PID identity mismatch")
        os.kill(info["pid"], signal.SIGTERM)
        deadline = time.monotonic() + 30
        while same_process(child_record):
            if time.monotonic() >= deadline:
                os.kill(info["pid"], signal.SIGKILL)
                _wait_stopped([child_record], timeout=10)
                break
            time.sleep(.2)
    state.pop("active_preflight", None)
    durable_json(Path(output) / "matrix.json", state)


def _finish_profile_children(workers):
    for child in workers:
        if child.poll() is None:
            child.terminate()
    for child in workers:
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)


def checked_child(command, log, gpu, *, state, output, stage, max_seconds=3600):
    """Profile worker with persistent heartbeat and a measured device peak."""
    child = spawn(command, log, gpu)
    peak, started = 0, time.monotonic()
    try:
        target = command[command.index("--output") + 1]
        state.update(status=stage, active_preflight={"children": [
            _profile_record(child, command, target, log)]}, updated_at=now())
        durable_json(output / "matrix.json", state)
        while child.poll() is None:
            peak = max(peak, gpu_memory_used().get(gpu["uuid"], 0))
            state.update(updated_at=now())
            durable_json(output / "matrix.json", state)
            if peak > gpu["budget_mib"] or time.monotonic() - started > max_seconds:
                raise RuntimeError(f"Preflight exceeded memory/time budget: {log}")
            time.sleep(2)
    finally:
        _finish_profile_children([child])
        state.pop("active_preflight", None)
        durable_json(output / "matrix.json", state)
    if child.returncode:
        raise RuntimeError(f"Preflight failed with exit {child.returncode}: {log}")
    return peak


def preflight(state, output, gpu=None):
    """Measure complete batch-16 lifecycles; concurrency follows memory, without a task cap."""
    recover_preflight_children(state, output)
    gpu = gpu or state["gpu"]
    profiles = gpu_profiles(state, gpu)
    folder = output / "preflight" / gpu["uuid"]
    folder.mkdir(parents=True, exist_ok=True)
    for job in state["jobs"]:
        if job["seed"] != 17:
            continue
        if job["variant"] in profiles:
            cached = profiles[job["variant"]]
            if (cached.get("source_identity") != state["source_identity"]
                    or cached.get("config_digest") != job["config_digest"]
                    or cached.get("manifest_sha256") != job["manifest_sha256"]
                    or cached.get("status") != "passed"):
                raise ValueError("Cached preflight differs from the frozen experiment")
            if cached["reservation_mib"] > gpu["budget_mib"]:
                raise RuntimeError(f"Batch 16 cannot fit measured memory budget: {job['variant']}")
            continue
        target = folder / f"{job['variant']}.json"
        command = [sys.executable, str(ROOT / "scripts/profile_capacity_phase.py"),
                   "--config", job["config"], "--manifest", job["manifest"], "--output", str(target)]
        peak = checked_child(command, target.with_suffix(".log"), gpu, state=state,
                             output=output, stage="preflight")
        result = read_json(target)
        if result.get("status") != "passed" or result.get("config") != str(Path(job["config"]).resolve()):
            raise ValueError("Preflight result does not identify the requested configuration")
        if result.get("batch_size") != 16:
            raise ValueError("Preflight must measure the unchanged batch size 16")
        peaks = [float(result[key]) for key in (
            "peak_reserved_mib", "training_peak_reserved_mib", "inference_peak_reserved_mib",
            "diagnostic_peak_reserved_mib") if result.get(key) is not None]
        if not peaks or any(not math.isfinite(value) or value <= 0 for value in peaks):
            raise ValueError("Preflight memory measurements are missing or invalid")
        result.update(nvml_peak_mib=peak, reservation_mib=math.ceil(max(peak, *peaks) + 512),
                      measured_gpu=dict(gpu), source_identity=state["source_identity"],
                      config_digest=job["config_digest"], manifest_sha256=job["manifest_sha256"])
        if result["reservation_mib"] > gpu["budget_mib"]:
            raise RuntimeError(f"Batch 16 cannot fit measured memory budget: {job['variant']}")
        durable_json(target, result)
        profiles[job["variant"]] = result
        durable_json(output / "matrix.json", state)
    state.pop("limits", None)
    state["admission_policy"] = {"strategy": "measured_reservations_and_live_nvml",
        "memory_fraction": .9, "batch_size": 16, "mixed_variants": True,
        "task_count_cap": None, "reservation_margin_mib": 512,
        "measurement_scope": "training, validation, scheduled diagnostics, all configured landmarks",
        "equivalent_gpu_measurements": "same GPU model and total memory"}
    durable_json(output / "matrix.json", state)
    return profiles


def update_memory_measurements(state, active, used, observations):
    """Combine reserved peaks with device NVML when host/container PIDs differ."""
    snapshot, grouped = {}, {}
    baseline = state.setdefault("gpu_foreign_baseline_mib", {})
    for gpu in suite_gpus(state):
        workers = [job for job in state["jobs"] if job["name"] in active
                   and job.get("gpu_uuid", state["gpu"]["uuid"]) == gpu["uuid"]]
        grouped[gpu["uuid"]] = workers
        profiles = gpu_profiles(state, gpu)
        for job in workers:
            record = observations.get(job.get("pid"), {})
            measured = record.get("mib", 0) if record.get("gpu_uuid") == gpu["uuid"] else 0
            job["observed_peak_mib"] = max(job.get("observed_peak_mib", 0), measured)
            profile = profiles.get(job["variant"])
            if profile and measured > 0 and measured + 512 > profile["reservation_mib"]:
                profile["reservation_mib"] = math.ceil(measured + 512)
                profile["runtime_adjustment"] = "Expanded after live per-process NVML measurement"
    # A shared profile may have expanded on either identical GPU above.
    for gpu in suite_gpus(state):
        workers, profiles = grouped[gpu["uuid"]], gpu_profiles(state, gpu)
        actual = used.get(gpu["uuid"], 0)
        if not workers:
            baseline[gpu["uuid"]] = min(baseline.get(gpu["uuid"], actual), actual)
        matched = [observations.get(job.get("pid"), {}) for job in workers]
        mapped = bool(workers) and all(record.get("gpu_uuid") == gpu["uuid"] and
                                      "mib" in record for record in matched)
        foreign = max(0, actual - sum(record["mib"] for record in matched)) if mapped else baseline.get(gpu["uuid"], 0)
        reserved = sum(profiles[job["variant"]]["reservation_mib"] for job in workers)
        snapshot[gpu["uuid"]] = {"budget_mib": gpu["budget_mib"], "nvml_used_mib": actual,
            "reserved_mib": reserved, "foreign_mib": foreign,
            "foreign_baseline_mib": baseline.get(gpu["uuid"], 0), "pid_mapping_available": mapped,
            "active_jobs": [job["name"] for job in workers]}
    state["gpu_memory_state"] = snapshot
    return snapshot


def choose_gpu(state, job, active, gpus, used, observations):
    """Reserve measured peak headroom even while newly spawned CUDA workers initialize."""
    memory = update_memory_measurements(state, active, used, observations)
    choices = []
    for gpu in gpus:
        profiles = gpu_profiles(state, gpu)
        if job["variant"] not in profiles:
            continue
        current = memory[gpu["uuid"]]
        workers = [item for item in state["jobs"] if item["name"] in current["active_jobs"]]
        if (workers and (job.get("exclusive_retry") or any(item.get("exclusive_retry") for item in workers))):
            continue
        required = profiles[job["variant"]]["reservation_mib"]
        committed = max(current["reserved_mib"] + current["foreign_mib"], current["nvml_used_mib"])
        if committed + required <= gpu["budget_mib"]:
            choices.append((committed / gpu["budget_mib"], gpu))
    return min(choices, key=lambda item: item[0])[1] if choices else None


def evaluate_job(output, name):
    import torch
    state = read_json(output / "matrix.json")
    validate_identity(state)
    job = next(j for j in state["jobs"] if j["name"] == name)
    run = Path(job["output"])
    receipt = run / "evaluation_complete.json"
    # Restart safely after receipt commit / partial deletion, without needing weights again.
    if receipt.exists():
        deleted = delete_evaluated_checkpoints(run, output, protected_paths=[state["codec"]])
        durable_json(run / "evaluation_done.json", {"completed": True, "completed_at": now(),
                     "config_digest": job["config_digest"], "cleanup": deleted})
        return
    queue = load_queue()
    report = queue.validate_completed_run(job, Path(job["manifest"]))
    checkpoint_epochs = {"best": report.get("best_epoch"), "last": report.get("completed_epochs")}
    if (any(type(epoch) is not int or epoch < 1 for epoch in checkpoint_epochs.values())
            or checkpoint_epochs["best"] > checkpoint_epochs["last"]):
        raise ValueError("Completed run lacks valid best/last checkpoint epoch identities")
    cfg = load_config(job["config"])
    diagnostic_path = run / "diagnostics.json"
    diagnostic_reports = read_json(diagnostic_path) if diagnostic_path.exists() else []
    expected_epochs = list(range(cfg.diagnostics_every_epochs, report["completed_epochs"] + 1,
                                 cfg.diagnostics_every_epochs)) if cfg.diagnostics_every_epochs else []
    if (not isinstance(diagnostic_reports, list)
            or [item.get("epoch") for item in diagnostic_reports] != expected_epochs):
        raise ValueError("Training diagnostics do not cover the scheduled completed epochs")
    diagnostic_record = run / "diagnostics_record.json"
    durable_json(diagnostic_record, {"schema": "verified_training_diagnostics_v1",
        "completed_epochs": report["completed_epochs"], "every_epochs": cfg.diagnostics_every_epochs,
        "reports": diagnostic_reports})
    required = {"training": {"path": str(run / "train_report.json"), "kind": "json"},
                "diagnostics": {"path": str(diagnostic_record), "kind": "json"}}
    # Classification main tables always use the validation-selected best checkpoint.
    for kind, split in (("best", "val"), ("best", "test"), ("last", "val")):
        target = run / f"evaluation_{kind}_{split}.json"
        result = evaluate(run / f"{kind}.pt", job["manifest"], target, split=split,
                          device="cuda:0", bootstrap=0)
        if kind == "best" and split == "val" and not math.isclose(
                result["selection_score"], report["best_selection_score"], rel_tol=0, abs_tol=1e-7):
            raise ValueError("Recomputed best validation F1 differs from selected checkpoint")
        result.update(checkpoint_kind=kind, checkpoint_epoch=checkpoint_epochs[kind])
        durable_json(target, result)
        required[f"{kind}_{split}"] = {"path": str(target), "kind": "classification", "split": split}
        gc.collect()
        torch.cuda.empty_cache()
    for kind, split in (("best", "val"), ("best", "test"), ("last", "val")):
        if not getattr(cfg.model, "enable_world_model", True):
            target = run / f"world_{kind}_{split}.json"
            durable_json(target, {"status": "not_applicable", "world_head_trained": False,
                "reason": "The direct T0 classification control has no world model branch",
                "split": split, "landmarks": cfg.landmarks,
                "checkpoint_kind": kind, "checkpoint_epoch": checkpoint_epochs[kind],
                "independence_scope": "downstream_only", "independent_test": False})
            required[f"world_{kind}_{split}"] = {
                "path": str(target), "kind": "world", "split": split, "applicable": False}
            continue
        model, cfg, saved = load_trained(run / f"{kind}.pt", "cuda:0")
        if saved.get("source_snapshot") != _source_snapshot():
            raise ValueError("World evaluation source differs from checkpoint")
        if saved.get("epoch_state", {}).get("completed_epochs") != checkpoint_epochs[kind]:
            raise ValueError("Checkpoint epoch identity differs from the completed training report")
        torch.set_num_threads(cfg.threads)
        store = RawMRIStore(job["manifest"], cfg.image_shape, cfg.allow_synthetic, saved["normalization"])
        result = evaluate_world(model, store, replace(cfg, device="cuda:0", precision="fp32"),
                                split=split, batch_size=cfg.validation_batch_size)
        result.update(status="evaluated", checkpoint_kind=kind, checkpoint_epoch=checkpoint_epochs[kind],
                      independence_scope="downstream_only",
                      independent_test=False, runtime_source_matches_checkpoint=True)
        target = run / f"world_{kind}_{split}.json"
        durable_json(target, result)
        required[f"world_{kind}_{split}"] = {"path": str(target), "kind": "world", "split": split}
        del model, saved, store
        gc.collect()
        torch.cuda.empty_cache()
    complete_evaluation_receipt(run, output, required, manifest=job["manifest"])
    deleted = delete_evaluated_checkpoints(run, output, protected_paths=[state["codec"]])
    durable_json(run / "evaluation_done.json", {"completed": True, "completed_at": now(),
                 "config_digest": job["config_digest"], "cleanup": deleted})


@contextmanager
def additional_gpu_locks(state):
    """Claim optional GPUs without interfering with an existing suite's workers."""
    with ExitStack() as locks:
        state["claimed_gpu_uuids"] = [state["gpu"]["uuid"]]
        for gpu in suite_gpus(state):
            if gpu["uuid"] == state["gpu"]["uuid"]:
                continue
            GPU_LOCK_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
            handle = (GPU_LOCK_ROOT / f"{gpu['uuid']}.lock").open("a")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            locks.enter_context(handle)
            state["claimed_gpu_uuids"].append(gpu["uuid"])
        yield


@contextmanager
def allocation_lock(allocation, output, state, journal):
    """Keep errors durable and return GPU allocation only when our workers are gone."""
    with allocation.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with additional_gpu_locks(state):
                yield
        except BaseException as error:
            state.update(status="controller_failed", updated_at=now(),
                         controller_error=f"{type(error).__name__}: {error}")
            durable_json(output / "matrix.json", state)
            discovery_failed, discovered = False, []
            try:
                discovered = recover_unregistered_workers(state, output, reject_duplicates=False)
            except Exception as discovery_error:
                # An incomplete scan cannot prove the GPU is safe to hand back.
                discovery_failed = True
                state["worker_recovery_error"] = f"{type(discovery_error).__name__}: {discovery_error}"
                try:
                    durable_json(output / "matrix.json", state)
                except Exception:
                    pass  # Preserve the original controller error if storage also failed.
            records = list(state["jobs"]) + state.get("active_preflight", {}).get("children", []) + discovered
            if (not discovery_failed and state.get("old_suite") and journal.exists()
                    and not any(same_process(r) for r in records)):
                try:
                    set_old_gpus(state["old_suite"], [], journal, restoring=True)
                except Exception as restore_error:
                    state["allocation_restore_error"] = f"{type(restore_error).__name__}: {restore_error}"
                    durable_json(output / "matrix.json", state)
            raise


def worker(output):
    output = output.resolve()
    with (output / "controller.lock").open("a") as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read_json(output / "matrix.json")
        validate_identity(state)
        for gpu in suite_gpus(state):
            if "total_mib" in gpu:
                gpu["budget_mib"] = math.floor(gpu["total_mib"] * .9)
        state["gpu"] = next(gpu for gpu in suite_gpus(state) if gpu["uuid"] == state["gpu"]["uuid"])
        state.update(controller_pid=os.getpid(), controller_start_ticks=process_info(os.getpid())["start_ticks"])
        journal = output / "old_gpu_allocation.json"
        # A fixed workspace lock also coordinates suites stored in different parents.
        allocation = GPU_LOCK_ROOT / f"{state['gpu']['uuid']}.lock"
        allocation.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with allocation_lock(allocation, output, state, journal):
            recover_unregistered_workers(state, output)
            active = {}
            # Adopt before looking at whole-device memory: our own surviving worker
            # must not be mistaken for a foreign allocation after controller restart.
            for job in state["jobs"]:
                info = same_process(job)
                if info:
                    if job["status"] not in {"running", "evaluating"} or job["name"] not in " ".join(info["command"]):
                        raise RuntimeError("Recorded worker identity mismatch")
                    job.setdefault("gpu_uuid", state["gpu"]["uuid"])
                    if job["gpu_uuid"] not in state["claimed_gpu_uuids"]:
                        raise RuntimeError("A surviving worker's GPU allocation is held by another controller")
                    active[job["name"]] = None
                elif job["status"] in {"running", "evaluating", "queued", "evaluation_pending"}:
                    done = Path(job["output"]) / "evaluation_done.json"
                    report = Path(job["output"]) / "train_report.json"
                    if report.exists() and read_json(report).get("completed"):
                        job["best_epoch"] = read_json(report).get("best_epoch")
                    job["status"] = ("completed" if done.exists() and read_json(done).get("completed") else
                        "evaluation_pending" if report.exists() and read_json(report).get("completed") else "queued")
            if state.get("old_suite"):
                gpu0 = gpu_inventory(["0"])
                set_old_gpus(state["old_suite"], gpu0, journal)
            if state.get("active_preflight"):
                recover_preflight_children(state, output)
            if not active and any(j["status"] != "completed" for j in state["jobs"]):
                while old_gpu_busy(state.get("old_suite") or output / "no_old_suite", state["gpu"]["uuid"]):
                    state.update(status="waiting_for_gpu", updated_at=now(),
                                 wait_reason="Training/evaluation or foreign workload still occupies the primary GPU")
                    durable_json(output / "matrix.json", state)
                    time.sleep(10)
                preflight(state, output)
            elif active and any(job["variant"] not in gpu_profiles(state, next(
                    gpu for gpu in suite_gpus(state) if gpu["uuid"] == job["gpu_uuid"]))
                    for job in state["jobs"] if job["name"] in active):
                raise RuntimeError("Live training workers require their already completed preflight policy")
            while True:
                for job in state["jobs"]:
                    if job["name"] not in active:
                        continue
                    child = active[job["name"]]
                    if (child is not None and child.poll() is None) or (child is None and same_process(job)):
                        progress = Path(job["output"]) / "progress.json"
                        if progress.exists():
                            job["progress"] = read_json(progress)
                        continue
                    del active[job["name"]]
                    code = child.returncode if child is not None else None
                    run = Path(job["output"])
                    if job["status"] == "running" and code in (None, 0) and (run / "train_report.json").exists() and read_json(run / "train_report.json").get("completed"):
                        job.update(status="evaluation_pending", best_epoch=read_json(run / "train_report.json").get("best_epoch"),
                                   exclusive_retry=False)
                    elif job["status"] == "evaluating" and code in (None, 0) and (run / "evaluation_done.json").exists():
                        job.update(status="completed", completed_at=now())
                    else:
                        log_path = run / ("training.log" if job["status"] == "running" else "evaluation.log")
                        log = log_path.read_bytes()[job.get("log_offset", 0):].decode(errors="replace")
                        stage = "queued" if job["status"] == "running" else "evaluation_pending"
                        if any(token in log.lower() for token in ("disk quota", "no space left", "edquot")):
                            job.update(status=stage, storage_blocked=True, failure_reason="Checkpoint write/quota failure")
                        elif "out of memory" in log.lower() and job.get("retries", 0) < 2:
                            failed_gpu = next(gpu for gpu in suite_gpus(state) if gpu["uuid"] == job["gpu_uuid"])
                            profile = gpu_profiles(state, failed_gpu)[job["variant"]]
                            profile["reservation_mib"] = math.ceil(max(
                                profile["reservation_mib"] * 1.25, job.get("observed_peak_mib", 0) + 512))
                            job.update(status=stage, retries=job.get("retries", 0) + 1,
                                       exclusive_retry=True,
                                       retry_reason="CUDA OOM: enlarge measured reservation and retry exclusively with batch 16")
                        elif job.get("retries", 0) < 2 and not any(marker in log for marker in (
                                "ValueError:", "TypeError:", "AssertionError:", "FloatingPointError:", "out of memory")):
                            job.update(status=stage, retries=job.get("retries", 0) + 1)
                        else:
                            job.update(status="failed", failure_reason=f"Worker exit {code}; see {log_path}")
                    job.update(pid=None, updated_at=now())
                pending = sorted([j for j in state["jobs"] if j["status"] in {"queued", "evaluation_pending"}],
                                 key=lambda j: j["status"] != "evaluation_pending")
                available = eligible_gpus(state)
                used, observations = gpu_memory_used(), process_memory()
                for gpu in available:
                    if set(gpu_profiles(state, gpu)) != {job["variant"] for job in state["jobs"]}:
                        if any(job["name"] in active and job["gpu_uuid"] == gpu["uuid"] for job in state["jobs"]):
                            raise RuntimeError("Live workers require completed memory measurements on their GPU")
                        if used.get(gpu["uuid"], 0) <= 1024:
                            preflight(state, output, gpu)
                            used, observations = gpu_memory_used(), process_memory()
                update_memory_measurements(state, active, used, observations)
                storage_error = None
                if pending:
                    try:
                        storage_probe(output, (len(active) + 1) * 24 * 2**30 + 8 * 2**30)
                    except OSError as error:
                        storage_error = str(error)
                for job in pending:
                    if storage_error:
                        break
                    if job.get("storage_blocked"):
                        # A small successful probe cannot prove a large quota failure has cleared.
                        storage_error = "A checkpoint quota failure requires a verified capacity repair before admission"
                        break
                    gpu = choose_gpu(state, job, active, available, used, observations)
                    if gpu is None:
                        continue
                    try:
                        storage_probe(output, (len(active) + 1) * 24 * 2**30 + 8 * 2**30)
                    except OSError as error:
                        storage_error = str(error)
                        break
                    run = Path(job["output"])
                    run.mkdir(parents=True, exist_ok=True, mode=0o700)
                    if job["status"] == "queued":
                        command = [sys.executable, str(ROOT / "train.py"), "train-vla-jepa",
                                   "--config", job["config"], "--manifest", job["manifest"], "--output", str(run)]
                        if (run / "last.pt").exists():
                            command.append("--resume")
                        job["status"], log = "running", run / "training.log"
                    else:
                        command = [sys.executable, str(ROOT / "scripts/run_unfrozen_ablation.py"),
                                   "--output", str(output), "--evaluate-job", job["name"]]
                        job["status"], log = "evaluating", run / "evaluation.log"
                    job["log_offset"] = log.stat().st_size if log.exists() else 0
                    child = spawn(command, log, gpu)
                    identity = process_info(child.pid)
                    job.update(pid=child.pid, start_ticks=identity["start_ticks"] if identity else None,
                               started_at=now(), gpu_uuid=gpu["uuid"])
                    active[job["name"]] = child
                    durable_json(output / "matrix.json", state)
                    used, observations = gpu_memory_used(), process_memory()
                complete = all(j["status"] == "completed" for j in state["jobs"])
                state.update(status="completed" if complete else "storage_blocked" if storage_error else
                             "running" if active else "waiting_for_memory" if pending else "failed",
                             updated_at=now(), storage_error=storage_error)
                durable_json(output / "matrix.json", state)
                write_suite_reports(output, state["jobs"], suite_complete=complete)
                if complete or (not active and not pending):
                    break
                time.sleep(10 if storage_error else 3)
            # Restore the old allocation while still owning the shared GPU lock.
            if state.get("old_suite"):
                set_old_gpus(state["old_suite"], [], journal, restoring=True)
        return 0 if state["status"] == "completed" else 1


def prepare(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (output / "matrix.json").exists():
        raise FileExistsError("Suite already prepared; use --resume")
    if not output.is_relative_to(Path("/personal_fast_storage")):
        raise ValueError("This suite's weights must remain on personal storage")
    jobs = prepare_suite(output, args.base_config, args.manifest, args.codec)
    runtime = freeze_runtime(output)
    signatures = {}
    for job in jobs:
        key = str(job["seed"])
        if key not in signatures:
            signatures[key] = stable_hash(RawMRIStore(job["manifest"]).signature())
    primary = gpu_inventory([args.gpu])[0]
    deferred_selector = getattr(args, "deferred_gpu", "0")
    gpus, deferred = [primary], []
    if deferred_selector and deferred_selector.lower() != "none":
        other = gpu_inventory([deferred_selector])[0]
        if other["uuid"] != primary["uuid"]:
            gpus.append(other)
            deferred.append(other["uuid"])
    for gpu in gpus:
        gpu["budget_mib"] = math.floor(gpu["total_mib"] * .9)
    state = {"schema": "vq_unfrozen_ablation_matrix_v1", "created_at": now(), "status": "prepared",
             "jobs": jobs, "runtime": str(runtime), "source_identity": stable_hash(_source_snapshot()),
             "runtime_hashes": runtime_hashes(runtime), "codec": str(args.codec.resolve()),
             "codec_sha256": digest(args.codec), "data_signatures": signatures,
             "gpu": primary, "gpus": gpus, "deferred_gpu_uuids": deferred,
             "profiles": {}, "gpu_profiles": {},
             "admission_policy": {"strategy": "measured_reservations_and_live_nvml", "memory_fraction": .9,
                                  "batch_size": 16, "mixed_variants": True, "task_count_cap": None},
             "old_suite": str(args.old_suite.resolve()) if args.old_suite else None,
             "checkpoint_retention": "delete_generated_checkpoints_only_after_all_outputs_verified",
             "test_release": "only_after_all_scheduled_runs_complete"}
    durable_json(output / "matrix.json", state)
    write_suite_reports(output, jobs, suite_complete=False)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--codec", type=Path)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--deferred-gpu", default="0", help="Join after old workers finish naturally; use none to disable")
    parser.add_argument("--old-suite", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--evaluate-job")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.evaluate_job:
        evaluate_job(args.output.resolve(), args.evaluate_job)
        return 0
    if args.worker:
        return worker(args.output)
    if args.report_only:
        state = read_json(args.output / "matrix.json")
        write_suite_reports(args.output, state["jobs"], suite_complete=state["status"] == "completed")
        return 0
    if not args.resume:
        if any(value is None for value in (args.base_config, args.manifest, args.codec)):
            parser.error("New suites require --base-config, --manifest, --codec")
        state = prepare(args)
    else:
        state = read_json(args.output / "matrix.json")
    if args.prepare_only:
        print({"status": state["status"], "jobs": len(state["jobs"]), "output": str(args.output.resolve())})
        return 0
    runtime = Path(state["runtime"])
    command = [sys.executable, str(runtime / "scripts/run_unfrozen_ablation.py"),
               "--worker", "--output", str(args.output.resolve())]
    env = {**os.environ, "PYTHONPATH": str(runtime / "src"), "PYTHONUNBUFFERED": "1"}
    with (args.output / "controller.log").open("a", buffering=1) as log:
        child = subprocess.Popen(command, cwd=runtime, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print({"controller_pid": child.pid, "jobs": len(state["jobs"]), "output": str(args.output.resolve())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
