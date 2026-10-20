#!/usr/bin/env python3
"""One custom-platform command for the fixed-test, three-task five-fold suite.

The controller never creates a CUDA context. Every GPU operation runs in an
isolated subprocess importing the suite's frozen runtime. No DDP launcher is
needed: one process may use each GPU, or several when measured resources allow.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import errno
import fcntl
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri_vla_jepa.io import digest, read_json, stable_hash


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_output_parent(spec):
    parent = Path(spec["paths"]["output_root"]).parent
    if not parent.is_dir():
        raise FileNotFoundError(f"Output filesystem is unavailable: {parent}. "
                                "Mount the projects storage before starting this suite")


def durable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def controller_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This suite already has an active controller") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({"pid": os.getpid(), "started_at": now()}))
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def process_info(pid):
    if not pid:
        return None
    try:
        root = Path("/proc") / str(pid)
        fields = (root / "stat").read_text().split(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "start_ticks": fields[19],
                "command": [p.decode() for p in (root / "cmdline").read_bytes().split(b"\0") if p]}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def same_process(record):
    info = process_info(record.get("pid"))
    return info if info and str(info["start_ticks"]) == str(record.get("start_ticks")) else None


def reject_distributed_launch(environ=None):
    env = os.environ if environ is None else environ
    for field in ("WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        if int(env.get(field, "1")) > 1:
            raise RuntimeError("Select platform framework 'custom'; do not launch this controller with DDP/torchrun")
    if int(env.get("RANK", "0")) != 0 or int(env.get("LOCAL_RANK", "0")) != 0:
        raise RuntimeError("Only one custom controller may launch the eight-GPU suite")


def _query(arguments):
    return subprocess.check_output(["nvidia-smi", *arguments], text=True, timeout=30)


def discover_gpus(expected=8, environ=None, query=None):
    """Keep only allocated devices, honoring CUDA_VISIBLE_DEVICES order."""
    env, query = os.environ if environ is None else environ, query or _query
    raw = query(["--query-gpu=index,uuid,name,memory.total,memory.used", "--format=csv,noheader,nounits"])
    inventory = []
    for row in csv.reader(io.StringIO(raw)):
        if not row:
            continue
        index, uuid, name, total, used = [v.strip() for v in row]
        inventory.append({"index": int(index), "uuid": uuid, "name": name,
                          "total_mib": int(total), "used_mib": int(used)})
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        # NVIDIA_VISIBLE_DEVICES may be the only allocation boundary in custom containers.
        visible = env.get("NVIDIA_VISIBLE_DEVICES")
        if visible == "all":
            visible = None
        elif visible in {"void", "none"}:
            visible = ""
    if visible is not None:
        selectors = [token.strip() for token in visible.split(",") if token.strip()]
        selected = []
        for token in selectors:
            matches = [g for g in inventory if token == str(g["index"]) or g["uuid"].startswith(token)]
            if len(matches) != 1:
                raise ValueError(f"Cannot map allocated GPU selector {token!r}; full non-MIG GPUs are required")
            selected.append(matches[0])
    else:
        selected = inventory
    if len(selected) != expected or len({g["uuid"] for g in selected}) != expected:
        raise ValueError(f"Expected exactly {expected} distinct allocated GPUs; found {len(selected)}")
    return [{**gpu, "logical_index": i} for i, gpu in enumerate(selected)]


def gpu_observations():
    raw = _query(["--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"])
    used = {r[0].strip(): int(r[1].strip()) for r in csv.reader(io.StringIO(raw)) if r}
    observations = {}
    try:
        raw = _query(["--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"])
        for row in csv.reader(io.StringIO(raw)):
            if len(row) == 3 and row[0].strip().isdigit() and row[2].strip().isdigit():
                observations[int(row[0])] = {"gpu_uuid": row[1].strip(), "mib": int(row[2])}
    except subprocess.SubprocessError:
        pass  # Device totals remain a conservative admission bound.
    return used, observations


def profile_key(gpu, variant):
    return stable_hash({"name": gpu["name"], "total_mib": gpu["total_mib"], "variant": variant})


def choose_gpu(gpus, profiles, active, used, observations=None, *, variant,
               memory_fraction=.9, margin_mib=768, exclusive=False):
    """Reserve full future peaks and all unknown device allocations.

    When NVML reports host PIDs that cannot be mapped, charge both reservations
    and the entire observed allocation. This is conservative rather than assuming
    unmatched memory belongs to our workers.
    """
    observations = observations or {}
    choices = []
    for gpu in gpus:
        profile = profiles.get(profile_key(gpu, variant))
        if not profile:
            continue
        workers = [a for a in active if a.get("gpu_uuid") == gpu["uuid"]]
        if workers and (exclusive or profile.get("exclusive_only")):
            continue
        if any(a.get("exclusive_retry") or profiles[profile_key(gpu, a["variant"])].get("exclusive_only")
               for a in workers):
            continue
        reserved = sum(profiles[profile_key(gpu, a["variant"])]["reservation_mib"] for a in workers)
        mapped = [observations.get(a.get("pid"), {}) for a in workers]
        own = sum(r.get("mib", 0) for r in mapped if r.get("gpu_uuid") == gpu["uuid"])
        foreign = max(0, used.get(gpu["uuid"], 0) - own)
        committed = max(reserved + foreign, used.get(gpu["uuid"], 0))
        budget = math.floor(gpu["total_mib"] * memory_fraction)
        if committed + profile["reservation_mib"] <= budget:
            choices.append((committed / budget, gpu["logical_index"], gpu))
    return min(choices, key=lambda v: v[:2])[2] if choices else None


def runtime_hashes(root):
    root = Path(root)
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def freeze_runtime(output):
    target = Path(output) / "runtime"
    if target.exists():
        # Preparation can be interrupted after the rename but before matrix.json.
        expected = {}
        for directory in ("src", "scripts"):
            for path in (ROOT / directory).rglob("*"):
                if (path.is_file() and "__pycache__" not in path.parts
                        and not any(p.endswith(".egg-info") for p in path.parts)):
                    expected[str(path.relative_to(ROOT))] = digest(path)
        expected["train.py"] = digest(ROOT / "train.py")
        if runtime_hashes(target) == expected:
            return target
        raise FileExistsError(f"Unjournaled runtime differs; refusing to overwrite: {target}")
    temporary = Path(output) / "runtime.preparing"
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(ROOT / "src", temporary / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    shutil.copytree(ROOT / "scripts", temporary / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "train.py", temporary / "train.py")
    os.replace(temporary, target)
    return target


def validate_runtime(state):
    root = Path(state["runtime"])
    if runtime_hashes(root) != state["runtime_hashes"]:
        raise ValueError("Frozen runtime changed; refusing mixed implementation results")


def enter_frozen_controller(state, arguments):
    """Resume/report with the original controller as well as frozen workers."""
    validate_runtime(state)
    runtime = Path(state["runtime"]).resolve()
    script = runtime / "scripts" / "run_t0_fivefold.py"
    if not script.is_file() or not script.resolve().is_relative_to(runtime):
        raise ValueError("Frozen controller is missing or outside its runtime")
    if Path(__file__).resolve() == script:
        import mri_vla_jepa.t0_cv_suite as suite
        if not Path(suite.__file__).resolve().is_relative_to(runtime / "src"):
            raise ValueError("Controller imported mutable suite/report implementation")
        return
    env = os.environ.copy()
    env.update(PYTHONPATH=str(runtime / "src"), PYTHONDONTWRITEBYTECODE="1")
    # Python opens the controller lock with CLOEXEC. Re-execution releases it;
    # the frozen controller acquires it again before touching mutable state.
    os.execve(sys.executable, [sys.executable, str(script), *arguments], env)
    raise RuntimeError("Frozen controller exec unexpectedly returned")


def storage_probe(output, required_bytes):
    output = Path(output)
    stat = os.statvfs(output)
    if stat.f_bavail * stat.f_frsize < required_bytes:
        raise OSError(errno.ENOSPC, "Insufficient capacity for active best/last checkpoints and atomic save copies")
    target = output / f".write_probe.{os.getpid()}"
    try:
        with target.open("wb") as stream:
            # Reserving blocks detects per-user/project quotas, unlike df or a
            # sparse truncate. This file is released before a worker is started.
            try:
                os.posix_fallocate(stream.fileno(), 0, max(2**20, int(required_bytes)))
            except OSError as exc:
                if exc.errno not in {errno.EOPNOTSUPP, errno.ENOSYS, errno.EINVAL}:
                    raise
                # Shared filesystems without fallocate need real writes to
                # demonstrate quota headroom; never accept a tiny-write proxy.
                block = b"\0" * 2**20
                remaining = max(2**20, int(required_bytes))
                while remaining:
                    count = min(remaining, len(block))
                    stream.write(block[:count])
                    remaining -= count
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        target.unlink(missing_ok=True)


def _read_number(path):
    try:
        value = Path(path).read_text().strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None


def host_resources():
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cpus = min(cpus, max(1, math.floor(int(quota) / int(period))))
    except (OSError, ValueError):
        quota = _read_number("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period = _read_number("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota and period and quota > 0:
            cpus = min(cpus, max(1, quota // period))
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        info[key] = int(value.strip().split()[0]) * 1024
    total, available = info["MemTotal"], info["MemAvailable"]
    limit = _read_number("/sys/fs/cgroup/memory.max") or _read_number("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    current = _read_number("/sys/fs/cgroup/memory.current")
    if current is None:
        current = _read_number("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if limit is not None and limit < total:
        total = limit
        available = min(available, max(0, limit - (current or 0)))
    return {"cpus": cpus, "total_bytes": total, "available_bytes": available}


def worker_environment(runtime, gpu):
    env = {k: v for k, v in os.environ.items() if k not in {
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK",
        "MASTER_ADDR", "MASTER_PORT"}}
    env.update(CUDA_VISIBLE_DEVICES=gpu["uuid"], PYTHONPATH=str(Path(runtime) / "src"),
               PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    return env


def worker_command(output, job, phase, profile_output=None):
    command = [sys.executable, "-m", "mri_vla_jepa.t0_cv_worker", "--suite", str(output),
               "--job", job["name"], "--phase", phase]
    if profile_output:
        command += ["--profile-output", str(profile_output)]
    return command


def spawn_worker(state, output, job, phase, gpu, profile_output=None):
    log = Path(job["output"]) / f"{phase}.log" if phase != "profile" else Path(profile_output).with_suffix(".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    command = worker_command(output, job, phase, profile_output)
    attempt_id = uuid.uuid4().hex
    attempt_status = Path(output) / "attempts" / f"{attempt_id}.json"
    with log.open("a", buffering=1) as stream:
        stream.seek(0, os.SEEK_END)
        log_start = stream.tell()
        env = worker_environment(state["runtime"], gpu)
        env.update(T0_CV_ATTEMPT_ID=attempt_id, T0_CV_ATTEMPT_STATUS=str(attempt_status),
                   T0_CV_LOG_START=str(log_start))
        child = subprocess.Popen(command, cwd=state["runtime"], env=env,
                                 stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    info = process_info(child.pid)
    if info is None and child.poll() is None:
        raise RuntimeError("Cannot journal worker process identity")
    return child, {"pid": child.pid, "start_ticks": info["start_ticks"] if info else None,
                   "gpu_uuid": gpu["uuid"], "phase": phase, "log": str(log), "command": command,
                   "log_start": log_start, "spawned_at": now(),
                   "attempt_id": attempt_id, "attempt_status": str(attempt_status)}


def recover_workers(state, output):
    """Adopt exact worker commands, including a crash between spawn and journaling."""
    found = {j["name"]: [] for j in state["jobs"]}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        info = process_info(int(entry.name))
        if info is None:
            continue
        command = info["command"]
        if "mri_vla_jepa.t0_cv_worker" not in command:
            continue
        try:
            if str(Path(command[command.index("--suite") + 1]).resolve()) != str(Path(output).resolve()):
                continue
            name = command[command.index("--job") + 1]
            phase = command[command.index("--phase") + 1]
        except (ValueError, IndexError):
            continue
        if name not in found or phase not in {"train", "evaluate", "profile"}:
            continue
        try:
            variables = dict(v.split(b"=", 1) for v in (entry / "environ").read_bytes().split(b"\0") if b"=" in v)
            uuid = variables.get(b"CUDA_VISIBLE_DEVICES", b"").decode()
        except (OSError, ValueError):
            raise RuntimeError("Cannot identify an existing suite worker's allocated GPU")
        if phase == "profile":
            # A preflight never affects a real training checkpoint; do not run a duplicate.
            if same_process(state.get("active_preflight", {})):
                continue
            raise RuntimeError("An unjournaled profile worker is still active; wait before resubmitting")
        if uuid not in {g["uuid"] for g in state.get("gpus", [])}:
            raise RuntimeError("A live worker uses a GPU outside this job's allocation")
        job = next(j for j in state["jobs"] if j["name"] == name)
        found[name].append({**info, "gpu_uuid": uuid, "phase": phase,
                            "log": str(Path(job["output"]) / f"{phase}.log"),
                            "log_start": int(variables.get(b"T0_CV_LOG_START", b"0")),
                            "attempt_id": variables.get(b"T0_CV_ATTEMPT_ID", b"").decode(),
                            "attempt_status": variables.get(b"T0_CV_ATTEMPT_STATUS", b"").decode(),
                            "status": "running" if phase == "train" else "evaluating"})
    for job in state["jobs"]:
        matches = found[job["name"]]
        if len(matches) > 1:
            raise RuntimeError(f"Duplicate live workers for {job['name']}; no new tasks dispatched")
        if matches:
            job.update(matches[0], recovered_at=now())
        # Keep dead running/evaluating attempts for the controller to inspect.
        # A durable exit receipt distinguishes a worker error from interruption.
    return found


def worker_failure_log(job):
    log = Path(job.get("log", ""))
    if not log.is_file():
        return ""
    with log.open("rb") as stream:
        stream.seek(job.get("log_start", 0))
        return stream.read().decode(errors="replace")[-20000:]


def recovered_exit_code(job, output):
    target = job.get("attempt_status")
    if target:
        path = Path(target).resolve()
        expected = Path(output).resolve() / "attempts" / f"{job.get('attempt_id')}.json"
        if path != expected:
            raise ValueError("Worker exit receipt is outside its journaled attempt")
        if path.is_file():
            receipt = read_json(path)
            if any(receipt.get(k) != job.get(k) for k in ("attempt_id", "pid", "phase")):
                raise ValueError("Worker exit receipt identity differs")
            if receipt.get("job") != job["name"] or receipt.get("exit_code") not in {0, 1}:
                raise ValueError("Invalid worker exit receipt")
            return receipt["exit_code"]
    # A quota failure can also prevent writing the small exit receipt. Read only
    # this attempt's log; SIGKILL without an error remains resumable interruption.
    message = worker_failure_log(job).lower()
    if any(s in message for s in ("traceback (most recent call last)", "out of memory", "disk quota",
                                  "no space left", "pytorchstreamwriter")):
        return 1
    return None


def training_complete(job):
    path = Path(job["output"]) / "train_report.json"
    return path.is_file() and read_json(path).get("completed") is True


def evaluation_complete(job, output):
    run = Path(job["output"])
    done = run / "evaluation_done.json"
    if not done.exists() or read_json(done).get("completed") is not True:
        return False
    from mri_vla_jepa.ablation_suite import delete_evaluated_checkpoints
    # Recheck all hashes/coverage even on a resumed or already-completed run.
    # The helper also finishes a deletion interrupted after durable receipt.
    delete_evaluated_checkpoints(run, output)
    return True


def _checkpoint_required(state, jobs, candidate=None):
    """Three full files/run cover best, last, and one atomic save copy."""
    todo = list(jobs) + ([candidate] if candidate is not None else [])
    total = 0
    for job in todo:
        sizes = [p.get("checkpoint_bytes", 0) for key, p in state.get("profiles", {}).items()
                 if p.get("variant") == job["variant"]]
        size = max(sizes, default=0)
        if size <= 0:
            raise ValueError("Preflight must measure or conservatively estimate checkpoint size")
        # Existing bytes already reduce statvfs free space; reserve only growth.
        run = Path(job["output"])
        existing = sum(p.stat().st_size for p in run.glob("*.pt*") if p.is_file()) if run.exists() else 0
        total += max(0, 3 * size - existing)
    factor = state["spec"]["resources"].get("storage_margin_factor", 1.25)
    return math.ceil(total * factor + 256 * 2**20)


def _commit(state, output):
    state["updated_at"] = now()
    durable_json(Path(output) / "matrix.json", state)


def profile_all(state, output, stopped):
    resources = state["spec"]["resources"]
    profiles = state.setdefault("profiles", {})
    if host_resources()["cpus"] < resources["threads_per_task"] + resources["loader_workers"]:
        raise RuntimeError("CPU allocation is smaller than one configured training worker; request more CPUs")
    variants = list(dict.fromkeys(j["variant"] for j in state["jobs"]))
    representatives = {}
    for gpu in state["gpus"]:
        representatives.setdefault((gpu["name"], gpu["total_mib"]), gpu)
    for gpu in representatives.values():
        for variant in variants:
            key = profile_key(gpu, variant)
            job = next(j for j in state["jobs"] if j["variant"] == variant)
            if key in profiles:
                if profiles[key].get("status") != "passed":
                    raise ValueError("Invalid cached GPU preflight")
                continue
            if stopped[0]:
                return False
            if any(same_process(j) and j.get("gpu_uuid") == gpu["uuid"] for j in state["jobs"]):
                raise RuntimeError("Cannot reprofile a GPU while a recovered training worker uses it")
            target = Path(output) / "preflight" / f"{gpu['uuid']}_{variant}.json"
            used, _ = gpu_observations()
            # Measurements with another workload's allocations are not transferable.
            if used.get(gpu["uuid"], 0) > 1024:
                raise RuntimeError(f"Preflight GPU is occupied: {gpu['uuid']}; retry when the allocated GPU is free")
            storage_probe(output, 256 * 2**20)
            child, record = spawn_worker(state, output, job, "profile", gpu, target)
            record.update(profile_output=str(target), profile_key=key)
            state["active_preflight"] = record
            _commit(state, output)
            peak, began = 0, time.monotonic()
            try:
                while child.poll() is None:
                    used, _ = gpu_observations()
                    peak = max(peak, used.get(gpu["uuid"], 0))
                    if stopped[0]:
                        child.terminate()
                        child.wait(timeout=30)
                        return False
                    if peak > gpu["total_mib"] * resources["memory_fraction"] or time.monotonic() - began > 3600:
                        child.terminate()
                        child.wait(timeout=30)
                        raise RuntimeError(f"GPU preflight exceeded its memory/time budget; see {record['log']}")
                    time.sleep(2)
                if child.returncode:
                    raise RuntimeError(f"GPU preflight failed: {record['log']}")
                result = read_json(target)
                if result.get("status") != "passed" or result.get("batch_size") != 16:
                    raise ValueError("GPU profile did not verify the required batch size 16")
                peaks = [result.get(k, 0) or 0 for k in (
                    "peak_reserved_mib", "training_peak_reserved_mib", "inference_peak_reserved_mib",
                    "diagnostic_peak_reserved_mib", "world_peak_reserved_mib")]
                if not max(peaks) > 0 or result.get("checkpoint_bytes", 0) <= 0:
                    raise ValueError("Preflight lacks memory or checkpoint-size measurements")
                result.update(variant=variant, measured_gpu=gpu, nvml_peak_mib=peak,
                              config_digest=job["config_digest"], manifest_sha256=job["manifest_sha256"],
                              runtime_identity=stable_hash(state["runtime_hashes"]),
                              reservation_mib=math.ceil(max(peak, *peaks) + resources["reservation_margin_mib"]))
                if result["reservation_mib"] > gpu["total_mib"] * resources["memory_fraction"]:
                    raise RuntimeError(f"Batch 16 cannot fit {gpu['name']} within the configured budget")
                profiles[key] = result
                durable_json(target, result)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=30)
                state.pop("active_preflight", None)
                _commit(state, output)
    return True


def _host_admits(state, active, candidate):
    limits = host_resources()
    resources = state["spec"]["resources"]
    slots = max(1, limits["cpus"] // (resources["threads_per_task"] + resources["loader_workers"]))
    if len(active) >= slots:
        return False
    host_profiles = [p.get("host_reservation_mib", p.get("host_peak_mib", 0))
                     for p in state["profiles"].values() if p.get("variant") == candidate["variant"]]
    peak = max(host_profiles, default=0) * 2**20
    reserved = sum(max((p.get("host_reservation_mib", p.get("host_peak_mib", 0)) for p in state["profiles"].values()
                        if p.get("variant") == j["variant"]), default=0) * 2**20 for j in active)
    resident = sum(process_tree_rss(j.get("pid")) for j in active)
    growth = max(0, reserved - resident)
    return (peak + growth <= limits["available_bytes"] and
            reserved + peak <= limits["total_bytes"] * resources["host_memory_fraction"])


def process_tree_rss(pid):
    if not pid:
        return 0
    try:
        root = Path("/proc") / str(pid)
        rss = next(int(line.split()[1]) * 1024 for line in (root / "status").read_text().splitlines()
                   if line.startswith("VmRSS:"))
        children = (root / "task" / str(pid) / "children").read_text().split()
        return rss + sum(process_tree_rss(int(child)) for child in children)
    except (OSError, ValueError, StopIteration):
        return 0


def validate_host_capacity(state):
    limits = host_resources()
    budget = limits["total_bytes"] * state["spec"]["resources"]["host_memory_fraction"]
    for profile in state["profiles"].values():
        required = profile.get("host_reservation_mib", profile.get("host_peak_mib", 0)) * 2**20
        if required > budget:
            raise RuntimeError(f"A single {profile['variant']} task needs {required / 2**30:.1f} GiB host RAM "
                               f"but the allocation budget is {budget / 2**30:.1f} GiB; request more RAM")


def run_controller(state, output, stopped):
    from mri_vla_jepa.t0_cv_suite import validate_inputs, write_reports
    resources = state["spec"]["resources"]
    children = {}
    recover_workers(state, output)
    _commit(state, output)
    while True:
        active = []
        for job in state["jobs"]:
            if job["status"] == "completed":
                continue
            child = children.get(job["name"])
            if child is not None:
                code = child.poll()
                alive = code is None
            else:
                alive = same_process(job) is not None
                code = None if alive else recovered_exit_code(job, output)
            if alive:
                active.append(job)
                continue
            if evaluation_complete(job, output):
                job.update(status="completed", completed_at=now())
                children.pop(job["name"], None)
                continue
            if job["status"] in {"running", "evaluating"}:
                phase = "train" if job["status"] == "running" else "evaluate"
                if phase == "train" and training_complete(job):
                    job["status"] = "evaluation_pending"
                elif code == 0 and phase == "evaluate":
                    raise RuntimeError(f"Evaluation exited without a completion receipt: {job['name']}")
                elif code == 0 and phase == "train":
                    raise RuntimeError(f"Training exited without a completion report: {job['name']}")
                elif code not in (None, 0):
                    log = Path(job.get("log", ""))
                    message = worker_failure_log(job)
                    lower = message.lower()
                    job.update(status="queued" if phase == "train" else "evaluation_pending",
                               last_exit_code=code, failure_log=str(log))
                    children.pop(job["name"], None)
                    if any(v in lower for v in ("disk quota", "no space left", "edquot", "pytorchstreamwriter")):
                        state.update(status="storage_blocked", storage_blocked=True,
                                     storage_error=f"Checkpoint write failed: {log}")
                        _commit(state, output)
                    elif "out of memory" in lower and job.get("oom_retries", 0) < 1:
                        job.update(exclusive_retry=True, oom_retries=job.get("oom_retries", 0) + 1)
                        gpu = next(g for g in state["gpus"] if g["uuid"] == job["gpu_uuid"])
                        # An exclusive retry avoids a concurrent peak collision.
                        # Keep the measured reservation: reserving the whole 90%
                        # would prevent admission on cards with nonzero idle use.
                        state["profiles"][profile_key(gpu, job["variant"])]["exclusive_only"] = True
                    else:
                        job["status"] = "failed"
                        state["status"] = "failed"
                        _commit(state, output)
                        raise RuntimeError(f"Worker failed; checkpoints preserved: {log}")
                else:
                    job["status"] = "evaluation_pending" if training_complete(job) else "queued"
            elif job["status"] == "queued" and training_complete(job):
                job["status"] = "evaluation_pending"
        if all(j["status"] == "completed" for j in state["jobs"]):
            state["status"] = "completed"
            _commit(state, output)
            write_reports(output, state)
            return 0
        if stopped[0]:
            # Preserve live workers/last checkpoints. Platform cgroup termination
            # may stop them later; a fresh controller adopts any that survived.
            state["status"] = "interrupted"
            _commit(state, output)
            return 130
        if any(j["status"] == "failed" for j in state["jobs"]):
            raise RuntimeError("A task failed; correct its recorded error before resubmitting")
        used, observations = gpu_observations()
        pending = sorted([j for j in state["jobs"] if j["status"] in {"queued", "evaluation_pending"}],
                         key=lambda j: (j["status"] != "evaluation_pending", j["fold_id"], j["variant"]))
        for job in pending:
            if not _host_admits(state, active, job):
                continue
            gpu = choose_gpu(state["gpus"], state["profiles"], active, used, observations,
                             variant=job["variant"], memory_fraction=resources["memory_fraction"],
                             margin_mib=resources["reservation_margin_mib"],
                             exclusive=job.get("exclusive_retry", False))
            if gpu is None:
                continue
            if job.get("exclusive_retry") and any(j.get("gpu_uuid") == gpu["uuid"] for j in active):
                continue
            validate_runtime(state)
            validate_inputs(state)
            try:
                # Evaluation can free large checkpoints even while new training
                # is storage-blocked; it only requires room for small reports.
                storage_probe(output, 4 * 2**20 if job["status"] == "evaluation_pending"
                              else _checkpoint_required(state, active, job))
            except OSError as exc:
                state.update(status="storage_blocked", storage_blocked=True, storage_error=str(exc))
                _commit(state, output)
                continue
            if job["status"] == "queued":
                state.pop("storage_error", None)
                state.pop("storage_blocked", None)
            phase = "evaluate" if job["status"] == "evaluation_pending" else "train"
            child, record = spawn_worker(state, output, job, phase, gpu)
            job.update(record, status="evaluating" if phase == "evaluate" else "running")
            children[job["name"]] = child
            active.append(job)
            _commit(state, output)
        state["status"] = "storage_blocked" if state.get("storage_blocked") else "running"
        state["active_jobs"] = [j["name"] for j in active]
        state["host_resources"] = host_resources()
        _commit(state, output)
        # OOF/ensemble are computed once after completion, not once per poll.
        print(json.dumps({"time": now(), "completed": sum(j["status"] == "completed" for j in state["jobs"]),
                          "total": len(state["jobs"]), "active": len(active),
                          "queued": sum(j["status"] == "queued" for j in state["jobs"]),
                          "evaluation_pending": sum(j["status"] == "evaluation_pending" for j in state["jobs"]),
                          "status": state["status"], "output": str(output)}, ensure_ascii=False), flush=True)
        time.sleep(min(float(resources["poll_seconds"]), 30))


def main(argv=None):
    from mri_vla_jepa.t0_cv_suite import load_suite_config, prepare_suite, validate_inputs, write_reports
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--report-only", action="store_true")
    launch_arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(launch_arguments)
    reject_distributed_launch()
    spec = load_suite_config(args.config.resolve())
    output = Path(spec["output_dir"]).resolve()
    validate_output_parent(spec)
    if args.report_only:
        # Reporting has its own report.lock and may inspect an atomic state
        # snapshot while the training controller continues dispatching workers.
        state = read_json(output / "matrix.json")
        enter_frozen_controller(state, launch_arguments)
        if state.get("spec") != spec:
            raise ValueError("Suite configuration changed; refusing mixed reports")
        validate_runtime(state)
        validate_inputs(state)
        write_reports(output, state)
        return 0
    stopped = [False]

    def stop(signum, frame):
        stopped[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with controller_lock(output):
        path = output / "matrix.json"
        if path.exists():
            state = read_json(path)
            enter_frozen_controller(state, launch_arguments)
            if state.get("spec") != spec:
                raise ValueError("Suite configuration changed; use a new suite_name/output directory")
            validate_runtime(state)
            validate_inputs(state)
        else:
            if args.report_only:
                raise FileNotFoundError("No prepared suite to report")
            state = prepare_suite(spec, output)
            runtime = freeze_runtime(output)
            from mri_vla_jepa.training import _source_snapshot
            state.update(spec=spec, runtime=str(runtime), runtime_hashes=runtime_hashes(runtime),
                         source_identity=stable_hash(_source_snapshot()),
                         created_at=now(), status="prepared", profiles={})
            _commit(state, output)
            enter_frozen_controller(state, launch_arguments)
        if args.prepare_only:
            write_reports(output, state, bootstrap=0)
            print(json.dumps({"status": state["status"], "jobs": len(state["jobs"]), "output": str(output)}))
            return 0
        if state["status"] == "completed":
            for job in state["jobs"]:
                if not evaluation_complete(job, output):
                    raise ValueError("Completed task lost its evaluation receipt")
            write_reports(output, state)
            return 0
        gpus = discover_gpus(expected=spec["resources"]["expected_gpus"])
        if state.get("active_preflight"):
            record = state["active_preflight"]
            if same_process(record):
                raise RuntimeError("An existing preflight is active; wait for it to exit before resubmitting")
            state.pop("active_preflight")
        old_gpus = state.get("gpus", [])
        if old_gpus and {g["uuid"] for g in old_gpus} != {g["uuid"] for g in gpus}:
            if any(same_process(j) for j in state["jobs"]):
                raise RuntimeError("Live workers remain on the previous GPU allocation")
        state["gpus"] = gpus
        recover_workers(state, output)
        _commit(state, output)
        if not profile_all(state, output, stopped):
            return 130
        validate_host_capacity(state)
        if args.preflight_only:
            state["status"] = "preflight_passed"
            _commit(state, output)
            print(json.dumps({"status": state["status"], "profiles": len(state["profiles"]), "gpus": len(gpus)}))
            return 0
        return run_controller(state, output, stopped)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"T0 five-fold suite failed: {exc}", file=sys.stderr, flush=True)
        raise
