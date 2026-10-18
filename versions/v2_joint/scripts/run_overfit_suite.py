#!/usr/bin/env python3
"""Run an immutable plan through a detached, restartable GPU memory queue.

Plan JSON: {"runs": [{"name": ..., "config": ..., "manifest": ...,
                     "output": ..., "reservation_mib": ...}]}.
Relative paths resolve against the plan directory. Existing completed or failed
runs are never retrained. An interrupted active run may resume its own last.pt.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("overfit_queue_helpers", SCRIPT.with_name("run_joint_suite.py"))
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
now = _helpers.now
atomic_json = _helpers.atomic_json
read_json = _helpers.read_json
process_info = _helpers.process_info
matches_process = _helpers.matches_process
training_environment = _helpers.training_environment
gpu_snapshot = _helpers.gpu_snapshot
TERMINAL = {"completed", "failed"}


def trainer_command(config, manifest, output, resume=False):
    # The update runner returns best_update. Its external entrypoint preserves
    # the frozen package source used for strict historical result replay.
    command = [sys.executable, str(SCRIPT.with_name("train_overfit.py")),
               "--config", str(config), "--manifest", str(manifest), "--output", str(output)]
    return command + (["--resume"] if resume else [])


def find_process(command, cwd, pid=None, birth=None):
    """A recorded PID must match its recorded birth; only launch intents scan."""
    if pid is not None:
        info = process_info(pid)
        return info if birth is not None and matches_process(info, command, cwd, birth) else None
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            info = process_info(entry.name)
            if matches_process(info, command, cwd):
                return info
    return None


def file_record(path):
    stat = Path(path).stat()
    return {"path": str(Path(path).resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_plan(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("runs"), list) or not data["runs"]:
        raise ValueError("Plan must contain a nonempty runs list")
    runs, names, outputs = [], set(), set()
    for item in data["runs"]:
        required = {"name", "config", "manifest", "output", "reservation_mib"}
        optional = {"algorithm", "profile", "seed"}
        if not isinstance(item, dict) or not required.issubset(item) or set(item) - required - optional:
            raise ValueError("Each run requires name, config, manifest, output, reservation_mib; optional algorithm/profile/seed")
        if (any(field in item and (not isinstance(item[field], str) or not item[field])
                for field in ("algorithm", "profile")) or ("seed" in item and type(item["seed"]) is not int)):
            raise ValueError("Optional algorithm/profile must be strings and seed must be an integer")
        name = item["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in names:
            raise ValueError("Run names must be safe and distinct")
        if type(item["reservation_mib"]) is not int or item["reservation_mib"] < 1:
            raise ValueError("Reservations must be positive integer MiB")
        run = dict(item)
        for field in ("config", "manifest", "output"):
            if not isinstance(item[field], str) or not item[field]:
                raise ValueError("Plan paths must be nonempty strings")
            run[field] = str((path.parent / item[field]).resolve())
        if not Path(run["config"]).is_file() or not Path(run["manifest"]).is_file():
            raise ValueError("Every config and manifest must already exist")
        output = Path(run["output"])
        if any(output == other or output.is_relative_to(other) or other.is_relative_to(output) for other in outputs):
            raise ValueError("Run output directories must be distinct and cannot overlap")
        names.add(name)
        outputs.add(output)
        runs.append(run)
    return {**data, "runs": runs}


def prepare_suite(project, output, plan_path):
    project, output = Path(project).resolve(), Path(output).resolve()
    plan = load_plan(plan_path)
    protocol = {"schema": "overfit_suite_plan_v1", "project": str(project), "plan": plan,
                "manifest_files": {run["manifest"]: file_record(run["manifest"]) for run in plan["runs"]}}
    saved = read_json(output / "status.json")
    if saved is not None:
        if (saved.get("schema") != "overfit_suite_queue_v1" or saved.get("protocol") != protocol
                or read_json(output / "plan.json") != protocol):
            raise ValueError("Existing suite plan identity changed; use a fresh suite directory")
        if len(saved["runs"]) != len(plan["runs"]):
            raise ValueError("Existing suite run identity changed")
        for run, wanted in zip(saved["runs"], plan["runs"]):
            if any(run[field] != wanted[field] for field in ("name", "manifest", "output", "reservation_mib")):
                raise ValueError("Existing suite run identity changed")
            if (run["source_config"] != wanted["config"]
                    or Path(run["config"]).read_text(encoding="utf-8") != Path(wanted["config"]).read_text(encoding="utf-8")):
                raise ValueError("Configuration changed; use a fresh suite directory")
        return saved
    if (output / "status.json").exists() or (output / "plan.json").exists():
        raise ValueError("Unreadable or orphaned suite metadata; refusing to overwrite")
    # Check every destination before creating any config snapshot.
    for run in plan["runs"]:
        folder = Path(run["output"])
        if folder.exists() and any(folder.iterdir()):
            raise ValueError("Run output already exists without this suite status")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    (output / "configs").mkdir(exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    runs = []
    for item in plan["runs"]:
        config = output / "configs" / f"{item['name']}.yaml"
        config.write_text(Path(item["config"]).read_text(encoding="utf-8"), encoding="utf-8")
        config.chmod(0o600)
        runs.append({**item, "source_config": item["config"], "config": str(config),
                     "log": str(output / "logs" / f"{item['name']}.log"),
                     "state": "queued", "attempts": [], "resume_next": False})
    status = {"schema": "overfit_suite_queue_v1", "protocol": protocol,
              "created_at": now(), "runs": runs}
    atomic_json(output / "plan.json", protocol)
    atomic_json(output / "status.json", status)
    return status


def terminal_training(output):
    """Use update-runner completion, not an epoch count or merely a checkpoint."""
    output = Path(output)
    completion = read_json(output / "completion.json")
    metrics = read_json(output / "metrics_best.json")
    try:
        return bool(isinstance(completion, dict) and completion.get("state") == "completed"
                    and isinstance(metrics, dict) and math.isfinite(float(metrics["selection"]))
                    and metrics["selection_name"] == "patient_equal_pcr_nll"
                    and all((output / name).is_file() and (output / name).stat().st_size > 0
                            for name in ("best.pt", "last.pt")))
    except (KeyError, TypeError, ValueError, OSError):
        return False


def worker_command(run, attempt):
    command = [sys.executable, str(SCRIPT), "--worker", "--config", run["config"],
               "--manifest", run["manifest"], "--output", run["output"],
               "--attempt", str(attempt["attempt"])]
    return command + (["--resume"] if attempt["resume"] else [])


def attempt_paths(run, attempt):
    folder = Path(run["output"]) / ".queue"
    return folder / f"attempt{attempt['attempt']}.worker.json", folder / f"attempt{attempt['attempt']}.exit.json"


def reserved_usage(running, snapshot):
    processes = snapshot["processes"]
    observed = [processes.get(run.get("trainer_pid"), processes.get(str(run.get("trainer_pid")), 0))
                for run in running]
    external = max(0, snapshot["used_mib"] - sum(observed))
    return external + sum(max(run["reservation_mib"], used) for run, used in zip(running, observed))


def can_launch(candidate, running, snapshot, budget=31232, max_concurrent=3):
    observed_free = snapshot.get("free_mib", snapshot["total_mib"] - snapshot["used_mib"])
    return (len(running) < max_concurrent and candidate["reservation_mib"] <= observed_free
            and reserved_usage(running, snapshot) + candidate["reservation_mib"]
            <= min(budget, snapshot["total_mib"]))


def update_run(run, handles, restarting=False):
    if run["state"] != "running" or not run["attempts"]:
        return
    attempt = run["attempts"][-1]
    worker_path, exit_path = attempt_paths(run, attempt)
    record = read_json(worker_path) or {}
    worker = find_process(worker_command(run, attempt), PROJECT,
                          attempt.get("worker_pid"), attempt.get("worker_start_ticks"))
    trainer = find_process(trainer_command(run["config"], run["manifest"], run["output"], attempt["resume"]),
                           PROJECT, record.get("trainer_pid", attempt.get("trainer_pid")),
                           record.get("trainer_start_ticks", attempt.get("trainer_start_ticks")))
    if worker or trainer:
        run["state"] = "running"
        if worker:
            attempt.update(worker_pid=worker["pid"], worker_start_ticks=worker["start_ticks"])
        if trainer:
            attempt.update(trainer_pid=trainer["pid"], trainer_start_ticks=trainer["start_ticks"])
            run["trainer_pid"] = trainer["pid"]
        return
    handle = handles.get(run["name"])
    if handle is not None:
        handle.poll()
    result = read_json(exit_path)
    if (restarting and not attempt.get("launch_complete") and result is None
            and not (Path(run["output"]) / "config.json").exists()
            and not (Path(run["output"]) / "last.pt").exists()):
        run["attempts"].pop()
        run.update(state="queued", resume_next=attempt["resume"])
        return
    attempt.update(exit_code=result.get("exit_code") if result else None,
                   ended_at=result.get("ended_at", now()) if result else now())
    run.pop("trainer_pid", None)
    if result is not None and result.get("exit_code") == 0 and terminal_training(run["output"]):
        run["state"] = "completed"
    elif (restarting and result is None and len(run["attempts"]) == 1
          and (Path(run["output"]) / "last.pt").is_file() and not terminal_training(run["output"])):
        run.update(state="queued", resume_next=True)
        run["recovery"] = "One checkpoint resume after controller interruption; no active trainer found"
    else:
        run["state"] = "failed"
        run["failure"] = (f"Trainer exit {result.get('exit_code')}; completion/artifacts incomplete" if result is not None
                          else "Worker and trainer disappeared without an exit record")


def launch_run(run, save_intent):
    if run["state"] != "queued" or terminal_training(run["output"]):
        raise RuntimeError("Refusing to launch a nonqueued or completed run")
    if run.get("resume_next") and not (Path(run["output"]) / "last.pt").is_file():
        raise RuntimeError("Recovery requires this run's last.pt")
    for resume in (False, True):
        if find_process(trainer_command(run["config"], run["manifest"], run["output"], resume), PROJECT):
            raise RuntimeError("An existing trainer already owns this output")
    attempt = {"attempt": len(run["attempts"]) + 1, "started_at": now(),
               "resume": run.pop("resume_next", False)}
    run["attempts"].append(attempt)
    run["state"] = "running"
    save_intent()
    Path(run["output"]).mkdir(parents=True, exist_ok=True, mode=0o700)
    with Path(run["log"]).open("a", encoding="utf-8") as log:
        log.write(f"\n[{attempt['started_at']}] attempt={attempt['attempt']} resume={attempt['resume']}\n")
        log.flush()
        process = subprocess.Popen(worker_command(run, attempt), cwd=PROJECT, env=training_environment(),
                                   stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True)
    info = process_info(process.pid) or {}
    attempt.update(worker_pid=process.pid, worker_start_ticks=info.get("start_ticks"), launch_complete=True)
    print(f"[{now()}] launched {run['name']} worker={process.pid} resume={attempt['resume']}", flush=True)
    return process


def worker_main(args):
    output = Path(args.output).resolve()
    folder = output / ".queue"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if terminal_training(output) or (args.resume and not (output / "last.pt").is_file()):
            raise RuntimeError("Output completed or recovery checkpoint absent")
        for resume in (False, True):
            command = trainer_command(str(Path(args.config).resolve()), str(Path(args.manifest).resolve()), str(output), resume)
            if find_process(command, PROJECT):
                raise RuntimeError("An exact matching trainer already owns this output")
        command = trainer_command(str(Path(args.config).resolve()), str(Path(args.manifest).resolve()), str(output), args.resume)
        process = subprocess.Popen(command, cwd=PROJECT, env=training_environment(), stdin=subprocess.DEVNULL,
                                   start_new_session=True)
        info = process_info(process.pid) or {}
        atomic_json(folder / f"attempt{args.attempt}.worker.json",
                    {"worker_pid": os.getpid(), "trainer_pid": process.pid,
                     "trainer_start_ticks": info.get("start_ticks"), "started_at": now()})
        code = process.wait()
        atomic_json(folder / f"attempt{args.attempt}.exit.json", {"exit_code": code, "ended_at": now()})
        return code


def controller(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    inherited = getattr(args, "controller_lock_fd", None)
    lock_handle = os.fdopen(inherited, "a") if inherited is not None else (output / ".controller.lock").open("a")
    with lock_handle as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another controller already owns this suite; no duplicate started")
        status = prepare_suite(PROJECT, output, args.plan)
        if any(run["reservation_mib"] > args.gpu_memory_budget_mib for run in status["runs"]):
            raise ValueError("A run reservation exceeds the entire queue memory budget")
        info = process_info(os.getpid()) or {}
        status["controller"] = {"pid": os.getpid(), "start_ticks": info.get("start_ticks"),
                                "started_at": now(), "state": "running"}
        status["queue_settings"] = {"max_concurrent": args.max_concurrent,
                                    "gpu_memory_budget_mib": args.gpu_memory_budget_mib,
                                    "poll_seconds": args.poll_seconds}
        handles = {}
        for run in status["runs"]:
            update_run(run, handles, restarting=True)
        atomic_json(output / "status.json", status)
        try:
            while not all(run["state"] in TERMINAL for run in status["runs"]):
                for run in status["runs"]:
                    update_run(run, handles)
                if all(run["state"] in TERMINAL for run in status["runs"]):
                    break
                snapshot = gpu_snapshot()
                status["gpu"] = snapshot
                for run in status["runs"]:
                    running = [item for item in status["runs"] if item["state"] == "running"]
                    if run["state"] == "queued" and can_launch(run, running, snapshot,
                            args.gpu_memory_budget_mib, args.max_concurrent):
                        handles[run["name"]] = launch_run(run, lambda: atomic_json(output / "status.json", status))
                        atomic_json(output / "status.json", status)
                running = [{"name": run["name"], "trainer_pid": run.get("trainer_pid"),
                            "worker_pid": run["attempts"][-1].get("worker_pid"),
                            "reservation_mib": run["reservation_mib"]}
                           for run in status["runs"] if run["state"] == "running"]
                with (output / "gpu_monitor.jsonl").open("a", encoding="utf-8") as monitor:
                    monitor.write(json.dumps({**snapshot, "running_count": len(running), "running": running}) + "\n")
                    monitor.flush()
                status["updated_at"] = now()
                atomic_json(output / "status.json", status)
                time.sleep(args.poll_seconds)
            status["controller"].update(state="completed", ended_at=now())
        except BaseException as exc:
            status["controller"].update(state="stopped", ended_at=now(), reason=str(exc))
            raise
        finally:
            atomic_json(output / "status.json", status)
        return 0 if all(run["state"] == "completed" for run in status["runs"]) else 1


def detach(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (output / ".controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            saved = read_json(output / "status.json") or {}
            if saved:
                prepare_suite(PROJECT, output, args.plan)  # validate the locked identity, without rewriting it
            print(json.dumps({"state": "already_running", "controller": saved.get("controller"),
                              "output": str(output), "message": "Existing controller retained; no duplicate started"}))
            return 0
        status = prepare_suite(PROJECT, output, args.plan)
        if all(run["state"] in TERMINAL for run in status["runs"]):
            print(json.dumps({"state": "already_terminal", "output": str(output),
                              "message": "Existing completed/failed runs retained; no training restarted"}))
            return 0
        command = [sys.executable, str(SCRIPT), "--plan", str(Path(args.plan).resolve()), "--output", str(output),
                   "--max-concurrent", str(args.max_concurrent), "--gpu-memory-budget-mib", str(args.gpu_memory_budget_mib),
                   "--poll-seconds", str(args.poll_seconds), "--controller-lock-fd", str(lock.fileno())]
        # Transfer the same flock to the detached child. There is no unlocked
        # interval in which a second launcher could create a duplicate controller.
        with (output / "controller.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=PROJECT, env=training_environment(), stdout=log,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                                       pass_fds=(lock.fileno(),))
    print(json.dumps({"state": "started", "controller_pid": process.pid, "output": str(output),
                      "status": str(output / "status.json")}))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--gpu-memory-budget-mib", type=int, default=31232)
    parser.add_argument("--poll-seconds", type=float, default=3)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", help=argparse.SUPPRESS)
    parser.add_argument("--manifest", help=argparse.SUPPRESS)
    parser.add_argument("--attempt", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--controller-lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if not args.config or not args.manifest or not args.attempt or args.attempt < 1:
            parser.error("Worker requires config, manifest and a positive attempt")
        return worker_main(args)
    if (not args.plan or not 1 <= args.max_concurrent <= 3 or args.gpu_memory_budget_mib < 1
            or not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0):
        parser.error("Plan required; concurrency 1..3; positive memory and finite poll interval")
    return detach(args) if args.detach else controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
