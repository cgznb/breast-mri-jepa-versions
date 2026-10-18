#!/usr/bin/env python3
"""Persistent, memory-budgeted queue for the fixed-T0/dynamic joint comparison.

The controller and trainers may survive the launching terminal. Restarting the
same command adopts exact matching workers; an interrupted checkpoint can be
resumed once. This script never changes training hyperparameters.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import yaml

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
RESERVATIONS = {"t0": 8192, "dynamic": 10240}
REQUIRED_ARTIFACTS = ("metrics_best.json", "best.pt", "last.pt", "predictions_val.csv")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temp.chmod(0o600)
    os.replace(temp, path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def process_info(pid):
    """Process birth tick prevents a recycled PID from being adopted."""
    try:
        folder = Path("/proc") / str(int(pid))
        fields = (folder / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "start_ticks": fields[19],
                "cmdline": (folder / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0"),
                "cwd": str((folder / "cwd").resolve(strict=True))}
    except (OSError, ValueError, IndexError):
        return None


def matches_process(info, command, cwd, birth=None):
    return bool(info and info["cmdline"] == command and info["cwd"] == str(Path(cwd).resolve())
                and (birth is None or info["start_ticks"] == birth))


def find_process(command, cwd, pid=None, birth=None):
    if pid and matches_process(process_info(pid), command, cwd, birth):
        return process_info(pid)
    for folder in Path("/proc").iterdir():
        if folder.name.isdigit():
            info = process_info(folder.name)
            if matches_process(info, command, cwd):
                return info
    return None


def training_environment(project=PROJECT):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(project / "src"), PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1", PYTORCH_ALLOC_CONF="expandable_segments:True",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    return env


def trainer_command(config, manifest, output, resume=False):
    command = [sys.executable, "-m", "response_jepa.cli", "train", "--config", str(config),
               "--manifest", str(manifest), "--output", str(output)]
    return command + (["--resume"] if resume else [])


def worker_command(run, manifest, attempt):
    command = [sys.executable, str(SCRIPT), "--worker", "--config", run["config"],
               "--manifest", str(manifest), "--output", run["output"],
               "--attempt", str(attempt["attempt"])]
    return command + (["--resume"] if attempt["resume"] else [])


def complete_artifacts(output):
    output = Path(output)
    return (all((output / name).is_file() and (output / name).stat().st_size > 0
                for name in REQUIRED_ARTIFACTS)
            and isinstance(read_json(output / "metrics_best.json"), dict))


def terminal_training(output):
    """A one-epoch smoke stop also writes final artifacts, so inspect last.pt."""
    if not complete_artifacts(output):
        return False
    try:
        # Only locally generated, trusted checkpoints belonging to this suite.
        import torch
        checkpoint = torch.load(Path(output) / "last.pt", map_location="cpu", weights_only=False)
        config = read_json(Path(output) / "config.json")
        metrics = read_json(Path(output) / "metrics_best.json")
        train = config["train"]
        return (checkpoint["schema"] == "response_jepa_checkpoint_local_v1"
                and checkpoint["config"] == config
                and (checkpoint["epoch"] == train["epochs"]
                     or checkpoint["bad_epochs"] >= train["patience"])
                and metrics["selection_name"] == "patient_equal_pcr_nll"
                and math.isfinite(float(metrics["selection"]))
                and math.isclose(float(metrics["selection"]), float(checkpoint["best_score"]),
                                 rel_tol=1e-6, abs_tol=1e-7))
    except Exception:
        return False


def gpu_snapshot():
    """Read total usage, including allocations belonging to other applications."""
    result = subprocess.run(["nvidia-smi", "--id=0",
                             "--query-gpu=uuid,memory.total,memory.used,utilization.gpu,power.draw",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True,
                            check=True, timeout=10)
    uuid, total, used, utilization, power = [item.strip() for item in result.stdout.strip().split(",")]
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True,
                            check=True, timeout=10)
    processes = {}
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 3 and fields[0] == uuid:
            try:
                processes[int(fields[1])] = int(fields[2])
            except ValueError:
                pass  # Total usage still accounts for unavailable per-process figures.
    def numeric(value):
        try:
            return float(value)
        except ValueError:
            return None
    return {"at": now(), "total_mib": int(total), "used_mib": int(used),
            "free_mib": int(total) - int(used), "utilization_percent": numeric(utilization),
            "power_watts": numeric(power), "processes": processes}


def reserved_usage(running, snapshot):
    observed = [snapshot["processes"].get(run.get("trainer_pid"), 0) for run in running]
    external = max(0, snapshot["used_mib"] - sum(observed))
    return external + sum(max(run.get("reservation_mib", RESERVATIONS[run["profile"]]), usage)
                          for run, usage in zip(running, observed))


def can_launch(candidate, running, snapshot, budget, max_concurrent):
    return (len(running) < max_concurrent and
            reserved_usage(running, snapshot) + candidate.get("reservation_mib", RESERVATIONS[candidate["profile"]])
            <= min(budget, snapshot["total_mib"]))


def prepare_suite(project, output, manifest, seeds):
    output, manifest = Path(output).resolve(), Path(manifest).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    specs = []
    for seed in seeds:
        for profile in ("dynamic", "t0"):
            template = (f"joint_dynamic_all_prefixes_batch16_seed{seed}.yaml" if profile == "dynamic"
                        else f"joint_t0_batch16_seed{seed}.yaml")
            source = project / "configs" / template
            text = source.read_text(encoding="utf-8")
            train = yaml.safe_load(text)["train"]
            if (train["phase"] != "joint" or train["profile"] != profile or train["seed"] != seed
                    or train["batch_size"] != 16 or train["accumulation"] != 1
                    or (profile == "dynamic" and train["prefix_sampling"] != "all_prefixes")):
                raise ValueError(f"Unexpected joint configuration: {source}")
            name = f"joint_{profile}_seed{seed}"
            specs.append((name, profile, seed, source, text))
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    saved = read_json(output / "status.json")
    if saved:
        if (saved.get("schema") != "joint_suite_queue_v1" or saved.get("manifest") != str(manifest)
                or saved.get("project") != str(project.resolve()) or saved.get("seeds") != seeds
                or [run["name"] for run in saved["runs"]] != [spec[0] for spec in specs]):
            raise ValueError("Existing suite identity differs; choose a fresh output directory")
        for run, spec in zip(saved["runs"], specs):
            if Path(run["config"]).read_text(encoding="utf-8") != spec[4]:
                raise ValueError("Training configuration changed; cannot resume this suite")
        return saved
    if (output / "status.json").exists():
        raise ValueError("Unreadable suite status; refusing to overwrite it")
    (output / "configs").mkdir(exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    runs = []
    for name, profile, seed, source, text in specs:
        config = output / "configs" / f"{name}.yaml"
        if config.exists() and config.read_text(encoding="utf-8") != text:
            raise ValueError(f"Config snapshot already differs: {config}")
        config.write_text(text, encoding="utf-8")
        outdir = output / name
        if (outdir / "config.json").exists():
            raise ValueError(f"Run already exists without suite status: {outdir}")
        runs.append({"name": name, "profile": profile, "seed": seed, "config": str(config),
                     "source_config": str(source), "output": str(outdir),
                     "log": str(output / "logs" / f"{name}.log"), "state": "queued",
                     "attempts": [], "resume_next": False})
    return {"schema": "joint_suite_queue_v1", "project": str(project.resolve()),
            "manifest": str(manifest), "seeds": seeds, "created_at": now(), "runs": runs}


def attempt_paths(run, attempt):
    folder = Path(run["output"]) / ".queue"
    return folder / f"attempt{attempt['attempt']}.worker.json", folder / f"attempt{attempt['attempt']}.exit.json"


def update_run(run, manifest, handles, restarting=False):
    if not run["attempts"] or run["state"] == "completed":
        return
    attempt = run["attempts"][-1]
    worker_path, exit_path = attempt_paths(run, attempt)
    worker = find_process(worker_command(run, manifest, attempt), PROJECT,
                          attempt.get("worker_pid"), attempt.get("worker_start_ticks"))
    record = read_json(worker_path) or {}
    trainer = find_process(trainer_command(run["config"], manifest, run["output"], attempt["resume"]),
                           PROJECT, record.get("trainer_pid"), record.get("trainer_start_ticks"))
    if worker or trainer:
        run["state"] = "running"
        if worker:
            attempt.update(worker_pid=worker["pid"], worker_start_ticks=worker["start_ticks"])
        if trainer:
            attempt.update(trainer_pid=trainer["pid"], trainer_start_ticks=trainer["start_ticks"])
            run["trainer_pid"] = trainer["pid"]
        return
    if (restarting and not attempt.get("launch_complete") and not read_json(exit_path)
            and not (Path(run["output"]) / "config.json").exists()):
        # A controller can stop after persisting launch intent but before Popen.
        run["attempts"].pop()
        run.update(state="queued", resume_next=attempt["resume"])
        return
    handle = handles.get(run["name"])
    if handle:
        handle.poll()  # Reap controller-owned workers before considering them ended.
    result = read_json(exit_path)
    if result:
        attempt.update(exit_code=result["exit_code"], ended_at=result["ended_at"])
    else:
        attempt.update(exit_code=None, ended_at=now())
    run.pop("trainer_pid", None)
    if result and result["exit_code"] == 0 and terminal_training(run["output"]):
        run["state"] = "completed"
        run.pop("failure", None)
    elif restarting and len(run["attempts"]) == 1 and (Path(run["output"]) / "last.pt").is_file():
        run.update(state="queued", resume_next=True)
        run["recovery"] = "One epoch-boundary resume after controller restart"
    else:
        run["state"] = "failed"
        run["failure"] = (f"Trainer exit code {result['exit_code']} or incomplete artifacts" if result
                          else "Worker/trainer disappeared without an exit record")


def launch_run(run, manifest, save_intent=None):
    attempt = {"attempt": len(run["attempts"]) + 1, "started_at": now(),
               "resume": run.pop("resume_next", False)}
    run["attempts"].append(attempt)
    run["state"] = "running"
    if save_intent:
        save_intent()
    Path(run["output"]).mkdir(parents=True, exist_ok=True, mode=0o700)
    with Path(run["log"]).open("a", encoding="utf-8") as log:
        log.write(f"\n[{attempt['started_at']}] attempt={attempt['attempt']} resume={attempt['resume']}\n")
        log.flush()
        process = subprocess.Popen(worker_command(run, manifest, attempt), cwd=PROJECT,
                                   env=training_environment(), stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    info = process_info(process.pid)
    attempt["worker_pid"] = process.pid
    attempt["launch_complete"] = True
    if info:
        attempt["worker_start_ticks"] = info["start_ticks"]
    run["state"] = "running"
    print(f"[{now()}] launched {run['name']} worker={process.pid} resume={attempt['resume']}", flush=True)
    return process


def worker_main(args):
    output = Path(args.output).resolve()
    folder = output / ".queue"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        command = trainer_command(Path(args.config).resolve(), Path(args.manifest).resolve(), output, args.resume)
        if find_process(command, PROJECT):
            raise RuntimeError("An exact matching trainer already owns this output directory")
        process = subprocess.Popen(command, cwd=PROJECT,
                                   env=training_environment(), stdin=subprocess.DEVNULL)
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
    with (output / ".controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another controller owns this suite; no duplicate trainers started")
        seeds = [int(seed.strip()) for seed in args.seeds.split(",")]
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("Seeds must be distinct")
        status = prepare_suite(PROJECT, output, args.manifest, seeds)
        reservations = {"t0": args.reservation_t0_mib, "dynamic": args.reservation_dynamic_mib}
        for run in status["runs"]:
            run["reservation_mib"] = reservations[run["profile"]]
        status["controller"] = {"pid": os.getpid(), "started_at": now(), "state": "running"}
        status["queue_settings"] = {"max_concurrent": args.max_concurrent,
                                    "gpu_memory_budget_mib": args.gpu_memory_budget_mib,
                                    "poll_seconds": args.poll_seconds,
                                    "reservations_mib": reservations}
        handles = {}
        for run in status["runs"]:
            update_run(run, status["manifest"], handles, restarting=True)
        atomic_json(output / "status.json", status)
        try:
            while True:
                for run in status["runs"]:
                    if run["state"] == "running":
                        previous = run["state"]
                        update_run(run, status["manifest"], handles)
                        if run["state"] != previous:
                            print(f"[{now()}] {run['name']}: {run['state']}", flush=True)
                if all(run["state"] in {"completed", "failed"} for run in status["runs"]):
                    break
                snapshot = gpu_snapshot()
                status["gpu"] = {"at": now(), **snapshot}
                with (output / "gpu_monitor.jsonl").open("a", encoding="utf-8") as monitor:
                    monitor.write(json.dumps({**snapshot, "running": [
                        {"name": run["name"], "trainer_pid": run.get("trainer_pid"),
                         "worker_pid": run["attempts"][-1].get("worker_pid"),
                         "reservation_mib": run["reservation_mib"]}
                        for run in status["runs"] if run["state"] == "running"]}) + "\n")
                for run in status["runs"]:
                    running = [item for item in status["runs"] if item["state"] == "running"]
                    if run["state"] == "queued" and can_launch(run, running, snapshot,
                            args.gpu_memory_budget_mib, args.max_concurrent):
                        handles[run["name"]] = launch_run(run, status["manifest"],
                                                         lambda: atomic_json(output / "status.json", status))
                        atomic_json(output / "status.json", status)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seeds", default="17,43,71")
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--gpu-memory-budget-mib", type=int, default=29696)
    parser.add_argument("--reservation-t0-mib", type=int, default=8192)
    parser.add_argument("--reservation-dynamic-mib", type=int, default=10240)
    parser.add_argument("--poll-seconds", type=float, default=3)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", help=argparse.SUPPRESS)
    parser.add_argument("--attempt", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker_main(args)
    if not 1 <= args.max_concurrent <= 3 or args.poll_seconds <= 0 or min(
            args.gpu_memory_budget_mib, args.reservation_t0_mib, args.reservation_dynamic_mib) < 1:
        parser.error("Concurrency must be 1..3; memory values and poll interval must be positive")
    if args.detach:
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=True, mode=0o700)
        command = [sys.executable, str(SCRIPT), "--output", str(output), "--manifest", str(Path(args.manifest).resolve()),
                   "--seeds", args.seeds, "--max-concurrent", str(args.max_concurrent),
                   "--gpu-memory-budget-mib", str(args.gpu_memory_budget_mib), "--poll-seconds", str(args.poll_seconds),
                   "--reservation-t0-mib", str(args.reservation_t0_mib),
                   "--reservation-dynamic-mib", str(args.reservation_dynamic_mib)]
        with (output / "controller.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=PROJECT, env=training_environment(), stdout=log,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({"controller_pid": process.pid, "output": str(output),
                          "status": str(output / "status.json")}, ensure_ascii=False))
        return 0
    return controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
