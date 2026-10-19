#!/usr/bin/env python3
"""Profile, freeze and run the 48 size/phase/training/seed experiments on GPUs."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import yaml

from mri_vla_jepa.data import RawMRIStore
from mri_vla_jepa.io import digest, load_checkpoint, read_json, stable_hash, write_json
from mri_vla_jepa.train_config import from_dict, load_config
from mri_vla_jepa.training import _source_snapshot

ROOT = Path(__file__).resolve().parents[1]
PROJECT = next((p for p in ROOT.parents if (p / "data/pretrained/codec.pt").is_file()), ROOT)
SIZES = ("large", "medium", "small")
PHASES = ("all", "pre_aqc0", "first_post_aqc1", "metadata_late")
MODES = ("teacher_forcing", "hybrid")
SEEDS = (17, 43)
MAX_WORKER_RETRIES = 4
STRUCTURE = ("dim", "heads", "encoder_depth", "fusion_depth", "predictor_depth",
             "attention_head_dim", "ffn_dim", "model_size")


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def load_queue():
    spec = importlib.util.spec_from_file_location("matrix_queue_helpers", ROOT / "scripts/run_registered_roi32_experiments.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def matrix_configs(base, *, batch_size=16):
    base = copy.deepcopy(base)
    for name in STRUCTURE:
        base["model"].pop(name, None)
    base.update(training_unit="epochs", max_epochs=200, early_stopping_patience=50,
                early_stopping_min_delta=0., selection_metric="f1", f1_threshold=.5,
                batch_size=batch_size, accumulation=1, precision="bf16", device="cuda:0",
                validation_batch_size=batch_size, diagnostics_batch_size=batch_size,
                landmarks="all_observed", image_cache=None,
                initialize_from_checkpoint=None, encoder_training="frozen",
                task_weight=1., jepa_weight=.5, flow_weight=0., reconstruction_weight=0.,
                variance_weight=0., variance_definition="off", lr=1e-4, weight_decay=.01)
    base["model"].update(encoder_type="vq_encoder", fixed_transition_queries=True,
                         pcr_readout="forecast", history_delta=True, world_residual=True,
                         dropout=.3, enable_flow=False, lesion_world_weighting=False,
                         lesion_pooling=False)
    result = []
    for size in SIZES:
        for phase in PHASES:
            for mode in MODES:
                for seed in SEEDS:
                    value = copy.deepcopy(base)
                    value["model"].update(model_size=size, mri_phases=phase)
                    value.update(prediction_training=mode, seed=seed,
                                 rollout_weight=.5 if mode == "hybrid" else 0.)
                    cfg = from_dict(value)
                    name = f"{size}_{phase}_{mode}_seed{seed}"
                    result.append((name, cfg))
    return result


def gpu_inventory(selectors):
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used",
                                   "--format=csv,noheader,nounits"], text=True)
    rows = []
    for line in raw.strip().splitlines():
        index, uuid, name, total, used = [v.strip() for v in line.split(",")]
        if index in selectors or uuid in selectors:
            total, used = int(total), int(used)
            rows.append({"index": int(index), "uuid": uuid, "name": name,
                         "total_mib": total, "used_mib": used,
                         "budget_mib": total - max(8192, math.ceil(total * .1))})
    if len(rows) != len(selectors):
        raise ValueError("Every requested GPU must identify exactly one available device")
    return rows


def process_memory():
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                                   "--format=csv,noheader,nounits"], text=True)
    result = {}
    for line in raw.strip().splitlines():
        values = [v.strip() for v in line.split(",")]
        if len(values) == 3 and values[0].isdigit() and values[2].isdigit():
            result[int(values[0])] = {"gpu_uuid": values[1], "mib": int(values[2])}
    return result


def gpu_memory_used():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"], text=True)
    return {uuid.strip(): int(value.strip()) for uuid, value in
            (line.split(",") for line in raw.strip().splitlines())}


def environment(gpu):
    return {**os.environ, "CUDA_VISIBLE_DEVICES": gpu["uuid"],
            "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"}


def spawn(command, log_path, gpu):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a")
    try:
        child = subprocess.Popen(command, cwd=ROOT, env=environment(gpu), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        log.close()
    return child


def source_identity():
    return stable_hash(_source_snapshot())


def prepare(output, base_config, codec, batch_size):
    base = load_config(base_config).to_dict()
    base["encoder_checkpoint"] = str(codec.resolve())
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    configs = output / "input_configs"
    configs.mkdir(exist_ok=True)
    result = []
    for name, cfg in matrix_configs(base, batch_size=batch_size):
        path = configs / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
        result.append({"name": name, "size": cfg.model.model_size, "phase": cfg.model.mri_phases,
                       "mode": cfg.prediction_training, "seed": cfg.seed, "config": str(path),
                       "config_digest": cfg.digest,
                       "output": str(output / name), "status": "queued", "retries": 0})
    return result


def preflight(output, jobs, manifest, gpus):
    tasks = [job for job in jobs if job["seed"] == SEEDS[0]]
    active, pending = {}, list(reversed(tasks))  # First confirm small configurations.
    profiles = {}
    folder = output / "preflight"
    folder.mkdir(exist_ok=True)

    def abort(error):
        for item in active.values():
            if item["child"].poll() is None:
                item["child"].terminate()
        for item in active.values():
            try:
                item["child"].wait(timeout=30)
            except subprocess.TimeoutExpired:
                item["child"].kill()
                item["child"].wait()
        raise error

    while pending or active:
        memory = process_memory()
        gpu_used = gpu_memory_used()
        for uuid, record in list(active.items()):
            # NVML may return host PIDs outside the container's PID namespace.
            # Profiling runs exactly one worker on each GPU, so its device total
            # is a conservative fallback when the PID cannot be resolved.
            record["peak"] = max(record["peak"], memory.get(record["child"].pid, {}).get("mib", gpu_used.get(uuid, 0)))
            code = record["child"].poll()
            if code is None:
                continue
            job = record["job"]
            if code != 0:
                log = record["log"].read_text(errors="replace")
                if "out of memory" in log.lower():
                    abort(torch_oom(f"Preflight OOM: {job['name']}"))
                abort(RuntimeError(f"Preflight failed: {job['name']}; see {record['log']}"))
            result = read_json(record["result"])
            result.update(nvml_peak_mib=record["peak"], nvml_scope="exclusive profiling device total if host PID is unavailable",
                          source_identity=source_identity(),
                          config_digest=load_config(job["config"]).digest)
            result["reservation_mib"] = math.ceil(max(result["peak_reserved_mib"], record["peak"]) + 512)
            if result["reservation_mib"] > record["gpu"]["budget_mib"]:
                abort(torch_oom(f"Preflight exceeds reserved GPU budget: {job['name']}"))
            write_json(record["result"], result)
            profiles[profile_key(job)] = result
            del active[uuid]
        for gpu in gpus:
            if gpu["uuid"] in active or not pending:
                continue
            job = pending.pop(0)
            target, log = folder / f"{job['name']}.json", folder / f"{job['name']}.log"
            child = spawn([sys.executable, str(ROOT / "scripts/profile_capacity_phase.py"),
                           "--config", job["config"], "--manifest", str(manifest),
                           "--output", str(target)], log, gpu)
            active[gpu["uuid"]] = {"child": child, "job": job, "gpu": gpu, "peak": 0,
                                     "result": target, "log": log}
        write_json(output / "preflight_status.json", {"updated_at": now(), "completed": len(profiles),
                   "total": len(tasks), "active": [v["job"]["name"] for v in active.values()]})
        if active:
            time.sleep(2)
    write_json(output / "profiles.json", profiles)
    return profiles


class torch_oom(RuntimeError):
    pass


def profile_key(job):
    return f"{job['size']}/{job['phase']}/{job['mode']}"


def benchmark(output, jobs, profiles, manifest, gpus, seconds=20):
    """Choose measured homogeneous limits; mixed admission also respects reservations."""
    folder = output / "throughput"
    folder.mkdir(exist_ok=True)
    limits, measurements = {}, []
    representatives = [j for j in jobs if j["seed"] == 17 and j["phase"] in ("all", "first_post_aqc1")]
    # Test one workload on each GPU at a time; independent groups share no parameters.
    for offset in range(0, len(representatives), len(gpus)):
        selected = list(zip(gpus, representatives[offset:offset + len(gpus)]))
        trials = {gpu["uuid"]: {"gpu": gpu, "job": job, "count": 1, "best": 0., "limit": 1}
                  for gpu, job in selected}
        while trials:
            workers, barriers = [], []
            for uuid, trial in list(trials.items()):
                job, count, gpu = trial["job"], trial["count"], trial["gpu"]
                reservation = profiles[profile_key(job)]["reservation_mib"]
                if count * reservation > gpu["budget_mib"]:
                    limits[benchmark_key(job)] = trial["limit"]
                    del trials[uuid]
                    continue
                stem = folder / f"{job['name']}_n{count}"
                start = stem.with_suffix(".start")
                start.unlink(missing_ok=True)
                barriers.append((start, []))
                for index in range(count):
                    target = Path(str(stem) + f"_{index}.json")
                    ready = Path(str(stem) + f"_{index}.ready")
                    ready.unlink(missing_ok=True)
                    log = target.with_suffix(".log")
                    child = spawn([sys.executable, str(ROOT / "scripts/profile_capacity_phase.py"),
                                   "--config", job["config"], "--manifest", str(manifest),
                                   "--output", str(target), "--benchmark-seconds", str(seconds),
                                   "--ready", str(ready), "--start", str(start)], log, gpu)
                    workers.append((child, target, ready, trial, log))
                    barriers[-1][1].append((child, ready))
            deadline = time.monotonic() + 900
            while any(not ready.exists() and child.poll() is None for child, _, ready, _, _ in workers):
                if time.monotonic() > deadline:
                    for child, *_ in workers:
                        child.terminate()
                    raise TimeoutError("Throughput warmup timed out")
                time.sleep(1)
            for start, _ in barriers:
                start.touch()
            for child, *_ in workers:
                child.wait()
            for uuid, trial in list(trials.items()):
                subset = [w for w in workers if w[3] is trial]
                failed = [w for w in subset if w[0].returncode]
                if failed:
                    if trial["count"] == 1:
                        raise RuntimeError(f"Single-worker throughput check failed: {trial['job']['name']}; see {failed[0][4]}")
                    limits[benchmark_key(trial["job"])] = trial["limit"]
                    measurements.append({"key": benchmark_key(trial["job"]), "count": trial["count"],
                                         "status": "failed", "logs": [str(w[4]) for w in failed]})
                    del trials[uuid]
                    continue
                throughput = sum(read_json(w[1])["patients_per_second"] for w in subset)
                measurements.append({"key": benchmark_key(trial["job"]), "count": trial["count"],
                                     "gpu_uuid": uuid, "patients_per_second": throughput, "status": "passed"})
                if throughput > trial["best"] * 1.05:
                    trial.update(best=throughput, limit=trial["count"], count=trial["count"] + 1)
                else:
                    limits[benchmark_key(trial["job"])] = trial["limit"]
                    del trials[uuid]
            write_json(output / "throughput.json", {"limits": limits, "measurements": measurements})
    return limits


def benchmark_key(job):
    return f"{job['size']}/{'all' if job['phase'] == 'all' else 'single'}/{job['mode']}"


def mixed_key(keys):
    return "+".join(sorted(keys))


def benchmark_mixed(output, jobs, profiles, manifest, gpus, seconds=20):
    """Admit two different sizes together only after measuring that exact pairing."""
    solo = {row["key"]: row["patients_per_second"] for row in read_json(output / "throughput.json")["measurements"]
            if row["count"] == 1 and row["status"] == "passed"}
    pending = []
    for phase in ("all", "first_post_aqc1"):
        for mode in MODES:
            for a, b in (("large", "small"), ("medium", "small"), ("large", "medium")):
                pair = [next(j for j in jobs if (j["size"], j["phase"], j["mode"], j["seed"]) ==
                             (size, phase, mode, 17)) for size in (a, b)]
                pending.append(pair)
    allowed, measurements = [], []
    while pending:
        trials = []
        for gpu in gpus:
            if not pending:
                break
            pair = pending.pop(0)
            key = mixed_key([benchmark_key(j) for j in pair])
            if sum(profiles[profile_key(j)]["reservation_mib"] for j in pair) > gpu["budget_mib"]:
                measurements.append({"key": key, "status": "memory_budget"})
                continue
            stem = output / "throughput" / ("mixed_" + key.replace("/", "_").replace("+", "__"))
            start = Path(str(stem) + ".start")
            start.unlink(missing_ok=True)
            workers = []
            for index, job in enumerate(pair):
                target = Path(str(stem) + f"_{index}.json")
                ready = target.with_suffix(".ready")
                ready.unlink(missing_ok=True)
                child = spawn([sys.executable, str(ROOT / "scripts/profile_capacity_phase.py"),
                               "--config", job["config"], "--manifest", str(manifest),
                               "--output", str(target), "--benchmark-seconds", str(seconds),
                               "--ready", str(ready), "--start", str(start)], target.with_suffix(".log"), gpu)
                workers.append((child, ready, target, job))
            trials.append((key, start, workers, gpu))
        deadline = time.monotonic() + 900
        while any(not ready.exists() and child.poll() is None for _, _, workers, _ in trials
                  for child, ready, _, _ in workers):
            if time.monotonic() > deadline:
                for _, _, workers, _ in trials:
                    for child, *_ in workers:
                        child.terminate()
                raise TimeoutError("Mixed throughput warmup timed out")
            time.sleep(1)
        for _, start, _, _ in trials:
            start.touch()
        for key, _, workers, gpu in trials:
            for child, *_ in workers:
                child.wait()
            failed = any(child.returncode for child, *_ in workers)
            rates = [] if failed else [read_json(target)["patients_per_second"] / solo[benchmark_key(job)]
                                      for _, _, target, job in workers]
            accepted = bool(rates and min(rates) >= .5 and sum(rates) > 1.05)
            if accepted:
                allowed.append(key)
            measurements.append({"key": key, "gpu_uuid": gpu["uuid"], "status": "failed" if failed else "passed",
                                 "normalized_worker_throughput": rates, "admitted": accepted})
        write_json(output / "mixed_throughput.json", {"allowed": allowed, "measurements": measurements})
    return allowed


def freeze_runtime(output):
    target = output / "runtime"
    if target.exists():
        raise FileExistsError("Runtime snapshot already exists")
    shutil.copytree(ROOT / "src", target / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    shutil.copytree(ROOT / "scripts", target / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "train.py", target / "train.py")
    return target


def runtime_hashes(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*.py"))
            if "__pycache__" not in p.parts}


def choose_gpu(job, active, gpus, profiles, limits, gpu_caps, mixed_allowed=()):
    reservation = profiles[profile_key(job)]["reservation_mib"]
    candidates = []
    for gpu in gpus:
        workers = [r for r in active.values() if r["gpu_uuid"] == gpu["uuid"]]
        if ((job.get("force_exclusive_retry") and workers)
                or any(r.get("force_exclusive_retry") for r in workers)):
            continue
        # Mixed workloads use the strictest measured concurrency limit.
        cap = min([limits[benchmark_key(job)], gpu_caps.get(gpu["uuid"], 999)] +
                  [limits[r["benchmark_key"]] for r in workers])
        keys = {benchmark_key(job)} | {r["benchmark_key"] for r in workers}
        if len(keys) > 1:
            if len(workers) != 1 or mixed_key(keys) not in mixed_allowed:
                continue
            cap = min(2, gpu_caps.get(gpu["uuid"], 999))
        reserved = sum(r["reservation_mib"] for r in workers)
        if len(workers) < cap and reserved + reservation <= gpu["budget_mib"]:
            candidates.append((len(workers), reserved, gpu["index"], gpu))
    return min(candidates, key=lambda v: v[:3])[-1] if candidates else None


def launch_candidates(jobs):
    """Seed distinct workloads on idle GPUs before filling matching groups."""
    rounds, candidates = {}, []
    for index, job in enumerate(jobs):
        if job["status"] not in ("queued", "evaluation_pending"):
            continue
        key = benchmark_key(job)
        turn = rounds.get(key, 0)
        rounds[key] = turn + 1
        candidates.append((SIZES.index(job["size"]), turn, index, job))
    return [item[-1] for item in sorted(candidates, key=lambda item: item[:3])]


def worker_failure_log(job, *, training):
    path = Path(job["output"]) / ("training.log" if training else "evaluation.log")
    return (path.read_bytes()[job.get("log_offset", 0):].decode(errors="replace")
            if path.exists() else "")


def retry_worker_failure(job, log, *, training):
    """Retry interrupted/I/O workers, while surfacing deterministic failures."""
    deterministic = any(marker in log for marker in (
        "ValueError:", "TypeError:", "AssertionError:", "FloatingPointError:",
        "KeyError:", "UnpicklingError:", "Error(s) in loading state_dict",
        "size mismatch for ", "invalid load key", "Resume requires last.pt"))
    # OOM has its own stricter, exclusive-GPU recovery policy.
    retryable = not deterministic and "out of memory" not in log.lower()
    job["retryable_failure"] = retryable
    if not retryable or job["retries"] >= MAX_WORKER_RETRIES:
        return False
    stage = "training" if training else job.get("evaluation_kind", "evaluation")
    job.update(status="queued" if training else "evaluation_pending",
               retries=job["retries"] + 1,
               retry_reason=f"Interrupted or transient {stage} worker; retry saved progress")
    job.setdefault("retry_events", []).append({"at": now(), "stage": stage,
                                              "retry": job["retries"], "reason": job["retry_reason"]})
    job.pop("failure_reason", None)
    return True


def aggregate(output, state):
    rows, groups = [], []
    for job in state["jobs"]:
        run = Path(job["output"])
        report_path = run / "train_report.json"
        if not report_path.exists():
            continue
        report = read_json(report_path)
        if not report.get("completed"):
            continue
        best = read_json(run / "evaluation_best.json") if (run / "evaluation_best.json").exists() else {}
        rows.append({**{k: job[k] for k in ("name", "size", "phase", "mode", "seed", "status")},
                     "best_f1": report.get("best_validation_f1"),
                     "best_epoch": report.get("best_epoch"), "stop_epoch": report.get("completed_epochs"),
                     "stop_reason": report.get("stop_reason"),
                     "evaluation": {k: best[k] for k in ("selection_score", "metrics", "per_landmark", "patient_equal_metrics") if k in best},
                     "world": {kind: read_json(run / f"world_{kind}.json") for kind in ("best", "last")
                               if (run / f"world_{kind}.json").exists()},
                     "profile": state["profiles"][profile_key(job)],
                     "wall_seconds": job.get("wall_seconds"), "nvml_peak_mib": job.get("nvml_peak_mib")})
    for size in SIZES:
        for phase in PHASES:
            for mode in MODES:
                subset = [r for r in rows if (r["size"], r["phase"], r["mode"]) == (size, phase, mode)
                          and r["status"] == "completed"]
                values = [r["best_f1"] for r in subset]
                groups.append({"size": size, "phase": phase, "mode": mode,
                               "seeds": [r["seed"] for r in subset], "complete_seed_pair": len(values) == 2,
                               "f1_mean": statistics.mean(values) if values else None,
                               "f1_std": statistics.stdev(values) if len(values) == 2 else None})
                group = groups[-1]
                group["per_landmark"] = {}
                for stage in ("T0", "T1", "T2", "T3"):
                    group["per_landmark"][stage] = {}
                    for metric in ("f1", "auroc", "auprc", "nll", "brier"):
                        scores = [r["evaluation"].get("per_landmark", {}).get(stage, {}).get(metric) for r in subset]
                        scores = [v for v in scores if v is not None]
                        group["per_landmark"][stage][metric] = {
                            "mean": statistics.mean(scores) if scores else None,
                            "std": statistics.stdev(scores) if len(scores) == 2 else None}
    complete = len(rows) == 48 and all(j["status"] == "completed" for j in state["jobs"])
    result = {"updated_at": now(), "completed": complete, "completed_runs": sum(j["status"] == "completed" for j in state["jobs"]),
              "planned_runs": 48, "split": "val", "independent_test": False,
              "training_mode_comparison": "TF: forecast classification from true sources, rollout loss 0; hybrid: autonomous classification, rollout loss .5",
              "groups": groups, "runs": rows}
    write_json(output / "summary.json", result)
    lines = ["# VLA-PCRV1 容量、相位与训练方式实验", "",
             f"更新：{result['updated_at']}；完成 {result['completed_runs']}/48。", "",
             "固定764人训练、102人验证，无独立测试集。F1阈值0.5，全部时点患者等权。", "",
             "|规模|相位|方式|种子|F1均值|F1标准差|", "|---|---|---|---|---:|---:|"]
    for row in groups:
        fmt = lambda v: "—" if v is None else f"{v:.6f}"
        lines.append(f"|{row['size']}|{row['phase']}|{row['mode']}|{row['seeds']}|{fmt(row['f1_mean'])}|{fmt(row['f1_std'])}|")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def work(output):
    queue = load_queue()
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read_json(output / "matrix.json")
        if state["source_identity"] != source_identity():
            raise ValueError("Frozen runtime source differs from matrix identity")
        if state.get("runtime_hashes") != runtime_hashes(ROOT):
            raise ValueError("Frozen runtime scripts changed")
        if digest(state["manifest"]) != state["manifest_sha256"]:
            raise ValueError("Frozen manifest changed")
        first_cfg = load_config(state["jobs"][0]["config"])
        if digest(first_cfg.encoder_checkpoint) != state["codec_sha256"]:
            raise ValueError("Frozen pretrained codec changed")
        for job in state["jobs"]:
            if load_config(job["config"]).digest != job["config_digest"]:
                raise ValueError(f"Frozen config changed: {job['name']}")
        if RawMRIStore(state["manifest"]).signature() != state["data_signature"]:
            raise ValueError("Frozen data identity changed")
        state.update(controller_pid=os.getpid(), status="running", updated_at=now())
        active = {}
        for job in state["jobs"]:
            info = queue.process_info(job.get("pid"))
            if info and info["start_ticks"] == job.get("start_ticks"):
                if job["config"] not in info["command"] or not any(
                        str(part).startswith(job["output"]) for part in info["command"]):
                    raise RuntimeError("Recorded PID identity mismatch")
                active[job["name"]] = {**job, "child": None, "adopted": True}
            elif job["status"] in ("running", "evaluating"):
                report_path = Path(job["output"]) / "train_report.json"
                completed = report_path.exists() and read_json(report_path).get("completed")
                if completed:
                    job["report"] = queue.validate_completed_run(job, Path(state["manifest"]))
                job["status"] = "evaluation_pending" if completed else "queued"
            elif job["status"] in ("failed", "evaluation_failed") and job.get("retryable_failure", True):
                training = job["status"] == "failed"
                retry_worker_failure(job, worker_failure_log(job, training=training), training=training)
        caps = state.setdefault("gpu_concurrency_caps", {})
        manifest = Path(state["manifest"])

        def persist():
            state["updated_at"] = now()
            state["gpus_live"] = [{**gpu, "running": [n for n, r in active.items() if r["gpu_uuid"] == gpu["uuid"]],
                                   "reserved_mib": sum(r["reservation_mib"] for r in active.values() if r["gpu_uuid"] == gpu["uuid"])}
                                  for gpu in state["gpus"]]
            write_json(output / "matrix.json", state)

        while True:
            memory = process_memory()
            used = gpu_memory_used()
            for gpu in state["gpus"]:
                gpu["live_used_mib"] = used.get(gpu["uuid"])
                gpu["peak_used_mib"] = max(gpu.get("peak_used_mib", 0), used.get(gpu["uuid"], 0))
            for job in state["jobs"]:
                item = active.get(job["name"])
                if item is None:
                    continue
                if item["pid"] in memory:
                    job["nvml_peak_mib"] = max(job.get("nvml_peak_mib", 0), memory[item["pid"]]["mib"])
                job["nvml_gpu_peak_mib"] = max(job.get("nvml_gpu_peak_mib", 0), used.get(item["gpu_uuid"], 0))
                progress = Path(job["output"]) / "progress.json"
                if progress.exists():
                    job["progress"] = read_json(progress)
                child = item["child"]
                info = queue.process_info(item["pid"])
                if child is not None and child.poll() is None:
                    continue
                if child is None and info and info["start_ticks"] == item["start_ticks"]:
                    continue
                del active[job["name"]]
                job["wall_seconds"] = job.get("wall_seconds", 0.) + time.time() - job["attempt_started"]
                job["pid"] = None
                code = child.returncode if child is not None else None
                run = Path(job["output"])
                if job["status"] == "running":
                    report = read_json(run / "train_report.json") if (run / "train_report.json").exists() else {}
                    if code in (None, 0) and report.get("completed"):
                        queue.validate_completed_run({**job, "config": job["config"]}, manifest)
                        job.update(status="evaluation_pending", report=report)
                        job.pop("force_exclusive_retry", None)
                    else:
                        log = worker_failure_log(job, training=True)
                        oom = "out of memory" in log.lower()
                        count = sum(r["gpu_uuid"] == item["gpu_uuid"] for r in active.values())
                        tried_alone = job.setdefault("oom_single_retries", {}).get("training", False)
                        if oom and job["retries"] < MAX_WORKER_RETRIES and (count or not tried_alone):
                            caps[item["gpu_uuid"]] = 1
                            job["oom_single_retries"]["training"] = True
                            job.update(status="queued", retries=job["retries"] + 1,
                                       force_exclusive_retry=True,
                                       retry_reason="CUDA OOM; reduce same-GPU concurrency")
                        elif not retry_worker_failure(job, log, training=True):
                            job.update(status="failed", failure_reason=f"Training exit {code}; see training.log")
                else:
                    checkpoint_kind = job["evaluation_kind"]
                    target = (run / f"{checkpoint_kind}.json" if checkpoint_kind.startswith("world_") else
                              run / f"evaluation_{checkpoint_kind}.json")
                    if code in (None, 0) and target.exists():
                        result = read_json(target)
                        if checkpoint_kind == "best":
                            expected = job["report"]["best_selection_score"]
                            if not math.isclose(result["selection_score"], expected, rel_tol=0, abs_tol=1e-7):
                                raise ValueError("Recomputed best F1 differs from saved selection score")
                        job.setdefault("evaluated", []).append(checkpoint_kind)
                        job.pop("force_exclusive_retry", None)
                        job["status"] = ("completed" if set(job["evaluated"]) == {"best", "last", "world_best", "world_last"}
                                         else "evaluation_pending")
                    else:
                        log = worker_failure_log(job, training=False)
                        count = sum(r["gpu_uuid"] == item["gpu_uuid"] for r in active.values())
                        tried_alone = job.setdefault("oom_single_retries", {}).get(checkpoint_kind, False)
                        if ("out of memory" in log.lower() and job["retries"] < MAX_WORKER_RETRIES and (count or not tried_alone)):
                            caps[item["gpu_uuid"]] = 1
                            job["oom_single_retries"][checkpoint_kind] = True
                            job.update(status="evaluation_pending", retries=job["retries"] + 1,
                                       force_exclusive_retry=True,
                                       retry_reason="Evaluation CUDA OOM; reduce same-GPU concurrency")
                        elif not retry_worker_failure(job, log, training=False):
                            job.update(status="evaluation_failed", failure_reason=f"Evaluation exit {code}")
                persist()
                aggregate(output, state)
            # Large-first list; both idle GPUs get a heavy worker before filling shared cards.
            candidates = launch_candidates(state["jobs"])
            for job in candidates:
                gpu = choose_gpu(job, active, state["gpus"], state["profiles"], state["limits"], caps,
                                 state.get("mixed_allowed", ()))
                if gpu is None:
                    continue
                run = Path(job["output"])
                run.mkdir(exist_ok=True)
                if job["status"] == "queued":
                    command = [sys.executable, str(ROOT / "train.py"), "train-vla-jepa",
                               "--config", job["config"], "--manifest", str(manifest), "--output", str(run)]
                    if (run / "last.pt").exists():
                        command.append("--resume")
                    job["status"], log = "running", run / "training.log"
                else:
                    kind = next(k for k in ("best", "last", "world_best", "world_last") if k not in job.get("evaluated", []))
                    world = kind.startswith("world_")
                    checkpoint = kind.removeprefix("world_")
                    command = [sys.executable, str(ROOT / "scripts/profile_capacity_phase.py"),
                               "--config", job["config"], "--manifest", str(manifest),
                               "--world-checkpoint" if world else "--evaluate-checkpoint", str(run / f"{checkpoint}.pt"),
                               "--output", str(run / (f"{kind}.json" if world else f"evaluation_{kind}.json"))]
                    job.update(status="evaluating", evaluation_kind=kind)
                    log = run / "evaluation.log"
                job["log_offset"] = log.stat().st_size if log.exists() else 0
                child = spawn(command, log, gpu)
                info = queue.process_info(child.pid)
                job.update(pid=child.pid, start_ticks=info["start_ticks"] if info else None,
                           gpu_uuid=gpu["uuid"], gpu_index=gpu["index"], attempt_started=time.time(),
                           reservation_mib=state["profiles"][profile_key(job)]["reservation_mib"],
                           benchmark_key=benchmark_key(job))
                active[job["name"]] = {**job, "child": child}
                persist()
            persist()
            if not active:
                if any(j["status"] in ("queued", "evaluation_pending") for j in state["jobs"]):
                    raise RuntimeError("Pending tasks cannot fit measured admission policy")
                break
            time.sleep(3)
        state.update(status="completed" if all(j["status"] == "completed" for j in state["jobs"]) else "failed", ended_at=now())
        persist()
        aggregate(output, state)
        return int(state["status"] != "completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data/registered_roi32/manifest.json")
    parser.add_argument("--base-config", type=Path, default=ROOT / "local_configs/encoder64_seed17.yaml")
    parser.add_argument("--codec", type=Path, default=PROJECT / "data/pretrained/codec.pt")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--benchmark-seconds", type=float, default=20)
    args = parser.parse_args()
    os.umask(0o077)
    output, manifest = args.output.resolve(), args.manifest.resolve()
    if args.status:
        state = read_json(output / "matrix.json")
        print(json.dumps({"status": state["status"], "updated_at": state["updated_at"],
                          "counts": {s: sum(j["status"] == s for j in state["jobs"]) for s in {j["status"] for j in state["jobs"]}},
                          "gpus": state.get("gpus_live"), "jobs": [{k: j.get(k) for k in
                          ("name", "status", "gpu_index", "progress", "failure_reason")} for j in state["jobs"]]}, indent=2))
        return 0
    if args.worker:
        return work(output)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (output / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "matrix.json").exists():
            if not args.resume:
                raise FileExistsError("Use --resume or choose a fresh matrix output")
            queue = load_queue()
            state = read_json(output / "matrix.json")
            info = queue.process_info(state.get("controller_pid"))
            if info and str(output) in info["command"]:
                raise RuntimeError("Matrix controller is already running")
            runtime = output / "runtime"
        else:
            gpus = gpu_inventory(args.gpus.split(","))
            for batch in (16, 8, 4, 2, 1):
                jobs = prepare(output, args.base_config.resolve(), args.codec.resolve(), batch)
                if args.prepare_only:
                    print(json.dumps({"planned": len(jobs), "output": str(output), "batch_size": batch}))
                    return 0
                try:
                    profiles = (read_json(output / "profiles.json") if args.skip_preflight else
                                preflight(output, jobs, manifest, gpus))
                    for job in jobs:
                        if job["seed"] == SEEDS[0]:
                            measured = profiles[profile_key(job)]
                            if (measured.get("source_identity") != source_identity()
                                    or measured.get("config_digest") != job["config_digest"]):
                                raise ValueError("Preflight source/config identity differs; rerun preflight")
                except torch_oom:
                    if batch == 1:
                        raise
                    continue
                break
            if args.preflight_only:
                print(json.dumps({"preflight": "passed", "profiles": len(profiles), "batch_size": batch}))
                return 0
            limits = benchmark(output, jobs, profiles, manifest, gpus, args.benchmark_seconds)
            mixed_allowed = benchmark_mixed(output, jobs, profiles, manifest, gpus, args.benchmark_seconds)
            store = RawMRIStore(manifest)
            if {k: len(store.by_split[k]) for k in ("train", "val", "test")} != {"train": 764, "val": 102, "test": 0}:
                raise ValueError("Matrix requires the agreed fixed 764/102/0 patient split")
            runtime = freeze_runtime(output)
            state = {"schema": "vla_capacity_phase_matrix_v1", "created_at": now(), "updated_at": now(),
                     "status": "prepared", "controller_pid": None, "manifest": str(manifest),
                     "manifest_sha256": digest(manifest), "codec_sha256": digest(args.codec),
                     "data_signature": store.signature(), "source_identity": source_identity(),
                     "runtime": str(runtime), "batch_size": batch, "gpus": gpus,
                     "runtime_hashes": runtime_hashes(runtime),
                     "profiles": profiles, "limits": limits, "mixed_allowed": mixed_allowed, "jobs": jobs}
            write_json(output / "matrix.json", state)
        command = [sys.executable, str(runtime / "scripts/run_capacity_phase_matrix.py"),
                   "--worker", "--output", str(output)]
        env = {**os.environ, "PYTHONPATH": str(runtime / "src"), "PYTHONUNBUFFERED": "1"}
        if args.detach:
            with (output / "controller.log").open("a") as log:
                child = subprocess.Popen(command, cwd=runtime, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write_json(output / "launch.json", {"controller_pid": child.pid, "launched_at": now()})
            print(json.dumps({"controller_pid": child.pid, "output": str(output), "planned_runs": 48}))
            return 0
        return subprocess.call(command, cwd=runtime, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
