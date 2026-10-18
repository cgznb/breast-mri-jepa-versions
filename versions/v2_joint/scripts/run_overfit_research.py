#!/usr/bin/env python3
"""Run five internal ablations, then a locked candidate and matched three-seed controls."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from response_jepa.config import from_dict, load_config
from run_joint_suite import atomic_json, now, read_json, training_environment, matches_process, process_info


OLD_DATA = Path("/root/autodl-tmp/line b/breast-mri-vla-jepa/data/registered_roi32")


def algorithm_config(algorithm, profile, seed, cache):
    cfg = load_config(PROJECT / "configs" / f"joint_{'t0' if profile == 't0' else 'dynamic_all_prefixes'}_batch16_seed17.yaml")
    cfg.train.seed = seed
    cfg.train.profile = profile
    cfg.train.image_cache = str(Path(cache).resolve())
    cfg.train.sample_weighting = "patient_equal" if algorithm in {"B1", "B4"} else "sample"
    cfg.model.forecast_detach = algorithm in {"B2", "B4"}
    cfg.loss.consistency = .1 if algorithm in {"B3", "B4"} else 0.
    cfg.loss.consistency_warmup_updates = 500
    cfg.train.reference_batch_size = 16
    cfg.train.max_updates = 3000
    cfg.train.warmup_updates = 200
    cfg.train.eval_every_updates = 100
    cfg.train.patience_updates = 500
    cfg.train.train_eval_patients = 64
    cfg.train.save_predictions = True
    cfg.train.world_eval = True
    return cfg.validate()


def reservation(algorithm, profile):
    # Real full B16 gates measured before launching this protocol. Room for
    # allocator/context drift remains above each measured process peak.
    if algorithm in {"B3", "B4"}:
        return 14336 if profile == "dynamic" else 12288
    return 10240 if profile == "dynamic" else 8192


def prepare_plan(directory, variants, manifest, cache):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    (directory / "input_configs").mkdir(exist_ok=True)
    runs = []
    for algorithm, profile, seed in variants:
        name = f"{algorithm}_{profile}_seed{seed}"
        cfg = algorithm_config(algorithm, profile, seed, cache)
        config = directory / "input_configs" / f"{name}.yaml"
        content = yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True)
        if config.exists() and config.read_text() != content:
            raise ValueError("Existing experiment config differs; refusing to overwrite")
        if not config.exists():
            config.write_text(content)
            config.chmod(0o600)
        runs.append({"name": name, "algorithm": algorithm, "profile": profile, "seed": seed,
                     "config": str(config), "manifest": str(Path(manifest).resolve()),
                     "output": str(directory / name), "reservation_mib": reservation(algorithm, profile)})
    plan = {"schema": "overfit_research_plan_v1", "runs": runs}
    path = directory / "experiment_plan.json"
    if path.exists() and read_json(path) != plan:
        raise ValueError("Existing experiment plan differs; refusing to mix experiments")
    if not path.exists():
        atomic_json(path, plan)
    return path


def choose_candidate(aggregate):
    if not aggregate.get("complete") or aggregate.get("errors"):
        raise ValueError("Screen results are not fully verified")
    rows = aggregate["runs"]
    baseline = next(row for row in rows if row["algorithm"] == "B0")
    candidates = [row for row in rows if row["algorithm"] != "B0"]
    winner = min(candidates, key=lambda row: (row["selection"], row["algorithm"]))
    return {"algorithm": winner["algorithm"], "selection": winner["selection"],
            "baseline_selection": baseline["selection"],
            "delta_nll": winner["selection"]-baseline["selection"],
            "engineering_target_met": winner["selection"] <= baseline["selection"]-.01,
            "rule": "lowest verified internal patient-equal NLL among B1/B2/B3/B4; algorithm-name tie-break",
            "interpretation": "single-seed development selection; no proven improvement until replication"}


def child(command, log):
    with Path(log).open("a") as handle:
        result = subprocess.run(command, cwd=PROJECT, env=training_environment(), stdin=subprocess.DEVNULL,
                                stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Child failed with exit {result.returncode}; see {log}")


def run_stage(plan, directory, args):
    command = [sys.executable, str(PROJECT / "scripts/run_overfit_suite.py"), "--plan", str(plan),
               "--output", str(directory), "--max-concurrent", "3", "--gpu-memory-budget-mib", str(args.budget)]
    while True:
        saved = read_json(directory / "status.json")
        if saved and all(run["state"] in {"completed", "failed"} for run in saved["runs"]):
            if any(run["state"] != "completed" for run in saved["runs"]):
                raise RuntimeError(f"Queue contains failed training; inspect {directory / 'status.json'}")
            break
        controller = saved.get("controller", {}) if saved else {}
        info = process_info(controller["pid"]) if controller.get("pid") else None
        if matches_process(info, command, PROJECT, controller.get("start_ticks")):
            # A surviving queue owns the trainers. Rejoin it after a research
            # watcher interruption instead of launching a second controller.
            time.sleep(10)
            continue
        child(command, directory / "queue.log")
    report = PROJECT / "reports" / args.output.name / directory.name
    report.mkdir(parents=True, exist_ok=True)
    child([sys.executable, str(PROJECT / "scripts/analyze_overfit_suite.py"), "--plan", str(plan),
           "--output", str(report)], directory / "analysis.log")
    return read_json(report / "aggregate.json"), report


def controller(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (output / ".research.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read_json(output / "research_status.json") or {}
        if state.get("state") == "completed":
            return 0
        def status(phase, **extra):
            state.pop("error", None)
            state.pop("ended_at", None)
            state.update(state="running", phase=phase, pid=os.getpid(),
                         start_ticks=(process_info(os.getpid()) or {}).get("start_ticks"), updated_at=now(), **extra)
            atomic_json(output / "research_status.json", state)
        try:
            if not args.screen_manifest.is_file() or not (args.screen_cache / "index.json").is_file():
                raise ValueError("Strict internal data preparation is not complete")
            status("screening", started_at=state.get("started_at", now()))
            protocol = {"created_at": now(), "scope": "five internal dynamic ablations, then paired full-training replication",
                "screen_patients": "original 764 training patients split internally; original 102 excluded",
                "screen_manifest": str(args.screen_manifest.resolve()), "screen_cache": str(args.screen_cache.resolve()),
                "screen_seed": 17, "screen_algorithms": ["B0", "B1", "B2", "B3", "B4"],
                "final_seeds": [17, 43, 71], "final_profiles": ["t0", "dynamic"],
                "max_updates": 3000, "warmup_updates": 200, "eval_every_updates": 100, "patience_updates": 500,
                "candidate_rule": "minimum verified internal patient-equal NLL among B1/B2/B3/B4",
                "final_control": "new B0 with identical budget, splits and seeds; old six runs descriptive only",
                "validation_scope": "development only; no independent test or efficacy claim",
                "new_algorithm_effect": "unknown; candidate replication still runs if screening target is not met"}
            if not (output / "protocol.json").exists():
                atomic_json(output / "protocol.json", protocol)
            screen = output / "screening"
            plan = prepare_plan(screen, [(f"B{i}", "dynamic", 17) for i in range(5)],
                                args.screen_manifest, args.screen_cache)
            aggregate, report = run_stage(plan, screen, args)
            choice = choose_candidate(aggregate)
            atomic_json(output / "selection.json", choice)
            status("replication", selected_algorithm=choice["algorithm"], screen_report=str(report))
            selected = choice["algorithm"]
            variants = []
            for seed in (17, 43, 71):
                for algorithm in ("B0", selected):
                    for profile in ("dynamic", "t0"):
                        # Patient weighting is exactly one in fixed T0, so B1's
                        # T0 run is numerically the same algorithm as B0. Reuse it.
                        if selected == "B1" and algorithm == "B1" and profile == "t0":
                            continue
                        variants.append((algorithm, profile, seed))
            final = output / "replication"
            final_plan = prepare_plan(final, variants, OLD_DATA / "manifest.json", OLD_DATA / "image_cache_20261004")
            aggregate, report = run_stage(final_plan, final, args)
            if not aggregate.get("complete") or aggregate.get("errors"):
                raise ValueError("Replication result checks did not pass")
            state.update(state="completed", phase="completed", ended_at=now(), final_report=str(report),
                         replication_runs=len(variants), selected_algorithm=selected,
                         fixed_t0_alias="B1 patient-equal weights are all one; reuse B0" if selected == "B1" else None)
            atomic_json(output / "research_status.json", state)
        except BaseException as exc:
            state.update(state="failed", ended_at=now(), error=f"{type(exc).__name__}: {exc}")
            atomic_json(output / "research_status.json", state)
            raise
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "runs/overfit_joint_20261009")
    parser.add_argument("--screen-manifest", type=Path, default=PROJECT / "data/overfit_development_20261009/manifest.json")
    parser.add_argument("--screen-cache", type=Path, default=PROJECT / "data/overfit_development_20261009/image_cache")
    parser.add_argument("--budget", type=int, default=31232)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.detach:
        args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
        command = [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output),
                   "--screen-manifest", str(args.screen_manifest.resolve()), "--screen-cache", str(args.screen_cache.resolve()),
                   "--budget", str(args.budget)]
        saved = read_json(args.output / "research_status.json") or {}
        if saved.get("state") == "completed":
            print(json.dumps({"state": "already_completed", "output": str(args.output)}))
            return 0
        info = process_info(saved["pid"]) if saved.get("pid") else None
        if matches_process(info, command, PROJECT, saved.get("start_ticks")):
            print(json.dumps({"state": "already_running", "research_pid": saved["pid"], "output": str(args.output)}))
            return 0
        with (args.output / "research.log").open("a") as log:
            process = subprocess.Popen(command, cwd=PROJECT, env=training_environment(), stdout=log,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({"research_pid": process.pid, "output": str(args.output)}))
        return 0
    return controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
