#!/usr/bin/env python3
"""Locked three-seed direct / forecast / gated readout ablations, T0 and dynamic."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from response_jepa.config import load_config
from run_overfit_suite import atomic_json, now, read_json, process_info, matches_process, training_environment

BASELINE = PROJECT / "runs/overfit_joint_20261009/replication"
VARIANTS = {"D_direct": "direct", "F_forecast": "forecast", "G_gated": "gated"}
SEEDS = (17, 43, 71)
PROFILES = ("dynamic", "t0")


def ablation_config(baseline, variant):
    cfg = load_config(baseline)
    if (cfg.model.readout != "gated" or cfg.train.phase != "joint"
            or cfg.train.max_updates != 3000 or cfg.train.sample_weighting != "sample"
            or cfg.model.forecast_detach or cfg.loss.consistency):
        raise ValueError("Expected the frozen B0 joint update baseline")
    cfg.model.readout = VARIANTS[variant]
    # A single equal-strength task objective in all three structural variants.
    # In a single-readout model 'fused' is its selected final output, not a mixture.
    cfg.loss.fused, cfg.loss.direct, cfg.loss.forecast = 1., 0., 0.
    return cfg.validate()


def prepare_plan(output, baseline=BASELINE):
    output, baseline = Path(output).resolve(), Path(baseline).resolve()
    old = read_json(baseline / "experiment_plan.json")
    if not old:
        raise ValueError("The completed matched-budget baseline plan is missing")
    references = [run for run in old["runs"] if run.get("algorithm") == "B0"]
    if {(r["profile"], r["seed"]) for r in references} != {(p, s) for p in PROFILES for s in SEEDS}:
        raise ValueError("Expected all six B0 profile/seed references")
    by_identity = {(r["profile"], r["seed"]): r for r in references}
    configs, runs, identities = {}, [], {}
    # Start one of each readout in dynamic mode, then fill vacancies across modes.
    for seed in SEEDS:
        for profile in PROFILES:
            for variant in VARIANTS:
                base = by_identity[profile, seed]
                cfg = ablation_config(base["config"], variant)
                name = f"{variant}_{profile}_seed{seed}"
                config = output / "input_configs" / f"{name}.yaml"
                configs[config] = yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False)
                runs.append({"name": name, "algorithm": "B0", "profile": profile, "seed": seed,
                             "config": str(config), "manifest": str(Path(base["manifest"]).resolve()),
                             "output": str(output / name),
                             "reservation_mib": 10240 if profile == "dynamic" else 8192})
                identities[name] = {"variant": variant, "readout": VARIANTS[variant],
                                    "profile": profile, "seed": seed, "baseline": base["name"]}
    plan = {"schema": "branch_ablation_plan_v1", "runs": runs, "variants": identities,
            "baseline_references": references,
            "task_loss": "1 * BCE(final selected output); direct and forecast auxiliary BCE weights zero",
            "retained_objectives": "A representation + B teacher-forced/free rollout in all variants",
            "comparison": "D vs F vs G: readout only. G vs original B0: auxiliary-head supervision removal.",
            "selection": "patient-equal validation NLL of final output; same max3000/warmup200/eval100/patience500",
            "validation_scope": "original764 training/102 reused development validation; no independent test",
            "source_policy": "unchanged response_jepa package; external orchestration only"}
    path = output / "experiment_plan.json"
    # Validate all existing inputs before writing anything new.
    if path.exists() and read_json(path) != plan:
        raise ValueError("Existing branch protocol differs; refusing to overwrite")
    if any(path.exists() and path.read_text() != content for path, content in configs.items()):
        raise ValueError("Existing branch config differs; refusing to overwrite")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    (output / "input_configs").mkdir(exist_ok=True)
    for config, content in configs.items():
        if not config.exists():
            config.write_text(content, encoding="utf-8")
            config.chmod(0o600)
    if not path.exists():
        atomic_json(path, plan)
    return path


def child(command, log):
    with Path(log).open("a") as handle:
        result = subprocess.run(command, cwd=PROJECT, env=training_environment(), stdin=subprocess.DEVNULL,
                                stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Child exit {result.returncode}; see {log}")


def controller(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (output / ".research.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read_json(output / "research_status.json") or {}
        if state.get("state") == "completed":
            return 0
        try:
            plan = prepare_plan(output)
            state.pop("error", None)
            state.pop("ended_at", None)
            state.update(state="running", phase="training", pid=os.getpid(),
                         start_ticks=(process_info(os.getpid()) or {}).get("start_ticks"),
                         started_at=state.get("started_at", now()), updated_at=now(), planned_runs=18)
            atomic_json(output / "research_status.json", state)
            command = [sys.executable, str(PROJECT / "scripts/run_overfit_suite.py"), "--plan", str(plan),
                       "--output", str(output), "--max-concurrent", "3", "--gpu-memory-budget-mib", str(args.budget)]
            while True:
                queue = read_json(output / "status.json")
                if queue and all(r["state"] in {"completed", "failed"} for r in queue["runs"]):
                    if any(r["state"] != "completed" for r in queue["runs"]):
                        raise RuntimeError("Branch queue has failed jobs; no automatic retraining")
                    break
                owner = queue.get("controller", {}) if queue else {}
                info = process_info(owner["pid"]) if owner.get("pid") else None
                if matches_process(info, command, PROJECT, owner.get("start_ticks")):
                    time.sleep(10)
                else:
                    child(command, output / "queue.log")
            state.update(phase="analysis", updated_at=now())
            atomic_json(output / "research_status.json", state)
            report = PROJECT / "reports" / output.name
            child([sys.executable, str(PROJECT / "scripts/analyze_branch_ablation.py"),
                   "--plan", str(plan), "--output", str(report)], output / "analysis.log")
            aggregate = read_json(report / "aggregate.json")
            if not aggregate or not aggregate.get("complete") or aggregate.get("errors"):
                raise RuntimeError("Branch final artifact verification failed")
            state.update(state="completed", phase="completed", ended_at=now(), final_report=str(report))
            atomic_json(output / "research_status.json", state)
        except BaseException as exc:
            state.update(state="failed", ended_at=now(), error=f"{type(exc).__name__}: {exc}")
            atomic_json(output / "research_status.json", state)
            raise
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "runs/branch_ablation_20261010")
    parser.add_argument("--budget", type=int, default=31232)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.prepare_only:
        print(prepare_plan(args.output))
        return 0
    if args.detach:
        args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
        command = [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output),
                   "--budget", str(args.budget)]
        saved = read_json(args.output / "research_status.json") or {}
        if saved.get("state") == "completed":
            print(json.dumps({"state": "already_completed", "output": str(args.output)}))
            return 0
        info = process_info(saved["pid"]) if saved.get("pid") else None
        if matches_process(info, command, PROJECT, saved.get("start_ticks")):
            print(json.dumps({"state": "already_running", "pid": saved["pid"]}))
            return 0
        with (args.output / "research.log").open("a") as log:
            process = subprocess.Popen(command, cwd=PROJECT, env=training_environment(), stdout=log,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({"state": "started", "research_pid": process.pid, "output": str(args.output)}))
        return 0
    return controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
