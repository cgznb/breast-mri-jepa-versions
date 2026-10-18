#!/usr/bin/env python3
"""Audit the six joint runs and publish aggregate-only, explicitly partial results.

Prediction and checkpoint inputs are trusted local research artifacts. All
checkpoint tensors are mapped to CPU; this script never runs MRI inference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from response_jepa.config import load_config
from response_jepa.data import canonical_key
from response_jepa.provenance import source_snapshot


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 43, 71)
PROFILES = ("t0", "dynamic")
BRANCHES = ("fused", "direct", "forecast")
METRICS = ("nll", "auroc", "auprc", "brier")
SCHEMA = "response_jepa_checkpoint_local_v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def close(a, b, where="metric", atol=1e-8):
    """Compare nested results without putting patient records in exceptions."""
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise ValueError(f"{where}: fields differ")
        for key in a:
            close(a[key], b[key], f"{where}.{key}", atol)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise ValueError(f"{where}: lengths differ")
        for x, y in zip(a, b):
            close(x, y, where, atol)
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        if not isinstance(b, (float, int)) or not math.isfinite(a) or not math.isfinite(b):
            raise ValueError(f"{where}: non-finite or missing value")
        if not math.isclose(a, b, rel_tol=1e-8, abs_tol=atol):
            raise ValueError(f"{where}: values differ")
    elif a != b:
        raise ValueError(f"{where}: values differ")


def classification(labels, probabilities):
    y, p = np.asarray(labels, dtype=float), np.asarray(probabilities, dtype=float)
    if not len(y):
        return {"n": 0, "positives": 0, **dict.fromkeys(METRICS)}
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid validation labels or probabilities")
    q = np.clip(p, 1e-12, 1-1e-12)
    return {"n": len(y), "positives": int(y.sum()),
            "nll": float(np.mean(-y*np.log(q)-(1-y)*np.log1p(-q))),
            "auroc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
            "auprc": float(average_precision_score(y, p)) if len(set(y)) == 2 else None,
            "brier": float(np.mean((y-p)**2))}


def read_predictions(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    converted, seen = [], set()
    for row in rows:
        item = {"patient_key": canonical_key(row["patient_key"]), "landmark": int(row["landmark"]),
                "label": float(row["label"]) if row.get("label") else None}
        identity = (item["patient_key"], item["landmark"])
        if identity in seen:
            raise ValueError("Duplicate patient-prefix prediction")
        seen.add(identity)
        for branch in BRANCHES:
            item[branch] = float(row[branch]) if row.get(branch) else None
        converted.append(item)
    return converted


def prediction_summary(rows):
    """Independent replay: average losses within patient, then across patients."""
    result = {}
    for branch in BRANCHES:
        selected = [r for r in rows if r["label"] is not None and r[branch] is not None]
        if not selected:
            result[branch] = None
            continue
        per_stage = {}
        for stage in range(4):
            sr = [r for r in selected if r["landmark"] == stage]
            per_stage[f"T{stage}"] = classification([r["label"] for r in sr], [r[branch] for r in sr])
        patients = defaultdict(list)
        for row in selected:
            patients[row["patient_key"]].append(row)
        losses = []
        for items in patients.values():
            if len({r["label"] for r in items}) != 1:
                raise ValueError("Patient labels disagree between prefixes")
            losses.append(classification([r["label"] for r in items], [r[branch] for r in items]))
        result[branch] = {"per_stage": per_stage, "patient_equal": {
            "patients": len(patients), "prefixes": len(selected),
            "nll": statistics.mean(r["nll"] for r in losses),
            "brier": statistics.mean(r["brier"] for r in losses)}}
    return result


def check_cohort(rows, fingerprint, profile):
    manifest = fingerprint["manifest"]
    expected = {}
    for patient in manifest["patients"]:
        if patient["split"] != "val":
            continue
        key = canonical_key(patient["patient_key"])
        stages = [0] if profile == "t0" else sorted({
            v["available_at"] for v in patient["visits"] if v is not None})
        for stage in stages:
            expected[key, stage] = patient.get("target", {}).get("pcr")
    actual = {(r["patient_key"], r["landmark"]): r["label"] for r in rows}
    if actual != expected:
        raise ValueError("Validation patient-prefix cohort or labels differ from checkpoint manifest")
    counts = Counter(r["landmark"] for r in rows)
    patients = len({r["patient_key"] for r in rows})
    if not manifest.get("synthetic", False):
        required = {0: 102} if profile == "t0" else {0: 102, 1: 100, 2: 94, 3: 86}
        if counts != required or patients != 102:
            raise ValueError("Unexpected real validation cohort: expected 102 patients and 102/382 prefixes")
    return {"patients": patients, "prefixes": len(rows),
            "per_stage": {f"T{s}": counts[s] for s in range(4)}}


def world_group(rows):
    patients = defaultdict(list)
    for row in rows:
        patients[row["patient_key"]].append(row)
    if not patients:
        return {"patients": 0, "targets": 0, "free_l1": None, "copy_l1": None,
                "ratio": None, "skill": None}
    free, copy = [], []
    for items in patients.values():
        free.append(statistics.mean(r["free_l1"] for r in items))
        copy.append(statistics.mean(r["copy_l1"] for r in items))
    free_mean, copy_mean = statistics.mean(free), statistics.mean(copy)
    ratio = free_mean/copy_mean if copy_mean > 1e-8 else None
    return {"patients": len(patients), "targets": len(rows), "free_l1": free_mean,
            "copy_l1": copy_mean, "ratio": ratio, "skill": 1-ratio if ratio is not None else None,
            "near_zero_copy_patients": sum(value <= 1e-8 for value in copy)}


def replay_world(path, metrics, origin, fingerprint, profile):
    if metrics.get("world") is None:
        return None
    if metrics.get("target_coordinate_origin") != origin:
        raise ValueError("World target coordinates differ from best checkpoint")
    with Path(path).open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    rows, seen = [], set()
    for row in source:
        item = {"patient_key": canonical_key(row["patient_key"]), "landmark": int(row["landmark"]),
                "target_stage": int(row["target_stage"]), "horizon": int(row["horizon"]),
                "complete_chain": row["complete_chain"] in {"True", "true", "1"},
                "free_l1": float(row["free_l1"]), "copy_l1": float(row["copy_l1"])}
        if any(not math.isfinite(item[k]) or item[k] < 0 for k in ("free_l1", "copy_l1")):
            raise ValueError("Invalid world/copy error")
        identity = (item["patient_key"], item["landmark"], item["target_stage"])
        if identity in seen:
            raise ValueError("Duplicate world target row")
        seen.add(identity)
        rows.append(item)
    expected = {}
    for patient in fingerprint["manifest"]["patients"]:
        if patient["split"] != "val":
            continue
        visits = patient["visits"]
        landmarks = [0] if profile == "t0" else sorted({v["available_at"] for v in visits if v is not None})
        for landmark in landmarks:
            observed = {stage for stage, visit in enumerate(visits)
                        if visit is not None and stage <= landmark and visit["available_at"] <= landmark}
            anchor = max(observed)
            known = observed | {stage for stage, visit in enumerate(visits) if visit is not None and stage > landmark}
            for target in range(landmark+1, 4):
                if visits[target] is not None:
                    identity = (canonical_key(patient["patient_key"]), landmark, target)
                    expected[identity] = (target-anchor, all(stage in known for stage in range(anchor, target+1)))
    actual = {(r["patient_key"], r["landmark"], r["target_stage"]):
              (r["horizon"], r["complete_chain"]) for r in rows}
    if actual != expected:
        raise ValueError("World target cohort, horizon or complete-chain mask differs from checkpoint manifest")
    result = {"all": world_group(rows),
              "horizons": {str(h): world_group([r for r in rows if r["horizon"] == h]) for h in (1, 2, 3)},
              "complete_chain": world_group([r for r in rows if r["complete_chain"]])}
    for key, value in result.items():
        close(value, metrics["world"][key], f"world.{key}")
    result["teacher_forcing_recorded_only"] = metrics["world"].get("teacher_forcing")
    result["scope"] = "Same evaluated checkpoint teacher; ratio of patient-equal means. TF rows are not saved."
    return result


def checkpoint_metadata(path):
    # These are our own local checkpoints. mmap avoids copying model/AdamW tensors.
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if state.get("schema") != SCHEMA:
        raise ValueError("Unsupported checkpoint schema")
    keys = ("config", "epoch", "updates", "best_score", "bad_epochs", "history", "normalization",
            "data_fingerprint", "source_identity", "environment", "target_coordinate_origin")
    return {key: state[key] for key in keys}


def replay_selection(history, min_delta):
    selected, best, bad = None, math.inf, 0
    for index, entry in enumerate(history, 1):
        if entry["epoch"] != index:
            raise ValueError("History epochs are not continuous")
        validation = entry["validation"]
        if validation["selection_name"] != "patient_equal_pcr_nll":
            raise ValueError("Unexpected checkpoint selection rule")
        score = validation["selection"]
        if not math.isfinite(score):
            raise ValueError("Non-finite validation selection")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in entry["train"].values()):
            raise ValueError("Non-finite recorded training term")
        close(score, validation["classification"]["fused"]["patient_equal"]["nll"], "history.selection")
        if score < best-min_delta:
            selected, best, bad = index, score, 0
        else:
            bad += 1
        close(best, entry["best_score"], "history.best_score")
        if bad != entry["bad_epochs"]:
            raise ValueError("History early-stopping counter differs")
    if selected is None:
        raise ValueError("Empty completed-run history")
    return selected, best, bad


def best_last_deltas(best_validation, last_validation, train_best, train_last):
    result = {"selection_nll": last_validation["selection"]-best_validation["selection"],
              "train_fused_bce": train_last["fused"]-train_best["fused"]
                  if "fused" in train_last and "fused" in train_best else None,
              "per_stage": {}}
    for branch in BRANCHES:
        bb, ll = best_validation["classification"].get(branch), last_validation["classification"].get(branch)
        if bb is None or ll is None:
            continue
        result["per_stage"][branch] = {}
        for stage in (f"T{s}" for s in range(4)):
            b, l = bb["per_stage"][stage], ll["per_stage"][stage]
            result["per_stage"][branch][stage] = {
                key: l[key]-b[key] if b[key] is not None and l[key] is not None else None for key in METRICS}
    return result


def audit_completed(directory, profile, seed, planned_config=None, current_source=None):
    config = read_json(directory/"config.json")
    train = config["train"]
    if train["phase"] != "joint" or train["profile"] != profile or train["seed"] != seed:
        raise ValueError("Run config does not match requested joint profile/seed")
    if train["batch_size"] != 16 or train["accumulation"] != 1:
        raise ValueError("Run is not physical batch16/accumulation1")
    if profile == "dynamic" and train["prefix_sampling"] != "all_prefixes":
        raise ValueError("Dynamic run does not enumerate all prefixes")
    if planned_config is not None and config != load_config(planned_config).to_dict():
        raise ValueError("Run config differs from queue's planned config")
    provenance = read_json(directory/"provenance.json")
    history = read_json(directory/"history.json")
    best = checkpoint_metadata(directory/"best.pt")
    last = checkpoint_metadata(directory/"last.pt")
    for checkpoint in (best, last):
        if checkpoint["config"] != config:
            raise ValueError("Checkpoint config differs from run config")
        for key in ("source_identity", "data_fingerprint", "normalization", "environment"):
            if checkpoint[key] != provenance[key]:
                raise ValueError(f"Checkpoint {key} differs from provenance")
        if current_source is not None and checkpoint["source_identity"] != current_source:
            raise ValueError("Checkpoint source differs from current runtime package")
    selected, score, bad = replay_selection(history, train["min_delta"])
    if best["epoch"] != selected or last["epoch"] != len(history):
        raise ValueError("Best/last checkpoint epochs do not match replayed history")
    if best["history"] != history[:selected] or last["history"] != history:
        raise ValueError("Checkpoint history differs from saved history")
    close(score, best["best_score"], "best_checkpoint.score")
    close(score, last["best_score"], "last_checkpoint.score")
    if last["bad_epochs"] != bad:
        raise ValueError("Last checkpoint early-stopping counter differs")
    if last["epoch"] < train["epochs"] and bad < train["patience"]:
        raise ValueError("Queue marks run complete before a configured terminal condition")
    metrics = read_json(directory/"metrics_best.json")
    close(metrics, history[selected-1]["validation"], "best.validation")
    rows = read_predictions(directory/"predictions_val.csv")
    cohort = check_cohort(rows, best["data_fingerprint"], profile)
    replayed = prediction_summary(rows)
    close(replayed, metrics["classification"], "prediction_replay")
    close(metrics["selection"], replayed["fused"]["patient_equal"]["nll"], "patient_equal_selection")
    world = replay_world(directory/"world_val.csv", metrics, best["target_coordinate_origin"],
                         best["data_fingerprint"], profile)
    for checkpoint in (best, last):
        origin = checkpoint["target_coordinate_origin"]
        if (origin.get("epoch") != checkpoint["epoch"] or origin.get("updates") != checkpoint["updates"]
                or origin.get("phase") != "joint" or origin.get("seed") != seed
                or origin.get("mode") != "ema" or Path(origin.get("run", "")).resolve() != directory.resolve()):
            raise ValueError("Teacher coordinate origin differs from checkpoint run/epoch/update")
        if checkpoint["updates"] != history[checkpoint["epoch"]-1]["updates"]:
            raise ValueError("Checkpoint updates differ from history")
        if history[checkpoint["epoch"]-1]["validation"].get("target_coordinate_origin") != origin:
            raise ValueError("History teacher coordinate origin differs from checkpoint")
    return {"best_epoch": selected, "last_epoch": last["epoch"], "updates": last["updates"],
            "stop_reason": "early_stopping" if bad >= train["patience"] else "max_epochs",
            "cohort": cohort, "selection": metrics["selection"], "classification": replayed,
            "world": world, "world_last_recorded": history[-1]["validation"].get("world"),
            "best_last": best_last_deltas(metrics, history[-1]["validation"],
                                          history[selected-1]["train"], history[-1]["train"]),
            "checks": {key: True for key in ("config", "source", "data", "normalization", "environment",
                                              "history", "best_selection", "prediction_replay", "world_replay")}}, {
                "config": config, "data": best["data_fingerprint"], "normalization": best["normalization"],
                "source": best["source_identity"]}


def summarize_values(values):
    valid = [value for value in values if value is not None]
    return {"seeds": len(valid), "mean": statistics.mean(valid) if valid else None,
            "sample_std": statistics.stdev(valid) if len(valid) > 1 else None}


def aggregate_results(runs):
    per_seed, aggregate = [], []
    verified = [run for run in runs if run.get("verified")]
    for run in verified:
        for branch, result in run["classification"].items():
            if result is None:
                continue
            for stage, score in result["per_stage"].items():
                if score["n"]:
                    per_seed.append({"run": run["name"], "profile": run["profile"], "seed": run["seed"],
                                     "branch": branch, "stage": stage, "best_epoch": run["best_epoch"], **score})
    groups = defaultdict(list)
    for row in per_seed:
        groups[row["profile"], row["branch"], row["stage"]].append(row)
    for (profile, branch, stage), rows in sorted(groups.items()):
        for metric in METRICS:
            aggregate.append({"profile": profile, "branch": branch, "stage": stage, "metric": metric,
                              "three_seeds_complete": len(rows) == len(SEEDS),
                              **summarize_values([r[metric] for r in rows])})
    selection = []
    for profile in PROFILES:
        items = [run for run in verified if run["profile"] == profile]
        selection.append({"profile": profile, "metric": "patient_equal_fused_nll",
                          "three_seeds_complete": len(items) == len(SEEDS),
                          **summarize_values([r["selection"] for r in items])})
    return per_seed, aggregate, selection


def world_rows(runs):
    rows = []
    for run in runs:
        if not run.get("verified"):
            continue
        for checkpoint, result in (("best", run.get("world")), ("last", run.get("world_last_recorded"))):
            if result is None:
                continue
            groups = {"all": result["all"], "complete_chain": result["complete_chain"],
                      **{f"horizon_{h}": score for h, score in result["horizons"].items()}}
            tf = result.get("teacher_forcing_recorded_only", result.get("teacher_forcing"))
            if tf is not None:
                groups["teacher_forcing"] = tf
            for group, score in groups.items():
                rows.append({"run": run["name"], "profile": run["profile"], "seed": run["seed"],
                             "checkpoint": checkpoint, "group": group,
                             "replayed_from_rows": checkpoint == "best" and group != "teacher_forcing",
                             **{key: score.get(key) for key in ("patients", "targets", "free_l1", "copy_l1", "ratio", "skill")}})
    return rows


def historical_reference(report, path):
    """Descriptive same-seed comparison; aggregate CSV cannot prove patient identity."""
    path = Path(path).resolve()
    if not report["cross_run_identity_consistent"]:
        return {"csv": str(path), "comparisons": [], "same_seed_summary": [],
                "scope": "Comparison withheld: new runs have inconsistent identities."}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {}
    for row in rows:
        profile, seed = row["profile"], int(row["seed"])
        experiment = {"t0": "A1", "dynamic": "L1"}.get(profile)
        if row["experiment"] != experiment or seed not in (17, 43):
            continue
        key = (profile, seed, row["stage"])
        if key in lookup:
            raise ValueError("Duplicate selected baseline experiment/profile/seed/stage")
        lookup[key] = row
    comparisons = []
    for row in report["per_seed"]:
        if row["branch"] != "fused" or row["seed"] not in (17, 43):
            continue
        old = lookup.get((row["profile"], row["seed"], row["stage"]))
        if old is None:
            raise ValueError("Missing selected baseline profile/seed/stage")
        if int(old["n"]) != row["n"] or int(old["positives"]) != row["positives"]:
            raise ValueError("Baseline and new result stage patient/positive counts differ")
        for metric in METRICS:
            value = float(old[metric]) if old[metric] else None
            if value is not None and not math.isfinite(value):
                raise ValueError("Non-finite baseline metric")
            delta = row[metric]-value if value is not None and row[metric] is not None else None
            comparisons.append({"profile": row["profile"], "seed": row["seed"], "stage": row["stage"],
                                "baseline_experiment": old["experiment"], "metric": metric,
                                "n": row["n"], "positives": row["positives"], "baseline": value,
                                "joint": row[metric], "delta_joint_minus_baseline": delta,
                                "joint_better": delta < 0 if metric in {"nll", "brier"} and delta is not None
                                    else delta > 0 if delta is not None else None})
    groups = defaultdict(list)
    for row in comparisons:
        groups[row["profile"], row["stage"], row["metric"]].append(row)
    summary = []
    for (profile, stage, metric), items in sorted(groups.items()):
        values = [r["delta_joint_minus_baseline"] for r in items]
        summary.append({"profile": profile, "stage": stage, "metric": metric,
                        "same_two_seeds_available": {r["seed"] for r in items} == {17, 43},
                        "improved_seeds": sum(r["joint_better"] is True for r in items),
                        **summarize_values(values)})
    return {"csv": str(path), "comparisons": comparisons, "same_seed_summary": summary,
            "scope": "Old A1 t0 and L1 dynamic, seeds17/43 only; stage counts/positives checked. "
                     "Aggregate rows cannot establish identical patient membership. Training budgets differ; "
                     "descriptive differences, no patient-paired statistical inference. "
                     "Old dynamic patient-equal overall NLL cannot be reconstructed from stage means. "
                     "No seed71 baseline exists, and no three-seed/two-seed mean comparison is claimed."}


def progress_preview(directory):
    path = directory/"history.json"
    if not path.exists():
        return None
    try:
        history = read_json(path)
        if not history:
            return None
        row = history[-1]
        return {"latest_epoch": row["epoch"], "updates": row["updates"],
                "current_nll": row["validation"]["selection"], "provisional_best_nll": row["best_score"],
                "last_epoch_seconds": row["seconds"], "scope": "unfinished history; excluded from final metrics"}
    except (KeyError, ValueError, OSError, TypeError):
        return {"scope": "history not readable at this snapshot; rerun later"}


def analyze_suite(suite_root, *, check_current_source=True):
    suite_root = Path(suite_root).resolve()
    status_path = suite_root/"status.json"
    status = read_json(status_path) if status_path.exists() else {"runs": []}
    entries = {row["name"]: row for row in status.get("runs", [])}
    source = source_snapshot() if check_current_source else None
    runs, identities, errors = [], [], []
    for profile in PROFILES:
        for seed in SEEDS:
            name = f"joint_{profile}_seed{seed}"
            entry = entries.get(name, {})
            directory = Path(entry.get("output", suite_root/name)).resolve()
            item = {"name": name, "profile": profile, "seed": seed, "state": entry.get("state", "not_recorded"),
                    "directory": str(directory), "verified": False}
            if item["state"] == "completed":
                try:
                    details, identity = audit_completed(directory, profile, seed,
                                                       entry.get("config"), current_source=source)
                    item.update(details, verified=True)
                    identities.append(identity)
                except (KeyError, ValueError, OSError, RuntimeError, TypeError) as exc:
                    item.update(state="invalid", error=str(exc))
                    errors.append(f"{name}: {exc}")
            else:
                item["progress"] = progress_preview(directory)
            runs.append(item)
    comparable = True
    if identities:
        first = identities[0]
        for identity in identities[1:]:
            for key in ("data", "normalization", "source"):
                if identity[key] != first[key]:
                    comparable = False
            a, b = first["config"], identity["config"]
            ta = {k: v for k, v in a["train"].items() if k not in {"profile", "seed", "prefix_sampling"}}
            tb = {k: v for k, v in b["train"].items() if k not in {"profile", "seed", "prefix_sampling"}}
            if a["model"] != b["model"] or a["loss"] != b["loss"] or ta != tb:
                comparable = False
    if not comparable:
        errors.append("Verified runs have inconsistent model/loss/config/source/data/normalization; comparison withheld")
    per_seed, aggregate, selection = aggregate_results(runs)
    complete = sum(run["verified"] for run in runs) == 6 and comparable
    return {"schema": "joint_suite_analysis_v1", "suite_root": str(suite_root),
            "complete": complete, "partial": not complete, "expected_runs": 6,
            "verified_completed_runs": sum(run["verified"] for run in runs),
            "cross_run_identity_consistent": comparable, "runs": runs, "errors": errors,
            "per_seed": per_seed, "aggregate": aggregate if comparable else [], "world_per_seed": world_rows(runs),
            "selection_aggregate": selection if comparable else [],
            "interpretation": {
                "selection": "patient-equal fused pCR NLL; lower is better",
                "seed_reduction": "mean and sample standard deviation (ddof=1); never best-of-seed",
                "stage_cohorts": "T0/T1/T2/T3 differ: 102/100/94/86 patients for dynamic",
                "budget": "Dynamic all-prefix has 2657 rows/167 updates per epoch; T0 764 rows/48 updates",
                "world": "World/copy share each checkpoint teacher; raw joint JEPA is not fixed across epochs",
                "validation": "102 repeatedly used development validation patients; no independent test",
                "replay": "Saved probability metrics replayed independently; no new MRI/model inference",
                "privacy": "Aggregate-only outputs; private patient rows stay in run directories"}}


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def formatted(value):
    return "—" if value is None else f"{value:.5f}"


def write_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output/"aggregate.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(output/"aggregate.json")
    write_csv(output/"pcr_per_seed.csv", report["per_seed"],
              ["run", "profile", "seed", "branch", "stage", "best_epoch", "n", "positives", *METRICS])
    write_csv(output/"pcr_seed_aggregate.csv", report["aggregate"],
              ["profile", "branch", "stage", "metric", "three_seeds_complete", "seeds", "mean", "sample_std"])
    write_csv(output/"selection_seed_aggregate.csv", report["selection_aggregate"],
              ["profile", "metric", "three_seeds_complete", "seeds", "mean", "sample_std"])
    write_csv(output/"world_per_seed.csv", report["world_per_seed"],
              ["run", "profile", "seed", "checkpoint", "group", "replayed_from_rows",
               "patients", "targets", "free_l1", "copy_l1", "ratio", "skill"])
    baseline = report.get("historical_reference")
    if baseline is not None:
        write_csv(output/"historical_same_seed_comparison.csv", baseline["comparisons"],
                  ["profile", "seed", "stage", "baseline_experiment", "metric", "n", "positives",
                   "baseline", "joint", "delta_joint_minus_baseline", "joint_better"])
    lines = ["# Joint 固定 T0 / 动态三种子结果", "",
             f"**{'全部完成' if report['complete'] else '部分结果：训练或核对尚未全部完成'}**。"
             f"已核对完成 {report['verified_completed_runs']} / 6 个训练。", "",
             "只把队列标记 completed 且配置、CPU checkpoint、数据、归一化、历史选模与预测重算均通过的训练纳入结果。"
             "未完成种子的临时 best 不进入最终指标。", "",
             "| Profile | Seed | 状态 | 最终轮次 | Best 轮次 | Patient-equal NLL |", "|---|---:|---|---:|---:|---:|"]
    for run in report["runs"]:
        preview = run.get("progress") or {}
        epoch = run.get("last_epoch", preview.get("latest_epoch", "—"))
        lines.append(f"| {run['profile']} | {run['seed']} | {run['state']} | {epoch} | "
                     f"{run.get('best_epoch', '—')} | {formatted(run.get('selection'))} |")
    lines.extend(["", "未完成状态及临时进度见 aggregate.json；所有患者预测仅在各 run 的私密 CSV 中，报告不导出患者行。", ""])
    complete_selections = [r for r in report["selection_aggregate"] if r["three_seeds_complete"]]
    if complete_selections:
        lines.extend(["## 三种子均值和样本标准差", "", "| Profile | Patient-equal fused NLL |", "|---|---:|"])
        for row in complete_selections:
            lines.append(f"| {row['profile']} | {formatted(row['mean'])} ± {formatted(row['sample_std'])} |")
        lines.extend(["", "完整的 fused/direct/forecast 各 T0–T3 NLL、AUROC、AUPRC、Brier 见 pcr_per_seed.csv 和 pcr_seed_aggregate.csv。", ""])
    else:
        lines.extend(["两个 profile 的三种子结果尚未齐全，不给出最终优劣结论。CSV 中少于三种子的统计明确标记 three_seeds_complete=false。", ""])
    verified = [r for r in report["runs"] if r["verified"]]
    if verified:
        lines.extend(["## Best 到 Last 与 world/copy", "",
                      "| Run | Last−Best NLL | Last−Best train fused BCE | Best free/copy 比值 |", "|---|---:|---:|---:|"])
        for run in verified:
            ratio = run["world"]["all"]["ratio"] if run.get("world") else None
            lines.append(f"| {run['name']} | {formatted(run['best_last']['selection_nll'])} | "
                         f"{formatted(run['best_last']['train_fused_bce'])} | {formatted(ratio)} |")
        lines.extend(["", "NLL/Brier 差值大于 0、AUROC/AUPRC 差值小于 0 表示 Last 恶化；使用已选 Best checkpoint。"
                      "world/copy 比值小于 1 表示同一 checkpoint 的预测优于复制锚点；TF 仅核对原记录，未保存逐患者 TF 行。", ""])
    if report["errors"]:
        lines.extend(["## 核对未通过", "", *[f"- {error}" for error in report["errors"]], ""])
    if baseline is not None:
        lines.extend(["## 旧主线同种子描述性参照", "",
                      "旧固定 T0 使用 A1，旧动态使用 L1；仅匹配 seed 17/43 与同阶段，seed 71 没有旧参照。"
                      "检查各阶段人数和阳性数一致，但 aggregate CSV 无法证明患者成员完全一致。"
                      "训练预算不同，不做患者配对显著性或三种子对两种子平均的比较。", "",
                      "| Profile | Stage | NLL Joint−旧参照均值 | 完成相同种子数 | NLL 改善种子数 |", "|---|---|---:|---:|---:|"])
        for row in baseline["same_seed_summary"]:
            if row["metric"] == "nll":
                lines.append(f"| {row['profile']} | {row['stage']} | {formatted(row['mean'])} | "
                             f"{row['seeds']} / 2 | {row['improved_seeds']} |")
        lines.extend(["", "所有指标和每个种子差值见 historical_same_seed_comparison.csv。"
                      "旧动态 overall patient-equal NLL 不能从阶段均值还原，报告只比较各阶段指标。", ""])
    lines.extend(["## 解释范围", "",
                  "动态输入是每位患者全部合法累计前缀；固定 T0 仅输入 T0。每 epoch 分别有 167 和 48 次更新，"
                  "相同 epoch 不等于相同训练预算。动态 T0–T3 的患者数分别 102/100/94/86，阶段间直接差异含有队列变化。", "",
                  "联合训练的 EMA teacher 随训练移动；不能把不同轮次原始 JEPA 值视为固定坐标的误差。"
                  "三种子标准差描述训练随机性，并非临床泛化置信区间。102 人为反复使用的开发验证集，无独立测试集。", "",
                  "本脚本在 CPU 读取本地 checkpoint，独立重算已保存预测指标，不进行新的 MRI 推理。", ""])
    (output/"README_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--output", default=str(ROOT/"reports"/"joint_three_seed_20261009"))
    parser.add_argument("--baseline-csv", help="Optional old pcr_per_seed.csv; A1 t0 / L1 dynamic seeds17/43 only")
    args = parser.parse_args()
    report = analyze_suite(args.suite_root)
    if args.baseline_csv:
        report["historical_reference"] = historical_reference(report, args.baseline_csv)
    write_report(report, args.output)
    print(f"Verified {report['verified_completed_runs']}/6 runs; "
          f"{'complete' if report['complete'] else 'partial'}; report: {args.output}")
    if report["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
