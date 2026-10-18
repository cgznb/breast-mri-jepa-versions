#!/usr/bin/env python3
"""Strict CPU replay of completed update-based runs; no new MRI inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import importlib.util
import math
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("overfit_replay_helpers", Path(__file__).with_name("analyze_joint_suite.py"))
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
close, read_json = _helpers.close, _helpers.read_json

from response_jepa.config import from_dict, load_config
from response_jepa.data import Store, canonical_key, planned_tasks
from response_jepa.engine import SCHEMA, make_loader, runtime_info, write_json
from response_jepa.metrics import task_summary
from response_jepa.provenance import source_snapshot

BRANCHES = ("fused", "direct", "forecast")
METRICS = ("nll", "auroc", "auprc", "brier")


def read_plan(path):
    path = Path(path).resolve()
    data = read_json(path)
    # Accept either the user's plan or the queue's frozen plan wrapper.
    data = data.get("plan", data)
    if not isinstance(data.get("runs"), list) or not data["runs"]:
        raise ValueError("Plan requires a nonempty runs list")
    runs, names, outputs = [], set(), set()
    for item in data["runs"]:
        if not {"name", "config", "manifest", "output", "reservation_mib"} <= item.keys():
            raise ValueError("Plan run is missing required fields")
        if item["name"] in names:
            raise ValueError("Plan run names must be distinct")
        run = dict(item)
        for key in ("config", "manifest", "output"):
            run[key] = str((path.parent / item[key]).resolve())
        if run["output"] in outputs:
            raise ValueError("Plan output directories must be distinct")
        names.add(item["name"])
        outputs.add(run["output"])
        runs.append(run)
    return runs


def algorithm_for(cfg):
    key = (cfg.train.sample_weighting == "patient_equal", cfg.model.forecast_detach,
           cfg.loss.consistency > 0)
    return {(False, False, False): "B0", (True, False, False): "B1",
            (False, True, False): "B2", (False, False, True): "B3",
            (True, True, True): "B4"}.get(key, "custom")


def checkpoint_metadata(path):
    # Trusted local artifacts only; map tensors to CPU and avoid materializing
    # model/AdamW arrays. Never construct a model or touch CUDA here.
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if state.get("schema") != SCHEMA or state.get("runner") != "optimizer_updates_v1":
        raise ValueError("Checkpoint schema/runner differs from update-based training")
    keys = ("config", "epoch", "cursor", "updates", "samples_seen", "best_score", "best_update",
            "bad_updates", "history", "normalization", "data_fingerprint", "source_identity",
            "environment", "target_coordinate_origin", "interval_samples", "interval_updates", "interval_sums")
    return {key: state[key] for key in keys}


def expected_cursor(store, cfg, entry, cache):
    """Count batch metadata without iterating datasets or reading MRI pixels."""
    epoch, cursor = entry["epoch"], entry["batch_cursor"]
    if type(epoch) is not int or type(cursor) is not int or min(epoch, cursor) < 0:
        raise ValueError("History contains an invalid epoch/batch cursor")
    for previous in range(epoch+1):
        if previous not in cache:
            tasks = planned_tasks(store, cfg, "train", previous)
            loader = make_loader(store, cfg, tasks, training=True)
            cache[previous] = [len(indices) for indices in loader.batch_sampler]
    if cursor > len(cache[epoch]):
        raise ValueError("History batch cursor exceeds the deterministic loader")
    updates = sum(len(cache[e]) for e in range(epoch)) + cursor
    samples = sum(sum(cache[e]) for e in range(epoch)) + sum(cache[epoch][:cursor])
    if entry["updates"] != updates or entry["samples_seen"] != samples:
        raise ValueError("History update/sample count differs from deterministic batch cursor")


def replay_history(history, cfg, store):
    if not history or history[0]["updates"] != 0:
        raise ValueError("History must include selection of the initial update-zero checkpoint")
    best, selected, selected_index = math.inf, None, None
    previous_update, previous_samples = -1, 0
    cursor_cache = {}
    for index, entry in enumerate(history):
        update = entry["updates"]
        if type(update) is not int or update <= previous_update or update > cfg.train.max_updates:
            raise ValueError("History updates are not strictly ordered within the budget")
        expected_cursor(store, cfg, entry, cursor_cache)
        if entry["training_patients_or_prefixes"] != entry["samples_seen"]-previous_samples:
            raise ValueError("History interval sample count differs from cursor progression")
        if index:
            next_regular = (previous_update // cfg.train.eval_every_updates + 1) * cfg.train.eval_every_updates
            expected_next = min(next_regular, selected+cfg.train.patience_updates, cfg.train.max_updates)
            if update != expected_next:
                raise ValueError("History assessment is missing or differs from the update-based protocol")
        validation = entry["validation"]
        if validation["selection_name"] != "patient_equal_pcr_nll":
            raise ValueError("History selection rule differs from patient-equal fused NLL")
        score = validation["selection"]
        close(score, validation["classification"]["fused"]["patient_equal"]["nll"], "history.selection")
        if not math.isfinite(score):
            raise ValueError("History selection is non-finite")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in entry["train"].values()):
            raise ValueError("History training terms are non-finite")
        if score < best-cfg.train.min_delta:
            best, selected, selected_index = score, update, index
        close(best, entry["best_score"], "history.best_score")
        if entry["best_update"] != selected or entry["bad_updates"] != update-selected:
            raise ValueError("History best-update or patience counter differs")
        previous_update, previous_samples = update, entry["samples_seen"]
    return selected_index, selected, best


def check_predictions(path, store, cfg):
    rows = _helpers.read_predictions(path)
    expected = {(canonical_key(store.patients[i]["patient_key"]), landmark):
                store.patients[i].get("target", {}).get("pcr")
                for i, landmark in planned_tasks(store, cfg, "val")}
    actual = {(row["patient_key"], row["landmark"]): row["label"] for row in rows}
    if actual != expected:
        raise ValueError("Prediction patient-prefix-label cohort differs from planned validation")
    for row in rows:
        for branch in BRANCHES:
            value = row[branch]
            optional = branch == "forecast" and cfg.model.readout == "direct"
            if value is None:
                if not optional:
                    raise ValueError("Required classification branch prediction is absent")
            elif not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Classification probability is invalid")
            elif optional:
                raise ValueError("Direct-only run unexpectedly contains a forecast prediction")
    counts = Counter(row["landmark"] for row in rows)
    return rows, {"patients": len({row["patient_key"] for row in rows}), "prefixes": len(rows),
                  "per_stage": {f"T{stage}": counts[stage] for stage in range(4)}}


def audit_completed(run, *, current_source=None):
    directory = Path(run["output"])
    cfg = from_dict(read_json(directory / "config.json"))
    wanted = load_config(run["config"])
    if cfg.to_dict() != wanted.to_dict():
        raise ValueError("Run config differs semantically from the planned config")
    t = cfg.train
    if t.phase != "joint" or t.max_updates is None or t.accumulation != 1:
        raise ValueError("Run does not use the joint optimizer-update protocol")
    if run.get("profile", t.profile) != t.profile or run.get("seed", t.seed) != t.seed:
        raise ValueError("Planned profile/seed differs from the run")
    algorithm = algorithm_for(cfg)
    if run.get("algorithm", algorithm) != algorithm:
        raise ValueError("Planned algorithm differs from the configured mechanisms")
    store = Store(run["manifest"], t.image_shape, allow_synthetic=t.allow_synthetic, image_cache=t.image_cache)
    fingerprint, normalization = store.fingerprint(), store.normalization
    provenance = read_json(directory / "provenance.json")
    best, last = [checkpoint_metadata(directory / name) for name in ("best.pt", "last.pt")]
    for state in (best, last):
        if from_dict(state["config"]).to_dict() != cfg.to_dict():
            raise ValueError("Checkpoint config differs semantically from the run")
        for key, expected in (("data_fingerprint", fingerprint), ("normalization", normalization),
                              ("environment", runtime_info())):
            if state[key] != expected or provenance[key] != expected:
                raise ValueError(f"Checkpoint/provenance {key} differs from current verified inputs")
        if state["source_identity"] != provenance["source_identity"]:
            raise ValueError("Checkpoint source differs from provenance")
        if current_source is not None and state["source_identity"] != current_source:
            raise ValueError("Checkpoint source differs from the frozen runtime package")
    history = read_json(directory / "history.json")
    selected_index, selected_update, selected_score = replay_history(history, cfg, store)
    selected_entry, final_entry = history[selected_index], history[-1]
    if best["history"] != history[:selected_index+1] or last["history"] != history:
        raise ValueError("Checkpoint history differs from saved history")
    for state, entry in ((best, selected_entry), (last, final_entry)):
        if state["updates"] != entry["updates"] or state["epoch"] != entry["epoch"]:
            raise ValueError("Checkpoint update/epoch differs from history")
        if state["cursor"] != {"epoch": entry["epoch"], "batch": entry["batch_cursor"]}:
            raise ValueError("Checkpoint batch cursor differs from history")
        if state["samples_seen"] != entry["samples_seen"]:
            raise ValueError("Checkpoint sample count differs from history")
        if state["best_update"] != selected_update or state["bad_updates"] != entry["updates"]-selected_update:
            raise ValueError("Checkpoint best-update or patience counter differs")
        close(state["best_score"], selected_score, "checkpoint.best_score")
        if state["interval_updates"] or state["interval_samples"] or state["interval_sums"]:
            raise ValueError("Completed assessment checkpoint contains an unfinished training interval")
        origin = state["target_coordinate_origin"]
        if (origin.get("mode") != "ema" or origin.get("phase") != "joint" or origin.get("seed") != t.seed
                or origin.get("epoch") != state["epoch"] or origin.get("updates") != state["updates"]
                or Path(origin.get("run", "")).resolve() != directory.resolve()):
            raise ValueError("Teacher coordinate origin differs from checkpoint identity")
        if t.world_eval and entry["validation"].get("target_coordinate_origin") != origin:
            raise ValueError("History teacher coordinate origin differs from checkpoint")
    completion = read_json(directory / "completion.json")
    if completion.get("state") != "completed":
        raise ValueError("Run has no completed terminal status")
    stopped = last["updates"]
    reason = "max_updates" if stopped == t.max_updates else "early_stop"
    if stopped > t.max_updates or (reason == "early_stop" and stopped-selected_update < t.patience_updates):
        raise ValueError("Completed run has not reached its update budget or early-stop condition")
    for key, expected in (("updates", stopped), ("max_updates", t.max_updates), ("best_update", selected_update),
                          ("samples_seen", last["samples_seen"]), ("bad_updates", stopped-selected_update),
                          ("stop_reason", reason)):
        if completion.get(key) != expected:
            raise ValueError(f"Completion {key} differs from replayed terminal state")
    close(completion["best_score"], selected_score, "completion.best_score")
    if Path(completion.get("output", "")).resolve() != directory.resolve():
        raise ValueError("Completion output differs from planned run")
    metrics = read_json(directory / "metrics_best.json")
    close(metrics, selected_entry["validation"], "best.validation", atol=1e-6)
    rows, cohort = check_predictions(directory / "predictions_val.csv", store, cfg)
    replayed = task_summary(rows)
    close(replayed, metrics["classification"], "prediction_replay", atol=1e-6)
    close(metrics["selection"], replayed["fused"]["patient_equal"]["nll"], "patient_equal_selection", atol=1e-6)
    if t.world_eval and metrics.get("world") is None:
        raise ValueError("Enabled world evaluation is absent from best metrics")
    world = _helpers.replay_world(directory / "world_val.csv", metrics, best["target_coordinate_origin"],
                                  fingerprint, t.profile)
    return {"algorithm": algorithm, "profile": t.profile, "seed": t.seed,
            "best_update": selected_update, "stop_update": stopped, "stop_reason": reason,
            "samples_seen": last["samples_seen"], "cohort": cohort, "selection": metrics["selection"],
            "classification": replayed, "world": world,
            "best_last": _helpers.best_last_deltas(metrics, final_entry["validation"],
                                                    selected_entry["train"], final_entry["train"]),
            "checks": {key: True for key in ("config", "source", "data", "normalization", "environment",
                "history", "cursor", "initial_selection", "best_selection", "terminal_budget",
                "prediction_cohort", "prediction_replay", "world_replay")}}


def summary(values):
    values = [value for value in values if value is not None]
    return {"seeds": len(values), "mean": statistics.mean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None}


def aggregate(runs):
    per_seed, grouped, selection_groups = [], defaultdict(list), defaultdict(list)
    for run in runs:
        if not run.get("verified"):
            continue
        identity = {key: run[key] for key in ("name", "data_partition", "algorithm", "profile", "seed")}
        selection_groups[run["data_partition"], run["algorithm"], run["profile"]].append(run)
        for branch, result in run["classification"].items():
            if result is None:
                continue
            for stage, scores in result["per_stage"].items():
                if not scores["n"]:
                    continue
                row = {**identity, "branch": branch, "stage": stage, "best_update": run["best_update"], **scores}
                per_seed.append(row)
                grouped[run["data_partition"], run["algorithm"], run["profile"], branch, stage].append(row)
    aggregates = [{"data_partition": partition, "algorithm": algorithm, "profile": profile,
                   "branch": branch, "stage": stage, "metric": metric,
                   **summary([row[metric] for row in rows])}
                  for (partition, algorithm, profile, branch, stage), rows in sorted(grouped.items())
                  for metric in METRICS]
    selections = [{"data_partition": partition, "algorithm": algorithm, "profile": profile,
                   "metric": "patient_equal_fused_nll", **summary([row["selection"] for row in rows])}
                  for (partition, algorithm, profile), rows in sorted(selection_groups.items())]
    return per_seed, aggregates, selections


def analyze_suite(plan_path):
    entries = read_plan(plan_path)
    partitions = {path: f"partition_{index+1}" for index, path in enumerate(sorted({r["manifest"] for r in entries}))}
    current_source = source_snapshot()
    runs, errors, identities = [], [], set()
    for entry in entries:
        run = {"name": entry["name"], "output": entry["output"],
               "data_partition": partitions[entry["manifest"]], "verified": False, "checks": {}}
        completion_path = Path(entry["output"]) / "completion.json"
        if not completion_path.is_file():
            run["state"] = "unfinished"
            runs.append(run)
            continue
        try:
            completion = read_json(completion_path)
            run["state"] = completion.get("state", "unfinished")
            if run["state"] != "completed":
                runs.append(run)
                continue
            audited = audit_completed(entry, current_source=current_source)
            identity = (run["data_partition"], audited["algorithm"], audited["profile"], audited["seed"])
            if identity in identities:
                raise ValueError("Duplicate algorithm/profile/seed within the same data partition")
            identities.add(identity)
            run.update(audited)
            run["verified"] = True
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            message = "required local artifact is inaccessible" if isinstance(exc, OSError) else str(exc)
            errors.append(f"{entry['name']}: {message}")
            run["error"] = message
        runs.append(run)
    per_seed, aggregates, selections = aggregate(runs)
    verified = sum(run["verified"] for run in runs)
    return {"schema": "overfit_suite_cpu_replay_v1", "plan": str(Path(plan_path).resolve()),
            "complete": verified == len(entries) and not errors, "partial": verified != len(entries) or bool(errors),
            "planned_runs": len(entries), "verified_completed_runs": verified, "errors": errors,
            "runs": runs, "per_seed": per_seed, "pcr_seed_aggregate": aggregates,
            "selection_seed_aggregate": selections,
            "scope": "Saved predictions replayed on CPU; no new MRI inference. Development validation only. "
                     "Different manifest partitions are never pooled; standard deviation describes seed variation."}


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(report, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "aggregate.json", report)
    write_csv(output / "pcr_per_seed.csv", report["per_seed"])
    write_csv(output / "pcr_seed_aggregate.csv", report["pcr_seed_aggregate"])
    write_csv(output / "selection_seed_aggregate.csv", report["selection_seed_aggregate"])
    lines = ["# 过拟合改进实验：严格结果回放", "",
             f"已核验完成 {report['verified_completed_runs']}/{report['planned_runs']} 个训练。"
             + ("全部完成。" if report["complete"] else "当前为部分结果；未完成或核验失败的训练不参与聚合。"), "",
             "| 数据分区 | 算法 | 输入模式 | 种子数 | 患者等权 NLL 均值 | 种子标准差 |",
             "|---|---|---|---:|---:|---:|"]
    for row in report["selection_seed_aggregate"]:
        deviation = f"{row['sample_std']:.6f}" if row["sample_std"] is not None else "—"
        lines.append(f"| {row['data_partition']} | {row['algorithm']} | {row['profile']} | {row['seeds']} | "
                     f"{row['mean']:.6f} | {deviation} |")
    lines.extend(["", "B0为同更新预算参考；B1患者等权；B2截断未来分类梯度；B3同前缀一致性；B4三项组合。",
                  "", "配置、源码、环境、数据和归一化、更新/批次游标、初始与最佳选模、终止预算均经过核对。"
                  "预测CSV逐分支、逐阶段及患者等权指标在CPU重算，验证患者/前缀/标签集合严格匹配manifest。"
                  "世界误差在同一最佳checkpoint教师坐标内回放free/copy，不代表MRI像素预测质量。", "",
                  "不同manifest分区分别聚合。种子标准差不代表临床泛化置信区间。内部开发验证用于筛选，"
                  "反复使用的原102人验证也属于开发验证；本报告没有独立测试结论。", "",
                  "本脚本不进行新的MRI推理，不运行GPU，不输出患者记录或校验值。", ""])
    if report["errors"]:
        lines.extend(["## 核验失败", "", *[f"- {error}" for error in report["errors"]], ""])
    (output / "README_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = analyze_suite(args.plan)
    write_report(report, args.output)
    print(f"Verified {report['verified_completed_runs']}/{report['planned_runs']} update runs; "
          f"{'complete' if report['complete'] else 'partial'}; report: {args.output}")
    return 2 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
