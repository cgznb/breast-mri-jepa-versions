#!/usr/bin/env python3
"""Validate saved dual-task artifacts and write patient-free partial/final results."""
from __future__ import annotations

import argparse
import runpy
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path
from statistics import mean
import sys
from zoneinfo import ZoneInfo

import torch

from mri_vla_jepa.arm import normalize_patient_key
from mri_vla_jepa.data import RawMRIStore
from mri_vla_jepa.io import load_checkpoint, read_json
from mri_vla_jepa.train_config import from_dict, load_config
from mri_vla_jepa.training import (
    CHECKPOINT_SCHEMA, _checkpoint_config_matches, _patient_partitions, _protocol,
    _scores, _selection_protocol, _source_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
_release = runpy.run_path(str(ROOT / "scripts/release_runtime.py"))
ORIGINAL = _release["version_directory"](ROOT, "v1")
PRIOR_VQ = _release["version_directory"](ROOT, "encoder")
DEFAULT_OUTPUT = ROOT / "runs/vq_dualtask_20261012"
DEFAULT_MANIFEST = ORIGINAL / "data/registered_roi32/manifest.json"
STAGES = ("raw", "frozen", "finetune")
KINDS = ("best", "last")
HORIZONS = ("1", "2", "3", "all")
METRICS = ("n", "positives", "nll", "auroc", "auprc", "brier")


class SnapshotChanged(ValueError):
    """A trainer replaced artifacts while the analysis was reading them."""


def _wrapper():
    spec = importlib.util.spec_from_file_location("dualtask_analysis_launcher", ROOT / "scripts/run_vq_dualtask_experiment.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _equal(actual, expected, label):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"{label}: aggregate keys differ")
        for key in expected:
            _equal(actual[key], expected[key], label)
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isfinite(actual) or abs(actual - expected) > 1e-12:
            raise ValueError(f"{label}: aggregate metric replay differs")
    elif actual != expected:
        raise ValueError(f"{label}: aggregate value differs")


def _finite(value):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("Checkpoint contains a nonfinite floating tensor")
    elif isinstance(value, dict):
        for child in value.values():
            _finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _finite(child)


def _population(store):
    expected_rows, stages = {}, {}
    for index in store.by_split["val"]:
        patient = store.patients[index]
        key = normalize_patient_key(patient["patient_key"])
        stages[key] = set(store.allowed_landmarks(index))
        for stage in stages[key]:
            expected_rows[key, stage] = patient.get("target", {}).get("pcr")
    return expected_rows, stages


def _validate_scores(score, stage_counts):
    if score.get("selection_metric") != _selection_protocol("all_observed"):
        raise ValueError("pCR selection metric differs from dynamic patient-equal NLL")
    selection = score.get("selection_score")
    if not isinstance(selection, (int, float)) or not math.isfinite(selection) or selection < 0:
        raise ValueError("Invalid pCR selection NLL")
    for stage, count in stage_counts.items():
        values = score["per_landmark"][stage]
        if values["n"] != count:
            raise ValueError("pCR stage cohort differs from the manifest")
        for metric in ("nll", "auroc", "auprc", "brier"):
            value = values[metric]
            if value is not None and (not math.isfinite(value) or value < 0 or
                                      (metric != "nll" and value > 1)):
                raise ValueError("Invalid saved pCR metric")


def _pcr(run, cfg, store, *, current_source):
    """Best prediction replay; last checkpoint/history aggregate agreement."""
    paths = [run / name for name in ("best.pt", "last.pt", "best_validation.json", "epoch_history.json")]
    if (run / "train_report.json").exists():
        paths.append(run / "train_report.json")
    identities = {path: _identity(path) for path in paths}
    saved = read_json(run / "best_validation.json")
    history = read_json(run / "epoch_history.json")
    report = read_json(run / "train_report.json") if run / "train_report.json" in identities else None
    states = {kind: load_checkpoint(run / f"{kind}.pt") for kind in KINDS}
    # Validate the read window before expensive tensor checks. All later checks
    # operate on this captured snapshot even if an active trainer saves again.
    if any(_identity(path) != identity for path, identity in identities.items()):
        raise SnapshotChanged("Artifacts changed while reading the pCR snapshot")
    rows_expected, stages = _population(store)
    stage_counts = {f"T{stage}": sum(stage in visits for visits in stages.values()) for stage in range(4)}
    rows = saved["rows"]
    index = {(normalize_patient_key(row["patient_key"]), row["landmark"]): row for row in rows}
    if len(index) != len(rows) or index.keys() != rows_expected.keys():
        raise ValueError("Saved prediction patients/prefixes differ from the manifest")
    if any(row["pcr"] != rows_expected[key] for key, row in index.items()):
        raise ValueError("Saved prediction labels differ from the manifest")
    replay = _scores(rows, cfg.landmarks)
    _equal({key: saved[key] for key in replay}, replay, "Best pCR")
    records = {}
    for kind in KINDS:
        path = run / f"{kind}.pt"
        identity = identities[path]
        state = states[kind]
        if (state.get("schema") != CHECKPOINT_SCHEMA or
                not _checkpoint_config_matches(state["config"], cfg) or
                state.get("protocol") != _protocol(cfg.landmarks) or
                state.get("selection_protocol") != _selection_protocol(cfg.landmarks)):
            raise ValueError("pCR checkpoint configuration/protocol mismatch")
        if current_source and state.get("source_snapshot") != _source_snapshot():
            raise ValueError("pCR checkpoint runtime source mismatch")
        if (state.get("data_signature") != store.signature() or
                state.get("normalization") != store.normalization_state() or
                state.get("patient_partitions") != _patient_partitions(store)):
            raise ValueError("pCR checkpoint data/normalization/population mismatch")
        _finite(state["model"])
        _finite(state["optimizer"])
        if current_source and cfg.initialize_from_checkpoint:
            parent_path = Path(cfg.initialize_from_checkpoint)
            parent_identity = _identity(parent_path)
            parent = load_checkpoint(parent_path)
            initialization = state.get("initialization") or {}
            if (initialization.get("type") != "frozen_stage_checkpoint" or
                    initialization.get("checkpoint") != parent_identity or
                    initialization.get("source_completed_steps") != parent["completed_steps"] or
                    initialization.get("source_best_validation_nll") != parent["best_nll"] or
                    initialization.get("model_weights_loaded") is not True or
                    initialization.get("optimizer_restored") is not False or
                    initialization.get("sampler_restored") is not False or
                    from_dict(parent["config"]).encoder_training != "frozen"):
                raise ValueError("Fine-tuning parent best checkpoint initialization identity mismatch")
            if parent_identity != _identity(parent_path):
                raise ValueError("Fine-tuning parent changed during analysis")
        epoch = (state.get("epoch_state") or {})["completed_epochs"]
        score = state["epoch_history"][-1]["validation"]
        if state["epoch_history"][-1]["epoch"] != epoch or epoch > len(history):
            raise ValueError("pCR checkpoint epoch/history mismatch")
        _equal(score, history[epoch - 1]["validation"], f"{kind} pCR history")
        if kind == "best":
            _equal(score, replay, "Best checkpoint prediction replay")
            if state["best_nll"] != saved["selection_score"]:
                raise ValueError("Best checkpoint selection differs from saved predictions")
        elif report and report.get("completed"):
            if epoch != len(history) or state["completed_steps"] != report["completed_steps"]:
                raise ValueError("Last checkpoint completed progress differs")
        _validate_scores(score, stage_counts)
        records[kind] = {"epoch": epoch, "steps": state["completed_steps"],
                         "checkpoint_identity": identity, "selection_nll": score["selection_score"],
                         "per_landmark": {stage: {name: value.get(name) for name in METRICS}
                                          for stage, value in score["per_landmark"].items()},
                         "validation": "saved_prediction_replay" if kind == "best" else "checkpoint_and_epoch_aggregates"}
    return records


def _world_counts(stages):
    chain, adjacent = {h: [] for h in HORIZONS}, {t: [] for t in ("T0_to_T1", "T1_to_T2", "T2_to_T3", "all")}
    for patient, visits in enumerate(stages.values()):
        for start in visits:
            for target in range(start + 1, 4):
                if all(stage in visits for stage in range(start, target + 1)):
                    chain[str(target - start)].append(patient)
                    chain["all"].append(patient)
            for source in range(start, 3):
                if source in visits and source + 1 in visits:
                    adjacent[f"T{source}_to_T{source + 1}"].append(patient)
                    adjacent["all"].append(patient)
    return ({f"horizon_{key}": (len(set(values)), len(values)) for key, values in chain.items()},
            {key: (len(set(values)), len(values)) for key, values in adjacent.items()})


def _ratio(values, *, adjacent=False):
    """Recompute ratios in one teacher coordinate; preserve null guard semantics."""
    output = {}
    for task in (("teacher_forcing",) if adjacent else ("teacher_forcing", "autonomous")):
        error, copy = values[f"{task}_l1"], values[f"{task}_copy_l1"]
        prefix = "" if adjacent else f"{task}_"
        ratio, skill = values[f"{prefix}relative_to_copy"], values[f"{prefix}skill_vs_copy"]
        if values["patients"] == 0:
            if any(value is not None for value in (error, copy, ratio, skill)):
                raise ValueError("Empty world cohort has nonnull metrics")
        else:
            if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                   for value in (error, copy)):
                raise ValueError("Nonfinite or negative world/copy loss")
            if copy <= 1e-8:
                if ratio is not None or skill is not None:
                    raise ValueError("World ratio violates the near-zero copy guard")
            else:
                _equal(ratio, error / copy, "World/copy ratio")
                _equal(skill, 1 - ratio, "World/copy skill")
        near = values["near_zero_copy_patients" if adjacent else f"{task}_near_zero_copy_patients"]
        if type(near) is not int or not 0 <= near <= values["patients"]:
            raise ValueError("Invalid near-zero-copy patient count")
        output[f"{task}_relative_to_copy"] = ratio
        output[f"{task}_skill_vs_copy"] = skill
        output[f"{task}_near_zero_copy_patients"] = near
    return output


def _world(path, checkpoint, stages, *, synthetic=False):
    data = read_json(path)
    identity = checkpoint["checkpoint_identity"]
    if (data.get("status") != "evaluated" or data.get("world_head_trained") is not True or
            data.get("checkpoint_file_unchanged") is not True or
            data.get("runtime_source_matches_checkpoint") is not True or
            data.get("checkpoint_identity") != identity or data.get("checkpoint") != identity["path"] or
            data.get("checkpoint_epoch") != checkpoint["epoch"] or
            data.get("checkpoint_steps") != checkpoint["steps"]):
        raise ValueError("World checkpoint/source/trained-head identity mismatch")
    if (data.get("split") != "val" or data.get("independent_test") is not False or
            data.get("development_validation") is not True or data.get("synthetic") is not synthetic or
            data.get("precision") != "fp32" or data.get("landmarks") != "all_observed" or
            data.get("patients") != len(stages) or data.get("legal_prefixes") != sum(map(len, stages.values())) or
            data.get("target_coordinates") != "same checkpoint EMA teacher for model and both copy baselines" or
            data.get("chain_policy") != "real continuous visits only; target and all intermediate visits required" or
            data.get("reduction") != "mean eligible prefix-targets within patient, then mean patients" or
            data.get("teacher_forcing_copy") != "real immediate predecessor" or
            data.get("autonomous_copy") != "real landmark state held constant"):
        raise ValueError("World cohort, precision, coordinate, or copy protocol mismatch")
    expected, adjacent_expected = _world_counts(stages)
    metrics, adjacent = {}, {}
    for name, (patients, targets) in expected.items():
        values = data["metrics"][name]
        if (values["patients"], values["prefix_targets"]) != (patients, targets):
            raise ValueError("World horizon cohort differs from the manifest")
        metrics[name] = {"patients": patients, "prefix_targets": targets, **_ratio(values)}
    for name, (patients, pairs) in adjacent_expected.items():
        values = data["teacher_forcing_all_adjacent"]["metrics"][name]
        if (values["patients"], values["prefix_pairs"]) != (patients, pairs):
            raise ValueError("Adjacent teacher-forcing cohort differs from the manifest")
        adjacent[name] = {"patients": patients, "prefix_pairs": pairs, **_ratio(values, adjacent=True)}
    if _identity(identity["path"]) != identity:
        raise ValueError("World checkpoint changed during analysis")
    return {"source": str(path.resolve()), "identity_validated": True,
            "metrics": metrics, "teacher_forcing_all_adjacent": adjacent}


def _references(store):
    records = []
    for group in ("l1_pcr_only", "vq_pcr_only_frozen", "vq_pcr_only_finetune"):
        for seed in (17, 43):
            run = (ORIGINAL / f"runs/registered_roi32_l1_l4_20261004/l1_dynamic_seed{seed}"
                   if group == "l1_pcr_only" else
                   PRIOR_VQ / f"runs/vq_encoder_20261012/{group.removeprefix('vq_pcr_only_')}_dynamic_seed{seed}")
            cfg = from_dict(read_json(run / "config.json"))
            if cfg.jepa_weight or cfg.flow_weight or not read_json(run / "train_report.json").get("completed"):
                raise ValueError("Historical pCR-only reference is not complete or uses world supervision")
            records.append({"group": group, "seed": seed, "source": str(run.resolve()),
                            "source_scope": "historical snapshot, not current dual-task source",
                            "pcr": _pcr(run, cfg, store, current_source=False)})
    return records


def _means(records):
    result = []
    for group in dict.fromkeys(record["group"] for record in records):
        subset = [record for record in records if record["group"] == group and "pcr" in record]
        if not subset:
            continue
        for kind in KINDS:
            values = [record["pcr"][kind] for record in subset]
            result.append({"group": group, "checkpoint": kind, "seeds": [record["seed"] for record in subset],
                           "complete_seed_pair": len(subset) == 2,
                           "selection_nll_mean": mean(value["selection_nll"] for value in values),
                           "per_landmark": {stage: {metric: (mean(value["per_landmark"][stage][metric] for value in values)
                               if all(value["per_landmark"][stage][metric] is not None for value in values) else None)
                               for metric in ("nll", "auroc", "auprc", "brier")} for stage in ("T0", "T1", "T2", "T3")}})
    return result


def _world_means(records):
    result = []
    for group in STAGES:
        for kind in KINDS:
            subset = [record for record in records if record["group"] == group and kind in record.get("world", {})]
            if not subset:
                continue
            row = {"group": group, "checkpoint": kind, "seeds": [record["seed"] for record in subset],
                   "complete_seed_pair": len(subset) == 2, "horizons": {}}
            for horizon in HORIZONS:
                values = [record["world"][kind]["metrics"][f"horizon_{horizon}"] for record in subset]
                row["horizons"][horizon] = {task: {
                    "ratio_mean": (mean(value[f"{task}_relative_to_copy"] for value in values)
                                   if all(value[f"{task}_relative_to_copy"] is not None for value in values) else None),
                    "beats_copy_seeds": sum(value[f"{task}_relative_to_copy"] is not None and
                                             value[f"{task}_relative_to_copy"] < 1 for value in values),
                } for task in ("teacher_forcing", "autonomous")}
            adjacent = [record["world"][kind]["teacher_forcing_all_adjacent"]["all"]["teacher_forcing_relative_to_copy"] for record in subset]
            row["all_adjacent_tf_ratio_mean"] = mean(adjacent) if all(value is not None for value in adjacent) else None
            result.append(row)
    return result


def _paired(records):
    index = {(record["group"], record["seed"]): record for record in records if "pcr" in record}
    result = []
    for candidate, control in (("raw", "l1_pcr_only"), ("frozen", "vq_pcr_only_frozen"),
                               ("finetune", "vq_pcr_only_finetune"), ("frozen", "raw"), ("finetune", "frozen")):
        for seed in (17, 43):
            if (candidate, seed) not in index or (control, seed) not in index:
                continue
            for kind in KINDS:
                new, old = [index[group, seed]["pcr"][kind] for group in (candidate, control)]
                result.append({"candidate": candidate, "control": control, "seed": seed, "checkpoint": kind,
                               "delta_nll": new["selection_nll"] - old["selection_nll"],
                               "delta_auroc": {stage: new["per_landmark"][stage]["auroc"] - old["per_landmark"][stage]["auroc"]
                                               for stage in ("T0", "T1", "T2", "T3")}})
    return result


def markdown(report):
    fmt = lambda value: "—" if value is None else f"{value:.6f}"
    lines = ["# VQ 双任务实验结果", "", f"核查时间：{report['checked_at']}（北京时间）。",
             f"状态：{report['status']}；完成训练 {report['completed_training_runs']}/6，已验证世界报告 {report['validated_world_reports']}/12。", "",
             "损失为 BCE(pCR) + 0.5×JEPA。以下均为同种子算术均值，非集成；只有种子 17/43 都到齐才是完整两种子比较。best 按患者平均开发验证 NLL 选取；last 为最新保存权重，已完成运行的 last 才是训练结束权重。", "",
             "| 方案 | 权重 | 种子 | NLL | T0 AUC | T1 AUC | T2 AUC | T3 AUC |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["pcr_means"]:
        lines.append(f"| {row['group']} | {row['checkpoint']} | {','.join(map(str,row['seeds']))} | {fmt(row['selection_nll_mean'])} | " +
                     " | ".join(fmt(row["per_landmark"][stage]["auroc"]) for stage in ("T0", "T1", "T2", "T3")) + " |")
    lines += ["", "| 双任务方案 | 权重 | 种子 | 相邻 TF/copy | 自主1步/copy | 自主2步/copy | 自主3步/copy | 自主全步/copy |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["world_means"]:
        lines.append(f"| {row['group']} | {row['checkpoint']} | {','.join(map(str,row['seeds']))} | {fmt(row['all_adjacent_tf_ratio_mean'])} | " +
                     " | ".join(fmt(row["horizons"][horizon]["autonomous"]["ratio_mean"]) for horizon in HORIZONS) + " |")
    lines += ["", "ratio 为同一 checkpoint 教师坐标内的 latent L1 / copy L1，小于 1 表示优于该 copy。TF copy 使用真实前一期，自主 copy 保持真实起点；两种分母不同。每个 checkpoint 的 EMA teacher 坐标不同，不能跨模型或 best/last 直接比较原始 latent 误差，也不能把 ratio 排名解释为共同坐标上的能力排名。", "",
              f"已列出 best pCR 的保存预测已独立重放并核对 manifest 人群与标签；last 核对完整 checkpoint、对应 epoch 的保存聚合指标，未重新推理患者预测。当前已核对 {report['validated_world_reports']} 份世界报告的 checkpoint 文件身份、源码、训练过的世界头、合法链和 copy 比值；这些报告由各自权重独立真实前向产生。", "",
              f"原 L1 和此前 VQ 冻结/微调为 pCR-only 历史参考。数据仍是重复使用的 {report['validation_patients']} 名开发验证患者，非独立测试；阶段人群随真实检查可用性变化。部分结果不代表未完成阶段的最终表现。",
              "", "完整聚合：[analysis.json](analysis.json)。患者行、模型权重和原始报告保留在各 run 的原位置。"]
    return "\n".join(lines) + "\n"


def analyze(output: Path, require_complete=True, *, manifest=DEFAULT_MANIFEST, include_references=True):
    output = Path(output).resolve()
    wrapper = _wrapper()
    queue = wrapper._load_queue()
    jobs = queue.prepared_jobs(output, wrapper.SUITE)
    cfg = load_config(jobs[0]["config"])
    store = RawMRIStore(manifest, cfg.image_shape, cfg.allow_synthetic, image_cache=cfg.image_cache)
    _, stages = _population(store)
    records = []
    completed, worlds = 0, 0
    for job in jobs:
        run = Path(job["output"])
        record = {"name": job["name"], "group": job["stage"], "seed": job["seed"], "source": str(run.resolve()), "world": {}}
        report = read_json(run / "train_report.json") if (run / "train_report.json").exists() else {}
        complete = report.get("completed") is True
        if complete:
            queue.validate_completed_run(job, manifest)
            completed += 1
        record["training_completed"] = complete
        if all((run / name).exists() for name in ("best.pt", "last.pt", "best_validation.json", "epoch_history.json")):
            for attempt in range(3):
                try:
                    record["pcr"] = _pcr(run, load_config(job["config"]), store, current_source=True)
                    break
                except SnapshotChanged:
                    if require_complete or complete or attempt == 2:
                        raise
            for kind in KINDS:
                path = run / f"world_{kind}.json"
                if path.exists():
                    record["world"][kind] = _world(path, record["pcr"][kind], stages,
                                                  synthetic=bool(store.manifest.get("synthetic", False)))
                    worlds += 1
        elif complete or any((run / f"world_{kind}.json").exists() for kind in KINDS):
            raise ValueError("Completed/world-evaluated run lacks its pCR artifacts")
        records.append(record)
    if require_complete and (completed != 6 or worlds != 12):
        raise ValueError(f"Final analysis requires 6 completed training runs and 12 valid world reports; found {completed}/{worlds}")
    references = _references(store) if include_references else []
    result = {"schema": "vq_dualtask_analysis_v1", "status": "completed" if completed == 6 and worlds == 12 else "partial",
              "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
              "output": str(output), "independent_test": False, "validation_patients": len(stages),
              "synthetic": bool(store.manifest.get("synthetic", False)),
              "stage_patients": {f"T{stage}": sum(stage in visits for visits in stages.values()) for stage in range(4)},
              "completed_training_runs": completed, "validated_world_reports": worlds,
              "runs": records, "pcr_references": references, "pcr_means": _means(records + references),
              "world_means": _world_means(records), "paired_pcr_differences": _paired(records + references),
              "world_comparison_scope": "Each ratio uses that checkpoint's EMA teacher and own copy; raw latent losses are not pooled across teachers.",
              "last_pcr_validation": "Checkpoint identity and saved checkpoint-epoch aggregates; no last-patient prediction replay."}
    output.mkdir(parents=True, exist_ok=True)
    for path, content in ((output / "analysis.json", json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                          (output / "RESULTS_ZH.md", markdown(result))):
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--partial", action="store_true", help="Allow unfinished training or pending world reports")
    args = parser.parse_args(argv)
    report = analyze(args.output, require_complete=not args.partial, manifest=args.manifest)
    print(f"Analysis {report['status']}: {report['completed_training_runs']}/6 training; {report['validated_world_reports']}/12 world reports; {args.output / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
