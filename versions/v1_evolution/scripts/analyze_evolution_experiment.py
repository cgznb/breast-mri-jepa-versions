#!/usr/bin/env python3
"""Replay and validate any number of staged evolution runs before aggregating."""
from __future__ import annotations

import argparse
import runpy
from datetime import datetime
import importlib.util
import json
from pathlib import Path
from statistics import mean
import sys
from zoneinfo import ZoneInfo

from mri_vla_jepa.data import RawMRIStore
from mri_vla_jepa.io import read_json
from mri_vla_jepa.train_config import from_dict, load_config


ROOT = Path(__file__).resolve().parents[1]
_release = runpy.run_path(str(ROOT / "scripts/release_runtime.py"))
ORIGINAL = _release["version_directory"](ROOT, "v1")
PRIOR_DUAL = _release["version_directory"](ROOT, "dualtask")
DEFAULT_OUTPUT = ROOT / "runs/v1_evolution_20261012"
DEFAULT_MANIFEST = ORIGINAL / "data/registered_roi32/manifest.json"
KINDS = ("best", "last")
HORIZONS = ("1", "2", "3", "all")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _wrapper():
    return _load(ROOT / "scripts/run_evolution_experiment.py", "evolution_analysis_launcher")


def _validators():
    # Reuse the independently replayed pCR metrics, legal-prefix populations,
    # checkpoint/source identity, and own-teacher world/copy checks.
    return _load(ROOT / "scripts/analyze_vq_dualtask_experiment.py", "evolution_artifact_validators")


def _store(manifest, cfg):
    return RawMRIStore(manifest, cfg.image_shape, cfg.allow_synthetic, image_cache=cfg.image_cache,
                       require_lesion_weights=cfg.model.lesion_world_weighting or cfg.model.lesion_pooling)


def _references(store, validators):
    records = []
    for seed in (17, 43):
        paths = {
            "l1_pcr_only": ORIGINAL / f"runs/registered_roi32_l1_l4_20261004/l1_dynamic_seed{seed}",
            "prior_raw_dual": PRIOR_DUAL / f"runs/vq_dualtask_20261012/raw_dynamic_seed{seed}",
            "prior_vq_frozen_dual": PRIOR_DUAL / f"runs/vq_dualtask_20261012/frozen_dynamic_seed{seed}",
        }
        for group, run in paths.items():
            cfg = from_dict(read_json(run / "config.json"))
            if not read_json(run / "train_report.json").get("completed"):
                raise ValueError(f"Historical reference is not completed: {group}")
            records.append({"group": group, "seed": seed, "source": str(run.resolve()),
                            "source_scope": "historical source snapshot; protected read-only reference",
                            "pcr": validators._pcr(run, cfg, store, current_source=False)})
    return records


def _world_means(records):
    result = []
    for group in dict.fromkeys(record["group"] for record in records):
        for kind in KINDS:
            subset = [record for record in records if record["group"] == group and kind in record.get("world", {})]
            if not subset:
                continue
            row = {"group": group, "checkpoint": kind, "seeds": [record["seed"] for record in subset],
                   "complete_seed_pair": {record["seed"] for record in subset} == {17, 43}, "horizons": {}}
            for horizon in HORIZONS:
                values = [record["world"][kind]["metrics"][f"horizon_{horizon}"] for record in subset]
                row["horizons"][horizon] = {task: {
                    "ratio_mean": (mean(value[f"{task}_relative_to_copy"] for value in values)
                                   if all(value[f"{task}_relative_to_copy"] is not None for value in values) else None),
                    "beats_copy_seeds": sum(value[f"{task}_relative_to_copy"] is not None and
                                             value[f"{task}_relative_to_copy"] < 1 for value in values),
                } for task in ("teacher_forcing", "autonomous")}
            values = [record["world"][kind]["teacher_forcing_all_adjacent"]["all"]["teacher_forcing_relative_to_copy"] for record in subset]
            row["all_adjacent_tf_ratio_mean"] = mean(values) if all(value is not None for value in values) else None
            result.append(row)
    return result


def _paired(records, variants):
    index = {(record["group"], record["seed"]): record for record in records if "pcr" in record}
    comparisons = [(variant["name"], variant["control"]) for variant in variants]
    comparisons += [("raw_fixed_direct", "prior_raw_dual"), ("vq_fixed_direct", "prior_vq_frozen_dual"),
                    ("vq_forecast", "vq_copy")]
    result = []
    for candidate, control in comparisons:
        for seed in (17, 43):
            if (candidate, seed) not in index or (control, seed) not in index:
                continue
            for kind in KINDS:
                new, old = [index[group, seed]["pcr"][kind] for group in (candidate, control)]
                delta = {}
                for stage in ("T0", "T1", "T2", "T3"):
                    values = [item["per_landmark"][stage]["auroc"] for item in (new, old)]
                    delta[stage] = values[0] - values[1] if all(value is not None for value in values) else None
                result.append({"candidate": candidate, "control": control, "seed": seed, "checkpoint": kind,
                               "delta_nll": new["selection_nll"] - old["selection_nll"], "delta_auroc": delta})
    return result


def markdown(report):
    fmt = lambda value: "—" if value is None else f"{value:.6f}"
    lines = ["# V1 四阶段结构实验", "", f"核查时间：{report['checked_at']}（北京时间）。",
             f"状态：{report['status']}；完成训练 {report['completed_training_runs']}/{report['planned_training_runs']}，已验证世界报告 {report['validated_world_reports']}/{report['expected_world_reports']}。", "",
             "各组均重新初始化，冻结旧 VQ 前端；raw control 使用 CNN。每组种子 17/43，764 人训练、102 人开发验证。best 依患者平均前缀 NLL 选择，last 为各次训练最新保存权重。均值是种子均值，不是模型集成。", "",
             "基础损失 BCE + 0.5×teacher-forcing JEPA；free 及其后续组另加 0.5×自主多步损失。history、residual、free 是依次增加的三次比较；lesion 和 encoder64 均单独与 history+residual+free 比较。", "",
             "| 方案 | 权重 | 种子 | NLL ↓ | T0 AUC | T1 AUC | T2 AUC | T3 AUC |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["pcr_means"]:
        lines.append(f"| {row['group']} | {row['checkpoint']} | {','.join(map(str,row['seeds']))} | {fmt(row['selection_nll_mean'])} | " +
                     " | ".join(fmt(row["per_landmark"][stage]["auroc"]) for stage in ("T0", "T1", "T2", "T3")) + " |")
    lines += ["", "| 方案 | 权重 | 种子 | 相邻 TF/copy | 自主 1 步/copy | 自主 2 步/copy | 自主 3 步/copy | 自主全步/copy |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["world_means"]:
        lines.append(f"| {row['group']} | {row['checkpoint']} | {','.join(map(str,row['seeds']))} | {fmt(row['all_adjacent_tf_ratio_mean'])} | " +
                     " | ".join(fmt(row["horizons"][horizon]["autonomous"]["ratio_mean"]) for horizon in HORIZONS) + " |")
    lines += ["", "世界指标分别由 best/last 权重真实前向产生。每个比值仅比较该 checkpoint 自己的 EMA teacher 坐标与自己的 copy；不能把跨模型 latent 误差或比值排序当作共同坐标上的状态预测能力排序。病灶加权组的训练空间权重也不构成跨模型评价尺度。", "",
              "历史参考属于独立源码快照，只读。所有 pCR 结果来自重复使用的开发验证集，没有独立测试或显著性结论。待完成组不能据部分结果排序。", "",
              "已保存 best 的患者预测与标签核对并重算指标；last 核对 checkpoint 对应 epoch 聚合指标，不重新推理患者。患者行保留在原 run，不复制进聚合报告。", "",
              "配对变化、阶段依赖和个别运行详见 [analysis.json](analysis.json)。"]
    return "\n".join(lines) + "\n"


def analyze(output, require_complete=True, *, manifest=DEFAULT_MANIFEST, include_references=True):
    output = Path(output).resolve()
    wrapper, validators = _wrapper(), _validators()
    queue = wrapper._load_queue()
    planned = queue.prepared_jobs(output, wrapper.SUITE)
    base_cfg = load_config(planned[0]["config"])
    base_store = _store(manifest, base_cfg)
    _, stages = validators._population(base_store)
    records, completed, worlds = [], 0, 0
    receipt = read_json(output / "controller.json") if (output / "controller.json").exists() else {}
    statuses = {job["name"]: job for job in receipt.get("jobs", [])}
    for job in planned:
        run = Path(job["output"])
        record = {"name": job["name"], "group": job["variant"], "stage": job["stage"], "step": job["step"],
                  "control": job["control"], "seed": job["seed"], "source": str(run.resolve()), "world": {},
                  "queue_status": statuses.get(job["name"], {}).get("status", "not_started"),
                  "depends_on": job["depends_on"]}
        saved = read_json(run / "train_report.json") if (run / "train_report.json").exists() else {}
        complete = saved.get("completed") is True
        if complete:
            queue.validate_completed_run(job, manifest)
            completed += 1
        record["training_completed"] = complete
        if all((run / name).exists() for name in ("best.pt", "last.pt", "best_validation.json", "epoch_history.json")):
            cfg = load_config(job["config"])
            store = _store(manifest, cfg) if cfg.model.lesion_world_weighting or cfg.model.lesion_pooling else base_store
            for attempt in range(3):
                try:
                    record["pcr"] = validators._pcr(run, cfg, store, current_source=True)
                    break
                except validators.SnapshotChanged:
                    if require_complete or complete or attempt == 2:
                        raise
            for kind in KINDS:
                path = run / f"world_{kind}.json"
                if path.exists():
                    record["world"][kind] = validators._world(path, record["pcr"][kind], stages,
                                                                synthetic=bool(store.manifest.get("synthetic", False)))
                    worlds += 1
        elif complete or any((run / f"world_{kind}.json").exists() for kind in KINDS):
            raise ValueError("Completed/world-evaluated run lacks its pCR artifacts")
        records.append(record)
    expected = len(planned)
    if require_complete and (completed != expected or worlds != 2 * expected):
        raise ValueError(f"Final analysis requires {expected} completed runs and {2 * expected} valid world reports; found {completed}/{worlds}")
    references = _references(base_store, validators) if include_references else []
    result = {"schema": "v1_evolution_analysis_v1", "status": "completed" if completed == expected and worlds == 2 * expected else "partial",
              "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
              "output": str(output), "independent_test": False, "validation_patients": len(stages),
              "synthetic": bool(base_store.manifest.get("synthetic", False)),
              "stage_patients": {f"T{stage}": sum(stage in visits for visits in stages.values()) for stage in range(4)},
              "planned_training_runs": expected, "expected_world_reports": 2 * expected,
              "completed_training_runs": completed, "validated_world_reports": worlds,
              "runs": records, "pcr_references": references, "pcr_means": validators._means(records + references),
              "world_means": _world_means(records), "paired_pcr_differences": _paired(records + references, wrapper.VARIANTS),
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
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args(argv)
    report = analyze(args.output, require_complete=not args.partial, manifest=args.manifest)
    print(f"Analysis {report['status']}: {report['completed_training_runs']}/{report['planned_training_runs']} training; "
          f"{report['validated_world_reports']}/{report['expected_world_reports']} world reports; {args.output / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
