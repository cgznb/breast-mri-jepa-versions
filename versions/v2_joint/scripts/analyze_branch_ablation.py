#!/usr/bin/env python3
"""Strict saved-artifact replay for trained readout ablations and original heads."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

import analyze_overfit_suite as replay
from response_jepa.config import load_config
from response_jepa.provenance import source_snapshot
from response_jepa.engine import write_json
from run_branch_ablation import VARIANTS, PROFILES, SEEDS, ablation_config


def analyze(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    entries = replay.read_plan(plan_path)
    identities = plan["variants"]
    if len(entries) != 18 or set(identities) != {r["name"] for r in entries}:
        raise ValueError("Branch plan must contain the locked 18 variants")
    expected = {(v, p, s) for v in VARIANTS for p in PROFILES for s in SEEDS}
    if {(r["variant"], r["profile"], r["seed"]) for r in identities.values()} != expected:
        raise ValueError("Branch variants differ from the locked readout/profile/seed grid")
    refs = {r["name"]: r for r in plan["baseline_references"]}
    if (len(refs) != 6 or {(r["profile"], r["seed"]) for r in refs.values()}
            != {(p, s) for p in PROFILES for s in SEEDS}):
        raise ValueError("Six matched original baseline references are required")
    source = source_snapshot()
    errors, runs, baseline = [], [], []
    for original in refs.values():
        try:
            item = replay.audit_completed(original, current_source=source)
            ablation_config(original["config"], "G_gated")  # original protocol guard
            baseline.append({"name": original["name"], "output": original["output"],
                             "data_partition": "original764_102", "verified": True,
                             **item, "algorithm": "O_original"})
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            errors.append(f"Reference {original['name']}: {type(exc).__name__}: {exc}")
    for entry in entries:
        identity = identities[entry["name"]]
        variant = identity["variant"]
        base = refs[identity["baseline"]]
        if (identity["readout"] != VARIANTS[variant]
                or (identity["profile"], identity["seed"]) != (entry["profile"], entry["seed"])
                or (base["profile"], base["seed"]) != (entry["profile"], entry["seed"])
                or entry["manifest"] != base["manifest"]
                or load_config(entry["config"]).to_dict() != ablation_config(base["config"], variant).to_dict()):
            raise ValueError("Planned readout differs from its matched baseline or locked loss controls")
        run = {"name": entry["name"], "output": entry["output"], "algorithm": variant,
               "profile": entry["profile"], "seed": entry["seed"],
               "data_partition": "original764_102", "verified": False, "checks": {}}
        if not (Path(entry["output"]) / "completion.json").is_file():
            run["state"] = "unfinished"
            runs.append(run)
            continue
        try:
            item = replay.audit_completed(entry, current_source=source)
            run.update(item, algorithm=variant, verified=True, state="completed")
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            run.update(state="verification_failed", error=f"{type(exc).__name__}: {exc}")
            errors.append(f"{entry['name']}: {run['error']}")
        runs.append(run)
    all_runs = runs + baseline
    per_seed, aggregates, selections = replay.aggregate(all_runs)
    primary = [r for r in per_seed if r["branch"] == "fused"]
    primary_aggregates = [r for r in aggregates if r["branch"] == "fused"]
    original_heads = [r for r in aggregates if r["algorithm"] == "O_original"]
    original_selection = []
    for profile in PROFILES:
        cohort = [r for r in baseline if r["profile"] == profile]
        for branch in replay.BRANCHES:
            original_selection.append({"profile": profile, "branch": branch,
                **replay.summary([r["classification"][branch]["patient_equal"]["nll"] for r in cohort])})
    # Paired seed differences are descriptive, not patient-level confidence intervals.
    pairs = []
    for profile in PROFILES:
        lookup = {(r["algorithm"], r["seed"]): r for r in all_runs if r["verified"] and r["profile"] == profile}
        for left, right in (("D_direct", "F_forecast"), ("D_direct", "G_gated"),
                            ("F_forecast", "G_gated"), ("O_original", "G_gated")):
            differences = [lookup[right, seed]["selection"]-lookup[left, seed]["selection"]
                           for seed in SEEDS if (left, seed) in lookup and (right, seed) in lookup]
            pairs.append({"profile": profile, "comparison": f"{right} minus {left}",
                          "metric": "patient_equal_nll", **replay.summary(differences)})
    verified = sum(r["verified"] for r in runs)
    return {"schema": "branch_ablation_cpu_replay_v1", "plan": str(plan_path),
            "complete": verified == 18 and len(baseline) == 6 and not errors,
            "planned_runs": 18, "verified_completed_runs": verified, "errors": errors,
            "runs": runs, "original_references": baseline, "primary_per_seed": primary,
            "primary_stage_aggregate": primary_aggregates, "selection_seed_aggregate": selections,
            "paired_seed_nll_deltas": pairs, "original_branch_stage_aggregate": original_heads,
            "original_branch_nll": original_selection,
            "scope": "D/F/G retrained from scratch with BCE(final)=1 and no auxiliary branch BCE. "
                     "A/B losses retained. Original heads come from joint original checkpoints selected by fused NLL. "
                     "Only final selected output is meaningful for single-readout training; unused diagnostic heads "
                     "may be untrained. All validation is development validation; stage cohorts differ."}


def value(row):
    if row.get("mean") is None:
        return "—"
    return (f"{row['mean']:.4f} ± {row['sample_std']:.4f}" if row.get("sample_std") is not None
            else f"{row['mean']:.4f}")


def write_report(report, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "aggregate.json", report)
    for filename, key in (("primary_per_seed.csv", "primary_per_seed"),
                          ("primary_stage_aggregate.csv", "primary_stage_aggregate"),
                          ("selection_seed_aggregate.csv", "selection_seed_aggregate"),
                          ("paired_seed_nll_deltas.csv", "paired_seed_nll_deltas"),
                          ("original_branch_stage_aggregate.csv", "original_branch_stage_aggregate"),
                          ("original_branch_nll.csv", "original_branch_nll")):
        replay.write_csv(output / filename, report[key])
    lines = ["# 直接分支／未来分支／融合读出的消融", "",
             f"新训练严格核验完成 {report['verified_completed_runs']}/18；"
             + ("全部完成。" if report["complete"] else "当前为部分结果，后台训练或核验尚未全部完成。"), "",
             "D_direct：直接预测；F_forecast：未来状态预测；G_gated：双分支概率门控融合。",
             "三组都从相同种子初始化，分类损失统一为1×最终输出BCE，关闭额外direct/forecast BCE；"
             "保留相同的A表征与B世界模型损失。固定T0和动态各17/43/71，实际batch16/累积1。", "",
             "本报告全部读出消融均属于第二版response_jepa；第一版mri_vla_jepa的损失消融另见旧项目报告。",
             "O_original复用第二版之前的B0模型，不是第一版模型，分类损失为融合1+直接0.5+未来0.25。"
             "D/F/G用于结构比较；G与O的区别是移除分支辅助监督，不能将D/F与O差异全部归因于结构。", "",
             "## 重新训练的主要结果", "",
             "所有模型按患者等权最终输出NLL选每个种子的最佳checkpoint。单读出模型的CSV中fused字段"
             "代表所选最终输出；未监督的备用head不纳入主比较。", "",
             "| 模式 | 训练方案 | 种子数 | 患者等权 NLL（均值 ± 标准差） |",
             "|---|---|---:|---:|"]
    for row in report["selection_seed_aggregate"]:
        lines.append(f"| {row['profile']} | {row['algorithm']} | {row['seeds']} | {value(row)} |")
    lines.extend(["", "## 原双分支模型各head的已有结果", "",
                  "以下为同一原模型、按融合NLL选模后的各head表现，不是分别重训结果。", "",
                  "| 模式 | 输出分支 | 种子数 | 患者等权 NLL（均值 ± 标准差） |",
                  "|---|---|---:|---:|"])
    for row in report["original_branch_nll"]:
        lines.append(f"| {row['profile']} | {row['branch']} | {row['seeds']} | {value(row)} |")
    for title, rows, label in (("新训练各阶段融合或所选输出", report["primary_stage_aggregate"], "algorithm"),
                               ("原动态模型各分支阶段表现", report["original_branch_stage_aggregate"], "branch")):
        lines.extend(["", f"## {title}", "", "| 模式 | 方案／分支 | 阶段 | NLL ↓ | AUROC ↑ | AUPRC ↑ | Brier ↓ |",
                      "|---|---|---|---:|---:|---:|---:|"])
        groups = defaultdict(dict)
        for row in rows:
            if label == "branch" and row["profile"] != "dynamic":
                continue
            groups[row["profile"], row[label], row["stage"]][row["metric"]] = row
        for (profile, name, stage), metrics in sorted(groups.items()):
            lines.append(f"| {profile} | {name} | {stage} | " + " | ".join(value(metrics[m]) for m in replay.METRICS) + " |")
    lines.extend(["", "配置、冻结包源码、数据、归一化、环境、训练更新游标、选模规则和预测均在CPU严格核验。",
                  "原102人已重复用于开发验证，没有独立测试结论；阶段患者集合不同。T3没有未来影像可预测，"
                  "其forecast head读取真实T3教师状态和条件，不能解释为未来预测收益。", ""])
    if report["errors"]:
        lines.extend(["## 核验错误", "", *[f"- {error}" for error in report["errors"]], ""])
    (output / "README_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = analyze(args.plan)
    write_report(report, args.output)
    print(f"Verified {report['verified_completed_runs']}/18 branch ablations; complete={report['complete']}")
    return 2 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
