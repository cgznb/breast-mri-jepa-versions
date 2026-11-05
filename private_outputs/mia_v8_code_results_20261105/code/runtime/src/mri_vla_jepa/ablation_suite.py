"""Patient-safe preparation, reporting and evaluated-weight retention for the 18-run suite.

This module does not launch workers or touch the historical manifest/codec.  Test
results are deliberately omitted from public reports until every run is complete.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
import fcntl
import html
import io
import math
import os
from pathlib import Path
import statistics

import sklearn
from sklearn.model_selection import train_test_split
import yaml

from .arm import normalize_patient_key
from .io import digest, read_json, write_json
from .landmark_profiles import profile_stages

SEEDS = (17, 43)
LONGITUDINAL_VARIANTS = ("warmup_cosine", "plateau", "weight_decay", "dropout", "ema", "attention", "t0_delta")
T0_VARIANTS = ("t0_forecast", "t0_direct")
VARIANTS = LONGITUDINAL_VARIANTS + T0_VARIANTS
METRICS = ("f1", "auroc", "auprc", "nll", "brier", "ece_10_equal_width")
COUNTS = ("n", "positives", "tp", "fp", "fn")
DIAGNOSTICS = ("pcr_spatial_attention_entropy", "pcr_spatial_effective_tokens")


def selection_cohort(variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown ablation variant")
    return "t0" if variant in T0_VARIANTS else "all_observed"


def _identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "size": path.stat().st_size, "sha256": digest(path)}


def _stratum(patient):
    label = patient.get("target", {}).get("pcr")
    if type(label) not in {int, float} or label not in (0, 1):
        raise ValueError("Splitting requires a binary pCR label for every patient")
    visits = patient.get("visits")
    if not isinstance(visits, list) or len(visits) != 4 or visits[0] is None:
        raise ValueError("Splitting requires four canonical visits including observed T0")
    return f"{int(label)}:" + "".join("1" if v is not None else "0" for v in visits)


def prepare_splits(source_manifest, output, *, seeds=SEEDS, codec=None,
                   expected_patients=866, val_size=87, test_size=173):
    """Write seed-specific manifests and audits; never alter source or MRI arrays."""
    source = Path(source_manifest).resolve()
    original = read_json(source)
    patients = sorted(deepcopy(original["patients"]),
                      key=lambda p: normalize_patient_key(p["patient_key"]))
    keys = [normalize_patient_key(p["patient_key"]) for p in patients]
    if len(patients) != expected_patients or len(keys) != len(set(keys)):
        raise ValueError("Unexpected cohort size or duplicate normalized patient keys")
    if type(val_size) is not int or type(test_size) is not int or min(val_size, test_size) < 1:
        raise ValueError("Validation and test sizes must be positive integers")
    if val_size + test_size >= len(patients):
        raise ValueError("Training partition must be nonempty")
    strata = [_stratum(p) for p in patients]
    source_identity = _identity(source)
    codec_identity = _identity(codec) if codec is not None else None
    # Relocation must retain exactly the original asset resolution.
    for patient in patients:
        for visit in patient["visits"]:
            if visit is not None:
                for field in ("image", "phases"):
                    if field not in visit:
                        continue
                    values = [visit[field]] if field == "image" else visit[field]
                    resolved = [str((source.parent / Path(p)).resolve()) for p in values]
                    visit[field] = resolved[0] if field == "image" else resolved
    result = {}
    for seed in seeds:
        if type(seed) is not int or seed < 0:
            raise ValueError("Split seed must be a nonnegative integer")
        development, test = train_test_split(list(range(len(patients))), test_size=test_size,
            random_state=seed, stratify=strata, shuffle=True)
        train, val = train_test_split(development, test_size=val_size,
            random_state=seed, stratify=[strata[i] for i in development], shuffle=True)
        indices = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
        assignment = {i: split for split, values in indices.items() for i in values}
        derived = deepcopy(original)
        derived["patients"] = [dict(deepcopy(p), split=assignment[i]) for i, p in enumerate(patients)]
        audit = {"schema": "vq_finetune_split_audit_v1", "seed": seed,
            "algorithm": "sklearn.model_selection.train_test_split: test first, then validation; same random_state",
            "sklearn_version": sklearn.__version__, "sort": "normalized_patient_key_lexicographic",
            "stratification": "pCR_label_x_four_visit_availability_pattern",
            "source_manifest": source_identity, "codec": codec_identity,
            "counts": {split: len(values) for split, values in indices.items()},
            "patients": {split: [patients[i]["patient_key"] for i in values] for split, values in indices.items()},
            "stratum_counts": {split: dict(sorted(Counter(strata[i] for i in values).items()))
                               for split, values in indices.items()},
            "landmark_counts": {split: {f"T{stage}": sum(stage in {
                v["available_at"] for v in patients[i]["visits"] if v is not None}
                for i in values) for stage in range(4)} for split, values in indices.items()},
            "independence_scope": "downstream_only", "patient_overlap": False}
        provenance = deepcopy(original.get("provenance", {}))
        provenance.update({"independence_scope": "downstream_only", "split_seed": seed,
            "source_manifest": source_identity, "pretrained_codec": codec_identity,
            "image_standardization": "historical_preprocessed_arrays_unchanged_not_refit_on_new_train",
            "codec_pretraining": "historical_pretrained_codec_original_population_not_redefined",
            "clinical_standardization": "fit_by_RawMRIStore_on_this_manifest_train_only",
            "split_algorithm": audit["algorithm"],
            "test_interpretation": "Held out from this downstream training and checkpoint selection only"})
        derived["provenance"] = provenance
        folder = Path(output) / f"seed{seed}"
        manifest_path, audit_path = folder / "manifest.json", folder / "split_audit.json"
        # Repreparation must be deterministic rather than silently rewriting inputs.
        for path, value in ((manifest_path, derived), (audit_path, audit)):
            if path.exists() and read_json(path) != value:
                raise ValueError(f"Existing prepared split differs: {path}")
            write_json(path, value)
            path.chmod(0o600)
        result[str(seed)] = {"manifest": str(manifest_path.resolve()), "audit": str(audit_path.resolve()),
            "manifest_sha256": digest(manifest_path), "audit_sha256": digest(audit_path),
            "split_counts": audit["counts"], "landmark_counts": audit["landmark_counts"]}
    return result


def variant_configs(base_config):
    """Return eighteen strict config objects, with no standalone baseline."""
    from .train_config import from_dict
    base = deepcopy(base_config)
    base.update(encoder_training="finetune", encoder_lr=1e-4, lr=1e-4,
        initialize_from_checkpoint=None, weight_decay=.01, lr_scheduler="none",
        training_unit="epochs", schedule="one_stage", max_epochs=200,
        early_stopping_patience=50, early_stopping_min_delta=0., batch_size=16, accumulation=1,
        representation_steps=0, freeze_encoder_after_representation=False,
        checkpoint_every=38, validation_every=1, validation_batch_size=16,
        diagnostics_every_epochs=5, diagnostics_patients=32,
        diagnostics_batch_size=16, diagnostics_seed=20261004, selection_metric="f1", f1_threshold=.5,
        landmarks="all_observed", prediction_training="teacher_forcing", task_weight=1.,
        jepa_weight=.5, rollout_weight=0., flow_weight=0., reconstruction_weight=0.,
        variance_weight=0., variance_definition="off", image_cache=None,
        warmup_epochs=10, warmup_start_factor=.1, cosine_end_epoch=100, cosine_min_lr=1e-6,
        plateau_factor=.5, plateau_patience=5, plateau_min_lr=1e-6, plateau_threshold=0.)
    base["model"].update(model_size="large", mri_phases="all", dropout=.3, ema_decay=.995,
        spatial_readout="mean", pcr_delta_reference="current", encoder_type="vq_encoder",
        vq_activation_checkpointing=True, fixed_transition_queries=True,
        pcr_readout="forecast", world_residual=True, history_delta=True, enable_flow=False,
        lesion_world_weighting=False, lesion_pooling=False, enable_world_model=True)
    # A small/custom template must fail rather than silently inherit an unintended
    # architecture. The caller supplies the frozen large/all/TF baseline template.
    from .model_config import MODEL_SIZE_PRESETS
    for key, expected in MODEL_SIZE_PRESETS["large"].items():
        if base["model"].get(key) != expected:
            raise ValueError(f"Suite requires the large template: {key} must be {expected!r}")
    if tuple(base["model"].get("token_grid", ())) != (2, 4, 4):
        raise ValueError("Suite requires the original 32-token grid [2,4,4]")
    if tuple(base.get("image_shape", ())) != (32, 128, 128):
        raise ValueError("Suite requires the original ROI32 image shape")
    result = []
    for variant in VARIANTS:
        for seed in SEEDS:
            value = deepcopy(base)
            value["seed"] = seed
            if variant in {"warmup_cosine", "plateau"}:
                value["lr_scheduler"] = variant
            elif variant == "weight_decay":
                value["weight_decay"] = .05
            elif variant == "dropout":
                value["model"]["dropout"] = .1
            elif variant == "ema":
                value["model"]["ema_decay"] = .999
            elif variant == "attention":
                value["model"]["spatial_readout"] = "attention"
            elif variant == "t0_delta":
                value["model"]["pcr_delta_reference"] = "t0"
            elif variant == "t0_forecast":
                value["landmarks"] = "t0"
            elif variant == "t0_direct":
                value["landmarks"] = "t0"
                value["jepa_weight"] = 0.
                value["model"].update(pcr_readout="zero_state", enable_world_model=False)
            result.append((f"{variant}_seed{seed}", from_dict(value)))
    return result


def prepare_suite(output, base_config_path, source_manifest, codec):
    """Prepare immutable protocol, manifests, configs and queued job dictionaries."""
    from .train_config import load_config
    output = Path(output).resolve()
    base = load_config(base_config_path).to_dict()
    base["encoder_checkpoint"] = str(Path(codec).resolve())
    splits = prepare_splits(source_manifest, output / "splits", codec=codec)
    protocol = {"schema": "vq_finetune_ablation_protocol_v1", "seeds": list(SEEDS),
        "variants": list(VARIANTS), "baseline_trained": False, "split_artifacts": splits,
        "base_config": _identity(base_config_path), "source_manifest": _identity(source_manifest),
        "codec": _identity(codec), "selection_metric": "validation_f1_by_selection_cohort",
        "selection_cohorts": {v: selection_cohort(v) for v in VARIANTS},
        "f1_threshold": .5, "test_visibility": "public_summary_only_when_entire_suite_complete",
        "independence_scope": "downstream_only", "retention": "delete_generated_weights_after_verified_all_evaluations"}
    protocol_path = output / "suite_protocol.json"
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise ValueError("Existing suite protocol differs")
    write_json(protocol_path, protocol)
    jobs = []
    for name, cfg in variant_configs(base):
        path = output / "input_configs" / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        value = cfg.to_dict()
        if path.exists() and yaml.safe_load(path.read_text()) != value:
            # YAML loads tuples as lists; config normalization handles that.
            if load_config(path).digest != cfg.digest:
                raise ValueError(f"Existing suite config differs: {path}")
        path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
        path.chmod(0o600)
        variant = name.rsplit("_seed", 1)[0]
        jobs.append({"name": name, "variant": variant, "seed": cfg.seed,
            "landmarks": cfg.landmarks, "selection_cohort": selection_cohort(variant),
            "world_applicable": cfg.model.enable_world_model,
            "config": str(path), "config_digest": cfg.digest, "config_sha256": digest(path),
            "manifest": splits[str(cfg.seed)]["manifest"],
            "manifest_sha256": splits[str(cfg.seed)]["manifest_sha256"],
            "output": str(output / "runs" / name), "status": "queued", "retries": 0})
    return jobs


def _expected_rows(manifest, split, profile="all_observed"):
    if profile not in {"all_observed", "t0", "t0_t2"}:
        raise ValueError("Expected all_observed, t0 or t0_t2 evaluation profile")
    admitted = profile_stages(profile)
    result = {}
    for patient in read_json(manifest)["patients"]:
        if patient["split"] == split:
            key = normalize_patient_key(patient["patient_key"])
            for stage in {v["available_at"] for v in patient["visits"] if v is not None}:
                if stage not in admitted:
                    continue
                result[(key, stage)] = patient["target"]["pcr"]
    return result


def validate_classification_report(path, *, manifest=None, split=None, profile=None):
    """Re-read predictions and recompute every metric, checking exact coverage."""
    from .training import _scores
    report = read_json(path)
    if split is not None and report.get("split") != split:
        raise ValueError("Evaluation split mismatch")
    split = report.get("split")
    actual_profile = report.get("landmarks")
    if (split not in {"val", "test"} or actual_profile not in {"all_observed", "t0", "t0_t2"}
            or (profile is not None and profile != actual_profile)):
        raise ValueError("Evaluation input/selection cohort mismatch")
    if report.get("f1_threshold") != .5 or report.get("metric_name") != "f1":
        raise ValueError("Evaluation must retain the fixed F1 selection protocol and threshold 0.5")
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Missing per-patient predictions")
    observed = {}
    for row in rows:
        key = (normalize_patient_key(row["patient_key"]), row["landmark"])
        if key in observed or type(key[1]) is not int or key[1] not in range(4):
            raise ValueError("Duplicate or invalid patient/landmark prediction")
        observed[key] = row["pcr"]
    if actual_profile == "t0" and any(stage != 0 for _, stage in observed):
        raise ValueError("T0-only evaluation cannot contain later-visit predictions")
    if actual_profile == "t0_t2" and any(stage not in profile_stages(actual_profile) for _, stage in observed):
        raise ValueError("T0-T2 evaluation cannot contain T3 predictions")
    if manifest is not None and observed != _expected_rows(manifest, split, actual_profile):
        raise ValueError("Evaluation prediction coverage/labels differ from manifest")
    calculated = _scores(rows, actual_profile, selection_metric="f1", f1_threshold=.5)
    for section in ("metrics", "patient_equal_metrics"):
        saved = report.get(section, {})
        for metric in METRICS + COUNTS:
            expected, actual = calculated[section].get(metric), saved.get(metric)
            if expected != actual and (expected is None or actual is None or
                    not math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-10)):
                raise ValueError(f"Saved evaluation metric differs: {section}/{metric}")
    stages = profile_stages(actual_profile) if actual_profile == "t0_t2" else range(4)
    if actual_profile == "t0_t2" and set(report.get("per_landmark", {})) != {f"T{stage}" for stage in stages}:
        raise ValueError("T0-T2 evaluation must report exactly T0, T1 and T2 landmark metrics")
    for stage in stages:
        saved = report.get("per_landmark", {}).get(f"T{stage}", {})
        computed = calculated["per_landmark"][f"T{stage}"]
        for metric in METRICS + COUNTS:
            expected, actual = computed.get(metric), saved.get(metric)
            if expected != actual and (expected is None or actual is None or
                    not math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-10)):
                raise ValueError(f"Saved landmark metric differs: T{stage}/{metric}")
    return report


def _table_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    temporary = Path(str(path) + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(stream.getvalue(), encoding="utf-8")
    os.replace(temporary, path)
    Path(path).chmod(0o600)


def _diagnostic_means(rows):
    result = {}
    for field in DIAGNOSTICS:
        values = [float(row[field]) for row in rows if field in row]
        if values:
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Nonfinite attention diagnostics")
            result[field + "_mean"] = statistics.mean(values)
            result[field + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.
    return result


def write_suite_reports(output, jobs, *, suite_complete=False):
    """Write public val summaries; include test only after all eighteen complete.

    Job ``evaluations`` optionally maps val/test to report paths.  Predictions
    remain in their original private run reports after weights are deleted.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # A --report-only process may run while the controller refreshes reports.
    # Serialize the shared atomic-temp filenames used by the JSON/CSV writers.
    with (output / ".report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _write_suite_reports(output, jobs, suite_complete=suite_complete)


def _write_suite_reports(output, jobs, *, suite_complete=False):
    expected_jobs = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    all_complete = (len(jobs) == len(expected_jobs) and
        {(job["variant"], job["seed"]) for job in jobs} == expected_jobs and
        all(job.get("status") == "completed" for job in jobs))
    reveal_test = bool(suite_complete and all_complete)
    overall, timepoints = [], []
    for job in jobs:
        run = Path(job["output"])
        cohort = selection_cohort(job["variant"])
        for split in ("val", "test"):
            path = Path(job.get("evaluations", {}).get(split, run / f"evaluation_best_{split}.json"))
            state = "withheld" if split == "test" and not reveal_test else "pending"
            report = None
            if state != "withheld" and path.exists():
                report = validate_classification_report(path, manifest=job.get("manifest"), split=split, profile=cohort)
                state = "evaluated"
            common = {"variant": job["variant"], "seed": job["seed"], "split": split,
                "selection_cohort": cohort, "world_diagnostics": "not_applicable" if job["variant"] == "t0_direct" else "applicable",
                "status": state, "checkpoint": "best", "epoch": job.get("best_epoch"),
                "f1_threshold": .5, "independence_scope": "downstream_only"}
            if report is not None:
                common["epoch"] = report.get("checkpoint_epoch", report.get("epoch", common["epoch"]))
            overall.append({**common, "timepoint": "overall",
                **{key: report["metrics"].get(key) if report else None for key in METRICS + COUNTS},
                **(_diagnostic_means(report["rows"]) if report else {})})
            for stage in range(4):
                applicable = cohort == "all_observed" or stage == 0
                metrics = report["per_landmark"][f"T{stage}"] if report else {}
                timepoints.append({**common, "timepoint": f"T{stage}",
                    "applicable": applicable, "status": state if applicable else "not_applicable",
                    "checkpoint": common["checkpoint"] if applicable else None,
                    "epoch": common["epoch"] if applicable else None,
                    **{key: metrics.get(key) if applicable else None for key in METRICS + COUNTS},
                    **(_diagnostic_means([r for r in report["rows"] if r["landmark"] == stage]) if report and applicable else {})})
    summaries = []
    for variant in VARIANTS:
        for split in ("val", "test"):
            for timepoint in ("overall", "T0", "T1", "T2", "T3"):
                selected = [row for row in overall + timepoints if row["variant"] == variant
                            and row["split"] == split and row["timepoint"] == timepoint]
                result = {"variant": variant, "split": split, "timepoint": timepoint,
                          "selection_cohort": selection_cohort(variant),
                          "status": "not_applicable" if variant in T0_VARIANTS and timepoint in {"T1", "T2", "T3"} else "applicable",
                          "completed_seeds": sum(row["status"] == "evaluated" for row in selected)}
                for metric in METRICS:
                    values = [row[metric] for row in selected if row[metric] is not None]
                    result[metric + "_mean"] = statistics.mean(values) if len(values) == len(SEEDS) else None
                    result[metric + "_std"] = statistics.stdev(values) if len(values) == len(SEEDS) else None
                summaries.append(result)
    rank_key = lambda r: (r["f1_mean"] is None, -(r["f1_mean"] or 0), VARIANTS.index(r["variant"]))
    rankings = {cohort: sorted([r for r in summaries if r["split"] == "val" and
        r["timepoint"] == "overall" and r["selection_cohort"] == cohort], key=rank_key)
        for cohort in ("all_observed", "t0")}
    ranking = rankings["all_observed"] + rankings["t0"]
    order = {row["variant"]: i for i, row in enumerate(ranking)}
    t0_order = {r["variant"]: i for i, r in enumerate(sorted(
        [r for r in summaries if r["split"] == "val" and r["timepoint"] == "T0"], key=rank_key))}
    t0_comparison = sorted([dict(r, comparison="T0_at_original_selection_checkpoint") for r in summaries
        if r["timepoint"] == "T0"], key=lambda r: (t0_order[r["variant"]], r["split"] != "val"))
    for rows in (overall, timepoints, summaries):
        rows.sort(key=lambda row: (order[row["variant"]], row["split"] != "val",
                                  row.get("seed", 0), row["timepoint"]))
    output.mkdir(parents=True, exist_ok=True)
    value = {"schema": "vq_finetune_ablation_report_v1", "test_visible": reveal_test,
        "suite_complete": all_complete, "ranking": {cohort: [r["variant"] for r in rows] for cohort, rows in rankings.items()},
        "timepoint_rows": len(timepoints), "applicable_timepoint_rows": sum(r["applicable"] for r in timepoints),
        "notes": ["No trained baseline: no net improvement claim against historical frozen runs.",
                  "Seed variation includes patient partition and training randomness.",
                  "Test independence applies to downstream training only; preprocessing and codec are historical.",
                  "Raw latent L1 is not directly comparable across EMA targets.",
                  "Main rankings are separate for all_observed and T0-only selection cohorts.",
                  "T0 comparison retains each model's original selected checkpoint; longitudinal models were selected on all prefixes."],
        "overall": overall, "timepoints": timepoints, "seed_summary": summaries, "t0_comparison": t0_comparison}
    write_json(output / "report.json", value)
    _table_csv(output / "per_config_seed_timepoint.csv", timepoints)
    _table_csv(output / "applicable_timepoints.csv", [row for row in timepoints if row["applicable"]])
    _table_csv(output / "overall.csv", overall)
    _table_csv(output / "seed_summary.csv", summaries)
    _table_csv(output / "t0_comparison.csv", t0_comparison)
    def table(rows, columns=None):
        columns = columns or ["variant", "seed", "selection_cohort", "split", "timepoint", "status", "checkpoint", "epoch", "world_diagnostics",
                              *METRICS, *COUNTS,
                              *[f"{field}_{stat}" for field in DIAGNOSTICS for stat in ("mean", "std")]]
        return "<table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in columns) + \
            "</tr></thead><tbody>" + "".join("<tr>" + "".join("<td>" + html.escape(
                "" if row.get(c) is None else str(row[c])) + "</td>" for c in columns) + "</tr>" for row in rows) + "</tbody></table>"
    artifact_rows = []
    for job in sorted(jobs, key=lambda j: (order[j["variant"]], j["seed"])):
        run = Path(job["output"])
        candidates = [("训练记录", run / "train_report.json"),
                      ("训练诊断", run / "diagnostics.json"),
                      ("诊断校验记录", run / "diagnostics_record.json")]
        for split, label in (("val", "验证"), ("test", "测试")):
            if split == "test" and not reveal_test:
                continue
            candidates.extend([
                (f"{label} best 逐例结果", Path(job.get("evaluations", {}).get(split,
                     run / f"evaluation_best_{split}.json"))),
                (f"{label} last 逐例结果", run / f"evaluation_last_{split}.json"),
                (f"{label} world best", run / f"world_best_{split}.json"),
                (f"{label} world last", run / f"world_last_{split}.json")])
        links = [f'<a href="{html.escape(os.path.relpath(path.resolve(), output.resolve()), quote=True)}">'
                 f'{html.escape(label)}</a>' for label, path in candidates if path.is_file()]
        if links:
            artifact_rows.append("<li>" + html.escape(f"{job['variant']} / seed {job['seed']}") +
                                 "：" + " · ".join(links) + "</li>")
    artifacts = "<h2>各实验诊断与逐例结果</h2><ul>" + "".join(artifact_rows) + "</ul>"
    document = "<!doctype html><html lang='zh'><meta charset='utf-8'><title>解冻优化实验</title>" \
        "<style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse;margin:16px 0}th,td{border:1px solid #ddd;padding:6px}th{position:sticky;top:0;background:#eee}</style>" \
        "<h1>large / all：9配置 × 2种子</h1><p>主排序按输入/选模范围分为all_observed与T0两组，组内按验证F1的两种子均值排序。测试仅在18次训练全部完成后展示。无独立解冻基线。权重评估后删除，预测及指标保留。</p>" \
        "<p>主表144行，其中120行适用，24行T0模型的T1–T3明确标为not_applicable；T0-only模型始终只输入T0。direct模型无world预测分支，world诊断标为not_applicable。</p>" \
        "<p>测试独立性限于本轮下游训练；影像标准化与codec沿用历史来源。种子同时影响划分和训练。</p>" \
        "<h2>两种子均值及样本标准差</h2>" + table(summaries,
            ["variant", "selection_cohort", "split", "timepoint", "status", "completed_seeds", *[f"{m}_{stat}" for m in METRICS for stat in ("mean", "std")]]) + \
        "<h2>T0比较</h2><p>沿用各配置原选模checkpoint；纵向配置按整体选模，T0配置按T0选模。</p>" + table(t0_comparison,
            ["variant", "selection_cohort", "split", "completed_seeds", *[f"{m}_{stat}" for m in METRICS for stat in ("mean", "std")]]) + \
        "<h2>整体</h2>" + table(overall) + "<h2>T0–T3</h2>" + table(timepoints) + artifacts + "</html>"
    (output / "report.html").write_text(document, encoding="utf-8")
    for name in ("report.json", "report.html"):
        (output / name).chmod(0o600)
    return value


def _contained(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if path == parent or parent not in path.parents:
        raise ValueError(f"Artifact is not strictly inside allowed directory: {path}")
    return path


def _checkpoint_files(run):
    return sorted(path for path in Path(run).rglob("*") if path.is_file() and
                  any(path.name.endswith(suffix) for suffix in (".pt", ".ckpt", ".pt.tmp", ".ckpt.tmp")))


def complete_evaluation_receipt(run_dir, suite_root, required_outputs, *, checkpoints=None,
                                manifest=None, status="evaluation_complete"):
    """Verify required outputs, then durably receipt their and checkpoint identities.

    required_outputs: {label: {path, kind='classification'|'world'|'json', split?}}.
    A classification val AND test report are mandatory. Pass every best/last
    and world output required by the runner. No checkpoint is removed here.
    """
    if status not in {"evaluation_complete", "completed"}:
        raise ValueError("Active or pending runs cannot be receipted")
    if manifest is None:
        raise ValueError("Manifest is required to verify complete patient/timepoint coverage")
    run = _contained(run_dir, suite_root)
    if not required_outputs or not isinstance(required_outputs, dict):
        raise ValueError("Required evaluation outputs must be explicitly supplied")
    checked, classified = {}, set()
    for label, specification in required_outputs.items():
        spec = {"path": specification, "kind": "json"} if isinstance(specification, (str, Path)) else dict(specification)
        path = _contained(spec["path"], run)
        kind = spec.get("kind", "json")
        if kind not in {"classification", "world", "json"}:
            raise ValueError("Unsupported evaluation output kind")
        if kind == "classification":
            value = validate_classification_report(path, manifest=manifest, split=spec.get("split"))
            classified.add(value["split"])
        else:
            value = read_json(path)
            if not isinstance(value, dict) or not value:
                raise ValueError("Empty or invalid evaluation JSON")
            if spec.get("split") is not None and value.get("split") != spec["split"]:
                raise ValueError("Evaluation split mismatch")
            if kind == "world":
                if value.get("split") not in {"val", "test"}:
                    raise ValueError("World evaluation requires validation/test split")
                if spec.get("applicable", True) is False:
                    if (value.get("status") != "not_applicable" or value.get("world_head_trained") is not False
                            or not isinstance(value.get("reason"), str) or not value["reason"].strip()):
                        raise ValueError("Disabled-world evaluation requires an explicit not_applicable record")
                else:
                    if value.get("status") != "evaluated":
                        raise ValueError("World evaluation is not complete")
                    expected_rows = _expected_rows(manifest, value.get("split"), value.get("landmarks", "all_observed"))
                    if (value.get("legal_prefixes") != len(expected_rows) or
                            value.get("patients") != len({key for key, _ in expected_rows})):
                        raise ValueError("World evaluation coverage differs from manifest")
        # The durable receipt authorizes irreversible weight deletion, so commit
        # every successfully re-read prediction/diagnostic file first.
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        checked[label] = {**spec, **_identity(path)}
    if classified != {"val", "test"}:
        raise ValueError("Both validation and test classification outputs are required")
    candidates = _checkpoint_files(run) if checkpoints is None else [Path(p) for p in checkpoints]
    identities = []
    for path in candidates:
        if path.is_symlink() or path.name == "codec.pt":
            raise ValueError("Cannot receipt a symlink or original codec as generated weight")
        path = _contained(path, run)
        if path not in _checkpoint_files(run):
            raise ValueError("Only generated checkpoint files can be receipted")
        identities.append(_identity(path))
    receipt = {"schema": "verified_evaluation_receipt_v1", "status": "evaluation_complete",
        "run": str(run), "suite_root": str(Path(suite_root).resolve()),
        "manifest": _identity(manifest) if manifest else None,
        "outputs": checked, "checkpoints": identities}
    write_json(run / "evaluation_complete.json", receipt)
    with (run / "evaluation_complete.json").open("rb") as handle:
        os.fsync(handle.fileno())
    directory_fd = os.open(run, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return receipt


def delete_evaluated_checkpoints(run_dir, suite_root, *, status="evaluation_complete", protected_paths=()):
    """Revalidate receipt/output identities and remove only receipted run weights."""
    if status not in {"evaluation_complete", "completed"}:
        raise ValueError("Active or pending checkpoint deletion is forbidden")
    run = _contained(run_dir, suite_root)
    receipt_path = run / "evaluation_complete.json"
    receipt = read_json(receipt_path)
    if (receipt.get("schema") != "verified_evaluation_receipt_v1" or
            receipt.get("status") != "evaluation_complete" or receipt.get("run") != str(run) or
            receipt.get("suite_root") != str(Path(suite_root).resolve())):
        raise ValueError("Invalid evaluation receipt")
    if receipt.get("manifest") and _identity(receipt["manifest"]["path"]) != receipt["manifest"]:
        raise ValueError("Receipted manifest changed")
    for output in receipt["outputs"].values():
        if _identity(_contained(output["path"], run)) != {k: output[k] for k in ("path", "size", "sha256")}:
            raise ValueError("Receipted output changed")
        if output.get("kind") == "classification":
            validate_classification_report(output["path"], manifest=receipt["manifest"]["path"] if receipt.get("manifest") else None,
                                           split=output.get("split"))
        else:
            read_json(output["path"])
    protected = {Path(p).resolve() for p in protected_paths}
    expected = {Path(item["path"]): item for item in receipt["checkpoints"]}
    current = _checkpoint_files(run)
    if any(p.resolve() not in expected for p in current):
        raise ValueError("Unreceipted checkpoint appeared after evaluation")
    # Validate every candidate before deleting any file.
    for path in current:
        if path.is_symlink() or path.resolve() in protected or path.name == "codec.pt":
            raise ValueError("Protected checkpoint cannot be removed")
        if _identity(_contained(path, run)) != expected[path.resolve()]:
            raise ValueError("Receipted checkpoint changed")
    deleted = []
    for path in current:
        path.unlink()
        deleted.append(expected[path.resolve()])
    record = {"schema": "evaluated_weight_cleanup_v1", "status": "weights_deleted",
              "receipt_sha256": digest(receipt_path), "deleted": deleted,
              "all_receipted_weights_absent": all(not path.exists() for path in expected)}
    # Preserve the original deletion inventory across idempotent retries.
    cleanup_path = run / "checkpoint_cleanup.json"
    if cleanup_path.exists():
        prior = read_json(cleanup_path)
        record["deleted"] = list({item["path"]: item for item in prior.get("deleted", []) + deleted}.values())
    write_json(cleanup_path, record)
    return record
