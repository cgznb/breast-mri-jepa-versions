#!/usr/bin/env python3
"""Reconstruct the v8 point metrics from this bundle's copied reports.

No original project paths, data manifests, MRI arrays, checkpoints, or GPU are accessed.
The original manuscript files remain unchanged. Patient identifiers and
probabilities are held in memory; outputs contain aggregate metrics only.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, roc_auc_score, roc_curve,
)

SEEDS = [42, 44, 71, 2026]
MODELS = ["direct_no_jepa", "direct_readout", "forecast_mean"]
LABELS = ["NoJEPA", "DirectReadout", "ForecastMean"]
TIMES = ["T0", "T1", "T2", "Macro"]
POLICIES = ["fixed_0.5", "oof_validation_youden"]
METRICS = [
    "auroc", "auprc", "f1", "accuracy", "sensitivity", "specificity",
    "ppv", "npv", "balanced_accuracy", "mcc", "youden_j", "brier", "nll",
    "ece_10_equal_width", "tp", "fp", "fn", "tn",
]


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_rows(report, landmark):
    rows = sorted(
        [row for row in report["rows"] if row["landmark"] == landmark],
        key=lambda row: row["patient_key"],
    )
    identities = [row["patient_key"] for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate patient in one report/landmark")
    labels = np.array([row["pcr"] for row in rows], dtype=int)
    probabilities = np.array([row["pcr_probability"] for row in rows], dtype=float)
    if not (np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()):
        raise ValueError("Invalid report probability")
    return identities, labels, probabilities


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def compute_metrics(labels, probabilities, threshold):
    """Use sklearn point estimators independently from the manuscript script."""
    prediction = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    bins = np.minimum((probabilities * 10).astype(int), 9)
    ece = sum(
        abs(float((probabilities[bins == index] - labels[bins == index]).sum()))
        for index in range(10)
    ) / len(labels)
    values = [
        roc_auc_score(labels, probabilities),
        average_precision_score(labels, probabilities),
        f1_score(labels, prediction, zero_division=0),
        float((prediction == labels).mean()),
        sensitivity, specificity, safe_divide(tp, tp + fp), safe_divide(tn, tn + fn),
        (sensitivity + specificity) / 2,
        matthews_corrcoef(labels, prediction), sensitivity + specificity - 1,
        float(((probabilities - labels) ** 2).mean()),
        float(-(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)).mean()),
        ece, int(tp), int(fp), int(fn), int(tn),
    ]
    return np.asarray(values, dtype=float)


def select_validation_threshold(labels, probabilities):
    fpr, tpr, thresholds = roc_curve(labels, probabilities, drop_intermediate=False)
    youden = tpr - fpr
    # roc_curve returns descending thresholds. Highest threshold wins ties.
    index = int(np.flatnonzero(np.isclose(youden, youden.max(), atol=1e-14, rtol=0))[0])
    threshold = float(thresholds[index])
    if not np.isfinite(threshold):
        threshold = float(np.nextafter(probabilities.max(), np.inf))
    return threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Default: BUNDLE/outputs/metric_reproduction")
    args = parser.parse_args()
    root = args.bundle_root.resolve()
    paper = root / "manuscript"
    out = args.output.resolve() if args.output else root / "outputs/metric_reproduction"
    outputs_root = (root / "outputs").resolve()
    if not outputs_root.is_relative_to(root) or not out.is_relative_to(outputs_root):
        parser.error("--output must resolve inside BUNDLE/outputs; source directories are protected")
    out.mkdir(parents=True, exist_ok=True)
    provenance = read_json(paper / "data/extended_provenance.json")
    sources = {(row["seed"], row["model"], row["fold"]): row for row in provenance["sources"]}
    if set(sources) != {(seed, label, fold) for seed in SEEDS for label in LABELS for fold in range(1, 6)}:
        raise ValueError("Manuscript provenance is not the expected 60 selected runs")
    probabilities = np.empty((4, 3, 5, 3, 175))
    thresholds = np.full((2, 4, 3, 3), 0.5)
    reference_ids = reference_labels = None
    reference_validation = [None, None, None]
    source_hash_checks = 0
    test_alignment_checks = validation_exclusion_checks = oof_alignment_checks = 0
    for si, seed in enumerate(SEEDS):
        matrix = read_json(root / f"experiments/seed{seed}/matrix.json")
        if not (matrix["status"] == "completed" and matrix["test_visible"]):
            raise ValueError("Copied experiment suite is not completed/test-visible")
        for mi, (variant, label) in enumerate(zip(MODELS, LABELS)):
            validation = [[], [], []]
            for fi in range(5):
                fold = fi + 1
                run = root / f"experiments/seed{seed}/runs/{variant}_seed{seed}_fold{fold}"
                source = sources[seed, label, fold]
                receipt = read_json(run / "evaluation_done.json")
                if receipt["completed"] is not True:
                    raise ValueError("Copied evaluation receipt is incomplete")
                config = read_json(run / "config.json")
                if config["seed"] != seed:
                    raise ValueError("Run configuration seed mismatch")
                for name, expected in [
                    ("evaluation_done.json", source["receipt_sha256"]),
                    ("config.json", source["config_sha256"]),
                    ("evaluation_best_test.json", source["test_sha256"]),
                    ("evaluation_best_val.json", source["validation_sha256"]),
                ]:
                    if sha256(run / name) != expected:
                        raise ValueError(f"Copied source hash mismatch: seed{seed}/{variant}/fold{fold}/{name}")
                    source_hash_checks += 1
                for registered in source["registered_reports"]:
                    name = Path(registered["path"]).name
                    if sha256(run / name) != registered["sha256"]:
                        raise ValueError("Copied registered report hash mismatch")
                    source_hash_checks += 1
                test = read_json(run / "evaluation_best_test.json")
                val = read_json(run / "evaluation_best_val.json")
                if not (test["split"] == "test" and test["independent_test"] is False and test["downstream_holdout"]):
                    raise ValueError("Wrong test report estimand")
                if val["split"] != "val":
                    raise ValueError("Wrong validation report split")
                for ti in range(3):
                    identities, labels, scores = report_rows(test, ti)
                    if len(identities) != 175 or labels.sum() != 56:
                        raise ValueError("Unexpected holdout denominator")
                    if reference_ids is None:
                        reference_ids, reference_labels = identities, labels
                    if identities != reference_ids or not np.array_equal(labels, reference_labels):
                        raise ValueError("Patient/label alignment differs across selected reports")
                    test_alignment_checks += 1
                    probabilities[si, mi, fi, ti] = scores
                    v_ids, v_labels, v_scores = report_rows(val, ti)
                    if set(v_ids) & set(identities):
                        raise ValueError("Validation/test patients overlap")
                    validation_exclusion_checks += 1
                    validation[ti].extend(zip(v_ids, v_labels.tolist(), v_scores.tolist()))
            for ti in range(3):
                rows = sorted(validation[ti])
                identities, labels, scores = zip(*rows)
                if len(identities) != len(set(identities)) or len(identities) != [801, 760, 700][ti]:
                    raise ValueError("Wrong OOF validation denominator or repeated patient")
                if reference_validation[ti] is None:
                    reference_validation[ti] = (identities, labels)
                if (identities, labels) != reference_validation[ti]:
                    raise ValueError("OOF patient/label vectors differ across models/seeds")
                oof_alignment_checks += 1
                thresholds[1, si, mi, ti] = select_validation_threshold(np.array(labels), np.array(scores))
    ensemble = probabilities.mean(axis=2)
    point = np.empty((2, 4, 3, 3, 18))
    for pi in range(2):
        for si in range(4):
            for mi in range(3):
                for ti in range(3):
                    point[pi, si, mi, ti] = compute_metrics(reference_labels, ensemble[si, mi, ti], thresholds[pi, si, mi, ti])
    per_seed = np.concatenate([point, point.mean(axis=3, keepdims=True)], axis=3)
    mean, sd = per_seed.mean(axis=1), per_seed.std(axis=1, ddof=1)
    original_individual = {(r["policy"], int(r["seed"]), r["model"], r["timepoint"]): r
                           for r in csv.DictReader((paper / "data/four_seed_individual_metrics.csv").open())}
    original_summary = {(r["policy"], r["model"], r["timepoint"]): r
                        for r in csv.DictReader((paper / "data/extended_summary.csv").open())}
    individual_rows, summary_rows = [], []
    comparisons = 0
    maximum_error = 0.0
    for pi, policy in enumerate(POLICIES):
        for mi, label in enumerate(LABELS):
            for ti, time in enumerate(TIMES):
                for si, seed in enumerate(SEEDS):
                    row = dict(policy=policy, seed=seed, model=label, timepoint=time,
                               patients=175, positives=56, folds=5,
                               threshold=float(thresholds[pi, si, mi, ti]) if ti < 3 else float("nan"))
                    row.update(zip(METRICS, map(float, per_seed[pi, si, mi, ti])))
                    individual_rows.append(row)
                    expected = original_individual[policy, seed, label, time]
                    actual = per_seed[pi, si, mi, ti]
                    target = np.array([float(expected[key]) for key in METRICS])
                    np.testing.assert_allclose(actual, target, atol=1e-12, rtol=0)
                    maximum_error = max(maximum_error, float(np.max(np.abs(actual - target))))
                    comparisons += len(METRICS)
                    if ti < 3:
                        np.testing.assert_allclose(row["threshold"], float(expected["threshold"]), atol=1e-12, rtol=0)
                        comparisons += 1
                row = dict(policy=policy, seed=",".join(map(str, SEEDS)), model=label, timepoint=time,
                           patients=175, positives=56, folds_per_seed=5, training_seed_count=4)
                expected = original_summary[policy, label, time]
                for ki, key in enumerate(METRICS):
                    row[key] = float(mean[pi, mi, ti, ki])
                    row[key + "_seed_sd"] = float(sd[pi, mi, ti, ki])
                    actual = [row[key], row[key + "_seed_sd"]]
                    target = [float(expected[key]), float(expected[key + "_seed_sd"])]
                    np.testing.assert_allclose(actual, target, atol=1e-12, rtol=0)
                    maximum_error = max(maximum_error, float(np.max(np.abs(np.asarray(actual) - target))))
                    comparisons += 2
                summary_rows.append(row)
    write_csv(out / "four_seed_individual_metrics_recomputed.csv", individual_rows)
    write_csv(out / "extended_point_summary_recomputed.csv", summary_rows)
    verification = dict(
        completed=True, checked_at_utc=datetime.now(timezone.utc).isoformat(),
        source_runs=60, registered_reports=180, copied_source_sha256_checks=source_hash_checks,
        test_patient_label_alignment_checks=test_alignment_checks,
        validation_test_exclusion_checks=validation_exclusion_checks,
        oof_unique_patient_label_alignment_checks=oof_alignment_checks,
        patients=175, positives=56, policies=POLICIES, numeric_comparisons=comparisons,
        maximum_absolute_error=maximum_error, absolute_tolerance=1e-12,
        aggregation="within-seed fivefold probability mean -> metric; Macro mean of T0/T1/T2; four-seed metric mean and sample SD (ddof=1)",
        source_access="bundle-local copied result JSON/CSV only; no source-project paths, manifests, MRI/training inputs, checkpoint loads or GPU",
        patient_level_outputs_written=False,
        scope="point metrics, validation thresholds and seed SD; frozen bootstrap intervals/p-values are included in manuscript/data but not rerun here",
        audit_limit="report-level patient/label alignment and validation/test exclusion only; original training-cohort manifest membership is not re-audited",
    )
    (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
