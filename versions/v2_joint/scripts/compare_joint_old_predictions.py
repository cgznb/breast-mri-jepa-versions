#!/usr/bin/env python3
"""CPU-only paired patient bootstrap of audited completed joint versus old runs.

Private prediction rows remain in memory. Output contains aggregate metrics only.
Intervals are conditional on the selected models and the development split.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from analyze_joint_suite import audit_completed, read_predictions
from response_jepa.data import canonical_key


ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = ROOT.parent / "breast-mri-vla-jepa"
METRICS = ("nll", "auroc", "auprc", "brier")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric_vector(y, p):
    """Metrics with exact ties, including bootstrap repetitions of a patient."""
    y, p = np.asarray(y), np.asarray(p)
    q = np.clip(p, 1e-12, 1-1e-12)
    result = {"nll": float(np.mean(-y*np.log(q)-(1-y)*np.log1p(-q))),
              "brier": float(np.mean((y-p)**2))}
    positive, negative = y.sum(), len(y)-y.sum()
    if not positive or not negative:
        return {**result, "auroc": np.nan, "auprc": np.nan}
    order = np.argsort(-p, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(p[order])), len(y)-1]
    tp = np.r_[0., np.cumsum(y[order])[ends]]
    fp = np.r_[0., (ends+1)-tp[1:]]
    result["auroc"] = float(np.trapezoid(tp/positive, fp/negative))
    result["auprc"] = float(np.sum(np.diff(tp)/positive * tp[1:]/(tp[1:]+fp[1:])))
    return result


def paired_rows(new, old, manifest, profile):
    expected = {}
    for patient in manifest["patients"]:
        if patient["split"] == "val":
            stages = [0] if profile == "t0" else sorted({
                v["available_at"] for v in patient["visits"] if v is not None})
            for stage in stages:
                expected[canonical_key(patient["patient_key"]), stage] = float(patient["target"]["pcr"])
    result = []
    for rows, label, probability in ((new, "label", "fused"), (old, "pcr", "pcr_probability")):
        index = {(canonical_key(r["patient_key"]), int(r["landmark"])): r for r in rows}
        if len(index) != len(rows) or set(index) != set(expected):
            raise ValueError("Prediction patient-prefix cohort differs from manifest")
        if any(float(index[key][label]) != y for key, y in expected.items()):
            raise ValueError("Prediction labels differ from manifest")
        probabilities = {key: float(r[probability]) for key, r in index.items()}
        if any(not np.isfinite(p) or p < 0 or p > 1 for p in probabilities.values()):
            raise ValueError("Invalid prediction probability")
        result.append(probabilities)
    return expected, *result


def stage_deltas(labels, new, old, draws):
    y = np.asarray(labels)
    nm, om = metric_vector(y, new), metric_vector(y, old)
    samples = defaultdict(list)
    for draw in draws:
        a, b = metric_vector(y[draw], new[draw]), metric_vector(y[draw], old[draw])
        for metric in METRICS:
            samples[metric].append(a[metric]-b[metric])
    return [{"metric": m, "joint": nm[m], "old": om[m], "delta_joint_minus_old": nm[m]-om[m],
             "lower_95": float(np.nanquantile(samples[m], .025)),
             "upper_95": float(np.nanquantile(samples[m], .975)),
             "valid_replicates": int(np.isfinite(samples[m]).sum())} for m in METRICS]


def compare(run, old_root, repetitions, rng):
    directory = Path(run["output"])
    profile, seed = run["profile"], run["seed"]
    audited, identity = audit_completed(directory, profile, seed, Path(run["config"]))
    relative = (f"runs/registered_roi32_a1_a2_20261004/a1_t0_seed{seed}" if profile == "t0"
                else f"runs/registered_roi32_l1_l4_20261004/l1_dynamic_seed{seed}")
    baseline = old_root / relative
    report, saved = read(baseline/"train_report.json"), read(baseline/"best_validation.json")
    if not report["completed"] or read(baseline/"config.json")["seed"] != seed:
        raise ValueError("Old baseline is incomplete or seed mismatched")
    manifest = read(old_root/"data/registered_roi32/manifest.json")
    if identity["data"]["manifest"] != manifest:
        raise ValueError("Joint and old manifest differ")
    nn, on = identity["normalization"], read(baseline/"normalization.json")
    if (nn["features"] != on["clinical_features"] or nn["fit_split"] != on["fit_split"]
            or not np.allclose(nn["mean"], on["clinical_mean"], rtol=1e-7, atol=1e-6)
            or not np.allclose(nn["std"], on["clinical_std"], rtol=1e-7, atol=1e-6)
            or identity["data"]["image_normalization"] != on["image_normalization"]):
        raise ValueError("Normalization differs beyond FP32 rounding")
    labels, new, old = paired_rows(read_predictions(directory/"predictions_val.csv"), saved["rows"], manifest, profile)
    stages, rows = ([0] if profile == "t0" else range(4)), []
    for stage in stages:
        keys = sorted(key for key in labels if key[1] == stage)
        y = np.array([labels[k] for k in keys])
        a, b = np.array([new[k] for k in keys]), np.array([old[k] for k in keys])
        draws = rng.integers(0, len(keys), size=(repetitions, len(keys)))
        for item in stage_deltas(y, a, b, draws):
            rows.append({"scope": f"T{stage}", "patients": len(keys), "positives": int(y.sum()), **item})
    patients = sorted({key[0] for key in labels})
    values = {name: [] for name in ("new_nll", "old_nll", "new_brier", "old_brier")}
    for patient in patients:
        keys = [key for key in labels if key[0] == patient]
        y = [labels[k] for k in keys]
        for tag, source in (("new", new), ("old", old)):
            metrics = metric_vector(y, [source[k] for k in keys])
            for metric in ("nll", "brier"):
                values[f"{tag}_{metric}"].append(metrics[metric])
    draws = rng.integers(0, len(patients), size=(repetitions, len(patients)))
    for metric in ("nll", "brier"):
        a, b = np.array(values[f"new_{metric}"]), np.array(values[f"old_{metric}"])
        estimates = (a-b)[draws].mean(axis=1)
        lower, upper = np.quantile(estimates, [.025, .975])
        rows.append({"scope": "patient_equal_prefix", "patients": len(patients),
                     "positives": int(sum(labels[p, 0] for p in patients)), "metric": metric,
                     "joint": float(a.mean()), "old": float(b.mean()),
                     "delta_joint_minus_old": float((a-b).mean()), "lower_95": float(lower),
                     "upper_95": float(upper), "valid_replicates": repetitions})
    return {"name": run["name"], "profile": profile, "seed": seed, "joint_best_epoch": audited["best_epoch"],
            "old_experiment": "A1" if profile == "t0" else "L1", "old_run": str(baseline),
            "checks": {name: True for name in ("joint_completed_audited", "same_seed", "same_manifest", "same_labels",
                                               "clinical_norm_equal_with_fp32_rounding", "image_normalization_equal")},
            "metrics": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, default=OLD_ROOT)
    parser.add_argument("--repetitions", type=int, default=5000)
    args = parser.parse_args(argv)
    if args.repetitions < 100:
        parser.error("Use at least 100 bootstrap repetitions")
    status = read(args.suite_root/"status.json")
    compared, excluded, errors = [], [], []
    for run in status["runs"]:
        if run["state"] != "completed" or run["seed"] not in (17, 43):
            excluded.append({"name": run["name"], "reason": "no matched old seed" if run["seed"] == 71 else "joint unfinished"})
            continue
        try:
            compared.append(compare(run, args.old_root, args.repetitions, np.random.default_rng(20261009)))
        except Exception as exc:
            errors.append({"name": run["name"], "error": str(exc)})
    result = {"schema": "joint_old_paired_patient_bootstrap_v1", "repetitions": args.repetitions,
              "complete_matched_four": len(compared) == 4 and not errors,
              "scope": "Selected fused joint versus selected direct old A1 fixedT0/L1 dynamic; development validation.",
              "interval_limit": "Patient resampling conditional on chosen checkpoints and split; excludes seed, split, model/epoch selection uncertainty and multiplicity correction.",
              "metric_direction": "joint-old: lower NLL/Brier and higher AUROC/AUPRC favor joint.",
              "comparisons": compared, "excluded": excluded, "errors": errors}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"aggregate.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    flat = [{"run": r["name"], "profile": r["profile"], "seed": r["seed"], "old_experiment": r["old_experiment"], **m}
            for r in compared for m in r["metrics"]]
    if flat:
        with (args.output/"paired_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    print(json.dumps({"compared_runs": len(compared), "excluded_runs": len(excluded), "errors": errors}))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
