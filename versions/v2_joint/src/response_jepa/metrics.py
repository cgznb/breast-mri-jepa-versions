"""Patient-equal evaluation, horizon-specific copy controls, paired bootstraps."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def classification(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if not len(y):
        return {"n": 0, "positives": 0, "auroc": None, "auprc": None, "nll": None, "brier": None}
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid labels/probabilities")
    q = np.clip(p, 1e-12, 1-1e-12)
    both = len(np.unique(y)) == 2
    return {"n": int(len(y)), "positives": int(y.sum()),
            "auroc": float(roc_auc_score(y, p)) if both else None,
            "auprc": float(average_precision_score(y, p)) if both else None,
            "nll": float(-(y*np.log(q) + (1-y)*np.log1p(-q)).mean()),
            "brier": float(((p-y)**2).mean())}


def task_summary(rows):
    result = {}
    labelled = [r for r in rows if r["label"] is not None]
    for name in ("direct", "forecast", "fused"):
        selected = [r for r in labelled if r.get(name) is not None]
        if not selected:
            result[name] = None
            continue
        stages = {}
        for stage in range(4):
            sr = [r for r in selected if r["landmark"] == stage]
            stages[f"T{stage}"] = classification([r["label"] for r in sr], [r[name] for r in sr])
        patients = defaultdict(list)
        for r in selected:
            patients[r["patient_key"]].append(r)
        ps = []
        for items in patients.values():
            if len({r["label"] for r in items}) != 1:
                raise ValueError("Patient labels disagree across prefixes")
            ps.append(classification([r["label"] for r in items], [r[name] for r in items]))
        result[name] = {"per_stage": stages, "patient_equal": {
            "patients": len(ps), "prefixes": len(selected),
            "nll": float(np.mean([x["nll"] for x in ps])),
            "brier": float(np.mean([x["brier"] for x in ps]))}}
    return result


def _world_group(rows):
    patients = defaultdict(list)
    for r in rows:
        patients[r["patient_key"]].append(r)
    if not patients:
        return {"patients": 0, "targets": 0, "free_l1": None, "copy_l1": None, "ratio": None, "skill": None}
    free = float(np.mean([np.mean([r["free_l1"] for r in rs]) for rs in patients.values()]))
    copy = float(np.mean([np.mean([r["copy_l1"] for r in rs]) for rs in patients.values()]))
    ratio = free / copy if copy > 1e-8 else None
    return {"patients": len(patients), "targets": len(rows), "free_l1": free, "copy_l1": copy,
            "ratio": ratio, "skill": 1-ratio if ratio is not None else None,
            "near_zero_copy_patients": sum(np.mean([r["copy_l1"] for r in rs]) <= 1e-8
                                           for rs in patients.values())}


def world_summary(rows, tf_rows=(), *, target_space="fixed Stage-A EMA encoder in B/C; not MRI pixel error"):
    result = {"all": _world_group(rows),
              "horizons": {str(h): _world_group([r for r in rows if r["horizon"] == h]) for h in (1, 2, 3)},
              "complete_chain": _world_group([r for r in rows if r["complete_chain"]]),
              "teacher_forcing": _world_group(tf_rows),
              "reduction": "mean targets/prefixes within patient, then mean patients; ratio of means",
              "horizon_basis": "target stage minus last real observed MRI stage",
              "target_space": target_space}
    return result


def paired_bootstrap(rows_a, rows_b, *, stage_a=0, stage_b=0, branch_a="fused", branch_b="fused",
                     repetitions=2000, seed=20261008):
    """Paired common-patient delta = B-A. Never bootstrap individual visits independently."""
    def index(rows, stage, branch):
        out = {}
        for r in rows:
            if int(r["landmark"]) != stage or r.get("label") in {None, ""} or r.get(branch) in {None, ""}:
                continue
            k = r["patient_key"]
            if k in out:
                raise ValueError("Duplicate patient-stage row; select one seed/fold model per patient first")
            out[k] = (float(r["label"]), float(r[branch]))
        return out
    a, b = index(rows_a, stage_a, branch_a), index(rows_b, stage_b, branch_b)
    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        raise ValueError("Fewer than two common labelled patients")
    if any(a[k][0] != b[k][0] for k in keys):
        raise ValueError("Labels disagree across model reports")
    y = np.array([a[k][0] for k in keys])
    pa, pb = np.array([a[k][1] for k in keys]), np.array([b[k][1] for k in keys])
    ma, mb = classification(y, pa), classification(y, pb)
    rng = np.random.default_rng(seed)
    deltas = {k: [] for k in ("auroc", "auprc", "nll", "brier")}
    for _ in range(repetitions):
        ix = rng.integers(0, len(y), len(y))
        aa, bb = classification(y[ix], pa[ix]), classification(y[ix], pb[ix])
        for k in deltas:
            if aa[k] is not None and bb[k] is not None:
                deltas[k].append(bb[k]-aa[k])
    return {"common_patients": len(keys), "available_a": len(a), "available_b": len(b),
            "stage_a": stage_a, "stage_b": stage_b, "model_a": ma, "model_b": mb,
            "delta_definition": "B minus A; positive AUROC/AUPRC or negative NLL/Brier is better",
            "interval_scope": "patient sampling conditional on these fitted models; not training/split uncertainty",
            "deltas": {k: {"point": mb[k]-ma[k] if ma[k] is not None and mb[k] is not None else None,
                           "lower": float(np.quantile(v, .025)) if v else None,
                           "upper": float(np.quantile(v, .975)) if v else None,
                           "valid_repetitions": len(v)} for k, v in deltas.items()}}
