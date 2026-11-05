"""Patient-aware classification and probabilistic diagnostics, without efficacy claims."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score


def classification_metrics(y,p, *, f1_threshold=0.5, sample_weight=None):
    y,p = np.asarray(y),np.asarray(p,dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("Metrics need matched nonempty vectors")
    if not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError("Invalid labels/probabilities")
    if (isinstance(f1_threshold, bool) or not isinstance(f1_threshold, (int, float))
            or not np.isfinite(f1_threshold) or not 0 <= f1_threshold <= 1):
        raise ValueError("f1_threshold must be finite and between zero and one")
    weights = np.ones(len(y), dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != y.shape or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Metrics need matched nonnegative sample weights with positive total")
    prediction = p >= f1_threshold
    tp = float(weights[(y == 1) & prediction].sum())
    fp = float(weights[(y == 0) & prediction].sum())
    fn = float(weights[(y == 1) & ~prediction].sum())
    denominator = 2 * tp + fp + fn
    clipped = np.clip(p,1e-12,1-1e-12)
    nll = float(np.average(-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)), weights=weights))
    ece,bins = 0.,[]
    edges = np.linspace(0,1,11)
    for index in range(10):
        mask = (p>=edges[index]) & ((p<=edges[index+1]) if index == 9 else (p<edges[index+1]))
        if mask.any() and weights[mask].sum() > 0:
            acc,conf = float(np.average(y[mask], weights=weights[mask])),float(np.average(p[mask], weights=weights[mask]))
            ece += weights[mask].sum()/weights.sum()*abs(acc-conf)
            bins.append({"count":int(mask.sum()),"mean_probability":conf,"event_rate":acc})
    both = len(np.unique(y[weights > 0])) == 2
    return {"n":len(y),"positives":int(y.sum()),"auroc":float(roc_auc_score(y,p,sample_weight=weights)) if both else None,
            "auprc":float(average_precision_score(y,p,sample_weight=weights)) if weights[y == 1].sum() else None,
            "brier":float(np.average((p-y)**2, weights=weights)),"nll":nll,
            "f1":float(2 * tp / denominator) if denominator else 0.0,
            "f1_threshold":float(f1_threshold),"tp":tp,"fp":fp,"fn":fn,
            "ece_10_equal_width":float(ece),"reliability_bins":bins}


def patient_bootstrap(y,p,patient_ids,repetitions=1000,seed=0, *, f1_threshold=0.5):
    """Cluster bootstrap. Multiple landmarks from a patient stay together."""
    y,p,ids = np.asarray(y),np.asarray(p),np.asarray(patient_ids)
    unique = np.unique(ids)
    rng = np.random.default_rng(seed)
    groups = {i:np.where(ids == i)[0] for i in unique}
    values = {k:[] for k in ("auroc","auprc","brier","nll","f1")}
    for _ in range(repetitions):
        chosen = rng.choice(unique,len(unique),replace=True)
        idx = np.concatenate([groups[i] for i in chosen])
        result = classification_metrics(y[idx],p[idx], f1_threshold=f1_threshold)
        for k in values:
            if result[k] is not None:
                values[k].append(result[k])
    return {k:{"lower":float(np.quantile(v,.025)),"upper":float(np.quantile(v,.975)),"valid_replicates":len(v)}
            if v else {"lower":None,"upper":None,"valid_replicates":0} for k,v in values.items()}
