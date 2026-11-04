#!/usr/bin/env python3
"""Post-hoc paired patient randomization tests of fixed seed44 ensembles.

Only aggregate test statistics are written. Randomization exchanges entire
within-patient trajectories; it does not retrain models or permute outcomes.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import numpy as np

from extended_metrics import LABELS, METRICS, TIMES, all_metrics

ROOT = Path(__file__).resolve().parents[1]
N_PERMUTATIONS = 99_999
RNG_SEED = 20261103
BATCH_SIZE = 1024
PRIMARY = 'principal_macro_9'
SECONDARY = 'secondary_81'
METRIC_LABELS = dict(zip(METRICS[:14], [
    'AUROC', 'AP', 'F1', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV',
    'NPV', 'Balanced accuracy', 'MCC', 'Youden index', 'Brier', 'NLL', 'ECE']))


def holm(p):
    """Holm step-down adjusted p-values, restored to original input order."""
    p = np.asarray(p, dtype=float)
    assert p.ndim == 1 and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty_like(p)
    result[order] = adjusted
    return result


def marker(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'


def batch_metrics(y, p):
    """Metrics for score rows (..., patients), including exact score-tie groups."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    shape, n = p.shape[:-1], p.shape[-1]
    p = p.reshape(-1, n)
    assert n == len(y) and 0 < y.sum() < n
    pos, neg = y.sum(), n-y.sum()
    order = np.argsort(-p, axis=1, kind='stable')
    ps = np.take_along_axis(p, order, axis=1)
    ys = y[order]
    indices = np.arange(n)[None, :]
    group_starts = np.c_[np.ones(len(p), bool), ps[:, 1:] != ps[:, :-1]]
    group_ends = np.c_[ps[:, :-1] != ps[:, 1:], np.ones(len(p), bool)]
    start = np.maximum.accumulate(np.where(group_starts, indices, 0), axis=1)
    end = np.minimum.accumulate(np.where(group_ends, indices, n)[:, ::-1], axis=1)[:, ::-1]
    cumulative = np.cumsum(ys, axis=1)
    tp_end = np.take_along_axis(cumulative, end, axis=1)
    before_start = np.take_along_axis(np.c_[np.zeros(len(p), int), cumulative], start, axis=1)
    auc = ((1-ys)*(tp_end-.5*(tp_end-before_start))).sum(axis=1)/(pos*neg)
    ap = (ys*tp_end/(end+1)).sum(axis=1)/pos
    pred = p >= .5
    tp = (pred*y).sum(axis=1).astype(float)
    fp = (pred*(1-y)).sum(axis=1).astype(float)
    fn, tn = pos-tp, neg-fp
    divide = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    sens, spec = tp/pos, tn/neg
    f1 = divide(2*tp, 2*tp+fp+fn)
    acc = (tp+tn)/n
    ppv, npv = divide(tp, tp+fp), divide(tn, tn+fn)
    mcc = divide(tp*tn-fp*fn, np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    brier = ((p-y)**2).mean(axis=1)
    clipped = np.clip(p, 1e-7, 1-1e-7)
    nll = -(y*np.log(clipped)+(1-y)*np.log(1-clipped)).mean(axis=1)
    bins = np.minimum((p*10).astype(int), 9)
    ece = sum(np.abs(np.where(bins == bi, p-y, 0).sum(axis=1))/n for bi in range(10))
    return np.stack([auc, ap, f1, acc, sens, spec, ppv, npv,
                     (sens+spec)/2, mcc, sens+spec-1, brier, nll, ece], axis=-1).reshape(*shape, 14)


def load_ensembles():
    provenance = json.loads((ROOT/'data/extended_provenance.json').read_text())
    assert provenance['training_seeds'] == [44]
    assert provenance['verified_completed_receipts'] == 20
    probs = np.empty((4, 5, 3, 175))
    ids_ref = y = None
    seen = set()
    for src in provenance['sources']:
        assert src['seed'] == 44
        path = Path(src['test_report'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == src['test_sha256']
        report = json.loads(path.read_text())
        assert report['split'] == 'test' and report['independent_test'] is False
        mi, fi = LABELS.index(src['model']), src['fold']-1
        for ti in range(3):
            rows = sorted((r for r in report['rows'] if r['landmark'] == ti), key=lambda r: r['patient_key'])
            ids = [r['patient_key'] for r in rows]
            yy = np.array([r['pcr'] for r in rows], dtype=int)
            if y is None:
                ids_ref, y = ids, yy
            assert ids == ids_ref and np.array_equal(yy, y)
            probs[mi, fi, ti] = [r['pcr_probability'] for r in rows]
            assert (mi, fi, ti) not in seen
            seen.add((mi, fi, ti))
    assert len(seen) == 60 and len(set(ids_ref)) == 175 and y.sum() == 56
    assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()
    return y, probs.mean(axis=1), provenance


def paired_test(y, a, b, permutations=N_PERMUTATIONS, seed=RNG_SEED):
    """Return observed differences and inclusive two-sided exceedance counts.

    a/b are (landmarks, patients). Each random bit exchanges all landmarks for
    one patient. The final statistic row is the mean of landmark statistics.
    """
    observed = batch_metrics(y, a)-batch_metrics(y, b)
    observed = np.concatenate([observed, observed.mean(axis=0, keepdims=True)])
    counts = np.zeros_like(observed, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for first in range(0, permutations, BATCH_SIZE):
        size = min(BATCH_SIZE, permutations-first)
        swap = rng.integers(0, 2, size=(size, 1, len(y)), dtype=np.uint8).astype(bool)
        ra, rb = np.where(swap, b, a), np.where(swap, a, b)
        delta = batch_metrics(y, ra)-batch_metrics(y, rb)
        delta = np.concatenate([delta, delta.mean(axis=1, keepdims=True)], axis=1)
        counts += (np.abs(delta) >= np.abs(observed)[None]-1e-12).sum(axis=0)
    return observed, counts


def main():
    y, ensemble, source = load_ensembles()
    points = batch_metrics(y, ensemble)
    reference = [r for r in csv.DictReader((ROOT/'data/extended_summary.csv').open())
                 if r['policy'] == 'fixed_0.5' and r['seed'] == '44']
    lookup = {(r['model'], r['timepoint']): r for r in reference}
    for mi, model in enumerate(LABELS):
        for ti, time in enumerate(TIMES[:3]):
            expected = np.array([float(lookup[model, time][k]) for k in METRICS[:14]])
            np.testing.assert_allclose(points[mi, ti], expected, atol=1e-12, rtol=0)
            np.testing.assert_allclose(points[mi, ti], all_metrics(y, ensemble[mi, ti], np.ones((1, len(y))))[0, :14], atol=1e-12, rtol=0)
    ci_rows = [r for r in csv.DictReader((ROOT/'data/extended_paired_bootstrap.csv').open())
               if r['policy'] == 'fixed_0.5' and r['seed'] == '44']
    cis = {(r['comparison'], r['timepoint'], r['metric']): r for r in ci_rows}
    temporal = [r for r in csv.DictReader((ROOT/'data/extended_longitudinal.csv').open())
                if r['policy'] == 'fixed_0.5' and r['seed'] == '44']
    temporal_ci = {(r['model'], r['metric']): r for r in temporal}
    tests = []

    def add(comparison, time, key, observed, count, ci, family, kind):
        assert abs(observed-float(ci['delta'])) < 1e-12
        p = (1+int(count))/(N_PERMUTATIONS+1)
        tests.append(dict(seed=44, policy='fixed_0.5', kind=kind, family=family,
                          comparison=comparison, timepoint=time, metric=key,
                          delta=float(observed), ci_low=float(ci['ci_low']), ci_high=float(ci['ci_high']),
                          p_raw=p, exceedances=int(count), random_permutations=N_PERMUTATIONS,
                          p_monte_carlo_se=float(np.sqrt(p*(1-p)/N_PERMUTATIONS))))

    # These families are fixed before generating any randomization p-values.
    for aa, bb in [(1, 0), (2, 0), (2, 1), (2, 3)]:
        comparison = LABELS[aa]+' minus '+LABELS[bb]
        observed, counts = paired_test(y, ensemble[aa], ensemble[bb])
        for ti, time in enumerate(TIMES):
            keys = METRICS[:14] if bb != 3 and time == 'Macro' else METRICS[:3]
            for ki, key in enumerate(keys):
                family = PRIMARY if bb != 3 and time == 'Macro' and ki < 3 else SECONDARY
                kind = 'pooling' if bb == 3 else 'model'
                add(comparison, time, key, observed[ti, ki], counts[ti, ki],
                    cis[comparison, time, key], family, kind)
        print('Computed paired trajectory test:', comparison, flush=True)
    for mi, model in enumerate(LABELS[:3]):
        observed, counts = paired_test(y, ensemble[mi, 2:3], ensemble[mi, 0:1])
        for ki, key in enumerate(METRICS[:3]):
            add(model+' T2 minus T0', 'T2 minus T0', key, observed[0, ki], counts[0, ki],
                temporal_ci[model, key], SECONDARY, 'longitudinal')
        print('Computed paired landmark test:', model, flush=True)
    assert len(tests) == 90
    for family, size in [(PRIMARY, 9), (SECONDARY, 81)]:
        selected = [r for r in tests if r['family'] == family]
        assert len(selected) == size
        adjusted = holm([r['p_raw'] for r in selected])
        for r, p in zip(selected, adjusted):
            r.update(p_holm_family=float(p), family_size=size,
                     significance=marker(p))
    for r, p in zip(tests, holm([r['p_raw'] for r in tests])):
        r.update(p_holm_all_90=float(p), significance_all_90=marker(p))
    with (ROOT/'data/significance_score_swap_sensitivity.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(tests[0]))
        writer.writeheader(); writer.writerows(tests)
    provenance = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(), training_seeds=[44],
        patients=175, positives=56, folds_per_ensemble=5, threshold=.5,
        random_permutations=N_PERMUTATIONS, rng_seed=RNG_SEED,
        numpy_version=np.__version__, batch_size=BATCH_SIZE,
        test='two-sided patient-paired Monte Carlo randomization',
        model_exchange='One independent fair swap bit per patient exchanges complete T0/T1/T2 probability trajectories; labels remain fixed.',
        longitudinal_exchange='One independent fair swap bit per patient exchanges T0 and T2 scores; labels remain fixed.',
        null='Conditional on outcomes, the two score trajectories (or paired landmark scores) are exchangeable. This is stronger than equality of a single metric.',
        p_formula='(1 + count(abs(permuted_delta) >= abs(observed_delta) - 1e-12)) / (99999 + 1)',
        macro='Arithmetic mean of separately recomputed landmark metric differences, never pooled patient-landmark rows.',
        metric_zero_division='PPV/NPV/MCC/F1 denominator zero is defined as zero in permutation draws; no undefined permutation is dropped.',
        families={PRIMARY: '9: three principal model contrasts x Macro AUROC/AP/F1',
                  SECONDARY: '81: principal landmark metrics 27; other principal Macro metrics 33; pooling 12; principal T2-T0 9'},
        multiplicity='Holm within primary 9 and secondary 81, separately; additional Holm across all 90 exported as sensitivity analysis. Family-wise adjustment alone does not control the union of both families.',
        intervals='Previously generated 2000 shared patient-bootstrap percentile 95% intervals; unadjusted, not inversions of these randomization tests.',
        interpretation='All analyses and families are post-hoc/exploratory. Conditions on fixed selected-seed ensembles; excludes retraining and seed/aggregation/model selection uncertainty.',
        excluded='No new tests of OOF thresholds, subgroups, DCA thresholds, or archived training probes.',
        independent_patients=175, exported_patient_data=False,
        source_provenance_sha256=hashlib.sha256((ROOT/'data/extended_provenance.json').read_bytes()).hexdigest(),
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_count=len(tests),
        family_significant_counts={family: sum(r['p_holm_family'] < .05 for r in tests if r['family'] == family) for family in [PRIMARY, SECONDARY]},
        global_significant_count=sum(r['p_holm_all_90'] < .05 for r in tests),
        fitted_model_source_count=len(source['sources']))
    (ROOT/'data/significance_score_swap_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps({k: provenance[k] for k in ['test_count', 'family_significant_counts', 'global_significant_count']}, indent=2))


if __name__ == '__main__':
    main()
