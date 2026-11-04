#!/usr/bin/env python3
"""Exploratory paired tests of metric differences between fixed seed44 ensembles.

AUROC: paired DeLong, including patient-correlated landmark Macro.
Other metrics: null-centered paired patient bootstrap, an approximate test.
Only aggregate results are exported; no patient rows or prediction arrays.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import numpy as np

from extended_metrics import LABELS, METRICS, TIMES, all_metrics
from significance_score_swap import load_ensembles, holm, marker

ROOT = Path(__file__).resolve().parents[1]
N_BOOTSTRAP = 99_999
RNG_SEED = 20261103
BATCH_SIZE = 1024
PRIMARY = 'principal_macro_9'
SECONDARY = 'secondary_81'


def auc_placements(y, p):
    """Return AUROCs and positive/negative DeLong placement vectors.

    p has arbitrary leading score dimensions and patients on the final axis.
    Ties contribute one half. No covariance independence across scores is used.
    """
    pos, neg = p[..., y == 1], p[..., y == 0]
    pairs = (pos[..., :, None] > neg[..., None, :]).astype(float)
    pairs += .5*(pos[..., :, None] == neg[..., None, :])
    vp, vn = pairs.mean(axis=-1), pairs.mean(axis=-2)
    return vp.mean(axis=-1), vp, vn


def delong_difference(y, a, b):
    """Paired DeLong test; average correlated landmark placements for Macro."""
    auc_a, ap, an = auc_placements(y, np.asarray(a))
    auc_b, bp, bn = auc_placements(y, np.asarray(b))
    delta = float(np.mean(auc_a-auc_b))
    dp, dn = np.mean(ap-bp, axis=0), np.mean(an-bn, axis=0)
    variance = float(np.var(dp, ddof=1)/len(dp)+np.var(dn, ddof=1)/len(dn))
    se = math.sqrt(max(variance, 0.))
    z = delta/se if se else (0. if delta == 0 else math.copysign(float('inf'), delta))
    return dict(delta=delta, standard_error=se, z=z,
                p_raw=math.erfc(abs(z)/math.sqrt(2)))


def hypotheses():
    """Freeze reporting families, without inspecting newly computed p-values."""
    specifications = []
    for aa, bb in [(1, 0), (2, 0), (2, 1), (2, 3)]:
        for ti, time in enumerate(TIMES):
            keys = METRICS[:14] if bb != 3 and time == 'Macro' else METRICS[:3]
            for ki, key in enumerate(keys):
                family = PRIMARY if bb != 3 and time == 'Macro' and ki < 3 else SECONDARY
                specifications.append(dict(aa=aa, bb=bb, ti=ti, ki=ki,
                    comparison=LABELS[aa]+' minus '+LABELS[bb], timepoint=time,
                    metric=key, family=family, kind='pooling' if bb == 3 else 'model'))
    for mi, model in enumerate(LABELS[:3]):
        for ki, key in enumerate(METRICS[:3]):
            specifications.append(dict(aa=mi, bb=mi, ti=2, ki=ki,
                comparison=model+' T2 minus T0', timepoint='T2 minus T0',
                metric=key, family=SECONDARY, kind='longitudinal'))
    assert len(specifications) == 90
    assert sum(s['family'] == PRIMARY for s in specifications) == 9
    return specifications


def metric_difference(values, spec):
    # values shape (..., model, landmark including Macro, metric)
    aa, bb, ti, ki = (spec[k] for k in ['aa', 'bb', 'ti', 'ki'])
    if spec['kind'] == 'longitudinal':
        return values[..., aa, 2, ki]-values[..., aa, 0, ki]
    return values[..., aa, ti, ki]-values[..., bb, ti, ki]


def main():
    y, ensemble, source = load_ensembles()
    specifications = hypotheses()
    point = np.array([[all_metrics(y, p, np.ones((1, len(y))))[0, :14]
                       for p in scores] for scores in ensemble])
    point = np.concatenate([point, point.mean(axis=1, keepdims=True)], axis=1)
    summary = {(r['model'], r['timepoint']): r
               for r in csv.DictReader((ROOT/'data/extended_summary.csv').open())
               if r['policy'] == 'fixed_0.5' and r['seed'] == '44'}
    for mi, model in enumerate(LABELS):
        for ti, time in enumerate(TIMES):
            np.testing.assert_allclose(point[mi, ti], [float(summary[model, time][k]) for k in METRICS[:14]], atol=1e-12, rtol=0)
    observed = np.array([metric_difference(point, s) for s in specifications])
    counts = np.zeros(len(specifications), dtype=np.int64)
    rng = np.random.default_rng(RNG_SEED)
    for first in range(0, N_BOOTSTRAP, BATCH_SIZE):
        size = min(BATCH_SIZE, N_BOOTSTRAP-first)
        weights = rng.multinomial(len(y), np.full(len(y), 1/len(y)), size=size)
        assert ((weights*y).sum(axis=1) > 0).all() and ((weights*(1-y)).sum(axis=1) > 0).all()
        values = np.empty((size, 4, 3, 14))
        for mi in range(4):
            for ti in range(3):
                values[:, mi, ti] = all_metrics(y, ensemble[mi, ti], weights)[:, :14]
        assert np.isfinite(values).all(), 'Do not silently omit undefined bootstrap replicates.'
        values = np.concatenate([values, values.mean(axis=2, keepdims=True)], axis=2)
        deltas = np.stack([metric_difference(values, s) for s in specifications], axis=1)
        counts += (np.abs(deltas-observed) >= np.abs(observed)-1e-12).sum(axis=0)
        if first == 0 or first+size == N_BOOTSTRAP:
            print(f'Computed shared patient bootstrap {first+size}/{N_BOOTSTRAP}', flush=True)
    ci_rows = [r for r in csv.DictReader((ROOT/'data/extended_paired_bootstrap.csv').open())
               if r['policy'] == 'fixed_0.5' and r['seed'] == '44']
    cis = {(r['comparison'], r['timepoint'], r['metric']): r for r in ci_rows}
    temporal = [r for r in csv.DictReader((ROOT/'data/extended_longitudinal.csv').open())
                if r['policy'] == 'fixed_0.5' and r['seed'] == '44']
    temporal_ci = {(r['model']+' T2 minus T0', r['metric']): r for r in temporal}
    tests = []
    for index, spec in enumerate(specifications):
        comparison, time, key = (spec[k] for k in ['comparison', 'timepoint', 'metric'])
        ci = temporal_ci[comparison, key] if spec['kind'] == 'longitudinal' else cis[comparison, time, key]
        assert abs(observed[index]-float(ci['delta'])) < 1e-12
        p_boot = (1+int(counts[index]))/(N_BOOTSTRAP+1)
        extra = dict(standard_error='', z='', bootstrap_exceedances=int(counts[index]),
                     bootstrap_replicates=N_BOOTSTRAP,
                     p_centered_bootstrap=p_boot,
                     p_bootstrap_monte_carlo_se=float(np.sqrt(p_boot*(1-p_boot)/N_BOOTSTRAP)))
        if key == 'auroc':
            aa, bb, ti = (spec[k] for k in ['aa', 'bb', 'ti'])
            if spec['kind'] == 'longitudinal':
                a, b = ensemble[aa, 2:3], ensemble[aa, 0:1]
            elif ti == 3:
                a, b = ensemble[aa], ensemble[bb]
            else:
                a, b = ensemble[aa, ti:ti+1], ensemble[bb, ti:ti+1]
            result = delong_difference(y, a, b)
            assert abs(result['delta']-observed[index]) < 1e-12
            p = result['p_raw']
            extra.update(standard_error=result['standard_error'], z=result['z'])
            method = 'paired_delong_asymptotic'
        else:
            p, method = p_boot, 'paired_null_centered_patient_bootstrap_approximate'
        tests.append(dict(seed=44, policy='fixed_0.5', kind=spec['kind'], family=spec['family'],
            comparison=comparison, timepoint=time, metric=key, method=method,
            delta=float(observed[index]), ci_low=float(ci['ci_low']), ci_high=float(ci['ci_high']),
            p_raw=float(p), **extra))
    for family, size in [(PRIMARY, 9), (SECONDARY, 81)]:
        selected = [r for r in tests if r['family'] == family]
        assert len(selected) == size
        for r, p in zip(selected, holm([r['p_raw'] for r in selected])):
            r.update(p_holm_family=float(p), family_size=size, significance=marker(p))
    for r, p in zip(tests, holm([r['p_raw'] for r in tests])):
        r.update(p_holm_all_90=float(p), significance_all_90=marker(p))
    with (ROOT/'data/significance_tests.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(tests[0]))
        writer.writeheader(); writer.writerows(tests)
    provenance = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(), training_seeds=[44],
        patients=175, positives=56, folds_per_ensemble=5, threshold=.5,
        bootstrap_replicates=N_BOOTSTRAP, rng_seed=RNG_SEED,
        numpy_version=np.__version__, batch_size=BATCH_SIZE,
        auroc_test='Two-sided paired DeLong asymptotic normal test. Macro averages positive and negative placement vectors across three landmarks before paired variance estimation.',
        other_metric_test='Two-sided null-centered paired patient bootstrap, approximate rather than finite-sample exact. Shared patient resampling preserves all model/landmark dependence.',
        bootstrap_p_formula='(1 + count(abs(bootstrap_delta - observed_delta) >= abs(observed_delta) - 1e-12)) / (99999 + 1)',
        null='Difference in the reported population metric is zero, conditional on the fitted models and evaluation policy.',
        macro='Arithmetic mean of separately recomputed landmark metrics, never pooled patient-landmark rows.',
        families={PRIMARY: '9: three principal model contrasts x Macro AUROC/AP/F1',
                  SECONDARY: '81: principal landmark metrics 27; other principal Macro metrics 33; pooling 12; principal T2-T0 9'},
        multiplicity='Holm within principal 9 and secondary 81 separately; additional Holm across all 90 exported as sensitivity analysis. Separate-family adjustment does not control the union of the two families.',
        intervals='Existing 2000 shared patient-bootstrap percentile 95% intervals; unadjusted, not exact inversions of these tests.',
        method_revision='After inspecting initial score-swap results, AUROC/other-metric primary methods were changed to target metric equality rather than conditional exchangeability of score trajectories. The original stronger-null randomization tests remain fully exported as sensitivity results; no method or family is claimed to be prospectively specified.',
        score_swap_sensitivity='data/significance_score_swap_sensitivity.csv: different, stronger score-exchangeability null; not used for table significance decisions.',
        interpretation='All analyses, method choices, and families are post-hoc/exploratory. Conditions on fixed selected-seed ensembles; excludes retraining and seed/aggregation/model selection uncertainty. Non-smooth metrics such as AP, F1, and ECE have finite-sample bootstrap limitations.',
        excluded='No new tests of OOF thresholds, subgroups, DCA thresholds, or archived training probes.',
        exported_patient_data=False,
        source_provenance_sha256=hashlib.sha256((ROOT/'data/extended_provenance.json').read_bytes()).hexdigest(),
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_count=len(tests),
        family_significant_counts={family: sum(r['p_holm_family'] < .05 for r in tests if r['family'] == family) for family in [PRIMARY, SECONDARY]},
        global_significant_count=sum(r['p_holm_all_90'] < .05 for r in tests),
        fitted_model_source_count=len(source['sources']))
    (ROOT/'data/significance_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps({k: provenance[k] for k in ['test_count', 'family_significant_counts', 'global_significant_count']}, indent=2))


if __name__ == '__main__':
    main()
