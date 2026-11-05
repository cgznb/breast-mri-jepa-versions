#!/usr/bin/env python3
"""Exploratory paired tests of four-seed mean metric differences.

All metrics: null-centered paired patient bootstrap, an approximate test.
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
from analysis_common import load_data,SEEDS,SEED_TEXT

def holm(p):
    p=np.asarray(p,float);order=np.argsort(p,kind="stable");result=np.empty_like(p)
    result[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)))
    return result

def marker(p):return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"

ROOT = Path(__file__).resolve().parents[1]
N_BOOTSTRAP = 99_999
RNG_SEED = 20261103
BATCH_SIZE = 1024
PRIMARY = 'principal_macro_9'
SECONDARY = 'secondary_81'


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
    data=load_data(); y,ensemble=data['y'],data['ensemble'];source=data
    specifications = hypotheses()
    per_seed=np.array([[[all_metrics(y,p,np.ones((1,len(y))))[0,:14] for p in scores] for scores in models] for models in ensemble])
    point=per_seed.mean(axis=0)
    point = np.concatenate([point, point.mean(axis=1, keepdims=True)], axis=1)
    summary = {(r['model'], r['timepoint']): r
               for r in csv.DictReader((ROOT/'data/extended_summary.csv').open())
               if r['policy'] == 'fixed_0.5' and r['seed'] == SEED_TEXT}
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
                values[:,mi,ti]=np.mean([all_metrics(y,ensemble[si,mi,ti],weights)[:,:14] for si in range(len(SEEDS))],axis=0)
        assert np.isfinite(values).all(), 'Do not silently omit undefined bootstrap replicates.'
        values = np.concatenate([values, values.mean(axis=2, keepdims=True)], axis=2)
        deltas = np.stack([metric_difference(values, s) for s in specifications], axis=1)
        counts += (np.abs(deltas-observed) >= np.abs(observed)-1e-12).sum(axis=0)
        if first == 0 or first+size == N_BOOTSTRAP:
            print(f'Computed shared patient bootstrap {first+size}/{N_BOOTSTRAP}', flush=True)
    ci_rows = [r for r in csv.DictReader((ROOT/'data/extended_paired_bootstrap.csv').open())
               if r['policy'] == 'fixed_0.5' and r['seed'] == SEED_TEXT]
    cis = {(r['comparison'], r['timepoint'], r['metric']): r for r in ci_rows}
    temporal = [r for r in csv.DictReader((ROOT/'data/extended_longitudinal.csv').open())
                if r['policy'] == 'fixed_0.5' and r['seed'] == SEED_TEXT]
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
        p,method=p_boot,'paired_null_centered_patient_bootstrap_four_seed_mean_approximate'
        tests.append(dict(seed=SEED_TEXT, policy='fixed_0.5', kind=spec['kind'], family=spec['family'],
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
        generated_at_utc=datetime.now(timezone.utc).isoformat(), training_seeds=SEEDS,
        patients=175, positives=56, folds_per_ensemble=5, threshold=.5,
        bootstrap_replicates=N_BOOTSTRAP, rng_seed=RNG_SEED,
        numpy_version=np.__version__, batch_size=BATCH_SIZE,
        metric_test='Two-sided null-centered paired patient bootstrap for all 14 metrics, including AUROC. Metric is recomputed separately for each seed then equally averaged; shared patient draws preserve seed/model/landmark dependence.',
        bootstrap_p_formula='(1 + count(abs(bootstrap_delta - observed_delta) >= abs(observed_delta) - 1e-12)) / (99999 + 1)',
        null='Difference in the mean population metric over these four fixed fitted seed-specific ensembles is zero.',
        macro='Arithmetic mean of separately recomputed landmark metrics within seed, then equal mean over seeds; never pooled patient-landmark rows or cross-seed probability ensembles.',
        families={PRIMARY: '9: three principal model contrasts x Macro AUROC/AP/F1',
                  SECONDARY: '81: principal landmark metrics 27; other principal Macro metrics 33; pooling 12; principal T2-T0 9'},
        multiplicity='Holm within principal 9 and secondary 81 separately; additional Holm across all 90 exported as sensitivity analysis. Separate-family adjustment does not control the union of the two families.',
        intervals='2000 shared patient-bootstrap percentile 95% intervals; unadjusted, not exact inversions of these tests.',
        interpretation='All analyses, selected seed set, and reporting families are post-hoc/exploratory. Conditions on fixed selected-seed ensembles; excludes retraining and seed/aggregation/model selection uncertainty. No seeds resampled. Sample SD over four selected seeds is descriptive. Non-smooth metrics such as AP, F1, and ECE have finite-sample bootstrap limitations.',
        excluded='No new tests of OOF thresholds, subgroups, DCA thresholds, or archived training probes.',
        exported_patient_data=False,
        source_provenance_sha256=hashlib.sha256((ROOT/'data/extended_provenance.json').read_bytes()).hexdigest(),
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_count=len(tests),
        family_significant_counts={family: sum(r['p_holm_family'] < .05 for r in tests if r['family'] == family) for family in [PRIMARY, SECONDARY]},
        global_significant_count=sum(r['p_holm_all_90'] < .05 for r in tests),
        fitted_model_source_count=len(data['sources']))
    (ROOT/'data/significance_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps({k: provenance[k] for k in ['test_count', 'family_significant_counts', 'global_significant_count']}, indent=2))


if __name__ == '__main__':
    main()
