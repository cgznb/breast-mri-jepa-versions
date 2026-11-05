#!/usr/bin/env python3
"""Larger independent simulation of the borderline T2 AUROC comparison.

Keep the uniform 99,999-draw 78-test table unchanged. This check repeats
ForecastMean minus NoJEPA at T2 with 999,999 paired patient resamples.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import numpy as np
from analysis_common import load_data, SEEDS
from extended_metrics import all_metrics
from significance_analysis import holm

ROOT = Path(__file__).resolve().parents[1]
B = 999_999
RNG = 20261104
BATCH = 2048


def weighted_auc(y, p, weights):
    order = np.argsort(-p, kind='stable')
    yy, pp, ww = y[order], p[order], weights[:, order]
    ends = np.r_[np.flatnonzero(np.diff(pp)), len(pp)-1]
    positives = weights @ y
    negatives = weights.sum(axis=1)-positives
    assert (positives > 0).all() and (negatives > 0).all()
    tpr = np.cumsum(ww*yy, axis=1)[:, ends]/positives[:, None]
    fpr = np.cumsum(ww*(1-yy), axis=1)[:, ends]/negatives[:, None]
    previous_tpr = np.c_[np.zeros(len(weights)), tpr[:, :-1]]
    previous_fpr = np.c_[np.zeros(len(weights)), fpr[:, :-1]]
    return ((tpr+previous_tpr)*(fpr-previous_fpr)/2).sum(axis=1)


def main():
    data = load_data()
    y, ensemble = data['y'], data['ensemble']
    assert SEEDS == [42, 44, 71, 2026] and ensemble.shape == (4, 3, 3, 175)
    rows = list(csv.DictReader((ROOT/'data/significance_tests.csv').open()))
    assert len(rows) == 78
    index = next(i for i, row in enumerate(rows)
                 if row['comparison'] == 'ForecastMean minus NoJEPA'
                 and row['timepoint'] == 'T2' and row['metric'] == 'auroc')
    source = rows[index]
    one = np.ones((1, len(y)))
    observed = float(np.mean([
        weighted_auc(y, ensemble[si, 2, 2], one)[0]
        - weighted_auc(y, ensemble[si, 0, 2], one)[0]
        for si in range(len(SEEDS))]))
    assert abs(observed-float(source['delta'])) < 1e-12
    rng = np.random.default_rng(RNG)
    extreme = 0
    for first in range(0, B, BATCH):
        weights = rng.multinomial(len(y), np.full(len(y), 1/len(y)),
                                  size=min(BATCH, B-first)).astype(float)
        differences = np.zeros(len(weights))
        for si in range(len(SEEDS)):
            for mi, sign in [(2, 1), (0, -1)]:
                auc = weighted_auc(y, ensemble[si, mi, 2], weights)
                if first == 0:
                    reference = all_metrics(y, ensemble[si, mi, 2], weights[:32])[:, 0]
                    np.testing.assert_allclose(auc[:32], reference, atol=1e-12, rtol=0)
                differences += sign*auc/len(SEEDS)
        extreme += int((np.abs(differences-observed) >= abs(observed)-1e-12).sum())
    p = (1+extreme)/(B+1)
    se = math.sqrt(p*(1-p)/B)
    z = 1.96
    denominator = 1+z*z/B
    center = (extreme/B+z*z/(2*B))/denominator
    half = z*math.sqrt((extreme/B)*(1-extreme/B)/B+z*z/(4*B*B))/denominator
    adjusted = {}
    for name, indices in [
            ('secondary_69', [i for i, row in enumerate(rows) if row['family'] == 'secondary_69']),
            ('all_78', list(range(78)))]:
        values = [p if i == index else float(rows[i]['p_raw']) for i in indices]
        adjusted[name] = float(holm(values)[indices.index(index)])
    result = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(), training_seeds=SEEDS,
        models=['NoJEPA', 'DirectReadout', 'ForecastMean'],
        comparison=source['comparison'], timepoint='T2', metric='auroc', delta=observed,
        reason='The uniform-table all-78 Holm p=0.04408 is close to 0.05 relative to its Monte Carlo uncertainty; repeat this same hypothesis in a larger independently seeded simulation.',
        primary_bootstrap_replicates=int(source['bootstrap_replicates']),
        primary_p_raw=float(source['p_raw']),
        primary_p_holm_secondary=float(source['p_holm_family']),
        primary_p_holm_all_78=float(source['p_holm_all_78']),
        extra_bootstrap_replicates=B, extra_rng_seed=RNG, extra_exceedances=extreme,
        extra_p_raw=p, extra_p_monte_carlo_se=se,
        monte_carlo_probability_wilson95=[center-half, center+half],
        adjusted_p_with_only_this_entry_replaced=adjusted,
        method='Same approximate null-centered paired patient bootstrap; separately recompute each seed-specific T2 AUROC from its fivefold probability ensemble, then equally average the four seed differences; other 77 p-values held fixed for this sensitivity check.',
        reporting='Separate numerical stability sensitivity, not a replacement for the uniformly generated 78-test table. All analyses and reporting families are exploratory.',
        source_significance_csv_sha256=hashlib.sha256((ROOT/'data/significance_tests.csv').read_bytes()).hexdigest(),
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        aggregate_only=True)
    (ROOT/'data/monte_carlo_stability.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
