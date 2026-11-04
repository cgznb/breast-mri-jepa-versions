#!/usr/bin/env python3
"""Independent numerical verification; requires scipy and scikit-learn.

Writes aggregate checks only. Patient-level source predictions stay in memory.
"""
from pathlib import Path
from datetime import datetime, timezone
from itertools import product
import csv
import hashlib
import json
import time
import warnings

import numpy as np
import scipy
from scipy.stats import binomtest, rankdata
import sklearn
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    recall_score, precision_score, balanced_accuracy_score, matthews_corrcoef,
    brier_score_loss, log_loss,
)

import significance_analysis as sa
import significance_score_swap as ss
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]


def reference_metrics(y, p):
    pred = p >= .5
    sensitivity = recall_score(y, pred, zero_division=0)
    specificity = recall_score(1-y, ~pred, zero_division=0)
    bins = np.digitize(p, np.arange(.1, 1., .1), right=False)
    # Multiplication by ten is the generating script's bin convention;
    # use integer lower boundaries to avoid independently rounded edges.
    bins = np.array([min(int(float(q)*10), 9) for q in p])
    ece = sum(abs(float((p[bins == k]-y[bins == k]).sum())) for k in range(10))/len(y)
    return np.array([
        roc_auc_score(y, p), average_precision_score(y, p),
        f1_score(y, pred, zero_division=0), accuracy_score(y, pred),
        sensitivity, specificity, precision_score(y, pred, zero_division=0),
        precision_score(1-y, ~pred, zero_division=0),
        balanced_accuracy_score(y, pred), matthews_corrcoef(y, pred),
        sensitivity+specificity-1, brier_score_loss(y, p),
        log_loss(y, np.clip(p, 1e-7, 1-1e-7), labels=[0, 1]), ece,
    ])


def rank_reference(y, scores):
    """AUC via average ranks; AP via positive/all threshold-rank ratio.

    This does not use the production cumulative group-end implementation.
    """
    pos = y.astype(bool)
    npos, nneg = int(pos.sum()), int((~pos).sum())
    ranks = rankdata(scores, method='average', axis=-1)
    auc = (ranks[..., pos].sum(axis=-1)-npos*(npos+1)/2)/(npos*nneg)
    all_reverse_ranks = rankdata(-scores, method='max', axis=-1)
    positive_reverse_ranks = rankdata(-scores[..., pos], method='max', axis=-1)
    ap = (positive_reverse_ranks/all_reverse_ranks[..., pos]).mean(axis=-1)
    predicted = scores >= .5
    tp = predicted[..., pos].sum(axis=-1)
    fp = predicted[..., ~pos].sum(axis=-1)
    f1 = 2*tp/(npos+tp+fp)
    return np.stack([auc, ap, f1], axis=-1)


def reference_holm(values):
    values = np.asarray(values)
    ranks = sorted(range(len(values)), key=lambda i: (values[i], i))
    adjusted = np.empty(len(values))
    for rank, idx in enumerate(ranks):
        adjusted[idx] = min(1., max((len(values)-j)*values[ranks[j]] for j in range(rank+1)))
    return adjusted


def main():
    started = time.monotonic()
    rng = np.random.default_rng(8439)
    y = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    cases = [rng.random((8,)) for _ in range(40)]
    cases += [rng.choice([0., .2, .5, .8, 1.], size=8) for _ in range(40)]
    cases += [np.zeros(8), np.ones(8), np.full(8, .5), np.full(8, .2),
              np.arange(8)/10, y.astype(float), 1-y.astype(float)]
    p = np.stack(cases)
    produced = ss.batch_metrics(y, p)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        expected = np.stack([reference_metrics(y, q) for q in p])
    metric_error = float(np.max(np.abs(produced-expected)))
    np.testing.assert_allclose(produced, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(rank_reference(y, p), expected[:, :3], atol=1e-12, rtol=0)

    toy_y = np.array([0, 1, 0, 1, 0, 1])
    a = np.array([[.2,.8,.5,.7,.1,.5], [.8,.3,.2,.6,.3,.9], [.1,.6,.4,.5,.3,.8]])
    b = np.array([[.6,.2,.4,.8,.5,.3], [.2,.9,.6,.5,.8,.2], [.7,.3,.1,.6,.5,.4]])
    bits = np.array(list(product([False, True], repeat=len(toy_y))))[:, None, :]
    perm_a, perm_b = np.where(bits, b, a), np.where(bits, a, b)
    exact_delta = ss.batch_metrics(toy_y, perm_a)-ss.batch_metrics(toy_y, perm_b)
    independent_delta = rank_reference(toy_y, perm_a)-rank_reference(toy_y, perm_b)
    np.testing.assert_allclose(exact_delta[..., :3], independent_delta, atol=1e-12, rtol=0)
    # Check every toy permutation against sklearn for the first landmark.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sklearn_delta = np.stack([reference_metrics(toy_y, x[0])-reference_metrics(toy_y, z[0]) for x,z in zip(perm_a, perm_b)])
    np.testing.assert_allclose(exact_delta[:, 0], sklearn_delta, atol=1e-12, rtol=0)
    observed = ss.batch_metrics(toy_y, a)-ss.batch_metrics(toy_y, b)
    binomial_checks = 0
    for key, eligible in [(3, np.ones(len(toy_y), bool)), (4, toy_y == 1), (5, toy_y == 0)]:
        pred_a, pred_b = a[0] >= .5, b[0] >= .5
        correct_a, correct_b = pred_a == toy_y, pred_b == toy_y
        wins = int((eligible & correct_a & ~correct_b).sum())
        losses = int((eligible & ~correct_a & correct_b).sum())
        reference_p = binomtest(wins, wins+losses, .5).pvalue if wins+losses else 1.
        exact_p = float((np.abs(exact_delta[:, 0, key]) >= abs(observed[0, key])-1e-12).mean())
        assert abs(reference_p-exact_p) < 1e-12
        binomial_checks += 1
    equal_delta, equal_counts = ss.paired_test(toy_y, a, a, permutations=127)
    assert np.all(equal_delta == 0) and np.all(equal_counts == 127)
    forward, fc = ss.paired_test(toy_y, a, b, permutations=127)
    reverse, rc = ss.paired_test(toy_y, b, a, permutations=127)
    np.testing.assert_allclose(forward, -reverse, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(fc, rc)
    check_rng = np.random.default_rng(sa.RNG_SEED)
    toy_swap = check_rng.integers(0,2,size=(127,1,len(toy_y)),dtype=np.uint8).astype(bool)
    d = rank_reference(toy_y, np.where(toy_swap,b,a))-rank_reference(toy_y,np.where(toy_swap,a,b))
    d = np.concatenate([d, d.mean(axis=1,keepdims=True)],axis=1)
    np.testing.assert_array_equal(fc[:, :3], (np.abs(d) >= np.abs(forward[:, :3])[None]-1e-12).sum(axis=0))
    holm_cases = [np.array([.01,.04,.03,.002]), np.array([0.,0.,.05,.05,1.]), rng.random(90)]
    for case in holm_cases:
        np.testing.assert_allclose(sa.holm(case), reference_holm(case), atol=1e-15, rtol=0)

    actual_y, ensemble, _ = sa.load_ensembles()
    actual_metrics = ss.batch_metrics(actual_y, ensemble)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        actual_reference = np.stack([reference_metrics(actual_y, q) for q in ensemble.reshape(-1,len(actual_y))]).reshape(4,3,14)
    np.testing.assert_allclose(actual_metrics, actual_reference, atol=1e-12, rtol=0)
    actual_error = float(np.max(np.abs(actual_metrics-actual_reference)))
    print('Independent metric, exhaustive toy, symmetry, McNemar, Holm checks passed.', flush=True)

    # Independent fast DeLong implementation uses joint midrank covariance,
    # rather than pairwise indicator placement vectors used in production.
    order = np.r_[np.flatnonzero(actual_y == 1),np.flatnonzero(actual_y == 0)]
    scores = ensemble.reshape(12,-1)[:,order]
    npos,nneg = int(actual_y.sum()),int((1-actual_y).sum())
    tx = rankdata(scores[:,:npos],method='average',axis=1)
    ty = rankdata(scores[:,npos:],method='average',axis=1)
    tz = rankdata(scores,method='average',axis=1)
    aucs = tz[:,:npos].sum(axis=1)/(npos*nneg)-(npos+1)/(2*nneg)
    v01 = (tz[:,:npos]-tx)/nneg
    v10 = 1-(tz[:,npos:]-ty)/npos
    covariance = np.cov(v01,ddof=1)/npos + np.cov(v10,ddof=1)/nneg
    specifications = sa.hypotheses()
    independent_delong = {}
    for spec in specifications:
        if spec['metric'] != 'auroc':continue
        c = np.zeros((4,3))
        if spec['kind'] == 'longitudinal':
            c[spec['aa'],2]=1;c[spec['aa'],0]=-1
        elif spec['ti'] == 3:
            c[spec['aa'],:]=1/3;c[spec['bb'],:]=-1/3
        else:
            c[spec['aa'],spec['ti']]=1;c[spec['bb'],spec['ti']]=-1
        contrast=c.ravel()
        delta=float(contrast@aucs)
        se=float(np.sqrt(max(float(contrast@covariance@contrast),0.)))
        p_value=float(2*norm.sf(abs(delta/se))) if se else (1. if delta == 0 else 0.)
        independent_delong[spec['comparison'],spec['timepoint'],spec['metric']] = (delta,se,p_value)
    # Explicit patient index resampling, followed by independent scipy rank
    # formulas. Shared weights are expanded into repeated actual patient rows.
    principal = [s for s in specifications if s['family'] == sa.PRIMARY]
    observed_values = np.concatenate([actual_reference,actual_reference.mean(axis=1,keepdims=True)],axis=1)
    observed_primary = np.array([sa.metric_difference(observed_values,s) for s in principal])
    counts = np.zeros(9,np.int64)
    actual_rng = np.random.default_rng(sa.RNG_SEED)
    patient_indices = np.arange(len(actual_y))
    for first in range(0,sa.N_BOOTSTRAP,sa.BATCH_SIZE):
        size=min(sa.BATCH_SIZE,sa.N_BOOTSTRAP-first)
        weights=actual_rng.multinomial(len(actual_y),np.full(len(actual_y),1/len(actual_y)),size=size)
        indices=np.stack([np.repeat(patient_indices,w) for w in weights])
        yy=actual_y[indices][:,None,None,:]
        sampled=ensemble[:3,:,indices].transpose(2,0,1,3)
        npositive=yy.sum(axis=-1)
        nnegative=len(actual_y)-npositive
        ranks=rankdata(sampled,method='average',axis=-1)
        auc=(np.where(yy,ranks,0).sum(axis=-1)-npositive*(npositive+1)/2)/(npositive*nnegative)
        # Among-positive threshold ranks are obtained by placing every
        # negative score below all positives, then ranking in descending order.
        all_threshold_ranks=rankdata(-sampled,method='max',axis=-1)
        positive_threshold_ranks=rankdata(-np.where(yy,sampled,-np.inf),method='max',axis=-1)
        ap=np.where(yy,positive_threshold_ranks/all_threshold_ranks,0).sum(axis=-1)/npositive
        prediction=sampled >= .5
        true_positive=(prediction*yy).sum(axis=-1)
        false_positive=(prediction*(1-yy)).sum(axis=-1)
        f1=2*true_positive/(npositive+true_positive+false_positive)
        vals=np.stack([auc,ap,f1],axis=-1).mean(axis=2)
        ds=np.stack([vals[:,s['aa'],s['ki']]-vals[:,s['bb'],s['ki']] for s in principal],axis=1)
        counts += (np.abs(ds-observed_primary)>=np.abs(observed_primary)-1e-12).sum(axis=0)
        if first == 0 or first+size == sa.N_BOOTSTRAP:
            print('Independent explicit-row bootstrap:',first+size,'/',sa.N_BOOTSTRAP,flush=True)
    independent_counts={(s['comparison'],s['metric']):int(n) for s,n in zip(principal,counts)}

    path = ROOT/'data/significance_tests.csv'
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 90 and len({(r['comparison'],r['timepoint'],r['metric']) for r in rows}) == 90
    for family, expected_n in [(sa.PRIMARY,9),(sa.SECONDARY,81)]:
        selected = [r for r in rows if r['family'] == family]
        assert len(selected) == expected_n
        independent_adjusted = reference_holm([float(r['p_raw']) for r in selected])
        for r, p_adj in zip(selected,independent_adjusted):
            assert int(r['family_size']) == expected_n
            assert r['seed'] == '44' and r['policy'] == 'fixed_0.5'
            p_boot = (1+int(r['bootstrap_exceedances']))/(1+int(r['bootstrap_replicates']))
            assert abs(float(r['p_centered_bootstrap'])-p_boot) < 1e-15
            if r['metric'] != 'auroc':
                assert abs(float(r['p_raw'])-p_boot) < 1e-15
            else:
                delta,se,p_delong=independent_delong[r['comparison'],r['timepoint'],r['metric']]
                assert abs(float(r['delta'])-delta)<1e-12
                assert abs(float(r['standard_error'])-se)<1e-12
                assert abs(float(r['p_raw'])-p_delong)<1e-12
            assert abs(float(r['p_holm_family'])-p_adj) < 1e-15
            star = '***' if p_adj < .001 else '**' if p_adj < .01 else '*' if p_adj < .05 else 'ns'
            assert r['significance'] == star
            if family == sa.PRIMARY:
                assert int(r['bootstrap_exceedances']) == independent_counts[r['comparison'],r['metric']]
    global_adjusted = reference_holm([float(r['p_raw']) for r in rows])
    for r,p_adj in zip(rows,global_adjusted):
        assert abs(float(r['p_holm_all_90'])-p_adj) < 1e-15
        assert r['significance_all_90'] == ('***' if p_adj < .001 else '**' if p_adj < .01 else '*' if p_adj < .05 else 'ns')
    provenance = json.loads((ROOT/'data/significance_provenance.json').read_text())
    assert provenance['analysis_script_sha256'] == hashlib.sha256(Path(sa.__file__).read_bytes()).hexdigest()
    assert provenance['source_provenance_sha256'] == hashlib.sha256((ROOT/'data/extended_provenance.json').read_bytes()).hexdigest()
    receipt = dict(
        verified_at_utc=datetime.now(timezone.utc).isoformat(),
        numpy_version=np.__version__,scipy_version=scipy.__version__,sklearn_version=sklearn.__version__,
        metric_comparisons=len(cases)*14,metric_max_absolute_error=metric_error,
        actual_point_metric_comparisons=4*3*14,actual_point_metric_max_absolute_error=actual_error,
        exact_toy_permutations=2**len(toy_y),exact_toy_shared_landmarks=3,
        toy_all14_sklearn_permutation_comparisons=2**len(toy_y)*14,
        exact_binomial_mcnemar_checks=binomial_checks,
        identical_predictions_p_equals_one=True,model_order_reversal_invariant=True,
        shared_landmark_swap_counts_verified=True,holm_independent_cases=len(holm_cases),
        actual_primary_bootstrap_replicates_independently_recomputed=sa.N_BOOTSTRAP,
        actual_primary_exceedance_counts_matched=9,
        independent_primary_method='Explicit repeated patient rows from shared bootstrap weights; scipy rankdata average-rank AUC and AP positive/all inclusive threshold rank ratios; repeated-row confusion-count F1',
        independent_delong_comparisons=len(independent_delong),
        independent_delong_method='Joint 12-score fast DeLong midranks and covariance, with contrast vectors for correlated Macro and landmarks',
        output_rows_verified=90,family_counts_verified={'principal_macro_9':9,'secondary_81':81},
        raw_p_values_recalculated=90,family_holm_values_verified=90,global_holm_values_verified=90,
        all_significance_markers_verified=True,source_hashes_verified=True,
        analysis_script_sha256=hashlib.sha256(Path(sa.__file__).read_bytes()).hexdigest(),
        verification_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        significance_csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        elapsed_seconds=round(time.monotonic()-started,3),aggregate_only=True,
    )
    (ROOT/'notes/significance_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
