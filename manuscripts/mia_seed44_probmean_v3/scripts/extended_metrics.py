#!/usr/bin/env python3
"""Seed44 patient probability-ensemble evaluation; aggregate-only outputs."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUITE = Path('/projects/xts0030/public/cxx/medium976_t0_t2_no_tf_cv5_multiseed_v1')
MODELS = ['direct_no_jepa', 'direct_readout', 'forecast_attention', 'forecast_mean']
LABELS = ['NoJEPA', 'DirectReadout', 'ForecastAttention', 'ForecastMean']
SEEDS = [44]
TIMES = ['T0', 'T1', 'T2', 'Macro']
POLICIES = ['fixed_0.5', 'oof_validation_youden']
METRICS = ['auroc', 'auprc', 'f1', 'accuracy', 'sensitivity', 'specificity', 'ppv',
           'npv', 'balanced_accuracy', 'mcc', 'youden_j', 'brier', 'nll',
           'ece_10_equal_width', 'tp', 'fp', 'fn', 'tn']
B = 2000
BOOTSTRAP_RNG = 20261102


def dump_csv(name, rows):
    with (ROOT / 'data' / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def ratio(a, b):
    return np.divide(a, b, out=np.full_like(np.asarray(a, float), np.nan), where=b != 0)


def all_metrics(y, p, weights, threshold=.5):
    """One row per weighted patient resample; tied probabilities handled by groups."""
    w = weights.astype(float, copy=False)
    total = w.sum(axis=1)
    pos = (w * y).sum(axis=1); neg = total - pos
    order = np.argsort(-p, kind='stable')
    ys, ps, ws = y[order], p[order], w[:, order]
    ends = np.r_[np.flatnonzero(np.diff(ps)), len(p)-1]
    tp_curve = np.cumsum(ws * ys, axis=1)[:, ends]
    fp_curve = np.cumsum(ws * (1-ys), axis=1)[:, ends]
    tpr = ratio(tp_curve, pos[:, None]); fpr = ratio(fp_curve, neg[:, None])
    prev_tpr = np.c_[np.zeros(len(w)), tpr[:, :-1]]
    prev_fpr = np.c_[np.zeros(len(w)), fpr[:, :-1]]
    auroc = ((tpr+prev_tpr)*(fpr-prev_fpr)/2).sum(axis=1)
    precision = np.divide(tp_curve, tp_curve+fp_curve, out=np.zeros_like(tp_curve), where=tp_curve+fp_curve > 0)
    ap = (precision*(tpr-prev_tpr)).sum(axis=1)
    pred = p >= threshold
    tp = (w * (y * pred)).sum(axis=1)
    fp = (w * ((1-y) * pred)).sum(axis=1)
    fn = pos-tp; tn = neg-fp
    sens, spec = ratio(tp,pos), ratio(tn,neg)
    ppv, npv = ratio(tp,tp+fp), ratio(tn,tn+fn)
    f1 = ratio(2*tp,2*tp+fp+fn)
    acc = ratio(tp+tn,total)
    mcc = ratio(tp*tn-fp*fn,np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    brier = (w * (p-y)**2).sum(axis=1)/total
    clipped = np.clip(p,1e-7,1-1e-7)
    losses = -(y*np.log(clipped)+(1-y)*np.log(1-clipped))
    nll = (w*losses).sum(axis=1)/total
    # Sum absolute (predicted positive mass - observed positive mass), divided by n.
    bins = np.minimum((p*10).astype(int),9)
    ece = np.zeros(len(w))
    for bi in range(10):
        selected = bins == bi
        ece += np.abs((w[:, selected]*(p[selected]-y[selected])).sum(axis=1))/total
    return np.stack([auroc,ap,f1,acc,sens,spec,ppv,npv,(sens+spec)/2,mcc,
                     sens+spec-1,brier,nll,ece,tp,fp,fn,tn],axis=-1)


def select_youden(y, p):
    """Validation-only optimum; highest threshold breaks exact Youden ties."""
    candidates = np.r_[np.nextafter(p.max(), np.inf), np.unique(p)[::-1]]
    pos,neg = y.sum(),len(y)-y.sum()
    assert pos > 0 and neg > 0
    pred = p[None,:] >= candidates[:,None]
    j = (pred*y).sum(axis=1)/pos + ((~pred)*(1-y)).sum(axis=1)/neg - 1
    best = int(np.flatnonzero(np.isclose(j,j.max(),atol=1e-14,rtol=0))[0])
    return float(candidates[best]),float(j[best])


def main():
    matrix = json.loads((SUITE / 'matrix.json').read_text())
    assert matrix['status'] == 'completed' and matrix['test_visible']
    probs = np.empty((4, 5, 3, 175))
    thresholds = np.full((2, 4, 3), .5)
    identities, validation, fold_rows = [], [], []
    oof = [[[] for _ in range(3)] for _ in MODELS]
    patient_ids = y = None
    reference_oof = [None] * 3
    verified_report_identities = 0
    for mi, model in enumerate(MODELS):
        for fi in range(5):
            run = SUITE / 'runs' / f'{model}_seed44_fold{fi+1}'
            receipt_path = run / 'evaluation_done.json'
            config_path = run / 'config.json'
            receipt = json.loads(receipt_path.read_text())
            assert receipt['completed'] is True
            manifest_path = Path(receipt['manifest']['path'])
            manifest_stat = manifest_path.stat()
            assert (manifest_stat.st_size, manifest_stat.st_mtime_ns) == (
                receipt['manifest']['bytes'], receipt['manifest']['mtime_ns'])
            manifest = json.loads(manifest_path.read_text())
            manifest_patients = {r['patient_key']: r for r in manifest['patients']}
            assert len(manifest_patients) == 976
            report_identities = []
            for identity in receipt['reports']:
                path = Path(identity['path'])
                st = path.stat()
                assert (st.st_size, st.st_mtime_ns) == (identity['bytes'], identity['mtime_ns'])
                report_identities.append(dict(path=str(path), bytes=st.st_size,
                                              mtime_ns=st.st_mtime_ns,
                                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                verified_report_identities += 1
            config = json.loads(config_path.read_text())
            assert config['seed'] == 44
            test_path = run / 'evaluation_best_test.json'
            val_path = run / 'evaluation_best_val.json'
            test = json.loads(test_path.read_text())
            val = json.loads(val_path.read_text())
            assert test['split'] == 'test' and test['downstream_holdout'] and test['independent_test'] is False
            assert val['split'] == 'val'
            identities.append(dict(model=LABELS[mi], seed=44, fold=fi+1,
                                   receipt=str(receipt_path),
                                   receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                                   config=str(config_path),
                                   config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                                   manifest=str(manifest_path),
                                   manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                                   test_report=str(test_path),
                                   test_sha256=hashlib.sha256(test_path.read_bytes()).hexdigest(),
                                   validation_report=str(val_path),
                                   validation_sha256=hashlib.sha256(val_path.read_bytes()).hexdigest(),
                                   registered_reports=report_identities))
            for ti in range(3):
                rows = sorted((r for r in test['rows'] if r['landmark'] == ti), key=lambda r: r['patient_key'])
                ids = [r['patient_key'] for r in rows]
                yy = np.array([r['pcr'] for r in rows])
                pp = np.array([r['pcr_probability'] for r in rows])
                if patient_ids is None:
                    patient_ids, y = ids, yy
                assert len(set(ids)) == 175
                assert ids == patient_ids and np.array_equal(yy, y) and len(yy) == 175 and yy.sum() == 56
                assert all(manifest_patients[k]['split'] == 'test' and
                           manifest_patients[k]['target']['pcr'] == label for k, label in zip(ids, yy))
                assert np.isfinite(pp).all() and ((pp >= 0) & (pp <= 1)).all()
                probs[mi, fi, ti] = pp
                vr = [r for r in val['rows'] if r['landmark'] == ti]
                assert not (set(ids) & {r['patient_key'] for r in vr})
                assert all(manifest_patients[r['patient_key']]['split'] == 'val' and
                           manifest_patients[r['patient_key']]['target']['pcr'] == r['pcr'] for r in vr)
                oof[mi][ti].extend(vr)
                # Individual-fold scores are checked only for source/audit consistency.
                m = all_metrics(y, pp, np.ones((1, 175)))[0]
                for key in ['auroc', 'auprc', 'f1', 'brier', 'nll', 'ece_10_equal_width']:
                    assert abs(m[METRICS.index(key)] - test['per_landmark'][TIMES[ti]][key]) < 1e-12, (key, mi, fi, ti)
        for ti in range(3):
            vr = sorted(oof[mi][ti], key=lambda r: r['patient_key'])
            ids = [r['patient_key'] for r in vr]
            assert len(ids) == len(set(ids)) == [801, 760, 700][ti]
            assert not (set(patient_ids) & set(ids))
            vy = np.array([r['pcr'] for r in vr])
            vp = np.array([r['pcr_probability'] for r in vr])
            if reference_oof[ti] is None:
                reference_oof[ti] = (ids, vy)
            assert ids == reference_oof[ti][0] and np.array_equal(vy, reference_oof[ti][1])
            threshold, j = select_youden(vy, vp)
            thresholds[1, mi, ti] = threshold
            validation.append(dict(policy=POLICIES[1], model=LABELS[mi], seed=44,
                                   timepoint=TIMES[ti], validation_patients=len(vy),
                                   validation_positives=int(vy.sum()), folds=5,
                                   threshold=threshold, validation_youden_j=j,
                                   score_type='one disjoint held-out-fold model score per development patient',
                                   application_score='test mean of five fold probabilities',
                                   tie_rule='largest threshold',
                                   caveat='OOF single-model and test ensemble score distributions differ; sensitivity analysis only'))
    assert verified_report_identities == 60 and len(identities) == 20
    # Essential estimand: mean the five probabilities for each patient FIRST.
    pbar = probs.mean(axis=1)
    weights = np.random.default_rng(BOOTSTRAP_RNG).multinomial(175, np.full(175, 1/175), size=B)
    assert ((weights*y).sum(axis=1) > 0).all() and ((weights*(1-y)).sum(axis=1) > 0).all()
    point = np.empty((2, 4, 3, len(METRICS)))
    boot = np.empty((B, 2, 4, 3, len(METRICS)))
    for pi, policy in enumerate(POLICIES):
        for mi, model in enumerate(LABELS):
            for ti in range(3):
                threshold = thresholds[pi, mi, ti]
                point[pi, mi, ti] = all_metrics(y, pbar[mi, ti], np.ones((1, 175)), threshold)[0]
                boot[:, pi, mi, ti] = all_metrics(y, pbar[mi, ti], weights, threshold)
                for fi in range(5):
                    fm = all_metrics(y, probs[mi, fi, ti], np.ones((1, 175)), threshold)[0]
                    fold_rows.append(dict(policy=policy, seed=44, model=model, fold=fi+1,
                                          timepoint=TIMES[ti], threshold=float(threshold),
                                          role='individual_fold_model_audit_only',
                                          **dict(zip(METRICS, map(float, fm)))))
    # Macro averages three landmark METRIC values, never probabilities across time.
    means = np.concatenate([point, point.mean(axis=2, keepdims=True)], axis=2)
    boot_means = np.concatenate([boot, boot.mean(axis=3, keepdims=True)], axis=3)
    ci = np.nanquantile(boot_means, [.025, .975], axis=0)
    summary = []
    for pi, policy in enumerate(POLICIES):
        for mi, model in enumerate(LABELS):
            for ti, time in enumerate(TIMES):
                row = dict(policy=policy, seed=44, model=model, timepoint=time,
                           folds_per_seed=5, training_seed_count=1, patients=175, positives=56,
                           aggregation='patient_probability_mean_before_metrics',
                           threshold=float(thresholds[pi, mi, ti]) if ti < 3 else float('nan'))
                for ki, key in enumerate(METRICS):
                    row[key] = float(means[pi, mi, ti, ki])
                    if ki < 14:
                        row[key+'_patient_ci_low'] = float(ci[0, pi, mi, ti, ki])
                        row[key+'_patient_ci_high'] = float(ci[1, pi, mi, ti, ki])
                summary.append(row)
    expected = np.array([[.7279, .5968, .5416], [.7399, .6075, .5678],
                         [.7507, .6283, .5604], [.7465, .6139, .5742]])
    assert np.array_equal(np.round(means[0, :, 3, :3], 4), expected), means[0, :, 3, :3]
    no_jepa = np.array([[.7275, .5954, .5354], [.7281, .5981, .5426], [.7281, .5971, .5469]])
    assert np.array_equal(np.round(means[0, 0, :3, :3], 4), no_jepa), means[0, 0, :3, :3]
    dump_csv('extended_summary.csv', summary)
    dump_csv('extended_fold_metrics.csv', fold_rows)
    dump_csv('extended_validation_thresholds.csv', validation)
    contrast, longitudinal = [], []
    for pi, policy in enumerate(POLICIES):
        for aa, bb in [(1, 0), (2, 0), (2, 1), (2, 3)]:
            delta = boot_means[:, pi, aa] - boot_means[:, pi, bb]
            ci_d = np.nanquantile(delta, [.025, .975], axis=0)
            for ti, time in enumerate(TIMES):
                for ki, key in enumerate(METRICS[:14]):
                    contrast.append(dict(policy=policy, seed=44,
                                         comparison=LABELS[aa]+' minus '+LABELS[bb], timepoint=time, metric=key,
                                         delta=float(means[pi, aa, ti, ki]-means[pi, bb, ti, ki]),
                                         ci_low=float(ci_d[0, ti, ki]), ci_high=float(ci_d[1, ti, ki])))
        for mi, model in enumerate(LABELS):
            delta = boot_means[:, pi, mi, 2] - boot_means[:, pi, mi, 0]
            ci_d = np.nanquantile(delta, [.025, .975], axis=0)
            for ki, key in enumerate(METRICS[:14]):
                longitudinal.append(dict(policy=policy, seed=44, model=model,
                                          comparison='T2 minus T0', metric=key,
                                          delta=float(means[pi, mi, 2, ki]-means[pi, mi, 0, ki]),
                                          ci_low=float(ci_d[0, ki]), ci_high=float(ci_d[1, ki])))
    dump_csv('extended_paired_bootstrap.csv', contrast)
    dump_csv('extended_longitudinal.csv', longitudinal)
    # DCA uses the same ensemble score at each hypothetical utility threshold.
    dca = []
    grid = np.round(np.arange(.05, .801, .01), 2)
    for mi, model in enumerate(LABELS):
        scores = []
        for ti in range(3):
            pred = pbar[mi, ti, :, None] >= grid
            scores.append(y[:, None]*pred - (1-y[:, None])*pred*grid/(1-grid))
        scores.append(np.mean(scores, axis=0))
        for ti, time in enumerate(TIMES):
            score = scores[ti]
            value = score.mean(axis=0)
            nb_boot = weights @ score / 175
            nb_ci = np.quantile(nb_boot, [.025, .975], axis=0)
            for gi, threshold in enumerate(grid):
                dca.append(dict(model=model, seed=44, timepoint=time, threshold=float(threshold),
                                net_benefit=float(value[gi]), patient_ci_low=float(nb_ci[0, gi]),
                                patient_ci_high=float(nb_ci[1, gi]),
                                treat_all=float(.32-.68*threshold/(1-threshold)), treat_none=0.0))
    dump_csv('extended_dca.csv', dca)
    table = [r'\begin{table*}[t]', r'\centering', r'\small',
             r'\caption{Extended operating-point metrics for seed 44 at probability threshold 0.5. Each patient score is the arithmetic mean of probabilities from the five fold models before any metric is calculated. All landmarks contain the same 175 holdout patients (56 pCR-positive). Macro equally averages the three landmark metric values. PPV and NPV denote positive and negative predictive value; BA is balanced accuracy.}',
             r'\label{tab:extended_operating}', r'\begin{tabular}{llrrrrrr}', r'\toprule',
             r'Model & Stage & Sens. & Spec. & PPV & NPV & BA & MCC \\', r'\midrule']
    for mi, model in enumerate(LABELS[:3]):
        for ti, time in enumerate(TIMES):
            table.append(model+' & '+time+' & '+' & '.join(f'{means[0, mi, ti, METRICS.index(k)]:.3f}' for k in ['sensitivity', 'specificity', 'ppv', 'npv', 'balanced_accuracy', 'mcc'])+r' \\')
        if mi < 2:
            table.append(r'\midrule')
    table += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    (ROOT/'tables'/'extended_operating.tex').write_text('\n'.join(table)+'\n')
    table = [r'\begin{table*}[t]', r'\centering', r'\small',
             r'\caption{Operating-point sensitivity analysis for seed-44 test ensemble scores, shown as three-landmark metric means. The alternative threshold maximizes Youden index on concatenated, disjoint out-of-fold development predictions, separately at each landmark; the largest threshold breaks ties. Each development patient contributes one score from a model that did not train on that patient. OOF scores arise from single fold models, whereas test scores average five model probabilities: this distribution mismatch makes this a sensitivity analysis, not validated ensemble calibration. No test labels select thresholds.}',
             r'\label{tab:extended_threshold}', r'\begin{tabular}{llrrrr}', r'\toprule',
             r'Model & Policy & F1 & Sens. & Spec. & MCC \\', r'\midrule']
    for mi, model in enumerate(LABELS[:3]):
        for pi, policy in enumerate(['0.5', 'OOF validation Youden']):
            table.append(model+' & '+policy+' & '+' & '.join(f'{means[pi, mi, 3, METRICS.index(k)]:.3f}' for k in ['f1', 'sensitivity', 'specificity', 'mcc'])+r' \\')
        if mi < 2:
            table.append(r'\midrule')
    table += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    (ROOT/'tables'/'extended_threshold.tex').write_text('\n'.join(table)+'\n')
    provenance = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(), training_seeds=SEEDS,
                      bootstrap_rng_seed=BOOTSTRAP_RNG, bootstrap_replicates=B, models=LABELS,
                      metrics=METRICS, patients=175, positives=56, aggregate_only=True,
                      verified_completed_receipts=20, verified_registered_report_identities=60,
                      verified_manifest_split_exclusion=True,
                      threshold_selection='one per model and landmark; Youden on disjoint OOF validation scores; largest threshold resolves ties',
                      validation_availability=[801, 760, 700],
                      threshold_caveat='single-model OOF score distribution differs from five-model ensemble test scores; sensitivity analysis, not ensemble calibration',
                      aggregation='seed44 only; arithmetic mean of five fold probabilities per patient and landmark -> metric; Macro equally averages T0/T1/T2 metric values',
                      bootstrap_scope='fixed trained models and thresholds; 2000 shared patient resamples across models and landmarks; excludes retraining, seed-selection, and threshold-selection uncertainty',
                      fold_metric_role='source consistency audit only; never used to derive ensemble performance', sources=identities)
    (ROOT/'data'/'extended_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    note = ['# Seed44 patient probability ensemble analysis', '',
            'All 20 completed receipts, 60 registered report byte/mtime identities, and 20 manifest references passed. Seed44 configs and all landmark/fold test label and ID alignments passed. Every validation prediction was checked against its source manifest: its patient belongs to that model fold validation split, not its training or test split. All test predictions belong to the fixed manifest test split and labels match. Raw IDs, prediction rows, and probability arrays are never exported.',
            'For every model, landmark, and patient, five fold probabilities are arithmetically averaged before metrics, patient bootstrap contrasts, temporal differences, or decision curves. This is not a mean of fold metrics. Macro averages three landmark metric values, not probabilities across landmarks.',
            'The 18 quantities comprise 14 metrics and TP/FP/FN/TN. Landmark confusion counts are integer counts from one 175-patient ensemble prediction vector; Macro counts are the arithmetic mean of landmark counts. Individual fold metrics are audit-only.',
            'All landmark vectors use exactly the same 175 holdout patients and 56 positive labels (prevalence 0.32). Every model has disjoint OOF development coverage 801/760/700 at T0/T1/T2 and no development/test patient overlap.',
            'There are 2,000 shared nonparametric patient bootstrap draws (RNG 20261102), reused across models and landmarks. Percentile 95% intervals condition on the fixed trained models and chosen thresholds; they exclude retraining, seed-selection, and threshold-selection uncertainty. They do not quantify variation across training seeds.',
            'The OOF Youden analysis concatenates exactly one held-out-fold score per available development patient and selects one threshold per model/landmark. OOF scores come from single fold models, while test scores are five-model ensemble means. Their score distributions differ: this is an operating-point sensitivity analysis, not validated ensemble calibration.',
            'DCA uses a hypothetical pCR-positive decision and uncalibrated model scores; its net benefit is exploratory and does not establish clinical usefulness. No clinical intervention or benefit was evaluated.', '',
            '## Fixed 0.5 ensemble results (Macro)', '',
            '| Model | AUROC | AP | F1 | Accuracy | Sensitivity | Specificity | PPV | NPV | BA | MCC | Brier | NLL | ECE |',
            '|---|'+'---|'*13]
    for mi, model in enumerate(LABELS):
        note.append('| '+model+' | '+' | '.join(f'{means[0, mi, 3, METRICS.index(k)]:.4f}' for k in ['auroc', 'auprc', 'f1', 'accuracy', 'sensitivity', 'specificity', 'ppv', 'npv', 'balanced_accuracy', 'mcc', 'brier', 'nll', 'ece_10_equal_width'])+' |')
    note += ['', '## OOF validation-Youden ensemble results (Macro)', '',
             '| Model | F1 | Sensitivity | Specificity | MCC |', '|---|---|---|---|---|']
    for mi, model in enumerate(LABELS):
        note.append('| '+model+' | '+' | '.join(f'{means[1, mi, 3, METRICS.index(k)]:.4f}' for k in ['f1', 'sensitivity', 'specificity', 'mcc'])+' |')
    note += ['', '## Paired Macro contrasts (fixed 0.5)', '']
    for r in contrast:
        if r['policy'] == 'fixed_0.5' and r['timepoint'] == 'Macro' and r['metric'] in ['auroc', 'auprc', 'f1', 'brier', 'nll', 'ece_10_equal_width']:
            note.append(f"- {r['comparison']}, {r['metric']}: {r['delta']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}].")
    note += ['', '## T2 minus T0 paired intervals (fixed 0.5)', '']
    for r in longitudinal:
        if r['policy'] == 'fixed_0.5' and r['metric'] in ['auroc', 'auprc', 'f1']:
            note.append(f"- {r['model']} {r['metric']}: {r['delta']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}].")
    note += ['', '## Exploratory Macro decision-curve point estimates', '',
             '| Hypothetical threshold | NoJEPA | DirectReadout | ForecastAttention | ForecastMean | Treat all |',
             '|---|---|---|---|---|---|']
    dca_lookup = {(r['model'], r['timepoint'], r['threshold']): r for r in dca}
    for threshold in [.1, .2, .3, .4, .5, .6, .7, .8]:
        row = [f'{threshold:.2f}']
        row += [f"{dca_lookup[model, 'Macro', threshold]['net_benefit']:.4f}" for model in LABELS]
        row += [f"{dca_lookup[LABELS[0], 'Macro', threshold]['treat_all']:.4f}"]
        note.append('| '+' | '.join(row)+' |')
    note += ['', '## Interpretation boundaries', '',
             '- ForecastAttention has the largest Macro AUROC and AP among the principal variants, but DirectReadout has higher F1 at 0.5; ForecastMean has the largest F1 among all four variants.',
             '- Predictive ranking, classification at 0.5, calibration, and hypothetical utility can disagree. A ranking advantage does not establish calibration or clinical decision benefit.',
             '- ForecastAttention has lower Macro net-benefit point estimates than DirectReadout at hypothetical thresholds 0.20, 0.30, and 0.40; its net benefit at 0.50 is negative (-0.0305). It has higher point estimates at some higher thresholds, so there is no uniform utility advantage. These thresholds were not prespecified clinical decision rules.',
             '- These data use one selected training seed and an internal downstream holdout. Patient bootstrap precision does not support a claim of robustness across seeds or external clinical validation.',
             '- The temporal comparison conditions on 175 patients with all three landmarks. It is descriptive on the common complete-prefix cohort and cannot establish benefit across all development patients.']
    (ROOT/'notes'/'extended_metrics.md').write_text('\n'.join(note)+'\n')
    print(json.dumps(dict(verified_runs=20, verified_report_identities=verified_report_identities,
                          summary_rows=len(summary), fold_rows=len(fold_rows), dca_rows=len(dca),
                          paired_rows=len(contrast), longitudinal_rows=len(longitudinal),
                          macro_fixed_0_5={model: dict(zip(METRICS[:14], map(float, means[0, mi, 3, :14]))) for mi, model in enumerate(LABELS)}), indent=2))


if __name__ == '__main__':
    main()
