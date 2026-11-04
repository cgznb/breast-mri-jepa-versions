#!/usr/bin/env python3
"""Rebuild aggregate-only manuscript assets from receipt-backed seed44 predictions."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone
import numpy as np

HERE = Path(__file__).resolve().parents[1]
SUITE = Path('/projects/xts0030/public/cxx/medium976_t0_t2_no_tf_cv5_multiseed_v1')
MODELS = ['direct_no_jepa', 'direct_readout', 'forecast_attention', 'forecast_mean']
LABELS = ['NoJEPA', 'DirectReadout', 'ForecastAttention', 'ForecastMean']
METRICS = ['auroc', 'auprc', 'f1', 'nll', 'brier', 'ece_10_equal_width']
TIMES = ['T0', 'T1', 'T2', 'Macro']
BOOTSTRAPS = 2000
BOOTSTRAP_RNG = 20261102


def write_csv(name, rows):
    with (HERE / 'data' / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def weighted_metrics(y, p, weights):
    """AUROC (including ties), average precision, and F1 on patient resamples."""
    order = np.argsort(-p, kind='stable')
    yy, pp = y[order], p[order]
    ww = weights[:, order]
    tp = np.cumsum(ww * yy, axis=1)
    fp = np.cumsum(ww * (1 - yy), axis=1)
    ends = np.r_[np.flatnonzero(np.diff(pp)), len(pp) - 1]
    tp, fp = tp[:, ends], fp[:, ends]
    positives = (weights * y).sum(axis=1)
    negatives = weights.sum(axis=1) - positives
    tpr, fpr = tp / positives[:, None], fp / negatives[:, None]
    prev_tpr = np.c_[np.zeros(len(weights)), tpr[:, :-1]]
    prev_fpr = np.c_[np.zeros(len(weights)), fpr[:, :-1]]
    auc = ((tpr + prev_tpr) * (fpr - prev_fpr) / 2).sum(axis=1)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
    ap = (precision * (tpr - prev_tpr)).sum(axis=1)
    pred = (p >= .5).astype(float)
    true_pos = (weights * y * pred).sum(axis=1)
    f1 = 2 * true_pos / (positives + (weights * pred).sum(axis=1))
    return np.stack([auc, ap, f1], axis=-1)


matrix = json.loads((SUITE / 'matrix.json').read_text())
assert matrix['status'] == 'completed' and matrix['test_visible']
fold_values = np.empty((4, 5, 3, 6))
probabilities = np.empty((4, 5, 3, 175))
source_records, training_rows = [], []
patient_ids = labels = None
for mi, model in enumerate(MODELS):
    for fi in range(5):
        run = SUITE / 'runs' / f'{model}_seed44_fold{fi + 1}'
        receipt = json.loads((run / 'evaluation_done.json').read_text())
        assert receipt['completed'] is True
        for identity in receipt['reports']:
            stat = Path(identity['path']).stat()
            assert (stat.st_size, stat.st_mtime_ns) == (identity['bytes'], identity['mtime_ns'])
        report_path = run / 'evaluation_best_test.json'
        report = json.loads(report_path.read_text())
        assert report['split'] == 'test' and report['downstream_holdout']
        assert report['independent_test'] is False and report['f1_threshold'] == .5
        config = json.loads((run / 'config.json').read_text())
        assert config['seed'] == 44
        train = json.loads((run / 'train_report.json').read_text())
        training_rows.append(dict(model=LABELS[mi], fold=fi + 1,
                                  best_epoch=train['best_epoch'], stopped_epoch=train['completed_epochs']))
        source_records.append(dict(model=model, fold=fi + 1, report=str(report_path),
                                   report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                                   receipt_sha256=hashlib.sha256((run / 'evaluation_done.json').read_bytes()).hexdigest()))
        for ti in range(3):
            rows = sorted((r for r in report['rows'] if r['landmark'] == ti), key=lambda r:r['patient_key'])
            ids = [r['patient_key'] for r in rows]
            y = np.array([r['pcr'] for r in rows])
            p = np.array([r['pcr_probability'] for r in rows])
            if patient_ids is None:
                patient_ids, labels = ids, y
            assert ids == patient_ids and np.array_equal(y, labels)
            assert len(rows) == 175 and y.sum() == 56
            probabilities[mi, fi, ti] = p
            m = report['per_landmark'][f'T{ti}']
            fold_values[mi, fi, ti] = [m[k] for k in METRICS]
            check = weighted_metrics(y, p, np.ones((1, len(y))))[0]
            assert np.max(np.abs(check - fold_values[mi, fi, ti, :3])) < 1e-12

rng = np.random.default_rng(BOOTSTRAP_RNG)
weights = rng.multinomial(175, np.full(175, 1/175), size=BOOTSTRAPS)
assert np.all((weights * labels).sum(axis=1) > 0)
boot = np.empty((BOOTSTRAPS, 4, 5, 3, 3))
for mi in range(4):
    for fi in range(5):
        for ti in range(3):
            boot[:, mi, fi, ti] = weighted_metrics(labels, probabilities[mi, fi, ti], weights)
means = fold_values.mean(axis=1)
sds = fold_values.std(axis=1, ddof=1)
means = np.concatenate([means, means.mean(axis=1, keepdims=True)], axis=1)
fold_macros = fold_values.mean(axis=2)
sds = np.concatenate([sds, fold_macros.std(axis=1, ddof=1)[:, None]], axis=1)
boot_mean = boot.mean(axis=2)
boot_mean = np.concatenate([boot_mean, boot_mean.mean(axis=2, keepdims=True)], axis=2)
intervals = np.quantile(boot_mean, [.025, .975], axis=0)
summary = []
for mi, label in enumerate(LABELS):
    for ti, time in enumerate(TIMES):
        r = dict(model=label, training_seed=44, timepoint=time, folds=5, patients=175, positives=56)
        for ki, metric in enumerate(METRICS):
            r[metric] = float(means[mi, ti, ki])
            r[metric + '_fold_sd'] = float(sds[mi, ti, ki])
            if ki < 3:
                r[metric + '_patient_ci_low'] = float(intervals[0, mi, ti, ki])
                r[metric + '_patient_ci_high'] = float(intervals[1, mi, ti, ki])
        summary.append(r)
write_csv('seed44_summary.csv', summary)
write_csv('training_epochs.csv', training_rows)
fold_rows = [dict(model=LABELS[mi], fold=fi + 1, timepoint=TIMES[ti],
                  **dict(zip(METRICS, map(float, fold_values[mi, fi, ti]))))
             for mi in range(4) for fi in range(5) for ti in range(3)]
write_csv('seed44_fold_metrics.csv', fold_rows)
contrasts = []
for a, b in [(1,0), (2,0), (2,1), (2,3)]:
    for ti,time in enumerate(TIMES):
        d = boot_mean[:, a, ti] - boot_mean[:, b, ti]
        ci = np.quantile(d, [.025,.975], axis=0)
        for ki,metric in enumerate(METRICS[:3]):
            contrasts.append(dict(comparison=LABELS[a]+' minus '+LABELS[b], timepoint=time,
                                  metric=metric, delta=float(means[a,ti,ki]-means[b,ti,ki]),
                                  ci_low=float(ci[0,ki]), ci_high=float(ci[1,ki])))
write_csv('paired_patient_bootstrap.csv', contrasts)

main = [r'\begin{table}[t]', r'\centering',
        r'\caption{Seed44 performance on the fixed 175-patient internal holdout. Values are means of five separately evaluated fold models; AP denotes average precision and F1 uses a threshold of 0.5. Macro averages the three landmarks. These are not probability-ensemble scores.}',
        r'\label{tab:main}', r'\small', r'\begin{tabular}{llrrr}', r'\toprule',
        r'Model & Landmark & AUROC & AP & F1 \\', r'\midrule']
for mi,label in enumerate(LABELS[:3]):
    for ti,time in enumerate(TIMES):
        main.append(label + ' & ' + time + ' & ' + ' & '.join(f'{v:.4f}' for v in means[mi,ti,:3]) + r' \\')
    if mi<2: main.append(r'\midrule')
main += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
(HERE/'tables'/'main_results.tex').write_text('\n'.join(main)+'\n')

cal = [r'\begin{table}[t]',r'\centering',
       r'\caption{Calibration and probability accuracy, averaged over landmarks and five fold models for seed44. Lower is better. ECE uses ten equal-width bins.}',
       r'\label{tab:calibration}',r'\begin{tabular}{lrrr}',r'\toprule',r'Model & NLL & Brier & ECE \\',r'\midrule']
for mi,label in enumerate(LABELS[:3]):
    cal.append(label+' & '+' & '.join(f'{x:.4f}' for x in means[mi,3,3:])+r' \\')
cal += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(HERE/'tables'/'calibration.tex').write_text('\n'.join(cal)+'\n')

sup = [r'\begin{table}[t]',r'\centering',
       r'\caption{Supplementary pooling comparison using seed44. ForecastMean and ForecastAttention share forecast-based pCR readout and future-state supervision, and differ in spatial pooling.}',
       r'\label{tab:pooling}',r'\begin{tabular}{llrrr}',r'\toprule',r'Model & Landmark & AUROC & AP & F1 \\',r'\midrule']
for mi in [3,2]:
    for ti,time in enumerate(TIMES):
        sup.append(LABELS[mi]+' & '+time+' & '+' & '.join(f'{x:.4f}' for x in means[mi,ti,:3])+r' \\')
    if mi==3:sup.append(r'\midrule')
sup += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(HERE/'tables'/'pooling.tex').write_text('\n'.join(sup)+'\n')

ci_table = [r'\begin{table}[t]',r'\centering',
            r'\caption{Exploratory paired patient-bootstrap intervals for three-landmark mean differences (2,000 shared patient resamples). Intervals condition on the fixed seed44 models; they exclude retraining and seed-selection uncertainty.}',
            r'\label{tab:bootstrap}',r'\small',r'\begin{tabular}{llr}',r'\toprule',r'Comparison & Metric & Difference [95\% interval] \\',r'\midrule']
for a,b in [(1,0),(2,0),(2,1)]:
    for metric in METRICS[:3]:
        r=next(r for r in contrasts if r['comparison']==LABELS[a]+' minus '+LABELS[b] and r['timepoint']=='Macro' and r['metric']==metric)
        ci_table.append(LABELS[a]+' -- '+LABELS[b]+' & '+{'auroc':'AUROC','auprc':'AP','f1':'F1'}[metric]+' & '+f"{r['delta']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"+r' \\')
ci_table += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(HERE/'tables'/'bootstrap.tex').write_text('\n'.join(ci_table)+'\n')

payload = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(), training_seed=44,
               bootstrap_replicates=BOOTSTRAPS, bootstrap_rng_seed=BOOTSTRAP_RNG,
               cohort=dict(patients=175,positives=56), aggregation='metric per fold -> five-fold mean -> equal landmark macro',
               no_probability_ensembling=True, patient_identifiers_exported=False, sources=source_records,
               summary=summary, contrasts=contrasts)
(HERE/'data'/'results_provenance.json').write_text(json.dumps(payload,indent=2)+'\n')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit('Install matplotlib to rebuild the publication figures.')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
colors=['#687782','#dc8b36','#1d7486']
fig,axs=plt.subplots(1,3,figsize=(10,3.1))
for ki,ax in enumerate(axs):
    for mi in range(3):
        ax.plot(range(3),means[mi,:3,ki],marker='o',color=colors[mi],label=LABELS[mi],linewidth=1.7)
    ax.set_xticks(range(3),TIMES[:3]);ax.set_ylabel(['AUROC','Average precision','F1 at 0.5'][ki]);ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False)
axs[0].legend(frameon=False,fontsize=8)
fig.tight_layout();fig.savefig(HERE/'figures'/'landmark_performance.pdf',bbox_inches='tight')
fig.savefig(HERE/'figures'/'landmark_performance.png',dpi=220,bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(10,3.1))
for ki,ax in enumerate(axs):
    for mi in range(3):
        value=means[mi,3,ki]
        ax.errorbar(mi,value,yerr=[[value-intervals[0,mi,3,ki]],[intervals[1,mi,3,ki]-value]],fmt='o',color=colors[mi],capsize=4)
    ax.set_xticks(range(3),LABELS[:3],rotation=18,ha='right');ax.set_ylabel(['Macro AUROC','Macro AP','Macro F1'][ki]);ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(HERE/'figures'/'patient_bootstrap.pdf',bbox_inches='tight');plt.close(fig)
print(json.dumps({'audited_seed44_runs':20,'bootstrap_replicates':BOOTSTRAPS,'aggregate_rows':len(summary),'macro_contrasts':[r for r in contrasts if r['timepoint']=='Macro']},indent=2))
