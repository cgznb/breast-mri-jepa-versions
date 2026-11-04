#!/usr/bin/env python3
"""Render 14 metrics of seed44 fivefold probability ensembles from aggregate CSV."""
from pathlib import Path
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['NoJEPA', 'DirectReadout', 'ForecastAttention']
METRICS = [
    ('auroc', 'AUROC'),
    ('auprc', 'AP'),
    ('f1', 'F1'),
    ('accuracy', 'Accuracy'),
    ('sensitivity', 'Sensitivity'),
    ('specificity', 'Specificity'),
    ('ppv', 'PPV'),
    ('npv', 'NPV'),
    ('balanced_accuracy', 'Balanced accuracy'),
    ('mcc', 'MCC'),
    ('youden_j', 'Youden index'),
    ('nll', r'NLL $\downarrow$'),
    ('brier', r'Brier score $\downarrow$'),
    ('ece_10_equal_width', r'ECE $\downarrow$'),
]


def build_table():
    source = ROOT / 'data' / 'extended_summary.csv'
    selected = [r for r in csv.DictReader(source.open())
                if r['seed'] == '44' and r['policy'] == 'fixed_0.5'
                and r['timepoint'] == 'Macro' and r['model'] in MODELS]
    assert len(selected) == len(MODELS)
    lookup = {r['model']: r for r in selected}
    assert set(lookup) == set(MODELS)
    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        r'\footnotesize',
        r'\setlength{\tabcolsep}{5pt}',
        r'\caption{All 14 macro metrics for the principal variants: point estimate [95\% patient-bootstrap percentile interval]. Each patient\textquotesingle s five seed44 fold-model probabilities are averaged first; metrics are then computed and equally averaged over T0--T2. Threshold-dependent metrics use 0.5. Identical patient resamples preserve pairing across models and landmarks. Entries are marginal model intervals; paired differences are reported separately. Intervals condition on the fitted ensembles and exclude retraining and retrospective selection uncertainty. Downward arrows identify metrics for which lower is better.}',
        r'\label{tab:extended_macro}',
        r'\begin{tabular}{lccc}',
        r'\toprule',
        r'Metric & NoJEPA & DirectReadout & ForecastAttention \\',
        r'\midrule',
    ]
    audit = []
    for key, label in METRICS:
        cells = []
        for model in MODELS:
            r = lookup[model]
            value, lo, hi = (float(r[k]) for k in [key, key+'_patient_ci_low', key+'_patient_ci_high'])
            assert all(math.isfinite(v) for v in [value, lo, hi]) and lo <= hi
            cells.append(f'{value:.4f} [{lo:.4f}, {hi:.4f}]')
            audit.append(dict(model=model, metric=key, estimate=value, ci_low=lo, ci_high=hi))
        lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    output = ROOT / 'tables' / 'extended_macro_metrics.tex'
    output.write_text('\n'.join(lines)+'\n')
    provenance = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      filter=dict(seed='44', policy='fixed_0.5', timepoint='Macro'),
                      aggregation='patient-wise arithmetic mean of five seed44 fold-model probabilities -> metric -> equal T0/T1/T2 Macro',
                      interval_scope='patient resampling conditional on fixed fitted ensembles; no refitting or retrospective selection uncertainty',
                      model_count=3, metric_rows=14, reported_values=audit)
    (ROOT / 'data' / 'extended_macro_table_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(f'Wrote {output.name}: 14 metrics, 3 models, 42 estimates and intervals.')


if __name__ == '__main__':
    build_table()
