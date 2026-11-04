#!/usr/bin/env python3
"""Create manuscript tables from the verified two-seed extended analysis."""
from pathlib import Path
import csv,json

ROOT=Path(__file__).resolve().parents[1]
LABELS=['NoJEPA','DirectReadout','ForecastAttention','ForecastMean']
TIMES=['T0','T1','T2','Macro']
rows=[r for r in csv.DictReader((ROOT/'data/extended_summary.csv').open()) if r['seed']=='mean21_44' and r['policy']=='fixed_0.5']
lookup={(r['model'],r['timepoint']):r for r in rows}
contrasts=[r for r in csv.DictReader((ROOT/'data/extended_paired_bootstrap.csv').open()) if r['seed']=='mean21_44' and r['policy']=='fixed_0.5']

def table(name,caption,label,cols,header,data):
    lines=[r'\begin{table*}[t]',r'\centering',r'\small',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+cols+'}',r'\toprule',header+r' \\',r'\midrule']
    lines += [' & '.join(row)+r' \\' for row in data]
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (ROOT/'tables'/name).write_text('\n'.join(lines)+'\n')

def formatted(model,t,metrics):
    r=lookup[model,t];return [f'{float(r[k]):.4f}' for k in metrics]

table('main_results.tex','Performance on the fixed internal holdout (175 patients, 56 pCR positive). Each value averages five fold metrics within a seed, then equally averages seeds 21 and 44. Macro averages the three landmarks. AP denotes average precision; F1 uses threshold 0.5.','tab:main','llrrr','Model & Landmark & AUROC & AP & F1',[[m,t,*formatted(m,t,['auroc','auprc','f1'])] for m in LABELS[:3] for t in TIMES])
table('calibration.tex','Probability quality averaged across landmarks after fold and seed aggregation. Lower is better. ECE uses ten equal-width bins.','tab:calibration','lrrr','Model & NLL & Brier & ECE',[[m,*formatted(m,'Macro',['nll','brier','ece_10_equal_width'])] for m in LABELS[:3]])
table('pooling.tex','Supplementary pooling comparison using the same two-seed aggregation as the main analysis. ForecastMean and ForecastAttention share forecast readout and future-state supervision, differing in spatial pooling.','tab:pooling','llrrr','Model & Landmark & AUROC & AP & F1',[[m,t,*formatted(m,t,['auroc','auprc','f1'])] for m in [LABELS[3],LABELS[2]] for t in TIMES])
table('bootstrap.tex','Exploratory paired patient-bootstrap intervals for three-landmark mean differences (2,000 shared patient draws). Intervals condition on the fitted models and exclude retraining and seed-selection uncertainty.','tab:bootstrap','llr','Comparison & Metric & Difference [95\\% interval]',[[r['comparison'].replace(' minus ',' -- '),{'auroc':'AUROC','auprc':'AP','f1':'F1'}[r['metric']],f"{float(r['delta']):+.4f} [{float(r['ci_low']):+.4f}, {float(r['ci_high']):+.4f}]"] for r in contrasts if r['timepoint']=='Macro' and r['metric'] in ['auroc','auprc','f1'] and not r['comparison'].endswith('ForecastMean')])
with (ROOT/'data/two_seed_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
provenance=json.loads((ROOT/'data/extended_provenance.json').read_text())
provenance['summary']=rows;provenance['contrasts']=contrasts
(ROOT/'data/results_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Rebuilt main tables with equal averaging of seeds 21 and 44.')
