#!/usr/bin/env python3
"""Render tables for seed44 patient-wise fivefold probability ensembles."""
from pathlib import Path
import csv,hashlib,json

ROOT=Path(__file__).resolve().parents[1]
LABELS=['NoJEPA','DirectReadout','ForecastAttention','ForecastMean']
TIMES=['T0','T1','T2','Macro']
source=ROOT/'data/extended_summary.csv'
contrast_source=ROOT/'data/extended_paired_bootstrap.csv'
provenance=json.loads((ROOT/'data/extended_provenance.json').read_text())
assert provenance['training_seeds']==[44], 'Regenerate seed44 probability-ensemble analysis first.'
assert 'probab' in provenance['aggregation'].lower(), 'Expected probability-first aggregation.'
rows=[r for r in csv.DictReader(source.open()) if r['seed']=='44' and r['policy']=='fixed_0.5']
lookup={(r['model'],r['timepoint']):r for r in rows}
assert len(rows)==len(LABELS)*len(TIMES) and set(lookup)=={(m,t) for m in LABELS for t in TIMES}
assert all(int(r['patients'])==175 and int(r['positives'])==56 for r in rows)
contrasts=[r for r in csv.DictReader(contrast_source.open()) if r['seed']=='44' and r['policy']=='fixed_0.5']
significance_source=ROOT/'data/significance_tests.csv'
significance=list(csv.DictReader(significance_source.open()))
assert len(significance)==90 and all(r['seed']=='44' and r['policy']=='fixed_0.5' for r in significance)
assert sum(r['family']=='principal_macro_9' for r in significance)==9

def table(name,caption,label,cols,header,data):
    lines=[r'\begin{table*}[t]',r'\centering',r'\small',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+cols+'}',r'\toprule',header+r' \\',r'\midrule']
    lines += [' & '.join(row)+r' \\' for row in data]
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (ROOT/'tables'/name).write_text('\n'.join(lines)+'\n')

def formatted(model,t,metrics):
    r=lookup[model,t];return [f'{float(r[k]):.4f}' for k in metrics]

table('main_results.tex','Performance on the fixed internal holdout (175 patients, 56 pCR positive). For each patient and landmark, the five seed44 fold-model probabilities are averaged before computing any metric. Macro averages T0--T2 metrics. AP denotes average precision; F1 uses threshold 0.5. Entries are point estimates; paired differences, patient intervals, and multiplicity-adjusted significance are reported in Table~\\ref{tab:bootstrap}.','tab:main','llrrr','Model & Landmark & AUROC & AP & F1',[[m,t,*formatted(m,t,['auroc','auprc','f1'])] for m in LABELS[:3] for t in TIMES])
table('calibration.tex','Probability quality of the seed44 fivefold probability ensemble, averaged over T0--T2 metrics. Lower is better. ECE uses ten equal-width bins.','tab:calibration','lrrr','Model & NLL & Brier & ECE',[[m,*formatted(m,'Macro',['nll','brier','ece_10_equal_width'])] for m in LABELS[:3]])
table('pooling.tex','Supplementary pooling comparison using seed44 fivefold probability ensembles. ForecastMean and ForecastAttention share forecast readout and future-state supervision, differing in spatial pooling. Probabilities are averaged before metrics; Macro averages the three landmarks.','tab:pooling','llrrr','Model & Landmark & AUROC & AP & F1',[[m,t,*formatted(m,t,['auroc','auprc','f1'])] for m in [LABELS[3],LABELS[2]] for t in TIMES])
metric_labels=dict(zip(['auroc','auprc','f1','accuracy','sensitivity','specificity','ppv','npv','balanced_accuracy','mcc','youden_j','brier','nll','ece_10_equal_width'],['AUROC','AP','F1','Accuracy','Sensitivity','Specificity','PPV','NPV','Balanced accuracy','MCC','Youden index','Brier','NLL','ECE']))
def short_comparison(text):
    return text.replace('DirectReadout','Direct').replace('ForecastAttention','Attention').replace('ForecastMean','Mean').replace(' minus ',' -- ')
def pformat(value):
    value=float(value)
    return r'$<0.0001$' if value<.0001 else f'{value:.4f}'
def stat_cells(r):
    return [f"{float(r['delta']):+.4f} [{float(r['ci_low']):+.4f}, {float(r['ci_high']):+.4f}]",pformat(r['p_raw']),pformat(r['p_holm_family']),r['significance']]
key=r' $^{*}p_{\mathrm{Holm}}<0.05$, $^{**}p_{\mathrm{Holm}}<0.01$, $^{***}p_{\mathrm{Holm}}<0.001$; ns, not significant after the stated correction. Intervals are unadjusted 95\% patient-bootstrap intervals (2,000 shared draws) and need not agree with adjusted test decisions. All analyses condition on the fitted, retrospectively selected seed44 ensembles.'
primary=[r for r in significance if r['family']=='principal_macro_9']
table('bootstrap.tex',r'Exploratory principal Macro comparisons. AUROC uses paired DeLong tests with correlated landmark placements averaged per patient. AP and F1 use approximate, null-centered paired patient-bootstrap tests with 99,999 shared resamples. All $p$ values are two-sided. Holm adjustment covers all nine entries in this table. Direct and Attention abbreviate DirectReadout and ForecastAttention.'+key,'tab:bootstrap','llrrrc',r'Comparison & Metric & Difference [95\% CI] & $p$ & $p_{\mathrm{Holm}}$ & Sig.',[[short_comparison(r['comparison']),metric_labels[r['metric']],*stat_cells(r)] for r in primary])
secondary_caption=r'AUROC uses paired DeLong tests; other metrics use approximate, null-centered paired patient-bootstrap tests (99,999 shared resamples). All $p$ values are two-sided. $p_{\mathrm{Holm}}$ adjusts jointly over the 81 secondary tests in Tables~\ref{tab:significance_landmarks}--\ref{tab:significance_longitudinal}. Direct and Attention abbreviate DirectReadout and ForecastAttention.'
landmarks=[r for r in significance if r['kind']=='model' and r['timepoint']!='Macro']
table('significance_landmarks.tex','Exploratory landmark comparisons. '+secondary_caption+key,'tab:significance_landmarks','lllrrrc',r'Comparison & Landmark & Metric & Difference [95\% CI] & $p$ & $p_{\mathrm{Holm}}$ & Sig.',[[short_comparison(r['comparison']),r['timepoint'],metric_labels[r['metric']],*stat_cells(r)] for r in landmarks])
extended=[r for r in significance if r['kind']=='model' and r['timepoint']=='Macro' and r['family']=='secondary_81']
table('significance_extended.tex','Exploratory Macro operating and probability-quality comparisons. Differences are first-named minus second-named model; negative Brier, NLL, or ECE differences favor the first model. '+secondary_caption+key,'tab:significance_extended','llrrrc',r'Comparison & Metric & Difference [95\% CI] & $p$ & $p_{\mathrm{Holm}}$ & Sig.',[[short_comparison(r['comparison']),metric_labels[r['metric']],*stat_cells(r)] for r in extended])
pooling=[r for r in significance if r['kind']=='pooling']
table('significance_pooling.tex','Exploratory pooling comparisons: ForecastAttention minus ForecastMean. '+secondary_caption+key,'tab:significance_pooling','llrrrc',r'Landmark & Metric & Difference [95\% CI] & $p$ & $p_{\mathrm{Holm}}$ & Sig.',[[r['timepoint'],metric_labels[r['metric']],*stat_cells(r)] for r in pooling])
longitudinal=[r for r in significance if r['kind']=='longitudinal']
table('significance_longitudinal.tex','Exploratory paired T2 minus T0 comparisons. DeLong placements and bootstrap draws retain within-patient dependence between landmarks. '+secondary_caption+key,'tab:significance_longitudinal','llrrrc',r'Model & Metric & Difference [95\% CI] & $p$ & $p_{\mathrm{Holm}}$ & Sig.',[[r['comparison'].replace(' T2 minus T0',''),metric_labels[r['metric']],*stat_cells(r)] for r in longitudinal])
with (ROOT/'data/seed44_ensemble_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# A copied v2 export must not be mistaken for a current result in the v3 package.
(ROOT/'data/two_seed_summary.csv').unlink(missing_ok=True)
provenance['summary']=rows;provenance['contrasts']=contrasts
provenance['table_sources']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,contrast_source,significance_source]}
provenance['table_filter']={'seed':'44','policy':'fixed_0.5'}
(ROOT/'data/results_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Rebuilt main tables from seed44 patient-wise fivefold probability ensembles.')
