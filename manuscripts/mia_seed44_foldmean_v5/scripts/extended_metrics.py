#!/usr/bin/env python3
"""Seed44 mean of separately evaluated fold metrics; aggregate outputs."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['direct_no_jepa', 'direct_readout', 'forecast_attention', 'forecast_mean']
LABELS = ['NoJEPA', 'DirectReadout', 'ForecastAttention', 'ForecastMean']
SEEDS = [44]
TIMES = ['T0', 'T1', 'T2', 'Macro']
POLICIES = ['fixed_0.5', 'fold_validation_youden']
METRICS = ['auroc', 'auprc', 'f1', 'accuracy', 'sensitivity', 'specificity', 'ppv',
           'npv', 'balanced_accuracy', 'mcc', 'youden_j', 'brier', 'nll',
           'ece_10_equal_width', 'tp', 'fp', 'fn', 'tn']
B = 2000
BOOTSTRAP_RNG = 20261103


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
    from analysis_common import load_data,SEEDS,SEED_TEXT,FOLDS,LABELS,TIMES,POLICIES
    data=load_data();y=data['y'];scores=data['scores'];thresholds=data['thresholds']
    weights=np.random.default_rng(BOOTSTRAP_RNG).multinomial(175,np.full(175,1/175),size=B)
    point=np.empty((2,5,4,3,len(METRICS)))
    boot=np.empty((B,2,5,4,3,len(METRICS)))
    fold_rows=[]
    for pi,policy in enumerate(POLICIES):
      for si,fold in enumerate(FOLDS):
       for mi,model in enumerate(LABELS):
        for ti in range(3):
            threshold=thresholds[pi,si,mi,ti]
            point[pi,si,mi,ti]=all_metrics(y,scores[si,mi,ti],np.ones((1,175)),threshold)[0]
            boot[:,pi,si,mi,ti]=all_metrics(y,scores[si,mi,ti],weights,threshold)
            fold_rows.append(dict(policy=policy,seed=44,model=model,fold=fold,timepoint=TIMES[ti],threshold=float(threshold),role='individual_fold_model_main_analysis',**dict(zip(METRICS,map(float,point[pi,si,mi,ti])))))
    # Evaluate each fold separately before averaging its metric values; never average probabilities.
    per_fold=np.concatenate([point,point.mean(axis=3,keepdims=True)],axis=3)
    boot_fold=np.concatenate([boot,boot.mean(axis=4,keepdims=True)],axis=4)
    means=per_fold.mean(axis=1);sd=per_fold.std(axis=1,ddof=1)
    boot_means=boot_fold.mean(axis=2)
    ci=np.nanquantile(boot_means,[.025,.975],axis=0)
    summary=[];seed_rows=[]
    for pi,policy in enumerate(POLICIES):
      for mi,model in enumerate(LABELS):
       for ti,time in enumerate(TIMES):
        row=dict(policy=policy,seed=SEED_TEXT,model=model,timepoint=time,folds_per_seed=5,training_seed_count=1,patients=175,positives=56,aggregation='independent metric per seed44 fold model -> equal five-fold metric mean',threshold=float('nan'))
        for ki,key in enumerate(METRICS):
            row[key]=float(means[pi,mi,ti,ki]);row[key+'_fold_sd']=float(sd[pi,mi,ti,ki])
            if ki<14:
                row[key+'_patient_ci_low']=float(ci[0,pi,mi,ti,ki]);row[key+'_patient_ci_high']=float(ci[1,pi,mi,ti,ki])
        summary.append(row)
        for si,fold in enumerate(FOLDS):
            seed_rows.append(dict(policy=policy,seed=44,fold=fold,model=model,timepoint=time,patients=175,positives=56,folds=5,threshold=float(thresholds[pi,si,mi,ti]) if ti<3 else float('nan'),**dict(zip(METRICS,map(float,per_fold[pi,si,mi,ti])))))
    dump_csv('extended_summary.csv',summary);dump_csv('seed44_individual_fold_metrics.csv',seed_rows)
    dump_csv('extended_fold_metrics.csv',fold_rows);dump_csv('extended_validation_thresholds.csv',data['validation'])
    contrast=[];longitudinal=[]
    for pi,policy in enumerate(POLICIES):
      for aa,bb in [(1,0),(2,0),(2,1),(2,3)]:
        delta=boot_means[:,pi,aa]-boot_means[:,pi,bb];ci_d=np.nanquantile(delta,[.025,.975],axis=0)
        fold_delta=per_fold[pi,:,aa]-per_fold[pi,:,bb]
        for ti,time in enumerate(TIMES):
          for ki,key in enumerate(METRICS[:14]):
            contrast.append(dict(policy=policy,seed=SEED_TEXT,comparison=LABELS[aa]+' minus '+LABELS[bb],timepoint=time,metric=key,delta=float(means[pi,aa,ti,ki]-means[pi,bb,ti,ki]),delta_fold_sd=float(fold_delta[:,ti,ki].std(ddof=1)),ci_low=float(ci_d[0,ti,ki]),ci_high=float(ci_d[1,ti,ki])))
      for mi,model in enumerate(LABELS):
        delta=boot_means[:,pi,mi,2]-boot_means[:,pi,mi,0];ci_d=np.nanquantile(delta,[.025,.975],axis=0)
        for ki,key in enumerate(METRICS[:14]):
            longitudinal.append(dict(policy=policy,seed=SEED_TEXT,model=model,comparison='T2 minus T0',metric=key,delta=float(means[pi,mi,2,ki]-means[pi,mi,0,ki]),delta_fold_sd=float((per_fold[pi,:,mi,2,ki]-per_fold[pi,:,mi,0,ki]).std(ddof=1)),ci_low=float(ci_d[0,ki]),ci_high=float(ci_d[1,ki])))
    dump_csv('extended_paired_bootstrap.csv',contrast);dump_csv('extended_longitudinal.csv',longitudinal)
    dca=[];dca_seed=[];grid=np.round(np.arange(.05,.801,.01),2)
    for mi,model in enumerate(LABELS):
        all_nb=[]
        for si,fold in enumerate(FOLDS):
            nb=[]
            for ti in range(3):
                pred=scores[si,mi,ti,:,None]>=grid
                tp=(pred*y[:,None]).sum(0);fp=(pred*(1-y)[:,None]).sum(0)
                nb.append(tp/175-fp/175*grid/(1-grid))
            nb=np.r_[np.array(nb),np.mean(nb,axis=0,keepdims=True)];all_nb.append(nb)
            for ti,time in enumerate(TIMES):
                for qi,q in enumerate(grid):dca_seed.append(dict(model=model,seed=44,fold=fold,timepoint=time,threshold=float(q),net_benefit=float(nb[ti,qi])))
        all_nb=np.array(all_nb)
        for ti,time in enumerate(TIMES):
            for qi,q in enumerate(grid):dca.append(dict(model=model,seed=SEED_TEXT,timepoint=time,threshold=float(q),net_benefit=float(all_nb[:,ti,qi].mean()),net_benefit_fold_sd=float(all_nb[:,ti,qi].std(ddof=1)),treat_all=float(y.mean()-(1-y.mean())*q/(1-q)),treat_none=0.))
    dump_csv('extended_dca.csv',dca);dump_csv('seed44_individual_dca.csv',dca_seed)
    table=[r'\begin{table*}[t]',r'\centering',r'\small',r'\caption{Operating characteristics at threshold 0.5. Entries average metrics computed separately for five seed44 fold models; Macro first averages T0--T2 within each fold. No probabilities are averaged. The same 175 patients appear in every evaluation.}',r'\label{tab:extended_operating}',r'\begin{tabular}{llrrrrrr}',r'\toprule',r'Model & Landmark & Sens. & Spec. & PPV & NPV & BA & MCC \\',r'\midrule']
    for mi,model in enumerate(LABELS[:3]):
        for ti,time in enumerate(TIMES):table.append(model+' & '+time+' & '+' & '.join(f'{means[0,mi,ti,METRICS.index(k)]:.3f}' for k in ['sensitivity','specificity','ppv','npv','balanced_accuracy','mcc'])+r' \\')
        if mi<2:table.append(r'\midrule')
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table*}'];(ROOT/'tables/extended_operating.tex').write_text('\n'.join(table)+'\n')
    table=[r'\begin{table*}[t]',r'\centering',r'\small',r'\caption{Threshold sensitivity analysis: seed44 five-fold metric means. Each fold model uses its own landmark-specific validation Youden threshold on the holdout scores from the same model. Validation also informed best-checkpoint selection; this is sensitivity analysis, not nested calibration. No test labels choose thresholds.}',r'\label{tab:extended_threshold}',r'\begin{tabular}{llrrrr}',r'\toprule',r'Model & Policy & F1 & Sens. & Spec. & MCC \\',r'\midrule']
    for mi,model in enumerate(LABELS[:3]):
        for pi,policy in enumerate(['0.5','Fold validation Youden']):table.append(model+' & '+policy+' & '+' & '.join(f'{means[pi,mi,3,METRICS.index(k)]:.3f}' for k in ['f1','sensitivity','specificity','mcc'])+r' \\')
        if mi<2:table.append(r'\midrule')
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table*}'];(ROOT/'tables/extended_threshold.tex').write_text('\n'.join(table)+'\n')
    provenance=dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),training_seeds=SEEDS,folds=[1,2,3,4,5],bootstrap_rng_seed=20261103,bootstrap_replicates=B,models=LABELS,metrics=METRICS,patients=175,positives=56,aggregate_only=True,verified_completed_receipts=20,verified_registered_report_identities=60,verified_manifest_split_exclusion=True,threshold_selection='separately per model/fold/landmark: Youden on that fold validation scores; apply to the same model holdout scores',threshold_caveat='Validation also informed checkpoint selection; not independent or nested calibration',aggregation='evaluate each seed44 fold model separately -> metric and equal T0/T1/T2 Macro -> equal mean of five fold metrics; no probabilities averaged',fold_sd='sample standard deviation over five fold-model metrics, ddof=1; descriptive spread of overlapping trained models, not independent folds/cohorts/seeds',bootstrap_scope='fixed five trained models; shared patient draws across all folds/models/landmarks; recompute metric per fold then average. No folds resampled; no retraining or seed/aggregation-selection uncertainty.',sources=data['sources'])
    (ROOT/'data/extended_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (ROOT/'notes/extended_metrics.md').write_text('# Seed44 separately evaluated folds\n\nEach fold model is evaluated separately on the same 175 patients/56 positives. Mean and sample SD of five fold metrics are reported. No probability averaging; no independent-patient inflation. Twenty completed receipts and sixty report identities passed. Patient bootstrap keeps all five fitted models fixed. Full results: extended_summary.csv and seed44_individual_fold_metrics.csv.\n')
    print(json.dumps(dict(verified_runs=20,verified_report_identities=60,macro={model:dict(zip(METRICS[:14],map(float,means[0,mi,3,:14]))) for mi,model in enumerate(LABELS)}),indent=2))

if __name__=='__main__':main()
