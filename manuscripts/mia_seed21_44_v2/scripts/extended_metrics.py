#!/usr/bin/env python3
"""Extended aggregate-only seeds 21/44 evaluation; no model or training mutation."""
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
SEEDS = [21, 44]
GROUPS = [(seed, mi, model) for seed in SEEDS for mi, model in enumerate(MODELS)]
OUT_GROUPS = [(seed, label) for seed in [21, 44, 'mean21_44'] for label in LABELS]
TIMES = ['T0', 'T1', 'T2', 'Macro']
POLICIES = ['fixed_0.5', 'validation_youden']
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
    matrix = json.loads((SUITE/'matrix.json').read_text())
    assert matrix['status'] == 'completed' and matrix['test_visible']
    probs = np.empty((8,5,3,175))
    thresholds = np.full((2,8,5,3),.5)
    identities, validation, fold_rows = [],[],[]
    patient_ids = y = None
    for gi,(seed, model_index, model) in enumerate(GROUPS):
        for fi in range(5):
            run=SUITE/'runs'/f'{model}_seed{seed}_fold{fi+1}'
            receipt=json.loads((run/'evaluation_done.json').read_text())
            assert receipt['completed'] is True
            for identity in receipt['reports']:
                st=Path(identity['path']).stat()
                assert (st.st_size,st.st_mtime_ns)==(identity['bytes'],identity['mtime_ns'])
            config=json.loads((run/'config.json').read_text()); assert config['seed']==seed
            test_path=run/'evaluation_best_test.json';val_path=run/'evaluation_best_val.json'
            test=json.loads(test_path.read_text()); val=json.loads(val_path.read_text())
            assert test['split']=='test' and test['downstream_holdout'] and test['independent_test'] is False
            assert val['split']=='val'
            identities.append(dict(model=LABELS[model_index],seed=seed,fold=fi+1,
                                   test_report=str(test_path),test_sha256=hashlib.sha256(test_path.read_bytes()).hexdigest(),
                                   validation_report=str(val_path),validation_sha256=hashlib.sha256(val_path.read_bytes()).hexdigest()))
            for ti in range(3):
                rows=sorted((r for r in test['rows'] if r['landmark']==ti),key=lambda r:r['patient_key'])
                ids=[r['patient_key'] for r in rows]
                yy=np.array([r['pcr'] for r in rows]); pp=np.array([r['pcr_probability'] for r in rows])
                if patient_ids is None: patient_ids,y=ids,yy
                assert ids==patient_ids and np.array_equal(yy,y) and len(yy)==175 and yy.sum()==56
                probs[gi,fi,ti]=pp
                vr=[r for r in val['rows'] if r['landmark']==ti]
                vy=np.array([r['pcr'] for r in vr]);vp=np.array([r['pcr_probability'] for r in vr])
                threshold,j=select_youden(vy,vp);thresholds[1,gi,fi,ti]=threshold
                validation.append(dict(model=LABELS[model_index],seed=seed,fold=fi+1,timepoint=TIMES[ti],
                                       validation_patients=len(vy),validation_positives=int(vy.sum()),
                                       threshold=threshold,validation_youden_j=j,tie_rule='largest threshold'))
                m=all_metrics(y,pp,np.ones((1,175)))[0]
                for key in ['auroc','auprc','f1','brier','nll','ece_10_equal_width']:
                    assert abs(m[METRICS.index(key)]-test['per_landmark'][TIMES[ti]][key])<1e-12,(key,seed,model_index,fi,ti)
    weights=np.random.default_rng(BOOTSTRAP_RNG).multinomial(175,np.full(175,1/175),size=B)
    assert ((weights*y).sum(axis=1)>0).all() and ((weights*(1-y)).sum(axis=1)>0).all()
    point=np.empty((2,8,5,3,len(METRICS)))
    boot=np.empty((B,2,8,5,3,len(METRICS)))
    for pi,policy in enumerate(POLICIES):
        for mi,(seed,model_index,_) in enumerate(GROUPS):
            model=LABELS[model_index]
            for fi in range(5):
                for ti in range(3):
                    p=probs[mi,fi,ti];threshold=thresholds[pi,mi,fi,ti]
                    point[pi,mi,fi,ti]=all_metrics(y,p,np.ones((1,175)),threshold)[0]
                    boot[:,pi,mi,fi,ti]=all_metrics(y,p,weights,threshold)
                    fold_rows.append(dict(policy=policy,seed=seed,model=model,fold=fi+1,timepoint=TIMES[ti],threshold=float(threshold),
                                          **dict(zip(METRICS,map(float,point[pi,mi,fi,ti])))))
    # Average fold metrics before landmarks; never average patient probabilities.
    means=np.nanmean(point,axis=2);sds=np.nanstd(point,axis=2,ddof=1)
    means=np.concatenate([means,means.mean(axis=2,keepdims=True)],axis=2)
    macro_fold=point.mean(axis=3)
    sds=np.concatenate([sds,macro_fold.std(axis=2,ddof=1)[:,:,None]],axis=2)
    boot_means=np.nanmean(boot,axis=3)
    boot_means=np.concatenate([boot_means,np.nanmean(boot_means,axis=3,keepdims=True)],axis=3)
    # Original eight groups are [seed21 models, seed44 models]. Append equal seed means.
    combined_means=means.reshape(2,2,4,4,len(METRICS)).mean(axis=1)
    seed_sd=means.reshape(2,2,4,4,len(METRICS)).std(axis=1,ddof=1)
    means=np.concatenate([means,combined_means],axis=1)
    sds=np.concatenate([sds,np.full_like(combined_means,np.nan)],axis=1)
    combined_boot=boot_means.reshape(B,2,2,4,4,len(METRICS)).mean(axis=2)
    boot_means=np.concatenate([boot_means,combined_boot],axis=2)
    ci=np.nanquantile(boot_means,[.025,.975],axis=0)
    summary=[]
    for pi,policy in enumerate(POLICIES):
        for mi,(seed,model) in enumerate(OUT_GROUPS):
            for ti,time in enumerate(TIMES):
                row=dict(policy=policy,seed=seed,model=model,timepoint=time,folds_per_seed=5,training_seed_count=2 if seed=='mean21_44' else 1,patients=175,positives=56)
                for ki,key in enumerate(METRICS):
                    row[key]=float(means[pi,mi,ti,ki]);row[key+'_fold_sd']=float(sds[pi,mi,ti,ki])
                    row[key+'_between_seed_sd']=float(seed_sd[pi,mi-8,ti,ki]) if mi>=8 else float('nan')
                    if ki<14:
                        row[key+'_patient_ci_low']=float(ci[0,pi,mi,ti,ki]);row[key+'_patient_ci_high']=float(ci[1,pi,mi,ti,ki])
                summary.append(row)
    dump_csv('extended_summary.csv',summary);dump_csv('extended_fold_metrics.csv',fold_rows)
    dump_csv('extended_validation_thresholds.csv',validation)
    contrast=[];longitudinal=[]
    for pi,policy in enumerate(POLICIES):
        for si,seed in enumerate([21,44,'mean21_44']):
            offset=si*4
            for aa,bb in [(1,0),(2,0),(2,1),(2,3)]:
                a,b=offset+aa,offset+bb
                delta=boot_means[:,pi,a]-boot_means[:,pi,b]
                ci_d=np.nanquantile(delta,[.025,.975],axis=0)
                for ti,time in enumerate(TIMES):
                    for ki,key in enumerate(METRICS[:14]):
                        contrast.append(dict(policy=policy,seed=seed,comparison=LABELS[aa]+' minus '+LABELS[bb],timepoint=time,metric=key,
                                             delta=float(means[pi,a,ti,ki]-means[pi,b,ti,ki]),
                                             ci_low=float(ci_d[0,ti,ki]),ci_high=float(ci_d[1,ti,ki])))
        for mi,(seed,model) in enumerate(OUT_GROUPS):
            delta=boot_means[:,pi,mi,2]-boot_means[:,pi,mi,0]
            ci_d=np.nanquantile(delta,[.025,.975],axis=0)
            for ki,key in enumerate(METRICS[:14]):
                longitudinal.append(dict(policy=policy,seed=seed,model=model,comparison='T2 minus T0',metric=key,
                                          delta=float(means[pi,mi,2,ki]-means[pi,mi,0,ki]),
                                          ci_low=float(ci_d[0,ki]),ci_high=float(ci_d[1,ki])))
    dump_csv('extended_paired_bootstrap.csv',contrast);dump_csv('extended_longitudinal.csv',longitudinal)
    # Decision curves are evaluated at specified probability thresholds, not Youden thresholds.
    dca=[];grid=np.round(np.arange(.05,.801,.01),2)
    for model_index,model in enumerate(LABELS):
        per_fold=np.empty((2,5,3,len(grid)))
        patient_score=np.empty((2,5,3,175,len(grid)))
        for si in range(2):
            gi=si*4+model_index
            for fi in range(5):
                for ti in range(3):
                    pred=probs[gi,fi,ti,:,None]>=grid
                    score=y[:,None]*pred-(1-y[:,None])*pred*grid/(1-grid)
                    patient_score[si,fi,ti]=score;per_fold[si,fi,ti]=score.mean(axis=0)
        for si,seed in enumerate([21,44,'mean21_44']):
            for ti,time in enumerate(TIMES):
                fold_nb=per_fold[:,:,ti] if ti<3 else per_fold.mean(axis=2)
                score=patient_score[:,:,ti] if ti<3 else patient_score.mean(axis=2)
                seed_nb=fold_nb.mean(axis=1)
                if si<2:
                    value=seed_nb[si];sd=fold_nb[si].std(axis=0,ddof=1)
                    scores=score[si].mean(axis=0);between_sd=np.full(len(grid),np.nan)
                else:
                    value=seed_nb.mean(axis=0);sd=np.full(len(grid),np.nan)
                    scores=score.mean(axis=1).mean(axis=0);between_sd=seed_nb.std(axis=0,ddof=1)
                nb_boot=weights@scores/175
                nb_ci=np.quantile(nb_boot,[.025,.975],axis=0)
                for gi,t in enumerate(grid):
                    dca.append(dict(model=model,seed=seed,timepoint=time,threshold=float(t),net_benefit=float(value[gi]),
                                    net_benefit_fold_sd=float(sd[gi]),net_benefit_between_seed_sd=float(between_sd[gi]),
                                    patient_ci_low=float(nb_ci[0,gi]),patient_ci_high=float(nb_ci[1,gi]),
                                    treat_all=float(.32-.68*t/(1-t)),treat_none=0.0))
    dump_csv('extended_dca.csv',dca)
    # Tables: detailed fixed operating point and validation-only threshold sensitivity analysis.
    table=[r'\begin{table*}[t]',r'\centering',r'\small',
           r'\caption{Extended threshold-dependent metrics at probability threshold 0.5. Values first average five fold metrics within each seed and then equally average seeds 21 and 44 on the same 175 holdout patients; Macro equally averages landmarks. PPV and NPV denote positive and negative predictive value; BA is balanced accuracy.}',
           r'\label{tab:extended_operating}',r'\begin{tabular}{llrrrrrr}',r'\toprule',
           r'Model & Stage & Sens. & Spec. & PPV & NPV & BA & MCC \\',r'\midrule']
    for mi,model in enumerate(LABELS[:3]):
        for ti,time in enumerate(TIMES):
            table.append(model+' & '+time+' & '+' & '.join(f'{means[0,mi+8,ti,METRICS.index(k)]:.3f}' for k in ['sensitivity','specificity','ppv','npv','balanced_accuracy','mcc'])+r' \\')
        if mi<2:table.append(r'\midrule')
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (ROOT/'tables'/'extended_operating.tex').write_text('\n'.join(table)+'\n')
    table=[r'\begin{table*}[t]',r'\centering',r'\small',
           r'\caption{Three-landmark mean operating-point sensitivity analysis, equally averaged over seeds 21 and 44 after within-seed five-fold aggregation. The alternative threshold maximizes Youden index separately on each fold and landmark validation set; the highest threshold breaks ties. No holdout predictions select thresholds. Probability-based AUROC and AP are unchanged by this operation.}',
           r'\label{tab:extended_threshold}',r'\begin{tabular}{llrrrr}',r'\toprule',
           r'Model & Policy & F1 & Sens. & Spec. & MCC \\',r'\midrule']
    for mi,model in enumerate(LABELS[:3]):
        for pi,policy in enumerate(['0.5','Validation Youden']):
            table.append(model+' & '+policy+' & '+' & '.join(f'{means[pi,mi+8,3,METRICS.index(k)]:.3f}' for k in ['f1','sensitivity','specificity','mcc'])+r' \\')
        if mi<2:table.append(r'\midrule')
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (ROOT/'tables'/'extended_threshold.tex').write_text('\n'.join(table)+'\n')
    receipt=dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),training_seeds=SEEDS,bootstrap_rng_seed=BOOTSTRAP_RNG,
                 bootstrap_replicates=B,models=LABELS,metrics=METRICS,patients=175,positives=56,
                 aggregate_only=True,threshold_selection='per fold and per landmark validation Youden; highest threshold resolves ties',
                 aggregation='metric per fold -> mean five folds within each seed -> equal mean of seeds 21/44 -> equal T0/T1/T2 Macro',
                 bootstrap_scope='fixed trained models; shared patient draws across folds, models and landmarks; thresholds fixed from validation',
                 sources=identities)
    (ROOT/'data'/'extended_provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    note=['# Extended seeds 21/44 analysis', '',
          'All 40 model/seed/fold receipts and their 120 report identities passed. Test predictions remain internal; only aggregate values are exported.',
          'The 18 reported quantities comprise 14 metrics plus TP/FP/FN/TN counts. Counts and metrics are averaged across separately evaluated folds.',
          'Alternative thresholds use per-landmark validation patients only, so missing validation prefixes do not require pooling repeated patient rows.',
          'Bootstrap intervals condition on the trained models and fixed validation-selected thresholds, and exclude retraining, threshold-selection and seed-selection uncertainty.',
          'Decision curves use a hypothetical risk-threshold utility with uncalibrated probabilities; their net benefit does not establish clinical usefulness.', '',
          '## Two-seed three-landmark fixed-threshold means', '', '| Model | Accuracy | Sensitivity | Specificity | PPV | NPV | BA | MCC |', '|---|---|---|---|---|---|---|---|']
    for mi,model in enumerate(LABELS):
        note.append('| '+model+' | '+' | '.join(f'{means[0,mi+8,3,METRICS.index(k)]:.4f}' for k in ['accuracy','sensitivity','specificity','ppv','npv','balanced_accuracy','mcc'])+' |')
    note += ['', '## Two-seed validation-Youden three-landmark means', '', '| Model | F1 | Sensitivity | Specificity | MCC |', '|---|---|---|---|---|']
    for mi,model in enumerate(LABELS):
        note.append('| '+model+' | '+' | '.join(f'{means[1,mi+8,3,METRICS.index(k)]:.4f}' for k in ['f1','sensitivity','specificity','mcc'])+' |')
    note += ['', '## T2 minus T0 paired intervals (fixed 0.5)', '']
    for r in longitudinal:
        if r['policy']=='fixed_0.5' and r['seed']=='mean21_44' and r['metric'] in ['auroc','auprc','f1']:
            note.append(f"- {r['model']} {r['metric']}: {r['delta']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}].")
    (ROOT/'notes'/'extended_metrics.md').write_text('\n'.join(note)+'\n')
    print(json.dumps(dict(verified_runs=40,summary_rows=len(summary),fold_rows=len(fold_rows),dca_rows=len(dca),paired_rows=len(contrast),longitudinal_rows=len(longitudinal)),indent=2))


if __name__=='__main__':main()
