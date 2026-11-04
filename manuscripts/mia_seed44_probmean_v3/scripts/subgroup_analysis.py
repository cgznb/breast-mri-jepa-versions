#!/usr/bin/env python3
"""Seed44 probability-ensemble subgroups and separate training-probe summaries.

For pCR, average each patient's five seed44 fold probabilities first, then
calculate metrics or resample patients. Learned latent coordinates are never
ensembled: archived mechanism diagnostics retain a descriptive fold mean.
Only group aggregates are exported.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import hashlib
import numpy as np

HERE = Path(__file__).resolve().parents[1]
SUITE = Path('/projects/xts0030/public/cxx/medium976_t0_t2_no_tf_cv5_multiseed_v1')
MANIFEST = Path('/personal_fast_storage/VLA-PCRV1/data/registered_roi32_medium976_v1/manifest.json')
MODELS = ['direct_no_jepa', 'direct_readout', 'forecast_attention']
LABELS = ['NoJEPA', 'DirectReadout', 'ForecastAttention']
TIMES = ['T0', 'T1', 'T2', 'Macro']
SEEDS = [44]
NBOOT = 2000
RNG_SEED = 20261102
METRICS = ['auroc', 'auprc', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1', 'nll', 'brier']


def write_csv(name, rows):
    with (HERE/'data'/name).open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def weighted_auc(y, p, weights):
    order=np.argsort(-p,kind='stable'); yy=y[order]; pp=p[order]; ww=weights[:,order]
    tp=np.cumsum(ww*yy,axis=1);fp=np.cumsum(ww*(1-yy),axis=1)
    ends=np.r_[np.flatnonzero(np.diff(pp)),len(pp)-1]
    tp,fp=tp[:,ends],fp[:,ends]
    positives=(weights*y).sum(1);negatives=weights.sum(1)-positives
    with np.errstate(invalid='ignore',divide='ignore'):
        tpr=tp/positives[:,None];fpr=fp/negatives[:,None]
        auc=((tpr+np.c_[np.zeros(len(weights)),tpr[:,:-1]])*(fpr-np.c_[np.zeros(len(weights)),fpr[:,:-1]])/2).sum(1)
    return auc


def metrics(y,p,allow_discrimination=True):
    n=len(y);np_=int(y.sum());nn=n-np_;pred=p>=.5
    tp=float((pred*y).sum());tn=float(((~pred)*(1-y)).sum());fp=float(pred.sum()-tp);fn=float(np_-tp)
    sens=tp/np_ if np_ else np.nan;spec=tn/nn if nn else np.nan
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    pp=np.clip(p,1e-7,1-1e-7)
    nll=float(-(y*np.log(pp)+(1-y)*np.log1p(-pp)).mean());brier=float(((p-y)**2).mean())
    auc=ap=np.nan
    if allow_discrimination and np_ and nn:
        auc=float(weighted_auc(y,p,np.ones((1,n)))[0])
        order=np.argsort(-p,kind='stable');yy=y[order];probs=p[order]
        tp_c=np.cumsum(yy);ends=np.r_[np.flatnonzero(np.diff(probs)),n-1]
        recall=tp_c[ends]/np_;precision=tp_c[ends]/(ends+1)
        ap=float((precision*np.diff(np.r_[0.,recall])).sum())
    return dict(zip(METRICS,[auc,ap,(sens+spec)/2,sens,spec,f1,nll,brier]))


def finite_value(value):
    return float(value) if np.isfinite(value) else None


manifest=json.loads(MANIFEST.read_text())
patients={r['patient_key']:r for r in manifest['patients'] if r['split']=='test'}
ids=sorted(patients);y=np.array([patients[i]['target']['pcr'] for i in ids])
assert len(ids)==175 and y.sum()==56
prob=np.empty((len(SEEDS),3,5,3,175));source_reports=[]
for si,seed in enumerate(SEEDS):
 for mi,model in enumerate(MODELS):
    for fi in range(5):
        run=SUITE/'runs'/f'{model}_seed{seed}_fold{fi+1}'
        receipt=json.loads((run/'evaluation_done.json').read_text());assert receipt['completed']
        for r in receipt['reports']:
            st=Path(r['path']).stat();assert (st.st_size,st.st_mtime_ns)==(r['bytes'],r['mtime_ns'])
        path=run/'evaluation_best_test.json';report=json.loads(path.read_text())
        source_reports.append(dict(model=LABELS[mi],seed=seed,fold=fi+1,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for ti in range(3):
            rows=sorted((r for r in report['rows'] if r['landmark']==ti),key=lambda r:r['patient_key'])
            assert [r['patient_key'] for r in rows]==ids
            assert np.array_equal([r['pcr'] for r in rows],y)
            prob[si,mi,fi,ti]=[r['pcr_probability'] for r in rows]

# This is the classifier used for every subgroup result. Average probabilities
# within each patient/landmark, not metrics or thresholded predictions.
ensemble_prob=prob[0].mean(axis=1)
assert ensemble_prob.shape==(3,3,175)

groupdefs=[('HR/HER2','HR+/HER2-','HR+/HER2-'),('HR/HER2','HR+/HER2+','HR+/HER2+'),
           ('HR/HER2','HR-/HER2+','HR-/HER2+'),('HR/HER2','HR-/HER2-','HR-/HER2-'),
           ('Age','Age <50','Age <50'),('Age','Age >=50','Age >=50'),
           ('Source namespace','ISPY2 namespace','ISPY2 namespace'),('Source namespace','ACRIN-6698 namespace','ACRIN-6698 namespace'),
           ('Baseline crop','Largest component contained','Largest component contained'),('Baseline crop','Largest component clipped','Largest component clipped')]
members={label:[] for _,label,_ in groupdefs}
for idx,i in enumerate(ids):
    r=patients[i];age,hr,her2,_=r['clinical']['values']
    geom=json.loads(Path(r['geometry']['paired_sampling_plan']).read_text())
    labels=[f"HR{'+' if hr else '-'}/HER2{'+' if her2 else '-'}",
            'Age <50' if age<50 else 'Age >=50',
            'ACRIN-6698 namespace' if geom['native_patient_id'].startswith('ACRIN') else 'ISPY2 namespace',
            'Largest component clipped' if geom['mask_qc']['largest_clipped'] else 'Largest component contained']
    for label in labels:members[label].append(idx)

rng=np.random.default_rng(RNG_SEED)
size_rows=[];metric_rows=[];delta_rows=[];fold_rows=[]
for family,label,display in groupdefs:
    ix=np.array(members[label]);yy=y[ix];n=len(ix);positives=int(yy.sum());negatives=n-positives
    allowed=n>=10 and positives>=5 and negatives>=5
    status='exploratory small subgroup' if n<20 else 'exploratory'
    if not allowed:status='discrimination suppressed: fewer than 5 outcomes in one class'
    size_rows.append(dict(family=family,subgroup=label,n=n,positives=positives,negatives=negatives,
                          prevalence=positives/n,discrimination_reported=allowed,status=status))
    fold_metrics=np.empty((len(SEEDS),3,5,3,len(METRICS)))
    for si,seed in enumerate(SEEDS):
     for mi,model in enumerate(LABELS):
        for fi in range(5):
            for ti in range(3):
                d=metrics(yy,prob[si,mi,fi,ti,ix],allowed)
                fold_metrics[si,mi,fi,ti]=[d[k] for k in METRICS]
                fold_rows.append(dict(family=family,subgroup=label,model=model,seed=seed,fold=fi+1,timepoint=TIMES[ti],n=n,positives=positives,
                                      **{k:finite_value(v) for k,v in d.items()}))
    # Per-fold metrics are audit-only. Manuscript subgroup metrics operate on
    # the patient-wise fivefold probability average above.
    means=np.empty((3,3,len(METRICS)))
    for mi in range(3):
        for ti in range(3):
            d=metrics(yy,ensemble_prob[mi,ti,ix],allowed)
            means[mi,ti]=[d[k] for k in METRICS]
    means=np.concatenate([means,means.mean(1,keepdims=True)],axis=1)
    if allowed:
        weights=rng.multinomial(n,np.full(n,1/n),size=NBOOT)
        valid=((weights*yy).sum(1)>0)&((weights*(1-yy)).sum(1)>0)
        boot=np.empty((NBOOT,3,3))
        for mi in range(3):
            for ti in range(3):
                boot[:,mi,ti]=weighted_auc(yy,ensemble_prob[mi,ti,ix],weights)
        boot=boot[valid]
        boot=np.concatenate([boot,boot.mean(2,keepdims=True)],axis=2)
        auc_ci=np.quantile(boot,[.025,.975],axis=0)
        delta=boot[:,2]-boot[:,0];delta_ci=np.quantile(delta,[.025,.975],axis=0)
    for mi,model in enumerate(LABELS):
        for ti,time in enumerate(TIMES):
            metric_rows.append(dict(family=family,subgroup=label,model=model,seed=44,timepoint=time,n=n,positives=positives,
                                    negatives=negatives,status=status,
                                    **{k:finite_value(means[mi,ti,ki]) for ki,k in enumerate(METRICS)},
                                    auroc_ci_low=float(auc_ci[0,mi,ti]) if allowed else None,
                                    auroc_ci_high=float(auc_ci[1,mi,ti]) if allowed else None))
    for ti,time in enumerate(TIMES):
        delta_rows.append(dict(family=family,subgroup=label,seed=44,timepoint=time,n=n,positives=positives,
                               comparison='ForecastAttention minus NoJEPA',
                               delta_auroc=float(means[2,ti,0]-means[0,ti,0]) if allowed else None,
                               ci_low=float(delta_ci[0,ti]) if allowed else None,
                               ci_high=float(delta_ci[1,ti]) if allowed else None,
                               bootstrap_valid=int(valid.sum()) if allowed else 0,bootstrap_requested=NBOOT,
                               status=status))
write_csv('subgroup_sizes.csv',size_rows);write_csv('subgroup_metrics.csv',metric_rows)
write_csv('subgroup_fold_metrics.csv',fold_rows);write_csv('subgroup_paired_differences.csv',delta_rows)

# Same seed44 probability-ensemble point estimates in the legacy audit schema.
# subgroup_fold_metrics.csv remains explicitly per-fold audit data only.
seed_rows=[{k:r[k] for k in ['family','subgroup','model','seed','timepoint','n','positives',*METRICS]}
           for r in metric_rows]
write_csv('subgroup_seed_metrics.csv',seed_rows)

# All archived diagnostics are fixed training probes, not validation/test evidence.
world_rows=[];gradient_rows=[];probe_sources=[]
for seed in SEEDS:
 for mi,model in enumerate(MODELS):
    for fi in range(5):
        path=SUITE/'runs'/f'{model}_seed{seed}_fold{fi+1}'/'diagnostics.json'
        diagnostics=json.loads(path.read_text())
        probe_sources.append(dict(model=LABELS[mi],seed=seed,fold=fi+1,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for d in diagnostics:
            assert 'training probe' in d['scope']
            batches=d['gradient_batches'];denom=sum(b['patients'] for b in batches)
            task_grad=sum(b['patients']*b['weighted_gradient_norms']['task']['world_predictor'] for b in batches)/denom
            roll_grad=sum(b['patients']*b['weighted_gradient_norms'].get('rollout',{}).get('world_predictor',0.) for b in batches)/denom
            cos_values=[b['task_auxiliary_cosines'].get('rollout',{}).get('shared_encoder_fusion') for b in batches]
            cosine=float(np.mean([v for v in cos_values if v is not None])) if any(v is not None for v in cos_values) else None
            gradient_rows.append(dict(model=LABELS[mi],seed=seed,fold=fi+1,epoch=d['epoch'],patients=d['patients'],
                                      scope='fixed training probe, not validation or test',task_predictor_gradient_norm=task_grad,
                                      rollout_predictor_gradient_norm=roll_grad,task_rollout_shared_gradient_cosine=cosine))
            world=d.get('world',{})
            if world.get('status')!='evaluated':continue
            assert world['target_coordinates']=='same checkpoint EMA teacher for model and both copy baselines'
            for horizon,m in world['metrics'].items():
                world_rows.append(dict(model=LABELS[mi],seed=seed,fold=fi+1,epoch=d['epoch'],horizon=horizon,
                                       patients=m['patients'],prefix_targets=m['prefix_targets'],world_head_trained=world['world_head_trained'],
                                       autonomous_relative_to_copy=m['autonomous_relative_to_copy'],
                                       autonomous_skill_vs_copy=m['autonomous_skill_vs_copy'],
                                       autonomous_l1=m['autonomous_l1'],autonomous_copy_l1=m['autonomous_copy_l1'],
                                       scope='fixed training probe, continuous chains only; not validation or test'))
write_csv('subgroup_world_probe.csv',world_rows);write_csv('subgroup_gradient_probe.csv',gradient_rows)
world_summary=[]
for model in LABELS:
    for epoch in [5,10,15,20,25,30,35,40,45,50]:
        for horizon in ['horizon_1','horizon_2','horizon_3','horizon_all']:
            rows=[r for r in world_rows if r['model']==model and r['epoch']==epoch and r['horizon']==horizon]
            if len(rows)!=5:continue
            vals=np.array([r['autonomous_skill_vs_copy'] for r in rows],dtype=float)
            grads=[r for r in gradient_rows if r['model']==model and r['epoch']==epoch]
            world_summary.append(dict(model=model,seed=44,epoch=epoch,horizon=horizon,seeds=1,folds_per_seed=5,
                                       skill_vs_copy_mean=float(vals.mean()),skill_vs_copy_run_sd_descriptive=float(np.std(vals,ddof=1)),
                                       probe_patients_min=min(r['patients'] for r in rows),probe_patients_max=max(r['patients'] for r in rows),
                                       task_predictor_gradient_mean=float(np.mean([r['task_predictor_gradient_norm'] for r in grads])),
                                       rollout_predictor_gradient_mean=float(np.mean([r['rollout_predictor_gradient_norm'] for r in grads])),
                                       scope='seed44 fixed training probe; fold mean of snapshot skills, not latent or pCR ensembling; fold spread descriptive only'))
write_csv('subgroup_world_summary.csv',world_summary)

table=[r'\begin{table*}[t]',r'\centering',r'\small',
       r'\caption{Exploratory subgroup discrimination for the seed44 fivefold probability ensemble. Each patient\textquotesingle s probabilities are averaged over five fold models before computing AUROC; Macro then averages T0--T2 AUROCs. Intervals are paired patient-bootstrap intervals for ForecastAttention minus NoJEPA, conditional on the fitted ensemble. Age 50 is an exploratory reporting cutpoint. Namespace groups are parts of one clinical release, not external cohorts.}',
       r'\label{tab:subgroups}',r'\begin{tabular}{lrrr}',r'\toprule',r'Subgroup ($n$/pCR+) & NoJEPA & Full & $\Delta$ AUROC [95\% interval] \\',r'\midrule']
for family,label,_ in groupdefs:
    if family=='Baseline crop':continue
    s=next(r for r in size_rows if r['subgroup']==label)
    a=next(r for r in metric_rows if r['subgroup']==label and r['model']=='NoJEPA' and r['timepoint']=='Macro')
    b=next(r for r in metric_rows if r['subgroup']==label and r['model']=='ForecastAttention' and r['timepoint']=='Macro')
    d=next(r for r in delta_rows if r['subgroup']==label and r['timepoint']=='Macro')
    name=label.replace('>=',r'$\geq$').replace('<',r'$<$')
    table.append(f"{name} ({s['n']}/{s['positives']}) & {a['auroc']:.3f} & {b['auroc']:.3f} & {d['delta_auroc']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"+r' \\')
table += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
(HERE/'tables'/'subgroup_results.tex').write_text('\n'.join(table)+'\n')
worldtable=[r'\begin{table*}[t]',r'\centering',r'\small',
            r'\caption{Archived seed44 epoch 20 fixed-training-probe diagnostics. These are descriptive means of five fold-specific skills, distinct from probability ensembling for pCR. Skill is $1-\mathrm{L1}_{forecast}/\mathrm{L1}_{copy}$ within each snapshot\textquotesingle s EMA coordinates; positive is better than holding the landmark state constant. Forecast targets require a continuous actually observed visit chain. These are training-probe summaries, not best-checkpoint holdout forecast evaluations. NoJEPA has no trained-world diagnostic.}',
            r'\label{tab:worldprobe}',r'\begin{lrrr}',r'\toprule']
# Keep the compact table schema readable and independent of raw latent scales.
worldtable=worldtable[:-2]+[r'\begin{tabular}{lrrr}',r'\toprule',r'Model & Horizon 1 skill & Horizon 2 skill & Horizon 3 skill \\',r'\midrule']
for model in LABELS:
    vals=[next((r['skill_vs_copy_mean'] for r in world_summary if r['model']==model and r['epoch']==20 and r['horizon']==f'horizon_{h}'),None) for h in [1,2,3]]
    worldtable.append(model+' & '+' & '.join(f'{v:+.3f}' if v is not None else '--' for v in vals)+r' \\')
worldtable += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
(HERE/'tables'/'subgroup_world_probe.tex').write_text('\n'.join(worldtable)+'\n')

provenance=dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),training_seeds=SEEDS,bootstrap_rng_seed=RNG_SEED,
                bootstrap_replicates=NBOOT,patient_rows_exported=False,manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                source_reports=source_reports,diagnostic_sources=probe_sources,subgroup_count=len(groupdefs),aggregation='seed44 only: patient-wise mean of five fold probabilities -> metric; Macro averages T0-T2 metrics',
                fold_metric_scope='subgroup_fold_metrics.csv is audit-only and is not averaged for manuscript pCR results',
                mechanism_aggregation='seed44 only: descriptive five-fold mean of per-snapshot copy-normalized skills and gradients; no latent-coordinate or pCR probability ensembling',
                missing_class_bootstraps='excluded from percentile interval and valid count reported',
                discrimination_rule='n>=10 and >=5 positives and >=5 negatives; otherwise AUROC/AP and their intervals suppressed',
                mechanism_scope='archived fixed32-patient training probes with continuous real-chain eligibility; no best-checkpoint validation/test forecast file exists')
(HERE/'data'/'subgroup_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps({'sizes':size_rows,'macro_deltas':[r for r in delta_rows if r['timepoint']=='Macro'],
                  'world_epoch 20':[r for r in world_summary if r['epoch']==20]},indent=2))
