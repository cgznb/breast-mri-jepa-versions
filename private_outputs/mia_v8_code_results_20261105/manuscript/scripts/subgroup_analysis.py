#!/usr/bin/env python3
"""Four-seed subgroup metrics and separately averaged training diagnostics.

Within each seed, average five fold probabilities before computing metrics.
Then average seed metrics, never probabilities across seeds. Patient bootstrap
resamples are shared across all fitted seed/model/landmark vectors. Only group
aggregates are exported; latent coordinates are never ensembled.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import hashlib
import numpy as np

HERE = Path(__file__).resolve().parents[1]
from analysis_common import SEEDS, MODELS, LABELS, run_dir
SEED_TAG = ",".join(map(str, SEEDS))
MANIFEST = Path('/personal_fast_storage/VLA-PCRV1/data/registered_roi32_medium976_v1/manifest.json')
TIMES = ['T0', 'T1', 'T2', 'Macro']
NBOOT = 2000
RNG_SEED = 20261103
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
prob=np.empty((len(SEEDS),len(LABELS),5,3,175));source_reports=[]
for si,seed in enumerate(SEEDS):
 for mi,model in enumerate(MODELS):
    for fi in range(5):
        run=run_dir(seed,model,fi+1)
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
ensemble_prob=prob.mean(axis=2)
assert ensemble_prob.shape==(len(SEEDS),len(LABELS),3,175)
assert SEEDS==[42,44,71,2026]
assert LABELS==['NoJEPA','DirectReadout','ForecastMean']

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
size_rows=[];metric_rows=[];delta_rows=[];fold_rows=[];seed_rows=[]
for family,label,display in groupdefs:
    ix=np.array(members[label]);yy=y[ix];n=len(ix);positives=int(yy.sum());negatives=n-positives
    allowed=n>=10 and positives>=5 and negatives>=5
    status='exploratory small subgroup' if n<20 else 'exploratory'
    if not allowed:status='discrimination suppressed: fewer than 5 outcomes in one class'
    size_rows.append(dict(family=family,subgroup=label,n=n,positives=positives,negatives=negatives,
                          prevalence=positives/n,discrimination_reported=allowed,status=status))
    fold_metrics=np.empty((len(SEEDS),len(LABELS),5,3,len(METRICS)))
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
    per_seed=np.empty((len(SEEDS),len(LABELS),3,len(METRICS)))
    for si,seed in enumerate(SEEDS):
        for mi in range(len(LABELS)):
            for ti in range(3):
                d=metrics(yy,ensemble_prob[si,mi,ti,ix],allowed)
                per_seed[si,mi,ti]=[d[k] for k in METRICS]
    per_seed=np.concatenate([per_seed,per_seed.mean(2,keepdims=True)],axis=2)
    means=per_seed.mean(axis=0)
    seed_sd=per_seed.std(axis=0,ddof=1)
    for si,seed in enumerate(SEEDS):
        for mi,model in enumerate(LABELS):
            for ti,time in enumerate(TIMES):
                seed_rows.append(dict(family=family,subgroup=label,model=model,seed=seed,
                    timepoint=time,n=n,positives=positives,
                    **{k:finite_value(per_seed[si,mi,ti,ki]) for ki,k in enumerate(METRICS)}))
    if allowed:
        weights=rng.multinomial(n,np.full(n,1/n),size=NBOOT)
        valid=((weights*yy).sum(1)>0)&((weights*(1-yy)).sum(1)>0)
        boot=np.empty((NBOOT,len(SEEDS),len(LABELS),3))
        for si in range(len(SEEDS)):
            for mi in range(len(LABELS)):
                for ti in range(3):
                    boot[:,si,mi,ti]=weighted_auc(yy,ensemble_prob[si,mi,ti,ix],weights)
        # Shared draws; average separately recomputed seed AUROCs.
        boot=boot[valid].mean(axis=1)
        boot=np.concatenate([boot,boot.mean(2,keepdims=True)],axis=2)
        auc_ci=np.quantile(boot,[.025,.975],axis=0)
        delta=boot[:,2]-boot[:,0];delta_ci=np.quantile(delta,[.025,.975],axis=0)
    for mi,model in enumerate(LABELS):
        for ti,time in enumerate(TIMES):
            row=dict(family=family,subgroup=label,model=model,seed=SEED_TAG,timepoint=time,n=n,positives=positives,
                     negatives=negatives,status=status,
                     **{k:finite_value(means[mi,ti,ki]) for ki,k in enumerate(METRICS)},
                     **{k+'_seed_sd':finite_value(seed_sd[mi,ti,ki]) for ki,k in enumerate(METRICS)},
                     auroc_ci_low=float(auc_ci[0,mi,ti]) if allowed else None,
                     auroc_ci_high=float(auc_ci[1,mi,ti]) if allowed else None)
            metric_rows.append(row)
    for ti,time in enumerate(TIMES):
        delta_rows.append(dict(family=family,subgroup=label,seed=SEED_TAG,timepoint=time,n=n,positives=positives,
                               comparison='ForecastMean minus NoJEPA',
                               delta_auroc=float(means[2,ti,0]-means[0,ti,0]) if allowed else None,
                               delta_auroc_seed_sd=float(np.std(per_seed[:,2,ti,0]-per_seed[:,0,ti,0],ddof=1)) if allowed else None,
                               ci_low=float(delta_ci[0,ti]) if allowed else None,
                               ci_high=float(delta_ci[1,ti]) if allowed else None,
                               bootstrap_valid=int(valid.sum()) if allowed else 0,bootstrap_requested=NBOOT,
                               status=status))
write_csv('subgroup_sizes.csv',size_rows);write_csv('subgroup_metrics.csv',metric_rows)
write_csv('subgroup_fold_metrics.csv',fold_rows);write_csv('subgroup_paired_differences.csv',delta_rows)

# Per-seed estimates remain visible for reproducibility and contradictory runs.
write_csv('subgroup_seed_metrics.csv',seed_rows)

# All archived diagnostics are fixed training probes, not validation/test evidence.
world_rows=[];gradient_rows=[];probe_sources=[]
for seed in SEEDS:
 for mi,model in enumerate(MODELS):
    for fi in range(5):
        path=run_dir(seed,model,fi+1)/'diagnostics.json'
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
world_seed_summary=[];world_summary=[]
for model in LABELS:
    epochs=sorted({r['epoch'] for r in world_rows if r['model']==model})
    for epoch in epochs:
        for horizon in ['horizon_1','horizon_2','horizon_3','horizon_all']:
            rows=[r for r in world_rows if r['model']==model and r['epoch']==epoch and r['horizon']==horizon]
            # A common epoch must contain all five fold snapshots in every seed.
            if len(rows)!=len(SEEDS)*5:continue
            seed_values=[];task_values=[];roll_values=[]
            for seed in SEEDS:
                rr=[r for r in rows if r['seed']==seed]
                assert len(rr)==5
                grads=[r for r in gradient_rows if r['model']==model and r['seed']==seed and r['epoch']==epoch]
                assert len(grads)==5
                skill=float(np.mean([r['autonomous_skill_vs_copy'] for r in rr]))
                task=float(np.mean([r['task_predictor_gradient_norm'] for r in grads]))
                roll=float(np.mean([r['rollout_predictor_gradient_norm'] for r in grads]))
                seed_values.append(skill);task_values.append(task);roll_values.append(roll)
                world_seed_summary.append(dict(model=model,seed=seed,epoch=epoch,horizon=horizon,
                    folds=5,skill_vs_copy_mean=skill,task_predictor_gradient_mean=task,
                    rollout_predictor_gradient_mean=roll,
                    probe_patients_min=min(r['patients'] for r in rr),probe_patients_max=max(r['patients'] for r in rr)))
            world_summary.append(dict(model=model,seed=SEED_TAG,epoch=epoch,horizon=horizon,seeds=len(SEEDS),folds_per_seed=5,
                skill_vs_copy_mean=float(np.mean(seed_values)),skill_vs_copy_seed_sd=float(np.std(seed_values,ddof=1)),
                probe_patients_min=min(r['patients'] for r in rows),probe_patients_max=max(r['patients'] for r in rows),
                task_predictor_gradient_mean=float(np.mean(task_values)),task_predictor_gradient_seed_sd=float(np.std(task_values,ddof=1)),
                rollout_predictor_gradient_mean=float(np.mean(roll_values)),rollout_predictor_gradient_seed_sd=float(np.std(roll_values,ddof=1)),
                scope='fixed training probe; mean of within-seed fold means, SD across four seed means; no latent or pCR probability ensembling'))
write_csv('subgroup_world_seed_summary.csv',world_seed_summary)
write_csv('subgroup_world_summary.csv',world_summary)

table=[r'\begin{table*}[t]',r'\centering',r'\small',
       r'\caption{Exploratory subgroup discrimination using seeds 42, 44, 71, and 2026. Within each seed, five fold probabilities are averaged per patient before computing AUROC; Macro averages T0--T2 AUROCs. Model entries are means $\pm$ sample SD across the four seed metrics. Paired patient-bootstrap intervals for the mean ForecastMean minus NoJEPA difference condition on all fitted models and keep the seed set fixed. Age 50 is an exploratory reporting cutpoint. Namespace groups are parts of one clinical release, not external cohorts.}',
       r'\label{tab:subgroups}',r'\begin{tabular}{lrrr}',r'\toprule',r'Subgroup ($n$/pCR+) & NoJEPA & Full & $\Delta$ AUROC [95\% interval] \\',r'\midrule']
for family,label,_ in groupdefs:
    if family=='Baseline crop':continue
    s=next(r for r in size_rows if r['subgroup']==label)
    a=next(r for r in metric_rows if r['subgroup']==label and r['model']=='NoJEPA' and r['timepoint']=='Macro')
    b=next(r for r in metric_rows if r['subgroup']==label and r['model']=='ForecastMean' and r['timepoint']=='Macro')
    d=next(r for r in delta_rows if r['subgroup']==label and r['timepoint']=='Macro')
    name=label.replace('>=',r'$\geq$').replace('<',r'$<$')
    table.append(f"{name} ({s['n']}/{s['positives']}) & {a['auroc']:.3f} $\\pm$ {a['auroc_seed_sd']:.3f} & {b['auroc']:.3f} $\\pm$ {b['auroc_seed_sd']:.3f} & {d['delta_auroc']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"+r' \\')
table += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
(HERE/'tables'/'subgroup_results.tex').write_text('\n'.join(table)+'\n')
worldtable=[r'\begin{table*}[t]',r'\centering',r'\small',
            r'\caption{Archived epoch 20 fixed-training-probe diagnostics for seeds 42, 44, 71, and 2026. Each entry is the mean $\pm$ sample SD of four seed-specific five-fold mean skills, distinct from probability ensembling for pCR. Skill is $1-\mathrm{L1}_{forecast}/\mathrm{L1}_{copy}$ within each snapshot\textquotesingle s EMA coordinates; positive is better than holding the landmark state constant. Forecast targets require a continuous actually observed visit chain. These are training-probe summaries, not best-checkpoint holdout forecast evaluations. NoJEPA has no trained-world diagnostic.}',
            r'\label{tab:worldprobe}',r'\begin{lrrr}',r'\toprule']
# Keep the compact table schema readable and independent of raw latent scales.
worldtable=worldtable[:-2]+[r'\begin{tabular}{lrrr}',r'\toprule',r'Model & Horizon 1 skill & Horizon 2 skill & Horizon 3 skill \\',r'\midrule']
for model in LABELS:
    vals=[next((r for r in world_summary if r['model']==model and r['epoch']==20 and r['horizon']==f'horizon_{h}'),None) for h in [1,2,3]]
    worldtable.append(model+' & '+' & '.join(f"{v['skill_vs_copy_mean']:+.3f} $\\pm$ {v['skill_vs_copy_seed_sd']:.3f}" if v is not None else '--' for v in vals)+r' \\')
worldtable += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
(HERE/'tables'/'subgroup_world_probe.tex').write_text('\n'.join(worldtable)+'\n')

provenance=dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),training_seeds=SEEDS,bootstrap_rng_seed=RNG_SEED,
                bootstrap_replicates=NBOOT,patient_rows_exported=False,manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                source_reports=source_reports,diagnostic_sources=probe_sources,subgroup_count=len(groupdefs),aggregation='Within each of seeds 42,44,71,2026: patient-wise mean of five fold probabilities -> metric; equally average seed metrics; Macro averages T0-T2 metrics. No across-seed probability ensemble.',
                fold_metric_scope='subgroup_fold_metrics.csv is audit-only and is not averaged for manuscript pCR results',
                mechanism_aggregation='Within each seed: descriptive five-fold mean of per-snapshot copy-normalized skills and gradients; then four-seed mean and sample SD; no latent-coordinate or pCR probability ensembling',
                missing_class_bootstraps='excluded from percentile interval and valid count reported',
                discrimination_rule='n>=10 and >=5 positives and >=5 negatives; otherwise AUROC/AP and their intervals suppressed',
                mechanism_scope='archived fixed32-patient training probes with continuous real-chain eligibility; no best-checkpoint validation/test forecast file exists')
(HERE/'data'/'subgroup_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps({'sizes':size_rows,'macro_deltas':[r for r in delta_rows if r['timepoint']=='Macro'],
                  'world_epoch 20':[r for r in world_summary if r['epoch']==20]},indent=2))
