#!/usr/bin/env python3
"""Independent seed44 fold-metric numerical verification (optional scipy/sklearn).

Only aggregate checks are written. Patient scores and repeated bootstrap rows
remain in memory. The main analysis does not depend on scipy/sklearn.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import time as run_time
import numpy as np
import scipy
from scipy.stats import rankdata
import sklearn
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    recall_score, precision_score, balanced_accuracy_score, matthews_corrcoef,
    brier_score_loss, log_loss, confusion_matrix, roc_curve,
)
from analysis_common import load_data, LABELS, TIMES, POLICIES
FOLDS=list(range(1,6))
SEED_TEXT="44"
from extended_metrics import METRICS, all_metrics
import significance_analysis as sa

ROOT=Path(__file__).resolve().parents[1]


def read(name):return list(csv.DictReader((ROOT/'data'/name).open()))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_metrics(y,p,threshold=.5):
    pred=p>=threshold
    sensitivity=recall_score(y,pred,zero_division=0)
    specificity=recall_score(1-y,~pred,zero_division=0)
    bins=np.array([min(int(float(q)*10),9) for q in p])
    ece=sum(abs(float((p[bins==k]-y[bins==k]).sum())) for k in range(10))/len(y)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return np.array([roc_auc_score(y,p),average_precision_score(y,p),
        f1_score(y,pred,zero_division=0),accuracy_score(y,pred),sensitivity,specificity,
        precision_score(y,pred,zero_division=0),precision_score(1-y,~pred,zero_division=0),
        balanced_accuracy_score(y,pred),matthews_corrcoef(y,pred),sensitivity+specificity-1,
        brier_score_loss(y,p),log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1]),ece,tp,fp,fn,tn])


def explicit_reference(y,scores,threshold=.5,only_primary=False):
    """Independent metric formulas on explicit repeated patient rows.

    AUC uses mean ranks; AP uses positive versus all inclusive threshold ranks.
    No weighted cumulative/group-end production implementation is reused.
    """
    y=np.broadcast_to(y,scores.shape)
    positive=y.astype(bool);npos=y.sum(axis=-1);nneg=scores.shape[-1]-npos
    ranks=rankdata(scores,method='average',axis=-1)
    auc=(np.where(positive,ranks,0).sum(axis=-1)-npos*(npos+1)/2)/(npos*nneg)
    all_ranks=rankdata(-scores,method='max',axis=-1)
    positive_ranks=rankdata(-np.where(positive,scores,-np.inf),method='max',axis=-1)
    ap=np.where(positive,positive_ranks/all_ranks,0).sum(axis=-1)/npos
    prediction=scores>=threshold
    tp=(prediction&positive).sum(axis=-1).astype(float)
    fp=(prediction&~positive).sum(axis=-1).astype(float)
    fn=npos-tp;tn=nneg-fp
    f1=2*tp/(npos+tp+fp)
    if only_primary:return np.stack([auc,ap,f1],axis=-1)
    divide=lambda a,b:np.divide(a,b,out=np.zeros_like(a,float),where=b!=0)
    sens=tp/npos;spec=tn/nneg
    acc=(tp+tn)/scores.shape[-1]
    ppv=divide(tp,tp+fp);npv=divide(tn,tn+fn)
    mcc=divide(tp*tn-fp*fn,np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    brier=((scores-y)**2).mean(axis=-1)
    clipped=np.clip(scores,1e-7,1-1e-7)
    nll=-(y*np.log(clipped)+(1-y)*np.log(1-clipped)).mean(axis=-1)
    bins=np.minimum((scores*10).astype(int),9)
    ece=sum(abs(np.where(bins==bi,scores-y,0).sum(axis=-1))/scores.shape[-1] for bi in range(10))
    return np.stack([auc,ap,f1,acc,sens,spec,ppv,npv,(sens+spec)/2,mcc,sens+spec-1,brier,nll,ece,tp,fp,fn,tn],axis=-1)


def reference_holm(values):
    values=np.asarray(values,float)
    ranks=sorted(range(len(values)),key=lambda i:(values[i],i))
    adjusted=np.empty(len(values))
    for rank,idx in enumerate(ranks):
        adjusted[idx]=min(1.,max((len(values)-j)*values[ranks[j]] for j in range(rank+1)))
    return adjusted


def difference(values,spec):
    if spec['kind']=='longitudinal':return values[...,spec['aa'],2,spec['ki']]-values[...,spec['aa'],0,spec['ki']]
    return values[...,spec['aa'],spec['ti'],spec['ki']]-values[...,spec['bb'],spec['ti'],spec['ki']]


def main():
    started=run_time.monotonic();data=load_data();y=data['y'];ensemble=data['scores'];thresholds=data['thresholds']
    individual={(r['policy'],r['fold'],r['model'],r['timepoint']):r for r in read('seed44_individual_fold_metrics.csv')}
    summary={(r['policy'],r['model'],r['timepoint']):r for r in read('extended_summary.csv')}
    threshold_checks=0
    for source in data['sources']:
        validation=json.loads(Path(source['validation_report']).read_text())
        fi=int(source['fold'])-1;mi=LABELS.index(source['model'])
        for ti in range(3):
            rr=[r for r in validation['rows'] if r['landmark']==ti]
            vy=np.array([r['pcr'] for r in rr]);vp=np.array([r['pcr_probability'] for r in rr])
            fpr,tpr,candidates=roc_curve(vy,vp,drop_intermediate=False)
            optimum=tpr-fpr;best=np.flatnonzero(np.isclose(optimum,optimum.max(),atol=1e-14,rtol=0))[0]
            expected=float(candidates[best])
            if np.isinf(expected):expected=np.nextafter(vp.max(),np.inf)
            assert abs(expected-thresholds[1,fi,mi,ti])<1e-12
            threshold_checks+=1
    per_seed=np.empty((2,5,4,3,18));max_error=0.;point_checks=0
    for pi,policy in enumerate(POLICIES):
      for si,seed in enumerate(FOLDS):
       for mi,model in enumerate(LABELS):
        for ti in range(3):
            expected=reference_metrics(y,ensemble[si,mi,ti],thresholds[pi,si,mi,ti])
            per_seed[pi,si,mi,ti]=expected
            row=individual[policy,str(seed),model,TIMES[ti]]
            actual=np.array([float(row[k]) for k in METRICS])
            np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
            max_error=max(max_error,float(np.max(abs(actual-expected))));point_checks+=18
    per_seed=np.concatenate([per_seed,per_seed.mean(axis=3,keepdims=True)],axis=3)
    for pi,policy in enumerate(POLICIES):
      for si,seed in enumerate(FOLDS):
       for mi,model in enumerate(LABELS):
        row=individual[policy,str(seed),model,'Macro'];actual=np.array([float(row[k]) for k in METRICS])
        np.testing.assert_allclose(actual,per_seed[pi,si,mi,3],atol=1e-12,rtol=0);point_checks+=18
    means=per_seed.mean(axis=1);sd=per_seed.std(axis=1,ddof=1)
    summary_checks=0
    for pi,policy in enumerate(POLICIES):
      for mi,model in enumerate(LABELS):
       for ti,t in enumerate(TIMES):
        row=summary[policy,model,t]
        np.testing.assert_allclose([float(row[k]) for k in METRICS],means[pi,mi,ti],atol=1e-12,rtol=0)
        np.testing.assert_allclose([float(row[k+'_fold_sd']) for k in METRICS],sd[pi,mi,ti],atol=1e-12,rtol=0)
        summary_checks+=36
    print('Independent sklearn point metrics, fold means, sample SD passed.',flush=True)

    # Independently evaluate all14 on 20 actual explicit resamples, against
    # production weighted metrics, both policies and all model/seed/landmarks.
    weights100=np.random.default_rng(6792).multinomial(len(y),np.full(len(y),1/len(y)),size=20)
    indices100=np.stack([np.repeat(np.arange(len(y)),w) for w in weights100])
    metric_boot_error=0.;weighted_checks=0
    for pi in range(2):
      for si in range(5):
       for mi in range(4):
        for ti in range(3):
            pp=ensemble[si,mi,ti];th=thresholds[pi,si,mi,ti]
            expected=np.array([reference_metrics(y[ix],pp[ix],th) for ix in indices100])
            actual=all_metrics(y,pp,weights100,th)
            np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
            metric_boot_error=max(metric_boot_error,float(abs(actual-expected).max()));weighted_checks+=actual.size
    print('Independent sklearn weighted bootstrap 20 draws passed.',flush=True)

    # Exact enumeration not needed: centered bootstrap is approximate, and its
    # target is the seed-mean metric, not a score-exchangeability randomization.
    specifications=sa.hypotheses();principal=[s for s in specifications if s['family']==sa.PRIMARY]
    observed_primary=np.array([difference(means[0],s) for s in principal]);counts=np.zeros(9,np.int64)
    bootstrap_values=np.empty((2000,2,4,4,14));actual_rng=np.random.default_rng(sa.RNG_SEED)
    principal_scores=ensemble[:,:3]
    for first in range(0,sa.N_BOOTSTRAP,sa.BATCH_SIZE):
        size=min(sa.BATCH_SIZE,sa.N_BOOTSTRAP-first)
        weights=actual_rng.multinomial(len(y),np.full(len(y),1/len(y)),size=size)
        indices=np.stack([np.repeat(np.arange(len(y)),w) for w in weights])
        for subfirst in range(0,size,256):
            subend=min(size,subfirst+256);ix=indices[subfirst:subend]
            sampled=principal_scores[...,ix].transpose(3,0,1,2,4)
            yy=y[ix][:,None,None,None,:]
            values=explicit_reference(yy,sampled,only_primary=True)
            vals=values.mean(axis=(1,3))
            delta=np.stack([vals[:,s['aa'],s['ki']]-vals[:,s['bb'],s['ki']] for s in principal],axis=1)
            counts+=(np.abs(delta-observed_primary)>=np.abs(observed_primary)-1e-12).sum(axis=0)
        # Independently rebuild all 2000 percentile interval draws, including
        # the mean-pooling model, operating metrics and alternative thresholds.
        if first<2000:
            upto=min(size,2000-first)
            for subfirst in range(0,upto,128):
                subend=min(upto,subfirst+128);ix=indices[subfirst:subend]
                sampled=ensemble[...,ix].transpose(3,0,1,2,4);yy=y[ix][:,None,None,None,:]
                for pi in range(2):
                    values=explicit_reference(yy,sampled,thresholds[pi][None,...,None])[...,:14].mean(axis=1)
                    values=np.concatenate([values,values.mean(axis=2,keepdims=True)],axis=2)
                    bootstrap_values[first+subfirst:first+subend,pi]=values
        if first==0 or first+size==sa.N_BOOTSTRAP or first//sa.BATCH_SIZE%25==0:
            print('Independent explicit-row bootstrap:',first+size,'/',sa.N_BOOTSTRAP,flush=True)
    independent_counts={(s['comparison'],s['metric']):int(v) for s,v in zip(principal,counts)}
    ci=np.quantile(bootstrap_values,[.025,.975],axis=0);interval_checks=0
    for pi,policy in enumerate(POLICIES):
      for mi,model in enumerate(LABELS):
       for ti,t in enumerate(TIMES):
        row=summary[policy,model,t]
        for ki,key in enumerate(METRICS[:14]):
            np.testing.assert_allclose([float(row[key+'_patient_ci_low']),float(row[key+'_patient_ci_high'])],ci[:,pi,mi,ti,ki],atol=1e-12,rtol=0);interval_checks+=2
    paired_rows=read('extended_paired_bootstrap.csv')
    for row in paired_rows:
        pi=POLICIES.index(row['policy']);ti=TIMES.index(row['timepoint']);ki=METRICS.index(row['metric'])
        a,b=row['comparison'].split(' minus ');aa,bb=LABELS.index(a),LABELS.index(b)
        expected=np.quantile(bootstrap_values[:,pi,aa,ti,ki]-bootstrap_values[:,pi,bb,ti,ki],[.025,.975])
        np.testing.assert_allclose([float(row['ci_low']),float(row['ci_high'])],expected,atol=1e-12,rtol=0)
        np.testing.assert_allclose(float(row['delta']),means[pi,aa,ti,ki]-means[pi,bb,ti,ki],atol=1e-12,rtol=0)
        np.testing.assert_allclose(float(row['delta_fold_sd']),(per_seed[pi,:,aa,ti,ki]-per_seed[pi,:,bb,ti,ki]).std(ddof=1),atol=1e-12,rtol=0)
        interval_checks+=2
    temporal_rows=read('extended_longitudinal.csv')
    for row in temporal_rows:
        pi=POLICIES.index(row['policy']);mi=LABELS.index(row['model']);ki=METRICS.index(row['metric'])
        expected=np.quantile(bootstrap_values[:,pi,mi,2,ki]-bootstrap_values[:,pi,mi,0,ki],[.025,.975])
        np.testing.assert_allclose([float(row['ci_low']),float(row['ci_high'])],expected,atol=1e-12,rtol=0)
        np.testing.assert_allclose(float(row['delta_fold_sd']),(per_seed[pi,:,mi,2,ki]-per_seed[pi,:,mi,0,ki]).std(ddof=1),atol=1e-12,rtol=0)
        interval_checks+=2
    print('All patient, paired and longitudinal percentile intervals independently matched.',flush=True)

    dca_checks=0;dca_error=0.
    for row in read('extended_dca.csv'):
        mi=LABELS.index(row['model']);time=row['timepoint'];q=float(row['threshold']);nb=[]
        for si in range(5):
            vals=[]
            for ti in range(3):
                selected=ensemble[si,mi,ti]>=q
                tp=np.count_nonzero(selected&(y==1));fp=np.count_nonzero(selected&(y==0))
                vals.append((tp-fp*q/(1-q))/len(y))
            nb.append(np.mean(vals) if time=='Macro' else vals[TIMES.index(time)])
        expected=float(np.mean(nb));err=abs(float(row['net_benefit'])-expected)
        assert err<1e-12;dca_error=max(dca_error,err)
        np.testing.assert_allclose(float(row['net_benefit_fold_sd']),np.std(nb,ddof=1),atol=1e-12,rtol=0)
        np.testing.assert_allclose(float(row['treat_all']),y.mean()-(1-y.mean())*q/(1-q),atol=1e-12,rtol=0)
        dca_checks+=1
    rows=read('significance_tests.csv');assert len(rows)==90
    assert len({(r['comparison'],r['timepoint'],r['metric']) for r in rows})==90
    for family,expected in [(sa.PRIMARY,9),(sa.SECONDARY,81)]:
        selected=[r for r in rows if r['family']==family];assert len(selected)==expected
        adjustments=reference_holm([float(r['p_raw']) for r in selected])
        for row,p_adj in zip(selected,adjustments):
            assert row['seed']==SEED_TEXT and row['policy']=='fixed_0.5'
            p=(1+int(row['bootstrap_exceedances']))/(1+int(row['bootstrap_replicates']))
            assert abs(float(row['p_raw'])-p)<1e-15
            assert abs(float(row['p_holm_family'])-p_adj)<1e-15
            star='***' if p_adj<.001 else '**' if p_adj<.01 else '*' if p_adj<.05 else 'ns'
            assert row['significance']==star
            if family==sa.PRIMARY:assert int(row['bootstrap_exceedances'])==independent_counts[row['comparison'],row['metric']]
    for row,p_adj in zip(rows,reference_holm([float(r['p_raw']) for r in rows])):
        assert abs(float(row['p_holm_all_90'])-p_adj)<1e-15
        assert row['significance_all_90']==('***' if p_adj<.001 else '**' if p_adj<.01 else '*' if p_adj<.05 else 'ns')
    provenance=json.loads((ROOT/'data/significance_provenance.json').read_text())
    assert provenance['analysis_script_sha256']==sha(sa.__file__)
    assert provenance['source_provenance_sha256']==sha(ROOT/'data/extended_provenance.json')
    receipt=dict(verified_at_utc=datetime.now(timezone.utc).isoformat(),numpy_version=np.__version__,scipy_version=scipy.__version__,sklearn_version=sklearn.__version__,training_seed=44,folds=FOLDS,aggregation='metric per fold prediction vector -> equal fivefold metric mean; fold sample SD ddof1; no probability ensembling',source_runs=20,point_scalar_metrics_verified_with_sklearn=point_checks,point_max_absolute_error=max_error,fold_mean_and_sd_scalar_checks=summary_checks,fold_validation_youden_thresholds_independently_verified=threshold_checks,weighted_bootstrap_reference_draws=20,weighted_bootstrap_scalar_metric_checks=weighted_checks,weighted_bootstrap_max_absolute_error=metric_boot_error,percentile_bootstrap_draws_independently_rebuilt=2000,all_14_percentile_interval_endpoints_checked=interval_checks,paired_contrast_rows_verified=len(paired_rows),longitudinal_contrast_rows_verified=len(temporal_rows),dca_rows_verified=dca_checks,dca_max_absolute_error=dca_error,actual_primary_bootstrap_draws_independently_recomputed=sa.N_BOOTSTRAP,actual_primary_exceedance_counts_matched=9,independent_primary_method='Explicit repeated patient rows; scipy rankdata average-rank AUC, positive/all threshold-rank AP, confusion-count F1. Compute separately for five folds and three landmarks, then mean.',output_significance_rows_verified=90,family_counts={'principal_macro_9':9,'secondary_81':81},raw_p_values_recalculated=90,family_holm_values_verified=90,global_holm_values_verified=90,all_significance_markers_verified=True,source_hashes_verified=True,analysis_script_sha256=sha(sa.__file__),verification_script_sha256=sha(__file__),significance_csv_sha256=sha(ROOT/'data/significance_tests.csv'),elapsed_seconds=round(run_time.monotonic()-started,3),aggregate_only=True,patient_ids_exported=False)
    (ROOT/'notes/numerical_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
