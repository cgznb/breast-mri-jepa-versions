#!/usr/bin/env python3
"""Audited loaders for four matched seed-specific fivefold ensembles."""
from pathlib import Path
import hashlib,json
import numpy as np
from extended_metrics import all_metrics,select_youden,METRICS

ROOT=Path(__file__).resolve().parents[1]
SEEDS=[42,44,2026,3407]
SEED_TEXT=','.join(map(str,SEEDS))
BASE=Path('/projects/xts0030/public/cxx')
SUITES={42:BASE/'medium976_t0_t2_no_tf_add3seed_17_29_42_v1',44:BASE/'medium976_t0_t2_no_tf_cv5_multiseed_v1',2026:BASE/'medium976_t0_t2_no_tf_add4seed_30_43_809_2026_v1',3407:BASE/'medium976_t0_t2_no_tf_add5seed_7_71_123_3407_2027_v1'}
MODELS=['direct_no_jepa','direct_readout','forecast_attention','forecast_mean']
LABELS=['NoJEPA','DirectReadout','ForecastAttention','ForecastMean']
TIMES=['T0','T1','T2','Macro']
POLICIES=['fixed_0.5','oof_validation_youden']

def run_dir(seed,model,fold):return SUITES[seed]/'runs'/f'{model}_seed{seed}_fold{fold}'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_data():
    prob=np.empty((len(SEEDS),len(MODELS),5,3,175)); thresholds=np.full((2,len(SEEDS),len(MODELS),3),.5)
    sources=[];validation=[];ids_ref=y=None;reference_oof=[None]*3
    reference_config={};reference_manifest={}
    for seed in SEEDS:
        matrix=json.loads((SUITES[seed]/'matrix.json').read_text())
        assert matrix['status']=='completed' and matrix['test_visible']
    for si,seed in enumerate(SEEDS):
      for mi,model in enumerate(MODELS):
        oof=[[] for _ in range(3)]
        for fi in range(5):
            run=run_dir(seed,model,fi+1)
            receipt_path=run/'evaluation_done.json'; receipt=json.loads(receipt_path.read_text())
            assert receipt['completed'] is True
            config_path=run/'config.json';config=json.loads(config_path.read_text());assert config['seed']==seed
            comparable={k:v for k,v in config.items() if k!='seed'}
            if (mi,fi) not in reference_config:reference_config[mi,fi]=comparable
            assert comparable==reference_config[mi,fi]
            manifest_path=Path(receipt['manifest']['path']);stat=manifest_path.stat()
            assert (stat.st_size,stat.st_mtime_ns)==(receipt['manifest']['bytes'],receipt['manifest']['mtime_ns'])
            if fi not in reference_manifest:reference_manifest[fi]=sha(manifest_path)
            assert sha(manifest_path)==reference_manifest[fi]
            manifest=json.loads(manifest_path.read_text());patients={r['patient_key']:r for r in manifest['patients']};assert len(patients)==976
            report_ids=[]
            for identity in receipt['reports']:
                path=Path(identity['path']);st=path.stat();assert (st.st_size,st.st_mtime_ns)==(identity['bytes'],identity['mtime_ns'])
                report_ids.append(dict(path=str(path),bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=sha(path)))
            test_path=run/'evaluation_best_test.json';val_path=run/'evaluation_best_val.json'
            test=json.loads(test_path.read_text());val=json.loads(val_path.read_text())
            assert test['split']=='test' and test['downstream_holdout'] and test['independent_test'] is False
            assert val['split']=='val'
            sources.append(dict(seed=seed,model=LABELS[mi],fold=fi+1,suite=str(SUITES[seed]),receipt=str(receipt_path),receipt_sha256=sha(receipt_path),config=str(config_path),config_sha256=sha(config_path),manifest=str(manifest_path),manifest_sha256=sha(manifest_path),test_report=str(test_path),test_sha256=sha(test_path),validation_report=str(val_path),validation_sha256=sha(val_path),registered_reports=report_ids))
            for ti in range(3):
                rows=sorted((r for r in test['rows'] if r['landmark']==ti),key=lambda r:r['patient_key'])
                ids=[r['patient_key'] for r in rows]; yy=np.array([r['pcr'] for r in rows],int);pp=np.array([r['pcr_probability'] for r in rows],float)
                if y is None:ids_ref,y=ids,yy
                assert ids==ids_ref and np.array_equal(yy,y) and len(set(ids))==175 and yy.sum()==56
                assert all(patients[k]['split']=='test' and patients[k]['target']['pcr']==v for k,v in zip(ids,yy))
                assert np.isfinite(pp).all() and ((pp>=0)&(pp<=1)).all()
                prob[si,mi,fi,ti]=pp
                check=all_metrics(y,pp,np.ones((1,175)))[0]
                for key in ['auroc','auprc','f1','brier','nll','ece_10_equal_width']:
                    assert abs(check[METRICS.index(key)]-test['per_landmark'][TIMES[ti]][key])<1e-12
                vr=[r for r in val['rows'] if r['landmark']==ti]
                assert not (set(ids)&{r['patient_key'] for r in vr})
                assert all(patients[r['patient_key']]['split']=='val' and patients[r['patient_key']]['target']['pcr']==r['pcr'] for r in vr)
                oof[ti].extend(vr)
        for ti in range(3):
            vr=sorted(oof[ti],key=lambda r:r['patient_key']);ids=[r['patient_key'] for r in vr];vy=np.array([r['pcr'] for r in vr]);vp=np.array([r['pcr_probability'] for r in vr])
            assert len(ids)==len(set(ids))==[801,760,700][ti] and not(set(ids)&set(ids_ref))
            if reference_oof[ti] is None:reference_oof[ti]=(ids,vy)
            assert ids==reference_oof[ti][0] and np.array_equal(vy,reference_oof[ti][1])
            threshold,j=select_youden(vy,vp);thresholds[1,si,mi,ti]=threshold
            validation.append(dict(policy=POLICIES[1],seed=seed,model=LABELS[mi],timepoint=TIMES[ti],validation_patients=len(vy),validation_positives=int(vy.sum()),folds=5,threshold=threshold,validation_youden_j=j,score_type='one held-out-fold model score per development patient',application_score='within-seed mean of five fold probabilities',tie_rule='largest threshold',caveat='Single-model OOF versus five-model ensemble score mismatch; sensitivity only.'))
    assert len(sources)==len(SEEDS)*len(MODELS)*5 and sum(len(s['registered_reports']) for s in sources)==len(sources)*3
    return dict(y=y,prob=prob,ensemble=prob.mean(axis=2),thresholds=thresholds,sources=sources,validation=validation)
