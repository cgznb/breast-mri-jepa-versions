#!/usr/bin/env python3
"""Receipt-backed seed44 predictions, retained separately for each fold model."""
from pathlib import Path
import hashlib,json
import numpy as np
from extended_metrics import all_metrics,select_youden,METRICS

ROOT=Path(__file__).resolve().parents[1]
SEEDS=[44]
SEED_TEXT='44'
FOLDS=[1,2,3,4,5]
SUITE=Path('/projects/xts0030/public/cxx/medium976_t0_t2_no_tf_cv5_multiseed_v1')
MODELS=['direct_no_jepa','direct_readout','forecast_attention','forecast_mean']
LABELS=['NoJEPA','DirectReadout','ForecastAttention','ForecastMean']
TIMES=['T0','T1','T2','Macro']
POLICIES=['fixed_0.5','fold_validation_youden']

def run_dir(seed,model,fold):
    assert seed==44 and fold in FOLDS
    return SUITE/'runs'/f'{model}_seed{seed}_fold{fold}'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_data():
    matrix=json.loads((SUITE/'matrix.json').read_text())
    assert matrix['status']=='completed' and matrix['test_visible']
    prob=np.empty((4,5,3,175));thresholds=np.full((2,5,4,3),.5)
    sources=[];validation=[];ids_ref=y=None;reference_val={};reference_manifest={}
    for mi,model in enumerate(MODELS):
      for fi,fold in enumerate(FOLDS):
        run=run_dir(44,model,fold)
        receipt_path=run/'evaluation_done.json';receipt=json.loads(receipt_path.read_text());assert receipt['completed'] is True
        config_path=run/'config.json';config=json.loads(config_path.read_text());assert config['seed']==44
        manifest_path=Path(receipt['manifest']['path']);st=manifest_path.stat()
        assert (st.st_size,st.st_mtime_ns)==(receipt['manifest']['bytes'],receipt['manifest']['mtime_ns'])
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
        sources.append(dict(seed=44,model=LABELS[mi],fold=fold,suite=str(SUITE),receipt=str(receipt_path),receipt_sha256=sha(receipt_path),config=str(config_path),config_sha256=sha(config_path),manifest=str(manifest_path),manifest_sha256=sha(manifest_path),test_report=str(test_path),test_sha256=sha(test_path),validation_report=str(val_path),validation_sha256=sha(val_path),registered_reports=report_ids))
        for ti in range(3):
            rows=sorted((r for r in test['rows'] if r['landmark']==ti),key=lambda r:r['patient_key'])
            ids=[r['patient_key'] for r in rows];yy=np.array([r['pcr'] for r in rows],int);pp=np.array([r['pcr_probability'] for r in rows],float)
            if y is None:ids_ref,y=ids,yy
            assert ids==ids_ref and np.array_equal(yy,y) and len(set(ids))==175 and yy.sum()==56
            assert all(patients[k]['split']=='test' and patients[k]['target']['pcr']==v for k,v in zip(ids,yy))
            assert np.isfinite(pp).all() and ((pp>=0)&(pp<=1)).all()
            prob[mi,fi,ti]=pp
            check=all_metrics(y,pp,np.ones((1,175)))[0]
            for key in ['auroc','auprc','f1','brier','nll','ece_10_equal_width']:
                assert abs(check[METRICS.index(key)]-test['per_landmark'][TIMES[ti]][key])<1e-12
            vr=sorted((r for r in val['rows'] if r['landmark']==ti),key=lambda r:r['patient_key'])
            vids=[r['patient_key'] for r in vr];vy=np.array([r['pcr'] for r in vr]);vp=np.array([r['pcr_probability'] for r in vr])
            assert len(vids)==len(set(vids)) and not(set(vids)&set(ids_ref))
            assert all(patients[r['patient_key']]['split']=='val' and patients[r['patient_key']]['target']['pcr']==r['pcr'] for r in vr)
            if (fi,ti) not in reference_val:reference_val[fi,ti]=(vids,vy)
            assert vids==reference_val[fi,ti][0] and np.array_equal(vy,reference_val[fi,ti][1])
            threshold,j=select_youden(vy,vp);thresholds[1,fi,mi,ti]=threshold
            validation.append(dict(policy=POLICIES[1],seed=44,fold=fold,model=LABELS[mi],timepoint=TIMES[ti],validation_patients=len(vy),validation_positives=int(vy.sum()),threshold=threshold,validation_youden_j=j,score_type='this fold model validation scores',application_score='same fold model holdout scores',tie_rule='largest threshold',caveat='Validation also selected checkpoint; sensitivity analysis, not nested calibration.'))
    assert len(sources)==20 and sum(len(s['registered_reports']) for s in sources)==60
    return dict(y=y,prob=prob,scores=prob.transpose(1,0,2,3),thresholds=thresholds,sources=sources,validation=validation)
