#!/usr/bin/env python3
"""Assemble only the v8 frozen code and result artifacts; no input data/weights."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

REPO=Path('/personal_fast_storage/VLA-PCRV1')
PAPER=REPO/'manuscripts/mia_mean_seed42_44_71_2026_probmean_v8'
OUT=REPO/'private_outputs/mia_v8_code_results_20261105'
AUDIT=json.loads((PAPER/'notes/source_audit.json').read_text())
PROV=json.loads((PAPER/'data/extended_provenance.json').read_text())
EXCLUDED={'.pt','.pth','.ckpt','.safetensors','.npz','.npy','.nii','.dcm','.xlsx','.xls','.parquet'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def identity(path):
    s=path.stat()
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def main():
    os.umask(0o077)
    if OUT.exists():raise RuntimeError('Destination already exists')
    assert len(PROV['sources'])==60
    tasks=[]
    def add(src,rel,category):
        src=Path(src)
        assert src.is_file() and not src.is_symlink(),src
        assert src.suffix.lower() not in EXCLUDED and not src.name.endswith('.nii.gz'),src
        tasks.append((src,Path(rel),category))
    def tree(src,rel,category):
        for p in sorted(src.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':
                add(p,Path(rel)/p.relative_to(src),category)
    plans={int(seed):Path('/projects/xts0030/public/cxx/planned_tasks')/Path(suite).name
           for seed,suite in AUDIT['suite_by_seed'].items()}
    runtime_hashes={}
    for seed,plan in plans.items():
        root=plan/'runtime'
        runtime_hashes[seed]={str(p.relative_to(root)):sha(p) for p in root.rglob('*')
                             if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
        assert len(runtime_hashes[seed])==92
    assert all(hashes==runtime_hashes[44] for hashes in runtime_hashes.values())
    assert runtime_hashes[44]['src/mri_vla_jepa/model.py']==AUDIT['mean_readout']['source_sha256']
    tree(plans[44]/'runtime','code/runtime','frozen_training_source')
    tree(REPO/'tools/medium976','code/original_scheduler_tools','scheduler_source')
    tree(PAPER/'scripts','code/analysis_original','original_analysis_source')
    tree(PAPER,'manuscript','original_manuscript_snapshot')
    tree(PAPER/'data','results/aggregated','final_aggregate_result')
    tree(PAPER/'figures','results/figures','final_figure')
    tree(PAPER/'tables','results/tables','final_latex_table')
    for name in ['source_audit.json','source_audit_ZH.md','methods_evidence.md','numerical_verification.json','build_verification.json']:
        add(PAPER/'notes'/name,Path('provenance/paper_audit')/name,'original_paper_audit')
    for name in ['AGENTS.md','LICENSE']:
        if (REPO/name).exists():add(REPO/name,Path('provenance/project_documents')/name,'project_document')
    for name in ['pyproject.toml','LICENSE']:
        p=REPO/'versions/v1_evolution'/name
        if p.exists():add(p,Path('code/environment_reference')/name,'current_environment_reference')
    licenses=REPO/'versions/v1_evolution/licenses'
    if licenses.exists():tree(licenses,'code/environment_reference/licenses','license_notice')
    index=[];verified_hashes=0
    for entry in PROV['sources']:
        run=Path(entry['receipt']).parent
        rc=json.loads((run/'evaluation_done.json').read_text())
        assert rc['completed'] is True
        assert json.loads((run/'training_done.json').read_text())['completed'] is True
        for name,key in [('evaluation_done.json','receipt_sha256'),('config.json','config_sha256'),
                         ('evaluation_best_test.json','test_sha256'),('evaluation_best_val.json','validation_sha256')]:
            assert sha(run/name)==entry[key],(entry['seed'],entry['model'],entry['fold'],name)
            verified_hashes+=1
        for report in entry['registered_reports']:
            path=Path(report['path'])
            assert sha(path)==report['sha256']
            assert (path.stat().st_size,path.stat().st_mtime_ns)==(report['bytes'],report['mtime_ns'])
            verified_hashes+=1
        seed=entry['seed'];fold=entry['fold'];name=run.name
        assert all(p.name!='last.pt' for p in run.iterdir())
        for p in sorted(run.iterdir()):
            # Reports, diagnostics, training-history results and run receipts only.
            if p.is_file() and p.suffix in {'.json','.log'} and not p.name.startswith('.'):
                add(p,Path(f'experiments/seed{seed}/runs')/name/p.name,'selected_run_result')
        add(run/'config.json',Path('configs')/f'{name}.json','original_run_config')
        cfg=json.loads((run/'config.json').read_text())
        index.append(dict(model=entry['model'],variant=name.split('_seed')[0],seed=seed,fold=fold,
            config=f'configs/{name}.json',reports=f'experiments/seed{seed}/runs/{name}',
            pcr_readout=cfg['model']['pcr_readout'],spatial_readout=cfg['model']['spatial_readout'],
            rollout_weight=cfg['rollout_weight'],jepa_weight=cfg['jepa_weight']))
    for seed,suite in AUDIT['suite_by_seed'].items():
        matrix=Path(suite)/'matrix.json'
        a=json.loads(matrix.read_text())
        assert a['status']=='completed' and a['test_visible']
        add(matrix,f'experiments/seed{seed}/matrix.json','original_suite_receipt')
    assert len({str(rel) for _,rel,_ in tasks})==len(tasks)
    OUT.mkdir(mode=0o700,parents=True)
    records=[]
    print(json.dumps({'phase':'copy_start','files':len(tasks),'bytes':sum(p.stat().st_size for p,_,_ in tasks)}),flush=True)
    for src,rel,category in tasks:
        before=identity(src);digest=sha(src)
        dst=OUT/rel;dst.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        shutil.copy2(src,dst)
        assert identity(src)==before and sha(dst)==digest,rel
        dst.chmod(0o600)
        records.append(dict(source=str(src),path=str(rel),category=category,bytes=before[2],
                            sha256=digest,source_mtime_ns=before[3],source_unchanged=True))
    with (OUT/'provenance/source_copy_inventory.jsonl').open('w') as stream:
        for record in records:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    write_json(OUT/'provenance/selected_runs.json',index)
    with (OUT/'configs/model_index.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(index[0]));writer.writeheader();writer.writerows(index)
    write_json(OUT/'provenance/runtime_identity_by_seed.json',runtime_hashes)
    version={}
    for name,args in [('git_commit',['git','rev-parse','HEAD']),('git_branch',['git','branch','--show-current'])]:
        version[name]=subprocess.check_output(args,cwd=REPO,text=True).strip()
    write_json(OUT/'provenance/package_scope.json',dict(schema='mia_v8_code_results_only_v1',
        created_at_utc=datetime.now(timezone.utc).isoformat(),paper_source=str(PAPER),bundle=str(OUT),
        models=['NoJEPA','DirectReadout','ForecastMean'],training_seeds=[42,44,71,2026],folds=[1,2,3,4,5],
        fitted_runs=60,registered_reports=180,holdout_patients=175,holdout_positives=56,
        included=['frozen_source','original_configs','aggregate_result_tables','figures','selected_run_reports','diagnostics','history','receipts','manuscript'],
        excluded=['MRI/NPZ','training_input_manifests','clinical_source_tables','pretrained_assets','fitted_checkpoint_weights','virtual_environments'],
        original_source_sha256_checks=verified_hashes,frozen_runtime_files=92,runtime_identity_equal_across_four_suites=True,
        sources_modified=False,symlinks_created=False,training_started=False,gpu_used=False,checkpoint_payloads_loaded=False,
        source_copy_files=len(records),source_copy_bytes=sum(r['bytes'] for r in records),version=version))
    print(json.dumps({'phase':'copy_complete','files':len(records),'bytes':sum(r['bytes'] for r in records),
                      'source_sha256_checks':verified_hashes,'source_and_copy_verified':True}),flush=True)

if __name__=='__main__':main()
