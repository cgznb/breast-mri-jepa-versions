#!/usr/bin/env python3
"""Run the original Medium T0-T2 no-TF jobs on the fixed Medium976 folds."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid

import admission
import capacity

TASK = Path('/projects/xts0030/public/cxx/planned_tasks/medium976_t0_t2_no_tf_cv5_multiseed_v1')
DEFAULT_SPEC = TASK / 'training_spec.json'
PLANNED_JOBS = 60
CHECKPOINT_RETENTION = 'best_only_after_verified_evaluation'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.chmod(0o600)
    temporary.replace(path)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def validate_input_identities(spec):
    for key in ('runtime_identities', 'config_identities', 'manifest_identities', 'asset_identities'):
        for expected in spec.get(key, []):
            if identity(expected['path']) != expected:
                raise ValueError(f'Prepared input changed: {expected["path"]}')


def select_jobs(spec, variants=None, seeds=None, folds=None):
    selections = [(variants, spec['variants'], 'variant'),
                  (seeds, spec['train_seeds'], 'seed'), (folds, spec['fold_ids'], 'fold')]
    for requested, available, name in selections:
        if requested is not None and (not requested or len(set(requested)) != len(requested)
                                       or not set(requested) <= set(available)):
            raise ValueError(f'Unknown or duplicate {name} selection')
    return [job for job in spec['jobs'] if (variants is None or job['variant'] in variants)
            and (seeds is None or job['seed'] in seeds) and (folds is None or job['fold_id'] in folds)]


def validate_gpus(gpus):
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError('Choose distinct physical GPUs')
    if any(not re.fullmatch(r'(?:\d+|GPU-[a-fA-F0-9-]+)', gpu) for gpu in gpus):
        raise ValueError('GPU identifiers must be physical indices or full GPU UUIDs')


def import_runtime(spec):
    runtime = Path(spec['runtime_src']).resolve()
    sys.path.insert(0, str(runtime))
    from mri_vla_jepa import data
    if not Path(data.__file__).resolve().is_relative_to(runtime):
        raise ValueError('The prepared private runtime was not imported')


def validate_plan(spec):
    if (spec['schema'] != 'medium976_t0_t2_no_tf_training_spec_v1'
            or len(spec['jobs']) != PLANNED_JOBS or spec['training_runs'] != PLANNED_JOBS
            or spec['train_seeds'] != [21, 44, 203]
            or spec['variants'] != ['forecast_attention', 'direct_no_jepa', 'direct_readout', 'forecast_mean']
            or spec['fold_ids'] != [1, 2, 3, 4, 5]
            or spec.get('checkpoint_retention') != CHECKPOINT_RETENTION):
        raise ValueError('Expected the fixed 60-job, three-seed Medium976 best-only plan')
    expected = {(v, s, f) for v in spec['variants'] for s in spec['train_seeds'] for f in spec['fold_ids']}
    actual = {(j['variant'], j['seed'], j['fold_id']) for j in spec['jobs']}
    names = [j['name'] for j in spec['jobs']]
    if actual != expected or len(set(names)) != PLANNED_JOBS:
        raise ValueError('The complete 60-job variant/seed/fold grid is required')


def preflight(spec, selected):
    validate_plan(spec)
    validate_input_identities(spec)
    import_runtime(spec)
    from mri_vla_jepa.data import RawMRIStore
    from mri_vla_jepa.landmark_profiles import allowed_landmarks
    from mri_vla_jepa.train_config import load_config
    stores, combinations, encoder_assets = {}, set(), set()
    output_root = Path(spec['output_root']).resolve()
    for job in spec['jobs']:
        combination = (job['variant'], job['seed'], job['fold_id'])
        if combination in combinations or Path(job['output']).resolve() != output_root / 'runs' / job['name']:
            raise ValueError('Duplicated job or output outside the isolated experiment')
        combinations.add(combination)
        cfg = load_config(job['config'])
        if (cfg.landmarks != 't0_t2' or cfg.prediction_training != 'hybrid'
                or cfg.jepa_weight != 0 or cfg.image_shape != (32, 128, 128)
                or cfg.initialize_from_checkpoint is not None or cfg.encoder_training != 'finetune'
                or cfg.seed != job['seed'] or cfg.device != 'cuda:0'
                or cfg.model.model_size != 'medium' or cfg.image_cache is not None):
            raise ValueError(f'Original Medium no-TF semantics changed: {job["name"]}')
        if not Path(cfg.encoder_checkpoint).is_file():
            raise ValueError('The original pretrained VQ encoder is missing')
        encoder_assets.add(str(Path(cfg.encoder_checkpoint).resolve()))
        expected_manifest = Path(spec['dataset_root']) / 'cv_manifests' / f'fold{job["fold_id"]}' / 'manifest.json'
        if Path(job['manifest']).resolve() != expected_manifest.resolve():
            raise ValueError('Job does not use the registered fixed-fold manifest')
        if job['fold_id'] not in stores:
            store = RawMRIStore(job['manifest'], cfg.image_shape)
            if len(store.patients) != 976 or len(store.by_split['test']) != 175:
                raise ValueError('Wrong patient population')
            if sum(v is not None for p in store.patients for v in p['visits']) != 3641:
                raise ValueError('Wrong effective visit population')
            if any(not all(p['visits'][i] is not None for i in (0, 1, 2))
                   for p in store.patients if p['split'] == 'test'):
                raise ValueError('The test175 must have actual T0/T1/T2')
            if any(3 in allowed_landmarks(store, i, 't0_t2') for i in range(len(store.patients))):
                raise ValueError('T3 entered the T0-T2 training profile')
            stores[job['fold_id']] = store
    expected = {(v, s, f) for v in spec['variants'] for s in spec['train_seeds'] for f in spec['fold_ids']}
    if combinations != expected:
        raise ValueError('The full variant/seed/fold grid changed')
    if encoder_assets != {item['path'] for item in spec.get('asset_identities', [])}:
        raise ValueError('The original VQ codec payload must be pinned in asset_identities')
    return {'status': 'passed', 'checked_at_beijing': now(), 'planned_jobs': PLANNED_JOBS,
            'selected_jobs': len(selected), 'folds': len(stores), 'patients': 976, 'visits': 3641,
            'test': 175, 'development': 801, 'landmarks': 't0_t2', 'image_shape': [32, 128, 128],
            'model_constructed': False, 'gpu_training_started': False,
            'test_visibility_gate_jobs': PLANNED_JOBS, 'checkpoint_retention': CHECKPOINT_RETENTION}


def run_identity(spec, job):
    return {'schema': 'medium976_no_tf_run_identity_v2', 'job': job,
            'config': identity(job['config']), 'manifest': identity(job['manifest']),
            'runtime_identities': spec['runtime_identities'],
            'asset_identities': spec.get('asset_identities', []),
            'entrypoint_identity': identity(Path(__file__)),
            'checkpoint_retention': spec.get('checkpoint_retention'),
            'allocator_policy': capacity.policy_identity(),
            'admission_identity': identity(Path(admission.__file__))}


def training_complete(job, spec=None):
    run = Path(job['output'])
    required = [run / name for name in ('train_report.json', 'best.pt',
                                        'run_identity.json', 'training_done.json')]
    if not all(path.is_file() for path in required):
        return False
    contract = read(run / 'run_identity.json')
    if spec is not None and contract != run_identity(spec, job):
        raise ValueError('Completed training belongs to different prepared inputs')
    if (contract.get('job') != job or contract.get('manifest') != identity(job['manifest'])
            or contract.get('config') != identity(job['config'])):
        raise ValueError('Run identity differs from the current job')
    done, report = read(run / 'training_done.json'), read(run / 'train_report.json')
    if not (report.get('completed') is True and report.get('landmarks') == 't0_t2'
            and done.get('completed') is True and done.get('run_identity') == identity(run / 'run_identity.json')
            and isinstance(done.get('files'), list) and len(done['files']) == 3
            and done['files'][:2] == [identity(run / name) for name in ('train_report.json', 'best.pt')]
            and done['files'][2].get('path') == str((run / 'last.pt').resolve())):
        return False
    try:
        last_identity = identity(run / 'last.pt')
    except FileNotFoundError:
        # Another evaluation worker may finish its journaled cleanup while
        # this controller or worker is checking the all-training gate.
        last_identity = None
    if last_identity is not None:
        return done['files'][2] == last_identity
    if not retention_receipt_valid(job, done):
        raise ValueError('Completed training has no last.pt and no valid best-only retention receipt')
    return True


@lru_cache(maxsize=10)
def manifest_patients(path, size, modified):
    return read(path)['patients']


def validate_evaluation_report(job, label, report):
    split = 'test' if label == 'best_test' else 'val'
    source = identity(job['manifest'])
    patients = manifest_patients(source['path'], source['bytes'], source['mtime_ns'])
    expected = {(p['patient_key'], i): p['target']['pcr'] for p in patients if p['split'] == split
                for i, visit in enumerate(p['visits'][:3]) if visit is not None}
    if (report.get('split') != split or report.get('landmarks') != 't0_t2'
            or report.get('checkpoint_landmarks') != 't0_t2' or report.get('synthetic') is not False
            or report.get('metric_name') != 'f1' or report.get('f1_threshold') != 0.5):
        raise ValueError('Evaluation split/profile/selection differs')
    rows = report.get('rows', [])
    actual = {(row['patient_key'], row['landmark']): row['pcr'] for row in rows}
    if len(rows) != len(actual) or actual != expected:
        raise ValueError('Evaluation patient/landmark/label coverage differs')
    if any(not math.isfinite(r['pcr_probability']) or not 0 <= r['pcr_probability'] <= 1 for r in rows):
        raise ValueError('Evaluation probabilities are invalid')
    if label == 'best_val':
        selected = read(Path(job['output']) / 'train_report.json')['best_selection_score']
        if not math.isclose(report['selection_score'], selected, rel_tol=0, abs_tol=1e-7):
            raise ValueError('Replayed best validation score differs from the selected checkpoint')


EVALUATION_CACHE = set()


def evaluation_artifacts_complete(job):
    """Validate formal reports independently of checkpoint-retention state."""
    path = Path(job['output']) / 'evaluation_done.json'
    if not path.is_file():
        return False
    receipt = read(path)
    expected_paths = [path.parent / f'evaluation_{label}.json' for label in ('best_val', 'last_val', 'best_test')]
    if not all(p.is_file() for p in expected_paths + [path.parent / 'best.pt', path.parent / 'train_report.json',
                  path.parent / 'training_done.json', path.parent / 'run_identity.json']):
        return False
    done = read(path.parent / 'training_done.json')
    if (receipt.get('completed') is not True
            or receipt.get('job') != job['name'] or receipt.get('manifest') != identity(job['manifest'])
            or receipt.get('reports') != [identity(p) for p in expected_paths]
            or receipt.get('best_checkpoint') != identity(path.parent / 'best.pt')
            or receipt.get('training_done') != identity(path.parent / 'training_done.json')
            or receipt.get('run_identity') != identity(path.parent / 'run_identity.json')
            or len(done.get('files', [])) != 3 or receipt.get('last_checkpoint') != done['files'][2]):
        return False
    cache_key = (str(path), identity(path)['mtime_ns'], tuple(p.stat().st_mtime_ns for p in expected_paths),
                 (path.parent / 'train_report.json').stat().st_mtime_ns)
    if cache_key not in EVALUATION_CACHE:
        for label, target in zip(('best_val', 'last_val', 'best_test'), expected_paths):
            validate_evaluation_report(job, label, read(target))
        EVALUATION_CACHE.add(cache_key)
    return True


def retention_contract(job, done):
    run = Path(job['output'])
    return {'schema': 'medium976_checkpoint_retention_v1', 'job': job['name'],
            'policy': CHECKPOINT_RETENTION, 'run_identity': identity(run / 'run_identity.json'),
            'training_done': identity(run / 'training_done.json'), 'best_checkpoint': identity(run / 'best.pt'),
            'last_checkpoint': done['files'][2], 'evaluation_receipt': identity(run / 'evaluation_done.json')}


def retention_receipt_valid(job, done):
    run = Path(job['output'])
    path = run / 'checkpoint_retention.json'
    if not path.is_file() or not evaluation_artifacts_complete(job):
        return False
    receipt = read(path)
    expected = retention_contract(job, done)
    return (receipt.get('status') in ('pending_delete', 'completed')
            and all(receipt.get(key) == value for key, value in expected.items()))


def evaluation_complete(job, spec=None):
    run = Path(job['output'])
    return (training_complete(job, spec) and evaluation_artifacts_complete(job)
            and not (run / 'last.pt').exists()
            and retention_receipt_valid(job, read(run / 'training_done.json'))
            and read(run / 'checkpoint_retention.json').get('status') == 'completed')


def finalize_checkpoint_retention(spec, job):
    """Delete last only after report validation, with crash-recoverable intent."""
    run = Path(job['output'])
    if spec.get('checkpoint_retention') != CHECKPOINT_RETENTION:
        raise ValueError('Unexpected checkpoint-retention policy')
    if not training_complete(job, spec) or not evaluation_artifacts_complete(job):
        raise ValueError('Retaining best only requires valid training and all three formal evaluation reports')
    done = read(run / 'training_done.json')
    contract = retention_contract(job, done)
    receipt_path = run / 'checkpoint_retention.json'
    if receipt_path.exists():
        if not retention_receipt_valid(job, done):
            raise ValueError('Checkpoint-retention receipt differs from verified evaluation artifacts')
        if read(receipt_path)['status'] == 'completed' and not (run / 'last.pt').exists():
            return
    else:
        write(receipt_path, dict(contract, status='pending_delete', updated_at_beijing=now()))
    last = run / 'last.pt'
    if last.exists():
        if identity(last) != contract['last_checkpoint']:
            raise ValueError('The last checkpoint changed after formal evaluation')
        last.unlink()
    write(receipt_path, dict(contract, status='completed', updated_at_beijing=now()))


def worker(spec, name, stage):
    validate_plan(spec)
    job = next(job for job in spec['jobs'] if job['name'] == name)
    validate_input_identities(spec)
    allocator_receipt = capacity.apply_allocator_policy()
    # Each process binds the pinned runtime before touching model construction.
    import_runtime(spec)
    from mri_vla_jepa.data import RawMRIStore
    from mri_vla_jepa.t0_cv_worker import install_paired_readout_initialization, validate_completed_run
    from mri_vla_jepa.train_config import load_config
    from mri_vla_jepa.training import evaluate, train
    install_paired_readout_initialization()
    cfg = load_config(job['config'])
    run = Path(job['output'])
    run.mkdir(parents=True, exist_ok=True, mode=0o700)
    write(run / f'allocator_policy_{stage}.json', allocator_receipt)
    marker = run / f'{stage}_worker_receipt.json'
    with (run / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            contract = run_identity(spec, job)
            contract_path = run / 'run_identity.json'
            if contract_path.exists():
                if read(contract_path) != contract:
                    raise ValueError('Run inputs differ; use a new experiment directory')
            elif any((run / name).exists() for name in ('best.pt', 'last.pt', 'train_report.json')):
                raise ValueError('Existing checkpoints have no Medium976 run identity; they cannot be reused')
            else:
                write(contract_path, contract)
            if stage == 'train':
                if (run / 'checkpoint_retention.json').exists():
                    raise ValueError('A best-only retention journal exists; recover evaluation cleanup instead of restarting training')
                store = RawMRIStore(job['manifest'], cfg.image_shape, require_lesion_weights=False)
                last = run / 'last.pt'
                if (run / 'best.pt').exists() and not last.exists():
                    raise ValueError('Partial run has best.pt but no last.pt; recovery must preserve the existing checkpoint')
                report = train(store, cfg, run, resume=last.is_file())
                if report.get('completed') is not True:
                    raise RuntimeError('Training paused before completion; test evaluation remains withheld')
                write(run / 'training_done.json', {'completed': True, 'run_identity': identity(contract_path),
                      'files': [identity(run / name) for name in ('train_report.json', 'best.pt', 'last.pt')],
                      'completed_at_beijing': now()})
            else:
                if not all(training_complete(item, spec) for item in spec['jobs']):
                    raise ValueError('Test evaluation requires all 60 planned jobs to finish training')
                if not evaluation_artifacts_complete(job):
                    if not (run / 'last.pt').is_file():
                        raise ValueError('Best-only outputs have incomplete evaluation artifacts; refusing to overwrite or retrain')
                    validate_completed_run(job, cfg)
                    reports = []
                    for label, checkpoint, split in [('best_val', 'best.pt', 'val'),
                                                      ('last_val', 'last.pt', 'val'),
                                                      ('best_test', 'best.pt', 'test')]:
                        path = run / f'evaluation_{label}.json'
                        report = evaluate(run / checkpoint, job['manifest'], path, split=split,
                                          device='cuda:0', landmark='t0_t2', bootstrap=0)
                        validate_evaluation_report(job, label, report)
                        reports.append(identity(path))
                    write(run / 'evaluation_done.json', {'completed': True, 'job': name,
                          'manifest': identity(job['manifest']), 'reports': reports,
                          'best_checkpoint': identity(run / 'best.pt'), 'last_checkpoint': identity(run / 'last.pt'),
                          'training_done': identity(run / 'training_done.json'),
                          'run_identity': identity(run / 'run_identity.json'),
                          'landmarks': 't0_t2', 'test_patients': 175, 'completed_at_beijing': now()})
                finalize_checkpoint_retention(spec, job)
            write(marker, {'status': 'completed', 'stage': stage, 'job': name, 'finished_at_beijing': now()})
        except BaseException as exc:
            write(marker, {'status': 'failed', 'stage': stage, 'job': name,
                           'error': str(exc), 'finished_at_beijing': now()})
            traceback.print_exc()
            raise


def pid_identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return {'pid': pid, 'start_ticks': int(stat[19]), 'state': stat[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def resolve_gpus(gpus):
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    mapping = {}
    for line in output.strip().splitlines():
        index, uuid = (s.strip() for s in line.split(','))
        mapping[index] = mapping[uuid] = uuid
    resolved = [mapping[gpu] for gpu in gpus]
    validate_gpus(resolved)
    return resolved


def checkpoint_bytes_on_disk(run):
    total = 0
    for path in Path(run).glob('*.pt'):
        try:
            total += path.stat().st_size
        except FileNotFoundError:
            pass  # A validated evaluation worker may just have removed last.
    return total


def controller(spec, selected, gpus, max_active, spec_path, *, capacity_profile=None,
               gpu_memory_fraction=.97, cpu_oversubscribe=2.0, poll_seconds=2):
    output = Path(spec['output_root'])
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (output / '.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        matrix_path = output / 'matrix.json'
        matrix = read(matrix_path) if matrix_path.exists() else {
            'schema': 'medium976_fixed60_no_tf_controller_v1', 'prepared_spec': spec,
            'status': 'prepared', 'test_visible': False, 'jobs': {j['name']: {'status': 'queued'} for j in spec['jobs']}}
        if matrix['prepared_spec'] != spec:
            raise ValueError('The prepared experiment changed; use a new output directory')
        matrix['scheduling'] = {'max_active': max_active, 'gpu_memory_fraction': gpu_memory_fraction,
                                'cpu_oversubscribe': cpu_oversubscribe, 'seed_scheduling': 'all',
                                'gpu_uuids': gpus, 'allocator_policy': capacity.policy_identity()}
        if capacity_profile is not None:
            matrix['capacity_profile'] = capacity_profile
        for entry in matrix['jobs'].values():
            previous = entry.get('process')
            current = pid_identity(previous['pid']) if previous else None
            if current and current['start_ticks'] == previous['start_ticks'] and current['state'] != 'Z':
                raise RuntimeError('A worker from the previous controller is still running; it continues saving checkpoints. Restart after it finishes.')

        def progress():
            trained = sum(training_complete(j, spec) for j in spec['jobs'])
            evaluated = sum(evaluation_complete(j, spec) for j in spec['jobs'])
            matrix.update(training_completed=trained, evaluation_completed=evaluated,
                          test_visible=evaluated == PLANNED_JOBS, updated_at_beijing=now())
            write(matrix_path, matrix)

        for stage in ('train', 'evaluate'):
            if stage == 'evaluate' and not all(training_complete(j, spec) for j in spec['jobs']):
                matrix['status'] = 'training_pending'
                progress()
                return
            complete = lambda job: (training_complete(job, spec) if stage == 'train' else evaluation_complete(job, spec))
            queue = [j for j in selected if not complete(j)]
            active = {}
            ownership = admission.GPUOwnership(spec_path) if queue else None
            matrix['status'] = 'training' if stage == 'train' else 'evaluating'
            try:
                while queue or active:
                    # Reap before admitting another worker, so released reservations
                    # immediately become available across every seed and variant.
                    for name, item in list(active.items()):
                        process, job, log = item['process'], item['job'], item['log']
                        code = process.poll()
                        if code is None:
                            continue
                        log.close()
                        del active[name]
                        entry = matrix['jobs'][name]
                        entry.pop('process', None)
                        entry['status'] = 'training_completed' if stage == 'train' else 'completed'
                        if code != 0 or not complete(job):
                            entry.update(status='failed', exit_code=code)
                            matrix['status'] = 'failed'
                            progress()
                            raise RuntimeError(f'{name} {stage} failed; see {entry["log"]}')
                        progress()
                    while queue and len(active) < max_active:
                        if capacity_profile is None:
                            raise ValueError('Multiworker admission requires the verified full-phase capacity profile')
                        snapshot = admission.gpu_snapshot(gpus)
                        records = [item['ownership'] for item in active.values()]
                        accounting = ownership.account(snapshot, records)
                        host = admission.effective_host()
                        for item in active.values():
                            item['reservation']['rss_mib'] = admission.process_tree_rss_mib(item['process'].pid)
                        reservations = [item['reservation'] for item in active.values()]
                        chosen, reasons = None, []
                        # A larger job must not block a smaller admissible job.
                        for index, candidate in enumerate(queue):
                            resource = capacity.reservation_for(capacity_profile, candidate['variant'], stage)
                            host_ok, reason = admission.host_admits(reservations, resource, host, cpu_oversubscribe)
                            if not host_ok:
                                reasons.append(reason)
                                continue
                            gpu = admission.choose_gpu(snapshot, reservations, accounting['owned_mib'],
                                                       resource['gpu_mib'], gpu_memory_fraction)
                            if gpu is not None:
                                chosen = (index, gpu, resource)
                                break
                            reasons.append('GPU full-peak reservations and foreign memory exceed the admission budget')
                        matrix['admission'] = {'sampled_at_beijing': now(), 'devices': snapshot,
                                               'ownership': accounting, 'host': host,
                                               'active_reservations': reservations,
                                               'blocked_reasons': sorted(set(reasons))}
                        if chosen is None:
                            progress()
                            break
                        index, gpu, resource = chosen
                        job = queue.pop(index)
                        run = Path(job['output'])
                        run.mkdir(parents=True, exist_ok=True, mode=0o700)
                        # Keep best+last until formal evaluation succeeds, plus
                        # room for each active worker's atomic checkpoint write.
                        remaining_bytes = sum(max(0, (1 if evaluation_complete(j, spec) else 2)
                            * resource['checkpoint_bytes'] - checkpoint_bytes_on_disk(j['output']))
                            for j in spec['jobs'])
                        temporary_slots = min(max_active, sum(math.floor(s['total_mib'] * gpu_memory_fraction /
                                                      resource['gpu_mib']) for s in snapshot.values()))
                        required_bytes = math.ceil(1.25 * (remaining_bytes + temporary_slots * resource['checkpoint_bytes']))
                        free_bytes = shutil.disk_usage(output).free
                        matrix['storage'] = {'free_bytes': free_bytes, 'estimated_required_bytes': required_bytes,
                                             'quota_proven': False}
                        if free_bytes < required_bytes:
                            raise RuntimeError('Insufficient filesystem free space for retained suite checkpoints')
                        log = (run / f'{stage}.log').open('a')
                        env = os.environ.copy()
                        attempt = uuid.uuid4().hex
                        env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=spec['runtime_src'],
                                   PYTORCH_CUDA_ALLOC_CONF=capacity.ALLOCATOR_CONF,
                                   MEDIUM976_ATTEMPT_ID=attempt,
                                   OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--spec', str(spec_path),
                               '--worker', job['name'], '--worker-stage', stage]
                        process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
                        entry = matrix['jobs'][job['name']]
                        record = ownership.record(process.pid, cmd, gpu, attempt)
                        reservation = dict(resource, gpu_uuid=gpu, reservation_mib=resource['gpu_mib'])
                        entry.update(status=stage, gpu_uuid=gpu, process=pid_identity(process.pid),
                                     command=cmd, attempt=attempt, reservation=reservation,
                                     log=str(run / f'{stage}.log'))
                        active[job['name']] = {'process': process, 'job': job, 'log': log,
                                              'ownership': record, 'reservation': reservation}
                        progress()
                        print(f'{now()} {stage}: {job["name"]} on {gpu}', flush=True)
                    if queue or active:
                        time.sleep(poll_seconds)
            except BaseException:
                if matrix['status'] != 'failed':
                    matrix['status'] = 'controller_interrupted_workers_continue'
                progress()
                raise
        progress()
        matrix['status'] = 'completed' if matrix['test_visible'] else 'evaluation_pending'
        progress()
        if matrix['test_visible']:
            write(output / 'results_index.json', {'status': 'completed', 'test_visible': True,
                  'jobs': [{'name': j['name'], 'variant': j['variant'], 'seed': j['seed'], 'fold_id': j['fold_id'],
                            'evaluation_receipt': str(Path(j['output']) / 'evaluation_done.json')} for j in spec['jobs']],
                  'aggregation_policy': 'Separate seeds; five-fold summaries and probability ensembles are distinct analyses.',
                  'completed_at_beijing': now()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=DEFAULT_SPEC)
    parser.add_argument('--gpus', nargs='+', default=['0'])
    parser.add_argument('--max-active', type=int, default=PLANNED_JOBS,
                        help='Global worker ceiling; measured GPU and host admission determine actual concurrency')
    parser.add_argument('--gpu-memory-fraction', type=float, default=.97,
                        help='GPU admission budget (default .97); full measured peak reservations still apply')
    parser.add_argument('--cpu-oversubscribe', type=float, default=2.0,
                        help='CPU thread sharing budget; RAM admission remains fully reserved')
    parser.add_argument('--capacity-profile', type=Path, default=TASK / 'capacity_profile.json')
    parser.add_argument('--profile-only', action='store_true', help='Measure capacity without starting formal training')
    parser.add_argument('--variants', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--folds', nargs='+', type=int)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--worker-stage', choices=['train', 'evaluate'], default='train', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    spec = read(args.spec)
    if args.worker:
        worker(spec, args.worker, args.worker_stage)
        return
    validate_gpus(args.gpus)
    if args.max_active < 1:
        raise ValueError('max-active must be positive')
    if not math.isfinite(args.gpu_memory_fraction) or not .5 <= args.gpu_memory_fraction <= .98:
        raise ValueError('gpu-memory-fraction must be between .5 and .98')
    if not math.isfinite(args.cpu_oversubscribe) or not 1 <= args.cpu_oversubscribe <= 4:
        raise ValueError('cpu-oversubscribe must be between 1 and 4')
    if args.preflight_only and args.profile_only:
        raise ValueError('Choose CPU preflight or GPU profiling')
    selected = select_jobs(spec, args.variants, args.seeds, args.folds)
    report = preflight(spec, selected)
    print(json.dumps(report, indent=2), flush=True)
    if args.preflight_only:
        return report
    gpus = resolve_gpus(args.gpus)
    profile = read(args.capacity_profile) if args.capacity_profile.is_file() else None
    try:
        if profile is None:
            raise ValueError('Capacity has not been measured')
        capacity.validate_profile(spec, profile, gpus)
    except ValueError as exc:
        print(f'{now()} Capacity profiling required: {exc}', flush=True)
        if profile is not None:
            archived = args.capacity_profile.with_name(args.capacity_profile.stem +
                                                       f'.superseded.{time.time_ns()}.json')
            args.capacity_profile.replace(archived)
        subprocess.run([sys.executable, '-u', str(Path(capacity.__file__).resolve()),
                        '--spec', str(args.spec.resolve()), '--gpu', gpus[0],
                        '--output', str(args.capacity_profile.resolve())], check=True)
        profile = read(args.capacity_profile)
        capacity.validate_profile(spec, profile, gpus)
    if args.profile_only:
        print(f'Capacity profile: {args.capacity_profile.resolve()}', flush=True)
        return profile
    controller(spec, selected, gpus, args.max_active, args.spec.resolve(), capacity_profile=profile,
               gpu_memory_fraction=args.gpu_memory_fraction, cpu_oversubscribe=args.cpu_oversubscribe)


if __name__ == '__main__':
    main()
