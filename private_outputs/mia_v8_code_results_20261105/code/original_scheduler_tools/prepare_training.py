#!/usr/bin/env python3
"""Prepare isolated Medium T0-T2 no-TF inputs; never construct or train a model."""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True

REPO = Path('/personal_fast_storage/VLA-PCRV1')
PUBLIC = Path('/projects/xts0030/public/cxx')
NAME = 'medium976_t0_t2_no_tf_cv5_multiseed_v1'
SOURCE_NAME = 'medium_t0_t2_no_tf_cv5_multiseed_v1'
SEEDS = [21, 44, 203]
VARIANTS = ['forecast_attention', 'direct_no_jepa', 'direct_readout', 'forecast_mean']
TRAINING_RUNS = len(SEEDS) * len(VARIANTS) * 5
CHECKPOINT_RETENTION = 'best_only_after_verified_evaluation'
TEST_VISIBILITY = f'withheld_until_all_{TRAINING_RUNS}_jobs_completed'
LOADER_FILES = {'mri_vla_jepa/data.py', 'mri_vla_jepa/data_pipeline.py'}
SCHEMA = 'medium976_t0_t2_no_tf_training_spec_v1'


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def source_files(directory):
    return sorted(p for p in directory.rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and p.suffix not in {'.pyc', '.pyo'})


def read_json(path):
    return json.loads(Path(path).read_text())


def create_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def job_definitions(args):
    jobs = []
    for seed in SEEDS:
        for variant in VARIANTS:
            for fold in range(1, 6):
                name = f'{variant}_seed{seed}_fold{fold}'
                jobs.append({'name': name, 'variant': variant, 'seed': seed, 'fold_id': fold,
                    'config': str(args.task_dir / 'input_configs' / f'{name}.yaml'),
                    'manifest': str(args.dataset_root / 'cv_manifests' / f'fold{fold}' / 'manifest.json'),
                    'output': str(args.output_root / 'runs' / name)})
    return jobs


def pretrained_asset_identities(jobs):
    import yaml

    paths = {Path(yaml.safe_load(Path(job['config']).read_text())['encoder_checkpoint']).resolve()
             for job in jobs}
    return [identity(path) for path in sorted(paths)]


def check_sources(args, jobs):
    if not (args.source_suite / 'runtime/src/mri_vla_jepa/landmark_profiles.py').is_file():
        raise FileNotFoundError('The source must be the frozen T0-T2 no-TF runtime')
    for rel in LOADER_FILES:
        if not (args.loader_runtime_src / rel).is_file():
            raise FileNotFoundError(args.loader_runtime_src / rel)
    for job in jobs:
        for path in (args.source_suite / 'input_configs' / Path(job['config']).name,
                     Path(job['manifest'])):
            if not path.is_file():
                raise FileNotFoundError(path)
    if not (args.dataset_root / 'completion_receipt.json').is_file():
        raise FileNotFoundError(args.dataset_root / 'completion_receipt.json')


def validate_prepared(args, spec, *, check_recorded_identities):
    """Read configs and actual data on CPU, without model/optimizer construction."""
    import yaml

    runtime_src = args.task_dir / 'runtime/src'
    source_src = args.source_suite / 'runtime/src'
    expected_runtime = {str(p.relative_to(source_src)): p for p in source_files(source_src)}
    actual_runtime = {str(p.relative_to(runtime_src)): p for p in source_files(runtime_src)}
    if set(actual_runtime) != set(expected_runtime):
        raise ValueError('The isolated runtime file set differs from the frozen source')
    for rel, target in actual_runtime.items():
        original = args.loader_runtime_src / rel if rel in LOADER_FILES else expected_runtime[rel]
        if target.read_bytes() != original.read_bytes():
            raise ValueError(f'Unexpected isolated runtime change: {rel}')
    if check_recorded_identities:
        for key in ('runtime_identities', 'config_identities', 'manifest_identities', 'asset_identities'):
            for entry in spec[key]:
                if identity(entry['path']) != entry:
                    raise ValueError(f'Prepared file identity changed: {entry["path"]}')

    sys.path.insert(0, str(runtime_src))
    from mri_vla_jepa.data import RAW_ROI32_STORAGE
    from mri_vla_jepa.landmark_profiles import allowed_landmarks, profile_stages
    from mri_vla_jepa.t0_cv_worker import _store
    from mri_vla_jepa.train_config import load_config
    import torch

    if Path(sys.modules['mri_vla_jepa.data'].__file__).resolve().parent != runtime_src / 'mri_vla_jepa':
        raise ValueError('CPU verification imported another runtime')
    torch.set_num_threads(2)
    expected_jobs = job_definitions(args)
    if (spec['schema'] != SCHEMA or spec['jobs'] != expected_jobs or len(expected_jobs) != 60
            or spec['train_seeds'] != SEEDS or spec['variants'] != VARIANTS
            or spec['training_runs'] != TRAINING_RUNS or spec['fold_ids'] != list(range(1, 6))):
        raise ValueError('The job matrix differs from the four-variant three-seed five-fold plan')
    if (spec.get('checkpoint_retention') != CHECKPOINT_RETENTION
            or spec.get('test_visibility') != TEST_VISIBILITY):
        raise ValueError('The three-seed checkpoint retention or test visibility policy changed')
    active_configs = {p.name for p in (args.task_dir / 'input_configs').glob('*.yaml')}
    if active_configs != {Path(job['config']).name for job in expected_jobs}:
        raise ValueError('Active input_configs must contain exactly the 60 registered jobs')
    if spec['asset_identities'] != pretrained_asset_identities(expected_jobs):
        raise ValueError('The prepared encoder assets differ from the configuration checkpoints')
    configs_by_fold = {}
    for job in expected_jobs:
        path = Path(job['config'])
        source = args.source_suite / 'input_configs' / path.name
        if path.read_bytes() != source.read_bytes():
            raise ValueError(f'Training config changed from its original baseline: {path.name}')
        value = yaml.safe_load(path.read_text())
        cfg = load_config(path)
        if (cfg.seed, cfg.landmarks, cfg.image_shape, cfg.model.model_size) != (
                job['seed'], 't0_t2', (32, 128, 128), 'medium'):
            raise ValueError(f'Unexpected Medium T0-T2 config: {path.name}')
        if (cfg.prediction_training, cfg.jepa_weight, cfg.rollout_weight) != (
                'hybrid', 0., 0. if job['variant'] == 'direct_no_jepa' else .5):
            raise ValueError(f'Unexpected no-TF objective: {path.name}')
        if cfg.image_cache is not None or cfg.device != 'cuda:0':
            raise ValueError('Keep original cache=null and logical device=cuda:0 settings')
        if value['initialize_from_checkpoint'] is not None or cfg.encoder_training != 'finetune':
            raise ValueError('Fresh runs must reuse only original pretrained encoder initialization')
        if not Path(cfg.encoder_checkpoint).is_file():
            raise FileNotFoundError(cfg.encoder_checkpoint)
        configs_by_fold.setdefault(job['fold_id'], cfg)

    folds, cohorts, tests, validation_sets = [], [], [], []
    for fold in range(1, 6):
        path = args.dataset_root / 'cv_manifests' / f'fold{fold}' / 'manifest.json'
        manifest = read_json(path)
        patients = manifest['patients']
        keys = [p['patient_key'] for p in patients]
        split_keys = {split: {p['patient_key'] for p in patients if p['split'] == split}
                      for split in ('train', 'val', 'test')}
        counts = {k: len(v) for k, v in split_keys.items()}
        expected_counts = {'train': 640 if fold == 1 else 641,
                           'val': 161 if fold == 1 else 160, 'test': 175}
        if len(keys) != 976 or len(set(keys)) != 976 or counts != expected_counts:
            raise ValueError(f'Unexpected patient partition in fold {fold}: {counts}')
        visits = sum(v is not None for p in patients for v in p['visits'])
        if visits != 3641:
            raise ValueError(f'Unexpected usable visit count in fold {fold}: {visits}')
        if manifest['image_preprocessing'] != {'mode': 'preprocessed_roi32_raw'}:
            raise ValueError('The dataset must declare raw ROI32 storage')
        if manifest['storage_image_normalization'] != RAW_ROI32_STORAGE:
            raise ValueError('The dataset storage declaration differs from the new loader')
        if manifest['image_normalization']['fit_split'] != 'train':
            raise ValueError('Fold intensity scale must be fitted only on training patients')
        special = next(p for p in patients if p['patient_key'] == 'ISPY2-893874')
        if any(special['visits'][i] is None for i in (0, 1, 2)) or special['visits'][3] is not None:
            raise ValueError('ISPY2-893874 must retain T0/T1/T2 and explicitly omit unusable T3')
        cfg = configs_by_fold[fold]
        store = _store(path, cfg)
        if not store.preprocessed_roi32_raw or tuple(profile_stages(cfg.landmarks)) != (0, 1, 2):
            raise ValueError('The isolated loader/profile combination is inconsistent')
        legal = [allowed_landmarks(store, i, cfg.landmarks) for i in range(len(patients))]
        if any(not stages or 3 in stages for stages in legal):
            raise ValueError('Every patient needs a T0-T2 source; T3 cannot be a classification landmark')
        probe_index = next(i for i, p in enumerate(patients) if p['patient_key'] == 'ISPY2-100899')
        inp, sup = store.batch([(probe_index, 2)])
        if inp.observed_mask[0, 3].item() or not sup.future_mask[0, 3].item():
            raise ValueError('T3 must remain a future target, excluded from the T2 input')
        if not torch.isfinite(inp.images).all().item():
            raise ValueError('Actual exported ROI32 batch contains nonfinite normalized data')
        folds.append({'fold_id': fold, 'manifest': str(path), 'patients': len(keys),
            'usable_visits': visits, 'split_counts': counts,
            'positive_counts': {s: sum(int(p['target']['pcr']) for p in patients if p['split'] == s)
                                for s in ('train', 'val', 'test')},
            'available_prefix_counts': dict(sorted(Counter(stage for stages in legal for stage in stages).items())),
            'image_normalization': {k: manifest['image_normalization'][k] for k in ('mean', 'std', 'fit_split')},
            'actual_batch_probe': {'patient_key': 'ISPY2-100899', 'source_stage': 2,
                'T3_input_observed': False, 'T3_future_supervision': True, 'all_finite': True}})
        cohorts.append(set(keys)); tests.append(split_keys['test']); validation_sets.append(split_keys['val'])
    if not all(cohort == cohorts[0] for cohort in cohorts) or not all(test == tests[0] for test in tests):
        raise ValueError('All folds must share the exact same cohort and fixed test patients')
    development = cohorts[0] - tests[0]
    if (set.union(*validation_sets) != development
            or sum(len(v) for v in validation_sets) != len(development)):
        raise ValueError('Validation folds must partition all 801 development patients exactly once')
    if torch.cuda.is_initialized():
        raise ValueError('CPU preparation unexpectedly initialized CUDA')
    return {'schema': 'medium976_training_preparation_validation_v1', 'status': 'passed',
        'checked_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'configs_strict_loaded': len(expected_jobs), 'all_configs_byte_identical_to_baseline': True,
        'train_seeds': SEEDS, 'training_runs': TRAINING_RUNS,
        'checkpoint_retention': CHECKPOINT_RETENTION, 'test_visibility': TEST_VISIBILITY,
        'all_runtime_files_except_loader_pair_byte_identical_to_baseline': True,
        'loader_pair_byte_identical_to_validated_raw_runtime': True,
        'runtime_files': len(actual_runtime), 'landmark_profile': 't0_t2',
        'classification_source_stages': [0, 1, 2], 'T3_retained_as_future_target': True,
        'cohort_patients': 976, 'fixed_test_patients': 175, 'development_patients': 801,
        'usable_visits': 3641, 'folds': folds,
        'model_constructed': False, 'optimizer_constructed': False,
        'GPU_used': False, 'training_started': False}


def migrate_three_seed_best_only(args):
    """Archive an unstarted six-seed preparation and retain its matching subset."""
    spec_path = args.task_dir / 'training_spec.json'
    report_path = args.task_dir / 'preparation_validation.json'
    old = read_json(spec_path)
    if args.output_root.exists() or Path(old['output_root']).exists():
        raise FileExistsError('A formal output exists; prepared-spec migration is forbidden')
    if (old.get('schema') != SCHEMA or old.get('train_seeds') != [43, 44, 809, 203, 30, 21]
            or old.get('training_runs') != 120 or len(old.get('jobs', [])) != 120
            or old.get('variants') != VARIANTS or old.get('training_started') is not False):
        raise ValueError('Migration accepts only the original unstarted six-seed preparation')
    for key in ('task_dir', 'output_root', 'dataset_root'):
        if Path(old[key]).resolve() != getattr(args, key):
            raise ValueError(f'Migration argument differs from original preparation: {key}')
    for key in ('runtime_identities', 'config_identities', 'manifest_identities', 'asset_identities'):
        if any(identity(entry['path']) != entry for entry in old[key]):
            raise ValueError(f'An original prepared input changed: {key}')
    configs = args.task_dir / 'input_configs'
    if {p.name for p in configs.glob('*.yaml')} != {Path(j['config']).name for j in old['jobs']}:
        raise ValueError('Original input config directory is incomplete or contains extra YAMLs')
    jobs = job_definitions(args)
    check_sources(args, jobs)
    for job in jobs:
        path = Path(job['config'])
        if path.read_bytes() != (args.source_suite / 'input_configs' / path.name).read_bytes():
            raise ValueError('The retained config differs from its original baseline')
    timestamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = args.task_dir / 'archived_preparations' / f'{timestamp}_120jobs_before_three_seed_best_only'
    archive.mkdir(parents=True, exist_ok=False)
    spec_path.rename(archive / spec_path.name)
    report_path.rename(archive / report_path.name)
    configs.rename(archive / configs.name)
    configs.mkdir()
    for job in jobs:
        target = Path(job['config'])
        shutil.copy2(archive / 'input_configs' / target.name, target)
    spec = dict(old)
    spec.update(created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                train_seeds=SEEDS, training_runs=TRAINING_RUNS, jobs=jobs,
                config_identities=[identity(job['config']) for job in jobs],
                test_visibility=TEST_VISIBILITY, checkpoint_retention=CHECKPOINT_RETENTION,
                migrated_from_preparation=str(archive), training_started=False)
    report = validate_prepared(args, spec, check_recorded_identities=True)
    spec['folds'] = report['folds']
    create_json(spec_path, spec)
    create_json(report_path, report)
    receipt = {'schema': 'medium976_three_seed_best_only_preparation_migration_v1',
        'status': 'completed', 'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'archived_preparation': str(archive), 'active_spec': str(spec_path),
        'previous_seeds': old['train_seeds'], 'active_seeds': SEEDS,
        'previous_training_runs': 120, 'active_training_runs': TRAINING_RUNS,
        'archived_input_configs': 120, 'active_input_configs': len(jobs),
        'retained_yaml_bytes_and_mtime_preserved': all(
            identity(job['config'])['bytes'] == identity(archive / 'input_configs' / Path(job['config']).name)['bytes']
            and identity(job['config'])['mtime_ns'] == identity(archive / 'input_configs' / Path(job['config']).name)['mtime_ns']
            for job in jobs),
        'runtime_reused_without_copy_or_modification': True,
        'data_or_yaml_content_modified': False, 'formal_output_created': False,
        'checkpoint_retention': CHECKPOINT_RETENTION,
        'GPU_used': False, 'training_started': False}
    create_json(archive / 'migration_receipt.json', receipt)
    print(json.dumps({'status': 'migrated', 'training_spec': str(spec_path),
                     'archive': str(archive), 'training_runs': TRAINING_RUNS,
                     'train_seeds': SEEDS, 'training_started': False}, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-dir', type=Path, default=PUBLIC / 'planned_tasks' / NAME)
    parser.add_argument('--output-root', type=Path, default=PUBLIC / NAME)
    parser.add_argument('--dataset-root', type=Path, default=REPO / 'data/registered_roi32_medium976_v1')
    parser.add_argument('--source-suite', type=Path, default=PUBLIC / SOURCE_NAME)
    parser.add_argument('--loader-runtime-src', type=Path,
                        default=REPO / 'private_outputs/medium976_roi32_v1/runtime/src')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--migrate-three-seed-best-only', action='store_true',
                        help='Archive an unstarted 120-job preparation and keep the 60-job subset')
    args = parser.parse_args(argv)
    for key in ('task_dir', 'output_root', 'dataset_root', 'source_suite', 'loader_runtime_src'):
        setattr(args, key, getattr(args, key).resolve())
    jobs = job_definitions(args)
    check_sources(args, jobs)
    spec_path = args.task_dir / 'training_spec.json'
    report_path = args.task_dir / 'preparation_validation.json'
    if args.validate_only and args.migrate_three_seed_best_only:
        parser.error('Choose validation or migration, not both')
    if args.migrate_three_seed_best_only:
        return migrate_three_seed_best_only(args)
    if args.validate_only:
        spec = read_json(spec_path)
        report = validate_prepared(args, spec, check_recorded_identities=True)
        print(json.dumps({'status': report['status'], 'configs_strict_loaded': report['configs_strict_loaded'],
                          'folds_checked': len(report['folds']), 'training_started': False}, indent=2))
        return 0

    destinations = [args.task_dir / 'runtime', args.task_dir / 'input_configs', spec_path, report_path]
    conflicts = [str(path) for path in destinations if path.exists()]
    if conflicts:
        raise FileExistsError('Preparation creates new artifacts only; existing destinations: ' + ', '.join(conflicts))
    args.task_dir.mkdir(parents=True, exist_ok=True)
    runtime_src = args.task_dir / 'runtime/src'
    source_src = args.source_suite / 'runtime/src'
    for source in source_files(source_src):
        rel = source.relative_to(source_src)
        selected = args.loader_runtime_src / rel if str(rel) in LOADER_FILES else source
        target = runtime_src / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        # Every destination is newly created; shared and historical files stay untouched.
        with target.open('xb') as stream:
            stream.write(selected.read_bytes())
        shutil.copystat(selected, target)
    (args.task_dir / 'input_configs').mkdir()
    for job in jobs:
        target = Path(job['config'])
        source = args.source_suite / 'input_configs' / target.name
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
        shutil.copystat(source, target)
    spec = {'schema': SCHEMA, 'suite_name': NAME,
        'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'task_dir': str(args.task_dir), 'output_root': str(args.output_root),
        'runtime_src': str(runtime_src), 'dataset_root': str(args.dataset_root),
        'baseline_source_suite': str(args.source_suite),
        'loader_source_runtime_src': str(args.loader_runtime_src),
        'train_seeds': SEEDS, 'variants': VARIANTS, 'fold_ids': list(range(1, 6)),
        'training_runs': len(jobs), 'jobs': jobs,
        'runtime_identities': [identity(p) for p in source_files(runtime_src)],
        'config_identities': [identity(job['config']) for job in jobs],
        'manifest_identities': [identity(args.dataset_root / 'cv_manifests' / f'fold{fold}' / 'manifest.json')
                                for fold in range(1, 6)],
        'asset_identities': pretrained_asset_identities(jobs),
        'test_visibility': TEST_VISIBILITY, 'checkpoint_retention': CHECKPOINT_RETENTION,
        'training_started': False}
    report = validate_prepared(args, spec, check_recorded_identities=True)
    spec['folds'] = report['folds']
    create_json(spec_path, spec)
    create_json(report_path, report)
    print(json.dumps({'status': 'prepared', 'training_spec': str(spec_path),
        'configs': len(jobs), 'runtime_files': report['runtime_files'],
        'folds_checked': len(report['folds']), 'model_constructed': False,
        'GPU_used': False, 'training_started': False}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
