#!/usr/bin/env python3
"""Package the audited shared ROI32 export into fixed-test, train-fit CV manifests."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys


REPO = Path('/personal_fast_storage/VLA-PCRV1')
OUTPUT = REPO / 'data/registered_roi32_medium976_v1'
SELECTION = REPO / 'private_outputs/medium_reselection_20261028/selection_index.json'
SPLIT_DIR = REPO / 'private_outputs/medium976_split_audit_20261028'
RUNTIME = REPO / 'private_outputs/medium976_roi32_v1/runtime/src'
CLINICAL = REPO / 'data/provenance/metadata/ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def fit_image_scale(records, train_ids):
    """Population scale over each unique train visit's nonzero DCE0 voxels."""
    visits = [visit for pid in sorted(train_ids) for visit in records[pid]['visits']]
    visit_ids = [visit['visit_id'] for visit in visits]
    if len(visit_ids) != len(set(visit_ids)):
        raise ValueError('A training visit would be counted more than once')
    count = sum(visit['DCE0_count'] for visit in visits)
    total = math.fsum(visit['DCE0_sum'] for visit in visits)
    square_total = math.fsum(visit['DCE0_square_sum'] for visit in visits)
    if count <= 0:
        raise ValueError('Empty training DCE0 foreground')
    mean = total / count
    variance = square_total / count - mean * mean
    if not math.isfinite(mean) or not math.isfinite(variance) or variance <= 0:
        raise ValueError('Invalid training DCE0 scale')
    scale = {
        'mean': mean, 'std': math.sqrt(variance), 'fit_split': 'train',
        'scope': 'unique_fold_train_actual_visits_DCE0_nonzero_crop_foreground_voxel_population',
        'background_policy': 'preserve_zero', 'already_normalized': True,
        'normalization_policy': (
            'Shared raw NPZ storage; the paired runtime applies this train-only scalar once to '
            'explicit foreground of all three phases. Background/padding stay zero; no per-patient '
            'or per-phase zscore. already_normalized describes RawMRIStore output.'),
    }
    report = {
        'schema': 'medium976_fold_train_image_scale_v1', 'fit_split': 'train',
        'fit_patient_ids': sorted(train_ids), 'fit_visit_ids': visit_ids,
        'fit_patient_count': len(train_ids), 'fit_visit_count': len(visits),
        'DCE0_foreground_voxels': count, 'DCE0_sum': total,
        'DCE0_square_sum': square_total, 'population_variance_ddof': 0,
        'image_normalization': scale,
        'fit_policy': 'Every actual MRI visit of each training patient once; DCE0 foreground only.',
        'excluded_from_fit': ['val', 'test'],
    }
    return scale, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--clinical', type=Path, default=CLINICAL)
    args = parser.parse_args()
    output = args.output.resolve()
    sys.path.insert(0, str(RUNTIME))
    from mri_vla_jepa.arm import CLINICAL_FEATURES, load_official_clinical, make_arm_record
    from mri_vla_jepa.constants import PHASES
    from mri_vla_jepa.data import RAW_ROI32_STORAGE, RawMRIStore, SCHEMA

    overrides_path = output / 'availability_overrides.json'
    overrides = read(overrides_path) if overrides_path.exists() else {'unusable_visits': []}
    unusable_visit_ids = {v['visit_id'] for v in overrides['unusable_visits']}
    if unusable_visit_ids and (unusable_visit_ids != {'ISPY2-893874:T3'}
                               or len(overrides['unusable_visits']) != 1):
        raise ValueError('Unexpected availability override; a new selection review is required')
    expected_visits = 3642 - len(unusable_visit_ids)
    export_summary = read(output / 'export_summary.json')
    if (export_summary.get('status') != 'completed'
            or export_summary.get('patients_completed') != 976
            or export_summary.get('visits_completed') != expected_visits):
        raise ValueError('The complete audited 976-patient export must finish first')
    selected = {p['patient_id']: p for p in read(SELECTION)['patients']}
    records = {path.stem: read(path) for path in sorted((output / 'records').glob('*.json'))}
    if len(selected) != 976 or set(selected) != set(records):
        raise ValueError('Selection and export patient populations differ')
    clinical = load_official_clinical(args.clinical)
    folds_doc = read(SPLIT_DIR / 'folds.json')
    folds = folds_doc['folds']
    if set(folds) != {f'fold{i}' for i in range(1, 6)}:
        raise ValueError('Exactly five folds required')
    old139 = set(read(SPLIT_DIR / 'old_test_T0T1T2_complete139.json'))
    fixed_test = set(folds['fold1']['test'])
    if len(fixed_test) != 175 or len(old139) != 139 or not old139 <= fixed_test:
        raise ValueError('The fixed175 test must retain the exact old139')
    validation_counts = Counter()
    for fold, splits in folds.items():
        populations = {name: set(ids) for name, ids in splits.items()}
        if (set(populations) != {'train', 'val', 'test'}
                or any(len(populations[name]) != len(splits[name]) for name in populations)
                or populations['test'] != fixed_test
                or populations['train'] & populations['val']
                or populations['train'] & fixed_test or populations['val'] & fixed_test
                or set.union(*populations.values()) != set(selected)):
            raise ValueError(f'Invalid fold partition: {fold}')
        validation_counts.update(populations['val'])
    if set(validation_counts) != set(selected) - fixed_test or set(validation_counts.values()) != {1}:
        raise ValueError('Each of the 801 development patients must be validation exactly once')

    patients = []
    actual_visit_count = 0
    for pid, selection in sorted(selected.items()):
        record = records[pid]
        if record['patient_id'] != pid or record['status'] != 'completed':
            raise ValueError('Incomplete/misattributed export record')
        official = clinical[pid.removeprefix('ISPY2-')]
        if official['pcr'] not in (0, 1) or float(selection['clinical']['pCR']) != official['pcr']:
            raise ValueError('Clinical pCR differs from selection or is missing')
        grid = f'registered_roi32_medium976_v1:{pid}'
        geometry = {
            'source_stage': 0, 'source_only': True, 'grid_id': grid,
            'orientation': 'RAS', 'roi_source': 'T0', 'resampling': 'source_grid',
            'shape_zyx': record['shape_zyx'], 'spacing_zyx_mm': record['spacing_zyx_mm'],
            'crop_affine_ras': record['crop_affine_ras'],
            'paired_sampling_plan': str(output / 'geometry' / f'{pid}.json'),
        }
        visits = [None] * 4
        for visit in record['visits']:
            if identity(visit['npz']['path']) != visit['npz']:
                raise ValueError('An exported NPZ changed after validation')
            stage = int(visit['visit'][1])
            if visits[stage] is not None or visit['visit_id'] != f'{pid}:T{stage}':
                raise ValueError('Duplicate or misattributed MRI visit')
            visits[stage] = {'stage': stage, 'available_at': stage, 'grid_id': grid,
                             'image': visit['npz']['path']}
        actual = [f'T{i}' for i, visit in enumerate(visits) if visit is not None]
        expected_stages = [stage for stage in sorted(selection['selected_visits'])
                           if f'{pid}:{stage}' not in unusable_visit_ids]
        if actual != expected_stages or visits[0] is None:
            raise ValueError('Actual export availability differs from selected MRI visits')
        if pid in fixed_test and any(v is None for v in visits[:3]):
            raise ValueError('Every test patient must have actual T0/T1/T2')
        actual_visit_count += len(actual)
        values = official['clinical_values']
        patients.append({
            'patient_key': pid,
            'clinical': {'values': values, 'known_at': [0 if v is not None else None for v in values]},
            'arm': make_arm_record(official['arm'], 0, 'assigned_arm_scenario'),
            'geometry': geometry, 'visits': visits,
            'target': {'pcr': official['pcr'], 'label_source': 'official_ispy2_clinical:pCR'},
        })
    if actual_visit_count != expected_visits:
        raise ValueError('Incorrect actual visit count')

    generated = datetime.now(timezone(timedelta(hours=8))).isoformat()
    old_patient_ids = {pid for pid, record in records.items()
                       if any(v['original_ROI32_replay_max_abs_error'] is not None for v in record['visits'])}
    restored_visits = [v for pid in old_patient_ids for v in records[pid]['visits']
                       if v['original_ROI32_replay_max_abs_error'] is None]
    summary = {
        'schema': 'medium976_fixed175_roi32_cv_manifest_summary_v1', 'status': 'completed',
        'generated_at_beijing': generated, 'patients': 976, 'actual_visits': actual_visit_count,
        'candidate_visits': 3642, 'unusable_visits': overrides['unusable_visits'],
        'image_shape': [3, 32, 128, 128], 'image_storage': 'raw_float32_shared_between_folds',
        'test_patients': 175, 'retained_old_test_complete139': 139, 'additional_test': 36,
        'development_patients': 801, 'T0_only_patients': sum(sum(v is not None for v in p['visits']) == 1 for p in patients),
        'available_patients_by_stage': {f'T{i}': sum(p['visits'][i] is not None for p in patients) for i in range(4)},
        'visit_pattern_counts': dict(Counter(''.join('1' if v is not None else '0' for v in p['visits']) for p in patients)),
        'baseline_patients': len(old_patient_ids),
        'baseline_visits': sum(v['original_ROI32_replay_max_abs_error'] is not None for r in records.values() for v in r['visits']),
        'new_patients': len(records) - len(old_patient_ids),
        'new_patient_visits': sum(len(r['visits']) for pid, r in records.items() if pid not in old_patient_ids),
        'restored_existing_visits': len(restored_visits),
        'existing_patients_with_restored_visits': len({v['patient_id'] for v in restored_visits}),
        'test_positive': sum(p['target']['pcr'] == 1 for p in patients if p['patient_key'] in fixed_test),
        'all_test_T0T1T2_complete': True, 'folds': {},
        'clinical_table': identity(args.clinical), 'runtime_source': str(RUNTIME),
        'each_development_patient_validation_exactly_once': True, 'training_started': False,
    }
    for fold, splits in sorted(folds.items()):
        scale, report = fit_image_scale(records, splits['train'])
        report.update(fold=fold, generated_at_beijing=generated,
                      train_test_overlap=False, train_validation_overlap=False)
        normalization_path = output / 'normalization' / f'{fold}.json'
        write(normalization_path, report)
        assignment = {pid: name for name, ids in splits.items() for pid in ids}
        fold_patients = deepcopy(patients)
        for patient in fold_patients:
            patient['split'] = assignment[patient['patient_key']]
        manifest = {
            'schema': SCHEMA, 'time_basis': 'stage_index', 'canonical_stages': [0, 1, 2, 3],
            'phase_order': PHASES, 'clinical_features': list(CLINICAL_FEATURES), 'synthetic': False,
            'image_preprocessing': {'mode': 'preprocessed_roi32_raw'},
            'storage_image_normalization': RAW_ROI32_STORAGE,
            'image_normalization': scale, 'patients': fold_patients,
            'provenance': {
                'cohort': 'medium976_reselection_20261028', 'fold': fold,
                'selection_index': str(SELECTION), 'split_definition': str(SPLIT_DIR / 'folds.json'),
                'availability_overrides': str(overrides_path) if unusable_visit_ids else None,
                'clinical_table': identity(args.clinical),
                'normalization_fit_report': str(normalization_path),
                'paired_runtime_source': str(RUNTIME),
                'roi_policy': 'Fixed source-available T0 predicted ROI at every actual visit; no follow-up segmentation.',
                'geometry_policy': 'Frozen matched metadata/mask/transform bundles; explicit array-index sampling, not placeholder NIfTI affines.',
                'registration_policy': 'Usable final registration including rigid fallback; no new NCC/FOV/FTV/full-mask-retention exclusion.',
                'arm_semantics': 'assigned_arm_scenario', 'arm_known_at_stage': 0, 'clinical_known_at_stage': 0,
                'availability_assumption': 'Assigned Arm and available baseline clinical fields assumed known at T0.',
                'treatment': 'Official assigned Arm; no dose, cycle or interval-treatment schedules inferred.',
                'data_policy': 'Only actual visits; missing canonical slots are null. Shared raw storage, per-fold train-only image and clinical normalization.',
                'test_policy': 'Fixed175 with actual T0/T1/T2; includes exact139 complete patients of the old Medium test173.',
                'historical_comparison_limit': '29 additional test patients were old development patients; old trained models are not held out on the whole new175.',
            },
        }
        path = output / 'cv_manifests' / fold / 'manifest.json'
        write(path, manifest)
        store = RawMRIStore(path)
        write(output / 'normalization' / f'{fold}_clinical_state.json', store.normalization_state())
        summary['folds'][fold] = {
            'manifest': str(path), 'image_normalization_fit': str(normalization_path),
            'clinical_normalization_state': str(output / 'normalization' / f'{fold}_clinical_state.json'),
            'splits': {name: {'patients': len(ids),
                             'positive': sum(clinical[pid.removeprefix('ISPY2-')]['pcr'] == 1 for pid in ids),
                             'actual_visits': sum(len(records[pid]['visits']) for pid in ids),
                             'available_patients_by_stage': {
                                 f'T{i}': sum(any(v['visit'] == f'T{i}' for v in records[pid]['visits']) for pid in ids)
                                 for i in range(4)}}
                       for name, ids in splits.items()},
            'mean': scale['mean'], 'std': scale['std'],
            'fit_patient_count': len(splits['train']), 'fit_visit_count': report['fit_visit_count'],
        }
        if fold == 'fold1':
            write(output / 'manifest.json', manifest)
    write(output / 'folds.json', folds_doc)
    write(output / 'manifest_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
