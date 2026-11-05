#!/usr/bin/env python3
"""Create audited ROI32 plans using paired historical metadata and T0 masks.

Public normalized native metadata must never be combined directly with an old
MAMA-MIA mask header. MRI arrays retain their saved ZYX index coordinates; the
plan records an explicit crop-to-array mapping and a public-coordinate bridge.
No source MRI, metadata, masks, or transforms are modified by this module.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from nibabel.spaces import vox2out_vox

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'data/provenance/source_preprocessing/MeWM-ISPY2'
sys.path.insert(0, str(UPSTREAM))
from mewm_ispy2.registered_roi32_data import (  # noqa: E402
    LPS_RAS, REVERSE, SHAPE, SPACING, acquisition_affine, centered_affine,
    image_affine, largest_region, linear_affine, resample_array,
)

BUNDLE = ROOT / 'data/provenance/rebuild_inputs'
PUBLIC = Path('/public_fast_storage/cxx26/organized_v1')
HISTORICAL = PUBLIC / '05_manifests_and_logs/normalization/20261011_portable_v1/original_json'
DEFAULT_SELECTION = ROOT / 'private_outputs/medium_reselection_20261028'
DEFAULT_OUTPUT = ROOT / 'private_outputs/medium976_geometry_audit_20261028'
DEFAULT_PHASE_SOURCES = ROOT / 'private_outputs/medium976_source_audit_20261028/phase_sources_all3642.json'
SCHEMA = 'medium976_paired_geometry_roi32_plan_v1'


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def max_error(actual, expected):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def historical_meta(native, visit, stage='registered'):
    path = HISTORICAL / stage / native / visit / 'meta' / f'{native}_{visit}_meta.json'
    require(path.is_file(), f'Missing preserved {stage} metadata: {path}')
    return path


def _geometry_close(left, right):
    return (left['shape_zyx'] == right['shape_zyx']
            and np.allclose(left['spacing_xyz'], right['spacing_xyz'], rtol=0, atol=1e-8)
            and np.allclose(left['direction'], right['direction'], rtol=0, atol=1e-8))


def _phase_record(meta):
    intra = meta['intra_visit_registration']
    require(intra['status'] == 'completed' and intra['reference_phase_id'] == 0,
            'T0 intravisit registration is incomplete or uses a different reference')
    selected = [r for r in intra['registrations'] if r['phase_id'] == 1 and r['registered_to_phase_id'] == 0]
    require(len(selected) == 1, 'Expected one saved phase1-to-phase0 transform')
    return selected[0]


def validate_paired_metadata(native_meta, t0_meta):
    """Check that acquisition and centered source grids describe the same array."""
    affine = acquisition_affine(native_meta)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    source = t0_meta['source_geometry']
    require(source['shape_zyx'] == [int(native_meta[k]) for k in ('n_slices', 'rows', 'cols')],
            'Historical native array and registered source geometry shapes differ')
    require(t0_meta['target_geometry']['shape_zyx'] == [int(t0_meta[k]) for k in ('n_slices', 'rows', 'cols')],
            'Historical T0 registered array and target geometry shapes differ')
    spacing_error = max_error(source['spacing_xyz'], spacing)
    direction_error = max_error(np.asarray(source['direction']).reshape(3, 3), affine[:3, :3] / spacing)
    require(spacing_error < 1e-5 and direction_error < 1e-5,
            'Historical native acquisition and registration-centered source spacing/direction differ')
    return {'native_vs_centered_source_spacing_max_error_mm': spacing_error,
            'native_vs_centered_source_direction_max_error': direction_error}


def build_patient_plan(task):
    """Recompute one T0 mapping, largest component, and fixed ROI32 geometry."""
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    patient, visits = task['patient'], task['visits']
    pid, native = patient['patient_id'], task['native_patient_id']
    native_path, t0_path = Path(task['native_metadata']), Path(task['t0_metadata'])
    phase_path, mask_path = Path(task['phase_transform']), Path(patient['T0_locator']['mask'])
    native_meta, t0_meta = read_json(native_path), read_json(t0_path)
    receipt = read_json(patient['T0_locator']['receipt'])
    require(receipt['status'] == 'completed' and receipt['qc']['tumor_voxels'] > 0,
            'T0 locator lacks a completed nonempty receipt')
    require(t0_meta['registration']['status'] == 'fixed_reference', 'T0 is not the fixed registered reference')
    phase_record = _phase_record(t0_meta)
    require(Path(phase_record['transform_path']).name == phase_path.name,
            'Paired T0 metadata names a different phase transform')
    phase = sitk.ReadTransform(str(phase_path))
    phase_affine = linear_affine(phase)

    mask_image = sitk.ReadImage(str(mask_path))
    mask = sitk.GetArrayFromImage(mask_image).astype(np.uint8)
    native_affine = acquisition_affine(native_meta)
    shape = (int(native_meta['n_slices']), int(native_meta['rows']), int(native_meta['cols']))
    reference = sitk.Image(list(shape[::-1]), sitk.sitkUInt8)
    native_spacing = np.linalg.norm(native_affine[:3, :3], axis=0)
    reference.SetSpacing(tuple(native_spacing))
    reference.SetDirection(tuple((native_affine[:3, :3] / native_spacing).ravel()))
    reference.SetOrigin(tuple(native_affine[:3, 3]))
    reference = sitk.DICOMOrient(reference, 'LPS')
    header_error = max_error(image_affine(mask_image), image_affine(reference))
    require(mask_image.GetSize() == reference.GetSize() and header_error <= 2e-5,
            'Old MAMA mask header disagrees with its paired historical native metadata')
    require(tuple(mask.shape) == tuple(sitk.GetArrayViewFromImage(reference).shape), 'Mask array axis/shape mismatch')
    require(int(np.count_nonzero(mask)) == receipt['qc']['tumor_voxels'], 'Mask voxel count changed from segmentation receipt')
    require(set(np.unique(mask)) <= {0, 1}, 'Expected a binary T0 MAMA mask')
    del reference

    target_ras = LPS_RAS @ acquisition_affine(t0_meta)
    strict_shape, strict_ras = vox2out_vox((tuple(t0_meta['target_geometry']['shape_zyx'])[::-1], target_ras),
                                          voxel_sizes=SPACING[::-1])
    strict_to_native_lps = (native_affine @ np.linalg.inv(centered_affine(t0_meta['source_geometry']))
                            @ phase_affine @ centered_affine(t0_meta['target_geometry'])
                            @ np.linalg.inv(target_ras) @ strict_ras)
    mask_to_strict = np.linalg.inv(strict_to_native_lps) @ image_affine(mask_image)
    # Independently execute TransformPoint instead of reusing its matrix.
    point_error = 0.
    for point in (np.zeros(3), (np.asarray(strict_shape) - 1) / 2,
                  np.asarray(strict_shape) - 1, np.array([3.2, 9.1, 7.4])):
        ras = strict_ras[:3, :3] @ point + strict_ras[:3, 3]
        index = np.linalg.solve(target_ras[:3, :3], ras - target_ras[:3, 3])
        target = centered_affine(t0_meta['target_geometry'])
        source_point = np.asarray(phase.TransformPoint(tuple(target[:3, :3] @ index + target[:3, 3])))
        source = centered_affine(t0_meta['source_geometry'])
        source_index = np.linalg.solve(source[:3, :3], source_point - source[:3, 3])
        native_point = native_affine[:3, :3] @ source_index + native_affine[:3, 3]
        expected = strict_to_native_lps[:3, :3] @ point + strict_to_native_lps[:3, 3]
        point_error = max(point_error, max_error(native_point, expected))
    require(point_error < 1e-8, 'Independent SITK point chain disagrees with matrix mapping')
    region = largest_region(mask, mask_to_strict)
    require(region is not None, 'T0 mask array is empty')
    center = region['center']
    start = center - (np.asarray(SHAPE) - 1) / 2
    translation = np.eye(4)
    translation[:3, 3] = start[::-1]
    crop_ras = strict_ras @ translation
    crop_to_array = REVERSE @ np.linalg.inv(target_ras) @ crop_ras @ REVERSE
    crop_to_mask = REVERSE @ np.linalg.inv(mask_to_strict) @ translation @ REVERSE
    mask_crop = resample_array(mask, crop_to_mask, 0) > 0
    largest_crop = resample_array(region['largest'], crop_to_mask, 0) > 0
    require(largest_crop.any(), 'Nonempty largest component disappears from its centered ROI32 crop')
    require(np.allclose(np.linalg.norm(crop_ras[:3, :3], axis=0), SPACING[::-1], rtol=0, atol=1e-10),
            'ROI32 crop spacing changed')

    checks = {**task['source_grid_checks'],
              'native_mask_header_max_error_mm': header_error, 'independent_point_chain_max_error_mm': point_error}
    baseline = task.get('baseline')
    if baseline is not None:
        expected = task['expected_patient']
        checks['legacy_mask_to_strict_max_error'] = max_error(mask_to_strict, expected['expected_mask_to_strict_xyz'])
        checks['legacy_largest_center_max_error_voxels'] = max_error(center, expected['expected_center_zyx'])
        checks['legacy_crop_affine_max_error_mm'] = max_error(crop_ras, baseline['geometry']['crop_affine_ras'])
        require(checks['legacy_mask_to_strict_max_error'] < 1e-6, 'Existing patient mapping changed')
        require(checks['legacy_largest_center_max_error_voxels'] < 1e-6, 'Existing patient largest component changed')
        require(checks['legacy_crop_affine_max_error_mm'] < 1e-6, 'Existing ROI32 crop affine changed')
        require(region['source_voxels'] == expected['expected_mask_voxels'], 'Existing T0 mask contents changed')

    public_phase = PUBLIC / '03_registered_nifti' / native / 'T0/transforms/intravisit/phase_1_to_phase_0.tfm'
    checks['public_vs_archived_phase_matrix_max_error'] = max_error(
        phase_affine, linear_affine(sitk.ReadTransform(str(public_phase))))
    # The selected paired transform is the authority for this crop. A separately
    # published version can differ; retain the matched image payload version.

    sources = {k: identity(v) for k, v in {
        'native_metadata': native_path, 't0_metadata': t0_path, 'phase_transform': phase_path,
        't0_mask': mask_path, 't0_segmentation_receipt': patient['T0_locator']['receipt'],
    }.items()}
    if task.get('phase1_image_source'):
        sources['paired_registered_phase1_image'] = identity(task['phase1_image_source'])
    planned_visits = []
    maximum_visit_grid_error = 0.
    for visit in visits:
        stage = visit['visit']
        archived_path = historical_meta(native, stage)
        archived = read_json(archived_path)
        require(_geometry_close(archived['target_geometry'], t0_meta['target_geometry']),
                f'{stage} target grid differs from saved T0 target grid')
        visit_grid_error = max_error(acquisition_affine(archived), acquisition_affine(t0_meta))
        maximum_visit_grid_error = max(maximum_visit_grid_error, visit_grid_error)
        require(visit_grid_error < 1e-6, f'{stage} acquisition grid differs from paired T0 grid')
        public_path = Path(visit['registered_metadata'])
        public_meta = read_json(public_path)
        require(tuple(public_meta['target_geometry']['shape_zyx']) == tuple(t0_meta['target_geometry']['shape_zyx']),
                f'{stage} public registered array shape changed')
        require(public_meta.get('registered_to_visit') == 'T0', f'{stage} public MRI is not registered to T0')
        public_target_ras = LPS_RAS @ acquisition_affine(public_meta)
        # Same array sampling in a different published coordinate frame.
        public_crop_ras = public_target_ras @ REVERSE @ crop_to_array @ REVERSE
        bridge_error = max_error(REVERSE @ np.linalg.inv(public_target_ras) @ public_crop_ras @ REVERSE,
                                 crop_to_array)
        require(bridge_error < 1e-8, 'Public-coordinate bridge changes array sampling')
        planned_visits.append({
            'visit': stage,
            'archived_metadata': identity(archived_path), 'public_metadata': identity(public_path),
            'shape_zyx': list(t0_meta['target_geometry']['shape_zyx']),
            'crop_to_registered_index_zyx': crop_to_array.tolist(),
            'public_crop_affine_ras': public_crop_ras.tolist(),
            'public_bridge_max_error_voxels': bridge_error,
        })
    checks['maximum_visit_grid_affine_error_mm'] = maximum_visit_grid_error
    result = {
        'schema': SCHEMA, 'status': 'passed', 'patient_id': pid, 'native_patient_id': native,
        'in_current_866': patient['in_current_866'],
        'coordinate_policy': 'Preserved paired native metadata, historical registered T0 grid, unchanged old mask header, saved phase1-to-phase0 transform.',
        'array_axis_order': 'zyx', 'spatial_units': 'mm', 'crop_shape_zyx': list(SHAPE),
        'shape_zyx': list(SHAPE), 'spacing_zyx_mm': SPACING.tolist(),
        'crop_affine_ras': crop_ras.tolist(), 'crop_affine_lps': (LPS_RAS @ crop_ras).tolist(),
        'crop_to_registered_index_zyx': crop_to_array.tolist(),
        'crop_to_native_mask_index_zyx': crop_to_mask.tolist(),
        'mask_to_strict_xyz': mask_to_strict.tolist(), 'strict_grid_affine_ras': strict_ras.tolist(),
        'center_strict_zyx': center.tolist(), 'start_strict_zyx': start.tolist(),
        'native_mask_shape_zyx': list(mask.shape), 'native_mask_affine_lps': image_affine(mask_image).tolist(),
        'phase_transform_direction': 'Saved SimpleITK resampling transform maps fixed phase0 centered physical coordinates to moving phase1 centered coordinates; inverse chain maps native phase1 mask to fixed T0.',
        'source_identities': sources,
        'phase_transform_version_policy': task['phase_transform_version_policy'],
        'actual_cropped_mask_voxels': int(mask_crop.sum()),
        'actual_cropped_largest_voxels': int(largest_crop.sum()),
        'mask_qc': {k: v for k, v in region.items() if k not in {'center', 'largest'}},
        'segmentation_review_status': receipt['qc'].get('review_status'),
        'segmentation_flags': receipt['qc'].get('flags', []),
        'checks': checks, 'visits': planned_visits,
        'verification_limits': ['Coordinate consistency and preserved sampling positions do not establish anatomical segmentation or registration accuracy.',
                                'No MRI voxel arrays have been loaded or exported by this geometry audit.'],
    }
    return result


def build_tasks(selection_dir):
    selection = read_json(selection_dir / 'selection_index.json')
    with (selection_dir / 'selected_visits.csv').open() as handle:
        by_patient = collections.defaultdict(list)
        for row in csv.DictReader(handle):
            by_patient[row['patient_id']].append(row)
    file_paths = {r['source_path']: BUNDLE / r['path'] for r in read_json(BUNDLE / 'INPUT_MANIFEST.json')['files']}
    old_inventory = read_json(BUNDLE / 'intermediates/registered_three_phase_roi32_v1/inventory.json')
    old_patients = {p['patient_id']: p for p in old_inventory['patients']}
    baseline = {p['patient_key']: p for p in read_json(ROOT / 'data/registered_roi32/manifest.json')['patients']}
    segmentation = read_json(BUNDLE / 'segmentation/inventory.json')
    records = {r['case_id']: r for r in segmentation['records'] if r['visit'] == 'T0'}
    phase_sources = ({(r['patient_id'], r['visit']): r for r in read_json(DEFAULT_PHASE_SOURCES)['visits']}
                     if DEFAULT_PHASE_SOURCES.is_file() else {})
    tasks = []
    for patient in selection['patients']:
        case = Path(patient['T0_locator']['mask']).name.removesuffix('.nii.gz')
        record = records[case]
        native = record['patient_id']
        task = {'patient': patient, 'native_patient_id': native, 'visits': by_patient[patient['patient_id']]}
        require({v['visit'] for v in task['visits']} == set(patient['selected_visits']), 'Selection visits disagree')
        if patient['in_current_866']:
            old = old_patients[patient['patient_id']]
            task.update(native_metadata=str(file_paths[old['native_metadata']['path']]),
                        t0_metadata=str(file_paths[old['t0_metadata']['path']]),
                        phase_transform=str(file_paths[old['phase_transform']['path']]),
                        baseline=baseline[patient['patient_id']], expected_patient=old,
                        phase_transform_version_policy='archived_paired_transform_preserves_existing_crop')
        else:
            task.update(native_metadata=str(historical_meta(native, 'T0', 'preprocessed')),
                        t0_metadata=str(historical_meta(native, 'T0')),
                        phase_transform=str(PUBLIC / '03_registered_nifti' / native / 'T0/transforms/intravisit/phase_1_to_phase_0.tfm'),
                        phase_transform_version_policy='published_transform_with_preserved_native_metadata')
            if (patient['patient_id'], 'T0') in phase_sources:
                source = phase_sources[patient['patient_id'], 'T0']
                phase1_path = Path(source['source_phases'][source['phase_indices'].index(1)]['path'])
                paired_phase = phase1_path.parent.parent / 'transforms/intravisit/phase_1_to_phase_0.tfm'
                require(paired_phase.is_file(), 'Selected T0 phase1 array lacks its paired saved transform')
                task.update(phase_transform=str(paired_phase), phase1_image_source=str(phase1_path),
                            phase_transform_version_policy='transform_paired_to_selected_registered_phase1_array')
        task['source_grid_checks'] = validate_paired_metadata(read_json(task['native_metadata']), read_json(task['t0_metadata']))
        tasks.append(task)
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-dir', type=Path, default=DEFAULT_SELECTION)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--only-new', action='store_true')
    parser.add_argument('--retry-failures', action='store_true', help='Retain passed plans and recompute only the previous failures.')
    parser.add_argument('--refresh-patients', nargs='+', default=[], help='Explicit patient IDs to recompute during failure retry.')
    args = parser.parse_args()
    tasks = build_tasks(args.selection_dir)
    if args.only_new:
        tasks = [t for t in tasks if not t['patient']['in_current_866']]
    if args.limit is not None:
        tasks = tasks[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plans, failures = [], []
    total = len(tasks)
    if args.retry_failures:
        require(not args.only_new and args.limit is None, 'Failure retry cannot change cohort scope')
        previous = read_json(args.output_dir / 'summary.json')
        require(previous['patients_total'] == total, 'Failure retry cohort size changed')
        require(previous['selection_index'] == identity(args.selection_dir / 'selection_index.json'), 'Selection changed since previous audit')
        failed_ids = {row['patient_id'] for row in previous['failures']} | set(args.refresh_patients)
        require(failed_ids <= {task['patient']['patient_id'] for task in tasks}, 'Refresh patient lies outside selection')
        for task in tasks:
            pid = task['patient']['patient_id']
            if pid in failed_ids:
                continue
            plan = read_json(args.output_dir / 'patients' / (pid + '.json'))
            require(plan['status'] == 'passed' and plan['patient_id'] == pid, 'Cached plan is not passed')
            for saved in plan['source_identities'].values():
                require(identity(saved['path']) == saved, 'Cached plan source changed')
            plan['checks'].update(task['source_grid_checks'])
            plan.setdefault('phase_transform_version_policy', 'archived_paired_transform_preserves_existing_crop'
                            if plan['in_current_866'] else 'published_transform_with_preserved_native_metadata')
            write_json(args.output_dir / 'patients' / (pid + '.json'), plan)
            plans.append(plan)
        tasks = [task for task in tasks if task['patient']['patient_id'] in failed_ids]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(build_patient_plan, task): task['patient']['patient_id'] for task in tasks}
        for future in as_completed(futures):
            pid = futures[future]
            try:
                plan = future.result()
                write_json(args.output_dir / 'patients' / (pid + '.json'), plan)
                plans.append(plan)
            except Exception as error:
                failures.append({'patient_id': pid, 'error': str(error), 'traceback': traceback.format_exc()})
            completed = len(plans) + len(failures)
            if completed % 25 == 0 or completed == total:
                progress = {'completed': completed, 'passed': len(plans), 'failed': len(failures), 'total': total,
                            'updated_at_utc': datetime.now(timezone.utc).isoformat()}
                write_json(args.output_dir / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    plans.sort(key=lambda p: p['patient_id'])
    maxima = collections.defaultdict(float)
    for plan in plans:
        for key, value in plan['checks'].items():
            maxima[key] = max(maxima[key], value)
    summary = {
        'schema': SCHEMA, 'status': 'failed' if failures else 'passed',
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'selection_index': identity(args.selection_dir / 'selection_index.json'),
        'patients_total': total, 'patients_passed': len(plans), 'patients_failed': len(failures),
        'existing_866_passed': sum(p['in_current_866'] for p in plans),
        'new_110_passed': sum(not p['in_current_866'] for p in plans),
        'largest_component_clipped': sum(p['mask_qc']['largest_clipped'] for p in plans),
        'all_components_clipped': sum(p['mask_qc']['all_clipped'] for p in plans),
        'existing_largest_component_clipped': sum(p['in_current_866'] and p['mask_qc']['largest_clipped'] for p in plans),
        'existing_all_components_clipped': sum(p['in_current_866'] and p['mask_qc']['all_clipped'] for p in plans),
        'max_errors': dict(maxima), 'failures': failures,
        'public_phase_version_differences': [
            {'patient_id': p['patient_id'],
             'in_current_866': p['in_current_866'],
             'matrix_max_abs_difference': p['checks']['public_vs_archived_phase_matrix_max_error'],
             'selected_transform': p['source_identities']['phase_transform']['path'],
             'preserved_crop_affine_max_error_mm': p['checks'].get('legacy_crop_affine_max_error_mm')}
            for p in plans if p['checks'].get('public_vs_archived_phase_matrix_max_error', 0) >= 1e-10],
        'geometry_rule': 'Do not combine corrected public native JSON with unchanged historical MAMA-MIA headers; preserved paired geometry is used here.',
        'limitations': ['This verifies coordinates and mask support, not anatomical QC or MRI voxel export correctness.',
                        'Clipped masks are recorded without an admission gate.'],
        'plans': [{'patient_id': p['patient_id'], 'path': str(args.output_dir / 'patients' / (p['patient_id'] + '.json'))} for p in plans],
    }
    write_json(args.output_dir / 'summary.json', summary)
    write_json(args.output_dir / 'plans.json', {'schema': SCHEMA, 'status': summary['status'], 'patients': plans})
    print(json.dumps({k: v for k, v in summary.items() if k not in {'plans', 'failures'}}, indent=2), flush=True)
    if failures:
        print(json.dumps(failures[:5], indent=2), flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
