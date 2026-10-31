#!/usr/bin/env python3
"""Export version-matched, shared raw ROI32 arrays for the selected 976 patients."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform, map_coordinates
import SimpleITK as sitk


REPO = Path('/personal_fast_storage/VLA-PCRV1')
AUDITS = REPO / 'private_outputs'
GEOMETRY = AUDITS / 'medium976_geometry_audit_20261028'
SOURCES = AUDITS / 'medium976_source_audit_20261028/phase_sources_all3642.json'
BASELINE = REPO / 'data/registered_roi32/manifest.json'
DEFAULT_OUTPUT = REPO / 'data/registered_roi32_medium976_v1'
SHAPE = (32, 128, 128)
PROBES = np.array([[0, 0, 0], [15, 63, 63], [31, 127, 127], [9, 22, 117],
                   [12, 91, 43], [24, 31, 78]], dtype=int)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def check_identity(expected):
    if identity(expected['path']) != expected:
        raise ValueError(f"Source identity changed: {expected['path']}")


def sample(array, mapping, order):
    return affine_transform(array, mapping[:3, :3], mapping[:3, 3], output_shape=SHAPE,
                            order=order, mode='constant', cval=0, prefilter=False)


def save_npz(path, **arrays):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def export_patient(task):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    pid, plan_path, visits, output, old_normalization = task
    output = Path(output)
    marker = output / 'records' / (pid + '.json')
    plan = read(plan_path)
    if plan['status'] != 'passed' or plan['patient_id'] != pid:
        raise ValueError('Patient geometry plan is invalid')
    source_ids = [identity(plan_path)]
    source_ids.extend(plan['source_identities'].values())
    for visit in plan['visits']:
        source_ids.extend([visit['archived_metadata'], visit['public_metadata']])
    for visit in visits:
        source_ids.extend(visit['source_phases'])
        if visit['old_roi32_npz']:
            source_ids.append(identity(visit['old_roi32_npz']))
    for entry in source_ids:
        check_identity(entry)
    if marker.exists():
        previous = read(marker)
        if previous['source_identities'] != source_ids:
            raise ValueError('Existing export source contract changed; use a new dataset directory')
        for receipt in previous['visits']:
            check_identity(receipt['npz'])
        return previous

    mask_image = sitk.ReadImage(plan['source_identities']['t0_mask']['path'])
    mask = sitk.GetArrayFromImage(mask_image).astype(np.uint8)
    roi = sample(mask, np.asarray(plan['crop_to_native_mask_index_zyx']), 0).astype(bool)[None]
    if int(roi.sum()) != plan['actual_cropped_mask_voxels']:
        raise ValueError('Exported T0 mask differs from the audited geometry plan')
    if not roi.any():
        raise ValueError('T0-localized ROI has an empty exported mask')
    del mask, mask_image
    geometry = {visit['visit']: visit for visit in plan['visits']}
    receipts = []
    for visit in visits:
        stage = visit['visit']
        mapping = np.asarray(geometry[stage]['crop_to_registered_index_zyx'], dtype=np.float64)
        shape = tuple(visit['source_shape_zyx'])
        if tuple(geometry[stage]['shape_zyx']) != shape:
            raise ValueError('Phase source shape differs from paired geometry plan')
        coverage = sample(np.ones(shape, dtype=np.uint8), mapping, 0).astype(bool)
        image = np.zeros((3, *SHAPE), dtype=np.float32)
        foreground = np.zeros(image.shape, dtype=bool)
        support = np.broadcast_to(coverage, image.shape).copy()
        interpolation_error = 0.0
        for phase, source in enumerate(visit['source_phases']):
            check_identity(source)
            nifti = nib.load(source['path'])
            if tuple(nifti.shape) != shape:
                raise ValueError(f"NIfTI payload shape differs: {source['path']}")
            raw = np.asarray(nifti.dataobj, dtype=np.float32)
            if not np.isfinite(raw).all():
                raise ValueError(f"Non-finite source image: {source['path']}")
            cropped = sample(raw, mapping, 1)
            positions = PROBES @ mapping[:3, :3].T + mapping[:3, 3]
            expected = map_coordinates(raw, positions.T, order=1, mode='constant', cval=0, prefilter=False)
            actual = cropped[tuple(PROBES.T)]
            error = float(np.max(np.abs(expected - actual)))
            interpolation_error = max(interpolation_error, error)
            if not np.allclose(actual, expected, atol=1e-3, rtol=2e-6):
                raise ValueError('Independent coordinate/interpolation sampler disagrees')
            valid = coverage & (cropped != 0)
            if not valid.any() or not np.isfinite(cropped).all():
                raise ValueError('MRI crop is empty or non-finite')
            image[phase, valid] = cropped[valid]
            foreground[phase] = valid
            del nifti, raw, cropped
        if not np.array_equal(foreground, support & (image != 0)) or np.any(image[~foreground] != 0):
            raise ValueError('Raw storage foreground/background contract differs')
        replay_error = None
        if visit['old_roi32_npz']:
            with np.load(visit['old_roi32_npz'], allow_pickle=False) as prior:
                prior_image, prior_support, prior_roi = prior['image'], prior['support'], prior['roi']
            replay = np.zeros_like(image)
            replay[foreground] = ((image[foreground] - old_normalization['mean']) / old_normalization['std'])
            replay_error = float(np.max(np.abs(replay - prior_image)))
            # Preserve the already-established ROI32/ROI64 replay acceptance contract.
            if not np.array_equal(support, prior_support):
                raise ValueError('Reproduced ROI32 acquisition support differs')
            if not np.array_equal(roi, prior_roi):
                raise ValueError('Reproduced T0 ROI differs from the existing archive')
            if not np.array_equal(replay == 0, prior_image == 0):
                raise ValueError('Reproduced ROI32 zero pattern differs')
            if not np.allclose(replay, prior_image, atol=1e-5, rtol=1e-5):
                raise ValueError('Reproduced ROI32 intensities differ; source/geometry bundle is inconsistent')
            del prior_image, prior_support, prior_roi, replay
        path = output / 'npz' / (pid + '_' + stage + '.npz')
        save_npz(path, image=image, foreground=foreground, support=support, roi=roi,
                 crop_affine_ras=np.asarray(plan['crop_affine_ras'], dtype=np.float64),
                 phase_indices=np.asarray(visit['phase_indices'], dtype=np.int16))
        # Read every array back after compressed export, rather than relying on a write receipt.
        with np.load(path, allow_pickle=False) as archive:
            for key, expected in [('image', image), ('foreground', foreground), ('support', support), ('roi', roi)]:
                actual = archive[key]
                if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
                    raise ValueError(f'NPZ round-trip validation failed for {key}')
        values = image[0, foreground[0]].astype(np.float64)
        receipts.append({
            'patient_id': pid, 'visit': stage, 'visit_id': visit['visit_id'], 'npz': identity(path),
            'image_shape': list(image.shape), 'image_dtype': str(image.dtype),
            'foreground_voxels_by_phase': [int(x.sum()) for x in foreground],
            'support_voxels_by_phase': [int(x.sum()) for x in support],
            'roi_voxels': int(roi.sum()), 'raw_min': float(image.min()), 'raw_max': float(image.max()),
            'DCE0_count': int(values.size), 'DCE0_sum': float(values.sum()),
            'DCE0_square_sum': float(np.square(values).sum()),
            'independent_interpolation_max_abs_error': interpolation_error,
            'original_ROI32_replay_max_abs_error': replay_error,
            'phase_indices': visit['phase_indices'], 'source_selection': visit['source_selection'],
            'crop_to_registered_index_zyx': mapping.tolist(),
            'all_arrays_roundtrip_equal': True, 'finite_source_and_crop_checked': True,
            'zero_background_verified': True,
        })
    result = {'schema': 'medium976_raw_roi32_patient_export_v1', 'status': 'completed',
              'patient_id': pid, 'source_identities': source_ids, 'geometry_plan': str(plan_path),
              'crop_affine_ras': plan['crop_affine_ras'], 'shape_zyx': list(SHAPE),
              'spacing_zyx_mm': plan['spacing_zyx_mm'], 'visits': receipts,
              'updated_at_utc': datetime.now(timezone.utc).isoformat()}
    write(marker, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--patients', nargs='*')
    parser.add_argument('--exclude-patients', nargs='*', default=[])
    args = parser.parse_args()
    args.output = args.output.resolve()
    for name in ['npz', 'records', 'geometry']:
        (args.output / name).mkdir(parents=True, exist_ok=True)
    geometry_summary = read(GEOMETRY / 'summary.json')
    if geometry_summary['status'] != 'passed' or geometry_summary['patients_passed'] != 976:
        raise ValueError('Complete 976-patient paired geometry audit must pass before export')
    source_index = read(SOURCES)
    overrides_path = args.output / 'availability_overrides.json'
    overrides = read(overrides_path) if overrides_path.exists() else {'unusable_visits': []}
    unusable = {visit['visit_id'] for visit in overrides['unusable_visits']}
    if unusable and (unusable != {'ISPY2-893874:T3'} or len(overrides['unusable_visits']) != 1):
        raise ValueError('Unexpected availability override; review the selection before exporting')
    by_patient = defaultdict(list)
    for visit in source_index['visits']:
        if visit['visit_id'] in unusable:
            continue
        by_patient[visit['patient_id']].append(visit)
    pids = sorted(set(args.patients or by_patient) - set(args.exclude_patients))
    if not set(pids) <= set(by_patient):
        raise ValueError('Unknown requested patient')
    old_norm = read(BASELINE)['image_normalization']
    tasks = []
    for pid in pids:
        plan = read(GEOMETRY / 'patients' / (pid + '.json'))
        target = args.output / 'geometry' / (pid + '.json')
        if target.exists():
            if read(target) != plan:
                raise ValueError('Frozen geometry plan changed')
        else:
            write(target, plan)
        tasks.append((pid, str(target), by_patient[pid], str(args.output), old_norm))
    expected_visits = sum(len(by_patient[pid]) for pid in pids)
    started = time.monotonic()
    completed, visits_done, failures = [], 0, []
    status_path = args.output / 'export_status.json'
    def progress(status):
        value = {'stage': 'raw_roi32_export', 'status': status,
                 'patients_total': len(tasks), 'patients_completed': len(completed),
                 'visits_total': expected_visits, 'visits_completed': visits_done,
                 'failed_patients': len(failures), 'elapsed_seconds': time.monotonic() - started,
                 'updated_at_utc': datetime.now(timezone.utc).isoformat()}
        write(status_path, value)
        return value
    progress('running')
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(export_patient, task): task[0] for task in tasks}
        for future in as_completed(pending):
            pid = pending[future]
            try:
                receipt = future.result()
                completed.append(receipt)
                visits_done += len(receipt['visits'])
            except Exception as exc:
                failures.append({'patient_id': pid, 'error': str(exc)})
                write(args.output / 'export_failures.json', failures)
            value = progress('running')
            if len(completed) % 20 == 0 or failures:
                print(json.dumps(value), flush=True)
    summary = progress('completed' if not failures else 'failed')
    replay = [v['original_ROI32_replay_max_abs_error'] for p in completed for v in p['visits']
              if v['original_ROI32_replay_max_abs_error'] is not None]
    summary.update(schema='medium976_raw_roi32_export_summary_v1',
                   candidate_visits=len(source_index['visits']),
                   unusable_visits=overrides['unusable_visits'],
                   visits_read_back_validated=visits_done, source_phase_files_checked=3 * visits_done,
                   original_ROI32_replay_visits=len(replay),
                   original_ROI32_replay_max_abs_error=max(replay, default=None),
                   independent_interpolation_max_abs_error=max((v['independent_interpolation_max_abs_error']
                       for p in completed for v in p['visits']), default=0),
                   image_storage='Raw ROI32, explicit foreground, per-fold train-only normalization on read.',
                   failures=failures, normalization_fitted=False, training_manifest_generated=False)
    write(args.output / 'export_summary.json', summary)
    if failures or visits_done != expected_visits:
        raise SystemExit(1)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
