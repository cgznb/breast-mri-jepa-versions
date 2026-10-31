#!/usr/bin/env python3
"""Validate and aggregate the complete 976-patient export without resampling.

The frozen geometry files are never rewritten. All NPZ identities and array
headers are checked; the exporter already read every exported array back and
recorded its numerical checks. Optional small samples can be reloaded here.
"""
from __future__ import annotations

import argparse
import ast
import collections
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import struct
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'data/registered_roi32_medium976_v1'
DEFAULT_GEOMETRY = ROOT / 'private_outputs/medium976_geometry_audit_20261028'
DEFAULT_SOURCES = ROOT / 'private_outputs/medium976_source_audit_20261028/phase_sources_all3642.json'
DEFAULT_REPORT = ROOT / 'private_outputs/medium976_roi32_v1/finalize_validation.json'
EXPECTED_PATIENTS, EXPECTED_VISITS, EXPECTED_REPLAY = 976, 3642, 3039
SCHEMA = 'medium976_raw_roi32_export_finalize_v1'
SHAPE = [3, 32, 128, 128]
# These fields cannot change the sampled arrays. Source identities, coordinate
# frames, all matrices, dimensions, mask counts, and visit data remain strict.
DIAGNOSTIC_FIELDS = frozenset({'checks', 'verification_limits', 'coordinate_policy',
                               'segmentation_review_status', 'segmentation_flags',
                               'phase_transform_version_policy'})
ARRAY_HEADERS = {
    'image': ((3, 32, 128, 128), '<f4', 4),
    'foreground': ((3, 32, 128, 128), '|b1', 1),
    'support': ((3, 32, 128, 128), '|b1', 1),
    'roi': ((1, 32, 128, 128), '|b1', 1),
    'crop_affine_ras': ((4, 4), '<f8', 8),
    'phase_indices': ((3,), '<i2', 2),
}


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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value, name, *, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value), f'Invalid numeric receipt: {name}')
    require(not nonnegative or value >= 0, f'Negative numeric receipt: {name}')
    return value


def differences(left, right, prefix=''):
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            path = prefix + '.' + key if prefix else key
            if key not in left or key not in right:
                yield path
            else:
                yield from differences(left[key], right[key], path)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            yield prefix + '.length'
        else:
            for index, (a, b) in enumerate(zip(left, right)):
                yield from differences(a, b, prefix + f'[{index}]')
    elif type(left) is not type(right) or left != right:
        yield prefix


def check_geometry(frozen, latest, pid):
    require(frozen['status'] == latest['status'] == 'passed', 'Frozen or latest geometry did not pass')
    require(frozen['patient_id'] == latest['patient_id'] == pid, 'Geometry patient identity differs')
    core_left = {k: v for k, v in frozen.items() if k not in DIAGNOSTIC_FIELDS}
    core_right = {k: v for k, v in latest.items() if k not in DIAGNOSTIC_FIELDS}
    core_diff = list(differences(core_left, core_right))
    require(not core_diff, 'Sampling/source geometry changed: ' + ', '.join(core_diff[:8]))
    require(frozen['crop_shape_zyx'] == [32, 128, 128], 'Frozen crop shape differs')
    require(frozen['spacing_zyx_mm'] == [2.0, 0.7032, 0.7032], 'Frozen crop spacing differs')
    for field in ('crop_affine_ras', 'crop_to_native_mask_index_zyx', 'crop_to_registered_index_zyx'):
        matrix = frozen[field]
        require(len(matrix) == 4 and all(len(row) == 4 for row in matrix), f'Invalid matrix: {field}')
        require(all(type(x) in (int, float) and math.isfinite(x) for row in matrix for x in row), f'Nonfinite matrix: {field}')
        require(matrix[3] == [0.0, 0.0, 0.0, 1.0], f'Invalid homogeneous matrix: {field}')
    return list(differences(frozen, latest))


def check_npz_headers(path):
    """Read ZIP directory and NPY headers, without loading the array payloads."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(ARRAY_HEADERS) and set(names) == {k + '.npy' for k in ARRAY_HEADERS},
                'NPZ array keys differ from the export contract')
        for key, (shape, dtype, width) in ARRAY_HEADERS.items():
            info = archive.getinfo(key + '.npy')
            with archive.open(info) as stream:
                require(stream.read(6) == b'\x93NUMPY', 'Invalid NPY magic')
                version = stream.read(2)
                require(version in (b'\x01\x00', b'\x02\x00', b'\x03\x00'), 'Unsupported NPY header version')
                size_width = 2 if version[0] == 1 else 4
                raw_size = stream.read(size_width)
                require(len(raw_size) == size_width, 'Truncated NPY header length')
                header_size = struct.unpack('<H' if size_width == 2 else '<I', raw_size)[0]
                require(0 < header_size <= 16384, 'Invalid NPY header size')
                raw_header = stream.read(header_size)
                require(len(raw_header) == header_size, 'Truncated NPY header')
                header = ast.literal_eval(raw_header.decode('utf-8' if version[0] == 3 else 'latin1').strip())
                require(header['shape'] == shape and header['descr'] == dtype and header['fortran_order'] is False,
                        f'NPZ array shape/dtype/order differs: {key}')
                require(info.file_size == stream.tell() + math.prod(shape) * width,
                        f'NPY payload length disagrees with header: {key}')


def expected_source_identities(frozen_path, plan, sources):
    entries = [identity(frozen_path), *plan['source_identities'].values()]
    for visit in plan['visits']:
        entries.extend((visit['archived_metadata'], visit['public_metadata']))
    for visit in sources:
        entries.extend(visit['source_phases'])
        if visit['old_roi32_npz']:
            entries.append(identity(visit['old_roi32_npz']))
    return entries


def availability_overrides(output, candidate_visit_ids):
    """Admit only the explicitly audited single unusable recovered T3 visit."""
    path = output / 'availability_overrides.json'
    if not path.is_file():
        return [], None
    payload = read(path)
    require(payload.get('schema') == 'medium976_verified_availability_overrides_v1', 'Unknown availability override schema')
    rows = payload.get('unusable_visits')
    require(isinstance(rows, list) and len(rows) == 1, 'Only one explicitly verified unusable visit is authorized')
    row = rows[0]
    require((row.get('patient_id'), row.get('visit'), row.get('visit_id')) ==
            ('ISPY2-893874', 'T3', 'ISPY2-893874:T3'), 'Availability override is outside the verified single-visit scope')
    require(row['visit_id'] in candidate_visit_ids, 'Availability override visit is absent from candidate source index')
    require(isinstance(row.get('reason'), str) and bool(row['reason'].strip()), 'Availability override lacks a reason')
    evidence = row.get('evidence')
    require(isinstance(evidence, list) and evidence, 'Availability override lacks saved evidence')
    require(all(isinstance(p, str) and Path(p).is_absolute() and Path(p).is_file() for p in evidence),
            'Availability override evidence must exist at absolute file paths')
    return rows, identity(path)


def check_visit_receipt(receipt, source, geometry, plan, output):
    stage, pid = source['visit'], plan['patient_id']
    require(receipt['patient_id'] == pid and receipt['visit'] == stage and receipt['visit_id'] == source['visit_id'],
            'Visit receipt identity differs from selected source')
    require(receipt['image_shape'] == SHAPE and receipt['image_dtype'] == 'float32', 'Image receipt shape/dtype differs')
    require(receipt['phase_indices'] == source['phase_indices'], 'Visit phase indices changed')
    require(len(source['phase_indices']) == 3 and source['phase_indices'][:2] == [0, 1]
            and source['phase_indices'][2] > 1, 'Invalid explicit three-phase selection')
    require(receipt['source_selection'] == source['source_selection'], 'Visit phase source selection/version changed')
    require(receipt['crop_to_registered_index_zyx'] == geometry['crop_to_registered_index_zyx'], 'Export receipt sampling matrix differs')
    require(tuple(source['source_shape_zyx']) == tuple(geometry['shape_zyx']), 'Source and geometry array shapes differ')
    for flag in ('all_arrays_roundtrip_equal', 'finite_source_and_crop_checked', 'zero_background_verified'):
        require(receipt[flag] is True, f'Numerical export verification did not pass: {flag}')
    foreground, support = receipt['foreground_voxels_by_phase'], receipt['support_voxels_by_phase']
    require(len(foreground) == len(support) == 3 and len(set(support)) == 1, 'Invalid phase support receipt')
    for fg, cov in zip(foreground, support):
        require(type(fg) is int and type(cov) is int and 0 < fg <= cov <= math.prod(SHAPE[1:]),
                'Empty or invalid phase foreground/support receipt')
    require(receipt['roi_voxels'] == plan['actual_cropped_mask_voxels'] and receipt['roi_voxels'] > 0,
            'Exported T0 mask count differs from paired geometry')
    require(receipt['DCE0_count'] == foreground[0], 'DCE0 statistics count differs from foreground')
    for name in ('raw_min', 'raw_max', 'DCE0_sum', 'DCE0_square_sum'):
        finite_number(receipt[name], name)
    require(receipt['raw_min'] <= receipt['raw_max'] and receipt['DCE0_square_sum'] >= 0, 'Invalid intensity statistics')
    finite_number(receipt['independent_interpolation_max_abs_error'], 'probe error', nonnegative=True)
    replay = receipt['original_ROI32_replay_max_abs_error']
    require((replay is not None) == source['in_original_3039'], 'Original-ROI32 replay coverage differs')
    if replay is not None:
        finite_number(replay, 'ROI32 replay error', nonnegative=True)
        require(source['roi64_contract_source_identities_match_now'] is True, 'Original source identity guard did not pass')
    path = output / 'npz' / f'{pid}_{stage}.npz'
    require(receipt['npz'] == identity(path), 'NPZ identity changed after export verification')
    check_npz_headers(path)
    return path


def reload_sample(path, receipt, plan):
    """Optional independent small sample; all ordinary validation stays light."""
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        image, foreground, support, roi = (archive[key] for key in ('image', 'foreground', 'support', 'roi'))
        require(np.isfinite(image).all(), 'Sampled NPZ has nonfinite image values')
        require(np.array_equal(foreground, support & (image != 0)), 'Sampled foreground/background contract differs')
        require(np.array_equal(archive['crop_affine_ras'], np.asarray(plan['crop_affine_ras'])), 'Sampled NPZ crop affine differs')
        require(np.array_equal(archive['phase_indices'], np.asarray(receipt['phase_indices'])), 'Sampled NPZ phase indices differ')
        require([int(a.sum()) for a in foreground] == receipt['foreground_voxels_by_phase'], 'Sampled foreground counts differ')
        require([int(a.sum()) for a in support] == receipt['support_voxels_by_phase'], 'Sampled support counts differ')
        require(int(roi.sum()) == receipt['roi_voxels'], 'Sampled T0 mask count differs')
        require(float(image.min()) == receipt['raw_min'] and float(image.max()) == receipt['raw_max'], 'Sampled intensity range differs')


def validate(output, geometry_dir, sources_path, sample_npz=0):
    started = time.monotonic()
    output = output.resolve()
    source_index = read(sources_path)
    candidate_by_patient = collections.defaultdict(list)
    for source in source_index['visits']:
        candidate_by_patient[source['patient_id']].append(source)
    expected_pids = set(candidate_by_patient)
    candidate_visit_ids = {source['visit_id'] for source in source_index['visits']}
    require(len(expected_pids) == EXPECTED_PATIENTS and len(source_index['visits']) == len(candidate_visit_ids) == EXPECTED_VISITS,
            'Source index is not the complete unique 976/3642 cohort')
    require(sum(source['in_original_3039'] for source in source_index['visits']) == EXPECTED_REPLAY,
            'Source index original-3039 coverage differs')
    unavailable, override_identity = availability_overrides(output, candidate_visit_ids)
    excluded_ids = {row['visit_id'] for row in unavailable}
    effective_sources = [source for source in source_index['visits'] if source['visit_id'] not in excluded_ids]
    require(sum(source['in_original_3039'] for source in effective_sources) == EXPECTED_REPLAY,
            'Availability override removes a protected original-3039 replay visit')
    by_patient = collections.defaultdict(list)
    for source in effective_sources:
        by_patient[source['patient_id']].append(source)
    require(set(by_patient) == expected_pids, 'Availability override removes a whole selected patient')
    expected_visit_ids = {source['visit_id'] for source in effective_sources}
    effective_visit_total = len(expected_visit_ids)
    audit_summary = read(geometry_dir / 'summary.json')
    require(audit_summary['status'] == 'passed' and audit_summary['patients_passed'] == EXPECTED_PATIENTS,
            'Latest complete geometry audit has not passed')
    frozen_paths = {p.stem: p for p in (output / 'geometry').glob('*.json')}
    record_paths = {p.stem: p for p in (output / 'records').glob('*.json')}
    npz_paths = set((output / 'npz').glob('*.npz'))
    errors, missing_records = [], sorted(expected_pids - set(record_paths))
    require(set(frozen_paths) == expected_pids, 'Frozen geometry coverage differs from all 976 selected patients')
    require(set(record_paths) <= expected_pids, 'Export contains an unexpected patient record')
    expected_npz_paths = {output / 'npz' / f"{source['patient_id']}_{source['visit']}.npz" for source in effective_sources}
    require(npz_paths <= expected_npz_paths, 'Export contains an unselected NPZ')
    diagonal_differences, diagonal_patients = collections.Counter(), []
    source_identity_cache, record_identities, visit_receipts = {}, [], []
    validated_patients, validated_geometry = [], 0
    npz_bytes, source_phase_instances = 0, 0
    phase_version_counts = collections.Counter()
    frozen_identities = []
    for pid in sorted(expected_pids):
        try:
            frozen_path = frozen_paths[pid]
            frozen, latest = read(frozen_path), read(geometry_dir / 'patients' / (pid + '.json'))
            diff = check_geometry(frozen, latest, pid)
            validated_geometry += 1
            frozen_identities.append(identity(frozen_path))
            if diff:
                diagonal_patients.append(pid)
                diagonal_differences.update(diff)
            sources = by_patient[pid]
            geometry = {v['visit']: v for v in frozen['visits']}
            require(set(geometry) == {v['visit'] for v in candidate_by_patient[pid]}, 'Frozen candidate geometry visit coverage differs')
            # Verify the selected phase1 anchor when the plan explicitly retains it.
            anchor = frozen['source_identities'].get('paired_registered_phase1_image')
            if anchor is not None:
                t0 = next(v for v in sources if v['visit'] == 'T0')
                require(anchor == t0['source_phases'][t0['phase_indices'].index(1)], 'Selected T0 phase1 image version changed')
                phase_path = Path(anchor['path']).parent.parent / 'transforms/intravisit/phase_1_to_phase_0.tfm'
                require(frozen['source_identities']['phase_transform']['path'] == str(phase_path), 'T0 phase transform and image bundle differ')
            if pid not in record_paths:
                continue
            record_path = record_paths[pid]
            before = identity(record_path)
            record = read(record_path)
            require(identity(record_path) == before, 'Export record changed while being read')
            require(record['schema'] == 'medium976_raw_roi32_patient_export_v1' and record['status'] == 'completed'
                    and record['patient_id'] == pid, 'Invalid completed patient export receipt')
            require(record['geometry_plan'] == str(frozen_path), 'Export record references a different frozen plan')
            require(record['crop_affine_ras'] == frozen['crop_affine_ras'], 'Export record crop affine changed')
            require(record['shape_zyx'] == [32, 128, 128] and record['spacing_zyx_mm'] == frozen['spacing_zyx_mm'],
                    'Export record crop shape/spacing differs')
            expected_sources = expected_source_identities(frozen_path, frozen, sources)
            require(record['source_identities'] == expected_sources, 'Export source identity list/order differs from frozen plan and final phase index')
            for saved in expected_sources:
                path = saved['path']
                current = source_identity_cache.get(path)
                if current is None:
                    current = identity(path)
                    source_identity_cache[path] = current
                require(current == saved, 'Source identity changed after export: ' + path)
            receipts = {v['visit']: v for v in record['visits']}
            require(len(receipts) == len(record['visits']) and set(receipts) == {v['visit'] for v in sources},
                    'Effective export receipt visits are missing/duplicated')
            for source in sources:
                receipt = receipts[source['visit']]
                npz = check_visit_receipt(receipt, source, geometry[source['visit']], frozen, output)
                npz_bytes += receipt['npz']['bytes']
                source_phase_instances += len(source['source_phases'])
                phase_version_counts.update(source['source_selection'])
                visit_receipts.append((npz, receipt, frozen))
            record_identities.append(before)
            validated_patients.append(pid)
        except Exception as error:
            errors.append({'patient_id': pid, 'error': str(error)})
    seen_visits = [receipt['visit_id'] for _, receipt, _ in visit_receipts]
    require(len(seen_visits) == len(set(seen_visits)), 'Aggregated visit receipts contain duplicates')
    sampled = []
    if sample_npz and visit_receipts:
        # Stable, evenly spaced visits, without repeatedly loading a whole cohort.
        ordered = sorted(visit_receipts, key=lambda row: row[1]['visit_id'])
        count = min(sample_npz, len(ordered))
        indices = sorted({round(i * (len(ordered) - 1) / max(1, count - 1)) for i in range(count)})
        for index in indices:
            path, receipt, plan = ordered[index]
            try:
                reload_sample(path, receipt, plan)
                sampled.append(receipt['visit_id'])
            except Exception as error:
                errors.append({'visit_id': receipt['visit_id'], 'error': str(error)})
    replay = [r['original_ROI32_replay_max_abs_error'] for _, r, _ in visit_receipts
              if r['original_ROI32_replay_max_abs_error'] is not None]
    historical_path = output / 'export_failures.json'
    historical_failures = read(historical_path) if historical_path.is_file() else []
    require(isinstance(historical_failures, list), 'Historical export failure log is not a list')
    resolved_history = []
    for failure in historical_failures:
        resolutions = [row for row in unavailable if row['patient_id'] == failure.get('patient_id')]
        if resolutions and failure.get('patient_id') in validated_patients:
            resolved_history.append({**failure, 'resolution': 'resolved_by_explicit_unusable_visit_override',
                                     'unusable_visit_id': resolutions[0]['visit_id']})
        else:
            resolved_history.append({**failure, 'resolution': 'completed_export_record_verified'
                                     if failure.get('patient_id') in validated_patients else 'not_yet_resolved'})
    complete = (not missing_records and not errors and len(validated_patients) == EXPECTED_PATIENTS
                and len(seen_visits) == effective_visit_total and len(replay) == EXPECTED_REPLAY
                and npz_paths == expected_npz_paths)
    if not missing_records and not complete and not errors:
        errors.append({'error': 'All patient markers exist but formal patient/effective-visit/original-replay coverage is incomplete'})
    status = 'failed' if errors else 'passed' if complete else 'incomplete'
    return {
        'schema': SCHEMA, 'status': status, 'complete': complete,
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'patients_total': EXPECTED_PATIENTS, 'patients_completed': len(validated_patients),
        'candidate_visits_total': EXPECTED_VISITS,
        'visits_total': effective_visit_total, 'effective_visits_total': effective_visit_total,
        'visits_completed': len(seen_visits),
        'unusable_visits_count': len(unavailable), 'unusable_visits': unavailable,
        'availability_overrides': override_identity,
        'frozen_geometry_plans_checked': validated_geometry,
        'frozen_geometry_files_rewritten': 0,
        'non_sampling_diagnostic_difference_patients': diagonal_patients,
        'non_sampling_diagnostic_difference_fields': dict(diagonal_differences),
        'sampling_and_source_plan_differences': 0 if validated_geometry == EXPECTED_PATIENTS else None,
        'patient_record_identities_checked': len(record_identities),
        'unique_source_file_identities_checked': len(source_identity_cache),
        'source_phase_files_checked': source_phase_instances,
        'source_phase_version_counts': dict(phase_version_counts),
        'npz_files_present': len(npz_paths), 'npz_identities_and_array_headers_checked': len(seen_visits),
        'npz_arrays_reloaded_for_samples': len(sampled), 'sample_visit_ids': sampled,
        'npz_bytes': npz_bytes, 'visits_read_back_validated_by_exporter': len(seen_visits),
        'all_arrays_roundtrip_equal': all(r['all_arrays_roundtrip_equal'] for _, r, _ in visit_receipts),
        'all_sources_and_crops_finite_checked': all(r['finite_source_and_crop_checked'] for _, r, _ in visit_receipts),
        'all_zero_background_verified': all(r['zero_background_verified'] for _, r, _ in visit_receipts),
        'original_ROI32_replay_visits': len(replay),
        'original_ROI32_replay_max_abs_error': max(replay, default=None),
        'original_ROI32_replay_acceptance': 'Exporter verified per-voxel allclose(atol=1e-5, rtol=1e-5), support/ROI/zero-pattern equality.',
        'independent_interpolation_max_abs_error': max((r['independent_interpolation_max_abs_error'] for _, r, _ in visit_receipts), default=None),
        'missing_patient_records': missing_records,
        'unreceipted_npz_files_present': len(npz_paths - {row[0] for row in visit_receipts}),
        'errors': errors, 'aggregation_elapsed_seconds': time.monotonic() - started,
        'historical_export_failure_log': identity(historical_path) if historical_path.is_file() else None,
        'historical_export_failure_log_preserved': True,
        'historical_export_failures': resolved_history,
        'historical_failures_resolved_by_explicit_unusable_visit_override': sum(
            row['resolution'] == 'resolved_by_explicit_unusable_visit_override' for row in resolved_history),
        'geometry_audit_summary': identity(geometry_dir / 'summary.json'),
        'phase_source_index': identity(sources_path),
        'validation_method': 'Read-only plan/receipt/source identity validation and every NPZ array header; no MRI resampling or whole-cohort NPZ reload.',
        'limitations': ['Numerical whole-array roundtrip/finiteness/replay checks are retained exporter receipts; this finalizer verifies their file identities and structure.',
                        'These checks establish coordinate/data consistency, not anatomical segmentation or registration quality.'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--geometry-dir', type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument('--sources', type=Path, default=DEFAULT_SOURCES)
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--check-only', action='store_true', help='Never update export status or summary.')
    parser.add_argument('--allow-incomplete', action='store_true', help='Allow a partial check while export workers are still running; requires --check-only.')
    parser.add_argument('--sample-npz', type=int, default=0, help='Optional number of evenly spaced NPZ arrays to reload.')
    args = parser.parse_args()
    require(not args.allow_incomplete or args.check_only, 'Incomplete mode requires --check-only')
    require(args.sample_npz >= 0, 'Sample count must be nonnegative')
    report = validate(args.output, args.geometry_dir, args.sources, args.sample_npz)
    write(args.report, report)
    if not args.check_only and report['complete']:
        summary = {
            **report, 'schema': 'medium976_raw_roi32_export_summary_v1', 'stage': 'raw_roi32_export',
            'status': 'completed', 'failed_patients': 0, 'failures': [],
            'visits_read_back_validated': report['visits_read_back_validated_by_exporter'],
            'image_storage': 'Raw ROI32, explicit foreground, per-fold train-only normalization on read.',
            'normalization_fitted': False, 'training_manifest_generated': False,
            'finalization_report': identity(args.report),
        }
        write(args.output / 'export_summary.json', summary)
        write(args.output / 'export_status.json', {
            'stage': 'raw_roi32_export', 'status': 'completed',
            'patients_total': EXPECTED_PATIENTS, 'patients_completed': EXPECTED_PATIENTS,
            'candidate_visits_total': EXPECTED_VISITS,
            'visits_total': report['effective_visits_total'], 'visits_completed': report['effective_visits_total'],
            'effective_visits_total': report['effective_visits_total'],
            'unusable_visits_count': report['unusable_visits_count'],
            'availability_overrides': report['availability_overrides'],
            'failed_patients': 0, 'updated_at_utc': report['updated_at_utc'],
            'finalized_from_all_patient_records': True, 'summary': str(args.output / 'export_summary.json'),
            'finalization_report': str(args.report),
        })
    print(json.dumps({k: v for k, v in report.items() if k not in {
        'missing_patient_records', 'non_sampling_diagnostic_difference_patients', 'limitations'}}, indent=2), flush=True)
    if report['errors'] or (not report['complete'] and not args.allow_incomplete):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
