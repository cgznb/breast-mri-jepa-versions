#!/usr/bin/env python3
"""Prepare a private patient-stratified internal cohort from upstream raw ROIs.

Intensity fitting uses only internal-training DCE0 foreground, then applies that
shared scale to all three phases. Additional phases are re-cropped with the
existing T0 geometry and the upstream raw-foreground policy; normalized zeros
are never used to infer foreground. Existing assets are read-only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.model_selection import train_test_split

from response_jepa.data import PHASES, Store, canonical_key
from response_jepa.provenance import file_identity, record_metadata

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path('/root/autodl-tmp/MAM/MeWM-ISPY2')
DEFAULT_MANIFEST = Path('/root/autodl-tmp/line b/breast-mri-vla-jepa/data/registered_roi32/manifest.json')
DEFAULT_RAW = UPSTREAM / 'runs/registered_dce0_roi32_firstpostmask_v1/data'
DEFAULT_PHASES = UPSTREAM / 'runs/registered_three_phase_roi32_v1/inventory.json'
SHAPE = (32, 128, 128)


def read_json(path):
    return json.loads(Path(path).read_text())


def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def private_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, encoding='utf-8') as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def split_patients(manifest, seed=20261009):
    """Split original training patients; never reuse the old validation cohort."""
    original = [p for p in manifest['patients'] if p['split'] == 'train']
    original.sort(key=lambda p: canonical_key(p['patient_key']))
    keys = [canonical_key(p['patient_key']) for p in original]
    labels = [p['target']['pcr'] for p in original]
    if len(set(keys)) != len(keys) or any(y not in (0, 1) for y in labels):
        raise ValueError('Internal development requires unique patients and binary labels')
    train, val = train_test_split(keys, test_size=.2, random_state=seed, stratify=labels)
    partitions = {'train': sorted(train), 'val': sorted(val)}
    old_validation = {canonical_key(p['patient_key']) for p in manifest['patients'] if p['split'] != 'train'}
    if set(train) & set(val) or (set(train) | set(val)) & old_validation:
        raise ValueError('Patient overlap in internal development split')
    return partitions


def raw_path(raw_root, visit_id):
    return Path(raw_root) / 'images' / (visit_id.replace(':', '_') + '.npz')


def read_raw(path):
    with np.load(path, allow_pickle=False) as archive:
        if not {'image', 'valid', 'coverage'}.issubset(archive.files):
            raise ValueError('Raw DCE0 archive requires image, valid and coverage')
        raw, valid, support = archive['image'], archive['valid'], archive['coverage']
    if (raw.shape != SHAPE or raw.dtype != np.float32 or valid.shape != SHAPE
            or support.shape != SHAPE or valid.dtype != np.bool_ or support.dtype != np.bool_
            or not np.isfinite(raw).all() or not valid.any() or np.any(valid & ~support)):
        raise ValueError('Invalid upstream raw ROI/foreground/coverage contract')
    # This validates the saved upstream policy against RAW intensities. It does
    # not infer a mask from values in a previously normalized image.
    if not np.array_equal(valid, support & (raw != 0)):
        raise ValueError('Saved raw foreground disagrees with upstream policy')
    return raw, valid, support


def raw_statistics(path):
    raw, valid, _ = read_raw(path)
    values = raw[valid].astype(np.float64)
    return len(values), float(values.sum()), float(np.square(values).sum())


def fit_image_normalization(records, train_keys, raw_root, workers=6):
    fitted = [r for r in records if canonical_key(r['patient_id']) in set(train_keys)]
    paths = [raw_path(raw_root, r['visit_id']) for r in fitted]
    if len(set(paths)) != len(paths) or not paths:
        raise ValueError('Image normalization requires unique training visits')
    with ThreadPoolExecutor(max_workers=workers) as executor:
        stats = list(executor.map(raw_statistics, paths))
    count = sum(s[0] for s in stats)
    mean = math.fsum(s[1] for s in stats) / count
    variance = math.fsum(s[2] for s in stats) / count - mean**2
    std = math.sqrt(max(0., variance))
    if not math.isfinite(std) or std <= 1e-8:
        raise ValueError('Degenerate internal training intensity statistics')
    return {'mean': mean, 'std': std, 'fit_split': 'train',
            'scope': 'unique_internal_train_dce0_crop_raw_foreground',
            'background_policy': 'preserve_zero', 'already_normalized': True,
            'normalization_policy': 'internal_train_dce0_shared_across_phases',
            'count': count, 'fit_patient_count': len(train_keys), 'fit_visit_count': len(paths)}


def normalize_raw(raw, valid, normalization):
    if raw.shape != valid.shape or valid.dtype != np.bool_ or not np.isfinite(raw).all():
        raise ValueError('Invalid raw phase or foreground')
    image = np.zeros_like(raw, dtype=np.float32)
    image[valid] = (raw[valid] - normalization['mean']) / normalization['std']
    if not np.isfinite(image).all():
        raise ValueError('Non-finite normalized image')
    return image


def write_arrays(path, arrays, *, compressed):
    path = Path(path)
    with tempfile.NamedTemporaryFile('wb', dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            if compressed:
                np.savez_compressed(handle, **arrays)
            else:
                np.save(handle, arrays['image'], allow_pickle=False)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_visit(task):
    record, raw_root, output, normalization, old_image, old_normalization, check_values = task
    sys.path.insert(0, str(UPSTREAM))
    from mewm_ispy2 import registered_roi32_data as single
    from mewm_ispy2.registered_three_phase_data import crop_phase
    source_raw = raw_path(raw_root, record['visit_id'])
    sources = [file_identity(source_raw), file_identity(record['metadata_source']['path']),
               file_identity(record['crop_report']['path'])]
    for identity in [record['metadata_source'], record['crop_report'], *record['phase_sources'][1:]]:
        single.verify_identity(identity)
    sources.extend(file_identity(s['path']) for s in record['phase_sources'][1:])
    name = record['visit_id'].replace(':', '_')
    output = Path(output)
    npz, npy, receipt = output/'images'/f'{name}.npz', output/'image_cache'/f'{name}.npy', output/'receipts'/f'{name}.json'
    binding = {'sources': sources, 'normalization': normalization}
    if receipt.is_file():
        previous = read_json(receipt)
        if (previous['binding'] == binding and previous['source'] == file_identity(npz)
                and previous['cached'] == file_identity(npy)):
            return previous
        raise ValueError('Prepared ROI source changed; use a fresh output directory')
    raw, valid, support = read_raw(source_raw)
    image_parts = [normalize_raw(raw, valid, normalization)]
    valid_parts, support_parts = [valid], [support]
    old_parts = [normalize_raw(raw, valid, old_normalization)] if check_values else []
    metadata, report = read_json(record['metadata_source']['path']), read_json(record['crop_report']['path'])
    for source in record['phase_sources'][1:]:
        raw, valid, support = crop_phase(source, metadata, report)
        image_parts.append(normalize_raw(raw, valid, normalization))
        valid_parts.append(valid)
        support_parts.append(support)
        if check_values:
            old_parts.append(normalize_raw(raw, valid, old_normalization))
    image, foreground, coverage = np.stack(image_parts), np.stack(valid_parts), np.stack(support_parts)
    if image.shape != (3, *SHAPE) or np.any(image[~foreground] != 0):
        raise ValueError('Prepared stacked image violates foreground/shape contract')
    # Independent replay checks upstream interpolation, phase order and masks
    # against the old values at its original scale, before applying the new one.
    replay = None
    with np.load(old_image, allow_pickle=False) as old:
        roi = old['roi']
        if check_values:
            old_values, old_support = old['image'], old['support']
            replay = float(np.abs(old_values - np.stack(old_parts)).max())
            if not np.array_equal(old_support, coverage) or not np.allclose(old_values, np.stack(old_parts), atol=2e-6, rtol=2e-6):
                raise ValueError('Raw reconstruction disagrees with original cached phase values/support')
    if roi.shape != (1, *SHAPE) or roi.dtype != np.bool_:
        raise ValueError('Invalid existing T0 ROI metadata')
    write_arrays(npz, {'image': image, 'valid': foreground, 'support': coverage, 'roi': roi}, compressed=True)
    write_arrays(npy, {'image': image}, compressed=False)
    with np.load(npz, allow_pickle=False) as archive:
        if not np.array_equal(archive['image'], np.load(npy, allow_pickle=False, mmap_mode='r')):
            raise ValueError('NPY cache differs from its normalized NPZ')
    result = {'binding': binding, 'source': file_identity(npz), 'cached': file_identity(npy),
              'foreground_values': int(foreground.sum()), 'outside_support_values': int((~coverage).sum()),
              'replay_max_abs_error': replay}
    write_json(receipt, result)
    return result


def prepare(manifest_path, phase_path, raw_root, output, *, seed=20261009, workers=6):
    started = time.monotonic()
    output = Path(output).resolve()
    manifest_path, phase_path, raw_root = Path(manifest_path).resolve(), Path(phase_path).resolve(), Path(raw_root).resolve()
    protected = [manifest_path.parent, phase_path.parent, raw_root]
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Output must be separate from all upstream assets')
    if shutil.disk_usage(ROOT).free < 50 * 1024**3:
        raise RuntimeError('Need 50 GiB free for private ROI/cache preparation')
    original, phase_inventory = read_json(manifest_path), read_json(phase_path)
    partitions = split_patients(original, seed)
    if [len(partitions[s]) for s in ('train', 'val')] != [611, 153]:
        raise ValueError('Expected the original 764 training patients split into 611/153')
    kept_keys = set(partitions['train']) | set(partitions['val'])
    records = [r for r in phase_inventory['visits'] if canonical_key(r['patient_id']) in kept_keys]
    by_visit = {(canonical_key(r['patient_id']), int(r['visit'][1:])): r for r in records}
    patients = [record_metadata(p) for p in original['patients'] if canonical_key(p['patient_key']) in kept_keys]
    if len(by_visit) != len(records) or len(records) != sum(v is not None for p in patients for v in p['visits']):
        raise ValueError('Upstream phases do not cover exactly the internal cohort')
    private_directory(output)
    for sub in ('images', 'image_cache', 'receipts'):
        private_directory(output/sub)
    binding = {'schema': 'overfit_development_preparation_v1', 'seed': seed,
               'manifest': file_identity(manifest_path), 'phase_inventory': file_identity(phase_path),
               'partitions': partitions, 'raw_root': str(raw_root)}
    if (output/'preparation.json').exists() and read_json(output/'preparation.json') != binding:
        raise ValueError('Preparation arguments changed; use a fresh output')
    write_json(output/'preparation.json', binding)
    write_json(output/'progress.json', {'stage': 'fit_train_only_normalization', 'updated_at': timestamp()})
    normalization = fit_image_normalization(records, partitions['train'], raw_root, workers)
    write_json(output/'image_normalization.json', normalization)
    print(json.dumps({'stage': 'normalization_fitted', 'fit_patients': 611,
                      'fit_visits': normalization['fit_visit_count'], 'seconds': time.monotonic()-started}), flush=True)
    train_keys = set(partitions['train'])
    tasks = []
    for patient in patients:
        key = canonical_key(patient['patient_key'])
        patient['split'] = 'train' if key in train_keys else 'val'
        for stage, visit in enumerate(patient['visits']):
            if visit is None:
                continue
            record = by_visit[(key, stage)]
            if not np.allclose(patient['geometry']['crop_affine_ras'], read_json(record['crop_report']['path'])['crop_affine_ras'], atol=0, rtol=0):
                raise ValueError('Manifest and upstream T0 crop geometry differ')
            old_image = str(Path(visit['image']).resolve())
            check = len(tasks) % 100 == 0
            tasks.append((record, str(raw_root), str(output), normalization, old_image, original['image_normalization'], check))
            visit['image'] = str(output/'images'/(record['visit_id'].replace(':', '_')+'.npz'))
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(prepare_visit, tasks):
            results.append(result)
            if len(results) % 100 == 0 or len(results) == len(tasks):
                seconds = time.monotonic() - started
                progress = {'stage': 'prepare_three_phase_images', 'completed': len(results), 'total': len(tasks),
                            'seconds': seconds, 'estimated_remaining_seconds': seconds/len(results)*(len(tasks)-len(results)),
                            'updated_at': timestamp()}
                write_json(output/'progress.json', progress)
                print(json.dumps(progress), flush=True)
    manifest = {k: record_metadata(v) for k, v in original.items() if k not in ('patients', 'provenance', 'image_normalization')}
    manifest.update(patients=patients, image_normalization=normalization,
                    provenance={'internal_development_seed': seed, 'original_manifest': file_identity(manifest_path),
                                'upstream_phase_inventory': file_identity(phase_path),
                                'patient_policy': 'Original train only; all prefixes of a patient remain in one partition',
                                'old_validation_policy': 'Original 102 development patients excluded from screening cohort',
                                'geometry_policy': 'Existing fixed T0 crop affine; no recentering or resizing',
                                'foreground_policy': 'Saved RAW DCE0 valid; other RAW phases coverage & nonzero before normalization'})
    cache = {'schema': 'registered_roi32_image_cache_v1', 'shape': [3, *SHAPE], 'dtype': 'float32',
             'phase_order': PHASES, 'image_normalization': normalization,
             'image_preprocessing': {'mode': 'preprocessed_roi32'},
             'images': [{k: r[k] for k in ('source', 'cached')} for r in results]}
    write_json(output/'manifest.json', manifest)
    write_json(output/'splits.json', partitions)
    write_json(output/'image_cache/index.json', cache)
    store = Store(output/'manifest.json', image_cache=output/'image_cache')
    write_json(output/'clinical_normalization.json', store.normalization)
    audit = store.audit(verify_pixels=True)
    tested = [r['replay_max_abs_error'] for r in results if r['replay_max_abs_error'] is not None]
    summary = {'schema': 'overfit_development_preparation_summary_v1', 'status': 'passed',
               'updated_at': timestamp(), 'seconds': time.monotonic()-started, 'seed': seed,
               'patient_counts': {s: len(store.splits[s]) for s in ('train', 'val', 'test')},
               'visit_counts': audit['visits'], 'label_counts': audit['labels'],
               'prefix_counts': {s: sum(len(store.allowed(i)) for i in store.splits[s]) for s in ('train', 'val')},
               'image_normalization': normalization, 'clinical_fit_patients': store.normalization['fit_patient_count'],
               'image_count': len(results), 'original_validation_excluded': True,
               'foreground_source': 'Raw saved DCE0 valid and upstream raw additional-phase crops',
               'replayed_old_scale_images': len(tested), 'max_old_scale_replay_error': max(tested),
               'npy_exact_verified_images': len(results), 'full_store_pixel_audit': True,
               'limitations': ['Single internal split is for screening, not independent test performance',
                               'Existing T0 ROI geometry and acquisition phase metadata are reused']}
    write_json(output/'COMPLETE.json', summary)
    write_json(output/'progress.json', {'stage': 'completed', 'updated_at': timestamp(), 'seconds': summary['seconds']})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--phase-inventory', type=Path, default=DEFAULT_PHASES)
    parser.add_argument('--raw-root', type=Path, default=DEFAULT_RAW)
    parser.add_argument('--output', type=Path, default=ROOT/'data/overfit_development_20261009')
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('workers must be 1..8')
    summary = prepare(args.manifest, args.phase_inventory, args.raw_root, args.output, seed=args.seed, workers=args.workers)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
