"""Patient leakage, raw foreground and strict cache compatibility gates."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import stat

import numpy as np
import pytest

from response_jepa.data import CLINICAL, PHASES, Store
from response_jepa.provenance import file_identity

SPEC = spec_from_file_location('prepare_overfit_development', Path(__file__).parents[1]/'scripts/prepare_overfit_development.py')
prep = module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


def test_stratified_split_is_patient_disjoint_and_excludes_old_validation():
    manifest = {'patients': [
        {'patient_key': f'ISPY2-fixture{i:04}', 'split': 'train' if i < 764 else 'val',
         'target': {'pcr': int(i % 4 == 0)}} for i in range(866)]}
    splits = prep.split_patients(manifest)
    assert [len(splits[s]) for s in ('train', 'val')] == [611, 153]
    assert splits == prep.split_patients(manifest)
    assert not set(splits['train']) & set(splits['val'])
    assert set(splits['train']) | set(splits['val']) == {f'fixture{i:04}' for i in range(764)}
    positive = lambda keys: sum(int(k[-4:]) % 4 == 0 for k in keys)
    assert abs(positive(splits['train'])/611 - positive(splits['val'])/153) < .01


def save_raw(root, visit, values, valid=None):
    path = prep.raw_path(root, visit)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(values, dtype=np.float32).reshape(2, 2, 2)
    support = np.ones_like(raw, dtype=bool)
    if valid is None:
        valid = raw != 0
    np.savez(path, image=raw, valid=valid, coverage=support)
    return path


def test_intensity_fit_uses_unique_raw_training_foreground_only(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, 'SHAPE', (2, 2, 2))
    records = [{'patient_id': 'ISPY2-train', 'visit_id': 'ISPY2-train:T0'},
               {'patient_id': 'ISPY2-train', 'visit_id': 'ISPY2-train:T1'},
               {'patient_id': 'ISPY2-val', 'visit_id': 'ISPY2-val:T0'}]
    save_raw(tmp_path, records[0]['visit_id'], [0, 0, 1, 1, 1, 1, 1, 1])
    save_raw(tmp_path, records[1]['visit_id'], [0, 3, 3, 3, 3, 3, 3, 3])
    save_raw(tmp_path, records[2]['visit_id'], [1000]*8)
    norm = prep.fit_image_normalization(records, ['train'], tmp_path, workers=2)
    expected = np.asarray([1.]*6+[3.]*7)
    assert norm['mean'] == pytest.approx(expected.mean())
    assert norm['std'] == pytest.approx(expected.std())
    assert norm['count'] == 13 and norm['fit_visit_count'] == 2
    assert norm['fit_patient_count'] == 1
    save_raw(tmp_path, records[2]['visit_id'], [9000]*8)
    assert norm == prep.fit_image_normalization(records, ['train'], tmp_path, workers=2)
    with pytest.raises(ValueError, match='unique'):
        prep.fit_image_normalization(records+[records[0]], ['train'], tmp_path)


def test_foreground_is_preserved_even_when_normalization_produces_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, 'SHAPE', (2, 2, 2))
    path = save_raw(tmp_path, 'fixture:T0', [0, 1, 2, 3, 4, 5, 6, 7])
    raw, valid, support = prep.read_raw(path)
    image = prep.normalize_raw(raw, valid, {'mean': 1., 'std': 2.})
    assert image.flat[0] == 0 and not valid.flat[0]
    assert image.flat[1] == 0 and valid.flat[1]
    assert np.all(image[~valid] == 0)
    # Acquisition support includes raw zero background; it cannot replace valid.
    assert support.flat[0] and not valid.flat[0]
    save_raw(tmp_path, 'bad:T0', [0, 1, 2, 3, 4, 5, 6, 7], valid=support)
    with pytest.raises(ValueError, match='foreground'):
        prep.read_raw(prep.raw_path(tmp_path, 'bad:T0'))


def test_private_manifest_cache_loads_strict_store_and_fits_train_clinical_only(tmp_path):
    output = prep.private_directory(tmp_path/'private')
    prep.private_directory(output/'image_cache')
    norm = {'mean': 2., 'std': 3., 'fit_split': 'train', 'scope': 'fixture_raw_train_foreground',
            'background_policy': 'preserve_zero', 'already_normalized': True,
            'normalization_policy': 'fixture_raw_dce0_shared'}
    patients, entries = [], []
    image = np.zeros((3, 32, 128, 128), dtype=np.float32)
    image[:, 0, 0, 0] = .5
    for i, (split, age) in enumerate([('train', 10.), ('train', 30.), ('val', 900.)]):
        source, cached = output/f'fixture{i}.npz', output/'image_cache'/f'fixture{i}.npy'
        prep.write_arrays(source, {'image': image}, compressed=True)
        prep.write_arrays(cached, {'image': image}, compressed=False)
        grid = f'fixture:{i}'
        patients.append({'patient_key': f'fixture{i}', 'split': split,
                         'geometry': {'source_only': True, 'source_stage': 0, 'roi_source': 'T0',
                                      'orientation': 'RAS', 'resampling': 'source_grid', 'grid_id': grid},
                         'clinical': {'values': [age, 0, 1, 0], 'known_at': [0]*4},
                         'arm': {'label': None, 'known_at_stage': None, 'semantics': 'assigned_arm_scenario'},
                         'target': {'pcr': i % 2}, 'visits': [
                             {'stage': 0, 'available_at': 0, 'grid_id': grid, 'image': str(source)}, None, None, None]})
        entries.append({'source': file_identity(source), 'cached': file_identity(cached)})
    manifest = {'schema': 'responsewm_raw_mri_v1', 'time_basis': 'stage_index', 'canonical_stages': [0, 1, 2, 3],
                'phase_order': PHASES, 'clinical_features': CLINICAL, 'image_preprocessing': {'mode': 'preprocessed_roi32'},
                'image_normalization': norm, 'patients': patients}
    index = {'schema': 'registered_roi32_image_cache_v1', 'shape': list(image.shape), 'dtype': 'float32',
             'phase_order': PHASES, 'image_normalization': norm,
             'image_preprocessing': {'mode': 'preprocessed_roi32'}, 'images': entries}
    prep.write_json(output/'manifest.json', manifest)
    prep.write_json(output/'image_cache/index.json', index)
    store = Store(output/'manifest.json', image_cache=output/'image_cache')
    assert store.normalization['mean'][0] == 20.
    assert store.normalization['std'][0] == 10.
    assert store.normalization['fit_patient_count'] == 2
    assert np.array_equal(store.read_image(patients[2]['visits'][0]).numpy(), image)
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for path in output.rglob('*'):
        if path.is_file():
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
    index['image_normalization']['mean'] = 999.
    prep.write_json(output/'image_cache/index.json', index)
    with pytest.raises(ValueError, match='metadata'):
        Store(output/'manifest.json', image_cache=output/'image_cache')
