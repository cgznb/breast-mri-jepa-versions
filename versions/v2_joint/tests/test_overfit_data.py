"""Patient contribution uses the complete legal population, even after slicing."""
from collections import defaultdict
from dataclasses import replace
import json

import pytest
import torch

from response_jepa.data import Store, PrefixDataset, collate_prefix, planned_tasks
from response_jepa.synthetic import small_config


def setup_dataset(cohort, *, phase="joint", profile="dynamic"):
    cfg = small_config(phase)
    cfg.train.profile = profile
    if profile == "dynamic":
        cfg.train.prefix_sampling = "all_prefixes"
    store = Store(cohort, cfg.train.image_shape, allow_synthetic=True)
    tasks = planned_tasks(store, cfg, "train", 0)
    dataset = PrefixDataset(store, tasks, sample_weighting="patient_equal",
                            profile=profile, phase=phase, load_images=False)
    return cfg, store, tasks, dataset


def test_patient_total_weight_equal_and_population_mean_one(cohort):
    _, _, tasks, dataset = setup_dataset(cohort)
    totals = defaultdict(float)
    for (patient, _), weight in zip(tasks, dataset.sample_weights):
        totals[patient] += weight
    assert len(tasks) == 31 and len(totals) == 8
    assert list(totals.values()) == pytest.approx([31 / 8] * 8)
    assert sum(dataset.sample_weights) == pytest.approx(len(tasks))
    batch = collate_prefix([dataset[0], dataset[1]])
    assert batch.sample_weight.dtype == torch.float32
    assert batch.sample_weight.tolist() == pytest.approx(dataset.sample_weights[:2])
    assert batch.keys == tuple(dataset[i][2] for i in range(2))


def test_weighting_is_fixed_for_reordered_epoch_and_partial_loader(cohort):
    cfg, store, tasks, dataset = setup_dataset(cohort)
    expected = dict(zip(tasks, dataset.sample_weights))
    for selected in (planned_tasks(store, cfg, "train", 4), tasks[3:9], tasks[-1:]):
        subset = PrefixDataset(store, selected, sample_weighting="patient_equal", load_images=False)
        assert subset.sample_weights == tuple(expected[task] for task in selected)
    # A patient with fewer legal prefixes must retain its larger row weight,
    # even if only one of its rows is in the resumed suffix.
    subset = PrefixDataset(store, [(6, 0)], sample_weighting="patient_equal", load_images=False)
    assert subset.sample_weights == pytest.approx([31 / (8 * 3)])


@pytest.mark.parametrize("profile", ["t0", "single_t0", "single_t3"])
def test_fixed_profile_weights_are_exactly_one(cohort, profile):
    _, _, _, dataset = setup_dataset(cohort, profile=profile)
    assert dataset.sample_weights == (1.0,) * len(dataset)


def test_weights_use_legal_availability_landmarks_not_scan_count(cohort):
    manifest = json.loads(cohort.read_text())
    manifest["patients"][0]["visits"][1]["available_at"] = 3
    cohort.write_text(json.dumps(manifest), encoding="utf-8")
    _, _, tasks, dataset = setup_dataset(cohort)
    assert len(tasks) == 30
    weights = dict(zip(tasks, dataset.sample_weights))
    assert (0, 1) not in weights
    assert weights[(0, 3)] == pytest.approx(30 / (8 * 3))
    assert weights[(1, 3)] == pytest.approx(30 / (8 * 4))


def test_world_only_weights_exclude_no_future_patients_and_prefixes(cohort):
    manifest = json.loads(cohort.read_text())
    manifest["patients"][0]["visits"][3] = None
    manifest["patients"][1]["visits"][1:] = [None, None, None]
    cohort.write_text(json.dumps(manifest), encoding="utf-8")
    _, _, tasks, dataset = setup_dataset(cohort, phase="B")
    assert len(tasks) == 19
    assert all(i != 1 and j < 3 for i, j in tasks)
    totals = defaultdict(float)
    for (patient, _), weight in zip(tasks, dataset.sample_weights):
        totals[patient] += weight
    assert len(totals) == 7
    assert list(totals.values()) == pytest.approx([19 / 7] * 7)


def test_default_samples_and_handmade_batches_remain_compatible(cohort, batch):
    _, store, tasks, _ = setup_dataset(cohort)
    dataset = PrefixDataset(store, tasks, load_images=False)
    assert dataset.sample_weights is None
    assert len(dataset[0]) == 3
    result = collate_prefix([dataset[0], dataset[1]])
    assert result.sample_weight is None
    assert batch.sample_weight is None
    assert result.to("cpu", non_blocking=True).sample_weight is None


def test_batch_to_and_pin_memory_preserve_optional_weights(batch, monkeypatch):
    weighted = replace(batch, sample_weight=torch.tensor([0.75, 1.25]))
    moved = weighted.to("meta", non_blocking=True)
    assert moved.inputs.images.device.type == "meta"
    assert moved.sample_weight.device.type == "meta"
    assert moved.keys == batch.keys
    # Avoid requiring a GPU/pinned allocator in CPU regression environments.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self.clone())
    assert batch.pin_memory().sample_weight is None
    pinned = weighted.pin_memory()
    torch.testing.assert_close(pinned.sample_weight, weighted.sample_weight)
    assert pinned.sample_weight is not weighted.sample_weight
    assert pinned.keys == weighted.keys


@pytest.mark.parametrize("tasks, match", [
    ([(0, 0), (0, 0)], "duplicate"),
    ([(0, 0), (8, 0)], "one split"),
    ([(6, 1)], "legal prefix"),
    ([], "nonempty"),
])
def test_patient_equal_rejects_bad_reference_rows(cohort, tasks, match):
    _, store, _, _ = setup_dataset(cohort)
    with pytest.raises(ValueError, match=match):
        PrefixDataset(store, tasks, sample_weighting="patient_equal", load_images=False)


def test_collate_rejects_mixed_or_invalid_weights(cohort):
    _, store, tasks, dataset = setup_dataset(cohort)
    plain = store.sample(*tasks[0], load_images=False)
    with pytest.raises(ValueError, match="consistently"):
        collate_prefix([plain, dataset[0]])
    for weight in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            collate_prefix([(*plain, weight)])
