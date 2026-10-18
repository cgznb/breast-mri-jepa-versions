"""Complete dynamic-prefix coverage, patient independence, and exact resume."""
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
import random

import numpy as np
import pytest
import torch

from response_jepa.data import Store, planned_tasks
from response_jepa.engine import evaluate, load_checkpoint, make_loader, train
from response_jepa.model import ResponseJEPA
from response_jepa.synthetic import small_config


def all_prefix_config(phase="joint"):
    cfg = small_config(phase)
    cfg.train.prefix_sampling = "all_prefixes"
    return cfg.validate()


def synthetic_store(cohort, cfg):
    return Store(cohort, cfg.train.image_shape, allow_synthetic=True)


@pytest.mark.parametrize("phase", ["B", "C", "joint"])
def test_all_prefixes_cover_each_eligible_patient_landmark_once(cohort, phase):
    cfg = all_prefix_config(phase)
    store = synthetic_store(cohort, cfg)
    tasks = planned_tasks(store, cfg, "train", 0)
    assert len(tasks) == len(set(tasks))
    assert {i for i, _ in tasks} == set(store.splits["train"])
    assert set(tasks) == {(i, j) for i in range(8) for j in range(3 if phase == "B" else 4)
                          if not (i == 6 and j == 1)}
    assert len(tasks) == (23 if phase == "B" else 31)


def test_all_prefix_epoch_shuffle_is_private_complete_and_reproducible(cohort):
    cfg = all_prefix_config()
    store = synthetic_store(cohort, cfg)
    global_rng = random.getstate()
    epoch0 = planned_tasks(store, cfg, "train", 0)
    assert epoch0 == planned_tasks(store, cfg, "train", 0)
    epoch1 = planned_tasks(store, cfg, "train", 1)
    assert epoch0 != epoch1 and set(epoch0) == set(epoch1)
    different_seed = deepcopy(cfg)
    different_seed.train.seed += 1
    assert epoch0 != planned_tasks(store, different_seed, "train", 0)
    assert random.getstate() == global_rng


def test_all_prefixes_keep_missing_slots_and_respect_delayed_availability(cohort):
    manifest = json.loads(cohort.read_text())
    patient = manifest["patients"][0]
    patient["visits"][1]["available_at"] = 3
    patient["visits"][3] = None
    cohort.write_text(json.dumps(manifest), encoding="utf-8")
    cfg = all_prefix_config()
    store = synthetic_store(cohort, cfg)
    tasks = planned_tasks(store, cfg, "train", 0)
    assert sorted(j for i, j in tasks if i == 0) == [0, 2, 3]
    assert sorted(j for i, j in tasks if i == 6) == [0, 2, 3]
    early, _, _ = store.sample(0, 2, load_images=False)
    late, _, _ = store.sample(0, 3, load_images=False)
    assert early.observed.tolist() == [True, False, True, False]
    assert late.observed.tolist() == [True, True, True, False]
    assert store.observed_stage(0, 2) == store.observed_stage(0, 3) == 2


def test_world_only_all_prefixes_exclude_patients_and_prefixes_without_future(cohort):
    manifest = json.loads(cohort.read_text())
    manifest["patients"][0]["visits"][3] = None
    manifest["patients"][1]["visits"][1:] = [None, None, None]
    cohort.write_text(json.dumps(manifest), encoding="utf-8")
    cfg = all_prefix_config("B")
    store = synthetic_store(cohort, cfg)
    tasks = planned_tasks(store, cfg, "train", 0)
    assert sorted(j for i, j in tasks if i == 0) == [0, 1]
    assert all(i != 1 for i, _ in tasks)
    assert all(j < 3 for _, j in tasks)
    cfg.train.phase = "joint"
    cfg.train.freeze_encoder = False
    cfg.validate()
    joint = planned_tasks(store, cfg, "train", 0)
    assert (0, 2) in joint and (1, 0) in joint


def test_all_prefix_validation_unchanged_and_selection_remains_patient_equal(cohort):
    manifest = json.loads(cohort.read_text())
    manifest["patients"][8]["visits"][1] = None
    manifest["patients"][9]["visits"][3] = None
    cohort.write_text(json.dumps(manifest), encoding="utf-8")
    cfg = all_prefix_config()
    store = synthetic_store(cohort, cfg)
    all_validation = planned_tasks(store, cfg, "val", 0)
    balanced = deepcopy(cfg)
    balanced.train.prefix_sampling = "balanced_cycle"
    assert all_validation == planned_tasks(store, balanced, "val", 0)
    assert all_validation == planned_tasks(store, cfg, "val", 9)
    assert len(all_validation) == 14
    model = ResponseJEPA(cfg.model)
    model.configure_trainable("joint", False)
    model.world_trained = True
    metrics, rows, _ = evaluate(model, store, cfg, "val")
    patient_losses = defaultdict(list)
    for row in rows:
        probability = row["fused"] if row["label"] else 1 - row["fused"]
        patient_losses[row["patient_key"]].append(-math.log(probability))
    manual_nll = sum(sum(v) / len(v) for v in patient_losses.values()) / len(patient_losses)
    assert metrics["selection_name"] == "patient_equal_pcr_nll"
    assert metrics["selection"] == pytest.approx(manual_nll, abs=1e-12)
    assert metrics["classification"]["fused"]["patient_equal"]["patients"] == 4
    assert metrics["classification"]["fused"]["patient_equal"]["prefixes"] == 14


@pytest.mark.parametrize("phase, profile", [
    ("A", "dynamic"), ("B", "t0"), ("C", "t0"), ("joint", "t0"),
    ("C", "single_t0"), ("C", "single_t3"),
])
def test_all_prefixes_reject_inapplicable_phase_or_profile(phase, profile):
    cfg = small_config(phase)
    cfg.train.profile = profile
    if profile.startswith("single_"):
        cfg.model.readout = "direct"
    cfg.train.prefix_sampling = "all_prefixes"
    with pytest.raises(ValueError, match="All-prefix sampling"):
        cfg.validate()


@pytest.mark.parametrize("sampling, epoch, expected", [
    ("balanced_cycle", 0, [(3, 0), (7, 2), (4, 3), (6, 2), (1, 0), (2, 1), (5, 3), (0, 0)]),
    ("balanced_cycle", 1, [(6, 3), (7, 3), (4, 0), (3, 1), (1, 1), (2, 2), (0, 1), (5, 0)]),
    ("random", 0, [(3, 1), (7, 3), (4, 1), (6, 2), (1, 0), (2, 0), (5, 2), (0, 3)]),
    ("random", 1, [(6, 2), (7, 1), (4, 2), (3, 0), (1, 2), (2, 3), (0, 2), (5, 0)]),
])
def test_existing_sampling_preserves_previous_task_order(cohort, sampling, epoch, expected):
    cfg = small_config("joint")
    cfg.train.prefix_sampling = sampling
    store = synthetic_store(cohort, cfg)
    assert planned_tasks(store, cfg, "train", epoch) == expected
    cfg.train.profile = "t0"
    assert planned_tasks(store, cfg, "train", epoch) == [(i, 0) for i, _ in expected]
    assert planned_tasks(store, cfg, "val", epoch) == [(i, 0) for i in range(8, 12)]


def test_joint_all_prefix_batches_keep_same_anchor_patients_distinct_and_all_samples(cohort):
    manifest = json.loads(cohort.read_text())
    for patient in manifest["patients"][:2]:
        patient["visits"][1]["available_at"] = 3
        patient["visits"][3] = None
    cohort.write_text(json.dumps(manifest), encoding="utf-8")
    cfg = all_prefix_config()
    cfg.train.batch_size = 3
    store = synthetic_store(cohort, cfg)
    tasks = planned_tasks(store, cfg, "train", 0)
    assert [(store.observed_stage(0, j), j) for j in [2, 3]] == [(2, 2), (2, 3)]
    seen = []
    for batch in make_loader(store, cfg, tasks, training=True, world=True):
        assert batch.inputs.anchor().unique().numel() == 1
        assert len(batch.keys) == len(set(batch.keys))
        assert len(batch.keys) >= 2
        seen.extend(zip(batch.keys, batch.inputs.landmark.tolist()))
    expected = [(store.patients[i]["patient_key"], j) for i, j in tasks]
    assert Counter(seen) == Counter(expected)
    assert len(seen) == len(set(seen))


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for av, bv in zip(a, b):
            assert_nested_equal(av, bv)
    else:
        assert a == b


def test_exact_cpu_all_prefix_joint_resume_with_complete_epochs(cohort, tmp_path):
    cfg = all_prefix_config()
    cfg.train.batch_size = 3
    cfg.train.accumulation = 2
    cfg.train.epochs = 2
    cfg.train.patience = 10
    cfg.train.warmup_epochs = 1
    cfg.train.augment = True
    cfg.model.dropout = 0.1
    train(cfg, cohort, tmp_path / "full")
    train(cfg, cohort, tmp_path / "resumed", stop_after_epoch=1)
    train(cfg, cohort, tmp_path / "resumed", resume=True)
    full = load_checkpoint(tmp_path / "full" / "last.pt")
    resumed = load_checkpoint(tmp_path / "resumed" / "last.pt")
    assert full["epoch"] == resumed["epoch"] == 2
    assert full["updates"] == resumed["updates"]
    assert full["best_score"] == resumed["best_score"]
    assert full["bad_epochs"] == resumed["bad_epochs"]
    for key in ("model", "optimizer", "rng"):
        assert_nested_equal(full[key], resumed[key])
    for uninterrupted, restarted in zip(full["history"], resumed["history"]):
        assert uninterrupted["training_patients_or_prefixes"] == restarted["training_patients_or_prefixes"] == 31
        assert_nested_equal(uninterrupted["train"], restarted["train"])
        assert_nested_equal(uninterrupted["validation"]["classification"], restarted["validation"]["classification"])
        assert uninterrupted["validation"]["selection"] == restarted["validation"]["selection"]
