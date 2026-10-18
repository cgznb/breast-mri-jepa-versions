"""Actual optimizer/resume/evaluation integration for autonomous pCR training."""
from dataclasses import replace

import pytest

from mri_vla_jepa import training
from mri_vla_jepa.io import load_checkpoint
from mri_vla_jepa.train_config import RawVLATrainConfig
from test_vq_encoder_experiment import codec, setup, assert_same, single_cpu_thread


@pytest.mark.parametrize("encoder_type", ["vq_prequant", "vq_encoder"])
def test_autonomous_joint_training_resume_is_exact(tmp_path, codec, encoder_type):
    cfg, store = setup(tmp_path, codec, max_epochs=2)
    cfg = replace(cfg, model=replace(cfg.model, encoder_type=encoder_type,
                  fixed_transition_queries=True, pcr_readout="forecast",
                  history_delta=True, world_residual=True),
                  jepa_weight=.5, rollout_weight=.5, reconstruction_weight=0.,
                  variance_weight=0., variance_definition="off")
    cfg.validate()
    full = training.train(store, cfg, tmp_path / "full")
    partial = training.train(store, cfg, tmp_path / "resumed", stop_after=1)
    assert full["completed"] and not partial["completed"]
    resumed = training.train(store, cfg, tmp_path / "resumed", resume=True)
    assert resumed["completed"]
    left = load_checkpoint(tmp_path / "full/last.pt")
    right = load_checkpoint(tmp_path / "resumed/last.pt")
    for key in ("model", "optimizer", "rng", "sampler_state", "history", "epoch_history", "epoch_state"):
        assert_same(left[key], right[key])
    assert any(row["rollout"] > 0 and row["rollout_count"] > 0 for row in left["history"])
    model, saved, _ = training.load_trained(tmp_path / "full/best.pt")
    assert saved.rollout_weight == .5 and saved.model.pcr_readout == "forecast"
    rows = training._rows(model, store, split="val", profile="all_observed", batch_size=2)
    assert len(rows) == 8
    assert all(0 <= row["pcr_probability"] <= 1 for row in rows)


@pytest.mark.parametrize("value", [-1., float("nan"), float("inf"), True])
def test_rollout_weight_requires_finite_nonnegative_number(value):
    with pytest.raises(ValueError, match="rollout_weight"):
        RawVLATrainConfig(rollout_weight=value).validate()


def test_secondary_roi_metrics_keep_patient_reduction_and_matched_copy():
    from mri_vla_jepa.diagnostics import aggregate_world_errors
    rows = []
    for patient, errors, copies in ((0, [2., 4.], [4., 8.]), (1, [9.], [18.])):
        for error, copy in zip(errors, copies):
            rows.append({"patient": patient, "horizon": 1,
                "teacher_forcing_l1": 1., "autonomous_l1": 1.,
                "teacher_forcing_copy_l1": 4., "autonomous_copy_l1": 4.,
                "lesion_weighted_teacher_forcing_l1": error,
                "lesion_weighted_autonomous_l1": error,
                "lesion_weighted_teacher_forcing_copy_l1": copy,
                "lesion_weighted_autonomous_copy_l1": copy})
    report = aggregate_world_errors(rows)
    assert report["metrics"]["horizon_all"]["autonomous_relative_to_copy"] == .25
    roi = report["lesion_weighted_metrics"]["horizon_all"]
    assert roi["patients"] == 2 and roi["prefix_targets"] == 3
    assert roi["autonomous_l1"] == 6. and roi["autonomous_copy_l1"] == 12.
    assert roi["autonomous_relative_to_copy"] == .5
