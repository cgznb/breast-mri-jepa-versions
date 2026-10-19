"""F1 selection must weight patients equally and survive checkpoint recovery."""
from dataclasses import replace

import numpy as np
import pytest

from mri_vla_jepa import training
from mri_vla_jepa.io import load_checkpoint
from mri_vla_jepa.metrics import classification_metrics, patient_bootstrap
from mri_vla_jepa.train_config import from_dict
from test_epoch_training import TinyModel, assert_same, make_setup


def row(patient, stage, label, probability):
    return {"patient_key": patient, "landmark": stage, "pcr": label,
            "pcr_probability": probability}


def test_patient_equal_f1_is_weighted_confusion_not_mean_patient_f1():
    # Patient A: TP=.5, FN=.5; B: FP=1; C: FN=1, each with unit total weight.
    rows = [row("A", 0, 1, .5), row("A", 1, 1, .1), row("B", 0, 0, .8)]
    rows.extend(row("C", stage, 1, .1) for stage in range(4))
    result = training._scores(rows, "all_observed", bootstrap=30, seed=2, selection_metric="f1")
    metrics = result["metrics"]
    assert result["selection_metric"] == "patient_equal_prefix_f1"
    assert result["selection_mode"] == "max"
    assert result["selection_score"] == pytest.approx(1 / 3.5)
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (.5, 1., 1.5)
    assert metrics["patients"] == 3 and metrics["prefixes"] == 7
    assert result["per_landmark"]["T0"]["f1"] == .5
    assert metrics["bootstrap"]["f1"]["valid_replicates"] == 30
    expected_nll = np.mean([np.mean([-np.log(.5), -np.log(.1)]), -np.log(.2), -np.log(.1)])
    assert metrics["nll"] == pytest.approx(expected_nll)


def test_f1_threshold_zero_division_and_bootstrap():
    assert classification_metrics([1, 0], [.5, .49])["f1"] == 1.
    assert classification_metrics([1, 0], [.5, .49], f1_threshold=.6)["f1"] == 0.
    assert classification_metrics([0, 0], [.1, .2])["f1"] == 0.
    result = patient_bootstrap([0, 0], [.1, .2], ["A", "B"], repetitions=10)
    assert result["f1"] == {"lower": 0., "upper": 0., "valid_replicates": 10}
    with pytest.raises(ValueError, match="inconsistent"):
        training._patient_equal_score([row("A", 0, 1, .6), row("A", 1, 0, .4)])


@pytest.mark.parametrize("changes", [
    {"selection_metric": "accuracy"}, {"f1_threshold": True}, {"f1_threshold": -1},
    {"f1_threshold": float("nan")}, {"prediction_training": "parallel"},
    {"prediction_training": "teacher_forcing", "rollout_weight": .5},
])
def test_selection_and_prediction_config_validation(changes):
    with pytest.raises(ValueError):
        from_dict(changes)


def mock_f1(monkeypatch, values):
    iterator = iter(values)

    def score(rows, profile, **kwargs):
        assert kwargs["selection_metric"] == "f1"
        f1 = next(iterator)
        return {"selection_score": f1, "selection_metric": "t0_direct_f1",
                "metrics": {"f1": f1, "nll": .8 - f1 / 10}}

    monkeypatch.setattr(training, "_scores", score)


def test_zero_first_f1_saved_ties_stop_after_fifty_bad_epochs(tmp_path, monkeypatch):
    cfg, _, store = make_setup(tmp_path, n_train=2)
    cfg = replace(cfg, selection_metric="f1", max_epochs=200, early_stopping_patience=50)
    monkeypatch.setattr(training, "RawVLAJEPA", TinyModel)
    mock_f1(monkeypatch, [0.] * 51)
    report = training.train(store, cfg, tmp_path / "run")
    assert report["completed_epochs"] == 51 and report["bad_epochs"] == 50
    assert report["best_selection_score"] == report["best_validation_f1"] == 0.
    assert report["best_validation_nll"] == .8
    assert report["best_epoch"] == report["best_step"] == 1
    best = load_checkpoint(tmp_path / "run/best.pt")
    assert best["epoch_state"]["completed_epochs"] == 1
    assert best["selection_mode"] == "max" and best["selection_metric"] == "f1"


def test_f1_improvement_scheduler_and_resume_are_exact(tmp_path, monkeypatch):
    cfg, _, store = make_setup(tmp_path, n_train=2)
    cfg = replace(cfg, selection_metric="f1", max_epochs=5, lr_scheduler="plateau",
                  plateau_patience=0, early_stopping_patience=2)
    monkeypatch.setattr(training, "RawVLAJEPA", TinyModel)
    values = [0., .1, .2, .2, .2]
    mock_f1(monkeypatch, values)
    training.train(store, cfg, tmp_path / "full")
    mock_f1(monkeypatch, values)
    training.train(store, cfg, tmp_path / "resumed", stop_after=2)
    training.train(store, cfg, tmp_path / "resumed", resume=True)
    left = load_checkpoint(tmp_path / "full/last.pt")
    right = load_checkpoint(tmp_path / "resumed/last.pt")
    for key in ("model", "optimizer", "scheduler_state", "rng", "sampler_state", "history",
                "epoch_history", "epoch_state", "best_selection_score", "best_validation_f1", "best_epoch", "best_step"):
        assert_same(left[key], right[key])
    assert left["scheduler_state"]["mode"] == "max"
    assert left["scheduler_state"]["best"] == .2
    assert left["epoch_history"][-1]["next_lr"] == cfg.lr / 4
    best = load_checkpoint(tmp_path / "full/best.pt")
    assert best["epoch_state"]["completed_epochs"] == 3
    assert best["best_epoch"] == 3


def test_checkpoint_defaults_preserve_legacy_nll_reading(tmp_path, monkeypatch):
    cfg, _, store = make_setup(tmp_path, n_train=2)
    cfg.max_epochs = 1
    monkeypatch.setattr(training, "RawVLAJEPA", TinyModel)
    training.train(store, cfg, tmp_path / "run")
    saved = load_checkpoint(tmp_path / "run/last.pt")
    for name in ("prediction_training", "selection_metric", "f1_threshold"):
        saved["config"].pop(name)
    monkeypatch.setattr(training, "load_checkpoint", lambda path: saved)
    _, restored, _ = training.load_trained("legacy.pt")
    assert restored.prediction_training == "hybrid"
    assert restored.selection_metric == "nll" and restored.f1_threshold == .5


def test_teacher_forcing_classification_loads_real_predecessors_without_auxiliary_loss(tmp_path, monkeypatch):
    cfg, _, store = make_setup(tmp_path, n_train=2)
    cfg = replace(cfg, prediction_training="teacher_forcing", max_epochs=1,
                  model=replace(cfg.model, fixed_transition_queries=True, pcr_readout="forecast"),
                  jepa_weight=0., rollout_weight=0., flow_weight=0.,
                  reconstruction_weight=0., variance_weight=0.)
    seen = []

    class SupervisionModel(TinyModel):
        def compute_loss(self, inp, sup, **kwargs):
            assert kwargs["prediction_training"] == "teacher_forcing"
            assert sup.future_mask[:, 1:].all()
            assert sup.future[:, 1:].abs().sum() > 0
            seen.append(True)
            return super().compute_loss(inp, sup, **kwargs)

    monkeypatch.setattr(training, "RawVLAJEPA", SupervisionModel)
    training.train(store, cfg, tmp_path / "run")
    assert seen
