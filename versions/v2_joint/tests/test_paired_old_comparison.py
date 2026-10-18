"""Patient correspondence and tied bootstrap metric regression checks."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

SCRIPTS = Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("paired_comparison", SCRIPTS/"compare_joint_old_predictions.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_tied_and_repeated_patient_predictions_match_standard_metrics():
    rng = np.random.default_rng(7)
    y, p = rng.integers(0, 2, 102), rng.integers(0, 5, 102)/4
    for _ in range(25):
        draw = rng.integers(0, 102, 102)
        actual = MODULE.metric_vector(y[draw], p[draw])
        assert actual["auroc"] == pytest.approx(roc_auc_score(y[draw], p[draw]), abs=1e-14)
        assert actual["auprc"] == pytest.approx(average_precision_score(y[draw], p[draw]), abs=1e-14)


def test_unmatched_and_duplicate_patient_prefix_are_rejected():
    manifest = {"patients": [{"patient_key": "ISPY2-1", "split": "val", "target": {"pcr": 1},
                              "visits": [{"available_at": 0}, {"available_at": 1}]}]}
    new = [{"patient_key": "1", "landmark": s, "label": 1, "fused": .7} for s in (0, 1)]
    old = [{"patient_key": "ISPY2-1", "landmark": s, "pcr": 1, "pcr_probability": .6} for s in (0, 1)]
    assert len(MODULE.paired_rows(new, old, manifest, "dynamic")[0]) == 2
    with pytest.raises(ValueError, match="cohort"):
        MODULE.paired_rows(new, old[:1], manifest, "dynamic")
    with pytest.raises(ValueError, match="cohort"):
        MODULE.paired_rows(new + new[:1], old, manifest, "dynamic")
    old[1]["pcr"] = 0
    with pytest.raises(ValueError, match="labels"):
        MODULE.paired_rows(new, old, manifest, "dynamic")


def test_paired_identical_models_have_exact_zero_interval():
    y, p = np.array([0, 1, 0, 1]), np.array([.2, .7, .4, .6])
    draws = np.tile(np.arange(4), (100, 1))
    for metric in MODULE.stage_deltas(y, p, p, draws):
        assert metric["delta_joint_minus_old"] == 0
        assert metric["lower_95"] == metric["upper_95"] == 0
