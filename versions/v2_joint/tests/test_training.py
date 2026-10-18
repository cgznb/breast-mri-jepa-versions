from copy import deepcopy
from dataclasses import replace
import torch
import pytest
from response_jepa.synthetic import small_config
from response_jepa.engine import train, load_checkpoint, load_for_inference, evaluate


def test_frozen_random_encoder_rejected(cohort, tmp_path):
    c = small_config("C")
    c.loss.tf = c.loss.rollout = 0
    c.model.readout = "direct"
    c.train.world_eval = False
    with pytest.raises(ValueError, match="frozen random"):
        train(c, cohort, tmp_path/"bad")


def test_exact_cpu_epoch_boundary_resume(cohort, tmp_path):
    cfg = small_config("C")
    cfg.train.freeze_encoder = False
    cfg.train.world_eval = False
    cfg.loss.tf = cfg.loss.rollout = 0
    cfg.model.readout = "direct"
    cfg.model.dropout = .1
    cfg.train.epochs = 3
    cfg.train.warmup_epochs = 1
    cfg.train.patience = 10
    cfg.train.accumulation = 2
    cfg.train.batch_size = 3  # exercises partial accumulation and final short microbatch
    train(cfg, cohort, tmp_path/"full")
    train(cfg, cohort, tmp_path/"resumed", stop_after_epoch=1)
    train(cfg, cohort, tmp_path/"resumed", resume=True)
    a, b = load_checkpoint(tmp_path/"full/last.pt"), load_checkpoint(tmp_path/"resumed/last.pt")
    assert a["epoch"] == b["epoch"] == 3
    assert a["updates"] == b["updates"]
    for k in a["model"]:
        torch.testing.assert_close(a["model"][k], b["model"][k], atol=0, rtol=0)
    assert a["best_score"] == b["best_score"]


def test_staged_training_preserves_teacher_coordinates(cohort, tmp_path):
    initial = None
    states = {}
    for phase in ("A", "B", "C"):
        cfg = small_config(phase)
        train(cfg, cohort, tmp_path/phase, init=initial)
        initial = tmp_path/phase/"best.pt"
        states[phase] = load_checkpoint(initial)
    assert states["A"]["target_coordinate_origin"] == states["B"]["target_coordinate_origin"]
    assert states["B"]["target_coordinate_origin"] == states["C"]["target_coordinate_origin"]
    for key, teacher in states["A"]["model"].items():
        if key.startswith("target_encoder."):
            torch.testing.assert_close(teacher, states["B"]["model"][key], atol=0, rtol=0)
            torch.testing.assert_close(teacher, states["C"]["model"][key], atol=0, rtol=0)
    m, s, cfg, _ = load_for_inference(initial, cohort, allow_synthetic=True)
    metrics, rows, world = evaluate(m, s, cfg, "test")
    assert metrics["classification"]["fused"]["patient_equal"]["patients"] == 4
    assert len(rows) == 16
    assert len(world) > 0
