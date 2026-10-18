from dataclasses import replace
import json
import numpy as np

import pytest
import torch

from response_jepa.data import Store
from response_jepa.engine import load_checkpoint, train
from response_jepa.model import ResponseJEPA
from response_jepa.synthetic import small_config
from response_jepa.update_engine import _probe_tasks, schedule_updates


def update_config():
    cfg = small_config("C")
    cfg.train.phase = "joint"
    cfg.train.profile = "dynamic"
    cfg.train.freeze_encoder = False
    cfg.train.prefix_sampling = "all_prefixes"
    cfg.train.sample_weighting = "patient_equal"
    cfg.train.reference_batch_size = 3
    cfg.train.batch_size = 3
    cfg.train.accumulation = 1
    cfg.train.max_updates = 7
    cfg.train.warmup_updates = 1
    cfg.train.eval_every_updates = 2
    cfg.train.patience_updates = 20
    cfg.train.train_eval_patients = 3
    cfg.train.augment = False
    cfg.loss.align = 0.
    cfg.loss.consistency = .1
    cfg.loss.consistency_warmup_updates = 3
    cfg.model.forecast_detach = True
    cfg.model.dropout = .1
    return cfg.validate()


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert a == b


def test_exact_cpu_mid_epoch_resume_with_weighting_and_consistency(cohort, tmp_path):
    cfg = update_config()
    train(cfg, cohort, tmp_path / "full")
    train(cfg, cohort, tmp_path / "resumed", stop_after_updates=3)
    intermediate = load_checkpoint(tmp_path / "resumed/last.pt")
    assert intermediate["updates"] == 3
    assert intermediate["cursor"]["batch"] == 3
    assert intermediate["interval_updates"] == 1
    train(cfg, cohort, tmp_path / "resumed", resume=True)
    a, b = [load_checkpoint(tmp_path / folder / "last.pt") for folder in ("full", "resumed")]
    for key in ("model", "optimizer", "rng", "updates", "best_update", "best_score", "cursor", "samples_seen"):
        assert_equal(a[key], b[key])
    for x, y in zip(a["history"], b["history"]):
        for key in ("train", "updates", "samples_seen", "best_score", "bad_updates"):
            assert_equal(x[key], y[key])
        assert x["validation"]["selection"] == y["validation"]["selection"]
    assert a["updates"] == cfg.train.max_updates
    assert [row["updates"] for row in a["history"]] == [0, 2, 4, 6, 7]


def test_update_boundary_resume_across_epoch(cohort, tmp_path):
    cfg = update_config()
    cfg.train.profile = "t0"
    cfg.train.prefix_sampling = "balanced_cycle"
    cfg.train.max_updates = 8
    train(cfg, cohort, tmp_path / "full")
    train(cfg, cohort, tmp_path / "resumed", stop_after_updates=3)
    train(cfg, cohort, tmp_path / "resumed", resume=True)
    a, b = [load_checkpoint(tmp_path / folder / "last.pt") for folder in ("full", "resumed")]
    for key in ("model", "optimizer", "rng", "cursor", "best_update", "best_score"):
        assert_equal(a[key], b[key])
    assert a["cursor"]["epoch"] >= 2


def test_teacher_updates_once_and_patience_uses_updates(cohort, tmp_path, monkeypatch):
    import response_jepa.update_engine as runner
    cfg = update_config()
    cfg.train.max_updates = 20
    cfg.train.patience_updates = 5
    calls = []
    original = ResponseJEPA.update_teacher
    monkeypatch.setattr(ResponseJEPA, "update_teacher", lambda self, decay: (calls.append(decay), original(self, decay))[1])
    monkeypatch.setattr(runner, "evaluate", lambda *args, **kwargs: ({"selection": .6}, [], []))
    result = train(cfg, cohort, tmp_path / "patience")
    assert result["updates"] == len(calls) == 5
    assert result["best_update"] == 0
    assert result["stop_reason"] == "early_stop"
    assert json.loads((tmp_path / "patience/completion.json").read_text())["state"] == "completed"


def test_update_schedule_has_exact_warmup_and_budget():
    cfg = update_config()
    cfg.train.warmup_updates = 2
    p = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([{"params": [p], "lr": .01, "base_lr": .01}])
    schedule_updates(optimizer, cfg, 1)
    assert optimizer.param_groups[0]["lr"] == .005
    schedule_updates(optimizer, cfg, 2)
    assert optimizer.param_groups[0]["lr"] == .01
    schedule_updates(optimizer, cfg, cfg.train.max_updates)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(.01*cfg.train.min_lr_ratio)


def test_fixed_probe_preserves_class_fraction(cohort):
    cfg = update_config()
    cfg.train.train_eval_patients = 5
    store = Store(cohort, cfg.train.image_shape, allow_synthetic=True)
    before = torch.get_rng_state().clone()
    tasks = _probe_tasks(store, cfg)
    ids = {i for i, _ in tasks}
    assert len(ids) == 5
    assert sum(store.patients[i]["target"]["pcr"] for i in ids) in (2, 3)
    assert tasks == _probe_tasks(store, cfg)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("field,value", [("max_updates", 0), ("warmup_updates", 7),
    ("eval_every_updates", 0), ("patience_updates", -1), ("sample_weighting", "bad"),
    ("reference_batch_size", 0)])
def test_invalid_update_contracts(field, value):
    cfg = update_config()
    setattr(cfg.train, field, value)
    with pytest.raises(ValueError):
        cfg.validate()
