"""Update-based warmup/cosine must preserve exact endpoint and resume behavior."""
import copy
from dataclasses import replace
import json

import pytest
import torch

from mri_vla_jepa import training
from mri_vla_jepa.data import RawMRIStore
from mri_vla_jepa.io import load_checkpoint, write_json
from mri_vla_jepa.train_config import from_dict
from test_epoch_training import TinyModel, assert_same, make_setup, mock_scores


def test_real_epoch_endpoints_and_parameter_group_ratios():
    cfg = from_dict({"lr_scheduler": "warmup_cosine", "training_unit": "epochs",
                     "accumulation": 1, "lr": 1e-4})
    first, second = torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([{"params": [first], "lr": 1e-4},
                                   {"params": [second], "lr": 1e-5}])
    scheduler = training._make_scheduler(optimizer, cfg, batches_per_epoch=48)
    used = []
    for _ in range(48 * 102):
        used.append([group["lr"] for group in optimizer.param_groups])
        optimizer.step()
        scheduler.step()
    assert used[0][0] == pytest.approx(1e-5)
    assert used[48 * 10 - 1][0] == pytest.approx(1e-4)
    assert used[48 * 100 - 1][0] == pytest.approx(1e-6)
    assert all(rates[0] == pytest.approx(1e-6) for rates in used[48 * 100:])
    assert all(rates[1] == pytest.approx(rates[0] / 10) for rates in used)
    assert all(a[0] <= b[0] for a, b in zip(used[:479], used[1:480]))
    assert all(a[0] >= b[0] for a, b in zip(used[479:], used[480:]))
    assert scheduler.last_epoch == 48 * 102


@pytest.mark.parametrize("stop_after", [1, 3, 4, 8, 12])
def test_warmup_boundary_cosine_and_floor_resume_exactly(tmp_path, monkeypatch, stop_after):
    cfg, _, store = make_setup(tmp_path, "dynamic")
    cfg = replace(cfg, max_epochs=5, lr_scheduler="warmup_cosine", warmup_epochs=1,
                  cosine_end_epoch=4)
    monkeypatch.setattr(training, "RawVLAJEPA", TinyModel)
    training.train(store, cfg, tmp_path / "full")
    training.train(store, cfg, tmp_path / "resumed", stop_after=stop_after)
    saved = load_checkpoint(tmp_path / "resumed/last.pt")
    assert saved["scheduler_state"]["last_epoch"] == stop_after
    training.train(store, cfg, tmp_path / "resumed", resume=True)
    full = load_checkpoint(tmp_path / "full/last.pt")
    resumed = load_checkpoint(tmp_path / "resumed/last.pt")
    for key in ("model", "optimizer", "scheduler_state", "rng", "sampler_state", "history",
                "epoch_history", "epoch_state", "best_nll", "best_epoch", "best_step"):
        assert_same(full[key], resumed[key])
    rates = [entry["lr"] for entry in resumed["history"]]
    assert rates[0] == pytest.approx(cfg.lr * cfg.warmup_start_factor)
    assert rates[2] == pytest.approx(cfg.lr)
    assert rates[11:] == pytest.approx([cfg.cosine_min_lr] * 4)
    for entry in resumed["history"]:
        assert entry["learning_rates"]["network"] == entry["lr"]
    assert resumed["epoch_history"][0]["lr"] == rates[2]


def test_warmup_resume_rejects_missing_stale_or_modified_schedule(tmp_path, monkeypatch):
    cfg, _, store = make_setup(tmp_path)
    cfg = replace(cfg, lr_scheduler="warmup_cosine", warmup_epochs=1, cosine_end_epoch=3)
    monkeypatch.setattr(training, "RawVLAJEPA", TinyModel)
    training.train(store, cfg, tmp_path / "run", stop_after=1)
    original = load_checkpoint(tmp_path / "run/last.pt")
    for kind in ("missing", "progress", "base", "consistent_but_wrong_rate", "step_count", "epoch_progress"):
        state = copy.deepcopy(original)
        if kind == "missing":
            state["scheduler_state"] = None
        elif kind == "progress":
            state["scheduler_state"]["last_epoch"] += 1
        elif kind == "base":
            state["scheduler_state"]["base_lrs"][0] *= 2
        elif kind == "consistent_but_wrong_rate":
            state["optimizer"]["param_groups"][0]["lr"] *= 2
            state["scheduler_state"]["_last_lr"][0] *= 2
        elif kind == "step_count":
            state["scheduler_state"]["_step_count"] += 1
        else:
            state["epoch_state"]["completed_epochs"] += 1
        monkeypatch.setattr(training, "load_checkpoint", lambda path: state)
        with pytest.raises(ValueError, match="scheduler|schedule"):
            training.train(store, cfg, tmp_path / "run", resume=True)


def test_warmup_keeps_existing_early_stopping_policy(tmp_path, monkeypatch):
    cfg, _, store = make_setup(tmp_path, n_train=2)
    cfg = replace(cfg, max_epochs=200, early_stopping_patience=2, lr_scheduler="warmup_cosine")
    monkeypatch.setattr(training, "RawVLAJEPA", TinyModel)
    mock_scores(monkeypatch, [.5] * 3)
    result = training.train(store, cfg, tmp_path / "run")
    assert result["stop_reason"] == "early_stopping" and result["completed_epochs"] == 3
    state = load_checkpoint(tmp_path / "run/last.pt")
    assert state["scheduler_state"]["last_epoch"] == 3


@pytest.mark.parametrize("changes", [
    {"training_unit": "steps"}, {"warmup_epochs": 0}, {"warmup_epochs": True},
    {"warmup_start_factor": 0}, {"warmup_start_factor": True}, {"warmup_start_factor": 1.1},
    {"warmup_start_factor": float("nan")}, {"cosine_end_epoch": 10},
    {"cosine_end_epoch": False}, {"cosine_min_lr": 0}, {"cosine_min_lr": True},
    {"cosine_min_lr": float("inf")}, {"cosine_min_lr": 1.},
])
def test_warmup_config_validation(changes):
    with pytest.raises(ValueError):
        from_dict({"lr_scheduler": "warmup_cosine", "training_unit": "epochs", "accumulation": 1,
                   **changes})


def test_single_update_warmup_is_rejected():
    cfg = from_dict({"lr_scheduler": "warmup_cosine", "training_unit": "epochs",
                     "accumulation": 1, "warmup_epochs": 1})
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=cfg.lr)
    with pytest.raises(ValueError, match="two warmup optimizer updates"):
        training._make_scheduler(optimizer, cfg, batches_per_epoch=1)


def test_downstream_holdout_provenance_does_not_claim_full_independence(tmp_path, monkeypatch):
    cfg, manifest, store = make_setup(tmp_path)
    model = TinyModel(cfg.model)
    state = {"normalization": store.normalization_state(), "schema": training.CHECKPOINT_SCHEMA,
             "patient_partitions": training._patient_partitions(store), "selection_protocol": "t0_direct_nll"}
    # New downstream test patients were excluded from this checkpoint's training and selection.
    data = json.loads(manifest.read_text())
    for patient in data["patients"]:
        if patient["split"] == "val":
            patient["split"] = "test"
    data["provenance"] = {"independence_scope": "downstream_only"}
    state["patient_partitions"]["val"] = []
    write_json(manifest, data)
    monkeypatch.setattr(training, "load_trained", lambda *args: (model, cfg, state))
    report = training.evaluate("unused.pt", manifest, tmp_path / "result.json",
                               split="test", allow_synthetic=True, bootstrap=0)
    assert report["independence_scope"] == "downstream_only"
    assert report["downstream_holdout"] and not report["independent_test"]
    assert report["rows"]


def test_attention_diagnostics_are_saved_per_patient_prefix(tmp_path):
    cfg, _, store = make_setup(tmp_path, "dynamic")

    class AttentionDiagnosticModel(TinyModel):
        def forward(self, inp):
            output = super().forward(inp)
            output["pcr_spatial_attention_entropy"] = torch.full((len(inp.images),), 1.25)
            output["pcr_spatial_effective_tokens"] = torch.full((len(inp.images),), 3.5)
            return output

    rows = training._rows(AttentionDiagnosticModel(cfg.model), store, split="val",
                          profile="all_observed", batch_size=2)
    assert {row["landmark"] for row in rows} == {0, 1, 2, 3}
    assert all(row["pcr_spatial_attention_entropy"] == 1.25 for row in rows)
    assert all(row["pcr_spatial_effective_tokens"] == 3.5 for row in rows)
