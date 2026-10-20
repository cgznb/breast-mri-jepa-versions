"""Run the T0-only classifier through training, evaluation and weight retirement."""
from pathlib import Path

import pytest

from mri_vla_jepa.ablation_suite import (
    complete_evaluation_receipt, delete_evaluated_checkpoints,
)
from mri_vla_jepa.data import RawMRIStore, make_raw_synthetic
from mri_vla_jepa.io import load_checkpoint, read_json, write_json
from mri_vla_jepa.training import evaluate, load_config, load_trained, train


ROOT = Path(__file__).resolve().parents[1]


def test_direct_t0_training_resume_evaluation_and_retirement(tmp_path, monkeypatch):
    cfg = load_config(ROOT / "configs/raw_vla_jepa_smoke_t0.yaml")
    cfg.model.enable_world_model = False
    cfg.model.pcr_readout = "zero_state"
    cfg.training_unit, cfg.max_epochs = "epochs", 2
    cfg.selection_metric, cfg.early_stopping_patience = "f1", 50
    cfg.prediction_training = "teacher_forcing"
    cfg.jepa_weight = cfg.rollout_weight = cfg.reconstruction_weight = cfg.variance_weight = 0.
    cfg.variance_definition = "off"
    cfg.diagnostics_every_epochs, cfg.diagnostics_patients, cfg.diagnostics_batch_size = 1, 4, 2
    cfg.validate()
    manifest = make_raw_synthetic(tmp_path / "data", n_train=4, n_val=4, image_shape=cfg.image_shape)
    data = read_json(manifest)
    for patient in data["patients"][-2:]:
        patient["split"] = "test"
    write_json(manifest, data)
    original_read = RawMRIStore._read_scan
    read_stages = []

    def t0_only(self, visit, reference=None):
        assert visit["stage"] == 0, "Direct T0 training/evaluation decoded a future visit"
        read_stages.append(visit["stage"])
        return original_read(self, visit, reference)

    monkeypatch.setattr(RawMRIStore, "_read_scan", t0_only)
    store = RawMRIStore(manifest, cfg.image_shape, allow_synthetic=True)
    suite, run = tmp_path / "suite", tmp_path / "suite/run"
    paused = train(store, cfg, run, stop_after=1)
    assert not paused["completed"]
    trained = train(store, cfg, run, resume=True)
    assert trained["completed"] and read_stages
    saved = load_checkpoint(run / "best.pt")
    assert saved["normalization"]["fit_split"] == "train"
    assert all(not name.startswith(("teacher_encoder.", "world_predictor.", "reconstruction.",
                                    "flow.", "dynamics_queries")) for name in saved["model"])
    model, _, _ = load_trained(run / "best.pt")
    assert model.teacher_encoder is None and model.world_predictor is None
    assert model.pcr_head[0].normalized_shape == (3 * cfg.model.dim,)
    inp, _ = store.batch([(store.by_split["val"][0], 0)], supervised=False)
    features = model(inp)["pcr_state_features"]
    assert features.shape == (1, 3 * cfg.model.dim)
    assert features[:, cfg.model.dim:].count_nonzero().item() == 0
    assert read_json(run / "diagnostics.json")[0]["world"]["status"] == "not_applicable"
    required = {}
    for split in ("val", "test"):
        target = run / f"evaluation_best_{split}.json"
        result = evaluate(run / "best.pt", manifest, target, split=split,
                          allow_synthetic=True, bootstrap=0)
        assert result["landmarks"] == "t0"
        assert len(result["rows"]) == 2 and {row["landmark"] for row in result["rows"]} == {0}
        assert result["metrics"]["n"] == 2
        assert all(result["per_landmark"][f"T{stage}"]["n"] == 0 for stage in (1, 2, 3))
        required[split] = {"path": str(target), "kind": "classification", "split": split}
    protected = tmp_path / "codec.pt"
    protected.write_bytes(b"historical codec, outside generated run")
    with pytest.raises(FileNotFoundError):
        delete_evaluated_checkpoints(run, suite, protected_paths=[protected])
    complete_evaluation_receipt(run, suite, required, manifest=manifest)
    delete_evaluated_checkpoints(run, suite, protected_paths=[protected])
    assert not list(run.glob("*.pt")) and protected.read_bytes().startswith(b"historical codec")
    assert (run / "train_report.json").exists() and (run / "evaluation_best_test.json").exists()
    assert read_json(run / "checkpoint_cleanup.json")["status"] == "weights_deleted"
