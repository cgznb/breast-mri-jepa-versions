"""One-batch debug must use the same VQ initialization and optimizer policy."""
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from mri_vla_jepa.data import RawMRIStore, make_raw_synthetic
from mri_vla_jepa.encoder import VisitMRIEncoder
from mri_vla_jepa.model import RawVLAJEPA
from mri_vla_jepa.model_config import RawVLAJEPAConfig
from mri_vla_jepa.train_config import RawVLATrainConfig
from mri_vla_jepa.vq_encoder import VQContinuousFrontend
import mri_vla_jepa.training as training


@pytest.fixture
def debug_module():
    path = Path(__file__).resolve().parents[1] / "scripts/debug_vla_jepa.py"
    spec = importlib.util.spec_from_file_location("debug_vla_jepa_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny_experiment(tmp_path):
    model_cfg = RawVLAJEPAConfig(
        dim=16, token_grid=(1, 2, 2), heads=2, encoder_depth=1,
        fusion_depth=1, predictor_depth=1, state_queries=2, pcr_queries=2,
        fm_channels=4, encoder_type="vq_prequant", vq_activation_checkpointing=True,
    )
    frontend = VQContinuousFrontend(model_cfg)
    original = {name: value.detach().clone() for name, value in frontend.state_dict().items()}
    codec = tmp_path / "codec.pt"
    torch.save({
        "schema": "symm_world_codec_v2",
        "numeric_contract": "registered_ROI32_shared_training_DCE0_normalization",
        "model_config": dict(frontend.architecture), "codec_state": original,
    }, codec)
    cfg = RawVLATrainConfig(
        model=model_cfg, image_shape=(8, 16, 16), accumulation=1,
        encoder_training="frozen", encoder_checkpoint=str(codec),
        jepa_weight=0., reconstruction_weight=0., variance_weight=0.,
        variance_definition="off", device="cpu", precision="fp32", threads=1,
        allow_synthetic=True,
    )
    manifest = make_raw_synthetic(tmp_path / "data", n_train=2, n_val=2,
                                  image_shape=cfg.image_shape)
    return cfg, manifest, original


def write_config(path, cfg):
    path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    return path


@pytest.mark.parametrize("mode", ["frozen", "finetune"])
@pytest.mark.parametrize("encoder_type", ["vq_prequant", "vq_encoder"])
def test_debug_vq_update_loads_weights_and_uses_production_parameter_groups(
        tmp_path, tiny_experiment, debug_module, monkeypatch, capsys, mode, encoder_type):
    cfg, manifest, original = tiny_experiment
    cfg = replace(cfg, model=replace(cfg.model, encoder_type=encoder_type),
                  encoder_training=mode, encoder_lr=1e-5 if mode == "finetune" else None)
    config = write_config(tmp_path / "config.yaml", cfg)
    captured = {}

    def capture_optimizer(model, config):
        captured["model"] = model
        captured["before"] = {name: value.detach().clone()
                              for name, value in model.encoder.frontend.state_dict().items()}
        optimizer = training._make_optimizer(model, config)
        captured["optimizer"] = optimizer
        return optimizer

    monkeypatch.setattr(debug_module, "_make_optimizer", capture_optimizer)
    before_files = set(tmp_path.rglob("*"))
    debug_module.main(["--config", str(config), "--manifest", str(manifest),
                       "--mode", "train-step"])
    assert set(tmp_path.rglob("*")) == before_files
    model = captured["model"]
    assert all(torch.equal(original[name], value) for name, value in captured["before"].items())
    groups = {group["name"]: group for group in captured["optimizer"].param_groups}
    assert groups["network"]["lr"] == cfg.lr
    frontend = model.encoder.frontend.state_dict()
    if mode == "frozen":
        assert set(groups) == {"network"}
        assert all(torch.equal(original[name], value) for name, value in frontend.items())
        assert all(parameter.grad is None for parameter in model.encoder.pretrained_parameters())
        assert not model.encoder.frontend.training
    else:
        assert groups["pretrained_encoder"]["lr"] == 1e-5
        assert any(not torch.equal(original[name], value) for name, value in frontend.items())
        assert all(parameter.grad is not None
                   for name, parameter in model.encoder.frontend.named_parameters()
                   if encoder_type != "vq_encoder" or not name.startswith("pre_quant."))
        if encoder_type == "vq_encoder":
            assert all(parameter.grad is None for parameter in model.encoder.frontend.pre_quant.parameters())
    assert not model.teacher_encoder.training
    assert all(parameter.grad is None for parameter in model.teacher_encoder.parameters())
    text = capsys.readouterr().out
    assert "weights: pretrained VQ encoder" in text
    assert "encoder.frontend.encoder:" in text
    assert ("encoder.frontend.pre_quant:" in text) == (encoder_type == "vq_prequant")
    assert "encoder.projection:" in text
    assert "backward/AdamW/EMA completed" in text


def test_debug_forecast_rollout_loss_reads_future_without_jepa(
        tmp_path, tiny_experiment, debug_module, monkeypatch):
    cfg, manifest, _ = tiny_experiment
    cfg = replace(cfg, model=replace(cfg.model, encoder_type="vq_encoder",
                  fixed_transition_queries=True, pcr_readout="forecast",
                  history_delta=True, world_residual=True), rollout_weight=.5)
    config = write_config(tmp_path / "forecast.yaml", cfg)
    captured = {}
    original = RawVLAJEPA.compute_loss

    def capture_loss(model, inp, sup, **weights):
        loss, terms = original(model, inp, sup, **weights)
        captured.update(weights=weights, future_count=int(sup.future_mask.sum()),
                        terms={name: float(value.detach()) for name, value in terms.items()})
        return loss, terms

    monkeypatch.setattr(RawVLAJEPA, "compute_loss", capture_loss)
    debug_module.main(["--config", str(config), "--manifest", str(manifest), "--mode", "loss"])
    assert captured["weights"]["rollout_weight"] == .5
    assert captured["future_count"] == 3
    terms = captured["terms"]
    assert terms["jepa"] == 0. and terms["rollout"] > 0.
    assert terms["rollout_count"] == 3
    assert terms["loss"] == pytest.approx(terms["task"] + .5 * terms["rollout"])


def test_saved_debug_uses_manifest_without_historical_image_cache(
        tmp_path, tiny_experiment, debug_module, monkeypatch):
    cfg, manifest, _ = tiny_experiment
    cfg = replace(cfg, model=replace(cfg.model, encoder_type="vq_encoder"),
                  image_cache=str(tmp_path / "missing_training_machine_cache"))
    normalization = RawMRIStore(manifest, cfg.image_shape, allow_synthetic=True).normalization_state()
    model = RawVLAJEPA(cfg.model)
    monkeypatch.setattr(debug_module, "load_trained", lambda path, device:
                        (model, cfg, {"normalization": normalization}))
    before_files = set(tmp_path.rglob("*"))
    debug_module.main(["--checkpoint", str(tmp_path / "saved.pt"),
                       "--manifest", str(manifest), "--mode", "predict"])
    assert cfg.image_cache == str(tmp_path / "missing_training_machine_cache")
    assert set(tmp_path.rglob("*")) == before_files


def test_debug_warmstart_config_fails_before_random_model_creation(
        tmp_path, tiny_experiment, debug_module, monkeypatch):
    cfg, _, _ = tiny_experiment
    cfg = replace(cfg, encoder_checkpoint=None, encoder_training="finetune", encoder_lr=1e-5,
                  initialize_from_checkpoint=str(tmp_path / "stage_a_best.pt"))
    config = write_config(tmp_path / "fine.yaml", cfg)

    def unexpected_model(*args, **kwargs):
        pytest.fail("Rejected warm-start config must not construct a random model")

    monkeypatch.setattr(debug_module, "RawVLAJEPA", unexpected_model)
    with pytest.raises(ValueError, match="Use --checkpoint PATH"):
        debug_module.load_debug_model(debug_module.parser().parse_args(["--config", str(config)]))


def test_raw_default_and_shape_trace_remain_compatible(tiny_experiment, debug_module, monkeypatch, capsys):
    cfg, _, _ = tiny_experiment
    raw = replace(cfg, model=replace(cfg.model, encoder_type="raw"),
                  encoder_checkpoint=None, encoder_training="finetune")
    requested = []
    monkeypatch.setattr(debug_module, "load_config", lambda path: requested.append(path) or raw)
    model, loaded, normalization = debug_module.load_debug_model(debug_module.parser().parse_args([]))
    assert requested == [debug_module.ROOT / "configs/registered_roi32_epoch200_t0_seed17.yaml"]
    assert isinstance(model.encoder, VisitMRIEncoder)
    assert loaded is raw and normalization is None
    handles = debug_module.attach_shape_trace(model)
    model.encoder(torch.randn(1, 3, 8, 16, 16))
    for handle in handles:
        handle.remove()
    assert "encoder.stem.0:" in capsys.readouterr().out


def test_saved_finetune_debug_uses_checkpoint_without_external_codec(
        tmp_path, tiny_experiment, debug_module, monkeypatch):
    cfg, _, _ = tiny_experiment
    cfg = replace(cfg, encoder_training="finetune", encoder_checkpoint=None, encoder_lr=1e-5,
                  initialize_from_checkpoint="unavailable_stage_a.pt")
    model = RawVLAJEPA(cfg.model)
    state = {"normalization": {"from": "checkpoint"}}
    monkeypatch.setattr(debug_module, "load_trained", lambda path, device: (model, cfg, state))
    monkeypatch.setattr(debug_module, "load_config", lambda path: pytest.fail("Checkpoint mode uses saved config"))
    loaded, saved_cfg, normalization = debug_module.load_debug_model(
        debug_module.parser().parse_args(["--checkpoint", str(tmp_path / "fine_best.pt")]))
    assert loaded is model and saved_cfg is cfg and normalization is state["normalization"]
    optimizer = debug_module.configure_debug_optimizer(model, saved_cfg, "train-step")
    assert {group["name"]: group["lr"] for group in optimizer.param_groups} == {
        "network": cfg.lr, "pretrained_encoder": cfg.encoder_lr,
    }
