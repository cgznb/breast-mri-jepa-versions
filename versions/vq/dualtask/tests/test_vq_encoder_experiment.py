"""CPU contracts for the isolated pretrained continuous VQ encoder experiment."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from mri_vla_jepa.data import RawMRIStore, make_raw_synthetic
from mri_vla_jepa.encoder import VisitMRIEncoder, VQVisitMRIEncoder
from mri_vla_jepa.io import load_checkpoint, save_checkpoint
from mri_vla_jepa.model import RawVLAJEPA
from mri_vla_jepa.model_config import RawVLAJEPAConfig
from mri_vla_jepa.train_config import RawVLATrainConfig, from_dict
from mri_vla_jepa.vq_encoder import VQContinuousFrontend
import mri_vla_jepa.training as training


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model_config(**updates):
    return RawVLAJEPAConfig(
        dim=16, token_grid=(1, 2, 2), heads=2, encoder_depth=1,
        fusion_depth=1, predictor_depth=1, state_queries=2, pcr_queries=2,
        fm_channels=4, ema_decay=.8, encoder_type="vq_prequant", **updates,
    )


@pytest.fixture(scope="module")
def codec(tmp_path_factory):
    """Safe synthetic export with the same frontend schema as the real codec."""
    directory = tmp_path_factory.mktemp("synthetic_vq")
    with torch.random.fork_rng():
        torch.manual_seed(814)
        frontend = VQContinuousFrontend(model_config())
    weights = {name: value.detach().clone() for name, value in frontend.state_dict().items()}
    payload = {
        "schema": "symm_world_codec_v2",
        "numeric_contract": "registered_ROI32_shared_training_DCE0_normalization",
        "model_config": dict(frontend.architecture),
        "codec_state": {
            **weights,
            "decoder.unused.weight": torch.tensor([97.]),
            "codebook.unused.weight": torch.tensor([89.]),
        },
    }
    path = directory / "synthetic_codec.pt"
    save_checkpoint(path, payload)
    return path, weights, payload


def setup(tmp_path, codec, mode="frozen", *, max_epochs=2, profile="all_observed"):
    cfg = RawVLATrainConfig(
        model=model_config(dropout=.1), image_shape=(8, 16, 16),
        training_unit="epochs", max_epochs=max_epochs, batch_size=2, accumulation=1,
        lr=1e-4, encoder_lr=1e-5 if mode == "finetune" else None,
        encoder_checkpoint=str(codec[0]), encoder_training=mode,
        landmarks=profile, validation_every=1, checkpoint_every=2,
        validation_batch_size=2, device="cpu", precision="fp32", threads=1,
        seed=719, deterministic=True, allow_synthetic=True,
    )
    manifest = make_raw_synthetic(tmp_path / "data", n_train=3, n_val=2,
                                  image_shape=cfg.image_shape)
    return cfg, RawMRIStore(manifest, cfg.image_shape, allow_synthetic=True)


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            assert_same(left[name], right[name])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_same(a, b)
    else:
        assert left == right


def clone_parameters(module):
    return {name: value.detach().clone() for name, value in module.named_parameters()}


def any_parameter_changed(before, module):
    return any(not torch.equal(before[name], value) for name, value in module.named_parameters())


def test_safe_export_loads_exact_frontend_weights_and_syncs_teacher(codec):
    model = RawVLAJEPA(model_config())
    metadata = model.initialize_pretrained_encoder(codec[0])
    assert metadata["numeric_contract"] == "registered_ROI32_shared_training_DCE0_normalization"
    assert_same(model.encoder.frontend.state_dict(), codec[1])
    assert_same(model.teacher_encoder.state_dict(), model.encoder.state_dict())
    assert not model.teacher_encoder.training
    assert all(not p.requires_grad for p in model.teacher_encoder.parameters())
    # The unused decoder/codebook tensors in the export never become modules.
    forbidden = ("decoder", "quantizer", "codebook", "discriminator")
    assert all(not any(part in name for part in forbidden)
               for name, _ in model.encoder.named_modules())
    assert all(name.startswith(("encoder.", "pre_quant.")) for name in codec[1])


@pytest.mark.parametrize("corruption", ["architecture", "missing", "shape", "nonfinite", "numeric"])
def test_codec_load_rejects_incompatible_export_before_loading(tmp_path, codec, corruption):
    payload = deepcopy(codec[2])
    name = next(iter(codec[1]))
    if corruption == "architecture":
        payload["model_config"]["embedding_dim"] = 9
    elif corruption == "missing":
        del payload["codec_state"][name]
    elif corruption == "shape":
        payload["codec_state"][name] = torch.zeros(1)
    elif corruption == "nonfinite":
        payload["codec_state"][name].flatten()[0] = float("nan")
    else:
        payload["numeric_contract"] = "other_normalization"
    path = tmp_path / "incompatible.pt"
    save_checkpoint(path, payload)
    frontend = VQContinuousFrontend(model_config())
    before = clone_parameters(frontend)
    with pytest.raises(ValueError):
        frontend.initialize_pretrained(path)
    assert_same(before, clone_parameters(frontend))


def test_three_phases_share_weights_and_preserve_concatenation_order(codec):
    cfg = replace(model_config(), dim=128, heads=4, token_grid=(2, 4, 4))
    encoder = VQVisitMRIEncoder(cfg).eval()
    encoder.initialize_pretrained(codec[0])
    # Distinct phases detect both accidentally shared input and changed order.
    images = torch.randn(2, 3, 8, 16, 16) + torch.arange(3)[None, :, None, None, None] * 3
    seen = []
    handle = encoder.pool.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    with torch.no_grad():
        result = encoder(images)
        expected = torch.cat([encoder.frontend(images[:, index:index + 1]) for index in range(3)], dim=1)
    handle.remove()
    assert result.shape == (2, 32, 128)
    assert seen[0].shape == (2, 24, 2, 4, 4)
    torch.testing.assert_close(seen[0], expected, atol=0, rtol=0)
    assert len([name for name, module in encoder.named_modules() if isinstance(module, VQContinuousFrontend)]) == 1


def test_frozen_frontend_stays_exact_while_adapter_fusion_and_pcr_update(tmp_path, codec):
    cfg, store = setup(tmp_path, codec)
    model = RawVLAJEPA(cfg.model)
    model.initialize_pretrained_encoder(codec[0])
    training._configure(model, cfg, "joint")
    optimizer = training._make_optimizer(model, cfg)
    frontend_before = clone_parameters(model.encoder.frontend)
    modules = [model.encoder.projection, model.encoder.spatial, model.fusion, model.pcr_head]
    before = [clone_parameters(module) for module in modules]
    inp, sup = store.batch([(0, 0), (1, 0)])
    loss, _ = model.compute_loss(inp, sup)
    loss.backward()
    assert all(p.grad is None and not p.requires_grad for p in model.encoder.pretrained_parameters())
    assert all(p.grad is None for p in model.teacher_encoder.parameters())
    assert not model.encoder.frontend.training
    optimizer.step()
    assert_same(frontend_before, clone_parameters(model.encoder.frontend))
    assert all(any_parameter_changed(saved, module) for saved, module in zip(before, modules, strict=True))
    ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert ids.isdisjoint(id(p) for p in model.encoder.pretrained_parameters())
    # Frozen weights must be copied exactly rather than accumulating EMA roundoff.
    for _ in range(20):
        model.update_teacher()
    assert_same(model.teacher_encoder.frontend.state_dict(), codec[1])


def test_finetune_frontend_receives_gradients_and_its_own_learning_rate(tmp_path, codec):
    cfg, store = setup(tmp_path, codec, "finetune")
    model = RawVLAJEPA(cfg.model)
    model.initialize_pretrained_encoder(codec[0])
    training._configure(model, cfg, "joint")
    optimizer = training._make_optimizer(model, cfg)
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert set(groups) == {"network", "pretrained_encoder"}
    assert groups["network"]["lr"] == 1e-4
    assert groups["pretrained_encoder"]["lr"] == 1e-5
    pretrained = {id(p) for p in model.encoder.pretrained_parameters()}
    assert {id(p) for p in groups["pretrained_encoder"]["params"]} == pretrained
    assert pretrained.isdisjoint(id(p) for p in groups["network"]["params"])
    optimizer_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(optimizer_ids) == len(set(optimizer_ids))
    assert set(optimizer_ids) == {id(p) for p in model.parameters() if p.requires_grad}
    before = clone_parameters(model.encoder.frontend)
    inp, sup = store.batch([(0, 0), (1, 0)])
    loss, _ = model.compute_loss(inp, sup)
    loss.backward()
    assert any(p.grad is not None and p.grad.count_nonzero() for p in model.encoder.pretrained_parameters())
    assert all(p.grad is None for p in model.teacher_encoder.parameters())
    optimizer.step()
    assert any_parameter_changed(before, model.encoder.frontend)


def test_teacher_ema_tracks_finetuned_frontend_and_adapter(codec):
    model = RawVLAJEPA(model_config())
    model.initialize_pretrained_encoder(codec[0])
    model.encoder.set_pretrained_trainable(True)
    before = clone_parameters(model.teacher_encoder)
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(.03125)
    model.update_teacher()
    for name, parameter in model.teacher_encoder.named_parameters():
        online = dict(model.encoder.named_parameters())[name]
        torch.testing.assert_close(parameter, before[name] * .8 + online * .2)
    model.train()
    assert not model.teacher_encoder.training
    assert all(p.grad is None and not p.requires_grad for p in model.teacher_encoder.parameters())


def test_activation_checkpointing_preserves_finetune_gradients(tmp_path, codec):
    cfg, store = setup(tmp_path, codec, "finetune")
    plain_cfg = replace(cfg.model, dropout=0., vq_activation_checkpointing=False)
    checkpoint_cfg = replace(plain_cfg, vq_activation_checkpointing=True)
    plain = RawVLAJEPA(plain_cfg)
    plain.initialize_pretrained_encoder(codec[0])
    checkpointed = RawVLAJEPA(checkpoint_cfg)
    checkpointed.load_state_dict(plain.state_dict())
    for model in (plain, checkpointed):
        model.encoder.set_pretrained_trainable(True)
        model.train()
    inp, sup = store.batch([(0, 0), (1, 0)])
    losses = []
    for model in (plain, checkpointed):
        loss, _ = model.compute_loss(inp, sup, jepa_weight=0., reconstruction_weight=0., variance_weight=0.)
        losses.append(loss.detach())
        loss.backward()
    torch.testing.assert_close(losses[0], losses[1], atol=0, rtol=0)
    for (left_name, left), (right_name, right) in zip(plain.named_parameters(), checkpointed.named_parameters(), strict=True):
        assert left_name == right_name
        if left.grad is None:
            assert right.grad is None
        else:
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)


def test_raw_encoder_defaults_keep_old_state_keys_and_optimizer_contract():
    cfg = RawVLATrainConfig(model=replace(model_config(), encoder_type="raw"),
                            device="cpu", precision="fp32")
    legacy = cfg.to_dict()
    for name in ("encoder_checkpoint", "encoder_training", "encoder_lr", "initialize_from_checkpoint"):
        legacy.pop(name)
    for name in list(legacy["model"]):
        if name == "encoder_type" or name.startswith("vq_"):
            legacy["model"].pop(name)
    restored = from_dict(legacy)
    model = RawVLAJEPA(restored.model)
    assert isinstance(model.encoder, VisitMRIEncoder)
    assert "encoder.stem.0.weight" in model.state_dict()
    assert not any("frontend" in name for name in model.state_dict())
    optimizer = training._make_optimizer(model, restored)
    assert len(optimizer.param_groups) == 1
    assert "name" not in optimizer.param_groups[0]
    assert training._checkpoint_config_matches(legacy, restored)


@pytest.mark.parametrize("mode", ["frozen", "finetune"])
def test_vq_synthetic_epoch_mid_epoch_resume_is_exact(tmp_path, codec, mode):
    cfg, store = setup(tmp_path, codec, mode)
    training.train(store, cfg, tmp_path / "full")
    partial = training.train(store, cfg, tmp_path / "resumed", stop_after=1)
    assert not partial["completed"]
    paused = load_checkpoint(tmp_path / "resumed/last.pt")
    assert paused["epoch_state"]["position"] == 2
    assert paused["epoch_state"]["batches_completed"] == 1
    training.train(store, cfg, tmp_path / "resumed", resume=True)
    full, resumed = [load_checkpoint(tmp_path / folder / "last.pt") for folder in ("full", "resumed")]
    for name in ("model", "optimizer", "rng", "sampler_state", "history", "epoch_history", "epoch_state", "best_nll"):
        assert_same(full[name], resumed[name])
    frontend = {name.removeprefix("encoder.frontend."): value for name, value in full["model"].items()
                if name.startswith("encoder.frontend.")}
    if mode == "frozen":
        assert_same(frontend, codec[1])
    else:
        assert any(not torch.equal(frontend[name], value) for name, value in codec[1].items())


@pytest.fixture
def stage_a(tmp_path, codec):
    cfg, store = setup(tmp_path, codec, max_epochs=1)
    training.train(store, cfg, tmp_path / "stage_a")
    best = tmp_path / "stage_a/best.pt"
    assert best.exists()
    return cfg, store, best


def test_stage_b_initializes_stage_a_best_but_starts_fresh_optimizer_and_epoch(tmp_path, stage_a, monkeypatch):
    cfg_a, store, best = stage_a
    cfg_b = replace(cfg_a, encoder_training="finetune", encoder_lr=1e-5, encoder_checkpoint=None,
                    initialize_from_checkpoint=str(best), max_epochs=2)
    saved_a = load_checkpoint(best)
    captured = {}
    make_optimizer = training._make_optimizer

    def capture_optimizer(model, cfg):
        captured["model"] = model
        optimizer = make_optimizer(model, cfg)
        assert not optimizer.state
        captured["optimizer"] = optimizer
        return optimizer

    monkeypatch.setattr(training, "_make_optimizer", capture_optimizer)
    model_loss = RawVLAJEPA.compute_loss

    def check_first_step(model, *args, **kwargs):
        if "checked" not in captured:
            assert_same(model.state_dict(), saved_a["model"])
            assert not captured["optimizer"].state
            captured["checked"] = True
        return model_loss(model, *args, **kwargs)

    monkeypatch.setattr(RawVLAJEPA, "compute_loss", check_first_step)
    training.train(store, cfg_b, tmp_path / "stage_b", stop_after=1)
    assert captured["checked"]
    saved_b = load_checkpoint(tmp_path / "stage_b/last.pt")
    assert saved_b["completed_steps"] == 1
    assert saved_b["epoch_state"]["completed_epochs"] == 0
    assert len(saved_b["history"]) == 1
    assert saved_a["completed_steps"] == 2
    assert all(value["step"] == 1 for value in saved_b["optimizer"]["state"].values())


@pytest.mark.parametrize("field,match", [
    ("normalization", "normalization"), ("patient_partitions", "population"),
    ("model", "model"), ("data_signature", "data"),
    ("source_snapshot", "source|implementation"), ("loss", "loss|sampling|config"),
])
def test_stage_b_rejects_stage_a_contract_mismatches(tmp_path, stage_a, field, match):
    cfg_a, store, best = stage_a
    payload = load_checkpoint(best)
    if field == "normalization":
        payload[field]["clinical_mean"][0] += 1
    elif field == "patient_partitions":
        payload[field]["train"] = payload[field]["train"][:-1]
    elif field == "model":
        payload["config"]["model"]["state_queries"] += 1
    elif field == "data_signature":
        payload[field]["image_shape"] = [4, 8, 8]
    elif field == "source_snapshot":
        payload[field]["vq_encoder.py"] = "different source"
    else:
        payload["config"]["jepa_weight"] += .1
    altered = tmp_path / "altered_stage_a" / "best.pt"
    save_checkpoint(altered, payload)
    cfg_b = replace(cfg_a, encoder_training="finetune", encoder_lr=1e-5, encoder_checkpoint=None,
                    initialize_from_checkpoint=str(altered), max_epochs=2)
    with pytest.raises(ValueError, match=match):
        training.train(store, cfg_b, tmp_path / "stage_b")


def test_trained_vq_checkpoint_is_self_contained(tmp_path, stage_a, monkeypatch):
    cfg, store, best = stage_a
    original, _, _ = training.load_trained(best)
    inp, _ = store.batch([(0, 0)], supervised=False)
    with torch.no_grad():
        expected = original(inp)["pcr_logit"]
    payload = load_checkpoint(best)
    payload["config"]["encoder_checkpoint"] = str(tmp_path / "missing_original_codec.pt")
    portable = tmp_path / "portable.pt"
    save_checkpoint(portable, payload)

    def forbidden_external_initialization(*args, **kwargs):
        raise AssertionError("A complete training checkpoint must not reopen the pretrained codec")

    monkeypatch.setattr(VQContinuousFrontend, "initialize_pretrained", forbidden_external_initialization)
    restored, _, _ = training.load_trained(portable)
    with torch.no_grad():
        actual = restored(inp)["pcr_logit"]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert_same(restored.encoder.state_dict(), original.encoder.state_dict())
