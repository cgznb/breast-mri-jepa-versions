"""Joint BCE/JEPA behavior and reproducibility of the isolated dual-task suite."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from mri_vla_jepa.io import load_checkpoint, save_checkpoint
from mri_vla_jepa.model import RawVLAJEPA
from mri_vla_jepa.train_config import load_config
import mri_vla_jepa.training as training

from test_vq_encoder_experiment import (
    assert_same, clone_parameters, codec, setup, single_cpu_thread,
)


ROOT = Path(__file__).resolve().parents[1]


def dual_setup(tmp_path, codec, mode="frozen", max_epochs=2):
    cfg, store = setup(tmp_path, codec, "finetune" if mode == "finetune" else "frozen",
                       max_epochs=max_epochs)
    cfg = replace(cfg, jepa_weight=.5, reconstruction_weight=0., variance_weight=0.,
                  variance_definition="off",
                  model=replace(cfg.model, vq_activation_checkpointing=mode != "raw"))
    if mode == "raw":
        cfg = replace(cfg, model=replace(cfg.model, encoder_type="raw"),
                      encoder_checkpoint=None, encoder_training="finetune", encoder_lr=None)
    return cfg.validate(), store


def fresh_model(cfg):
    model = RawVLAJEPA(cfg.model)
    if cfg.model.encoder_type == "vq_prequant":
        model.initialize_pretrained_encoder(cfg.encoder_checkpoint)
    training._configure(model, cfg, cfg.phase(0))
    return model


def dual_loss(model, inp, sup):
    return model.compute_loss(inp, sup, task_weight=1., jepa_weight=.5, flow_weight=0.,
                              reconstruction_weight=0., variance_weight=0.,
                              variance_definition="off")


def has_gradient(module):
    return any(parameter.grad is not None and parameter.grad.count_nonzero()
               for parameter in module.parameters())


@pytest.mark.parametrize("mode", ["raw", "frozen", "finetune"])
def test_dual_task_updates_shared_fusion_both_heads_and_only_trainable_encoder(
        tmp_path, codec, mode):
    cfg, store = dual_setup(tmp_path, codec, mode)
    model = fresh_model(cfg)
    optimizer = training._make_optimizer(model, cfg)
    modules = [model.fusion, model.pcr_head, model.world_predictor]
    before = [clone_parameters(module) for module in modules]
    if mode != "raw":
        frontend_before = clone_parameters(model.encoder.frontend)
    inp, sup = store.batch([(0, 0), (1, 1)])
    loss, terms = dual_loss(model, inp, sup)
    assert terms["pair_count"] == 5 and terms["jepa"] > 0 and terms["task"] > 0
    assert terms["flow"] == terms["reconstruction"] == terms["variance"] == 0
    torch.testing.assert_close(loss, terms["task"] + .5 * terms["jepa"], atol=0, rtol=0)
    # Both losses independently reach the shared fusion. Each head has its own loss.
    shared = next(model.fusion.parameters())
    for name in ("task", "jepa"):
        gradient = torch.autograd.grad(terms[name], shared, retain_graph=True)[0]
        assert gradient.count_nonzero() and torch.isfinite(gradient).all()
    assert torch.autograd.grad(terms["task"], next(model.world_predictor.parameters()),
                               retain_graph=True, allow_unused=True)[0] is None
    assert torch.autograd.grad(terms["jepa"], next(model.pcr_head.parameters()),
                               retain_graph=True, allow_unused=True)[0] is None
    loss.backward()
    assert all(has_gradient(module) for module in modules)
    assert has_gradient(model.encoder)
    assert not model.teacher_encoder.training
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in model.teacher_encoder.parameters())
    if mode == "frozen":
        assert all(parameter.grad is None for parameter in model.encoder.pretrained_parameters())
        assert not model.encoder.frontend.training
        assert has_gradient(model.encoder.projection) and has_gradient(model.encoder.spatial)
    elif mode == "finetune":
        assert has_gradient(model.encoder.frontend)
        assert {group["name"]: group["lr"] for group in optimizer.param_groups} == {
            "network": 1e-4, "pretrained_encoder": 1e-5,
        }
    optimizer.step()
    model.update_teacher()
    assert all(any(not torch.equal(saved[name], parameter)
                   for name, parameter in module.named_parameters())
               for saved, module in zip(before, modules, strict=True))
    if mode == "frozen":
        assert_same(frontend_before, clone_parameters(model.encoder.frontend))
        assert_same(model.teacher_encoder.frontend.state_dict(), model.encoder.frontend.state_dict())
    elif mode == "finetune":
        assert any(not torch.equal(frontend_before[name], parameter)
                   for name, parameter in model.encoder.frontend.named_parameters())


@pytest.mark.parametrize("mode", ["frozen", "finetune"])
def test_real_future_changes_jepa_supervision_without_changing_pcr(tmp_path, codec, mode):
    cfg, store = dual_setup(tmp_path, codec, mode)
    model = fresh_model(cfg).eval()
    inp, sup = store.batch([(0, 0), (1, 1)])
    changed = sup.future.clone()
    changed[sup.future_mask] = torch.randn_like(changed[sup.future_mask]) * 7
    altered = replace(sup, future=changed)
    seen = []
    hook = model.pcr_head.register_forward_hook(lambda module, args, value: seen.append(value.detach().clone()))
    with torch.no_grad():
        _, initial = dual_loss(model, inp, sup)
        _, modified = dual_loss(model, inp, altered)
    hook.remove()
    torch.testing.assert_close(seen[0], seen[1], atol=0, rtol=0)
    torch.testing.assert_close(initial["task"], modified["task"], atol=0, rtol=0)
    assert initial["jepa"] != modified["jepa"]
    assert inp.images[~inp.observed_mask].count_nonzero() == 0
    leaked = inp.images.clone()
    leaked[sup.future_mask] = sup.future[sup.future_mask]
    with pytest.raises(ValueError, match="Unobserved MRI slots must be zero"):
        model(replace(inp, images=leaked))


@pytest.mark.parametrize("mode", ["frozen", "finetune"])
def test_terminal_t3_has_zero_world_loss_and_keeps_pcr_gradient(tmp_path, codec, mode):
    cfg, store = dual_setup(tmp_path, codec, mode)
    model = fresh_model(cfg)
    inp, sup = store.batch([(0, 3), (1, 3)])
    sup = replace(sup, future=torch.full_like(sup.future, float("nan")))
    assert not inp.query_mask.any() and not sup.future_mask.any()
    loss, terms = dual_loss(model, inp, sup)
    assert torch.isfinite(loss) and terms["pair_count"] == terms["jepa"] == 0
    torch.testing.assert_close(loss, terms["task"], atol=0, rtol=0)
    loss.backward()
    assert has_gradient(model.pcr_head) and has_gradient(model.fusion)
    assert not has_gradient(model.world_predictor)
    assert all(parameter.grad is None for parameter in model.teacher_encoder.parameters())


@pytest.mark.parametrize("missing,expected", [
    ((1,), [False, False, True]), ((2,), [True, False, False]),
    ((1, 2), [False, False, False]),
])
@pytest.mark.parametrize("mode", ["frozen", "finetune"])
def test_missing_visits_remove_adjacent_transitions_and_ignore_masked_nans(
        tmp_path, codec, mode, missing, expected):
    cfg, store = dual_setup(tmp_path, codec, mode)
    model = fresh_model(cfg)
    inp, sup = store.batch([(0, 0), (1, 0)])
    mask, future = sup.future_mask.clone(), sup.future.clone()
    mask[:, missing] = False
    future[~mask] = float("nan")
    sup = replace(sup, future_mask=mask, future=future)
    dynamics = model(inp)["dynamics_features"]
    world, detail = model.teacher_forcing_loss(inp, sup, dynamics, return_details=True)
    assert detail["pair_mask"].tolist() == [expected, expected]
    assert detail["prediction"].shape == (2, 3, 4, 16)
    assert not detail["teacher_states"].requires_grad
    assert torch.isfinite(world) and torch.isfinite(detail["teacher_states"]).all()
    if any(expected):
        manual = detail["pair_losses"][:, expected].mean()
        torch.testing.assert_close(world, manual)
    else:
        assert world == 0
    loss, terms = dual_loss(model, inp, sup)
    assert terms["pair_count"] == 2 * sum(expected)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


@pytest.mark.parametrize("mode", ["frozen", "finetune"])
def test_autonomous_latent_generation_has_expected_shape_and_backward(tmp_path, codec, mode):
    cfg, store = dual_setup(tmp_path, codec, mode)
    model = fresh_model(cfg)
    inp, _ = store.batch([(0, 0), (1, 1)], supervised=False)
    states = model._rollout(inp)
    assert states.shape == (2, 4, 4, 16) and torch.isfinite(states).all()
    assert states[~inp.query_mask].count_nonzero() == 0
    states.square().mean().backward()
    assert has_gradient(model.world_predictor) and has_gradient(model.fusion)
    assert all(parameter.grad is None for parameter in model.teacher_encoder.parameters())
    if mode == "frozen":
        assert not has_gradient(model.encoder.frontend)
    else:
        assert has_gradient(model.encoder.frontend)
    # This experiment predicts latent states; the optional voxel-flow is disabled.
    assert model.flow is None
    with pytest.raises(ValueError, match="Generation is disabled"):
        model.generate(inp)


@pytest.mark.parametrize("mode", ["raw", "frozen", "finetune"])
def test_dual_task_cpu_mid_epoch_resume_is_exact(tmp_path, codec, mode):
    cfg, store = dual_setup(tmp_path, codec, mode)
    training.train(store, cfg, tmp_path / "full")
    training.train(store, cfg, tmp_path / "resumed", stop_after=1)
    partial = load_checkpoint(tmp_path / "resumed/last.pt")
    assert partial["epoch_state"]["completed_epochs"] == 0
    assert partial["epoch_state"]["position"] == 2
    training.train(store, cfg, tmp_path / "resumed", resume=True)
    full, resumed = [load_checkpoint(tmp_path / folder / "last.pt") for folder in ("full", "resumed")]
    for name in ("model", "optimizer", "rng", "sampler_state", "history", "epoch_history", "epoch_state", "best_nll"):
        assert_same(full[name], resumed[name])
    assert any(entry["jepa"] > 0 and entry["pair_count"] > 0 for entry in full["history"])
    if mode == "frozen":
        frontend = {name.removeprefix("encoder.frontend."): value for name, value in full["model"].items()
                    if name.startswith("encoder.frontend.")}
        assert_same(frontend, codec[1])


@pytest.fixture
def dual_stage_a(tmp_path, codec):
    cfg, store = dual_setup(tmp_path, codec, max_epochs=1)
    training.train(store, cfg, tmp_path / "stage_a")
    return cfg, store, tmp_path / "stage_a/best.pt"


def test_dual_finetune_loads_matching_a_best_with_fresh_optimizer_and_progress(
        tmp_path, dual_stage_a, monkeypatch):
    cfg_a, store, best = dual_stage_a
    saved_a = load_checkpoint(best)
    cfg_b = replace(cfg_a, encoder_training="finetune", encoder_lr=1e-5,
                    encoder_checkpoint=None, initialize_from_checkpoint=str(best), max_epochs=2)
    captured = {}
    make_optimizer = training._make_optimizer
    original_loss = RawVLAJEPA.compute_loss

    def capture_optimizer(model, cfg):
        optimizer = make_optimizer(model, cfg)
        assert not optimizer.state
        captured["optimizer"] = optimizer
        return optimizer

    def check_initial_model(model, *args, **kwargs):
        if not captured.get("checked"):
            assert_same(model.state_dict(), saved_a["model"])
            assert not captured["optimizer"].state
            captured["checked"] = True
        return original_loss(model, *args, **kwargs)

    monkeypatch.setattr(training, "_make_optimizer", capture_optimizer)
    monkeypatch.setattr(RawVLAJEPA, "compute_loss", check_initial_model)
    training.train(store, cfg_b, tmp_path / "stage_b", stop_after=1)
    saved_b = load_checkpoint(tmp_path / "stage_b/last.pt")
    assert captured["checked"] and saved_b["completed_steps"] == 1
    assert saved_b["epoch_state"]["completed_epochs"] == 0 and len(saved_b["history"]) == 1
    assert saved_b["history"][0]["jepa"] > 0
    assert all(state["step"] == 1 for state in saved_b["optimizer"]["state"].values())
    init = saved_b["initialization"]
    stat = best.stat()
    assert init["checkpoint"] == {"path": str(best.resolve()), "size_bytes": stat.st_size,
                                   "mtime_ns": stat.st_mtime_ns}
    assert init["source_completed_steps"] == saved_a["completed_steps"]
    assert init["source_best_validation_nll"] == saved_a["best_nll"]
    assert init["model_weights_loaded"] and not init["optimizer_restored"] and not init["sampler_restored"]


@pytest.mark.parametrize("corruption,match", [
    ("jepa_weight", "loss|sampling|configuration"),
    ("source_snapshot", "source|implementation"),
    ("patient_partitions", "population"),
])
def test_dual_finetune_rejects_incompatible_parent_identity(
        tmp_path, dual_stage_a, corruption, match):
    cfg, store, best = dual_stage_a
    payload = load_checkpoint(best)
    if corruption == "jepa_weight":
        payload["config"]["jepa_weight"] = 0.
    elif corruption == "source_snapshot":
        payload["source_snapshot"]["model.py"] = "different implementation"
    else:
        payload["patient_partitions"]["train"] = payload["patient_partitions"]["train"][:-1]
    changed = tmp_path / "incompatible_best.pt"
    save_checkpoint(changed, payload)
    cfg_b = replace(cfg, encoder_training="finetune", encoder_lr=1e-5,
                    encoder_checkpoint=None, initialize_from_checkpoint=str(changed))
    with pytest.raises(ValueError, match=match):
        training.train(store, cfg_b, tmp_path / "stage_b")


@pytest.mark.parametrize("seed", [17, 43])
def test_formal_dual_task_configs_match_the_locked_protocol_and_new_parent(seed):
    configurations = {}
    for stage in ("raw", "frozen", "finetune"):
        name = f"vq_dualtask_{stage}_dynamic_seed{seed}.yaml"
        path = ROOT / "configs" / name
        packaged = ROOT / "src/mri_vla_jepa/resources/configs" / name
        assert path.read_bytes() == packaged.read_bytes()
        cfg = configurations[stage] = load_config(path)
        assert cfg.seed == seed and cfg.landmarks == "all_observed"
        assert cfg.task_weight == 1. and cfg.jepa_weight == .5
        assert cfg.flow_weight == cfg.reconstruction_weight == cfg.variance_weight == 0.
        assert cfg.variance_definition == "off" and not cfg.model.enable_flow
        assert cfg.batch_size == 16 and cfg.accumulation == 1
        assert cfg.training_unit == "epochs" and cfg.max_epochs == 200
        assert cfg.early_stopping_patience == 50 and cfg.model.dropout == .3 and cfg.lr == 1e-4
    raw, frozen, fine = (configurations[stage] for stage in ("raw", "frozen", "finetune"))
    assert raw.model.encoder_type == "raw" and not raw.model.vq_activation_checkpointing
    assert raw.encoder_checkpoint is raw.encoder_lr is raw.initialize_from_checkpoint is None
    assert frozen.model.encoder_type == fine.model.encoder_type == "vq_prequant"
    assert frozen.model.vq_activation_checkpointing and fine.model.vq_activation_checkpointing
    assert frozen.model == fine.model
    assert frozen.encoder_training == "frozen" and frozen.initialize_from_checkpoint is None
    assert frozen.encoder_checkpoint == "/root/autodl-tmp/MAM/symm-fm-world-v2/data/registered_roi32/codec.pt"
    assert fine.encoder_training == "finetune" and fine.encoder_checkpoint is None and fine.encoder_lr == 1e-5
    import runpy
    release = runpy.run_path(str(ROOT / "scripts/release_runtime.py"))
    assert release["portable_config"](fine.to_dict(), ROOT)["initialize_from_checkpoint"] == str(
        ROOT / f"runs/vq_dualtask_20261012/frozen_dynamic_seed{seed}/best.pt")
    allowed = {"model", "encoder_checkpoint", "encoder_training", "encoder_lr", "initialize_from_checkpoint"}
    snapshots = [cfg.to_dict() for cfg in (raw, frozen, fine)]
    assert all({key: value for key, value in snapshot.items() if key not in allowed} == {
        key: value for key, value in snapshots[0].items() if key not in allowed} for snapshot in snapshots[1:])
    model_allowed = {"encoder_type", "vq_activation_checkpointing"}
    assert {key: value for key, value in snapshots[0]["model"].items() if key not in model_allowed} == {
        key: value for key, value in snapshots[1]["model"].items() if key not in model_allowed}
    original = load_config(ROOT / f"configs/registered_roi32_epoch200_l1_dynamic_seed{seed}.yaml").to_dict()
    changed = {"jepa_weight", "image_cache"}
    assert {key: value for key, value in snapshots[0].items() if key not in changed} == {
        key: value for key, value in original.items() if key not in changed}
