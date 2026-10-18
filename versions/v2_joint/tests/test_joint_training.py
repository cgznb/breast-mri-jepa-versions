"""Joint objectives, legal observed anchors, and recoverable CPU training."""
from dataclasses import replace
import json
import math

import pytest
import torch

from response_jepa.contracts import Batch
from response_jepa.data import Store, planned_tasks
from response_jepa.engine import evaluate, load_checkpoint, load_for_inference, make_loader, train
from response_jepa.model import ResponseJEPA
from response_jepa.objectives import joint_loss
from response_jepa.synthetic import small_config


def joint_config(profile="dynamic"):
    cfg = small_config("C")
    cfg.train.phase = "joint"
    cfg.train.profile = profile
    cfg.train.freeze_encoder = False
    cfg.train.augment = False
    cfg.loss.align = 0.0
    return cfg.validate()


def baseline_batch(batch):
    observed = torch.zeros_like(batch.inputs.observed)
    observed[:, 0] = True
    inp = replace(batch.inputs,
                  images=batch.targets.future.masked_fill(~observed[:, :, None, None, None, None], 0),
                  observed=observed, landmark=torch.zeros_like(batch.inputs.landmark),
                  query_mask=~observed)
    return Batch(inp, replace(batch.targets, future_mask=~observed), batch.keys)


def has_gradient(module):
    return any(p.grad is not None and bool(p.grad.count_nonzero()) for p in module.parameters())


def test_joint_objective_trains_all_branches_and_only_ema_updates_teacher(batch):
    cfg = joint_config("t0")
    model = ResponseJEPA(cfg.model)
    model.configure_trainable("joint", False)
    model.train()
    batch = baseline_batch(batch)
    future = batch.targets.future.detach().clone().requires_grad_()
    batch = Batch(batch.inputs, replace(batch.targets, future=future), batch.keys)
    teacher_before = {k: v.detach().clone() for k, v in model.target_encoder.named_parameters()}

    loss, terms, out = joint_loss(model, batch, cfg, augment=False)
    assert torch.isfinite(loss)
    assert out["forecast_logit"] is not None
    assert terms["fused"] > 0
    assert terms["tf"] > 0
    assert terms["rollout"] > 0
    loss.backward()

    for module in (model.encoder, model.fusion, model.world.output, model.direct_head,
                   model.terminal_head, model.mask_predictor, model.repr_projector):
        assert has_gradient(module), type(module).__name__
    assert future.grad is None
    assert all(not p.requires_grad and p.grad is None for p in model.target_encoder.parameters())
    assert model.encoder.training and not model.target_encoder.training

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    optimizer.step()
    for name, param in model.target_encoder.named_parameters():
        torch.testing.assert_close(param, teacher_before[name], atol=0, rtol=0)
    decay = 0.9
    model.update_teacher(decay)
    changed = False
    online = dict(model.encoder.named_parameters())
    for name, param in model.target_encoder.named_parameters():
        expected = torch.lerp(teacher_before[name], online[name].detach(), 1 - decay)
        torch.testing.assert_close(param, expected, atol=0, rtol=0)
        changed |= not torch.equal(param, teacher_before[name])
    assert changed


def test_joint_future_targets_cannot_change_classification_or_rollout(batch):
    cfg = joint_config("t0")
    model = ResponseJEPA(cfg.model)
    model.configure_trainable("joint", False)
    model.eval()
    # Nonzero residual weights exercise the learned forecast, beyond the initial copy state.
    torch.nn.init.normal_(model.world.output.weight, std=0.02)
    batch = baseline_batch(batch)
    torch.manual_seed(20261009)
    first_loss, _, first = joint_loss(model, batch, cfg, augment=False)
    changed = Batch(batch.inputs, replace(batch.targets, future=batch.targets.future + 30), batch.keys)
    torch.manual_seed(20261009)
    changed_loss, _, second = joint_loss(model, changed, cfg, augment=False)
    for name in ("logit", "direct_logit", "forecast_logit", "states"):
        torch.testing.assert_close(first[name], second[name], atol=0, rtol=0)
    assert not torch.equal(first_loss, changed_loss)


def test_joint_auxiliary_uses_actual_legal_anchor_not_landmark(batch, monkeypatch):
    import response_jepa.objectives as objectives
    cfg = joint_config()
    model = ResponseJEPA(cfg.model)
    model.configure_trainable("joint", False)
    model.eval()
    observed = torch.zeros_like(batch.inputs.observed)
    observed[0, 0] = True
    observed[1, :2] = True
    inp = replace(batch.inputs,
                  images=batch.targets.future.masked_fill(~observed[:, :, None, None, None, None], 0),
                  observed=observed, landmark=torch.full_like(batch.inputs.landmark, 2),
                  query_mask=torch.tensor([[False, False, False, True]] * 2))
    sup = replace(batch.targets, future_mask=inp.query_mask)
    captured = []
    original = objectives.representation_loss

    def spy(m, representation_batch, c, *, augment=True):
        captured.append({k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                         for k, v in representation_batch.items()})
        return original(m, representation_batch, c, augment=augment)

    monkeypatch.setattr(objectives, "representation_loss", spy)
    joint_loss(model, Batch(inp, sup, batch.keys), cfg, augment=False)
    assert len(captured) == 1
    anchors = inp.anchor()
    assert anchors.tolist() == [0, 1]
    torch.testing.assert_close(captured[0]["stage"], anchors, atol=0, rtol=0)
    torch.testing.assert_close(captured[0]["image"], inp.images[torch.arange(2), anchors], atol=0, rtol=0)


def test_joint_training_batches_group_actual_stages_with_delayed_availability(cohort):
    manifest = json.loads(cohort.read_text())
    for i, patient in enumerate(manifest["patients"][:4]):
        patient["visits"][2]["available_at"] = 3
        patient["visits"][1]["available_at"] = 2 if i < 2 else 3
    cohort.write_text(json.dumps(manifest), encoding="utf-8")
    cfg = joint_config()
    cfg.train.batch_size = 3
    store = Store(cohort, cfg.train.image_shape, allow_synthetic=True)
    tasks = [(i, 2) for i in store.splits["train"]]
    anchors = [store.observed_stage(i, landmark) for i, landmark in tasks]
    assert anchors[:4] == [1, 1, 0, 0]
    seen = []
    for batch in make_loader(store, cfg, tasks, training=True, world=True):
        assert batch.inputs.anchor().unique().numel() == 1
        seen.extend(batch.keys)
    assert len(seen) == len(set(seen)) == 8


def test_joint_dynamic_keeps_terminal_prefixes_and_patient_equal_validation(cohort):
    cfg = joint_config()
    store = Store(cohort, cfg.train.image_shape, allow_synthetic=True)
    validation = planned_tasks(store, cfg, "val")
    assert len(validation) == 16
    assert {landmark for _, landmark in validation} == {0, 1, 2, 3}
    # World-only B excludes terminal prefixes, whereas joint keeps their pCR supervision.
    training = planned_tasks(store, cfg, "train", epoch=0)
    assert len(training) == len(store.splits["train"])
    assert len({i for i, _ in training}) == len(training)
    model = ResponseJEPA(cfg.model)
    model.configure_trainable("joint", False)
    model.world_trained = True
    metrics, rows, world = evaluate(model, store, cfg, "val")
    assert metrics["selection_name"] == "patient_equal_pcr_nll"
    fused = metrics["classification"]["fused"]["patient_equal"]
    assert fused["patients"] == 4
    assert metrics["selection"] == fused["nll"]
    assert len(rows) == 16 and len(world) > 0


def test_joint_teacher_updates_once_per_optimizer_step(cohort, tmp_path, monkeypatch):
    cfg = joint_config("t0")
    cfg.train.batch_size = 3
    cfg.train.accumulation = 2
    calls = []
    original = ResponseJEPA.update_teacher

    def tracked(model, decay):
        calls.append(decay)
        return original(model, decay)

    monkeypatch.setattr(ResponseJEPA, "update_teacher", tracked)
    train(cfg, cohort, tmp_path / "joint_updates")
    state = load_checkpoint(tmp_path / "joint_updates" / "last.pt")
    expected = cfg.train.epochs * math.ceil(math.ceil(8 / 3) / cfg.train.accumulation)
    assert state["updates"] == len(calls) == expected
    assert calls == [cfg.train.ema] * expected


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for av, bv in zip(a, b):
            assert_nested_equal(av, bv)
    else:
        assert a == b


def test_exact_cpu_joint_resume_with_dropout_and_partial_accumulation(cohort, tmp_path):
    cfg = joint_config("t0")
    cfg.train.epochs = 3
    cfg.train.patience = 10
    cfg.train.warmup_epochs = 1
    cfg.train.batch_size = 3
    cfg.train.accumulation = 2
    cfg.model.dropout = 0.1
    train(cfg, cohort, tmp_path / "joint_full")
    train(cfg, cohort, tmp_path / "joint_resumed", stop_after_epoch=1)
    train(cfg, cohort, tmp_path / "joint_resumed", resume=True)
    full = load_checkpoint(tmp_path / "joint_full" / "last.pt")
    resumed = load_checkpoint(tmp_path / "joint_resumed" / "last.pt")
    assert full["epoch"] == resumed["epoch"] == 3
    assert full["updates"] == resumed["updates"]
    assert full["best_score"] == resumed["best_score"]
    assert_nested_equal(full["model"], resumed["model"])
    assert_nested_equal(full["optimizer"], resumed["optimizer"])
    assert "target_coordinate_origin" in resumed
    model, store, saved_cfg, state = load_for_inference(tmp_path / "joint_resumed" / "best.pt",
                                                      cohort, allow_synthetic=True)
    metrics, rows, world = evaluate(model, store, saved_cfg, "val")
    assert saved_cfg.train.phase == "joint"
    assert state["world_trained"] and len(rows) == 4 and len(world) > 0
    assert metrics["selection_name"] == "patient_equal_pcr_nll"
    assert model.encoder_frozen and not model.target_encoder.training


@pytest.mark.parametrize("change", ["frozen", "single_patient", "no_mri", "no_world", "no_representation"])
def test_joint_invalid_training_contracts_rejected(change):
    cfg = joint_config()
    if change == "frozen":
        cfg.train.freeze_encoder = True
    elif change == "single_patient":
        cfg.train.batch_size = 1
    elif change == "no_mri":
        cfg.model.use_mri = False
    elif change == "no_world":
        cfg.loss.tf = cfg.loss.rollout = 0.0
    else:
        cfg.loss.masked = cfg.loss.variance = cfg.loss.covariance = cfg.loss.token_variance = cfg.loss.repr_pcr = 0.0
    with pytest.raises(ValueError):
        cfg.validate()


def test_joint_rejects_staged_warm_start(cohort, tmp_path):
    cfg = joint_config()
    with pytest.raises(ValueError, match="(?i)joint"):
        train(cfg, cohort, tmp_path / "joint_bad_init", init=tmp_path / "unused.pt")
