from dataclasses import replace
import torch
import pytest
from response_jepa.contracts import Batch
from response_jepa.objectives import task_world_loss, representation_loss, variance_covariance
from response_jepa.model import ResponseJEPA


def test_output_requests_cannot_change_pcr(model, batch):
    all_out = model(batch.inputs, request_states=True)
    last = torch.zeros_like(batch.inputs.query_mask)
    last[:, 3] = True
    last_out = model(replace(batch.inputs, query_mask=last), request_states=True)
    none = model(replace(batch.inputs, query_mask=torch.zeros_like(last)), request_states=True)
    assert torch.equal(all_out["logit"], last_out["logit"])
    assert torch.equal(all_out["logit"], none["logit"])
    assert torch.equal(all_out["states"], last_out["states"])
    assert not none["requested_states"].count_nonzero()


def test_nonzero_future_input_rejected(model, batch):
    x = batch.inputs.images.clone()
    x[0, 3] = 1
    with pytest.raises(ValueError, match="Hidden"):
        model(replace(batch.inputs, images=x))


def test_future_supervision_does_not_change_pcr(model, batch, config):
    _, _, a = task_world_loss(model, batch, config)
    changed = replace(batch.targets, future=batch.targets.future + 30)
    _, _, b = task_world_loss(model, Batch(batch.inputs, changed, batch.keys), config)
    assert torch.equal(a["logit"], b["logit"])
    assert torch.equal(a["states"], b["states"])


def test_teacher_receives_no_grad(model, batch, config):
    future = batch.targets.future.clone().requires_grad_()
    sup = replace(batch.targets, future=future)
    loss, _, _ = task_world_loss(model, Batch(batch.inputs, sup, batch.keys), config)
    loss.backward()
    assert future.grad is None
    assert all(p.grad is None and not p.requires_grad for p in model.target_encoder.parameters())


def test_zero_residual_starts_at_copy(model, batch):
    out = model(batch.inputs, need_world=True)
    ar = torch.arange(2)
    initial = out["states"][ar, batch.inputs.anchor()]
    for i in range(2):
        for j in range(int(batch.inputs.anchor()[i]), 4):
            torch.testing.assert_close(out["states"][i, j], initial[i], atol=0, rtol=0)


def test_temporal_causality_in_teacher_forcing(model):
    states = torch.randn(2, 3, 4, 32)
    transitions = torch.randn(2, 3, 2, 32)
    active = torch.ones(2, 3, dtype=torch.bool)
    # Test nontrivial predictions rather than relying on zero initialized output.
    torch.nn.init.normal_(model.world.output.weight, std=.02)
    a = model.world(states, transitions, active)
    altered = states.clone()
    altered[:, 1:] += 100
    b = model.world(altered, transitions, active)
    torch.testing.assert_close(a[:, 0], b[:, 0], atol=1e-6, rtol=1e-6)


def test_inactive_world_nan_is_isolated(model):
    s = torch.randn(2, 3, 4, 32)
    t = torch.randn(2, 3, 2, 32)
    active = torch.tensor([[True, False, False], [False, False, True]])
    s[~active] = float("nan")
    t[~active] = float("nan")
    out = model.world(s, t, active)
    assert torch.isfinite(out).all()


def test_outcome_gradient_reaches_free_predictor(model, batch, config):
    config.loss.tf = config.loss.rollout = 0
    loss, _, _ = task_world_loss(model, batch, config)
    loss.backward()
    assert model.world.output.weight.grad is not None
    assert model.world.output.weight.grad.norm() > 0


def test_detach_control_blocks_outcome_gradient_to_world(batch, config):
    config.loss.tf = config.loss.rollout = 0
    config.model.forecast_detach = True
    m = ResponseJEPA(config.model)
    m.configure_trainable("C", True)
    m.eval()
    loss, _, _ = task_world_loss(m, batch, config)
    loss.backward()
    g = m.world.output.weight.grad
    assert g is None or g.count_nonzero() == 0


def test_missing_intermediate_still_supervises_later_target(model, batch, config):
    observed = batch.inputs.observed.clone()
    observed[1, 1] = False
    image = batch.inputs.images.masked_fill(~observed[:, :, None, None, None, None], 0)
    inp = replace(batch.inputs, images=image, observed=observed, landmark=torch.zeros(2, dtype=torch.long),
                  query_mask=torch.tensor([[False, True, True, True]]*2))
    future_mask = torch.tensor([[False, False, True, False]]*2)
    sup = replace(batch.targets, future_mask=future_mask)
    loss, terms, out = task_world_loss(model, Batch(inp, sup, batch.keys), config)
    assert terms["rollout"] > 0
    assert terms["tf"] == 0
    assert torch.isfinite(out["states"]).all()
    loss.backward()
    assert model.world.output.weight.grad.norm() > 0


def test_t3_has_zero_world_loss(model, batch, config):
    obs = torch.ones_like(batch.inputs.observed)
    inp = replace(batch.inputs, images=batch.targets.future, observed=obs,
                  landmark=torch.full((2,), 3, dtype=torch.long), query_mask=torch.zeros_like(obs))
    sup = replace(batch.targets, future_mask=torch.zeros_like(obs))
    loss, terms, out = task_world_loss(model, Batch(inp, sup, batch.keys), config)
    assert terms["tf"] == terms["rollout"] == 0
    assert torch.isfinite(loss)


def test_teacher_stays_eval(model):
    model.train()
    assert not model.target_encoder.training
    assert not model.encoder.training


def test_patient_axis_regularizer_detects_spatial_only_variation():
    # Every patient has exactly the same pooled representation, despite spatial variation.
    x = torch.randn(1, 8, 32).expand(4, -1, -1)
    var, cov = variance_covariance(x.mean(1))
    assert var > .9
    assert cov == 0


def test_clinical_ablation_ignores_mri_and_arm(batch, config):
    config.model.use_mri = False
    config.model.use_arm = False
    config.model.readout = "direct"
    m = ResponseJEPA(config.model).eval()
    out = m(batch.inputs)["logit"]
    changed = replace(batch.inputs, images=batch.inputs.images*100, arm_id=torch.tensor([13, 12]))
    assert torch.equal(out, m(changed)["logit"])


def test_same_visit_pretraining_backward(config, batch):
    config.train.phase = "A"
    m = ResponseJEPA(config.model)
    m.configure_trainable("A", False)
    data = {"image": batch.inputs.images[:, 0], "stage": torch.zeros(2, dtype=torch.long),
            "label": batch.targets.label, "label_mask": batch.targets.label_mask}
    loss, terms = representation_loss(m, data, config)
    assert terms["regularized_stages"] == 1
    loss.backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in m.encoder.parameters())
    assert all(p.grad is None for p in m.world.parameters())


def test_copy_readout_preserves_the_learned_world_objective(batch, config):
    from copy import deepcopy
    full = ResponseJEPA(config.model).eval()
    torch.nn.init.normal_(full.world.output.weight, std=.02)
    copycfg = deepcopy(config)
    copycfg.model.readout = "copy_gated"
    cp = ResponseJEPA(copycfg.model).eval()
    cp.load_state_dict(full.state_dict())
    _, tfull, ofull = task_world_loss(full, batch, config)
    _, tcopy, ocopy = task_world_loss(cp, batch, copycfg)
    assert torch.equal(ofull["states"], ocopy["states"])
    assert torch.equal(tfull["tf"], tcopy["tf"])
    assert torch.equal(tfull["rollout"], tcopy["rollout"])
    assert not torch.equal(ofull["forecast_logit"], ocopy["forecast_logit"])


def test_linear_world_control_is_differentiable(batch, config):
    config.model.world_arch = "linear"
    m = ResponseJEPA(config.model).eval()
    loss, _, out = task_world_loss(m, batch, config)
    assert out["states"].shape == (2, 4, 4, 32)
    loss.backward()
    assert m.world.output.weight.grad.norm() > 0


def test_absolute_world_control_is_differentiable(batch, config):
    config.model.residual_world = False
    m = ResponseJEPA(config.model).eval()
    loss, _, out = task_world_loss(m, batch, config)
    loss.backward()
    assert torch.isfinite(out["states"]).all()
    assert m.world.output.weight.grad.norm() > 0


def test_covariance_really_stays_fp32_under_autocast():
    x = torch.randn(8, 32)
    va, co = variance_covariance(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        va_amp, co_amp = variance_covariance(x)
    assert va_amp.dtype == co_amp.dtype == torch.float32
    torch.testing.assert_close(va, va_amp, atol=0, rtol=0)
    torch.testing.assert_close(co, co_amp, atol=0, rtol=0)
