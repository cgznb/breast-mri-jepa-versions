"""Dataset-risk weights, stochastic consistency and legal-input gradient paths."""
from copy import deepcopy
from dataclasses import fields, replace
import math

import pytest
import torch
from torch.nn import functional as F

from response_jepa.contracts import Batch
from response_jepa.model import ResponseJEPA
from response_jepa.objectives import (additive_average, bce, joint_loss, patient_average,
                                      representation_loss, symmetric_bernoulli_kl,
                                      task_world_loss)


def weighted_config(config, reference=16):
    config.train.sample_weighting = "patient_equal"
    config.train.reference_batch_size = reference
    return config


def weighted_batch(batch, weights=(.5, 1.5)):
    result = deepcopy(batch)
    result.sample_weight = torch.tensor(weights)
    return result


def test_complete_epoch_patient_risk_and_tail_gradients_are_equal():
    # Patient A has four identical prefixes, B has two. Despite stage batching
    # and a one-item tail, each patient must get half the dataset objective.
    prefix_patient = torch.tensor([0, 0, 1, 0, 1, 0])
    losses = torch.tensor([2., 5.], requires_grad=True)
    weights = torch.tensor([.75, .75, 1.5, .75, 1.5, .75])
    batches = ([0, 1, 2], [3, 4], [5])
    reduced = sum(additive_average(losses[prefix_patient[ix]], weights[ix], 3) for ix in batches)
    # Sum of updates is M / batch_ref times the equal-patient risk.
    torch.testing.assert_close(reduced, torch.tensor(7.))
    reduced.backward()
    torch.testing.assert_close(losses.grad, torch.ones(2))


def test_reference_denominator_never_self_normalizes_batch_weights():
    value = torch.tensor([2., 5.], requires_grad=True)
    full = additive_average(value, torch.tensor([.5, 1.5]), 16)
    split = (additive_average(value[:1], torch.tensor([.5]), 16)
             + additive_average(value[1:], torch.tensor([1.5]), 16))
    torch.testing.assert_close(full, split)
    full.backward()
    torch.testing.assert_close(value.grad, torch.tensor([.5, 1.5]) / 16)
    # A single tail sample keeps its fixed coefficient, rather than becoming
    # an entire one-patient batch mean.
    assert additive_average(torch.tensor([5.]), torch.tensor([1.5]), 16) == 7.5 / 16


def test_no_reference_preserves_original_mean_and_fixed_t0_unit_weights():
    values = torch.tensor([1., 2., 5.])
    torch.testing.assert_close(additive_average(values), values.mean(), atol=0, rtol=0)
    torch.testing.assert_close(additive_average(values, torch.ones(3)), values.mean(), atol=0, rtol=0)


@pytest.mark.parametrize("weights", [torch.tensor([1.]), torch.tensor([1., -1.]),
                                      torch.tensor([1., float("nan")])])
def test_invalid_dataset_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="weights"):
        additive_average(torch.tensor([1., 2.]), weights, 16)


def test_future_target_count_and_no_future_do_not_change_patient_denominator():
    errors = torch.tensor([[2., 4., 100.], [9., 100., 100.], [100., 100., 100.]])
    mask = torch.tensor([[True, True, False], [True, False, False], [False, False, False]])
    loss, values, counts = patient_average(errors, mask, sample_weight=torch.tensor([.5, 1., 4.]),
                                          reference_batch_size=16)
    torch.testing.assert_close(values, torch.tensor([3., 9., 0.]))
    assert counts.tolist() == [2, 1, 0]
    assert loss == 10.5 / 16


def test_missing_labels_contribute_zero_without_renormalizing():
    logits = torch.zeros(3, requires_grad=True)
    loss = bce(logits, torch.tensor([0., 1., 0.]), torch.tensor([True, False, True]),
               sample_weight=torch.tensor([.5, 100., 1.5]), reference_batch_size=16)
    torch.testing.assert_close(loss, torch.tensor(2 * math.log(2) / 16))
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([.25, 0., .75]) / 16)


def test_patient_equal_requires_dataset_weights(model, batch, config):
    weighted_config(config)
    with pytest.raises(ValueError, match="fixed dataset sample weights"):
        task_world_loss(model, batch, config)


def test_task_terms_use_patient_weights_before_reference_reduction(model, batch, config):
    # Evaluate each patient independently to recover their complete trajectory
    # and classification risk, then verify the joint physical batch preserves it.
    original = deepcopy(config)
    single_terms = []
    for row in range(2):
        inp = replace(batch.inputs, **{field.name: getattr(batch.inputs, field.name)[row:row+1]
                                      for field in fields(batch.inputs)})
        target = replace(batch.targets, **{field.name: getattr(batch.targets, field.name)[row:row+1]
                                          for field in fields(batch.targets)})
        _, terms, _ = task_world_loss(model, Batch(inp, target, (batch.keys[row],)), original)
        single_terms.append(terms)
    weighted_config(config, reference=4)
    _, terms, _ = task_world_loss(model, weighted_batch(batch), config)
    for name in ("tf", "rollout", "direct", "forecast", "fused", "align"):
        expected = (.5 * single_terms[0][name] + 1.5 * single_terms[1][name]) / 4
        torch.testing.assert_close(terms[name], expected, atol=1e-6, rtol=1e-5)


def test_masked_weights_leave_batch_regularizers_unchanged(model, batch, config):
    weighted_config(config, reference=2)
    data = {"image": batch.inputs.images[:, 0], "stage": torch.zeros(2, dtype=torch.long),
            "label": batch.targets.label, "label_mask": batch.targets.label_mask,
            "sample_weight": torch.ones(2)}
    torch.manual_seed(32)
    _, first = representation_loss(model, data, config, augment=False)
    data["sample_weight"] = torch.full((2,), 3.)
    torch.manual_seed(32)
    _, heavier = representation_loss(model, data, config, augment=False)
    torch.testing.assert_close(heavier["masked"], 3 * first["masked"])
    for name in ("variance", "covariance", "token_variance", "actual_mask_fraction"):
        torch.testing.assert_close(first[name], heavier[name], atol=0, rtol=0)


def test_symmetric_kl_matches_bernoulli_definition_and_has_both_gradients():
    first = torch.tensor([-1.2, 0., 2.], requires_grad=True)
    second = torch.tensor([.3, 1., -1.], requires_grad=True)
    p, q = first.detach().double().sigmoid(), second.detach().double().sigmoid()
    kl_pq = p * (p / q).log() + (1-p) * ((1-p) / (1-q)).log()
    kl_qp = q * (q / p).log() + (1-q) * ((1-q) / (1-p)).log()
    expected = .5 * (kl_pq + kl_qp)
    actual = symmetric_bernoulli_kl(first, second)
    torch.testing.assert_close(actual.double(), expected, atol=1e-7, rtol=1e-6)
    actual.sum().backward()
    assert first.grad.abs().min() > 0
    assert second.grad.abs().min() > 0
    torch.testing.assert_close(symmetric_bernoulli_kl(first, first), torch.zeros(3), atol=0, rtol=0)


def test_extreme_kl_is_finite_in_fp32_under_autocast():
    first = torch.tensor([-10000., 10000., -10000., 10000.], requires_grad=True)
    second = torch.tensor([10000., -10000., -10000., 10000.], requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        values = symmetric_bernoulli_kl(first, second)
    assert values.dtype == torch.float32
    assert values.tolist() == [10000., 10000., 0., 0.]
    values.sum().backward()
    assert torch.isfinite(first.grad).all() and torch.isfinite(second.grad).all()


def test_rdrop_averages_two_classifiers_and_computes_world_once(model, batch, config, monkeypatch):
    config.loss.consistency = .1
    config.loss.consistency_warmup_updates = 500
    model.train()
    forwards, teachers = [], []
    original_forward, original_teacher = model.forward, model.teacher_forcing

    def forward(inp, **kwargs):
        out = original_forward(inp, **kwargs)
        forwards.append((inp, out))
        return out

    def teacher(*args, **kwargs):
        teachers.append(1)
        return original_teacher(*args, **kwargs)

    monkeypatch.setattr(model, "forward", forward)
    monkeypatch.setattr(model, "teacher_forcing", teacher)
    loss, terms, out = task_world_loss(model, batch, config, updates=250)
    assert len(forwards) == 2 and teachers == [1]
    assert forwards[0][0] is forwards[1][0] is batch.inputs
    assert out is forwards[0][1]
    for term, logit in (("fused", "logit"), ("direct", "direct_logit"), ("forecast", "forecast_logit")):
        expected = sum(F.binary_cross_entropy_with_logits(view[logit].float(), batch.targets.label)
                       for _, view in forwards) / 2
        torch.testing.assert_close(terms[term], expected)
    assert terms["consistency_weight"] == .05
    expected = sum(getattr(config.loss, name) * terms[name]
                   for name in ("tf", "rollout", "direct", "forecast", "fused", "align"))
    torch.testing.assert_close(loss, expected + .05 * terms["consistency"])
    for _, view in forwards:
        view["direct_logit"].retain_grad()
    loss.backward()
    assert all(view["direct_logit"].grad.abs().sum() > 0 for _, view in forwards)


def test_zero_warmup_immediately_enables_consistency(model, batch, config):
    config.loss.consistency = .1
    config.loss.consistency_warmup_updates = 0
    _, terms, _ = task_world_loss(model, batch, config, updates=0)
    assert terms["consistency_weight"] == .1


def test_consistency_joint_computes_representation_once(model, batch, config, monkeypatch):
    import response_jepa.objectives as objectives
    config.train.phase = "joint"
    config.loss.consistency = .1
    model.configure_trainable("joint", False)
    calls = []
    original = objectives.representation_loss

    def representation(m, data, cfg, **kwargs):
        calls.append(data)
        return original(m, data, cfg, **kwargs)

    monkeypatch.setattr(objectives, "representation_loss", representation)
    loss, terms, _ = joint_loss(model, batch, config, augment=False, updates=1)
    assert len(calls) == 1 and terms["masked"] > 0
    assert torch.isfinite(loss)


def test_both_consistency_views_are_independent_of_future_supervision(model, batch, config, monkeypatch):
    config.loss.consistency = .1
    model.train()
    captures = []
    original = model.forward

    def forward(inp, **kwargs):
        out = original(inp, **kwargs)
        captures.append(out["logit"].detach().clone())
        return out

    monkeypatch.setattr(model, "forward", forward)
    torch.manual_seed(222)
    task_world_loss(model, batch, config, updates=500)
    changed = Batch(batch.inputs, replace(batch.targets, future=batch.targets.future + 50), batch.keys)
    torch.manual_seed(222)
    task_world_loss(model, changed, config, updates=500)
    assert len(captures) == 4
    for first, second in zip(captures[:2], captures[2:]):
        torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_detach_blocks_two_view_classification_world_path_but_keeps_shared_encoder(batch, config):
    config.model.forecast_detach = True
    config.loss.tf = config.loss.rollout = config.loss.align = 0
    config.loss.consistency = .1
    model = ResponseJEPA(config.model)
    model.configure_trainable("joint", False)
    loss, _, _ = task_world_loss(model, batch, config, updates=500)
    loss.backward()
    assert all(p.grad is None or not p.grad.count_nonzero() for p in model.world.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.terminal_head.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())
