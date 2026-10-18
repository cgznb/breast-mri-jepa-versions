"""Request-independent classification, legal rollout and real gradient routes."""
from dataclasses import replace

import pytest
import torch

from mri_vla_jepa.contracts import RawMRIInput, RawMRISupervision
from mri_vla_jepa.model import RawVLAJEPA, TimeCausalWorldPredictor
from mri_vla_jepa.model_config import RawVLAJEPAConfig


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(314)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def cfg():
    return RawVLAJEPAConfig(dim=24, token_grid=(1, 2, 2), heads=4,
                            encoder_depth=1, fusion_depth=1, predictor_depth=1,
                            state_queries=2, pcr_queries=2, fixed_transition_queries=True)


@pytest.fixture
def batch():
    observed = torch.tensor([[True, False, False, False], [True, True, False, False]])
    images = torch.zeros(2, 4, 3, 4, 8, 8)
    images[observed] = torch.randn(3, 3, 4, 8, 8)
    queries = torch.tensor([[False, True, True, True], [False, False, True, True]])
    inp = RawMRIInput(images, observed, torch.tensor([0, 1]), torch.randn(2, 4),
                      torch.ones(2, 4, dtype=torch.bool), torch.tensor([1, 13]),
                      torch.ones(2, dtype=torch.bool), torch.tensor([0, 0]), queries)
    future = torch.full_like(images, float("nan"))
    future[queries] = torch.randn(int(queries.sum()), 3, 4, 8, 8)
    sup = RawMRISupervision(future, queries.clone(), torch.tensor([1., 0.]),
                            torch.ones(2, dtype=torch.bool))
    return inp, sup


def has_grad(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


@pytest.mark.parametrize("readout", ["direct", "copy", "forecast_detach", "forecast"])
def test_output_request_does_not_change_classifier_or_internal_slots(cfg, batch, readout):
    model = RawVLAJEPA(replace(cfg, pcr_readout=readout)).eval()
    inp, _ = batch
    complete = model.forecast(inp)
    sparse_mask = torch.zeros_like(inp.query_mask)
    sparse_mask[:, 3] = True
    sparse = model.forecast(replace(inp, query_mask=sparse_mask))
    empty = model.forecast(replace(inp, query_mask=torch.zeros_like(inp.query_mask)))
    for alternate in (sparse, empty):
        torch.testing.assert_close(alternate["pcr_logit"], complete["pcr_logit"], atol=0, rtol=0)
        torch.testing.assert_close(alternate["dynamics_features"], complete["dynamics_features"], atol=0, rtol=0)
    torch.testing.assert_close(sparse["state_prediction"][:, 3], complete["state_prediction"][:, 3], atol=0, rtol=0)
    assert sparse["state_prediction"][:, :3].count_nonzero() == 0
    assert empty["state_prediction"].count_nonzero() == 0
    assert complete["dynamics_features"][:, 0].count_nonzero() == 0
    assert complete["dynamics_features"][:, 1:].count_nonzero() > 0


def test_readout_capacities_and_detachment_are_matched(cfg, batch):
    models = {name: RawVLAJEPA(replace(cfg, pcr_readout=name)).eval()
              for name in ("copy", "forecast_detach", "forecast")}
    reference = models["forecast"].state_dict()
    for model in models.values():
        model.load_state_dict(reference)
    assert len({sum(p.numel() for p in model.pcr_head.parameters()) for model in models.values()}) == 1
    inp, sup = batch
    detach_output = models["forecast_detach"](inp)
    forecast_output = models["forecast"](inp)
    torch.testing.assert_close(detach_output["pcr_logit"], forecast_output["pcr_logit"], atol=0, rtol=0)
    for name, model in models.items():
        loss, terms = model.compute_loss(inp, sup, jepa_weight=0, rollout_weight=0,
                                         flow_weight=0, reconstruction_weight=0, variance_weight=0)
        loss.backward()
        assert has_grad(model.pcr_head) and has_grad(model.fusion) and has_grad(model.encoder)
        assert has_grad(model.world_predictor) == (name == "forecast")
        assert all(p.grad is None and not p.requires_grad for p in model.teacher_encoder.parameters())
        assert terms["rollout"] == terms["rollout_count"] == 0


@pytest.mark.parametrize("readout", ["forecast_detach", "forecast"])
def test_pcr_rollout_never_accepts_teacher_forced_targets(cfg, batch, readout, monkeypatch):
    model = RawVLAJEPA(replace(cfg, pcr_readout=readout, history_delta=True)).eval()
    inp, sup = batch
    def forbidden(*args, **kwargs):
        raise AssertionError("Classifier must not invoke future-target teacher encoding")
    monkeypatch.setattr(model, "_teacher_states", forbidden)
    before = model(inp)
    modified = replace(sup, future=sup.future * 10000)
    _, terms = model.compute_loss(inp, modified, jepa_weight=0, rollout_weight=0,
                                  flow_weight=0, reconstruction_weight=0, variance_weight=0)
    after = model(inp)
    torch.testing.assert_close(before["pcr_logit"], after["pcr_logit"], atol=0, rtol=0)
    assert torch.isfinite(terms["task"])


def test_terminal_current_is_copy_and_needs_no_world(cfg, batch, monkeypatch):
    inp, _ = batch
    images = torch.randn_like(inp.images)
    terminal = replace(inp, images=images, observed_mask=torch.ones_like(inp.observed_mask),
                       landmark=torch.tensor([3, 3]), query_mask=torch.zeros_like(inp.query_mask))
    copy = RawVLAJEPA(replace(cfg, pcr_readout="copy")).eval()
    forecast = RawVLAJEPA(replace(cfg, pcr_readout="forecast")).eval()
    forecast.load_state_dict(copy.state_dict())
    def forbidden(*args, **kwargs):
        raise AssertionError("A real T3 state needs no transition")
    monkeypatch.setattr(forecast.world_predictor, "forward", forbidden)
    output = forecast(terminal)
    torch.testing.assert_close(output["pcr_logit"], copy(terminal)["pcr_logit"], atol=0, rtol=0)
    assert output["pcr_state_features"][:, -cfg.dim:].count_nonzero() == 0


def test_missing_landmark_rollout_starts_at_actual_latest_observation(cfg, batch, monkeypatch):
    inp, _ = batch
    # Requesting a T3 classifier does not invent a T3 scan when it is missing.
    missing = replace(inp, landmark=torch.tensor([3, 3]), query_mask=torch.zeros_like(inp.query_mask))
    model = RawVLAJEPA(replace(cfg, pcr_readout="forecast")).eval()
    calls = []
    original = model.world_predictor.forward
    def record(states, transitions, source_mask, **kwargs):
        calls.append((states.detach().clone(), source_mask.clone()))
        return original(states, transitions, source_mask, **kwargs)
    monkeypatch.setattr(model.world_predictor, "forward", record)
    output = model(missing)
    assert len(calls) == 3
    assert torch.equal(calls[0][1], torch.tensor([[True, False, False], [False, False, False]]))
    assert torch.equal(calls[1][1], torch.tensor([[True, True, False], [False, True, False]]))
    with torch.no_grad():
        latest_state = model.teacher_encoder(inp.images[inp.observed_mask])[-1]
    torch.testing.assert_close(calls[1][0][1, 1], latest_state, atol=0, rtol=0)
    assert torch.isfinite(output["pcr_logit"]).all()
    assert output["pcr_state_features"][:, cfg.dim:2 * cfg.dim].abs().sum() > 0


def test_zero_initialized_residual_is_exact_copy_including_multistep(cfg, batch):
    model = RawVLAJEPA(replace(cfg, world_residual=True, history_delta=True)).eval()
    inp, _ = batch
    states, _ = model._observed_teacher_states(inp)
    predicted = model.forecast_states(inp)
    for row, latest in enumerate(inp.landmark.tolist()):
        for target in range(latest + 1, 4):
            torch.testing.assert_close(predicted[row, target], states[row, latest], atol=0, rtol=0)


def test_history_latest_predecessor_and_missing_validity(cfg):
    model = RawVLAJEPA(replace(cfg, history_delta=True))
    states = torch.randn(2, 4, 4, cfg.dim)
    known = torch.tensor([[True, False, True, False], [False, True, False, False]])
    history, valid = model._source_history(states, known)
    assert torch.equal(valid, torch.tensor([[False, True, True], [False, False, True]]))
    torch.testing.assert_close(history[0, 2], states[0, 0], atol=0, rtol=0)
    torch.testing.assert_close(history[1, 2], states[1, 1], atol=0, rtol=0)
    assert history[~valid].count_nonzero() == 0


def test_history_invalid_nan_isolation_and_block_causality(cfg):
    world = TimeCausalWorldPredictor(replace(cfg, history_delta=True)).eval()
    source = torch.randn(2, 3, 4, cfg.dim)
    transition = torch.randn(2, 3, 2, cfg.dim)
    active = torch.tensor([[True, True, True], [False, True, True]])
    history = torch.randn_like(source)
    valid = torch.tensor([[False, True, True], [False, False, True]])
    output = world(source, transition, active, history_states=history, history_mask=valid)
    poisoned = history.clone()
    poisoned[~valid] = float("nan")
    isolated = world(source, transition, active, history_states=poisoned, history_mask=valid)
    torch.testing.assert_close(output, isolated, atol=0, rtol=0)
    changed = history.clone()
    changed[:, 2] += 100 * torch.randn_like(changed[:, 2])
    modified = world(source, transition, active, history_states=changed, history_mask=valid)
    torch.testing.assert_close(output[:, :2], modified[:, :2], atol=0, rtol=0)
    assert not torch.allclose(output[:, 2], modified[:, 2])


def test_autonomous_supervision_is_patient_equal_and_differentiable(cfg, batch):
    inp, sup = batch
    model = RawVLAJEPA(replace(cfg, history_delta=True)).eval()
    dynamic = model(inp)["dynamics_features"]
    loss, detail = model.autonomous_rollout_loss(inp, sup, dynamic, return_details=True)
    expected = torch.stack((detail["target_losses"][0, 1:].mean(),
                            detail["target_losses"][1, 2:].mean())).mean()
    torch.testing.assert_close(loss, expected)
    assert detail["patient_count"] == 2
    total, terms = model.compute_loss(inp, sup, task_weight=0, jepa_weight=0, rollout_weight=.7,
                                     flow_weight=0, reconstruction_weight=0, variance_weight=0)
    torch.testing.assert_close(total, terms["rollout"] * .7)
    assert terms["rollout_count"] == 5 and terms["pair_count"] == 0
    total.backward()
    assert has_grad(model.world_predictor) and has_grad(model.fusion)
    assert not has_grad(model.pcr_head)
    assert all(p.grad is None for p in model.teacher_encoder.parameters())


def test_detached_classifier_still_trains_world_with_independent_free_loss(cfg, batch):
    model = RawVLAJEPA(replace(cfg, pcr_readout="forecast_detach", history_delta=True)).eval()
    inp, sup = batch
    loss, _ = model.compute_loss(inp, sup, jepa_weight=0, rollout_weight=.5,
                                 flow_weight=0, reconstruction_weight=0, variance_weight=0)
    loss.backward()
    assert has_grad(model.world_predictor) and has_grad(model.pcr_head)


def test_missing_rollout_targets_reduce_to_finite_zero(cfg, batch):
    inp, sup = batch
    sup = replace(sup, future_mask=torch.zeros_like(sup.future_mask),
                  future=torch.full_like(sup.future, float("nan")))
    model = RawVLAJEPA(replace(cfg, history_delta=True)).eval()
    loss, terms = model.compute_loss(inp, sup, task_weight=0, jepa_weight=0, rollout_weight=1,
                                     flow_weight=0, reconstruction_weight=0, variance_weight=0)
    assert loss.requires_grad and loss == terms["rollout"] == terms["rollout_count"] == 0
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_lesion_loss_weights_regions_without_rescaling_uniform_errors(cfg, batch):
    inp, _ = batch
    mask = torch.zeros(2, 1, 4, 8, 8)
    mask[:, :, :, :4, :4] = 1
    inp = replace(inp, lesion_weights=mask)
    model = RawVLAJEPA(replace(cfg, lesion_world_weighting=True)).eval()
    target = torch.zeros(2, 3, 4, cfg.dim)
    prediction = torch.arange(4.).reshape(1, 1, 4, 1).expand_as(target)
    loss = model._state_l1(prediction, target, inp)
    torch.testing.assert_close(loss, torch.full((2, 3), .75))
    torch.testing.assert_close(model._state_l1(torch.ones_like(target), target, inp),
                               torch.ones(2, 3))
    with pytest.raises(ValueError, match="lesion_weights"):
        model._state_l1(prediction, target, replace(inp, lesion_weights=None))


@pytest.mark.parametrize("name,value", [("fixed_transition_queries", 1), ("history_delta", 1),
    ("world_residual", 1), ("pcr_readout", "unknown"), ("lesion_weight_strength", float("nan"))])
def test_evolution_configuration_rejects_invalid_flags(cfg, name, value):
    with pytest.raises(ValueError):
        replace(cfg, **{name: value}).validate()


def test_forecast_readout_requires_request_independent_slots(cfg):
    with pytest.raises(ValueError, match="fixed_transition_queries"):
        replace(cfg, fixed_transition_queries=False, pcr_readout="forecast").validate()
