"""Shared parallel TF training and observed-only public prediction contracts."""
from dataclasses import replace

import pytest
import torch

from mri_vla_jepa.contracts import RawMRIInput, RawMRISupervision
from mri_vla_jepa.model import RawVLAJEPA
from mri_vla_jepa.model_config import RawVLAJEPAConfig
from mri_vla_jepa.data import RawMRIStore, make_raw_synthetic
from mri_vla_jepa.diagnostics import make_probe_tasks, run_training_probe, weighted_gradient_metrics
from mri_vla_jepa.train_config import RawVLATrainConfig


@pytest.fixture(autouse=True)
def tiny_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(719)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def cfg():
    return RawVLAJEPAConfig(dim=16, heads=2, token_grid=(1, 2, 2),
        encoder_depth=1, fusion_depth=1, predictor_depth=1, state_queries=2,
        pcr_queries=2, fm_channels=4, fixed_transition_queries=True,
        pcr_readout="forecast", history_delta=True, dropout=0.)


@pytest.fixture
def batch():
    landmark = torch.tensor([0, 0, 0, 3, 2, 1])
    observed = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                             [1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 0, 0]], dtype=torch.bool)
    future_mask = torch.tensor([[0, 1, 1, 1], [0, 1, 0, 0], [0, 0, 1, 0],
                                [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.bool)
    images = torch.zeros(6, 4, 3, 4, 8, 8)
    images[observed] = torch.randn(int(observed.sum()), 3, 4, 8, 8)
    future = torch.full_like(images, float("nan"))
    future[future_mask] = torch.randn(int(future_mask.sum()), 3, 4, 8, 8)
    query = torch.arange(4)[None] > landmark[:, None]
    inp = RawMRIInput(images, observed, landmark, torch.randn(6, 4),
        torch.ones(6, 4, dtype=torch.bool), torch.zeros(6, dtype=torch.long),
        torch.zeros(6, dtype=torch.bool), torch.full((6,), -1, dtype=torch.long), query)
    sup = RawMRISupervision(future, future_mask, torch.tensor([1., 0., 1., 0., 1., 0.]),
                            torch.ones(6, dtype=torch.bool))
    return inp, sup


def tf_loss(model, inp, sup, **kwargs):
    options = dict(prediction_training="teacher_forcing", flow_weight=0,
                   reconstruction_weight=0, variance_weight=0)
    return model.compute_loss(inp, sup, **(options | kwargs))


def has_grad(module):
    return any(value.grad is not None and value.grad.abs().sum() > 0
               for value in module.parameters())


def test_tf_shares_one_context_teacher_and_world_forward_with_real_gradients(cfg, batch, monkeypatch):
    model = RawVLAJEPA(cfg).train()
    inp, sup = batch
    calls = {name: 0 for name in ("encoder", "teacher_encoder", "fusion", "world_predictor")}
    handles = []
    for name in calls:
        def record(module, args, output, name=name):
            calls[name] += 1
        handles.append(getattr(model, name).register_forward_hook(record))
    def forbidden(*args, **kwargs):
        raise AssertionError("TF training must not use an autonomous rollout")
    monkeypatch.setattr(model, "_autonomous_states", forbidden)
    monkeypatch.setattr(model, "_rollout", forbidden)
    try:
        loss, terms = tf_loss(model, inp, sup)
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    assert calls == dict.fromkeys(calls, 1)
    assert terms["pair_count"] == 5
    assert terms["rollout_count"] == 0
    for name in ("encoder", "fusion", "world_predictor", "pcr_head"):
        assert has_grad(getattr(model, name))
    assert all(value.grad is None and not value.requires_grad for value in model.teacher_encoder.parameters())


def test_farthest_source_valid_prediction_and_observed_only_fallback(cfg, batch, monkeypatch):
    model = RawVLAJEPA(replace(cfg, world_residual=True)).eval()
    inp, sup = batch
    recorded = {}
    original = model._teacher_forced_readout
    def readout(inp, output, teacher, details=None):
        result = original(inp, output, teacher, details)
        recorded.update(result)
        return result
    monkeypatch.setattr(model, "_teacher_forced_readout", readout)
    _, terms = tf_loss(model, inp, sup)
    assert recorded["pcr_read_stage"].tolist() == [3, 2, 3, 3, 0, 3]
    assert recorded["pcr_observation_fallback"].tolist() == [False, False, False, True, True, False]
    assert [int(terms[f"pcr_stage_T{stage}_count"]) for stage in range(4)] == [1, 0, 1, 4]
    assert terms["pcr_observation_fallback_count"] == 2
    with torch.no_grad():
        states, _ = model._teacher_states(inp, sup)
    terminal = recorded["pcr_state_features"][:, cfg.dim:2 * cfg.dim]
    # Residual-zero initialization copies the true predecessor of each selected
    # prediction, including missing-target T2/T3; fallback never reads true T3.
    expected_source = [2, 1, 2, 3, 0, 2]
    expected = torch.stack([states[row, stage].mean(0)
                            for row, stage in enumerate(expected_source)])
    torch.testing.assert_close(terminal, expected)


def test_tf_pcr_uses_true_predecessors_even_without_jepa_but_public_forward_does_not(cfg, batch):
    model = RawVLAJEPA(cfg).eval()
    inp, sup = batch
    before = model.forecast(inp)
    changed = replace(sup, future=sup.future * 50 + 10)
    loss, terms = tf_loss(model, inp, sup, jepa_weight=0)
    changed_loss, _ = tf_loss(model, inp, changed, jepa_weight=0)
    assert not torch.allclose(loss, changed_loss)
    assert terms["pair_count"] == terms["jepa"] == 0
    loss.backward()
    assert has_grad(model.world_predictor)
    after = model.forecast(inp)
    torch.testing.assert_close(before["pcr_logit"], after["pcr_logit"], atol=0, rtol=0)
    torch.testing.assert_close(before["state_prediction"], after["state_prediction"], atol=0, rtol=0)
    with pytest.raises(ValueError, match="finite"):
        tf_loss(model, inp, replace(sup, future=torch.full_like(sup.future, float("nan"))), jepa_weight=0)


def test_terminal_or_no_source_needs_no_world_forward(cfg, batch, monkeypatch):
    model = RawVLAJEPA(cfg).eval()
    inp, sup = batch
    rows = torch.tensor([3, 4])
    inp = RawMRIInput(**{name: (value[rows] if value is not None else None)
                         for name, value in vars(inp).items()})
    sup = RawMRISupervision(**{name: value[rows] for name, value in vars(sup).items()})
    def forbidden(*args, **kwargs):
        raise AssertionError("No valid requested predecessor needs no world forward")
    monkeypatch.setattr(model.world_predictor, "forward", forbidden)
    loss, terms = tf_loss(model, inp, sup)
    assert torch.isfinite(loss) and terms["pair_count"] == 0
    assert terms["pcr_observation_fallback_count"] == 2
    loss.backward()


@pytest.mark.parametrize("phase", ["all", "first_post_aqc1"])
def test_tf_flow_shares_prediction_uses_real_pairs_and_selected_channels(cfg, batch, phase, monkeypatch):
    model = RawVLAJEPA(replace(cfg, enable_flow=True, mri_phases=phase)).eval()
    inp, sup = batch
    count = 3 if phase == "all" else 1
    calls, conditions = [], []
    hook = model.world_predictor.register_forward_hook(lambda module, args, output: calls.append(output))
    original_flow = model.flow.forward
    def flow(value, tau, source, latent, stage):
        assert value.shape[1] == source.shape[1] == count
        conditions.append((latent, stage))
        return original_flow(value, tau, source, latent, stage)
    monkeypatch.setattr(model.flow, "forward", flow)
    def forbidden(*args, **kwargs):
        raise AssertionError("TF flow must reuse the parallel TF prediction")
    monkeypatch.setattr(model, "_rollout", forbidden)
    monkeypatch.setattr(model, "_autonomous_states", forbidden)
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, terms = tf_loss(model, inp, sup, flow_weight=1, reconstruction_weight=.1)
        loss.backward()
    finally:
        hook.remove()
    assert len(calls) == len(conditions) == 1
    assert len(conditions[0][0]) == int(terms["pair_count"]) == 5
    assert has_grad(model.flow) and has_grad(model.world_predictor)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("phase", ["all", "pre_aqc0", "first_post_aqc1", "metadata_late"])
def test_generation_returns_phase_metadata_and_selected_channel_shape(cfg, batch, phase):
    model = RawVLAJEPA(replace(cfg, enable_flow=True, mri_phases=phase)).train()
    inp, _ = batch
    generated = model.generate(inp, steps=1, seed=10)
    count = 3 if phase == "all" else 1
    assert generated["future_images"].shape == (6, 4, count, 4, 8, 8)
    assert generated["future_images"][~inp.query_mask].count_nonzero() == 0
    assert generated["phase_order"] == (["pre_aqc0", "first_post_aqc1", "metadata_late"]
                                         if phase == "all" else [phase])
    assert model.training and not model.teacher_encoder.training


def test_tf_rejects_autonomous_objective_and_unknown_policy(cfg, batch):
    model = RawVLAJEPA(cfg)
    inp, sup = batch
    with pytest.raises(ValueError, match="rollout_weight=0"):
        tf_loss(model, inp, sup, rollout_weight=.1)
    with pytest.raises(ValueError, match="prediction_training"):
        model.compute_loss(inp, sup, prediction_training="implicit")


def test_gradient_probe_preserves_tf_training_policy_when_jepa_is_disabled(cfg, tmp_path, monkeypatch):
    manifest = make_raw_synthetic(tmp_path, n_train=4, n_val=2, image_shape=(4, 8, 8))
    store = RawMRIStore(manifest, (4, 8, 8), allow_synthetic=True)
    train_cfg = RawVLATrainConfig(model=cfg, prediction_training="teacher_forcing",
        jepa_weight=0, flow_weight=0, reconstruction_weight=0, variance_weight=0,
        variance_definition="off", device="cpu", image_shape=(4, 8, 8),
        diagnostics_patients=4, diagnostics_batch_size=2, allow_synthetic=True)
    model = RawVLAJEPA(cfg)
    original_batch, original_loss = store.batch, model.compute_loss
    calls = []
    def supervised_batch(tasks, **kwargs):
        assert kwargs.get("future_supervision", True)
        return original_batch(tasks, **kwargs)
    def loss(inp, sup, **kwargs):
        calls.append(kwargs["prediction_training"])
        return original_loss(inp, sup, **kwargs)
    monkeypatch.setattr(store, "batch", supervised_batch)
    monkeypatch.setattr(model, "compute_loss", loss)
    report, _ = run_training_probe(model, store, train_cfg, 0, make_probe_tasks(store, "t0", 4))
    assert calls == ["teacher_forcing", "teacher_forcing"]
    assert report["world_head_trained"]
    assert all(row["weighted_gradient_norms"]["task"]["world_predictor"] > 0
               for row in report["gradient_batches"])


def test_grouped_gradient_scalars_match_direct_vector_norms_and_cosines():
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(2, 3)
    model.fusion = torch.nn.Linear(3, 1)
    encoded = model.encoder(torch.tensor([[1., 2.], [-2., 1.]]))
    terms = {"task": model.fusion(encoded).square().mean(), "jepa": encoded.square().mean()}
    weights = {"task": 1., "jepa": .5}
    parameters = tuple(model.parameters())
    actual = weighted_gradient_metrics(model, terms, weights)
    vectors = {}
    for key, weight in weights.items():
        gradients = torch.autograd.grad(weight * terms[key], parameters, retain_graph=True, allow_unused=True)
        vectors[key] = torch.cat([(value if value is not None else torch.zeros_like(parameter)).flatten()
            for value, parameter in zip(gradients, parameters)])
        assert actual["weighted_gradient_norms"][key]["all_trainable"] == pytest.approx(
            float(vectors[key].norm()), rel=1e-6)
    cosine = torch.nn.functional.cosine_similarity(vectors["task"], vectors["jepa"], dim=0)
    assert actual["task_auxiliary_cosines"]["jepa"]["shared_encoder_fusion"] == pytest.approx(
        float(cosine), rel=1e-6)
    assert all(parameter.grad is None for parameter in parameters)
