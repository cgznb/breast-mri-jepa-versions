"""Paired spatial readout ablations preserve legal state sources and gradients."""
from dataclasses import replace

import pytest
import torch

from mri_vla_jepa.contracts import RawMRIInput, RawMRISupervision
from mri_vla_jepa.model import RawVLAJEPA
from mri_vla_jepa.model_config import RawVLAJEPAConfig


ATTENTION_READOUTS = ("attention", "change_residual_attention")


def score_weight(model):
    scorer = model.pcr_spatial_scorer
    return scorer.score.weight if hasattr(scorer, "score") else scorer[-1].weight


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(17043)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def cfg():
    return RawVLAJEPAConfig(dim=16, heads=2, token_grid=(1, 2, 2),
        encoder_depth=1, fusion_depth=1, predictor_depth=1, state_queries=2,
        pcr_queries=2, fixed_transition_queries=True, pcr_readout="forecast",
        history_delta=True, dropout=0.)


@pytest.fixture
def batch():
    observed = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    landmark = torch.tensor([0, 1, 3])
    queries = torch.arange(4)[None] > landmark[:, None]
    images = torch.zeros(3, 4, 3, 4, 8, 8)
    images[observed] = torch.randn(int(observed.sum()), 3, 4, 8, 8)
    inp = RawMRIInput(images, observed, landmark, torch.randn(3, 4),
        torch.ones(3, 4, dtype=torch.bool), torch.zeros(3, dtype=torch.long),
        torch.zeros(3, dtype=torch.bool), torch.full((3,), -1, dtype=torch.long), queries)
    future_mask = queries.clone()
    future_mask[1, 2] = False
    future = torch.full_like(images, float("nan"))
    future[future_mask] = torch.randn(int(future_mask.sum()), 3, 4, 8, 8)
    sup = RawMRISupervision(future, future_mask, torch.tensor([1., 0., 1.]),
                            torch.ones(3, dtype=torch.bool))
    return inp, sup


def loss_options(policy="teacher_forcing"):
    return dict(prediction_training=policy, jepa_weight=0., rollout_weight=0.,
                flow_weight=0., reconstruction_weight=0., variance_weight=0.)


@pytest.mark.parametrize("pool", ATTENTION_READOUTS)
def test_attention_initialization_preserves_shared_parameters_and_rng(cfg, batch, pool):
    torch.manual_seed(841)
    mean = RawVLAJEPA(cfg).eval()
    after_mean = torch.get_rng_state().clone()
    torch.manual_seed(841)
    attention = RawVLAJEPA(replace(cfg, spatial_readout=pool)).eval()
    assert torch.equal(torch.get_rng_state(), after_mean)
    for key, value in mean.state_dict().items():
        torch.testing.assert_close(attention.state_dict()[key], value, atol=0, rtol=0)
    assert not any(key.startswith("pcr_spatial_scorer.") for key in mean.state_dict())
    assert score_weight(attention).count_nonzero() == 0
    # Mean checkpoints retain their exact original parameter names and load strictly.
    RawVLAJEPA(cfg).load_state_dict(mean.state_dict(), strict=True)
    inp, sup = batch
    first, second = mean(inp), attention(inp)
    torch.testing.assert_close(first["pcr_state_features"], second["pcr_state_features"])
    torch.testing.assert_close(first["pcr_logit"], second["pcr_logit"])
    torch.testing.assert_close(mean.compute_loss(inp, sup, **loss_options())[0],
                               attention.compute_loss(inp, sup, **loss_options())[0])
    assert "pcr_spatial_attention" not in first


@pytest.mark.parametrize("grid", [(1, 2, 2), (2, 4, 4)])
@pytest.mark.parametrize("pool", ATTENTION_READOUTS)
def test_attention_weights_pool_terminal_and_delta_together(cfg, batch, grid, pool):
    model = RawVLAJEPA(replace(cfg, token_grid=grid, spatial_readout=pool)).eval()
    inp, _ = batch
    count = model.world_predictor.tokens
    terminal = torch.randn(3, count, cfg.dim, requires_grad=True)
    reference = torch.randn_like(terminal)
    pcr = torch.randn(3, cfg.pcr_queries, cfg.dim)
    uniform, stats = model._pcr_state_readout(inp, pcr, terminal, reference, reference)
    expected = torch.cat((pcr.mean(1), terminal.mean(1), (terminal - reference).mean(1)), -1)
    torch.testing.assert_close(uniform, expected)
    torch.testing.assert_close(stats["pcr_spatial_attention"], torch.full((3, count), 1 / count))
    torch.testing.assert_close(stats["pcr_spatial_attention_entropy"], torch.full((3,), torch.tensor(count).log()))
    torch.testing.assert_close(stats["pcr_spatial_effective_tokens"], torch.full((3,), float(count)))
    with torch.no_grad():
        score_weight(model).normal_()
    features, stats = model._pcr_state_readout(inp, pcr, terminal, reference, reference)
    weights = stats["pcr_spatial_attention"]
    assert torch.isfinite(weights).all() and (weights >= 0).all()
    torch.testing.assert_close(weights.sum(1), torch.ones(3))
    if pool == "change_residual_attention":
        mix = stats["pcr_spatial_attention_mix"]
        learned = stats["pcr_spatial_learned_attention"]
        torch.testing.assert_close(mix, torch.full((3,), cfg.pcr_attention_mix_init))
        torch.testing.assert_close(weights, (1 - mix[:, None]) / count + mix[:, None] * learned)
        assert (weights >= (1 - mix[:, None]) / count).all()
    torch.testing.assert_close(features[:, :cfg.dim], pcr.mean(1))
    torch.testing.assert_close(features[:, cfg.dim:2 * cfg.dim], (terminal * weights[..., None]).sum(1))
    torch.testing.assert_close(features[:, -cfg.dim:], ((terminal - reference) * weights[..., None]).sum(1))
    assert all(not value.requires_grad for value in stats.values())
    zero_change, _ = model._pcr_state_readout(inp, pcr, terminal, terminal, reference)
    assert zero_change[:, -cfg.dim:].count_nonzero() == 0


@pytest.mark.parametrize("policy", ["teacher_forcing", "hybrid"])
@pytest.mark.parametrize("detach", [False, True])
@pytest.mark.parametrize("pool", ATTENTION_READOUTS)
def test_attention_and_t0_keep_classification_gradient_routes(cfg, batch, policy, detach, pool):
    model = RawVLAJEPA(replace(cfg, spatial_readout=pool, pcr_delta_reference="t0",
                             pcr_readout="forecast_detach" if detach else "forecast")).train()
    inp, sup = batch
    counts = dict.fromkeys(("encoder", "teacher_encoder", "world_predictor"), 0)
    handles = []
    for name in counts:
        def record(module, args, output, name=name):
            counts[name] += 1
        handles.append(getattr(model, name).register_forward_hook(record))
    try:
        loss, _ = model.compute_loss(inp, sup, **loss_options(policy))
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    assert torch.isfinite(loss)
    assert score_weight(model).grad.abs().sum() > 0
    world_gradient = sum(float(p.grad.abs().sum()) for p in model.world_predictor.parameters() if p.grad is not None)
    assert bool(world_gradient) == (not detach)
    assert all(p.grad is None and not p.requires_grad for p in model.teacher_encoder.parameters())
    assert counts["encoder"] == counts["teacher_encoder"] == 1
    if policy == "teacher_forcing":
        assert counts["world_predictor"] == 1


@pytest.mark.parametrize("policy", ["teacher_forcing", "hybrid"])
def test_t0_difference_uses_teacher_baseline_and_preserves_first_two_branches(cfg, batch, policy, monkeypatch):
    original = RawVLAJEPA(cfg).eval()
    baseline = RawVLAJEPA(replace(cfg, pcr_delta_reference="t0")).eval()
    baseline.load_state_dict(original.state_dict(), strict=True)
    inp, sup = batch
    recorded = {}
    for name, model in (("current", original), ("t0", baseline)):
        helper = model._pcr_state_readout
        def record(inp, pcr, terminal, current, t0, *, name=name, helper=helper):
            result = helper(inp, pcr, terminal, current, t0)
            recorded[name] = (result[0], terminal, current, t0)
            return result
        monkeypatch.setattr(model, "_pcr_state_readout", record)
        if policy == "teacher_forcing":
            model.compute_loss(inp, sup, **loss_options())
        else:
            model(inp)
    old, new = recorded["current"][0], recorded["t0"][0]
    torch.testing.assert_close(new[:, :2 * cfg.dim], old[:, :2 * cfg.dim], atol=0, rtol=0)
    torch.testing.assert_close(new[0], old[0], atol=0, rtol=0)
    with torch.no_grad():
        teacher_t0 = baseline.teacher_encoder(inp.images[:, 0])
    torch.testing.assert_close(recorded["t0"][3], teacher_t0, atol=0, rtol=0)
    expected = (recorded["t0"][1] - teacher_t0).mean(1)
    torch.testing.assert_close(new[:, -cfg.dim:], expected, atol=0, rtol=0)
    # Real T3 has no transition: its accumulated change is retained only by t0.
    assert old[2, -cfg.dim:].count_nonzero() == 0
    assert new[2, -cfg.dim:].abs().sum() > 0
    assert not recorded["t0"][3].requires_grad


@pytest.mark.parametrize("policy", ["teacher_forcing", "hybrid"])
def test_t0_readout_rejects_missing_baseline(cfg, batch, policy):
    inp, sup = batch
    observed, images = inp.observed_mask.clone(), inp.images.clone()
    observed[1, 0], images[1, 0] = False, 0
    inp = replace(inp, observed_mask=observed, images=images)
    model = RawVLAJEPA(replace(cfg, pcr_delta_reference="t0")).eval()
    with pytest.raises(ValueError, match="observed T0"):
        if policy == "teacher_forcing":
            model.compute_loss(inp, sup, **loss_options())
        else:
            model(inp)


@pytest.mark.parametrize("policy", ["teacher_forcing", "hybrid"])
@pytest.mark.parametrize("pool", ATTENTION_READOUTS)
def test_attention_t0_readout_is_finite_under_bfloat16(cfg, batch, policy, pool):
    # Match the production preset's custom attention path, avoiding PyTorch's
    # separate CPU fused-Transformer autocast limitation.
    model = RawVLAJEPA(replace(cfg, spatial_readout=pool, pcr_delta_reference="t0",
                             attention_head_dim=8)).train()
    inp, sup = batch
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = model.compute_loss(inp, sup, **loss_options(policy))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(score_weight(model).grad).all()
    model.eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model.forecast(inp)
    for key in ("pcr_logit", "pcr_spatial_attention", "pcr_spatial_attention_entropy",
                "pcr_spatial_effective_tokens"):
        assert torch.isfinite(output[key]).all()
    torch.testing.assert_close(output["pcr_spatial_attention"].sum(1), torch.ones(3))


@pytest.mark.parametrize("pool,reference", [("attention", "current"), ("mean", "t0"), ("attention", "t0"),
    ("change_residual_attention", "current"), ("change_residual_attention", "t0")])
def test_readout_restores_strictly_and_public_inference_never_uses_future(cfg, batch, pool, reference, monkeypatch):
    config = replace(cfg, spatial_readout=pool, pcr_delta_reference=reference)
    model, restored = RawVLAJEPA(config).eval(), RawVLAJEPA(config).eval()
    if pool in ATTENTION_READOUTS:
        with torch.no_grad():
            score_weight(model).normal_()
    restored.load_state_dict(model.state_dict(), strict=True)
    inp, _ = batch
    def forbidden(*args, **kwargs):
        raise AssertionError("Public inference cannot encode future supervision")
    monkeypatch.setattr(restored, "_teacher_states", forbidden)
    complete = restored.forecast(inp)
    empty = restored.forecast(replace(inp, query_mask=torch.zeros_like(inp.query_mask)))
    torch.testing.assert_close(complete["pcr_logit"], model.forecast(inp)["pcr_logit"], atol=0, rtol=0)
    torch.testing.assert_close(empty["pcr_logit"], complete["pcr_logit"], atol=0, rtol=0)
    assert empty["state_prediction"].count_nonzero() == 0


def test_change_conditioning_query_changes_spatial_distribution(cfg):
    model = RawVLAJEPA(replace(cfg, spatial_readout="change_residual_attention")).eval()
    scorer = model.pcr_spatial_scorer
    count = model.world_predictor.tokens
    current = torch.randn(3, count, cfg.dim)
    change = torch.randn_like(current)
    query = torch.randn(3, cfg.dim, requires_grad=True)
    with torch.no_grad():
        scorer.score.weight.normal_(std=.3)
    weights = scorer(current, change, query).float().softmax(1)
    changed = scorer(current, change, query.roll(1, dims=-1)).float().softmax(1)
    # A query-dependent scalar added after a linear scorer would cancel in softmax.
    assert not torch.allclose(weights, changed, atol=1e-6, rtol=1e-5)
    (weights * torch.arange(count, dtype=weights.dtype)).sum().backward()
    assert query.grad.abs().sum() > 0
    assert scorer.query_proj.weight.grad.abs().sum() > 0


def test_change_residual_attention_learns_after_uniform_initialization(cfg, batch):
    model = RawVLAJEPA(replace(cfg, spatial_readout="change_residual_attention")).train()
    scorer = model.pcr_spatial_scorer
    optimizer = torch.optim.SGD(scorer.parameters(), lr=.5)
    inp, sup = batch
    count = model.world_predictor.tokens
    terminal = torch.randn(3, count, cfg.dim)
    current = torch.randn_like(terminal)
    pcr = torch.randn(3, cfg.pcr_queries, cfg.dim, requires_grad=True)

    def classification_loss():
        features, _ = model._pcr_state_readout(inp, pcr, terminal, current, current)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            model.pcr_head(features).squeeze(-1), sup.label)

    classification_loss().backward()
    assert scorer.score.weight.grad.abs().sum() > 0
    # The score starts at zero: hidden projections become trainable after its first update.
    assert scorer.query_proj.weight.grad.count_nonzero() == 0
    optimizer.step()
    model.zero_grad(set_to_none=True)
    pcr.grad = None
    classification_loss().backward()
    for projection in (scorer.current_proj, scorer.change_proj, scorer.query_proj):
        assert torch.isfinite(projection.weight.grad).all()
        assert projection.weight.grad.abs().sum() > 0
    assert scorer.mix_logit.grad.abs().sum() > 0
    assert pcr.grad.abs().sum() > 0


@pytest.mark.parametrize("update", [dict(spatial_readout="maximum"), dict(spatial_readout=[]),
    dict(pcr_delta_reference="future"), dict(pcr_delta_reference=None),
    dict(pcr_attention_dim=0), dict(pcr_attention_dim=-1), dict(pcr_attention_dim=1.5),
    dict(pcr_attention_dim=True), dict(pcr_attention_mix_init=0.), dict(pcr_attention_mix_init=1.),
    dict(pcr_attention_mix_init=float("nan")), dict(pcr_attention_mix_init=float("inf")),
    dict(pcr_attention_mix_init=True)])
def test_readout_configuration_rejects_invalid_values(cfg, update):
    with pytest.raises(ValueError):
        replace(cfg, **update).validate()
