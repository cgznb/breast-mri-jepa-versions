"""Strict T0 classification, absent world branches and their CPU probes."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mri_vla_jepa.data import RawMRIStore, make_raw_synthetic
from mri_vla_jepa.diagnostics import evaluate_world, make_probe_tasks, run_training_probe
from mri_vla_jepa.io import save_checkpoint
from mri_vla_jepa.model import RawVLAJEPA
from mri_vla_jepa.model_config import RawVLAJEPAConfig
from mri_vla_jepa.train_config import RawVLATrainConfig, from_dict
from mri_vla_jepa.training import _checkpoint_config_matches, _configure, _make_optimizer
from mri_vla_jepa.vq_encoder import VQContinuousFrontend


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def configuration(**updates):
    return RawVLAJEPAConfig(**(dict(dim=16, heads=2, token_grid=(1, 2, 2),
        encoder_depth=1, fusion_depth=1, predictor_depth=1,
        state_queries=2, pcr_queries=2, fm_channels=4,
        enable_world_model=False, enable_flow=False, pcr_readout="direct") | updates))


def training_configuration(model=None, **updates):
    return RawVLATrainConfig(**(dict(model=model or configuration(),
        image_shape=(8, 16, 16), device="cpu", precision="fp32", threads=1,
        allow_synthetic=True, landmarks="t0", prediction_training="teacher_forcing",
        jepa_weight=0, rollout_weight=0, flow_weight=0, reconstruction_weight=0,
        variance_weight=0, variance_definition="off", diagnostics_batch_size=2,
        diagnostics_patients=4) | updates)).validate()


@pytest.fixture
def store(tmp_path):
    manifest = make_raw_synthetic(tmp_path / "data", n_train=4, n_val=2, image_shape=(8, 16, 16))
    return RawMRIStore(manifest, (8, 16, 16), allow_synthetic=True)


def loss_arguments(**updates):
    return dict(task_weight=1, jepa_weight=0, rollout_weight=0, flow_weight=0,
        reconstruction_weight=0, variance_weight=0, variance_definition="off",
        prediction_training="teacher_forcing") | updates


def test_direct_omits_entire_world_architecture_and_keeps_shared_initialization():
    torch.manual_seed(91)
    forecast = RawVLAJEPA(configuration(enable_world_model=True, fixed_transition_queries=True,
                                        pcr_readout="forecast"))
    torch.manual_seed(91)
    direct = RawVLAJEPA(configuration())
    assert direct.teacher_encoder is None and direct.world_predictor is None
    assert direct.dynamics_queries is None and direct.reconstruction is None and direct.flow is None
    assert not any(name.startswith(("teacher_encoder.", "world_predictor.", "dynamics_queries",
                                   "reconstruction.", "flow.")) for name in direct.state_dict())
    for name in ("encoder", "fusion", "fusion_norm", "stage", "clinical", "arm", "arm_known_at"):
        actual, expected = getattr(direct, name).state_dict(), getattr(forecast, name).state_dict()
        assert actual.keys() == expected.keys()
        assert all(torch.equal(value, expected[key]) for key, value in actual.items())
    assert torch.equal(direct.pcr_queries, forecast.pcr_queries)
    assert sum(p.numel() for p in direct.parameters()) < sum(p.numel() for p in forecast.parameters())


@pytest.mark.parametrize("readout", ["direct", "zero_state"])
def test_direct_bce_ignores_future_targets_and_never_calls_world(store, monkeypatch, readout):
    model = RawVLAJEPA(configuration(pcr_readout=readout)).eval()
    inp, sup = store.batch([(0, 0), (1, 0)])
    changed = replace(sup, future=torch.full_like(sup.future, float("nan")))
    def forbidden(*args, **kwargs):
        raise AssertionError("direct pCR called a world or teacher branch")
    for name in ("_teacher_states", "_observed_teacher_states", "_teacher_forcing_predictions", "_rollout"):
        monkeypatch.setattr(model, name, forbidden)
    original, terms = model.compute_loss(inp, sup, **loss_arguments())
    comparison, other = model.compute_loss(inp, changed, **loss_arguments())
    assert torch.equal(original, comparison) and torch.equal(terms["task"], other["task"])
    assert torch.isfinite(original)
    assert "dynamics_features" not in model(inp)


def test_direct_rejects_forecasts_and_auxiliary_losses(store):
    model = RawVLAJEPA(configuration())
    inp, sup = store.batch([(0, 0)], future_supervision=False)
    for method in (model.forecast_states, model.forecast):
        with pytest.raises(ValueError, match="disabled"):
            method(inp)
    with pytest.raises(ValueError, match="BCE-only"):
        model.compute_loss(inp, sup, **loss_arguments(jepa_weight=.5))
    with pytest.raises(ValueError, match="BCE-only"):
        training_configuration(jepa_weight=.5)
    with pytest.raises(ValueError, match="direct pCR"):
        configuration(pcr_readout="forecast", fixed_transition_queries=True).validate()


@pytest.mark.parametrize("readout", ["direct", "zero_state"])
def test_direct_pretrained_vq_frontend_is_initialized_and_updated(store, tmp_path, readout):
    cfg = training_configuration(model=configuration(encoder_type="vq_encoder", pcr_readout=readout,
        vq_hidden_channels=4, vq_embedding_dim=2, vq_num_groups=4,
        vq_bottleneck_blocks=0, vq_activation_checkpointing=True), encoder_lr=1e-4)
    frontend = VQContinuousFrontend(cfg.model)
    codec = tmp_path / "codec.pt"
    save_checkpoint(codec, {"schema": "symm_world_codec_v2",
        "numeric_contract": "registered_ROI32_shared_training_DCE0_normalization",
        "model_config": dict(frontend.architecture), "codec_state": frontend.state_dict()})
    model = RawVLAJEPA(cfg.model)
    model.initialize_pretrained_encoder(codec)
    assert all(torch.equal(value, model.encoder.frontend.state_dict()[key])
               for key, value in frontend.state_dict().items())
    _configure(model, cfg, "joint")
    optimizer = _make_optimizer(model, cfg)
    assert {group["name"]: group["lr"] for group in optimizer.param_groups} == {
        "network": 1e-4, "pretrained_encoder": 1e-4}
    before = {name: value.detach().clone() for name, value in model.encoder.frontend.encoder.named_parameters()}
    inp, sup = store.batch([(0, 0), (1, 0)], future_supervision=False)
    loss, _ = model.compute_loss(inp, sup, **loss_arguments())
    loss.backward()
    assert any(p.grad is not None and bool(p.grad.count_nonzero()) for p in model.encoder.pretrained_parameters())
    optimizer.step()
    assert any(not torch.equal(value, before[name])
               for name, value in model.encoder.frontend.encoder.named_parameters())


@pytest.mark.parametrize("readout", ["direct", "zero_state"])
def test_direct_probe_does_not_read_followups_and_replays_without_teacher(store, monkeypatch, readout):
    cfg = training_configuration(model=configuration(pcr_readout=readout))
    model = RawVLAJEPA(cfg.model)
    original = store._read_scan
    def baseline_only(visit, *args):
        assert visit["stage"] == 0, "direct probe read a future MRI"
        return original(visit, *args)
    monkeypatch.setattr(store, "_read_scan", baseline_only)
    tasks = make_probe_tasks(store, "t0", 4)
    report, private = run_training_probe(model, store, cfg, 0, tasks)
    assert report["teacher_present"] is False and report["teacher_frozen"] is None
    assert report["teacher_representation"]["status"] == "not_applicable"
    assert report["teacher_drift"]["status"] == "not_applicable"
    assert report["world"]["status"] == "not_applicable" and not report["world"]["applicable"]
    assert private["teacher_states"] is None and private["known_mask"] is None
    assert report["observed_representation"]["per_stage"]["T0"]["patients"] == 4
    assert all(batch["weighted_gradient_norms"]["task"]["encoder"] > 0
               for batch in report["gradient_batches"])
    replay, _ = run_training_probe(model, store, cfg, 5, tasks, private)
    assert report["gradient_batches"] == replay["gradient_batches"]
    assert evaluate_world(model, store, cfg)["status"] == "not_applicable"
    json.dumps(report, allow_nan=False)


def test_t0_forecast_hybrid_future_changes_jepa_but_not_bce_or_prediction(store):
    cfg = configuration(enable_world_model=True, pcr_readout="forecast", fixed_transition_queries=True)
    model = RawVLAJEPA(cfg).eval()
    inp, sup = store.batch([(0, 0), (1, 0)])
    assert bool(inp.observed_mask[:, 0].all()) and not bool(inp.observed_mask[:, 1:].any())
    changed = replace(sup, future=sup.future * 5 + sup.future_mask[:, :, None, None, None, None] * 7)
    args = loss_arguments(prediction_training="hybrid", jepa_weight=.5)
    first_prediction = model(inp)["pcr_logit"].detach().clone()
    _, before = model.compute_loss(inp, sup, **args)
    _, after = model.compute_loss(inp, changed, **args)
    assert torch.equal(before["task"], after["task"])
    assert not torch.equal(before["jepa"], after["jepa"])
    assert torch.equal(first_prediction, model(inp)["pcr_logit"])
    assert before["pair_count"] == 6


@pytest.mark.parametrize("prediction_training", ["teacher_forcing", "hybrid"])
def test_zero_state_keeps_three_part_forecast_head_and_exact_zero_inputs(store, prediction_training):
    torch.manual_seed(184)
    forecast = RawVLAJEPA(configuration(enable_world_model=True, fixed_transition_queries=True,
                                        pcr_readout="forecast")).eval()
    torch.manual_seed(184)
    model = RawVLAJEPA(configuration(pcr_readout="zero_state")).eval()
    assert model.pcr_head[0].normalized_shape == forecast.pcr_head[0].normalized_shape == (3 * model.cfg.dim,)
    assert model.pcr_head[-1].in_features == forecast.pcr_head[-1].in_features == 3 * model.cfg.dim
    assert all(torch.equal(value, forecast.pcr_head.state_dict()[key])
               for key, value in model.pcr_head.state_dict().items())
    assert model.teacher_encoder is None and model.world_predictor is None and model.dynamics_queries is None
    assert model.reconstruction is None and model.flow is None
    inp, sup = store.batch([(0, 0), (1, 0)], future_supervision=False)
    assert bool(inp.observed_mask[:, 0].all()) and not bool(inp.observed_mask[:, 1:].any())
    output = model(inp)
    query, terminal, change = output["pcr_state_features"].chunk(3, -1)
    torch.testing.assert_close(query, output["pcr_features"].mean(1), rtol=0, atol=0)
    assert torch.count_nonzero(terminal) == torch.count_nonzero(change) == 0
    assert output["pcr_state_features"].shape == (2, 3 * model.cfg.dim)
    changed_images = inp.images.clone()
    changed_images[:, 1:] = float("nan")
    # Public inputs reject even masked future image values; no future MRI can
    # be supplied as a hidden placeholder to the T0-only classifier.
    with pytest.raises(ValueError, match="Unobserved MRI slots must be zero"):
        model(replace(inp, images=changed_images))
    captured = []
    hook = model.pcr_head.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    try:
        loss, _ = model.compute_loss(inp, sup, **loss_arguments(prediction_training=prediction_training))
    finally:
        hook.remove()
    assert captured and all(value.shape == (2, 3 * model.cfg.dim) for value in captured)
    assert all(torch.count_nonzero(value[:, model.cfg.dim:]) == 0 for value in captured)
    loss.backward()
    assert model.pcr_queries.grad is not None and bool(model.pcr_queries.grad.count_nonzero())
    assert any(p.grad is not None and bool(p.grad.count_nonzero()) for p in model.encoder.parameters())


def test_zero_state_configuration_roundtrips_and_requires_absent_world_model():
    cfg = training_configuration(model=configuration(pcr_readout="zero_state"))
    assert from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="enable_world_model=false"):
        configuration(pcr_readout="zero_state", enable_world_model=True).validate()


def test_legacy_configs_default_to_world_model_and_checkpoint_keys_match():
    cfg = training_configuration(model=configuration(enable_world_model=True), jepa_weight=.5)
    saved = cfg.to_dict()
    del saved["model"]["enable_world_model"]
    assert from_dict(saved).model.enable_world_model is True
    assert _checkpoint_config_matches(saved, cfg)
    torch.manual_seed(67)
    expected = RawVLAJEPA(cfg.model)
    actual = RawVLAJEPA(from_dict(saved).model)
    actual.load_state_dict(expected.state_dict(), strict=True)
    assert any(name.startswith("world_predictor.") for name in actual.state_dict())


def test_profile_t0_stages_and_patient_requirements():
    path = Path(__file__).parents[1] / "scripts" / "profile_capacity_phase.py"
    spec = importlib.util.spec_from_file_location("t0_capacity_profile", path)
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    direct = training_configuration()
    forecast = replace(direct, model=configuration(enable_world_model=True,
        pcr_readout="forecast", fixed_transition_queries=True), jepa_weight=.5, prediction_training="hybrid")
    store = SimpleNamespace(by_split={"train": [0, 1]}, patients=[
        {"visits": [{}, None, None, None]}, {"visits": [{}, {}, {}, {}]}])
    assert profile.profile_stages(direct) == profile.profile_stages(forecast) == (0,)
    assert profile.profile_patients(store, direct) == [0, 1]
    assert profile.profile_patients(store, forecast) == [1]
    assert not profile.needs_future_supervision(direct)
    assert profile.needs_future_supervision(forecast)
    assert profile.profile_stages(replace(forecast, landmarks="all_observed")) == (0, 1, 2, 3)
