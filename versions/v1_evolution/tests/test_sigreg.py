"""SIGReg gradients, data boundaries, disabled behavior and checkpoint recovery."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from mri_vla_jepa.contracts import RawMRIInput, RawMRISupervision
from mri_vla_jepa.data import RawMRIStore, make_raw_synthetic
from mri_vla_jepa.io import load_checkpoint
from mri_vla_jepa.model import RawVLAJEPA, RawVLAJEPAConfig
from mri_vla_jepa.train_config import load_config
from mri_vla_jepa.training import _checkpoint_config_matches, _source_snapshot, train


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(831)
    yield
    torch.set_num_threads(previous)


def _case():
    observed = torch.tensor([[True, False, False, False],
                             [True, True, False, False],
                             [True, True, True, False]])
    images = torch.zeros(3, 4, 3, 4, 8, 8)
    images[observed] = torch.randn(int(observed.sum()), 3, 4, 8, 8)
    landmarks = torch.arange(3)
    query = torch.arange(4)[None] > landmarks[:, None]
    inp = RawMRIInput(images, observed, landmarks, torch.randn(3, 4),
                      torch.ones(3, 4, dtype=torch.bool), torch.tensor([1, 2, 3]),
                      torch.ones(3, dtype=torch.bool), torch.zeros(3, dtype=torch.long), query)
    future = torch.full_like(images, float("nan"))
    future[query] = torch.randn(int(query.sum()), 3, 4, 8, 8)
    sup = RawMRISupervision(future, query, torch.tensor([0., 1., 0.]),
                            torch.ones(3, dtype=torch.bool))
    model = RawVLAJEPA(RawVLAJEPAConfig(
        dim=24, heads=4, token_grid=(1, 2, 2), encoder_depth=1,
        fusion_depth=1, predictor_depth=1, state_queries=2,
        pcr_queries=2, fm_channels=8, dropout=0., pcr_readout="direct"))
    return model, inp, sup


def _loss(model, inp, sup, **kwargs):
    options = dict(task_weight=1., jepa_weight=0., rollout_weight=0., flow_weight=0.,
                   reconstruction_weight=0., variance_weight=0., variance_definition="off")
    options.update(kwargs)
    return model.compute_loss(inp, sup, **options)


def _has_grad(module):
    return any(p.grad is not None and bool(p.grad.abs().sum()) for p in module.parameters())


def test_disabled_sigreg_preserves_exact_loss_terms_rng_and_predictions():
    model, inp, sup = _case()
    model.eval()
    prediction = model(inp)["pcr_logit"].detach().clone()
    rng = torch.get_rng_state().clone()
    loss, terms = _loss(model, inp, sup)
    explicit, disabled_terms = _loss(model, inp, sup, sigreg_weight=0.,
                                    sigreg_num_projections=3, sigreg_num_points=2)
    torch.testing.assert_close(loss, explicit, rtol=0, atol=0)
    assert terms.keys() == disabled_terms.keys() and "sigreg" not in terms
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(model(inp)["pcr_logit"], prediction, rtol=0, atol=0)


def test_sigreg_is_additive_and_uses_no_extra_online_encoding():
    model, inp, sup = _case()
    model.eval()
    calls = []
    handle = model.encoder.register_forward_hook(lambda *args: calls.append(1))
    base, _ = _loss(model, inp, sup)
    baseline_calls = len(calls)
    calls.clear()
    enabled, terms = _loss(model, inp, sup, sigreg_weight=.09,
                            sigreg_num_projections=11, sigreg_num_points=7)
    assert len(calls) == baseline_calls == 1
    handle.remove()
    assert terms["sigreg"] > 0 and terms["sigreg_stage_count"] == 2
    torch.testing.assert_close(enabled, base + .09 * terms["sigreg"])


def test_sigreg_alone_updates_online_encoder_without_teacher_or_predictor_gradients():
    model, inp, sup = _case()
    loss, terms = _loss(model, inp, sup, task_weight=0., sigreg_weight=.09,
                        sigreg_num_projections=9, sigreg_num_points=7)
    loss.backward()
    assert _has_grad(model.encoder)
    assert not _has_grad(model.teacher_encoder)
    assert not _has_grad(model.world_predictor)
    assert not _has_grad(model.fusion)
    assert not _has_grad(model.pcr_head)
    assert all(p.grad is None for p in model.teacher_encoder.parameters())
    assert terms["sigreg_stage_count"] == 2


def test_sigreg_ignores_hidden_future_images_and_pcr_labels():
    model, inp, sup = _case()
    model.eval()
    torch.manual_seed(9)
    _, before = _loss(model, inp, sup, task_weight=0., sigreg_weight=.09,
                       sigreg_num_projections=9, sigreg_num_points=7)
    changed = replace(sup, future=torch.full_like(sup.future, float("nan")), label=1-sup.label)
    torch.manual_seed(9)
    _, after = _loss(model, inp, changed, task_weight=0., sigreg_weight=.09,
                      sigreg_num_projections=9, sigreg_num_points=7)
    torch.testing.assert_close(before["sigreg"], after["sigreg"], rtol=0, atol=0)


def _assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_same(a, b)
    else:
        assert left == right


def test_training_resume_restores_random_projection_sequence_and_optimizer(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/raw_vla_jepa_smoke_t0.yaml")
    cfg = replace(cfg, joint_steps=4, batch_size=2, accumulation=1,
                  jepa_weight=0., rollout_weight=0., flow_weight=0.,
                  reconstruction_weight=0., variance_weight=0., variance_definition="off",
                  sigreg_weight=.09, sigreg_num_projections=9, sigreg_num_points=7,
                  validation_every=1).validate()
    manifest = make_raw_synthetic(tmp_path / "data", n_train=4, n_val=2, image_shape=cfg.image_shape)
    store = RawMRIStore(manifest, cfg.image_shape, allow_synthetic=True)
    train(store, cfg, tmp_path / "full")
    train(store, cfg, tmp_path / "resume", stop_after=2)
    for key, value in (("sigreg_weight", .1), ("sigreg_num_projections", 10), ("sigreg_num_points", 9)):
        with pytest.raises(ValueError, match="Resume config/data identity differs"):
            train(store, replace(cfg, **{key: value}), tmp_path / "resume", resume=True)
    train(store, cfg, tmp_path / "resume", resume=True)
    full = load_checkpoint(tmp_path / "full/last.pt")
    resumed = load_checkpoint(tmp_path / "resume/last.pt")
    for key in ("history", "model", "optimizer", "rng", "sampler_state"):
        _assert_same(full[key], resumed[key])
    assert all("sigreg" in row and row["sigreg"] > 0 for row in full["history"])
    assert "sigreg.py" in full["source_snapshot"]


def test_sigreg_defaults_preserve_old_config_loading_and_bind_new_fields():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/raw_vla_jepa_smoke_t0.yaml")
    old = cfg.to_dict()
    for key in ("sigreg_weight", "sigreg_num_projections", "sigreg_num_points"):
        old.pop(key)
    assert cfg.sigreg_weight == 0 and _checkpoint_config_matches(old, cfg)
    for key, invalid in (("sigreg_weight", float("nan")), ("sigreg_weight", -.1),
                         ("sigreg_num_projections", True), ("sigreg_num_projections", 0),
                         ("sigreg_num_points", 1), ("sigreg_num_points", 3.5)):
        with pytest.raises(ValueError):
            replace(cfg, **{key: invalid}).validate()
    assert "sigreg.py" in _source_snapshot()
