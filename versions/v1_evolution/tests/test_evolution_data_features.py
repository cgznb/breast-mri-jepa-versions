"""Observed T0 masks and pretrained continuous-feature ablations, on CPU."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from mri_vla_jepa.data import RawMRIStore, make_raw_synthetic
from mri_vla_jepa.data_pipeline import ROI32T0LesionCache
from mri_vla_jepa.encoder import VQVisitMRIEncoder, _SpatialGridPool, _pool_with_lesion_weights
from mri_vla_jepa.io import read_json, write_json
from mri_vla_jepa.model_config import RawVLAJEPAConfig
from mri_vla_jepa.vq_encoder import VQContinuousFrontend


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def config(kind="vq_encoder"):
    return RawVLAJEPAConfig(dim=16, heads=2, token_grid=(1, 2, 2),
                            encoder_depth=1, encoder_type=kind).validate()


@pytest.fixture
def registered_masks(tmp_path):
    path = make_raw_synthetic(tmp_path, n_train=1, n_val=1)
    manifest = read_json(path)
    manifest["image_preprocessing"] = {"mode": "preprocessed_roi32"}
    manifest["image_normalization"] = {
        "mean": 1., "std": 2., "fit_split": "train", "scope": "training foreground",
        "background_policy": "preserve_zero", "already_normalized": True,
        "normalization_policy": "Existing training scale",
    }
    manifest["provenance"] = {"roi_policy": "Fixed source-available T0 predicted ROI"}
    roi = np.zeros((1, 32, 128, 128), dtype=np.bool_)
    roi[:, 3:7, 9:13, 17:23] = True
    image = np.zeros((3, 32, 128, 128), dtype=np.float32)
    for patient in manifest["patients"]:
        for stage, visit in enumerate(patient["visits"]):
            if visit is None:
                continue
            archive = tmp_path / (patient["patient_key"] + f"_T{stage}.npz")
            # Accessing follow-up ROI or latent would fail under allow_pickle=False.
            hidden = np.array(["must not be read"], dtype=object)
            np.savez(archive, image=image, roi=roi if stage == 0 else hidden, latent=hidden)
            visit["image"] = str(archive)
    write_json(path, manifest)
    return path, roi


def test_optional_none_survives_device_and_pin_transfers(tmp_path, monkeypatch):
    path = make_raw_synthetic(tmp_path, n_train=1, n_val=1)
    inp, _ = RawMRIStore(path, image_shape=(8, 16, 16), allow_synthetic=True).batch([(0, 0)])
    assert inp.lesion_weights is None
    assert inp.to("cpu").lesion_weights is None
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: True)
    assert inp.pin_memory().lesion_weights is None


def test_real_t0_mask_coordinates_and_future_privacy(registered_masks):
    path, roi = registered_masks
    store = RawMRIStore(path, allow_synthetic=True, require_lesion_weights=True)
    inp, _ = store.batch([(0, 0), (1, 3)], supervised=False)
    assert inp.lesion_weights.shape == (2, 1, 32, 128, 128)
    for row in range(2):
        np.testing.assert_array_equal(inp.lesion_weights[row].numpy(), roi)
    assert inp.to("cpu").validate().lesion_weights is not None
    # Even supervised future image reads never load those object-valued ROIs.
    _, sup = store.batch([(0, 0)], supervised=True)
    assert sup.future_mask.any()
    signature = store.signature()["lesion_weights"]
    assert signature["member"] == "roi"
    assert len(signature["baseline_archives"]) == 2
    assert all("T0.npz" in item["path"] for item in signature["baseline_archives"])


def test_baseline_without_lesion_flag_never_requires_masks(registered_masks):
    path, _ = registered_masks
    manifest = read_json(path)
    for patient in manifest["patients"]:
        archive = patient["visits"][0]["image"]
        with np.load(archive, allow_pickle=False) as current:
            image = current["image"]
        np.savez(archive, image=image)
    store = RawMRIStore(path, allow_synthetic=True)
    inp, _ = store.batch([(0, 0)], supervised=False)
    assert inp.lesion_weights is None
    assert "lesion_weights" not in store.signature()
    with pytest.raises(ValueError, match="real T0 roi"):
        RawMRIStore(path, allow_synthetic=True, require_lesion_weights=True)


@pytest.mark.parametrize("bad", ["empty", "shape", "dtype"])
def test_mask_cache_rejects_invalid_real_regions(tmp_path, bad):
    roi = np.ones((1, 32, 128, 128), dtype=bool)
    if bad == "empty":
        roi[:] = False
    elif bad == "shape":
        roi = roi[:, :16]
    else:
        roi = roi.astype(np.float32)
    path = tmp_path / "baseline.npz"
    np.savez(path, roi=roi)
    cache = ROI32T0LesionCache([path])
    with pytest.raises(ValueError):
        cache.read(path)


def test_mask_cache_rejects_undeclared_future_and_changed_archive(tmp_path):
    path = tmp_path / "baseline.npz"
    np.savez(path, roi=np.ones((1, 32, 128, 128), dtype=bool))
    cache = ROI32T0LesionCache([path])
    with pytest.raises(ValueError, match="declared baseline"):
        cache.read(tmp_path / "future.npz")
    cache.read(path)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        cache.read(path)


@pytest.mark.parametrize("bad", ["nan", "negative", "over_one", "shape", "gradient", "unobserved_T0"])
def test_input_contract_rejects_illegal_lesion_weights(tmp_path, bad):
    path = make_raw_synthetic(tmp_path, n_train=1, n_val=1)
    inp, _ = RawMRIStore(path, image_shape=(8, 16, 16), allow_synthetic=True).batch([(0, 1)])
    mask = torch.ones(1, 1, 8, 16, 16)
    if bad == "nan":
        mask.flatten()[0] = float("nan")
    elif bad == "negative":
        mask.flatten()[0] = -.1
    elif bad == "over_one":
        mask.flatten()[0] = 1.1
    elif bad == "shape":
        mask = mask[..., :8]
    elif bad == "gradient":
        mask.requires_grad_()
    else:
        images = inp.images.clone()
        images[:, 1] = images[:, 0]
        images[:, 0] = 0
        observed = torch.tensor([[False, True, False, False]])
        inp = replace(inp, images=images, observed_mask=observed)
    with pytest.raises(ValueError, match="lesion_weights"):
        replace(inp, lesion_weights=mask).validate()


def test_weighted_pool_concentrates_real_lesion_and_keeps_empty_bins():
    images = torch.zeros(1, 3, 4, 4, 4)
    features = torch.arange(64.).reshape(1, 1, 4, 4, 4).requires_grad_()
    mask = torch.zeros(1, 1, 4, 4, 4)
    mask[..., 0, 0, 0] = 1
    pool = _SpatialGridPool((1, 2, 2))
    out = _pool_with_lesion_weights(features, images, mask, pool)
    assert out[0, 0, 0, 0, 0] == features[0, 0, 0, 0, 0]
    torch.testing.assert_close(out[..., 1, 1], pool(features)[..., 1, 1])
    torch.testing.assert_close(_pool_with_lesion_weights(features, images, mask * 0, pool), pool(features))
    out.sum().backward()
    assert features.grad.count_nonzero() > 0


def test_encoder64_replays_same_pretrained_weights_and_bypasses_prequant(tmp_path):
    compressed = VQContinuousFrontend(config("vq_prequant")).eval()
    path = tmp_path / "codec.pt"
    torch.save({"schema": "symm_world_codec_v2",
                "numeric_contract": "registered_ROI32_shared_training_DCE0_normalization",
                "model_config": compressed.architecture,
                "codec_state": compressed.state_dict()}, path)
    expanded = VQContinuousFrontend(config()).eval()
    expanded.initialize_pretrained(path)
    assert expanded.state_dict().keys() == compressed.state_dict().keys()
    assert all(torch.equal(value, compressed.state_dict()[name])
               for name, value in expanded.state_dict().items())
    images = torch.randn(1, 1, 8, 16, 16)
    compressed_result = compressed(images)
    expected = expanded.encoder(images)
    handle = expanded.pre_quant.register_forward_pre_hook(
        lambda *_: pytest.fail("encoder64 must bypass pre_quant"))
    result = expanded(images)
    handle.remove()
    assert result.shape == (1, 64, 2, 4, 4)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert compressed_result.shape == (1, 8, 2, 4, 4)


def test_encoder64_frozen_adapter_and_finetuned_frontend_gradients():
    encoder = VQVisitMRIEncoder(config()).train()
    images = torch.randn(2, 3, 8, 16, 16)
    frozen = {name: value.detach().clone() for name, value in encoder.frontend.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in encoder.parameters() if p.requires_grad], lr=1e-3)
    out = encoder(images)
    assert encoder.projection.in_channels == 192
    assert out.shape == (2, 4, 16)
    (out * torch.randn_like(out)).sum().backward()
    assert encoder.projection.weight.grad is not None
    assert all(p.grad is None for p in encoder.frontend.parameters())
    optimizer.step()
    assert all(torch.equal(value, frozen[name]) for name, value in encoder.frontend.named_parameters())
    encoder.zero_grad(set_to_none=True)
    encoder.set_pretrained_trainable(True)
    out = encoder(images)
    (out * torch.randn_like(out)).sum().backward()
    assert encoder.frontend.encoder.input.weight.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in encoder.frontend.pre_quant.parameters())
