"""Capacity, causal-mask and single-phase isolation contracts."""
from dataclasses import asdict

import pytest
import torch
from torch import nn

from mri_vla_jepa.attention import build_transformer
from mri_vla_jepa.encoder import VisitMRIEncoder, VQVisitMRIEncoder
from mri_vla_jepa.flow import ConditionalMRIFlow
from mri_vla_jepa.model_config import MODEL_SIZE_PRESETS, RawVLAJEPAConfig
from mri_vla_jepa.phases import PHASES, phase_indices, select_phases, selected_phases
from mri_vla_jepa.train_config import from_dict


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny(**updates):
    return RawVLAJEPAConfig(**(dict(dim=24, heads=3, encoder_depth=1, fusion_depth=1,
                                   predictor_depth=1, token_grid=(1, 2, 2), fm_channels=4)
                              | updates))


@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_presets_expand_and_roundtrip(size):
    cfg = RawVLAJEPAConfig(model_size=size).validate()
    for name, value in MODEL_SIZE_PRESETS[size].items():
        assert getattr(cfg, name) == value
    assert from_dict({"model": asdict(cfg)}).model == cfg
    with pytest.raises(ValueError, match="conflicts"):
        RawVLAJEPAConfig(model_size=size, dim=32)


def test_custom_legacy_structure_and_default_keys_are_preserved():
    cfg = tiny()
    assert cfg.model_size is None and cfg.attention_head_dim is None
    torch.manual_seed(15)
    actual = build_transformer(cfg, 2).eval()
    torch.manual_seed(15)
    expected = nn.TransformerEncoder(nn.TransformerEncoderLayer(
        cfg.dim, cfg.heads, cfg.dim * 4, cfg.dropout, activation="gelu",
        batch_first=True, norm_first=True), 2, enable_nested_tensor=False).eval()
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for key, value in actual.state_dict().items():
        assert torch.equal(value, expected.state_dict()[key])
    values = torch.randn(2, 5, cfg.dim)
    torch.testing.assert_close(actual(values), expected(values), rtol=0, atol=0)


def test_independent_attention_matches_pytorch_mask_and_prenorm_math():
    cfg = tiny(attention_head_dim=8, ffn_dim=53)
    actual = build_transformer(cfg, 1).eval()
    expected = nn.TransformerEncoderLayer(cfg.dim, cfg.heads, cfg.ffn_dim, 0.,
                                         activation="gelu", batch_first=True,
                                         norm_first=True).eval()
    layer = actual.layers[0]
    with torch.no_grad():
        layer.self_attn.qkv.weight.copy_(expected.self_attn.in_proj_weight)
        layer.self_attn.qkv.bias.copy_(expected.self_attn.in_proj_bias)
        layer.self_attn.out_proj.load_state_dict(expected.self_attn.out_proj.state_dict())
        for name in ("linear1", "linear2", "norm1", "norm2"):
            getattr(layer, name).load_state_dict(getattr(expected, name).state_dict())
    values = torch.randn(2, 6, cfg.dim)
    blocked = torch.ones(6, 6, dtype=torch.bool).triu(1)
    padding = torch.tensor([[False, False, True, False, False, False],
                            [False, False, False, True, False, False]])
    torch.testing.assert_close(actual(values, mask=blocked, src_key_padding_mask=padding),
                               expected(values, src_mask=blocked, src_key_padding_mask=padding),
                               rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual(values, is_causal=True, src_key_padding_mask=padding),
                               actual(values, mask=blocked, src_key_padding_mask=padding))


def test_wider_attention_blocks_future_and_padding_and_backpropagates():
    cfg = tiny(attention_head_dim=16, ffn_dim=71)
    model = build_transformer(cfg, 2).eval()
    assert model.layers[0].self_attn.qkv.out_features == 3 * 3 * 16
    source = torch.randn(2, 6, cfg.dim, requires_grad=True)
    blocked = torch.ones(6, 6, dtype=torch.bool).triu(1)
    padding = torch.tensor([[False, True, False, False, False, False]]).expand(2, -1)
    original = model(source, mask=blocked, src_key_padding_mask=padding)
    changed = source.detach().clone()
    changed[:, 1] = torch.randn_like(changed[:, 1]) * 100
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 100
    comparison = model(changed, mask=blocked, src_key_padding_mask=padding)
    torch.testing.assert_close(original[:, [0, 2, 3]], comparison[:, [0, 2, 3]])
    original.square().mean().backward()
    assert torch.isfinite(source.grad).all()
    assert model.layers[0].self_attn.qkv.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("phase", ["all", *PHASES])
@pytest.mark.parametrize("encoder_type", ["raw", "vq_prequant", "vq_encoder"])
def test_encoders_select_only_requested_phases(phase, encoder_type):
    cfg = tiny(mri_phases=phase, encoder_type=encoder_type)
    encoder = (VisitMRIEncoder(cfg) if encoder_type == "raw" else VQVisitMRIEncoder(cfg)).eval()
    values = torch.randn(2, 3, 8, 8, 8)
    counts = []
    handle = None
    if encoder_type != "raw":
        handle = encoder.frontend.register_forward_hook(lambda *_: counts.append(1))
        assert encoder.projection.in_channels == len(selected_phases(cfg)) * encoder.frontend.output_channels
    else:
        assert encoder.stem[0].in_channels == len(selected_phases(cfg))
    first = encoder(values)
    assert first.shape == (2, 4, cfg.dim)
    if handle is not None:
        handle.remove()
        assert len(counts) == len(selected_phases(cfg))
    if phase != "all":
        changed = values.clone()
        for index in set(range(3)) - set(phase_indices(cfg)):
            changed[:, index] = float("nan")
        torch.testing.assert_close(encoder(changed), first, rtol=0, atol=0)


@pytest.mark.parametrize("phase", ["all", *PHASES])
def test_flow_channels_and_phase_helpers(phase):
    cfg = tiny(mri_phases=phase)
    images = torch.randn(2, 4, 3, 8, 8, 8)
    selected = select_phases(images, cfg)
    channels = len(selected_phases(cfg))
    assert selected.shape == (2, 4, channels, 8, 8, 8)
    flow = ConditionalMRIFlow(cfg)
    assert flow.stem.in_channels == 2 * channels + cfg.fm_channels
    output = flow(selected[:, 0], torch.rand(2), selected[:, 1],
                  torch.randn(2, 4, cfg.dim), torch.tensor([1, 2]))
    assert output.shape == selected[:, 0].shape
    output.mean().backward()
    assert flow.output.weight.grad is not None


def test_phase_and_capacity_validation():
    for update in (dict(mri_phases="aqc2"), dict(attention_head_dim=0),
                   dict(ffn_dim=True), dict(model_size="extra_large")):
        with pytest.raises(ValueError):
            tiny(**update).validate()
    with pytest.raises(ValueError, match="three-phase"):
        select_phases(torch.randn(2, 1, 8, 8, 8), "first_post_aqc1")
