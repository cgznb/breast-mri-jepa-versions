"""Checkpoint-key-compatible extraction of cgznb/symm-fm's 3D MRI VQGAN.

Based on workflows/first_post_three_phase/mewm_ispy2/vqgan.py at
92265d3b1749ae3b686f2089843c49da129fd4d2. Original encoder/decoder/quantizer
state-dict names and tensor geometry are preserved. Lightning training and
unrelated modality utilities removed. Frozen decode now preserves input grads.
EMA updates use no_grad and dead-code restarts also reset accumulated counts.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class VQGANConfig:
    image_channels: int = 1
    hidden_channels: int = 16
    embedding_dim: int = 8
    n_codes: int = 16384
    downsample_factor: int = 4
    bottleneck_blocks: int = 0
    num_groups: int = 32
    commitment_weight: float = .25
    ema_decay: float = .99
    ema_smoothing: float = 1e-7
    restart_threshold: float = 1.
    nearest_chunk_size: int = 4096
    def __post_init__(self):
        if self.image_channels != 1 or self.downsample_factor != 4:
            raise ValueError("Original three-phase codec is shared, single-channel, 4x spatial compression")
        if self.hidden_channels * 2 % self.num_groups:
            raise ValueError("VQ normalization groups do not divide channels")
        if self.n_codes < 2 or self.nearest_chunk_size < 1 or self.bottleneck_blocks < 0:
            raise ValueError("Invalid VQ codebook or depth")


def _triple(x):
    return (x, x, x) if isinstance(x, int) else tuple(x)


def _same_pad(kernel, stride):
    p = [(k-s, ) for k, s in zip(kernel, stride)]
    pairs = [(v[0]//2 + v[0]%2, v[0]//2) for v in p]
    return tuple(x for pair in reversed(pairs) for x in pair)


class SamePadConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type="replicate"):
        kernel, steps = _triple(kernel_size), _triple(stride)
        super().__init__(in_channels, out_channels, kernel, stride=steps, padding=0, bias=bias)
        self.pad_input, self.padding_type = _same_pad(kernel, steps), padding_type
    def forward(self, value):
        return super().forward(F.pad(value, self.pad_input, mode=self.padding_type) if any(self.pad_input) else value)


class SamePadConvTranspose3d(nn.ConvTranspose3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type="replicate"):
        kernel, steps = _triple(kernel_size), _triple(stride)
        super().__init__(in_channels, out_channels, kernel, stride=steps,
                         padding=tuple(s-1 for s in kernel), bias=bias)
        self.pad_input, self.padding_type = _same_pad(kernel, steps), padding_type
    def forward(self, value, output_size=None):
        return super().forward(F.pad(value, self.pad_input, mode=self.padding_type) if any(self.pad_input) else value,
                               output_size=output_size)


class ResidualBlock3D(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels, eps=1e-6)
        self.conv1 = SamePadConv3d(channels, channels, 3)
        self.norm2 = nn.GroupNorm(groups, channels, eps=1e-6)
        self.conv2 = SamePadConv3d(channels, channels, 3)
    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        return x + self.conv2(F.silu(self.norm2(h)))


def _identity_residual_blocks(count, channels, groups):
    blocks = nn.ModuleList([ResidualBlock3D(channels, groups) for _ in range(count)])
    for b in blocks:
        nn.init.zeros_(b.conv2.weight)
        nn.init.zeros_(b.conv2.bias)
    return blocks


class MRIEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        h, g = config.hidden_channels, config.num_groups
        self.input = SamePadConv3d(1, h, 3)
        self.downsamples = nn.ModuleList([SamePadConv3d(h, h*2, 4, 2), SamePadConv3d(h*2, h*4, 4, 2)])
        self.residuals = nn.ModuleList([ResidualBlock3D(h*2, g), ResidualBlock3D(h*4, g)])
        self.bottleneck = _identity_residual_blocks(config.bottleneck_blocks, h*4, g)
        self.final_norm = nn.GroupNorm(g, h*4, eps=1e-6)
        self.output_channels = h*4
    def forward(self, x):
        x = self.input(x)
        for down, res in zip(self.downsamples, self.residuals):
            x = res(down(x))
        for res in self.bottleneck:
            x = res(x)
        return F.silu(self.final_norm(x))


class MRIDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        h, g = config.hidden_channels, config.num_groups
        self.final_norm = nn.GroupNorm(g, h*4, eps=1e-6)
        self.bottleneck = _identity_residual_blocks(config.bottleneck_blocks, h*4, g)
        self.upsamples = nn.ModuleList([SamePadConvTranspose3d(h*4, h*4, 4, 2), SamePadConvTranspose3d(h*4, h*2, 4, 2)])
        self.residual_one = nn.ModuleList([ResidualBlock3D(h*4, g), ResidualBlock3D(h*2, g)])
        self.residual_two = nn.ModuleList([ResidualBlock3D(h*4, g), ResidualBlock3D(h*2, g)])
        self.output = SamePadConv3d(h*2, 1, 3)
    def forward(self, x):
        x = F.silu(self.final_norm(x))
        for res in self.bottleneck:
            x = res(x)
        for up, r1, r2 in zip(self.upsamples, self.residual_one, self.residual_two):
            x = r2(r1(up(x)))
        return self.output(x)


class VectorQuantizerEMA(nn.Module):
    def __init__(self, config):
        super().__init__()
        embeddings = F.normalize(torch.randn(config.n_codes, config.embedding_dim), dim=1)
        self.register_buffer("embeddings", embeddings)
        self.register_buffer("cluster_size", torch.ones(config.n_codes))
        self.register_buffer("embedding_sum", embeddings.clone())
        self.n_codes, self.embedding_dim = config.n_codes, config.embedding_dim
        self.decay, self.smoothing = config.ema_decay, config.ema_smoothing
        self.restart_threshold, self.commitment_weight = config.restart_threshold, config.commitment_weight
        self.chunk_size = config.nearest_chunk_size
    @torch.no_grad()
    def _nearest(self, flat):
        e = self.embeddings.float()
        norms = e.square().sum(1)
        return torch.cat([(x.float().square().sum(1, keepdim=True) - 2*x.float() @ e.T + norms).argmin(1)
                          for x in flat.split(self.chunk_size)])
    @torch.no_grad()
    def _update(self, flat, indices):
        counts = torch.bincount(indices, minlength=self.n_codes).to(self.cluster_size.dtype)
        sums = torch.zeros_like(self.embedding_sum)
        sums.index_add_(0, indices, flat.detach().to(sums.dtype))
        self.cluster_size.mul_(self.decay).add_(counts, alpha=1-self.decay)
        self.embedding_sum.mul_(self.decay).add_(sums, alpha=1-self.decay)
        total = self.cluster_size.sum()
        smooth = (self.cluster_size+self.smoothing) / (total+self.n_codes*self.smoothing) * total
        self.embeddings.copy_(self.embedding_sum / smooth[:, None])
        under = self.cluster_size < self.restart_threshold
        if under.any():
            candidates = flat.detach().to(self.embeddings.dtype)
            if len(candidates) < self.n_codes:
                candidates = candidates.repeat(math.ceil(self.n_codes / len(candidates)), 1)
                candidates = candidates + torch.randn_like(candidates)*(.01/math.sqrt(self.embedding_dim))
            choices = candidates[torch.randperm(len(candidates), device=flat.device)[:self.n_codes]]
            self.embeddings[under] = choices[under]
            self.cluster_size[under] = max(1., self.restart_threshold)
            self.embedding_sum[under] = choices[under] * self.cluster_size[under, None]
    def forward(self, latent):
        flat = latent.movedim(1, -1).reshape(-1, self.embedding_dim)
        indices = self._nearest(flat)
        # Clone avoids version-counter changes when updating the codebook below.
        qflat = F.embedding(indices, self.embeddings).clone()
        if self.training:
            self._update(flat, indices)
        q = qflat.reshape(latent.shape[0], *latent.shape[2:], -1).movedim(-1, 1).to(latent.dtype)
        loss = self.commitment_weight * F.mse_loss(latent.float(), q.detach().float())
        probability = torch.bincount(indices, minlength=self.n_codes).float()
        probability /= probability.sum().clamp_min(1)
        return latent + (q-latent).detach(), {
            "indices": indices.reshape(latent.shape[0], *latent.shape[2:]), "commitment_loss": loss,
            "perplexity": (-(probability * probability.clamp_min(1e-10).log()).sum()).exp()}


class MRILevelVQGAN(nn.Module):
    def __init__(self, config=VQGANConfig()):
        super().__init__()
        self.config = config
        self.encoder = MRIEncoder(config)
        self.pre_quant = SamePadConv3d(self.encoder.output_channels, config.embedding_dim, 1)
        self.quantizer = VectorQuantizerEMA(config)
        self.post_quant = SamePadConv3d(config.embedding_dim, self.encoder.output_channels, 1)
        self.decoder = MRIDecoder(config)
    def encode_continuous(self, x):
        if x.ndim != 5 or x.shape[1] != 1 or any(s % 4 for s in x.shape[-3:]):
            raise ValueError("VQ input must be B,1,D,H,W with spatial dimensions divisible by four")
        return self.pre_quant(self.encoder(x))
    def encode(self, x):
        return self.quantizer(self.encode_continuous(x))
    def decode(self, z):
        return self.decoder(self.post_quant(z))
    def forward(self, x):
        z, info = self.encode(x)
        return self.decode(z), info


class ThreePhaseCodec(nn.Module):
    def __init__(self, codec: MRILevelVQGAN, latent_mode="prequant", decode_mode="requantize"):
        super().__init__()
        if latent_mode not in ("prequant", "quantized") or decode_mode not in ("requantize", "continuous"):
            raise ValueError("Unsupported latent/decode mode")
        self.codec = codec.eval().requires_grad_(False)
        self.latent_mode, self.decode_mode = latent_mode, decode_mode
        self.checkpointing = False
    @property
    def channels(self):
        return 3 * self.codec.config.embedding_dim
    def train(self, mode=True):
        super().train(False)
        return self
    @torch.no_grad()
    def encode(self, images):
        if images.ndim != 5 or images.shape[1] != 3:
            raise ValueError("Three phases in fixed pre/first-post/metadata-late order required")
        result = []
        for x in images.split(1, dim=1):
            z = self.codec.encode_continuous(x)
            if self.latent_mode == "quantized":
                z, _ = self.codec.quantizer(z)
            result.append(z)
        return torch.cat(result, 1)
    def decode(self, joint):
        """No no_grad decorator: a frozen decoder still differentiates its INPUT.

        Quantizer is eval-only and uses the original straight-through gradient.
        Thus codebook entries never change during flow/semantic optimization.
        """
        if joint.ndim != 5 or joint.shape[1] != self.channels:
            raise ValueError("Joint latent channel count does not match the frozen codec")
        result = []
        with torch.autocast(joint.device.type, enabled=False):
            for z in joint.float().split(self.codec.config.embedding_dim, 1):
                if self.decode_mode == "requantize":
                    z, _ = self.codec.quantizer(z)
                if self.checkpointing and torch.is_grad_enabled() and z.requires_grad:
                    result.append(checkpoint(self.codec.decode,z,use_reentrant=False))
                else:
                    result.append(self.codec.decode(z))
        return torch.cat(result, 1)


def load_codec(path, *, trusted_legacy=False):
    """Legacy pickle opt-in only; never load untrusted checkpoint objects."""
    p = torch.load(Path(path), map_location="cpu", weights_only=not trusted_legacy)
    if p.get("schema") in ("trjepa_codec_v1", "first_post_unregistered_tumor_roi_v1", "symm_world_codec_v2"):
        cfg, state = p["model_config"], p["codec_state"]
    else:
        marker = p.get("mewm_ispy2_vqgan_identity", {})
        if (marker.get("numeric_contract") != "ispy2_first_post_unregistered_train_zscore_v1"
                or marker.get("phase_index") != 1 or marker.get("registered") is not False):
            raise ValueError("Unknown codec checkpoint contract; conversion must be explicit")
        cfg = marker["architecture_contract"]["config"]
        state = {k.removeprefix("autoencoder."): v for k, v in p["state_dict"].items() if k.startswith("autoencoder.")}
    model = MRILevelVQGAN(VQGANConfig(**cfg))
    model.load_state_dict(state, strict=True)
    return model


class PatchDiscriminator3D(nn.Module):
    """Multi-scale PatchGAN discriminator; spectral-normalized convolution blocks."""
    def __init__(self, width=32, levels=3):
        super().__init__()
        from torch.nn.utils.parametrizations import spectral_norm
        layers, inc = [], 1
        for i in range(levels):
            out = width * min(2**i, 8)
            layers.extend([spectral_norm(nn.Conv3d(inc, out, 3, 2, 1)), nn.LeakyReLU(.2)])
            inc = out
        layers.append(spectral_norm(nn.Conv3d(inc, 1, 3, 1, 1)))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
