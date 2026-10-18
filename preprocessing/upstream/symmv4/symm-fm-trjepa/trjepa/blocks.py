"""Adapted research blocks, with 3D positions and PyTorch SDPA.

ViT attention / pre-LN residual pattern: Meta I-JEPA vision_transformer.py.
AdaLN-Zero modulation: Meta DiT models.py. Both CC-BY-NC-4.0.
ISAB/PMA topology and MAB residual: Juho Lee Set Transformer, MIT.
Changes: modern SDPA, 3D coordinates, explicit missing-key masking,
pre-normalized cross attention and independently gated cross-attention in DiT.
See docs/SOURCES.md and licenses/ (no upstream pretrained weights included).
"""
from __future__ import annotations
import math
from typing import Sequence
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def fourier(t: Tensor, dim: int, max_period: float = 10000.) -> Tensor:
    if dim < 2:
        raise ValueError("Fourier width must be >= 2")
    half = dim // 2
    f = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device).float() / half)
    a = t.float().unsqueeze(-1) * f
    x = torch.cat((a.cos(), a.sin()), -1)
    return F.pad(x, (0, dim - x.shape[-1]))


def pos3d(grid: Sequence[int], dim: int, device=None, dtype=torch.float32) -> Tensor:
    """ZYX order, physical axis semantics are not video-time semantics."""
    if dim % 6:
        raise ValueError("3D sine/cosine dimension must be divisible by 6")
    axes = torch.meshgrid(*(torch.arange(n, device=device) for n in grid), indexing="ij")
    return torch.cat([fourier(a.reshape(-1), dim // 3) for a in axes], -1).to(dtype)[None]


def gather(x: Tensor, indices: Tensor) -> Tensor:
    return x.gather(1, indices[..., None].expand(-1, -1, x.shape[-1]))


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float = 4., out: int | None = None):
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim * ratio))
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(int(dim * ratio), out or dim)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, cross: bool = False):
        super().__init__()
        if dim % heads:
            raise ValueError("Attention dimension must be divisible by heads")
        self.heads, self.cross = heads, cross
        if cross:
            self.q = nn.Linear(dim, dim)
            self.kv = nn.Linear(dim, 2 * dim)
        else:
            self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x: Tensor, context: Tensor | None = None, valid: Tensor | None = None):
        b, n, d = x.shape
        if self.cross:
            if context is None:
                raise ValueError("Cross-attention requires context")
            q = self.q(x).reshape(b, n, self.heads, d // self.heads).transpose(1, 2)
            kv = self.kv(context).reshape(b, -1, 2, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
        else:
            q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        mask = None
        if valid is not None:
            if not valid.any(1).all():
                raise ValueError("All-missing attention row: provide an explicit missing token")
            mask = valid[:, None, None, :].bool()
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
        return self.proj(y.transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.attn, self.mlp = Attention(dim, heads), MLP(dim)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class CrossBlock(nn.Module):
    """Perceiver-style latent query update, followed by query self-attention."""
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.qn, self.kn = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.cross = Attention(dim, heads, cross=True)
        self.block = Block(dim, heads)
    def forward(self, x, context, valid=None):
        return self.block(x + self.cross(self.qn(x), self.kn(context), valid))


class MAB(nn.Module):
    """Set Transformer MAB, adapted from Juho Lee's official modules.py.

    Original scaling sqrt(dim_V) is preserved (rather than SDPA's sqrt(head_dim)).
    Padding mask is the only functional extension to the original MAB equations.
    """
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError("Set attention heads do not divide dimension")
        self.dim_V, self.num_heads = dim, heads
        self.fc_q, self.fc_k, self.fc_v = [nn.Linear(dim, dim) for _ in range(3)]
        self.fc_o = nn.Linear(dim, dim)
        self.ln0, self.ln1 = nn.LayerNorm(dim), nn.LayerNorm(dim)
    def forward(self, Q, K, valid=None):
        q, k, v = self.fc_q(Q), self.fc_k(K), self.fc_v(K)
        w = self.dim_V // self.num_heads
        q_, k_, v_ = [torch.cat(t.split(w, 2), 0) for t in (q, k, v)]
        score = q_.bmm(k_.transpose(1, 2)) / math.sqrt(self.dim_V)
        if valid is not None:
            if not valid.any(1).all():
                raise ValueError("Set has no valid key")
            score = score.masked_fill(~valid.repeat(self.num_heads, 1)[:, None], -torch.inf)
        a = score.softmax(2)
        o = torch.cat((q_ + a.bmm(v_)).split(q.size(0), 0), 2)
        o = self.ln0(o)
        return self.ln1(o + F.relu(self.fc_o(o)))


class ISAB(nn.Module):
    def __init__(self, dim, heads, inducing=16):
        super().__init__()
        self.I = nn.Parameter(torch.empty(1, inducing, dim))
        nn.init.xavier_uniform_(self.I)
        self.mab0, self.mab1 = MAB(dim, heads), MAB(dim, heads)
    def forward(self, x, valid=None):
        h = self.mab0(self.I.expand(x.shape[0], -1, -1), x, valid)
        return self.mab1(x, h)


class PMA(nn.Module):
    def __init__(self, dim, heads, seeds=4):
        super().__init__()
        self.S = nn.Parameter(torch.empty(1, seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, heads)
    def forward(self, x, valid=None):
        return self.mab(self.S.expand(x.shape[0], -1, -1), x, valid)


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    """Meta DiT adaLN-Zero block + separately gated semantic cross-attention."""
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.normc = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.context_norm = nn.LayerNorm(dim)
        self.attn, self.cross, self.mlp = Attention(dim, heads), Attention(dim, heads, True), MLP(dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    def forward(self, x, c, context):
        s1, a1, g1, sc, ac, gc, s2, a2, g2 = self.adaLN_modulation(c).chunk(9, 1)
        x = x + g1[:, None] * self.attn(modulate(self.norm1(x), s1, a1))
        x = x + gc[:, None] * self.cross(modulate(self.normc(x), sc, ac), self.context_norm(context))
        return x + g2[:, None] * self.mlp(modulate(self.norm2(x), s2, a2))


def run_blocks(blocks, x, *, checkpointing=False, args=()):
    for block in blocks:
        if checkpointing and torch.is_grad_enabled() and block.training:
            x = checkpoint(block, x, *args, use_reentrant=False)
        else:
            x = block(x, *args)
    return x
