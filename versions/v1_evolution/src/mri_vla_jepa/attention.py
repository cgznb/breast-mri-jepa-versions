"""Pre-norm Transformers with optional independent attention inner width."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _WideSelfAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.qkv = nn.Linear(dim, 3 * heads * head_dim)
        self.out_proj = nn.Linear(heads * head_dim, dim)
        self.dropout = dropout

    def forward(self, value, mask=None, key_padding_mask=None, is_causal=False):
        batch, length, _ = value.shape
        qkv = self.qkv(value).reshape(batch, length, 3, self.heads, self.head_dim)
        query, key, val = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # Transformer masks mark True as blocked. SDPA bool masks mark True as
        # allowed, so use an additive mask to avoid reversing this convention.
        additive = None
        if mask is not None:
            if mask.ndim == 2 and mask.shape == (length, length):
                mask = mask[None, None]
            elif mask.ndim == 3 and mask.shape == (batch * self.heads, length, length):
                mask = mask.reshape(batch, self.heads, length, length)
            else:
                raise ValueError("Attention mask must be [L,L] or [B*heads,L,L]")
            additive = self._additive(mask, query)
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, length):
                raise ValueError("Key padding mask must be [B,L]")
            padding = self._additive(key_padding_mask[:, None, None], query)
            additive = padding if additive is None else additive + padding
        if is_causal:
            blocked = torch.ones(length, length, dtype=torch.bool, device=value.device).triu(1)
            causal = self._additive(blocked[None, None], query)
            additive = causal if additive is None else additive + causal
        output = F.scaled_dot_product_attention(
            query, key, val, attn_mask=additive,
            dropout_p=self.dropout if self.training else 0.,
        )
        return self.out_proj(output.transpose(1, 2).reshape(batch, length, -1))

    @staticmethod
    def _additive(mask, reference):
        if mask.dtype == torch.bool:
            return torch.zeros_like(mask, dtype=reference.dtype, device=reference.device).masked_fill(
                mask.to(reference.device), float("-inf"))
        if not mask.is_floating_point():
            raise ValueError("Attention masks must be boolean or floating point")
        return mask.to(device=reference.device, dtype=reference.dtype)


class _WideTransformerLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = _WideSelfAttention(cfg.dim, cfg.heads, cfg.attention_head_dim, cfg.dropout)
        self.linear1 = nn.Linear(cfg.dim, cfg.ffn_dim or cfg.dim * 4)
        self.linear2 = nn.Linear(cfg.ffn_dim or cfg.dim * 4, cfg.dim)
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.norm2 = nn.LayerNorm(cfg.dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=False):
        src = src + self.dropout1(self.self_attn(
            self.norm1(src), mask=mask, key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        ))
        return src + self.dropout2(self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(src))))))


class WideTransformerEncoder(nn.Module):
    """Encoder with the TransformerEncoder call and blocked-mask semantics."""
    def __init__(self, cfg, depth):
        super().__init__()
        self.layers = nn.ModuleList(_WideTransformerLayer(cfg) for _ in range(depth))
        self.num_layers = depth

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        for layer in self.layers:
            src = layer(src, mask=mask, src_key_padding_mask=src_key_padding_mask,
                        is_causal=bool(is_causal))
        return src


def build_transformer(cfg, depth):
    """Keep historical keys/behavior unless independent attention is requested."""
    if cfg.attention_head_dim is None:
        layer = nn.TransformerEncoderLayer(
            cfg.dim, cfg.heads, cfg.ffn_dim or cfg.dim * 4, cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
    return WideTransformerEncoder(cfg, depth)
