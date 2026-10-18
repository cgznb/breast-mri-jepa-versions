"""Multi-scale MRI encoder + shared query backbone + anchored residual world model.

Derived conceptually from the user's VLA-JEPA architecture, not a Qwen/V-JEPA2
checkpoint port. Teacher targets and teacher-forced predictions NEVER enter pCR.
"""
from __future__ import annotations
from dataclasses import replace
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .config import ModelConfig
from .contracts import Inputs


def groups(channels):
    return math.gcd(channels, 8)

class TransformerStack(nn.Module):
    """Independent initialization of each Pre-LN layer; no cloned initial weights."""
    def __init__(self, dim, heads, depth, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(depth)])
    def forward(self, x, mask=None, padding=None):
        for layer in self.layers:
            x = layer(x, src_mask=mask, src_key_padding_mask=padding)
        return x

class Residual3D(nn.Module):
    """Depthwise spatial mixing + pointwise expansion; GroupNorm is batch-size independent."""
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv3d(channels, channels, 3, padding=1, groups=channels)
        self.norm = nn.GroupNorm(groups(channels), channels)
        self.expand = nn.Conv3d(channels, 2 * channels, 1)
        self.project = nn.Conv3d(2 * channels, channels, 1)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1, 1), 1e-3))
    def forward(self, x):
        y = self.project(F.gelu(self.expand(self.norm(self.depthwise(x)))))
        return x + self.scale * y

class MultiScaleEncoder(nn.Module):
    """Default [3,32,128,128] -> [256,128], retaining a 4x8x8 local grid.
    No claim of lesion segmentation or identifiable anatomy/response disentanglement.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.grid = tuple(cfg.token_grid)
        w = cfg.widths
        self.stem = nn.Sequential(nn.Conv3d(3, w[0], (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
                                  nn.GroupNorm(groups(w[0]), w[0]), nn.GELU())
        self.stages = nn.ModuleList()
        for i in range(4):
            layers = []
            if i:
                layers += [nn.Conv3d(w[i-1], w[i], 3, stride=2, padding=1),
                           nn.GroupNorm(groups(w[i]), w[i]), nn.GELU()]
            layers += [Residual3D(w[i]) for _ in range(cfg.depths[i])]
            self.stages.append(nn.Sequential(*layers))
        self.lateral = nn.ModuleList([nn.Conv3d(w[i], cfg.dim, 1) for i in (2, 3)])
        self.position = nn.Parameter(torch.randn(1, math.prod(self.grid), cfg.dim) * .02)
        self.spatial = TransformerStack(cfg.dim, cfg.heads, cfg.spatial_depth, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.dim)

    def forward(self, images):
        if images.ndim != 5 or images.shape[1] != 3:
            raise ValueError("Visit encoder expects [visits,3,D,H,W]")
        x = self.stem(images)
        scales = []
        for i, stage in enumerate(self.stages):
            if self.cfg.activation_checkpoint and self.training and torch.is_grad_enabled():
                x = checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
            if i in (2, 3):
                scales.append(F.adaptive_avg_pool3d(self.lateral[i-2](x), self.grid))
        x = (scales[0] + scales[1]) / math.sqrt(2)
        tokens = x.flatten(2).transpose(1, 2) + self.position
        return self.norm(self.spatial(tokens))

class LegacySmallEncoder(nn.Module):
    """Upstream encoder topology as a control, NOT an exact historical-run reproduction.
    The v2 trainer, query masking and layer initialization are intentionally different.
    """
    def __init__(self, cfg):
        super().__init__()
        hidden = max(8, cfg.dim // 2)
        self.stem = nn.Sequential(
            nn.Conv3d(3, hidden, 3, stride=2, padding=1),
            nn.GroupNorm(groups(hidden), hidden), nn.GELU(),
            nn.Conv3d(hidden, cfg.dim, 3, stride=2, padding=1),
            nn.GroupNorm(groups(cfg.dim), cfg.dim), nn.GELU(),
            nn.AdaptiveAvgPool3d(cfg.token_grid))
        self.position = nn.Parameter(torch.randn(1, math.prod(cfg.token_grid), cfg.dim) * .02)
        self.spatial = TransformerStack(cfg.dim, cfg.heads, cfg.spatial_depth, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.dim)
    def forward(self, x):
        return self.norm(self.spatial(self.stem(x).flatten(2).transpose(1, 2) + self.position))

class AttentionPool(nn.Module):
    """Learned queries read spatial tokens without destroying the world target grid."""
    def __init__(self, dim, heads, queries, dropout):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, queries, dim) * .02)
        self.qnorm, self.knorm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4*dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(4*dim, dim))
    def forward(self, context):
        q = self.queries.expand(len(context), -1, -1)
        kv = self.knorm(context)
        y = self.attention(self.qnorm(q), kv, kv, need_weights=False)[0]
        q = q + y
        return q + self.ffn(q)

class WorldPredictor(nn.Module):
    """Causal [transition queries, source-state spatial tokens] blocks.

    With residual_world=True and zero delta projection, the initial model is
    exactly copy, even though the internal Transformer is randomly initialized.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n = math.prod(cfg.token_grid)
        self.k = cfg.state_queries
        self.state_embed = nn.Linear(cfg.dim, cfg.dim)
        self.action_embed = nn.Linear(cfg.dim, cfg.dim)
        self.spatial_position = nn.Parameter(torch.randn(1, 1, self.n, cfg.dim) * .02)
        self.query_position = nn.Parameter(torch.randn(1, 1, self.k, cfg.dim) * .02)
        self.time = nn.Embedding(3, cfg.dim)
        self.root = nn.Parameter(torch.randn(1, 1, cfg.dim) * .02)
        self.blocks = TransformerStack(cfg.dim, cfg.heads, cfg.predictor_depth, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.dim)
        self.output = nn.Linear(cfg.dim, cfg.dim)
        if cfg.residual_world:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        else:
            nn.init.xavier_uniform_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, source, transition, active):
        b, t, n, d = source.shape
        if t > 3 or t < 1 or n != self.n or transition.shape != (b, t, self.k, d):
            raise ValueError("Invalid world input shape")
        if active.shape != (b, t) or active.dtype != torch.bool:
            raise ValueError("World active mask must be boolean [B,T]")
        mask = active[:, :, None, None]
        # Inactive NaNs cannot contaminate active attention through linear layers.
        source = source.masked_fill(~mask, 0)
        transition = transition.masked_fill(~mask, 0)
        time = self.time.weight[:t][None, :, None]
        state = self.state_embed(source) + self.spatial_position + time
        act = self.action_embed(transition) + self.query_position + time
        length = self.k + self.n
        x = torch.cat((act, state), 2).flatten(1, 2)
        x = torch.cat((self.root.expand(b, -1, -1), x), 1)
        padding = torch.cat((torch.zeros(b, 1, device=x.device, dtype=torch.bool),
                             ~active[:, :, None].expand(-1, -1, length).flatten(1)), 1)
        times = torch.cat((torch.tensor([-1], device=x.device),
                           torch.arange(t, device=x.device).repeat_interleave(length)))
        causal = times[None, :] > times[:, None]
        y = self.blocks(x, mask=causal, padding=padding)
        y = y[:, 1:].reshape(b, t, length, d)[:, :, self.k:]
        y = self.output(self.norm(y))
        if self.cfg.residual_world:
            y = source + y
        return y.masked_fill(~mask, 0)

class LinearWorldPredictor(nn.Module):
    """Token-wise linear residual control, conditioned on the mean transition query.
    The shared context is still nonlinear: this is a linear TRANSITION control,
    not a claim that the whole medical model is a linear system.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.output = nn.Linear(2 * cfg.dim, cfg.dim)
        nn.init.zeros_(self.output.bias)
        if cfg.residual_world:
            nn.init.zeros_(self.output.weight)
        else:
            nn.init.xavier_uniform_(self.output.weight)

    def forward(self, source, transition, active):
        mask = active[:, :, None, None]
        source = source.masked_fill(~mask, 0)
        transition = transition.masked_fill(~mask, 0)
        condition = transition.mean(2, keepdim=True).expand(-1, -1, source.shape[2], -1)
        result = self.output(torch.cat((source, condition), -1))
        if self.cfg.residual_world:
            result = result + source
        return result.masked_fill(~mask, 0)

class ResponseJEPA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        encoder = MultiScaleEncoder if cfg.encoder == "multiscale" else LegacySmallEncoder
        self.encoder = encoder(cfg)
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False).eval()
        self.context_pool = AttentionPool(cfg.dim, cfg.heads, cfg.context_queries, cfg.dropout)
        self.stage = nn.Embedding(4, cfg.dim)
        self.clinical_value = nn.Linear(2, cfg.dim)
        self.clinical_field = nn.Parameter(torch.randn(1, 4, cfg.dim) * .02)
        self.arm = nn.Embedding(14, cfg.dim)
        self.arm_known_at = nn.Embedding(5, cfg.dim)  # zero = unknown, one..four = T0..T3
        self.root = nn.Parameter(torch.randn(1, 1, cfg.dim) * .02)
        self.dynamics_queries = nn.Parameter(torch.randn(3, cfg.state_queries, cfg.dim) * .02)
        self.pcr_queries = nn.Parameter(torch.randn(1, cfg.pcr_queries, cfg.dim) * .02)
        self.fusion = TransformerStack(cfg.dim, cfg.heads, cfg.fusion_depth, cfg.dropout)
        self.fusion_norm = nn.LayerNorm(cfg.dim)
        self.direct_head = nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, 1))
        self.world = (WorldPredictor(cfg) if cfg.world_arch == "transformer" else LinearWorldPredictor(cfg))
        self.terminal_pool = AttentionPool(cfg.dim, cfg.heads, cfg.pcr_queries, cfg.dropout)
        self.terminal_head = nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, 1))
        self.gate = nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, cfg.gate_bias)
        self.mask_predictor = nn.Sequential(
            TransformerStack(cfg.dim, cfg.heads, 2, cfg.dropout),
            nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, cfg.dim))
        self.repr_projector = nn.Sequential(nn.Linear(cfg.dim, 2*cfg.dim), nn.GELU(),
                                             nn.Linear(2*cfg.dim, cfg.dim))
        self.repr_head = nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, 1))
        self.encoder_frozen = False

    def train(self, mode=True):
        super().train(mode)
        self.target_encoder.eval()
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def configure_trainable(self, phase, freeze_encoder):
        self.requires_grad_(True)
        self.target_encoder.requires_grad_(False)
        if phase == "A":
            self.requires_grad_(False)
            for module in (self.encoder, self.mask_predictor, self.repr_projector, self.repr_head):
                module.requires_grad_(True)
            self.encoder_frozen = False
        elif phase == "joint":
            self.encoder.requires_grad_(True)
            self.encoder_frozen = False
        else:
            for module in (self.mask_predictor, self.repr_projector, self.repr_head):
                module.requires_grad_(False)
            self.encoder.requires_grad_(not freeze_encoder and self.cfg.use_mri)
            self.encoder_frozen = freeze_encoder or not self.cfg.use_mri
            if phase == "B":
                for module in (self.direct_head, self.terminal_head, self.terminal_pool, self.gate):
                    module.requires_grad_(False)
                self.pcr_queries.requires_grad_(False)
        self.train()

    @torch.no_grad()
    def update_teacher(self, decay):
        for teacher, online in zip(self.target_encoder.parameters(), self.encoder.parameters(), strict=True):
            teacher.lerp_(online, 1 - decay)
        for teacher, online in zip(self.target_encoder.buffers(), self.encoder.buffers(), strict=True):
            teacher.copy_(online)

    def encode_selected(self, images, mask, teacher=False):
        module = self.target_encoder if teacher else self.encoder
        n, d = math.prod(self.cfg.token_grid), self.cfg.dim
        out = images.new_zeros(len(images), 4, n, d)
        selected = images[mask]
        if not len(selected):
            return out
        chunks = []
        with torch.set_grad_enabled(torch.is_grad_enabled() and not teacher and not self.encoder_frozen):
            for offset in range(0, len(selected), self.cfg.encoder_chunk):
                chunks.append(module(selected[offset:offset + self.cfg.encoder_chunk]))
            values = torch.cat(chunks, 0)
        # CopySlices retains gradients for online encoder fine-tuning.
        out[mask] = values.to(out.dtype)
        return out

    def condition_tokens(self, inp):
        c = inp.clinical if self.cfg.use_clinical else torch.zeros_like(inp.clinical)
        cm = inp.clinical_mask if self.cfg.use_clinical else torch.zeros_like(inp.clinical_mask)
        clinical = self.clinical_value(torch.stack((c, cm.to(c.dtype)), -1)) + self.clinical_field
        aid = inp.arm_id if self.cfg.use_arm else torch.zeros_like(inp.arm_id)
        when = inp.arm_known_at + 1 if self.cfg.use_arm else torch.zeros_like(inp.arm_known_at)
        arm = self.arm(aid) + self.arm_known_at(when)
        return torch.cat((clinical, arm[:, None]), 1)

    def backbone(self, inp: Inputs):
        inp.validate()
        b, d, q = len(inp.images), self.cfg.dim, self.cfg.context_queries
        packed_mri = inp.images.new_zeros(b, 4, q, d)
        online = inp.images.new_zeros(b, 4, math.prod(self.cfg.token_grid), d)
        if self.cfg.use_mri:
            online = self.encode_selected(inp.images, inp.observed)
            pooled = self.context_pool(online[inp.observed])
            packed_mri[inp.observed] = pooled.to(packed_mri.dtype)
            mri_mask = inp.observed
        else:
            mri_mask = torch.zeros_like(inp.observed)
        packed_mri = packed_mri + self.stage.weight[None, :, None]
        cond = self.condition_tokens(inp)
        context = torch.cat((self.root.expand(b, -1, -1), packed_mri.flatten(1, 2), cond), 1)
        context_padding = torch.cat((torch.zeros(b, 1, device=inp.images.device, dtype=torch.bool),
                                      ~mri_mask[:, :, None].expand(-1, -1, q).flatten(1),
                                      torch.zeros(b, 5, device=inp.images.device, dtype=torch.bool)), 1)
        # Fixed INTERNAL slots. inp.query_mask is deliberately never consulted.
        dyn = self.dynamics_queries[None] + self.stage.weight[None, 1:, None]
        dyn = dyn.expand(b, -1, -1, -1)
        pcr = self.pcr_queries.expand(b, -1, -1) + self.stage(inp.landmark)[:, None]
        x = torch.cat((context, dyn.flatten(1, 2), pcr), 1)
        padding = F.pad(context_padding, (0, 3*self.cfg.state_queries + self.cfg.pcr_queries), value=False)
        causal = torch.ones(len(x[0]), len(x[0]), device=x.device, dtype=torch.bool).triu(1)
        causal[:context.shape[1], :context.shape[1]] = False
        y = self.fusion_norm(self.fusion(x, mask=causal, padding=padding))
        start = context.shape[1]
        transitions = y[:, start:start + 3*self.cfg.state_queries].reshape(b, 3, self.cfg.state_queries, d)
        feature = y[:, -self.cfg.pcr_queries:].mean(1)
        return {"direct_logit": self.direct_head(feature).squeeze(-1), "feature": feature,
                "transitions": transitions, "condition_tokens": cond, "online_states": online}

    def rollout(self, inp, transitions, *, copy_only=False):
        """Differentiable free rollout from the latest legal real MRI; supports missing intermediates."""
        if not self.cfg.use_mri:
            raise ValueError("World forecasting requires MRI")
        anchor = inp.anchor()
        rows = torch.arange(len(inp.images), device=inp.images.device)
        anchor_mask = F.one_hot(anchor, 4).bool()
        states = self.encode_selected(inp.images, anchor_mask, teacher=True)
        initial = states[rows, anchor]
        if copy_only:
            # The capacity-control head sees no learned future change.
            for j in range(4):
                active = anchor <= j
                states = states.clone()
                states[active, j] = initial[active]
            return states, anchor
        for j in range(3):
            updating = anchor <= j
            if not updating.any():
                continue
            source_stages = torch.arange(j+1, device=inp.images.device)[None]
            active = (source_stages >= anchor[:, None]) & updating[:, None]
            predicted = self.world(states[:, :j+1], transitions[:, :j+1], active)
            states = states.clone()     # avoid in-place overwrite of autograd predecessors
            states[updating, j+1] = predicted[updating, j].to(states.dtype)
        return states, anchor

    def forward(self, inp: Inputs, *, need_world=False, request_states=False):
        out = self.backbone(inp)
        direct = out["direct_logit"]
        future = self.cfg.readout != "direct"
        states = None
        if future or need_world or request_states:
            states, anchor = self.rollout(inp, out["transitions"])
            out.update(states=states, anchor=anchor)
        out.update(forecast_logit=None, gate=None)
        if future:
            # Copy readout changes ONLY the task head input, not the auxiliary rollout objective.
            terminal = (states[torch.arange(len(states), device=states.device), anchor]
                        if self.cfg.readout == "copy_gated" else states[:, 3])
            if self.cfg.forecast_detach:
                terminal = terminal.detach()
            tokens = torch.cat((terminal, out["condition_tokens"], self.stage(inp.landmark)[:, None]), 1)
            f = self.terminal_head(self.terminal_pool(tokens).mean(1)).squeeze(-1)
            out["forecast_logit"] = f
            if self.cfg.readout == "forecast":
                final = f
            else:
                gate_logit = self.gate(out["feature"]).squeeze(-1).float()
                # Stable probability mixture in log space, rather than averaging logits.
                logp = torch.logaddexp(F.logsigmoid(-gate_logit) + F.logsigmoid(direct.float()),
                                       F.logsigmoid(gate_logit) + F.logsigmoid(f.float()))
                logq = torch.logaddexp(F.logsigmoid(-gate_logit) + F.logsigmoid(-direct.float()),
                                       F.logsigmoid(gate_logit) + F.logsigmoid(-f.float()))
                final = logp - logq
                out["gate"] = gate_logit.sigmoid()
        else:
            final = direct
        out["logit"] = final
        if request_states:
            # Requested outputs only: classification and internal rollout are invariant.
            out["requested_states"] = states.masked_fill(~inp.query_mask[:, :, None, None], 0)
        return out

    @torch.no_grad()
    def teacher_states(self, inp, targets):
        """LOSS-ONLY helper. No caller may route its future values to the task head."""
        targets.validate(inp)
        images = inp.images.clone()
        images[targets.future_mask] = targets.future[targets.future_mask]
        known = inp.observed | targets.future_mask
        return self.encode_selected(images, known, teacher=True), known

    def teacher_forcing(self, inp, targets, transitions):
        states, known = self.teacher_states(inp, targets)
        source_stages = torch.arange(3, device=inp.images.device)[None]
        # Only requested-in-time FUTURE targets, independent of the UI output request.
        active = known[:, :-1] & (source_stages + 1 > inp.landmark[:, None])
        pair_mask = active & known[:, 1:]
        predictions = self.world(states[:, :-1], transitions, active)
        return predictions, states, known, pair_mask
