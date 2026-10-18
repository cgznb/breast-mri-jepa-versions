"""3D I-JEPA-style representation pretraining; no temporal targets in stage B."""
from __future__ import annotations
from dataclasses import dataclass
import copy
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .blocks import Block, CrossBlock, pos3d, gather, run_blocks


@dataclass
class State:
    spatial: Tensor                  # [B,N,d], fixed ZYX patch order
    global_token: Tensor             # [B,1,d]
    tumor_token: Tensor              # [B,1,d], source ROI or global fallback
    grid: tuple[int, int, int]
    roi_weights: Tensor              # [B,N], never a future mask in online branch
    def detached(self):
        return State(self.spatial.detach(), self.global_token.detach(), self.tumor_token.detach(),
                     self.grid, self.roi_weights.detach())
    def tokens(self):
        return torch.cat((self.spatial, self.global_token, self.tumor_token), 1)


def state_from_tokens(tokens: Tensor, grid, weights: Tensor | None = None) -> State:
    global_token = tokens.mean(1, keepdim=True)
    if weights is None:
        weights = tokens.new_zeros(tokens.shape[:2])
    mass = weights.sum(1, keepdim=True)
    roi = (tokens * weights[..., None]).sum(1, keepdim=True) / mass.clamp_min(1e-6)[..., None]
    roi = torch.where((mass > 0)[..., None], roi, global_token)
    return State(tokens, global_token, roi, tuple(grid), weights)


class MedicalEncoder3D(nn.Module):
    """Independent 3-channel ViT, non-overlapping 3D patchification.

    The masked encoder selects visible tokens BEFORE any self-attention. No
    overlapping convolution, spatial normalization, or teacher feature can
    communicate a masked patch's pixels to the visible pathway.
    """
    def __init__(self, *, dim=384, depth=12, heads=6, patch=(8,16,16), checkpointing=True):
        super().__init__()
        self.dim, self.patch, self.checkpointing = dim, tuple(patch), checkpointing
        if dim % 6 or dim % heads or depth < 1:
            raise ValueError("Invalid 3D semantic Transformer configuration")
        self.patch_embed = nn.Conv3d(3, dim, self.patch, self.patch)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
    def patch_tokens(self, x):
        if x.ndim != 5 or x.shape[1] != 3 or any(s % p for s,p in zip(x.shape[-3:], self.patch)):
            raise ValueError("Semantic input must be [B,3,D,H,W], divisible by 3D patch shape")
        z = self.patch_embed(x)
        grid = tuple(z.shape[-3:])
        tokens = z.flatten(2).transpose(1,2)
        return tokens + pos3d(grid, self.dim, x.device, tokens.dtype), grid
    def forward(self, x, roi=None, visible=None):
        tokens, grid = self.patch_tokens(x)
        if visible is not None:
            if visible.shape[0] != x.shape[0] or visible.shape[1] < 1:
                raise ValueError("At least one visible token per sample is required")
            tokens = gather(tokens, visible)
        tokens = self.norm(run_blocks(self.blocks, tokens, checkpointing=self.checkpointing))
        if visible is not None:
            return tokens
        weights = None
        if roi is not None:
            if roi.shape != (x.shape[0], 1, *x.shape[-3:]):
                raise ValueError("ROI must match the current image geometry")
            weights = F.avg_pool3d(roi.float(), self.patch, self.patch).flatten(1)
        # Deterministic pooling avoids freezing random, never-pretrained ROI queries.
        # Learnable attention readout lives in the response/dynamics network instead.
        return state_from_tokens(tokens, grid, weights)


class MedicalMasker:
    """Mixed anatomical block, ROI, boundary and multiscale masking.

    All modes return equal-length complementary indices so no batch padding
    accidentally leaks masked tokens. ROI selection is a pretraining curriculum,
    not evidence that a token has learned a particular medical interpretation.
    """
    MODES = ("anatomical", "tumor", "boundary", "multiscale")
    def __init__(self, ratio=.5):
        if not 0.05 <= ratio <= .9:
            raise ValueError("Mask ratio must be between .05 and .9")
        self.ratio = ratio
    def __call__(self, batch_size, grid, device, roi=None, mode=None):
        n = math.prod(grid)
        k = max(1, min(n-1, round(n*self.ratio)))
        if n < 2:
            raise ValueError("Masked JEPA requires at least two spatial patches")
        mode = mode or self.MODES[int(torch.randint(len(self.MODES), ()).item())]
        if mode not in self.MODES:
            raise ValueError("Unknown mask curriculum")
        score = torch.rand(batch_size, *grid, device=device)
        for b in range(batch_size):
            blocks = 3 if mode == "multiscale" else 1
            for _ in range(blocks):
                widths = [max(1, int(s*(.2 + .5*torch.rand(()).item()))) for s in grid]
                starts = [int(torch.randint(s-w+1, ()).item()) for s,w in zip(grid,widths)]
                slices = tuple(slice(a,a+w) for a,w in zip(starts,widths))
                score[(b,*slices)] += 2.
        if roi is not None and mode in ("tumor", "boundary"):
            r = F.adaptive_max_pool3d(roi.float(), grid)
            if mode == "boundary":
                dilated = F.max_pool3d(r, 3, 1, 1)
                eroded = -F.max_pool3d(-r, 3, 1, 1)
                r = (dilated-eroded).clamp(0,1)
            score += r[:,0] * 4.
        order = score.flatten(1).argsort(1, descending=True)
        return order[:,k:].sort(1).values, order[:,:k].sort(1).values


class MaskPredictor(nn.Module):
    def __init__(self, dim=384, predictor_dim=192, depth=6, heads=6, checkpointing=True):
        super().__init__()
        self.dim, self.checkpointing = predictor_dim, checkpointing
        self.predictor_embed = nn.Linear(dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1,1,predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=.02)
        self.blocks = nn.ModuleList([Block(predictor_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(predictor_dim, eps=1e-6)
        self.out = nn.Linear(predictor_dim, dim)
    def forward(self, context, visible, target, grid):
        b = context.shape[0]
        p = pos3d(grid, self.dim, context.device, context.dtype).expand(b,-1,-1)
        c = self.predictor_embed(context) + gather(p,visible)
        q = self.mask_token.expand(b,target.shape[1],-1) + gather(p,target)
        sequence = torch.cat((c,q),1)
        y = run_blocks(self.blocks,sequence,checkpointing=self.checkpointing)
        return self.out(self.norm(y[:,c.shape[1]:]))


class MaskedJEPA(nn.Module):
    def __init__(self, encoder: MedicalEncoder3D, predictor: MaskPredictor, mask_ratio=.5):
        super().__init__()
        self.online = encoder
        self.target = copy.deepcopy(encoder).eval().requires_grad_(False)
        self.predictor = predictor
        self.masker = MedicalMasker(mask_ratio)
    def train(self, mode=True):
        super().train(mode)
        self.target.eval()
        return self
    @torch.no_grad()
    def update_teacher(self, momentum):
        if not 0 <= momentum < 1:
            raise ValueError("EMA momentum must be in [0,1)")
        for target, online in zip(self.target.parameters(), self.online.parameters()):
            target.lerp_(online, 1-momentum)
        for target, online in zip(self.target.buffers(), self.online.buffers()):
            target.copy_(online)
    def forward(self, image, roi=None, mode=None):
        grid = tuple(s//p for s,p in zip(image.shape[-3:],self.online.patch))
        visible, target = self.masker(image.shape[0],grid,image.device,roi,mode)
        context = self.online(image,visible=visible)
        with torch.no_grad():
            teacher = self.target(image).spatial
            targets = gather(teacher,target)
        prediction = self.predictor(context,visible,target,grid)
        loss = F.l1_loss(prediction.float(),targets.float())
        # Small explicit variance term; diagnostics still required to detect collapse.
        std = context.float().flatten(0,1).std(0,unbiased=False)
        variance = F.relu(.5-std).mean()
        return loss+.01*variance, {"masked_l1":loss.detach(), "variance_penalty":variance.detach(),
                                   "context_std":std.mean().detach(), "teacher_std":teacher.float().std(1).mean().detach()}


class TreatmentResponse(nn.Module):
    def __init__(self, dim=384, heads=6, depth=4, queries=8, checkpointing=True):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1,queries,dim)*.02)
        self.blocks = nn.ModuleList([CrossBlock(dim,heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.checkpointing = checkpointing
    def forward(self, state: State, clinical_tokens):
        context = torch.cat((state.tokens(),clinical_tokens),1)
        q = self.queries.expand(state.spatial.shape[0],-1,-1)
        return self.norm(run_blocks(self.blocks,q,checkpointing=self.checkpointing,args=(context,)))


class DynamicsPredictor(nn.Module):
    """A NEW longitudinal predictor. It is not the discarded MaskPredictor."""
    def __init__(self, dim=384, heads=6, depth=6, checkpointing=True):
        super().__init__()
        self.blocks = nn.ModuleList([CrossBlock(dim,heads) for _ in range(depth)])
        self.norm, self.delta = nn.LayerNorm(dim), nn.Linear(dim,dim)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        self.checkpointing = checkpointing
    def forward(self, source: State, response):
        y = run_blocks(self.blocks,source.spatial,checkpointing=self.checkpointing,args=(response,))
        future = source.spatial + self.delta(self.norm(y))
        # Pool using SOURCE ROI only. No target lesion location is supplied.
        return state_from_tokens(future,source.grid,source.roi_weights)


def semantic_loss(pred: State, target: State, *, roi=None, spatial_valid=None, cosine_weight=.1):
    """Spatial terms require verified correspondence; global comparison is separate."""
    a,b = pred.spatial.float(),target.spatial.detach().float()
    weights = torch.ones_like(a[:,:,0])
    if roi is not None:
        weights = weights + 2*F.adaptive_max_pool3d(roi.float(),pred.grid).flatten(1)
    spatial = ((a-b).abs().mean(-1) + cosine_weight*(1-F.cosine_similarity(a,b,dim=-1)))
    per_sample = (spatial*weights).sum(1)/weights.sum(1)
    if spatial_valid is not None:
        w = spatial_valid.to(per_sample.dtype)
        spatial_term = (per_sample*w).sum()/w.sum().clamp_min(1)
    else:
        spatial_term = per_sample.mean()
    global_term = F.l1_loss(pred.global_token.float(),target.global_token.detach().float())
    return spatial_term + .25*global_term
