"""Phase-specific, equal-case losses. Acquisition support is not foreground.

These are project-specific medical adaptations, not claims of a new FM theorem.
ROI supervision is a separate normalized term; it cannot vanish just because a
lesion occupies 0.1% of the crop. Empty ROIs do not divide by zero or invent labels.
"""
from __future__ import annotations
import torch
from torch import Tensor
import torch.nn.functional as F


def latent_support(support: Tensor, shape, channels: int, *, threshold: float=0.999) -> Tensor:
    if support.ndim!=5 or support.shape[1]!=3 or channels%3:
        raise ValueError("Expected three acquisition supports and phase-ordered latent channels")
    pooled=F.adaptive_avg_pool3d(support.float(),shape)
    # Conservative block support avoids marking an unobserved phase as observed.
    # It is NOT a guarantee that the whole encoder receptive field is supported.
    return (pooled>=threshold).float().repeat_interleave(channels//3,dim=1)


def _phase_average(error: Tensor, weight: Tensor, phases: int=3) -> Tensor:
    weight=weight.expand_as(error).float()
    if error.shape[1]%phases:
        raise ValueError("Error channels must preserve the three-phase order")
    b,c=error.shape[:2]
    num=(error.float()*weight).reshape(b,phases,-1).sum(-1)
    den=weight.reshape(b,phases,-1).sum(-1)
    usable=den>0
    per=num/den.clamp_min(1.)
    # Equal phase weight within each case, then equal case weight, not voxel mass.
    case=per.sum(-1)/usable.sum(-1).clamp_min(1)
    present=usable.any(-1)
    return case.sum()/present.sum().clamp_min(1)


def balanced_loss(pred: Tensor, target: Tensor, support: Tensor, *, roi: Tensor|None=None,
                  roi_weight: float=0., kind: str="mse", require_support: bool=True) -> Tensor:
    if pred.shape!=target.shape:raise ValueError("Prediction and supervision shapes differ")
    if require_support and not (support.reshape(support.shape[0],-1).sum(1)>0).all():
        raise ValueError("A training case has no valid phase support; inspect registration/FOV instead of training on padding")
    if roi_weight<0:raise ValueError("ROI loss weight must be nonnegative")
    error=(pred.float()-target.float()).abs()
    if kind=="mse":error=error.square()
    elif kind!="l1":raise ValueError("Unknown balanced loss kind")
    loss=_phase_average(error,support)
    if roi is not None and roi_weight:
        lesion=F.adaptive_max_pool3d(roi.float(),pred.shape[-3:])
        loss=loss+roi_weight*_phase_average(error,support*lesion)
    return loss


def teacher_forced_endpoint(state, velocity, tau, source, mode):
    from .flow import estimate_endpoint
    return estimate_endpoint(state,tau,velocity,source,mode)


def sample_flow_time(batch:int,device,*,strategy:str="uniform",logit_mean:float=0.,logit_std:float=1.):
    if strategy=="uniform":return torch.rand(batch,device=device)
    if strategy=="logit_normal":
        if logit_std<=0:raise ValueError("logit_std must be positive")
        return (torch.randn(batch,device=device)*logit_std+logit_mean).sigmoid()
    raise ValueError("time sampling must be uniform or logit_normal")
