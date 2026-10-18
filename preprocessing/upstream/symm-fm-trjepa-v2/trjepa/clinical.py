"""Masked clinical outcomes, with landmark-relative piecewise exponential survival.

No minibatch Cox risk sets: the parametric likelihood is patient-separable.
Labels live in TargetBatch and are never passed to condition encoders.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .blocks import MLP


class ClinicalHeads(nn.Module):
    def __init__(self,dim=384,response_classes=3,survival_edges=(0,180,365,730,1095)):
        super().__init__()
        edges=list(survival_edges)
        if edges[0]!=0 or any(b<=a for a,b in zip(edges,edges[1:])):
            raise ValueError("Survival bin starts must begin at zero and strictly increase")
        self.register_buffer("survival_edges",torch.tensor(edges,dtype=torch.float32))
        self.fusion=nn.Sequential(nn.LayerNorm(dim*4),nn.Linear(dim*4,dim),nn.GELU(),MLP(dim),nn.LayerNorm(dim))
        self.pcr=nn.Linear(dim,1)
        self.response=nn.Linear(dim,response_classes)
        self.survival=nn.Linear(dim,len(edges))
    def forward(self,s0,response,s1,structured):
        features=torch.cat((s0.global_token[:,0],response.mean(1),s1.global_token[:,0],structured.mean(1)),-1)
        h=self.fusion(features)
        return {"pcr_logit":self.pcr(h)[:,0],"response_logits":self.response(h),
                "survival_rates_per_day":F.softplus(self.survival(h).float())/365.25,
                "features":h}
    def survival_probability(self,rates,times):
        # Last rate covers all follow-up beyond the last edge (not truncated censoring).
        edges=self.survival_edges.to(rates)
        times=torch.as_tensor(times,device=rates.device,dtype=rates.dtype)
        widths=torch.cat((edges[1:]-edges[:-1],edges.new_tensor([torch.inf])))
        exposure=(times[...,None]-edges).clamp_min(0).minimum(widths)
        return torch.exp(-torch.einsum("bk,tk->bt",rates,exposure))
    def loss(self,pred,labels,*,pcr_positive_weight=1.):
        zero=pred["pcr_logit"].sum()*0
        terms={"pcr":zero,"response":zero,"survival":zero}
        idx=[i for i,y in enumerate(labels) if y.get("pcr") is not None]
        if idx:
            y=pred["pcr_logit"].new_tensor([labels[i]["pcr"] for i in idx])
            if not torch.all((y==0)|(y==1)):
                raise ValueError("pCR targets must be 0 or 1")
            terms["pcr"]=F.binary_cross_entropy_with_logits(pred["pcr_logit"][idx].float(),y.float(),
                pos_weight=y.new_tensor(pcr_positive_weight))
        idx=[i for i,y in enumerate(labels) if y.get("response_class") is not None]
        if idx:
            y=torch.tensor([labels[i]["response_class"] for i in idx],device=zero.device,dtype=torch.long)
            terms["response"]=F.cross_entropy(pred["response_logits"][idx].float(),y)
        idx=[i for i,y in enumerate(labels) if y.get("survival") is not None]
        if idx:
            surv=[labels[i]["survival"] for i in idx]
            rates=pred["survival_rates_per_day"][idx]
            t=rates.new_tensor([r["duration_days"] for r in surv])
            e=rates.new_tensor([r["event"] for r in surv])
            if (t<=0).any() or not torch.all((e==0)|(e==1)):
                raise ValueError("Invalid survival observations")
            edges=self.survival_edges.to(rates)
            widths=torch.cat((edges[1:]-edges[:-1],edges.new_tensor([torch.inf])))
            exposure=(t[:,None]-edges).clamp_min(0).minimum(widths)
            bin_index=torch.bucketize(t,edges[1:],right=True)
            event_rate=rates.gather(1,bin_index[:,None])[:,0]
            terms["survival"]=((rates*exposure).sum(1)-e*event_rate.clamp_min(1e-10).log()).mean()
        return terms


def change_consistency(generated,predicted,source,valid):
    """Direction + relative magnitude, not a redundant shifted absolute L1.

    Predicted zero-change tokens are excluded from angular loss. Magnitude remains
    active. We do not assign a meaning such as necrosis to an individual token.
    """
    a=(generated.spatial-source.spatial).float()
    b=(predicted.spatial-source.spatial).detach().float()
    an,bn=a.norm(dim=-1),b.norm(dim=-1)
    active=(bn>1e-4)&valid[:,None]
    angle=1-F.cosine_similarity(a,b,dim=-1,eps=1e-6)
    angular=(angle*active).sum()/active.sum().clamp_min(1)
    magnitude=(an-bn).abs()/(1+bn)
    magnitude=(magnitude*valid[:,None]).sum()/(valid.sum()*a.shape[1]).clamp_min(1)
    return angular+magnitude
