"""End-to-end source-only semantic predictor and stochastic three-phase renderer."""
from __future__ import annotations
from dataclasses import dataclass
import copy
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .codec import ThreePhaseCodec
from .semantic import MedicalEncoder3D, TreatmentResponse, DynamicsPredictor, State, semantic_loss, state_from_tokens
from .conditions import ConditionEncoder
from .flow import ConditionBridge,FlowDiT3D,make_path,estimate_endpoint,integrate,build_flow
from .clinical import ClinicalHeads,change_consistency
from .losses import balanced_loss,latent_support,sample_flow_time
from .data import SourceBatch,TargetBatch


@dataclass
class Prediction:
    source_state: State
    response: Tensor
    future_state: State
    structured: Tensor
    clinical: dict[str,Tensor]


class WorldModel(nn.Module):
    def __init__(self,cfg,schema,statistics,codec:ThreePhaseCodec):
        super().__init__()
        self.cfg=copy.deepcopy(cfg)
        self.codec=codec
        if len(statistics["mean"])!=codec.channels or statistics["fit_split"]!="train":
            raise ValueError("Latent normalization does not match the codec/train-only contract")
        if statistics["latent_mode"]!=codec.latent_mode:
            raise ValueError("Cannot mix prequant and quantized latent statistics")
        self.register_buffer("latent_mean",torch.tensor(statistics["mean"]).view(1,-1,1,1,1))
        self.register_buffer("latent_std",torch.tensor(statistics["std"]).view(1,-1,1,1,1))
        s=cfg["semantic"]; d=s["dim"]; heads=s["heads"]
        self.encoder=MedicalEncoder3D(**s)
        self.target_encoder=copy.deepcopy(self.encoder).eval().requires_grad_(False)
        self.condition_encoder=ConditionEncoder(schema,d,heads,**cfg.get("condition_options",{}))
        self.response_encoder=TreatmentResponse(dim=d,heads=heads,**cfg["response"])
        self.dynamics=DynamicsPredictor(dim=d,heads=heads,**cfg["dynamics"])
        f=cfg["flow"]
        self.bridge=ConditionBridge(semantic_dim=d,flow_dim=f["dim"],bridge_channels=f["bridge_channels"],**cfg["bridge"])
        self.flow=build_flow(f,codec.channels)
        self.heads=ClinicalHeads(dim=d,**cfg["clinical"])
        self.outcome_available={"pcr":False,"response_class":False,"survival":False}
        self.stage="C"
        self.configure_stage("C")
    def initialize_encoder(self,state):
        self.encoder.load_state_dict(state,strict=True)
        self.target_encoder.load_state_dict(state,strict=True)
    def normalize(self,z):
        return (z-self.latent_mean)/self.latent_std
    def denormalize(self,z):
        return z*self.latent_std+self.latent_mean
    def configure_stage(self,stage):
        if stage not in ("C","D","E"):
            raise ValueError("WorldModel is used only for stages C, D and E")
        self.stage=stage
        self.requires_grad_(False)
        modules=[self.condition_encoder,self.response_encoder,self.dynamics,self.heads] if stage=="C" else [self.bridge,self.flow]
        if self.cfg.get("renderer_only",False) and stage=="D":modules += [self.condition_encoder]
        if stage=="E":
            modules += [self.condition_encoder,self.response_encoder,self.dynamics,self.heads]
        for module in modules:
            module.requires_grad_(True)
        self.train(True)
    def train(self,mode=True):
        super().train(mode)
        # The main implementation deliberately freezes both semantic coordinates.
        # No unvalidated online-encoder unfreezing with drifting fixed targets.
        self.codec.eval();self.encoder.eval();self.target_encoder.eval()
        if self.stage=="D":
            if not self.cfg.get("renderer_only",False):self.condition_encoder.eval()
            self.response_encoder.eval();self.dynamics.eval();self.heads.eval()
        if self.stage=="C":
            self.bridge.eval();self.flow.eval()
        return self
    def predict(self,source:SourceBatch)->Prediction:
        """Deployment path: signature cannot receive TargetBatch, masks1 or labels."""
        if not isinstance(source,SourceBatch):
            raise TypeError("predict requires SourceBatch only")
        structured=self.condition_encoder(source.conditions)
        if self.cfg.get("renderer_only",False):
            # Source identity comes from the full VQ anchor. No untrained semantic
            # encoder, response predictor or clinical head contributes to this arm.
            b=source.image.shape[0];d=self.cfg["semantic"]["dim"]
            s0=state_from_tokens(source.latent.new_zeros(b,1,d),(1,1,1))
            response=source.latent.new_zeros(b,1,d)
            clinical=self.heads(s0,response,s0,structured)
            return Prediction(s0,response,s0,structured,clinical)
        s0=self.encoder(source.image,source.roi)
        response=self.response_encoder(s0,structured)
        future=self.dynamics(s0,response)
        clinical=self.heads(s0,response,future,structured)
        return Prediction(s0,response,future,structured,clinical)
    def condition(self,source,prediction,drop=None):
        return self.bridge(self.normalize(source.latent),prediction.source_state,prediction.response,
                           prediction.future_state,prediction.structured,drop=drop)
    def training_loss(self,source:SourceBatch,target:TargetBatch,*,step=0,loss_cfg=None):
        """CFM/SymmFlow regression plus explicitly optional auxiliary terms.

        Shared latent standardization is applied before forming residuals.
        Endpoint loss is teacher-forced and is never reported as generation.
        """
        weights=loss_cfg if loss_cfg is not None else self.cfg["loss"]
        pred=self.predict(source)
        zero=pred.response.sum()*0
        terms={k:zero for k in ("jepa","fm","endpoint","semantic","change","pcr","response","survival")}
        if self.stage in ("C","E"):
            with torch.no_grad():
                future_target=self.target_encoder(target.image,target.roi)
            terms["jepa"]=semantic_loss(pred.future_state,future_target,roi=target.roi,
                spatial_valid=target.spatial_valid,support=target.support,roi_balance=weights.get("jepa_roi_balance",0.))
            terms.update(self.heads.loss(pred.clinical,target.labels,pcr_positive_weight=weights.get("pcr_positive_weight",1.)))
        if self.stage in ("D","E"):
            z0,z1=self.normalize(source.latent),self.normalize(target.latent)
            tau=sample_flow_time(z0.shape[0],z0.device,strategy=weights.get("time_sampling","uniform"),
                                logit_mean=weights.get("logit_mean",0.),logit_std=weights.get("logit_std",1.))
            nx,ny=torch.randn_like(z0),torch.randn_like(z0)
            state,velocity_target=make_path(z0,z1,tau,nx,mode=self.flow.mode,noise_y=ny)
            # Validation must use the fully conditioned branch, not CFG training dropout.
            prob=weights.get("condition_dropout",.1) if self.training else 0.
            drop=torch.rand(z0.shape[0],device=z0.device)<prob
            condition=self.condition(source,pred,drop)
            v=self.flow(state,tau,condition)
            support=latent_support(target.support,z0.shape[-3:],z0.shape[1],
                                   threshold=weights.get("support_threshold",.999))
            if self.flow.mode.startswith("residual") and source.support is not None:
                support=support*latent_support(source.support,z0.shape[-3:],z0.shape[1],
                                             threshold=weights.get("support_threshold",.999))
            terms["fm"]=balanced_loss(v[:,:z0.shape[1]],velocity_target[:,:z0.shape[1]],support,
                                     roi=target.roi,roi_weight=weights.get("fm_roi_balance",.25))
            if "symm" in self.flow.mode:
                sw=torch.ones_like(support) if source.support is None else latent_support(source.support,z0.shape[-3:],z0.shape[1])
                terms["fm"]=terms["fm"]+weights.get("reverse_weight",1.)*balanced_loss(
                    v[:,z0.shape[1]:],velocity_target[:,z0.shape[1]:],sw,roi=source.roi,
                    roi_weight=weights.get("fm_roi_balance",.25))
            if weights.get("endpoint",0)>0:
                z_est=estimate_endpoint(state,tau,v,z0,self.flow.mode)
                terms["endpoint"]=balanced_loss(z_est,z1,support,roi=target.roi,
                    roi_weight=weights.get("fm_roi_balance",.25),kind="l1")
            start=int(weights.get("semantic_start",0))
            every=max(1,int(weights.get("semantic_every",8)))
            active=self.stage=="E" and step>=start and (step-start)%every==0
            if active and (weights.get("semantic",0)>0 or weights.get("change",0)>0):
                full_condition=self.condition(source,pred)
                mode=weights.get("consistency_mode","endpoint")
                if mode=="endpoint":
                    z_est=estimate_endpoint(state,tau,self.flow(state,tau,full_condition),z0,self.flow.mode)
                elif mode=="ode":
                    z_est=integrate(self.flow,full_condition,torch.randn_like(z0),
                                    steps=int(weights.get("consistency_steps",4)),solver="euler")
                else:raise ValueError("consistency_mode must be endpoint or ode")
                generated_image=self.codec.decode(self.denormalize(z_est))
                generated_state=self.encoder(generated_image,source.roi)
                ramp=min(1.,(step-start+1)/max(1,int(weights.get("semantic_ramp",1))))
                # This is a sparse regularizer. It is NOT multiplied by its period;
                # the configured weight is per active update, as recorded in logs.
                terms["semantic"]=ramp*semantic_loss(generated_state,pred.future_state.detached(),
                    spatial_valid=target.spatial_valid,support=target.support)
                terms["change"]=ramp*change_consistency(generated_state,pred.future_state,pred.source_state,target.spatial_valid)
        total=sum(terms[k]*float(weights.get(k,0.)) for k in terms)
        if not torch.isfinite(total):raise FloatingPointError("Nonfinite objective; checkpoint not updated")
        return total,{k:float(v.detach()) for k,v in terms.items()}
    @torch.no_grad()
    def sample(self,source:SourceBatch,*,samples=1,steps=20,solver="heun",seed=0,guidance=1.):
        if samples<1:
            raise ValueError("At least one sample is required")
        previous=self.training
        self.eval()
        try:
            p=self.predict(source)
            c=self.condition(source,p)
            null=self.condition(source,p,drop=torch.ones(source.image.shape[0],device=source.image.device,dtype=torch.bool)) if guidance!=1 else None
            generator=torch.Generator(device=source.image.device).manual_seed(seed)
            images=[];latents=[]
            for _ in range(samples):
                noise=torch.randn(c.anchor.shape,device=c.anchor.device,dtype=c.anchor.dtype,generator=generator)
                z=integrate(self.flow,c,noise,steps=steps,solver=solver,guidance=guidance,null_condition=null)
                latents.append(self.denormalize(z))
                images.append(self.codec.decode(latents[-1]))
            return {"images":torch.stack(images),"latents":torch.stack(latents),
                    "future_semantic":p.future_state.spatial,"response_tokens":p.response,
                    "pcr_probability":p.clinical["pcr_logit"].sigmoid() if self.outcome_available["pcr"] else torch.full_like(p.clinical["pcr_logit"],float("nan")),
                    "pcr_available":torch.tensor(self.outcome_available["pcr"],device=source.image.device),
                    "response_probability":p.clinical["response_logits"].softmax(-1) if self.outcome_available["response_class"] else torch.full_like(p.clinical["response_logits"],float("nan")),
                    "survival_rates_per_day":p.clinical["survival_rates_per_day"] if self.outcome_available["survival"] else torch.full_like(p.clinical["survival_rates_per_day"],float("nan"))}
        finally:
            self.train(previous)
    @torch.no_grad()
    def rollout(self,source:SourceBatch,condition_sequence,*,steps=20,seed=0):
        """Free-running IMAGE-state rollout on one fixed source grid.

        Each later semantic state is re-encoded from the previous generated image;
        this is not unobserved real-visit teacher forcing. Source ROI remains an
        explicitly approximate attention prior, not a predicted tumor mask.
        """
        results=[]
        current=source
        for i,conditions in enumerate(condition_sequence):
            current=SourceBatch(current.image,current.latent,current.roi,conditions,current.ids,current.support)
            out=self.sample(current,steps=steps,seed=seed+i)
            results.append(out)
            current=SourceBatch(out["images"][0],out["latents"][0],source.roi,conditions,source.ids,source.support)
        return results
