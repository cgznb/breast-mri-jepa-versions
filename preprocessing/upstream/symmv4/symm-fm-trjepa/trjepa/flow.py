"""Conditional 3D DiT renderer and explicit CFM / original-form symmetric paths.

DiT core adapted from Meta's official models.py (CC BY-NC 4.0): adaLN-Zero,
Transformer depth, Fourier flow time, non-affine final normalization and zero
velocity output. Extensions: anisotropic 3D patch/unpatch, semantic cross-attn,
immutable current-image anchor and semantic-to-generative spatial bridge.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .blocks import DiTBlock, fourier, pos3d, run_blocks, modulate
from .semantic import State


@dataclass
class FlowCondition:
    anchor: Tensor                 # normalized source VQ latent, never noised
    tokens: Tensor                 # only online-predicted semantics
    global_vector: Tensor
    spatial: Tensor                # semantic bridge on latent grid
    def detached(self):
        return FlowCondition(*(x.detach() for x in (self.anchor,self.tokens,self.global_vector,self.spatial)))


class ConditionBridge(nn.Module):
    def __init__(self, semantic_dim=384, flow_dim=384, bridge_channels=24, token_grid=(3,4,4),
                 use_response=True,use_future=True,use_structured=True,use_spatial=True):
        super().__init__()
        self.token_grid=tuple(token_grid)
        self.use_response,self.use_future,self.use_structured,self.use_spatial=use_response,use_future,use_structured,use_spatial
        self.token_proj=nn.Linear(semantic_dim,flow_dim)
        self.global_proj=nn.Sequential(nn.Linear(semantic_dim*3,flow_dim),nn.SiLU(),nn.Linear(flow_dim,flow_dim))
        # Spatial adapter is not the predictive backbone; two residual 3D blocks follow projection.
        self.spatial_proj=nn.Sequential(nn.Conv3d(semantic_dim,bridge_channels,1),nn.SiLU(),
            nn.Conv3d(bridge_channels,bridge_channels,3,padding=1),nn.SiLU(),nn.Conv3d(bridge_channels,bridge_channels,3,padding=1))
        self.null_token=nn.Parameter(torch.randn(1,1,flow_dim)*.02)
        self.null_global=nn.Parameter(torch.zeros(1,flow_dim))
    def forward(self,z0,source:State,response,future:State,structured,*,drop=None,spatial_valid=None):
        b,n,d=future.spatial.shape
        future_map=future.spatial.transpose(1,2).reshape(b,d,*future.grid)
        reduced=F.adaptive_avg_pool3d(future_map,self.token_grid).flatten(2).transpose(1,2)
        # Current global remains; ALL correlated treatment/response/future routes are dropped together for CFG.
        parts=[source.global_token]
        if self.use_response: parts.append(response)
        if self.use_future: parts.append(reduced)
        if self.use_structured: parts.append(structured)
        tokens=self.token_proj(torch.cat(parts,1))
        rg=response.mean(1) if self.use_response else torch.zeros_like(source.global_token[:,0])
        fg=future.global_token[:,0] if self.use_future else torch.zeros_like(source.global_token[:,0])
        global_vector=self.global_proj(torch.cat((source.global_token[:,0],rg,fg),-1))
        bridge=F.interpolate(self.spatial_proj(future_map),size=z0.shape[-3:],mode="trilinear",align_corners=False)
        if not self.use_spatial or not self.use_future:
            bridge=bridge*0
        if spatial_valid is not None:
            bridge=bridge*spatial_valid[:,None,None,None,None].to(bridge.dtype)
        if drop is not None:
            tokens=torch.where(drop[:,None,None],self.null_token.expand_as(tokens),tokens)
            global_vector=torch.where(drop[:,None],self.null_global.expand_as(global_vector),global_vector)
            bridge=bridge*(~drop)[:,None,None,None,None]
        return FlowCondition(z0,tokens,global_vector,bridge)


class FlowDiT3D(nn.Module):
    def __init__(self, *, latent_channels=24, dim=384, depth=12, heads=6, patch=(2,4,4),
                 bridge_channels=24, mode="residual_cfm", checkpointing=True):
        super().__init__()
        if mode not in ("residual_cfm","direct_cfm","residual_symm","direct_symm"):
            raise ValueError("Unknown transport objective")
        self.mode,self.patch,self.dim,self.latent_channels=mode,tuple(patch),dim,latent_channels
        self.state_channels=latent_channels*(2 if "symm" in mode else 1)
        self.checkpointing=checkpointing
        self.patch_embed=nn.Conv3d(self.state_channels+latent_channels+bridge_channels,dim,self.patch,self.patch)
        self.time_embed=nn.Sequential(nn.Linear(dim,dim),nn.SiLU(),nn.Linear(dim,dim))
        self.blocks=nn.ModuleList([DiTBlock(dim,heads) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim,elementwise_affine=False,eps=1e-6)
        self.final_mod=nn.Sequential(nn.SiLU(),nn.Linear(dim,2*dim))
        self.final=nn.Linear(dim,math.prod(self.patch)*self.state_channels)
        nn.init.zeros_(self.final_mod[-1].weight); nn.init.zeros_(self.final_mod[-1].bias)
        nn.init.zeros_(self.final.weight); nn.init.zeros_(self.final.bias)
    def forward(self,x:Tensor,tau:Tensor,condition:FlowCondition):
        if x.ndim!=5 or x.shape[1]!=self.state_channels:
            raise ValueError("Wrong flow state channel/shape contract")
        if any(s%p for s,p in zip(x.shape[-3:],self.patch)):
            raise ValueError("Latent grid must be divisible by the 3D flow patch shape")
        if tau.shape!=(x.shape[0],):
            raise ValueError("Flow time tau must have shape [B]")
        h=self.patch_embed(torch.cat((x,condition.anchor,condition.spatial),1))
        grid=h.shape[-3:]
        h=h.flatten(2).transpose(1,2)+pos3d(grid,self.dim,x.device,h.dtype)
        c=self.time_embed(fourier(tau,self.dim))+condition.global_vector
        h=run_blocks(self.blocks,h,checkpointing=self.checkpointing,args=(c,condition.tokens))
        shift,scale=self.final_mod(c).chunk(2,1)
        h=self.final(modulate(self.norm(h),shift,scale))
        b=x.shape[0]; gz,gy,gx=grid; pz,py,px=self.patch
        h=h.reshape(b,gz,gy,gx,pz,py,px,self.state_channels)
        return h.permute(0,7,1,4,2,5,3,6).reshape(b,self.state_channels,gz*pz,gy*py,gx*px)


def make_path(source,target,tau,noise,*,mode="residual_cfm",noise_y=None):
    """Analytic path; no stochastic numerical ODE is needed for training.

    For symmetric mode, original sigma=0 equations are preserved in coordinates
    [future-or-residual, source]: x_t=(1-t)eps_x+t*y;
    y_t=(1-t)source+t*eps_y. Immutable source is an additional condition.
    This is NOT biological reversal. It is not identical to legacy conditioning.
    """
    if source.shape!=target.shape or noise.shape!=target.shape:
        raise ValueError("Source, target and noise latent shapes must agree")
    if tau.shape!=(source.shape[0],) or (tau<0).any() or (tau>1).any():
        raise ValueError("Flow time outside [0,1]")
    endpoint=target-source if mode.startswith("residual") else target
    t=tau[:,None,None,None,None]
    x=(1-t)*noise+t*endpoint
    vx=endpoint-noise
    if "symm" in mode:
        if noise_y is None or noise_y.shape!=source.shape:
            raise ValueError("Symmetric path requires independent second-branch noise")
        y=(1-t)*source+t*noise_y
        return torch.cat((x,y),1),torch.cat((vx,noise_y-source),1)
    return x,vx


def estimate_endpoint(state,tau,velocity,source,mode):
    """First-branch endpoint estimator: x_t + (1-t)*v. Surrogate, not a free sample."""
    c=source.shape[1]
    estimate=state[:,:c]+(1-tau[:,None,None,None,None])*velocity[:,:c]
    return estimate+source if mode.startswith("residual") else estimate


def integrate(model,condition,noise,*,steps=20,solver="heun",guidance=1.,null_condition=None):
    """Differentiable ODE; caller decides torch.no_grad vs closed-loop training.

    CFG acts on ALL output channels. Source anchor remains immutable throughout.
    """
    if steps<1 or solver not in ("euler","heun"):
        raise ValueError("Invalid solver settings")
    if guidance!=1 and null_condition is None:
        raise ValueError("Guidance requires a jointly dropped condition")
    state=torch.cat((noise,condition.anchor),1) if "symm" in model.mode else noise
    dt=1./steps
    def vf(y,t):
        tau=y.new_full((y.shape[0],),t)
        v=model(y,tau,condition)
        if guidance!=1:
            v0=model(y,tau,null_condition)
            v=v0+guidance*(v-v0)
        return v
    for i in range(steps):
        t=i*dt
        k1=vf(state,t)
        proposal=state+dt*k1
        state=proposal if solver=="euler" else state+dt*.5*(k1+vf(proposal,(i+1)*dt))
    end=state[:,:condition.anchor.shape[1]]
    return end+condition.anchor if model.mode.startswith("residual") else end


class FlowMONAIUNet3D(nn.Module):
    """Optional reuse of the original repo's MONAI 1.5.1 U-Net backbone family.

    Not a silent fallback and not a legacy checkpoint loader: conditioning/input
    channels change. The official multi-resolution ResBlocks and Transformer
    cross-attention are reused directly from MONAI. This optional backend requires
    external MONAI installation and was not executable in the offline test host.
    """
    def __init__(self,*,latent_channels=24,dim=384,mode="residual_cfm",bridge_channels=24,
                 channels=(128,256,384),attention_levels=(False,True,True),num_head_channels=(32,64,64),
                 num_res_blocks=2,transformer_num_layers=2,norm_num_groups=32,checkpointing=True,**unused):
        super().__init__()
        try:
            from monai.networks.nets import DiffusionModelUNet
        except ImportError as e:
            raise ImportError("MONAI backend selected explicitly; install monai==1.5.1. No toy fallback will be used.") from e
        self.mode,self.latent_channels=mode,latent_channels
        self.state_channels=latent_channels*(2 if "symm" in mode else 1)
        self.checkpointing=checkpointing
        self.net=DiffusionModelUNet(spatial_dims=3,in_channels=self.state_channels+latent_channels+bridge_channels,
            out_channels=self.state_channels,channels=tuple(channels),num_res_blocks=num_res_blocks,
            attention_levels=tuple(attention_levels),norm_num_groups=norm_num_groups,
            num_head_channels=tuple(num_head_channels),with_conditioning=True,
            transformer_num_layers=transformer_num_layers,cross_attention_dim=dim)
    def forward(self,x,tau,condition):
        from torch.utils.checkpoint import checkpoint
        inputs=torch.cat((x,condition.anchor,condition.spatial),1)
        tokens=torch.cat((condition.tokens,condition.global_vector[:,None]),1)
        def f(x,t,c):return self.net(x=x,timesteps=t,context=c)
        if self.checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(f,inputs,tau,tokens,use_reentrant=False)
        return f(inputs,tau,tokens)


def build_flow(config,latent_channels):
    config=dict(config)
    backend=config.pop("backbone","dit3d")
    if backend=="dit3d":return FlowDiT3D(latent_channels=latent_channels,**config)
    if backend=="monai_unet3d":return FlowMONAIUNet3D(latent_channels=latent_channels,**config)
    raise ValueError("flow.backbone must be dit3d or monai_unet3d")
