#!/usr/bin/env python3
"""Controlled 8-pair latent memorization with the actual 3D DiT/CFM/ODE code.

Not MRI, not a generalization/clinical benchmark, and not evidence that v2
outperforms v1. Evaluates a new fixed noise draw on the SAME training anchors.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import math
import torch
from trjepa.flow import FlowDiT3D,FlowCondition,make_path,integrate
from trjepa.losses import balanced_loss
from trjepa.data import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='validation/flow_learnability.json')
    p.add_argument('--steps',type=int,default=600);a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(2026)
    model=FlowDiT3D(latent_channels=24,dim=48,depth=4,heads=4,patch=(1,1,1),
                    bridge_channels=6,mode='residual_cfm',checkpointing=False,time_scale=1000.,init_scheme='dit')
    source=torch.randn(8,24,2,2,2)*.4
    signal=torch.linspace(-1,1,8).view(8,1,1,1,1)
    target=.8*source+signal*.25
    condition=FlowCondition(source,torch.randn(8,3,48),torch.randn(8,48),torch.zeros(8,6,2,2,2))
    evaluation_noise=torch.randn_like(source)
    @torch.no_grad()
    def score():
        model.eval()
        generated=integrate(model,condition,evaluation_noise,steps=32,solver='heun')
        return float((generated-target).square().mean())
    before=score();opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=.01,betas=(.9,.95))
    rows=[]
    for step in range(a.steps):
        model.train();opt.zero_grad(set_to_none=True)
        tau=torch.rand(8);noise=torch.randn_like(source)
        state,velocity=make_path(source,target,tau,noise,mode='residual_cfm')
        estimate=model(state,tau,condition)
        loss=balanced_loss(estimate,velocity,torch.ones_like(source))
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        for g in opt.param_groups:g['lr']=8e-4*(.1+.9*(1+math.cos(math.pi*(step+1)/a.steps))/2)
        if (step+1)%100==0 or step+1==a.steps:
            rows.append({'step':step+1,'flow_loss':float(loss.detach()),'fixed_noise_sample_mse':score()})
    after=score()
    result={'task':'same-eight-pair latent memorization; unseen fixed noise only',
            'real_images':False,'clinical_validation':False,'compared_to_v1':False,
            'device':'cpu','torch':str(torch.__version__),'seed':2026,'steps':a.steps,
            'depth':4,'dim':48,'latent_shape':[8,24,2,2,2],'parameters':sum(p.numel() for p in model.parameters()),
            'solver':'heun','solver_steps':32,'before_sample_mse':before,'after_sample_mse':after,
            'copy_source_mse':float((source-target).square().mean()),'rows':rows,
            'relative_mse_reduction':1-after/before,'passed_memorization_check':after<.1*before}
    write_json(a.output,result)
    print(result)
    if not result['passed_memorization_check']:raise SystemExit('Memorization failed: inspect before attempting real training')
if __name__=='__main__':main()
