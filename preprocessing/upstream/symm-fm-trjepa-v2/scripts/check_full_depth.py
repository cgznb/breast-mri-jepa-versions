#!/usr/bin/env python3
"""Full production depth/width on a SMALL synthetic spatial grid, not a GPU benchmark."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import torch
from trjepa.config import load_config
from trjepa.codec import MRILevelVQGAN,VQGANConfig,ThreePhaseCodec
from trjepa.conditions import fit_schema
from trjepa.data import SourceBatch
from trjepa.world import WorldModel

def main():
    torch.set_num_threads(1);torch.manual_seed(2026)
    root=Path(__file__).resolve().parents[1]
    cfg=load_config(root/'configs/optimized.yaml')
    codec=ThreePhaseCodec(MRILevelVQGAN(VQGANConfig(**cfg['codec'])))
    cond={'clinical':{'age':50,'sex':'female'},'treatment':{'regimen':'synthetic','known_at_source':True,'drugs':[]},'source_day':0,'delta_days':90,'horizon_kind':'query'}
    schema=fit_schema([{'conditions':cond,'split':'train'}],**cfg['condition_fields'])
    # Explicit architecture-only synthetic statistics, not fitted medical values.
    stats={'mean':[0.]*24,'std':[1.]*24,'fit_split':'train','latent_mode':'prequant'}
    model=WorldModel(cfg,schema,stats,codec).eval()
    x=torch.randn(1,3,16,32,32);roi=torch.ones(1,1,16,32,32)
    source=SourceBatch(x,codec.encode(x),roi,[cond],['SYNTHETIC_ARCHITECTURE_ONLY'])
    out=model.sample(source,steps=1,samples=1,solver='euler',seed=2026)
    report={'passed':bool(torch.isfinite(out['images']).all()),'scope':'full configured depth and width, small synthetic spatial input; untrained',
        'clinical_validation':False,'gpu_tested':False,'input_shape':list(x.shape),'output_shape':list(out['images'].shape),
        'semantic_layers':cfg['semantic']['depth'],'response_layers':cfg['response']['depth'],'dynamics_layers':cfg['dynamics']['depth'],
        'flow_layers':cfg['flow']['depth'],'width':cfg['semantic']['dim'],
        'parameters_by_module':{n:sum(p.numel() for p in m.parameters()) for n,m in model.named_children()},
        'total_parameters_including_frozen_teacher_codec':sum(p.numel() for p in model.parameters())}
    (root/'validation/full_depth.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
