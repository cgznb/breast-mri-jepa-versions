#!/usr/bin/env python3
"""Execute minimal counterexamples in the original v1 package and in v2.
Pass an extracted, unmodified original ZIP root. No real patient input required.
"""
import argparse,json,subprocess,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from trjepa.data import write_json,sha256

PROBE=r'''
import json,tempfile
from pathlib import Path
import numpy as np
import torch
from trjepa.prepare import load_visit
from trjepa.train import EMA
from trjepa.blocks import fourier
root=Path(tempfile.mkdtemp())
g={'shape_zyx':[8,8,8],'spacing_xyz_mm':[1.,1.,1.],
   'origin_lps_mm':[0.,0.,0.],'direction_lps':[1.,0.,0.,0.,1.,0.,0.,0.,1.]}
x=np.zeros((3,8,8,8),np.float32);x[:,2:6,2:6,2:6]=20
np.savez(root/'raw.npz',image=x,roi=np.zeros((1,8,8,8),np.float32),support=np.ones_like(x))
visit={'array_file':'raw.npz','crop_geometry':g,'already_normalized':False}
a=load_visit(visit,g,root,{'mean':10.,'std':5.,'background_policy':'preserve_zero'})
image=a['image'] if isinstance(a,dict) else a[0]
support=torch.ones(1,3,8,8,8);support[:,2]=0
try:
    from trjepa.losses import latent_support
    weight=latent_support(support,(2,2,2),24)
except ImportError:
    weight=torch.nn.functional.adaptive_avg_pool3d(support,(2,2,2)).mean(1,keepdim=True).expand(1,24,2,2,2)
value=torch.ones_like(weight,requires_grad=True)
(value.square()*weight).sum().backward()
m=torch.nn.Linear(1,1,bias=False);m.weight.data.zero_()
try:ema=EMA(m,.999,warmup=True)
except TypeError:ema=EMA(m,.999)
m.weight.data.fill_(1.);ema.update(m)
torch.manual_seed(2026)
pos=torch.randn(1,16,24).repeat(4,1,1)
print(json.dumps({'normalized_raw_background':float(image[0,0,0,0]),
 'missing_third_phase_supervision_weight':float(weight[:,16:].mean()),
 'missing_third_phase_gradient_L1':float(value.grad[:,16:].abs().sum()),
 'ema_after_first_constant_update':float(ema.shadow['weight'].item()),
 'position_only_old_variance':float(pos.flatten(0,1).std(0).mean()),
 'position_only_same_coordinate_cross_patient_std':float(pos.std(0,unbiased=False).mean())}))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--v1-root',required=True);p.add_argument('--output',default='validation/reproductions.json')
    a=p.parse_args();v1=Path(a.v1_root).resolve();v2=Path(__file__).resolve().parents[1]
    result={'scope':'synthetic counterexamples in actual v1 and v2 implementations; not a clinical experiment'}
    for name,root in [('v1',v1),('v2',v2)]:
        command='import sys;sys.path.insert(0,'+repr(str(root))+')\n'+PROBE
        run=subprocess.run([sys.executable,'-c',command],capture_output=True,text=True,check=True)
        result[name]=json.loads(run.stdout.strip().splitlines()[-1])
    result['v1_source_sha256']={f:sha256(v1/f) for f in ['trjepa/prepare.py','trjepa/world.py','trjepa/train.py']}
    write_json(a.output,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
