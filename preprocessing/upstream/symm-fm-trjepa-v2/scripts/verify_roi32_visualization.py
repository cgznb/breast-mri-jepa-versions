"""Independent numerical/pixel replay and a frozen-model sampling replay."""
from pathlib import Path
import argparse
import json
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
from PIL import Image
import torch
from trjepa.data import read_json,write_json,verify_identity,source_only
from trjepa.evaluate import load_world


def load(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def numpy_score(image,target,support,roi):
    e=np.abs(image.astype(np.float64)-target.astype(np.float64))
    s=support.astype(np.float64);r=s*roi
    phase=(e*s).reshape(3,-1).sum(1)/s.reshape(3,-1).sum(1)
    denom=r.reshape(3,-1).sum(1);valid=denom>0
    region=(e*r).reshape(3,-1).sum(1)/np.maximum(denom,1)
    return float(phase.mean()+.25*region[valid].mean())


def verify(args):
    torch.set_num_threads(4)
    root=Path(args.root);protocol=read_json(root/'protocol.json');report=read_json(root/'report.json')
    assert len({c['patient_id'] for c in protocol['cases']})==30
    assert all(c['seed']==p['seed'] for c,p in zip(report['cases'],protocol['cases']))
    metric_errors=[];pixel_checks=0;array_checks=0
    methods=('copy_source','best_sample','best_mean','last_sample','target_codec')
    for c in report['cases']:
        name=c['case'];ref=load(root/'volumes'/f'{name}_reference.npz')
        best=load(root/'volumes'/f'{name}_best.npz');last=load(root/'volumes'/f'{name}_last.npz')
        for data in (ref,best,last):
            for arr in data.values():assert np.isfinite(arr).all();array_checks+=1
        assert best['images'].shape==(4,3,32,128,128) and last['images'].shape==(1,3,32,128,128)
        assert c['center']==int(ref['source_roi'][0].sum((-1,-2)).argmax())
        assert np.array_equal(c['window'],ref['window'])
        arrays={'source':ref['source'],'target':ref['target'],'best_sample':best['images'][0],
                'best_mean':best['images'].mean(0),'last_sample':last['images'][0],'target_codec':ref['target_codec']}
        for method in methods:
            arr=arrays['source' if method=='copy_source' else method]
            expected=numpy_score(arr,ref['target'],ref['support'],ref['roi'])
            metric_errors.append(abs(expected-c['metrics'][method]['selection_score']))
        expected=np.mean([numpy_score(x,ref['target'],ref['support'],ref['roi']) for x in best['images']])
        metric_errors.append(abs(expected-c['metrics']['best_expected_sample']['selection_score']))
        for kind,arr in arrays.items():
            im=np.array(Image.open(root/'gallery/atlases'/f'{name}_{kind}.png'))
            assert im.shape==(1536,1024,3)
            lo,hi=c['window']
            for phase in range(3):
                for z in range(32):
                    for y,x in ((0,0),(32,96),(64,64),(85,40),(127,127)):
                        expected=np.uint8(np.rint(np.clip((arr[phase,z,y,x]-lo)/(hi-lo),0,1)*255))
                        actual=im[(phase*4+z//8)*128+y,(z%8)*128+x]
                        assert np.all(actual==expected),(name,kind,phase,z,y,x)
                        pixel_checks+=1
        assert read_json(root/'volumes'/f'{name}_best.json')['target_reads_before_prediction']==0
        assert read_json(root/'volumes'/f'{name}_last.json')['target_reads_before_prediction']==0
    assert max(metric_errors)<1e-6
    for binding in protocol['bindings']:verify_identity(binding)
    pair=protocol['cases'][0];data=REPO/'data/registered_roi32'
    inv=read_json(data/'inventory.json');record=next(p for p in inv['pairs'] if p['pair_id']==pair['pair_id'])
    codec=next(b['path'] for b in protocol['bindings'] if Path(b['path']).name=='codec.pt')
    model,saved=load_world(root.parent/'D/best.pt',codec,'cuda:0');model.eval().requires_grad_(False)
    source,audit=source_only(data,{k:record[k] for k in ('pair_id','source_file','conditions')})
    quantized_indices=[]
    def capture_indices(module,inputs,output):
        quantized_indices.append(output[1]['indices'].detach().cpu().numpy().copy())
    handle=model.codec.codec.quantizer.register_forward_hook(capture_indices)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        generated=model.sample(source.to('cuda:0'),samples=1,steps=32,solver='heun',seed=pair['seed'])
        repeated=[model.codec.decode(generated['latents'][0])[0].float().cpu().numpy() for _ in range(2)]
    handle.remove()
    before=load(root/'volumes'/f'{pair["case"]}_best.npz')
    replay=generated['images'][0,0].float().cpu().numpy()
    image_error=float(np.abs(replay-before['images'][0]).max())
    image_mae=float(np.abs(replay-before['images'][0]).mean())
    latent_error=float(np.abs(generated['latents'][0,0].float().cpu().numpy()-before['latents'][0]).max())
    ref=load(root/'volumes'/f'{pair["case"]}_reference.npz')
    gray_step=float((ref['window'][1]-ref['window'][0])/255)
    decode_max=[float(np.abs(x-before['images'][0]).max()) for x in (replay,*repeated)]
    decode_mae=[float(np.abs(x-before['images'][0]).mean()) for x in (replay,*repeated)]
    repeat_max=[float(np.abs(x-replay).max()) for x in repeated]
    repeat_mae=[float(np.abs(x-replay).mean()) for x in repeated]
    indices_equal=all(np.array_equal(x,quantized_indices[i%3]) for i,x in enumerate(quantized_indices))
    assert len(quantized_indices)==9 and indices_equal
    # cuDNN FP32 convolution can vary between calls; bound error in display units.
    assert latent_error==0,latent_error
    assert max(decode_max+repeat_max)<.25*gray_step,(decode_max,repeat_max,gray_step)
    assert max(decode_mae+repeat_mae)<.01*gray_step,(decode_mae,repeat_mae,gray_step)
    assert audit['target_reads']==0
    from pypdf import PdfReader
    pdf=PdfReader(root/'gallery/comparison_30cases.pdf')
    pdf_pages=len(pdf.pages);assert pdf_pages==30
    for i,page in enumerate(pdf.pages,1):
        operations=page.get_contents().operations
        names=[args[0] for args,op in operations if op==b'Do']
        assert len(names)==1
        embedded=np.array(page.images[names[0]].image.convert('RGB'))
        matrices=[args for args,op in operations if op==b'cm']
        assert len(matrices)==2 and list(matrices[0])==[1,0,0,1,0,0]
        a,b,c,d,e,f=matrices[1]
        assert a>0 and d<0 and b==c==e==0 and abs(f+d)<1e-5
        # Matplotlib reverses image rows and restores orientation in the PDF transform.
        embedded=embedded[::-1]
        figure=np.array(Image.open(root/f'gallery/figures/C{i:02d}.png').convert('RGB'))
        assert np.array_equal(embedded,figure),i
    browser=read_json(root/'browser_verification.json');assert browser['passed']
    pixels=np.array(Image.open(root/'browser/rendered_source.png').convert('RGB'))
    assert len(np.unique(pixels))>32 and float(pixels.std())>10
    for binding in protocol['bindings']:verify_identity(binding)
    result={'passed':True,'cases':30,'unique_patients':30,'finite_array_checks':array_checks,
            'independent_metric_checks':len(metric_errors),'max_metric_error':max(metric_errors),
            'grayscale_atlas_pixel_checks':pixel_checks,'pixel_errors':0,'pdf_pages':pdf_pages,
            'sampling_replay_image_max_error':image_error,'sampling_replay_latent_max_error':latent_error,
            'sampling_replay_image_mae':image_mae,'decode_repeat_max_errors':repeat_max,
            'decode_repeat_mae':repeat_mae,'quantizer_indices_identical_across_decodes':indices_equal,
            'display_gray_level_width':gray_step,'decode_max_error_in_gray_levels':max(decode_max)/gray_step,
            'max_error_tolerance_gray_levels':.25,'mean_error_tolerance_gray_levels':.01,
            'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'pdf_figures_exact_pixel_matches':pdf_pages,'browser_render_nonblank':True,
            'source_only_routes':60,'training_bindings_unchanged':True,'optimizer_updates':0}
    write_json(root/'verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);verify(p.parse_args())
