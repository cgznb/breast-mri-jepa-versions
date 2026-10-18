"""Render scientific MRI comparisons, slice atlases and an offline viewer."""
from pathlib import Path
import argparse
import csv
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image,ImageDraw,ImageFont

FONT=Path('/usr/share/fonts/truetype/dejavu')
KINDS=['source','target','best_sample','best_mean','last_sample','target_codec']
LABELS=['Source MRI','Real future','Best EMA / sample 1','Best EMA / mean of 4','Last EMA / sample 1','Target VQ reconstruction']


def read(path):return json.loads(Path(path).read_text())
def load(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def font(size,bold=False):return ImageFont.truetype(str(FONT/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')),size)
def text(draw,xy,value,size=20,fill='#26383d',bold=False,anchor=None):
    draw.text(xy,str(value),font=font(size,bold),fill=fill,anchor=anchor)


def pixels(volume,window,kind='gray',support=None):
    low,high=window
    normalized=np.clip((volume-low)/(high-low),0,1)
    if kind=='gray':result=np.repeat(np.rint(normalized[...,None]*255).astype(np.uint8),3,axis=-1)
    else:result=colormaps[kind](normalized,bytes=True)[...,:3].copy()
    if support is not None:result[support<.5]=[86,92,96]
    return result


def atlas(values):
    result=Image.new('RGBA' if values.shape[-1]==4 else 'RGB',(1024,1536))
    for phase in range(3):
        for z in range(32):result.paste(Image.fromarray(values[phase,z]),((z%8)*128,(phase*4+z//8)*128))
    return result


def case_figure(case,views):
    size=228;gap=10;left=118;top=137;width=left+6*(size+gap)+14;height=top+3*(size+gap)+68
    canvas=Image.new('RGB',(width,height),'white');d=ImageDraw.Draw(canvas)
    text(d,(20,16),f"TR-JEPA optimized v2 | {case['case']} | {case['transition']}",27,bold=True)
    qc='verified correspondence' if case['spatial_verified'] else 'voxel correspondence UNVERIFIED'
    text(d,(20,56),f"{case['delta_days']} days | source-ROI slice z={case['center']} | {qc}",19)
    text(d,(20,86),'Heun 32 | common noise | best step 144 / last step 576 | development cohort',17,fill='#667276')
    for col,label in enumerate(LABELS):text(d,(left+col*(size+gap)+size/2,top-23),label,16,bold=True,anchor='mm')
    for phase,label in enumerate(('Pre','Early','Late')):
        y=top+phase*(size+gap);text(d,(16,y+size/2),label,20,bold=True,anchor='lm')
        for col,kind in enumerate(KINDS):
            image=Image.fromarray(views[kind][phase,case['center']]).resize((size,size),Image.Resampling.NEAREST)
            canvas.paste(image,(left+col*(size+gap),y))
    text(d,(20,height-53),f"Shared source window [{case['window'][0]:.3f}, {case['window'][1]:.3f}] | no per-image contrast adjustment",17)
    score=case['metrics'];text(d,(20,height-28),f"Volume MAE: copy {score['copy_source']['supported_mae']:.4f} | best sample {score['best_sample']['supported_mae']:.4f} | best mean {score['best_mean']['supported_mae']:.4f} | last {score['last_sample']['supported_mae']:.4f}",17)
    return canvas


def charts(audit,path):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(12.5,7.4),layout='constrained')
    for ax,stage,color in ((axes[0,0],'B','#148077'),(axes[0,1],'C','#985743')):
        r=audit['stages'][stage];v=r['validations']
        ax.plot([x['step'] for x in v],[x['selection_score'] for x in v],marker='o',ms=4,color=color)
        ax.set(title=f"Stage {stage}: validation objective",xlabel='Actual optimizer updates',ylabel='Loss')
        ax.grid(alpha=.18)
    ax=axes[1,0];r=audit['stages']['D'];v=r['validations']
    ax.plot([x['step'] for x in v],[x['validation_generation'] for x in v],marker='o',ms=4,label='Generated MRI',color='#246fa4')
    ax.axhline(v[0]['validation_copy_source'],color='#ad6843',linestyle='--',label='Copy source')
    ax.axvline(144,color='#168276',linestyle=':',label='Selected update 144')
    ax.set(title='Stage D: generation selection',xlabel='Actual optimizer updates',ylabel='Supported MAE + 0.25 ROI MAE')
    ax.legend(fontsize=8);ax.grid(alpha=.18)
    ax=axes[1,1];labels=list('BCD');x=np.arange(3)
    ref=[audit['stages'][s]['reference_updates'] for s in labels];actual=[audit['stages'][s]['actual_updates'] for s in labels]
    ax.bar(x-.18,ref,.35,label='ZIP reference (batch 8)',color='#bcc8cc')
    ax.bar(x+.18,actual,.35,label='Actual enlarged-batch run',color='#168276')
    ax.set_xticks(x,labels);ax.set_yscale('log');ax.set(title='Optimizer update counts differ',ylabel='Updates (log scale)')
    for loc,n in zip(x-.18,ref):ax.text(loc,n*1.12,f'{n:,}',ha='center',fontsize=8)
    for loc,n in zip(x+.18,actual):ax.text(loc,n*1.12,f'{n:,}',ha='center',fontsize=8)
    ax.set_ylim(70,100000);ax.legend(fontsize=8)
    fig.savefig(path,dpi=150);plt.close(fig)


def build(root):
    root=Path(root);report=read(root/'report.json');audit=read(root/'training_audit.json')
    gallery=root/'gallery';(gallery/'atlases').mkdir(parents=True,exist_ok=True);(gallery/'figures').mkdir(exist_ok=True)
    pages=[];overview=Image.new('RGB',(1260,1590),'white');od=ImageDraw.Draw(overview)
    text(od,(20,17),'TR-JEPA v2 | six prespecified examples',27,bold=True)
    text(od,(20,54),'Early contrast | source-ROI slice | same window / noise | development patients',18)
    for col,label in enumerate(('Source MRI','Real future','Best EMA 144','Last EMA 576')):text(od,(210+col*253+115,104),label,19,bold=True,anchor='mm')
    layout={}
    for case in report['cases']:
        number=case['case'];ref=load(root/'volumes'/f'{number}_reference.npz')
        best=load(root/'volumes'/f'{number}_best.npz')['images'];last=load(root/'volumes'/f'{number}_last.npz')['images'][0]
        arrays={'source':ref['source'],'target':ref['target'],'best_sample':best[0],'best_mean':best.mean(0),
                'last_sample':last,'target_codec':ref['target_codec']}
        views={kind:pixels(arr,case['window']) for kind,arr in arrays.items()}
        views['error']=pixels(np.abs(arrays['best_mean']-ref['target']),(0,2),'magma',ref['support'])
        views['sd']=pixels(best.std(0),(0,1),'viridis')
        for kind,values in views.items():atlas(values).save(gallery/'atlases'/f'{number}_{kind}.png')
        mask=ref['source_roi'][0]>.5
        interior=mask.copy();interior[:,1:-1,1:-1]&=mask[:,:-2,1:-1]&mask[:,2:,1:-1]&mask[:,1:-1,:-2]&mask[:,1:-1,2:]
        boundary=mask&~interior;rgba=np.zeros((3,32,128,128,4),np.uint8)
        rgba[:,:,:,:,:3]=[245,193,60];rgba[:,:,:,:,3]=boundary[None]*255
        atlas(rgba).save(gallery/'atlases'/f'{number}_roi.png')
        page=case_figure(case,views);page.save(gallery/'figures'/f'{number}.png');pages.append(page)
        index=int(number[1:])-1
        if index%5==0:
            row=index//5;y=129+row*237;z=case['center']
            text(od,(18,y+69),number,23,bold=True);text(od,(18,y+104),case['transition'],20)
            text(od,(18,y+137),f'z = {z}',17,fill='#637075')
            for col,kind in enumerate(('source','target','best_sample','last_sample')):
                overview.paste(Image.fromarray(views[kind][1,z]).resize((224,224),Image.Resampling.NEAREST),(210+col*253,y))
        layout[number]={'center':case['center'],'shape':[32,128,128],'window':case['window'],'atlas_columns':8}
    text(od,(20,1565),'Examples selected before predictions; voxel correspondence may be unverified. Full three-phase views in the gallery.',16)
    overview.save(gallery/'overview.png')
    with PdfPages(gallery/'comparison_30cases.pdf') as pdf:
        for page in pages:
            fig=plt.figure(figsize=(page.width/130,page.height/130),dpi=130)
            ax=fig.add_axes([0,0,1,1]);ax.imshow(np.asarray(page),interpolation='none');ax.axis('off')
            pdf.savefig(fig,dpi=130);plt.close(fig)
    charts(audit,gallery/'training_curves.png')
    with (gallery/'metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['case','transition','spatial_verified','method','supported_mae','roi_mae','selection_score']);writer.writeheader()
        for c in report['cases']:
            for method,metrics in c['metrics'].items():writer.writerow({**{k:c[k] for k in ('case','transition','spatial_verified')},'method':method,**{k:metrics[k] for k in ('supported_mae','roi_mae','selection_score')}})
    template=Path(__file__).with_name('roi32_gallery.html').read_text()
    (gallery/'index.html').write_text(template.replace('__REPORT__',json.dumps(report,ensure_ascii=False)),encoding='utf-8')
    shutil.copyfile('/root/autodl-tmp/.codex/roi32_viewer_browser_20260917/node_modules/lucide/dist/umd/lucide.min.js',gallery/'lucide.min.js')
    (gallery/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (gallery/'render_layout.json').write_text(json.dumps(layout,indent=2)+'\n')
    print(json.dumps({'gallery':str(gallery),'cases':30,'atlases':270,'pdf_pages':len(pages)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();build(a.root)
