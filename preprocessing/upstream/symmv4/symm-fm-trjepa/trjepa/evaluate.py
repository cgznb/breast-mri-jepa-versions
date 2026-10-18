"""Source-audited sampling, patient-macro metrics and representation export."""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import torch
import torch.nn.functional as F
from .data import read_json,write_json,source_only,CacheReader,atomic_npz,sha256,file_identity,validate_inventory
from .train import make_world


def load_world(checkpoint,codec_path,device="cpu",trusted_legacy=False):
    saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if saved.get("stage") not in ("C","D","E"):
        raise ValueError("Sampling needs a longitudinal world-model checkpoint")
    if (saved["contract"].get("codec_identity")!=file_identity(codec_path) if "codec_identity" in saved["contract"]
            else saved["contract"]["codec_sha256"]!=sha256(codec_path)):
        raise ValueError("Wrong VQ checkpoint: weights must match training and caches")
    model=make_world(saved["config"],saved["contract"],codec_path,torch.device(device),trusted_legacy)
    model.configure_stage(saved["stage"])
    model.load_state_dict(saved.get("inference_state",saved["model_state"]),strict=True)
    return model.eval(),saved


def image_metrics(pred,target,support,*,data_range=6.):
    """MAE/RMSE on original normalized intensities; PSNR/SSIM on fixed clipped window.

    A declared range [-range/2,+range/2] is used for every patient and method.
    SSIM is local 3D uniform-window SSIM, not 2D SSIM or LPIPS.
    """
    if data_range<=0:raise ValueError("data_range must be positive")
    pred,target,support=(v.float() for v in (pred,target,support))
    mask=support>0.5
    if not mask.any():return None
    error=pred-target
    mae=error[mask].abs().mean();rmse=error[mask].square().mean().sqrt()
    a=pred.clamp(-data_range/2,data_range/2);b=target.clamp(-data_range/2,data_range/2)
    mse=(a-b).square()[mask].mean()
    psnr=float(10*torch.log10(torch.tensor(data_range**2)/mse.clamp_min(1e-12)))
    k=min(7,min(a.shape[-3:]));k=k if k%2 else k-1
    a=a[None,None];b=b[None,None]
    pool=lambda x:F.avg_pool3d(x,k,1)
    ma,mb=pool(a),pool(b)
    va=(pool(a*a)-ma.square()).clamp_min(0);vb=(pool(b*b)-mb.square()).clamp_min(0)
    cov=pool(a*b)-ma*mb
    c1=(.01*data_range)**2;c2=(.03*data_range)**2
    ssim=((2*ma*mb+c1)*(2*cov+c2))/((ma.square()+mb.square()+c1)*(va+vb+c2))
    valid=pool(mask.float()[None,None])>.999
    return {"mae":float(mae),"rmse":float(rmse),"psnr_clipped":psnr,
            "ssim3d_box_clipped":float(ssim[valid].mean()) if valid.any() else None,
            "support_voxels":int(mask.sum())}


def binary_metrics(prob,labels):
    p=np.asarray(prob,float);y=np.asarray(labels,int)
    if not len(y):return {"n":0}
    if not np.isin(y,[0,1]).all():raise ValueError("Binary labels must be zero or one")
    auc=None
    if len(np.unique(y))==2:
        # Pairwise definition handles tied probabilities without rank assumptions.
        pos=p[y==1][:,None];neg=p[y==0][None]
        auc=float(((pos>neg)+.5*(pos==neg)).mean())
    return {"n":len(y),"brier":float(np.mean((p-y)**2)),"auroc":auc,
            "accuracy_threshold_0_5":float(np.mean((p>=.5)==y))}


def mask_metrics(pred,true,spacing_xyz):
    """Externally supplied segmentations, never fabricated by thresholding MRI."""
    a=np.asarray(pred,dtype=bool);b=np.asarray(true,dtype=bool)
    if a.shape!=b.shape:raise ValueError("Segmentation grids differ")
    voxel=float(np.prod(spacing_xyz));va=float(a.sum())*voxel;vb=float(b.sum())*voxel
    denom=a.sum()+b.sum()
    return {"dice":float(2*(a&b).sum()/denom) if denom else 1.,
            "pred_volume_mm3":va,"target_volume_mm3":vb,"volume_absolute_error_mm3":abs(va-vb),
            "volume_relative_error":abs(va-vb)/vb if vb>0 else None,
            "target_empty":bool(vb==0)}


def _case(p):
    return {k:p[k] for k in ("source_file","conditions","pair_id") if k in p}


@torch.no_grad()
def evaluate(model,root,output,*,split="val",steps=20,samples=1,seed=2026,solver="heun",data_range=6.,limit=None):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    inv=read_json(root/"inventory.json");validate_inventory(inv)
    device=next(model.parameters()).device
    rows=[];pcr_by_patient=defaultdict(list)
    pairs=[p for p in inv["pairs"] if p["split"]==split]
    if limit is not None:pairs=pairs[:limit]
    if not pairs:raise ValueError("No evaluation pairs")
    for ordinal,p in enumerate(pairs):
        source,audit=source_only(root,_case(p));source=source.to(device)
        result=model.sample(source,steps=steps,samples=samples,seed=seed+ordinal*1000003,solver=solver)
        # True future is opened ONLY AFTER source-only prediction is complete.
        target=CacheReader(root).read(p["target_file"])
        target_image=target["image"];support=target["support"]
        target_codec=model.codec.decode(target["latent"][None].to(device))[0].cpu()
        prediction=result["images"][:,0].cpu();baseline=source.image[0].cpu()
        generation={};copy_source={};ceiling={};roi_metrics={}
        from . import PHASES
        for i,phase in enumerate(PHASES):
            sample_metrics=[image_metrics(x[i],target_image[i],support[i],data_range=data_range) for x in prediction]
            keys=(sample_metrics[0] or {}).keys()
            generation[phase]={k:float(np.mean([m[k] for m in sample_metrics if m is not None and m[k] is not None]))
                               if any(m is not None and m[k] is not None for m in sample_metrics) else None for k in keys}
            copy_source[phase]=image_metrics(baseline[i],target_image[i],support[i],data_range=data_range)
            ceiling[phase]=image_metrics(target_codec[i],target_image[i],support[i],data_range=data_range)
            roi_metrics[phase]=image_metrics(prediction.mean(0)[i],target_image[i],support[i]*target["roi"][0],data_range=data_range)
        filename=f"pair_{ordinal:06d}.npz"
        atomic_npz(out/filename,prediction=prediction.numpy(),mean=prediction.mean(0).numpy(),
                   variance=prediction.var(0,unbiased=False).numpy())
        prob=float(result["pcr_probability"][0]) if bool(result["pcr_available"]) else None
        label=p.get("labels",{}).get("pcr")
        if label is not None and prob is not None:pcr_by_patient[p["patient_id"]].append((prob,label))
        rows.append({"pair_id":p["pair_id"],"patient_id":p["patient_id"],"transition":p.get("transition"),
            "generation_expected_sample_metric":generation,"copy_source":copy_source,"target_codec":ceiling,
            "roi_ensemble_mean":roi_metrics,"pcr_probability":prob,"source_only_audit":audit,"file":filename,
            "minimum_target_coverage":float(support.mean((1,2,3)).min())})
    def macro(method):
        results={}
        from . import PHASES
        for phase in PHASES:
            results[phase]={}
            for metric in ("mae","rmse","psnr_clipped","ssim3d_box_clipped"):
                patients=defaultdict(list)
                for row in rows:
                    item=row[method][phase]
                    if item is not None and item.get(metric) is not None:patients[row["patient_id"]].append(item[metric])
                results[phase][metric]=float(np.mean([np.mean(v) for v in patients.values()])) if patients else None
        return results
    probs=[];labels=[]
    for patient,v in pcr_by_patient.items():
        if len({x[1] for x in v})!=1:raise ValueError("Patient has inconsistent pCR labels")
        probs.append(np.mean([x[0] for x in v]));labels.append(v[0][1])
    report={"split":split,"pair_count":len(rows),"patient_count":len({r['patient_id'] for r in rows}),
            "solver":solver,"steps":steps,"samples":samples,"data_range":data_range,
            "image_metric_protocol":"raw MAE/RMSE; fixed symmetric clipping for 3D box-SSIM and PSNR",
            "summary":{m:macro(m) for m in ("generation_expected_sample_metric","copy_source","target_codec")},
            "pcr_patient_mean_across_available_pairs":binary_metrics(probs,labels),"records":rows,
            "notes":["Clinical reporting must additionally stratify by source landmark; mixing all visits can be optimistic.",
                     "Sample variance is not calibrated clinical uncertainty; semantic pCR head is deterministic.",
                     "Validation used for checkpoint selection is not an independent test set.",
                     "No segmentation volume or HD95 is invented without an external validated segmenter."]}
    write_json(out/"report.json",report)
    return report


@torch.no_grad()
def export_features(model,root,output):
    inv=read_json(Path(root)/"inventory.json");validate_inventory(inv)
    device=next(model.parameters()).device
    data=defaultdict(list)
    for p in inv["pairs"]:
        source,audit=source_only(root,_case(p));source=source.to(device)
        pred=model.predict(source)
        for key,value in {"source":pred.source_state.global_token[:,0],"response":pred.response.mean(1),
                          "future":pred.future_state.global_token[:,0],"vq":source.latent.mean((2,3,4))}.items():
            data[key].append(value[0].cpu().numpy())
        data["label"].append(-1 if p.get("labels",{}).get("pcr") is None else p["labels"]["pcr"])
        data["patient_id"].append(p["patient_id"]);data["split"].append(p["split"]);data["pair_id"].append(p["pair_id"])
        data["source_day"].append(p.get("source_day",float("nan")))
    atomic_npz(output,**{k:np.asarray(v) for k,v in data.items()})
    return {"file":str(output),"pairs":len(data["label"]),"future_image_reads":0}


def linear_probe(features_path,output,*,features="future",epochs=300,seed=2026):
    """Patient-level pCR probe. One specified source landmark per patient REQUIRED."""
    with np.load(features_path,allow_pickle=False) as f:data={k:np.array(f[k]) for k in f.files}
    keep=data["label"]>=0
    patients=data["patient_id"][keep]
    if len(patients)!=len(set(patients)):
        raise ValueError("Select one fixed source landmark/transition per patient before probing; do not count repeated visits as independent patients")
    names=features.split("+")
    if any(n not in ("source","response","future","vq") for n in names):raise ValueError("Unknown feature set")
    x=np.concatenate([data[n] for n in names],1)[keep];y=data["label"][keep];splits=data["split"][keep]
    train=splits=="train";val=splits=="val";test=splits=="test"
    if train.sum()<2 or not val.any() or len(np.unique(y[train]))!=2:
        raise ValueError("Probe needs both training classes and a validation set")
    mean=x[train].mean(0);std=x[train].std(0).clip(1e-6)
    xt=torch.tensor((x-mean)/std,dtype=torch.float32);yt=torch.tensor(y,dtype=torch.float32)
    results=[];best=None
    for decay in (0.,.001,.01,.1):
        torch.manual_seed(seed)
        head=torch.nn.Linear(xt.shape[1],1)
        opt=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=decay)
        for _ in range(epochs):
            opt.zero_grad();loss=F.binary_cross_entropy_with_logits(head(xt[train])[:,0],yt[train]);loss.backward();opt.step()
        with torch.no_grad():p=head(xt)[:,0].sigmoid().numpy()
        vm=binary_metrics(p[val],y[val]);row={"weight_decay":decay,"validation":vm}
        results.append(row)
        if best is None or vm["brier"]<best["validation"]["brier"]:
            best={**row,"test":binary_metrics(p[test],y[test])}
    report={"features":features,"selected":best,"candidates":results,"selection":"validation Brier; no test tuning",
            "n_train":int(train.sum()),"n_val":int(val.sum()),"n_test":int(test.sum())}
    write_json(output,report);return report
