"""Data/VQ/representation/generation failure localization, without fabricated diagnoses."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import torch
from .data import (read_json,write_json,CacheReader,PairDataset,collate,verify_codec,validate_inventory,
                   validate_inference_inventory,SourceBatch)
from .codec import ThreePhaseCodec,load_codec
from .validation import selection_indices,validate_generation
from .losses import balanced_loss
from .semantic import semantic_loss


def summarize_metrics(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if line.strip():rows.append(json.loads(line))
    if not rows:return {"rows":0}
    numeric={k for r in rows for k,v in r.items() if isinstance(v,(float,int)) and not isinstance(v,bool)}
    return {"rows":len(rows),"first_step":rows[0].get("step"),"last_step":rows[-1].get("step"),
            "metrics":{k:{"first":next(r[k] for r in rows if k in r),
                          "last":next(r[k] for r in reversed(rows) if k in r),
                          "minimum":min(r[k] for r in rows if k in r)} for k in sorted(numeric)}}


@torch.no_grad()
def diagnose(cfg,root,codec_path,output,*,checkpoint=None,device="cpu",limit=4,split="train",steps=(16,32),
             metrics_path=None,trusted_legacy=False):
    if limit<1:raise ValueError("limit must be positive")
    root=Path(root);device=torch.device(device)
    torch.set_num_threads(cfg["train"].get("cpu_threads",4))
    inv=read_json(root/"inventory.json");validate_inventory(inv)
    verify_codec(inv,codec_path)
    ds=PairDataset(root,split)
    selected=selection_indices(ds.records,limit,cfg.get("seed",2026))
    codec=ThreePhaseCodec(load_codec(codec_path,trusted_legacy=trusted_legacy),inv["latent_mode"],cfg.get("decode_mode","requantize")).to(device)
    norms=inv.get("image_normalization",{})
    alerts=[]
    if "background_policy" not in norms:
        alerts.append("Image background policy is undocumented. Inspect raw zeros versus -mean/std; "
                      "do not infer a normalization failure from normalized pixels alone.")
    if not all(ds.records[i].get("spatial_correspondence_verified",False) for i in selected):
        alerts.append("At least one inspected pair lacks spatial correspondence QC. Residuals may represent pose/FOV changes.")
    cases=[]
    for index in selected:
        s,t=collate([ds[index]]);s,t=s.to(device),t.to(device)
        p=ds.records[index];row={"pair_id":p["pair_id"],"patient_id":p["patient_id"],"phases":[]}
        z=codec.encode(s.image)
        row["reencoded_latent_rmse"]=float((z-s.latent).square().mean().sqrt())
        row["reencoded_latent_relative_rmse"]=row["reencoded_latent_rmse"]/max(1e-8,float(s.latent.square().mean().sqrt()))
        copy=s.image
        reconstructed=codec.decode(t.latent)
        for phase in range(3):
            support=t.support[:,phase:phase+1]
            mask=support>0
            arr=t.image[:,phase:phase+1][mask]
            mae=lambda x:float((x[:,phase:phase+1]-t.image[:,phase:phase+1]).abs()[mask].mean()) if mask.any() else None
            roi=mask & (t.roi>0)
            rec_roi=float((reconstructed[:,phase:phase+1]-t.image[:,phase:phase+1]).abs()[roi].mean()) if roi.any() else None
            row["phases"].append({"phase":phase,"coverage":float(support.mean()),
                "target_quantiles":torch.quantile(arr.float(),torch.tensor([0.,.01,.5,.99,1.],device=device)).tolist() if arr.numel() else None,
                "zero_fraction":float((arr==0).float().mean()) if arr.numel() else None,
                "copy_source_mae":mae(copy),"target_codec_mae":mae(reconstructed),"target_codec_roi_mae":rec_roi})
        cases.append(row)
    del codec
    report={"scope":"inspected cases only, not a full clinical validation", "split":split,"cases":cases,"alerts":alerts,
            "latent_mode":inv["latent_mode"],"image_normalization":norms,"spatial_policy":inv["spatial_policy"]}
    if metrics_path:report["training_log"]=summarize_metrics(metrics_path)
    if checkpoint:
        from .evaluate import load_world
        saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
        is_pretrain=saved["stage"]=="B"
        if is_pretrain:
            from .semantic import MedicalEncoder3D
            encoder=MedicalEncoder3D(**saved["config"]["semantic"]).to(device).eval()
            encoder.load_state_dict(saved["encoder_state"],strict=True)
            model=None
        else:model,saved=load_world(checkpoint,codec_path,device,trusted_legacy)
        validate_inference_inventory(saved["contract"],root)
        features=[];semantic_rows=[]
        for index in selected:
            s,t=collate([ds[index]]);s,t=s.to(device),t.to(device)
            if is_pretrain:
                features.append(encoder(s.image,s.roi).spatial.float().cpu())
                continue
            pred=model.predict(s)
            if not model.cfg.get("renderer_only",False):
                truth=model.target_encoder(t.image,t.roi)
                semantic_rows.append({"pair_id":ds.records[index]["pair_id"],
                    "predicted_future_loss":float(semantic_loss(pred.future_state,truth,spatial_valid=t.spatial_valid,support=t.support)),
                    "copy_source_state_loss":float(semantic_loss(pred.source_state,truth,spatial_valid=t.spatial_valid,support=t.support))})
                features.append(pred.source_state.spatial.float().cpu())
        if features and len({ds.records[i]["patient_id"] for i in selected})==len(features) and len(features)>1:
            shapes={tuple(x.shape) for x in features}
            if len(shapes)==1:
                stack=torch.cat(features)
                # Variation across patients at the SAME token coordinates. V1's
                # across-space statistic cannot reject position-only solutions.
                report["same_position_across_patient_std"]=float(stack.std(0,unbiased=False).mean())
                pooled=stack.mean(1);centered=pooled-pooled.mean(0)
                sv=torch.linalg.svdvals(centered)
                prob=sv.square()/sv.square().sum().clamp_min(1e-12)
                report["global_feature_effective_rank"]=float((-(prob*prob.clamp_min(1e-12).log()).sum()).exp()) if sv.square().sum()>0 else 0.
                report["feature_rank_maximum"]=len(features)-1
        report["semantic_comparison"]=semantic_rows
        if saved["stage"] in ("D","E"):
            # Sweep only these inspected cases, with equal case/seed protocol.
            sweeps=[]
            for weights in ("raw","ema"):
                state=saved["model_state"] if weights=="raw" else saved.get("inference_state",saved["model_state"])
                model.load_state_dict(state,strict=True);model.eval()
                for nsteps in steps:
                    result=validate_generation(model,ds,device,max_pairs=limit,steps=nsteps,
                        seed=cfg.get("seed",2026),solver="heun",samples=1)
                    sweeps.append({"weights":weights,**result})
            report["generation_sweep"]=sweeps
        report["checkpoint_stage"]=saved["stage"]
        report["checkpoint_step"]=saved["step"]
    report["interpretation"]=[
        "A bad target VQ reconstruction is a codec/preprocessing bottleneck, not fixed by deeper flow alone.",
        "A low JEPA loss does not establish image-dependent or disease-dependent representation.",
        "Solver/raw-EMA comparisons isolate numerical and averaging effects, not clinical generalization.",
        "Do not tune the test cohort; choose settings on validation and report independent test results once."]
    write_json(output,report)
    return {"report":str(output),"inspected_pairs":len(cases),"real_data_supplied_by_caller":not inv.get("synthetic",False)}
