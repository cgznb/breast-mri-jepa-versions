"""Measure complete production-model updates on actual ROI32 patient batches."""
from pathlib import Path
import argparse
import json
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from trjepa.config import load_config
from trjepa.data import PairDataset,RepresentationDataset,collate,write_json
from trjepa.semantic import MedicalEncoder3D,MaskPredictor,MaskedJEPA
from trjepa.train import train_contract,make_world,seed_all,EMA,autocast,_optimizer


def profile(cfg,root,codec,stage,batch,updates=4,device="cuda"):
    if updates<2:raise ValueError("Profiling requires warmup plus at least one measured update")
    device=torch.device(device)
    seed_all(cfg.get("seed",2026));torch.set_num_threads(cfg["train"].get("cpu_threads",4))
    contract=train_contract(cfg,root,codec)
    result={"stage":stage,"batch":batch,"updates":updates,"precision":cfg["train"]["precision"],
            "production_architecture":True,"real_patient_data":True,"device":str(device),
            "E_every_profile_update_includes_decode_and_semantic_backward":stage=="E"}
    try:
        if stage=="B":
            model=MaskedJEPA(MedicalEncoder3D(**cfg["semantic"]),
                             MaskPredictor(dim=cfg["semantic"]["dim"],**cfg["mask_predictor"]),cfg["mask_ratio"]).to(device)
            dataset=RepresentationDataset(root)
            xs,rs=zip(*(dataset[i%len(dataset)] for i in range(batch)))
            inputs=(torch.stack(xs),torch.stack(rs))
        else:
            model=make_world(cfg,contract,codec,device);model.configure_stage(stage)
            dataset=PairDataset(root,"train")
            if stage=="E":
                dataset.records.sort(key=lambda p:not p["spatial_correspondence_verified"])
            inputs=collate([dataset[i%len(dataset)] for i in range(batch)])
        parameters=[p for p in model.parameters() if p.requires_grad]
        optimizer,_=_optimizer(model,cfg,stage,100)
        ema=None if stage=="B" else EMA(model,cfg["train"]["ema_decay"])
        if device.type=="cuda":
            result.update(gpu=torch.cuda.get_device_name(device),device_total_mib=torch.cuda.get_device_properties(device).total_memory/1024**2)
            torch.cuda.reset_peak_memory_stats(device)
        seconds=[];losses=[];norms=[]
        for step in range(updates):
            if device.type=="cuda":torch.cuda.synchronize(device)
            started=time.perf_counter();model.train();optimizer.zero_grad(set_to_none=True)
            a,b=(v.to(device) for v in inputs)
            with autocast(device,cfg["train"]["precision"]):
                loss,terms=(model(a,b) if stage=="B" else model.training_loss(a,b,step=step*cfg["loss"]["semantic_every"]))
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters,cfg["train"]["clip_grad"],error_if_nonfinite=True)
            optimizer.step()
            if ema:ema.update(model)
            else:model.update_teacher(.996)
            if device.type=="cuda":torch.cuda.synchronize(device)
            seconds.append(time.perf_counter()-started);losses.append(float(loss.detach()));norms.append(float(norm))
            del a,b,loss
        if ema:
            with ema.average(model):
                if not all(torch.isfinite(p).all() for p in parameters):
                    raise FloatingPointError("Nonfinite EMA")
        result.update(passed=True,losses=losses,gradient_norms=norms,seconds=seconds,
                      examples_per_second=batch/(sum(seconds[1:])/max(1,len(seconds)-1)),
                      trainable_parameters=sum(p.numel() for p in parameters),last_terms={k:float(v) for k,v in terms.items()})
        if device.type=="cuda":
            result.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2,
                          peak_reserved_mib=torch.cuda.max_memory_reserved(device)/1024**2)
    except torch.OutOfMemoryError:
        result.update(passed=False,reason="cuda_out_of_memory")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True);parser.add_argument("--data",required=True)
    parser.add_argument("--codec",required=True);parser.add_argument("--stage",choices=list("BCDE"),required=True)
    parser.add_argument("--batch",type=int,required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--updates",type=int,default=4);parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    report=profile(load_config(args.config),args.data,args.codec,args.stage,args.batch,args.updates,args.device)
    write_json(args.output,report);print(json.dumps(report,default=float),flush=True)
