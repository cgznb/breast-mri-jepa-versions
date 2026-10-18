"""Profile real full-model optimizer updates, including the v2 variance branch."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from trjepa.config import load_config,for_stage
from trjepa.data import PairDataset,RepresentationDataset,collate,write_json
from trjepa.semantic import MedicalEncoder3D,MaskPredictor,MaskedJEPA
from trjepa.sampling import training_indices,augment_representation
from trjepa.train import train_contract,make_world,seed_all,EMA,autocast,_optimizer


def profile(cfg,root,codec,stage,batch,updates=5,device="cuda:0"):
    if updates<3:raise ValueError("Profile needs optimizer warmup and repeated measured updates")
    experiment=cfg;cfg=for_stage(cfg,stage);device=torch.device(device)
    cfg["train"]["stage_batches"]={stage:batch}
    seed_all(cfg["seed"]);torch.set_num_threads(cfg["train"].get("cpu_threads",8))
    contract=train_contract(experiment,root,codec)
    report={"stage":stage,"batch":batch,"updates":updates,"precision":cfg["train"]["precision"],
            "production_architecture":True,"real_patient_data":True,"device":str(device)}
    try:
        if stage=="B":
            model=MaskedJEPA(MedicalEncoder3D(**cfg["semantic"]),
                MaskPredictor(dim=cfg["semantic"]["dim"],**cfg["mask_predictor"]),
                cfg["mask_ratio"],**cfg.get("jepa_pretrain",{})).to(device)
            dataset=RepresentationDataset(root)
        else:
            model=make_world(cfg,contract,codec,device);model.configure_stage(stage)
            dataset=PairDataset(root,"train")
        ids=training_indices(dataset,0,0,batch,cfg["seed"],cfg["train"]["sampling"])
        with ThreadPoolExecutor(max_workers=8) as pool:items=list(pool.map(dataset.__getitem__,ids))
        if stage=="B":
            xs,rs=zip(*items);inputs=(torch.stack(xs),torch.stack(rs))
        else:inputs=collate(items)
        del items
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer,_=_optimizer(model,cfg,stage,100)
        ema=None if stage=="B" else EMA(model,cfg["train"]["ema_decay"],cfg["train"].get("ema_warmup",False))
        report.update(gpu=torch.cuda.get_device_name(device),device_total_mib=torch.cuda.get_device_properties(device).total_memory/1024**2)
        torch.cuda.reset_peak_memory_stats(device)
        seconds=[];losses=[];norms=[]
        for step in range(updates):
            torch.cuda.synchronize(device);start=time.perf_counter()
            model.train();optimizer.zero_grad(set_to_none=True)
            a,b=(v.to(device) for v in inputs)
            if stage=="B":a,b=augment_representation(a,b,cfg.get("augmentation",{}))
            with autocast(device,cfg["train"]["precision"]):
                loss,terms=(model(a,b,mode=model.masker.MODES[step%4]) if stage=="B" else model.training_loss(a,b,step=step))
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(params,cfg["train"]["clip_grad"],error_if_nonfinite=True)
            optimizer.step()
            if ema:ema.update(model)
            else:model.update_teacher(.996)
            torch.cuda.synchronize(device)
            seconds.append(time.perf_counter()-start);losses.append(float(loss.detach()));norms.append(float(norm))
            del a,b,loss
        optimizer.zero_grad(set_to_none=True)
        if ema:
            with ema.average(model):
                if not all(torch.isfinite(p).all() for p in params):raise FloatingPointError("Nonfinite EMA")
        report.update(passed=True,losses=losses,gradient_norms=norms,seconds=seconds,
            examples_per_second=batch/(sum(seconds[1:])/len(seconds[1:])),
            trainable_parameters=sum(p.numel() for p in params),last_terms={k:float(v) for k,v in terms.items()},
            peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2,
            peak_reserved_mib=torch.cuda.max_memory_reserved(device)/1024**2)
    except torch.OutOfMemoryError:
        report.update(passed=False,reason="cuda_out_of_memory")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("config","data","codec","stage","output"):p.add_argument("--"+key,required=True)
    p.add_argument("--batch",required=True,type=int);p.add_argument("--updates",type=int,default=5)
    p.add_argument("--device",default="cuda:0")
    a=p.parse_args();result=profile(load_config(a.config),a.data,a.codec,a.stage,a.batch,a.updates,a.device)
    write_json(a.output,result);print(json.dumps(result),flush=True)

