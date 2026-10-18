"""Five explicit training stages, checkpoint contracts and deterministic replay.

Single-process CPU/single-GPU trainer. Gradient accumulation, AMP, gradient
checkpointing, parameter EMA and exact RNG resume are supported. No untested
DDP implementation is advertised. Large datasets can be read from local NVMe.
"""
from __future__ import annotations
from dataclasses import asdict
from contextlib import contextmanager,nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import copy
import json
import math
import os
import random
import shutil
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .data import (read_json,write_json,file_identity,verify_codec,fit_statistics,PairDataset,RepresentationDataset,
                   collate,ensure_assets,validate_inventory)
from .conditions import fit_schema
from .codec import load_codec,MRILevelVQGAN,VQGANConfig,ThreePhaseCodec,PatchDiscriminator3D
from .semantic import MedicalEncoder3D,MaskPredictor,MaskedJEPA
from .world import WorldModel
from .config import for_stage
from .sampling import training_indices,augment_representation
from .validation import validate_generation,selection_indices


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def rng_state():
    a=np.random.get_state()
    return {"python":random.getstate(),"numpy":[a[0],a[1].tolist(),a[2],a[3],a[4]],
            "torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(s):
    random.setstate(s["python"])
    a=s["numpy"];np.random.set_state((a[0],np.asarray(a[1],dtype=np.uint32),a[2],a[3],a[4]))
    torch.set_rng_state(s["torch"])
    if s["cuda"] and torch.cuda.is_available():torch.cuda.set_rng_state_all(s["cuda"])


def cpu_tree(obj):
    if isinstance(obj,torch.Tensor):return obj.detach().cpu().clone()
    if isinstance(obj,dict):return {k:cpu_tree(v) for k,v in obj.items()}
    if isinstance(obj,list):return [cpu_tree(v) for v in obj]
    if isinstance(obj,tuple):return tuple(cpu_tree(v) for v in obj)
    return obj


def atomic_save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as f:
            torch.save(cpu_tree(obj),f);f.flush();os.fsync(f.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


class EMA:
    def __init__(self,model,decay=.999,warmup=False):
        self.decay=decay
        self.warmup=warmup
        self.num_updates=0
        self.shadow={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    @torch.no_grad()
    def update(self,model,reference_updates=1.):
        self.num_updates+=reference_updates
        decay=min(self.decay,(1+self.num_updates)/(10+self.num_updates)) if self.warmup else self.decay
        decay=decay**reference_updates
        for n,p in model.named_parameters():
            if n in self.shadow:self.shadow[n].lerp_(p,1-decay)
    def load(self,state,device,num_updates=0):
        self.num_updates=float(num_updates)
        if set(state)!=set(self.shadow):raise ValueError("EMA trainable parameter set changed")
        self.shadow={n:v.to(device) for n,v in state.items()}
    @contextmanager
    def average(self,model):
        old={}
        with torch.no_grad():
            for n,p in model.named_parameters():
                if n in self.shadow:
                    old[n]=p.detach().clone();p.copy_(self.shadow[n])
        try:yield
        finally:
            with torch.no_grad():
                for n,p in model.named_parameters():
                    if n in old:p.copy_(old[n])


def autocast(device,precision):
    if precision=="fp32":return nullcontext()
    if precision not in ("bf16","fp16"):raise ValueError("precision must be fp32/bf16/fp16")
    if device.type=="cpu" and precision=="fp16":raise ValueError("fp16 CPU training unsupported")
    return torch.autocast(device.type,dtype=torch.bfloat16 if precision=="bf16" else torch.float16)


def _indices(n,step,micro,batch,seed):
    g=torch.Generator().manual_seed(seed+1000003*step+1009*micro)
    return torch.randint(n,(batch,),generator=g).tolist()


def train_contract(cfg,root,codec_path,require_spatial=None):
    inv=read_json(Path(root)/"inventory.json")
    validate_inventory(inv,require_spatial=cfg["train"].get("require_spatial",False) if require_spatial is None else require_spatial)
    if inv["latent_mode"]!=cfg["latent_mode"]:
        raise ValueError("Configured latent_mode differs from cached latent coordinates; re-encode, do not relabel")
    verify_codec(inv,codec_path)
    stats_path=Path(root)/"statistics.json"
    if stats_path.exists():
        stats=read_json(stats_path)
        if stats.get("inventory_identity")!=file_identity(Path(root)/"inventory.json"):
            raise ValueError("Cached train statistics belong to another inventory")
        if stats.get("codec_identity")!=file_identity(codec_path):
            raise ValueError("Statistics codec differs")
    else:stats=fit_statistics(root,inv)
    schema=fit_schema(inv["pairs"],**cfg["condition_fields"])
    return {"inventory_identity":file_identity(Path(root)/"inventory.json"),"codec_identity":file_identity(codec_path),
            "statistics":stats,"condition_schema":schema,"config":cfg,
            "image_normalization":inv.get("image_normalization"),"phase_order":inv.get("phase_order"),
            "spatial_policy":inv.get("spatial_policy"),
            "task_observations":{k:sum(p.get("labels",{}).get(k) is not None for p in inv["pairs"] if p["split"]=="train")
                                 for k in ("pcr","response_class","survival")}}


def make_world(cfg,contract,codec_path,device,trusted_legacy=False):
    codec=ThreePhaseCodec(load_codec(codec_path,trusted_legacy=trusted_legacy),
                          cfg["latent_mode"],cfg.get("decode_mode","requantize"))
    model=WorldModel(cfg,contract["condition_schema"],contract["statistics"],codec).to(device)
    model.data_contract=copy.deepcopy(contract)
    model.outcome_available={k:(not cfg.get("renderer_only",False)) and contract.get("task_observations",{}).get(k,0)>0 and cfg["loss"].get(w,0)>0
                             for k,w in (("pcr","pcr"),("response_class","response"),("survival","survival"))}
    return model


def validate_initialization(saved,contract,cfg,stage,*,warmstart=False):
    for key in ("inventory_identity","codec_identity","statistics","condition_schema"):
        if saved["contract"].get(key)!=contract.get(key):
            raise ValueError(f"Initialization {key} differs: do not mix caches or latent normalization")
    old=saved["config"]
    for key in ("latent_mode","decode_mode","renderer_only"):
        if old.get(key,False if key=="renderer_only" else None)!=cfg.get(key,False if key=="renderer_only" else None):
            raise ValueError(f"Initialization changes {key}; explicit model migration required")
    if saved["stage"]!="B":
        if old["flow"]["mode"]!=cfg["flow"]["mode"]:
            raise ValueError("Cannot reuse a velocity checkpoint in a different flow coordinate system")
    changed=[]
    for section,key,default in (("flow","time_scale",1.),("condition_options","time_scale_days",365.25)):
        if old.get(section,{}).get(key,default)!=cfg.get(section,{}).get(key,default):changed.append(f"{section}.{key}")
    if saved["stage"]!="B" and "condition_options.time_scale_days" in changed:
        raise ValueError("A trained clinical time embedding cannot be reparameterized silently; use a matching config or retrain stage C")
    if saved["stage"] in ("D","E") and "flow.time_scale" in changed:
        raise ValueError("A trained flow time embedding cannot be rescaled at load; use warmstart_existing.yaml or train a new D renderer")
    return {"source_stage":saved["stage"],"stage":stage,"warmstart":warmstart,
            "behavior_changes":changed,"optimizer_reset":True,"weights":"EMA when available"}


def _optimizer(model,cfg,stage,total_steps):
    base=cfg["train"]["learning_rate"]
    fast=[];slow=[]
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        if stage=="E" and name.split(".")[0] in ("response_encoder","dynamics","condition_encoder"):
            slow.append(p)
        else:fast.append(p)
    groups=[{"params":fast,"lr":base}]
    if slow:groups.append({"params":slow,"lr":base*cfg["train"].get("joint_semantic_lr_ratio",.1)})
    opt=torch.optim.AdamW(groups,weight_decay=cfg["train"].get("weight_decay",.01),betas=(.9,.95))
    warm=min(stage_budget(cfg,stage)["warmup_steps"],max(1,total_steps//4))
    def schedule(s):
        if s<warm:return max(.01,(s+1)/warm)
        progress=min(1.,(s-warm)/max(1,total_steps-warm))
        return .05+.95*.5*(1+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,schedule)
    return opt,scheduler


def stage_budget(cfg,stage):
    settings=for_stage(cfg,stage)["train"]
    batch=int(settings.get("stage_batches",{}).get(stage,settings.get("batch_size",1)))
    acc=int(settings.get("accumulation",1));reference=int(settings.get("reference_batch_size",0))
    if min(batch,acc)<1 or (reference and acc!=1):
        raise ValueError("Sample-budget training requires positive batch and accumulation=1")
    examples=int(settings["steps"][stage])*(reference or batch*acc)
    effective=batch*acc;scale=reference/effective if reference else 1.
    return {"batch":batch,"accumulation":acc,"steps":math.ceil(examples/effective),"examples":examples,
            "reference_batch":reference or effective,
            "warmup_steps":max(1,math.ceil(settings.get("warmup_steps",100)*scale)),
            "validation_every":max(1,math.ceil(settings.get("validation_every",100)*scale)),
            "checkpoint_every":max(1,math.ceil(settings.get("checkpoint_every",100)*scale))}


def validate(model,dataset,device,stage,*,seed=2026,max_batches=4,precision="fp32"):
    previous=model.training;saved=rng_state();seed_all(seed)
    model.eval();losses=[]
    try:
        with torch.no_grad(),autocast(device,precision):
            for i in selection_indices(dataset.records,max_batches,seed):
                if stage=="B":
                    # A validation source image, with no treatment and no longitudinal target.
                    source,_,_=dataset[i]
                    loss,_=model(source["image"][None].to(device),source["roi"][None].to(device))
                else:
                    source,target=collate([dataset[i]])
                    # Validation uses endpoint loss without expensive semantic ODE graphs.
                    weight=copy.deepcopy(model.cfg["loss"])
                    if stage=="E":weight.update(semantic=0.,change=0.)
                    weight["condition_dropout"]=0.
                    loss,_=model.training_loss(source.to(device),target.to(device),loss_cfg=weight)
                losses.append(float(loss))
        return sum(losses)/len(losses)
    finally:
        restore_rng(saved);model.train(previous)


def train_stage(cfg,root,codec_path,output,stage,*,init=None,resume=None,device="cpu",trusted_legacy=False,stop_after=None,warmstart=False,init_weights="ema",should_stop=None):
    if stage not in ("B","C","D","E"):raise ValueError("Use train_codec for stage A")
    if cfg.get("renderer_only",False) and stage!="D":
        raise ValueError("renderer_only is a stage D isolation baseline, not a semantic/clinical model")
    if init and resume:raise ValueError("Choose initialization OR exact resume, not both")
    if init_weights not in ("raw","ema"):raise ValueError("init_weights must be raw or ema")
    experiment_cfg=copy.deepcopy(cfg)
    cfg=for_stage(cfg,stage)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    device=torch.device(device)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA was requested but is unavailable")
    seed_all(cfg.get("seed",2026))
    torch.set_num_threads(cfg["train"].get("cpu_threads",4))
    contract=train_contract(experiment_cfg,root,codec_path,require_spatial=False if stage=="B" else cfg["train"].get("require_spatial",False))
    ensure_assets(root,read_json(Path(root)/"inventory.json"))
    if (output/"contract.json").exists() and read_json(output/"contract.json")!=contract:
        raise ValueError("Existing output contract differs; use a new directory")
    if (output/"last.pt").exists() and resume is None:
        raise ValueError("Output already contains training; use --resume or a new directory")
    write_json(output/"contract.json",contract)
    if stage=="B":
        encoder=MedicalEncoder3D(**cfg["semantic"])
        predictor=MaskPredictor(dim=cfg["semantic"]["dim"],**cfg["mask_predictor"])
        model=MaskedJEPA(encoder,predictor,cfg.get("mask_ratio",.5),**cfg.get("jepa_pretrain",{})).to(device)
        dataset=RepresentationDataset(root,preload=cfg["train"].get("preload",False))
        if init:
            if not warmstart:raise ValueError("Stage B --init requires --warmstart")
            saved=torch.load(init,map_location="cpu",weights_only=True)
            if saved["stage"]!="B":raise ValueError("Stage B warmstart requires a B checkpoint")
            validate_initialization(saved,contract,cfg,stage,warmstart=True)
            model.load_state_dict(saved["model_state"],strict=True)
    else:
        model=make_world(cfg,contract,codec_path,device,trusted_legacy)
        if init and not resume:
            saved=torch.load(init,map_location="cpu",weights_only=True)
            if not ((saved["stage"],stage) in (("B","C"),("C","D"),("D","E")) or (warmstart and saved["stage"]==stage)):
                raise ValueError("Use B->C->D->E or explicit --warmstart for same-stage optimization")
            provenance=validate_initialization(saved,contract,cfg,stage,warmstart=warmstart)
            provenance["weights"]=init_weights
            if saved["stage"]=="B" and stage=="C":
                encoder_state=saved["encoder_state"] if init_weights=="ema" else {k.removeprefix("online."):v for k,v in saved["model_state"].items() if k.startswith("online.")}
                model.initialize_encoder(encoder_state)
            else:
                state=saved["model_state"] if init_weights=="raw" else saved.get("inference_state",saved["model_state"])
                if saved["stage"]=="C" and stage=="D":
                    # C's renderer/bridge are untrained placeholders. Keep this
                    # run's declared initialization, not old random saved tensors.
                    state={k:(model.state_dict()[k] if k.startswith(("bridge.","flow.")) else v) for k,v in state.items()}
                    provenance["renderer_initialization"]="fresh per requested D config"
                model.load_state_dict(state,strict=True)
            write_json(output/"initialization.json",provenance)
        elif not resume and not cfg.get("allow_untrained_smoke",False) and not (stage=="D" and cfg.get("renderer_only",False)):
            raise ValueError(f"Stage {stage} needs the preceding trained checkpoint; random encoders are smoke-only")
        model.configure_stage(stage)
        dataset=PairDataset(root,"train",require_spatial=cfg["train"].get("require_spatial",False),preload=cfg["train"].get("preload",False))
    validation=PairDataset(root,"val",require_spatial=False if stage=="B" else cfg["train"].get("require_spatial",False),preload=cfg["train"].get("preload",False))
    budget=stage_budget(cfg,stage)
    total_steps=budget["steps"]
    stop=total_steps if stop_after is None else min(total_steps,int(stop_after))
    if stop<1:raise ValueError("stop_after must be positive")
    opt,scheduler=_optimizer(model,cfg,stage,total_steps)
    ema=None if stage=="B" else EMA(model,cfg["train"].get("ema_decay",.999),cfg["train"].get("ema_warmup",False))
    precision=cfg["train"].get("precision","fp32")
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda" and precision=="fp16")
    start=0;best=None;best_loss=None;bad_validations=0
    if resume:
        s=torch.load(resume,map_location="cpu",weights_only=True)
        if s["stage"]!=stage or s["contract"]!=contract:
            raise ValueError("Resume stage/config/data contract differs")
        model.load_state_dict(s["model_state"],strict=True)
        opt.load_state_dict(s["optimizer_state"]);scheduler.load_state_dict(s["scheduler_state"])
        scaler.load_state_dict(s["scaler_state"])
        if ema:ema.load(s["ema_state"],device,s.get("ema_updates",s["step"]))
        restore_rng(s["rng"]);start=s["step"];best=s["best"]
        best_loss=s.get("best_loss");bad_validations=s.get("bad_validations",0)
    batch=budget["batch"];acc=budget["accumulation"]
    if min(batch,acc)<1:raise ValueError("Batch size/accumulation must be positive")
    interval=budget["validation_every"]
    checkpoint_every=budget["checkpoint_every"]
    if stage=="B" and cfg.get("jepa_pretrain",{}).get("image_variance_weight",0)>0 and batch<2:
        raise ValueError("Stage B image variance needs actual microbatch>=2; gradient accumulation is not a substitute")
    if cfg["train"].get("limit_train_pairs"):
        if not hasattr(dataset,"records"):raise ValueError("limit_train_pairs applies to C/D/E only")
        chosen=selection_indices(dataset.records,int(cfg["train"]["limit_train_pairs"]),cfg.get("seed",2026))
        dataset.records=[dataset.records[i] for i in chosen]
    if cfg["train"].get("subset_pairs"):
        if not hasattr(dataset,"records"):raise ValueError("subset_pairs applies to C/D/E only")
        wanted=set(cfg["train"]["subset_pairs"])
        dataset.records=[p for p in dataset.records if p["pair_id"] in wanted]
        if len(dataset.records)!=len(wanted):raise ValueError("subset_pairs contains unknown/non-training pairs")
    logs=output/"metrics.jsonl"
    write_json(output/"effective_config.json",cfg)
    write_json(output/"training_budget.json",{**budget,"microbatch":batch,"effective_batch":batch*acc,
        "training_items":len(dataset),"max_updates":total_steps,"approximate_item_passes":budget["examples"]/len(dataset),
        "sampling":cfg["train"].get("sampling","pair_uniform"),"not_a_convergence_guarantee":True})
    if resume and s.get("training_complete",False):
        return read_json(output/"status.json")
    if cfg["train"].get("preload",False):
        filenames=dataset.files if stage=="B" else sorted({p[k] for p in dataset.records for k in ("source_file","target_file")})
        print(json.dumps({"stage":stage,"event":"preloading","views":len(filenames)}),flush=True)
        with ThreadPoolExecutor(max_workers=cfg["train"].get("preload_workers",8)) as pool:
            for _ in pool.map(dataset.reader.read,filenames):pass
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    done=start;interrupted=False;early_stop=False
    for step in range(start,stop):
        if device.type=="cuda":torch.cuda.synchronize(device)
        started=time.perf_counter()
        model.train();opt.zero_grad(set_to_none=True);mean_loss=0.;metrics={}
        current_batch=min(batch,budget["examples"]-step*batch) if cfg["train"].get("reference_batch_size",0) else batch
        for micro in range(acc):
            ids=training_indices(dataset,step,micro,current_batch,cfg.get("seed",2026),cfg["train"].get("sampling","pair_uniform"))
            if stage=="B":
                xs,rs=zip(*(dataset[i] for i in ids))
                image=torch.stack(xs).to(device);roi=torch.stack(rs).to(device)
                image,roi=augment_representation(image,roi,cfg.get("augmentation",{}))
                with autocast(device,precision):loss,terms=model(image,roi)
            else:
                s,t=collate([dataset[i] for i in ids]);s=s.to(device);t=t.to(device)
                with autocast(device,precision):loss,terms=model.training_loss(s,t,step=step)
            scaler.scale(loss/acc).backward()
            mean_loss+=float(loss.detach())/acc
            for k,v in terms.items():metrics[k]=metrics.get(k,0.)+float(v)/acc
            del loss
            if stage=="B":del image,roi
            else:del s,t
        scaler.unscale_(opt)
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg["train"].get("clip_grad",1.))
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient; no EMA/teacher/checkpoint update made")
        old_scale=scaler.get_scale();scaler.step(opt);scaler.update()
        if scaler.get_scale()<old_scale:
            raise FloatingPointError("AMP overflow; resume last valid checkpoint with safer precision")
        scheduler.step()
        if stage=="B":
            lo=cfg["train"].get("teacher_ema_start",.996)
            momentum=min(.99999,lo+(1-lo)*((step+1)/max(1,total_steps)))
            model.update_teacher(momentum**(current_batch*acc/budget["reference_batch"]))
        else:ema.update(model,current_batch*acc/budget["reference_batch"])
        if device.type=="cuda":torch.cuda.synchronize(device)
        done=step+1;interrupted=bool(should_stop and should_stop())
        row={"step":step+1,"stage":stage,"loss":mean_loss,"gradient_norm":float(norm),"lr":opt.param_groups[0]["lr"],**metrics}
        row.update(seconds=time.perf_counter()-started,sampled_examples=min(done*batch*acc,budget["examples"]),
                   target_examples=budget["examples"],configured_steps=total_steps,batch_size=current_batch)
        if device.type=="cuda":
            row.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2,
                       peak_reserved_mib=torch.cuda.max_memory_reserved(device)/1024**2)
        is_best=False;is_best_loss=False;early_stop=False
        opt.zero_grad(set_to_none=True)
        if not interrupted and ((step+1)%interval==0 or step+1==stop):
            with ema.average(model) if ema else nullcontext():
                score=validate(model,validation,device,stage,seed=cfg.get("seed",2026)+77,
                    max_batches=cfg["train"].get("validation_batches",4),precision=precision)
                row["validation_loss"]=score
                if best_loss is None or score<best_loss:best_loss=score;is_best_loss=True
                select=cfg["train"].get("selection","loss")
                if stage in ("D","E") and select=="generation":
                    settings=cfg.get("validation",{})
                    with autocast(device,precision):
                        gen=validate_generation(model,validation,device,seed=cfg.get("seed",2026)+77,
                            max_pairs=cfg["train"].get("validation_batches",8),steps=settings.get("steps",32),
                            solver=settings.get("solver","heun"),samples=settings.get("samples",1),
                            roi_weight=settings.get("roi_weight",.25))
                    score=gen["score"]
                    row.update(validation_generation=score,validation_copy_source=gen["copy_source"],validation_target_codec=gen["target_codec"])
                    write_json(output/"validation"/f"{step+1:07d}.json",gen)
                elif select not in ("generation","loss"):raise ValueError("selection must be loss or generation")
                row["selection_score"]=score
            if best is None or score<best:
                best=score;is_best=True;bad_validations=0
            else:bad_validations+=1
            patience=int(cfg["train"].get("early_stopping_patience",0))
            early_stop=patience>0 and bad_validations>=patience
        with logs.open("a") as f:f.write(json.dumps(row,allow_nan=False)+"\n")
        write_json(output/"progress.json",{**row,"status":"stopping" if interrupted else "running"})
        if (step+1)%cfg["train"].get("log_every",10)==0 or step==start:
            print(json.dumps(row,allow_nan=False),flush=True)
        if (step+1)%checkpoint_every==0 or step+1==stop or is_best or is_best_loss or early_stop or interrupted:
            if shutil.disk_usage(output).free<cfg["train"].get("disk_reserve_gib",0)*1024**3:
                raise RuntimeError("Insufficient disk reserve for atomic training checkpoint")
            state={"kind":"trjepa_training_v2","implementation_version":"0.2.0","stage":stage,"step":step+1,"best":best,"best_loss":best_loss,
                "bad_validations":bad_validations,"config":cfg,"contract":contract,
                "model_state":model.state_dict(),"optimizer_state":opt.state_dict(),"scheduler_state":scheduler.state_dict(),
                "scaler_state":scaler.state_dict(),"rng":rng_state(),"budget":budget,
                "sampled_examples":row["sampled_examples"],"training_complete":done==total_steps or early_stop}
            if stage=="B":state["encoder_state"]=model.target.state_dict()
            else:
                state["ema_state"]=ema.shadow
                state["ema_updates"]=ema.num_updates
                # One CPU copy for sampling; large checkpoints intentionally include optimizer + raw + EMA.
                with ema.average(model):state["inference_state"]=cpu_tree(model.state_dict())
            atomic_save(output/"last.pt",state)
            if is_best:atomic_save(output/"best.pt",state)
            if is_best_loss:atomic_save(output/"best_loss.pt",state)
        if early_stop or interrupted:break
    report={"checkpoint":str(output/"last.pt"),"best_checkpoint":str(output/"best.pt"),"stage":stage,"step":done,
            "device":str(device),"clinical_validation":False,"complete":done==total_steps or early_stop,
            "status":"complete" if done==total_steps or early_stop else "stopped",
            "early_stopped":early_stop,"budget":budget}
    write_json(output/"status.json",report)
    write_json(output/"progress.json",{**row,"status":report["status"]})
    return report


class CodecImageDataset(torch.utils.data.Dataset):
    def __init__(self,root,split="train"):
        self.root=Path(root);inv=read_json(self.root/"inventory.json")
        if inv["schema"] not in ("trjepa_images_only_v1","trjepa_three_phase_source_grid_v1"):
            raise ValueError("Stage A expects prepared raw normalized images, not pre-existing latent-only caches")
        self.files=sorted({v["file"] for v in inv["visits"] if v["split"]==split})
        if not self.files:raise ValueError("No images for codec training/validation")
    def __len__(self):return len(self.files)*3
    def __getitem__(self,index):
        phase=index%3
        with np.load(self.root/self.files[index//3],allow_pickle=False) as f:
            x=torch.from_numpy(np.array(f["image"][phase:phase+1],dtype=np.float32))
            r=torch.from_numpy(np.array(f["roi"],dtype=np.float32))
        return x,r


def reconstruction_terms(x,recon,roi):
    weights=1+2*roi
    rec=((x-recon).abs()*weights).sum()/weights.sum().clamp_min(1)
    multiscale=x.new_zeros(())
    for k in (2,4):
        if min(x.shape[-3:])>=k:multiscale=multiscale+F.l1_loss(F.avg_pool3d(x,k,k),F.avg_pool3d(recon,k,k))
    gradient=x.new_zeros(())
    for axis in (-3,-2,-1):
        gradient=gradient+F.l1_loss(torch.diff(x,dim=axis),torch.diff(recon,dim=axis))
    return rec,.5*multiscale+.1*gradient


def train_codec(cfg,root,output,*,device="cpu",resume=None,init=None,stop_after=None):
    """Optional stage A. Existing compatible VQ is the recommended main path.

    ROI reconstruction + multiscale/gradient reconstruction + commitment + 3D
    PatchGAN. No pretrained perceptual network is silently fabricated: this
    baseline does NOT call the multiscale term 'medical perceptual supervision'.
    """
    device=torch.device(device);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    seed_all(cfg.get("seed",2026));torch.set_num_threads(cfg["train"].get("cpu_threads",4))
    model=load_codec(init) if init else MRILevelVQGAN(VQGANConfig(**cfg["codec"]))
    model=model.to(device).train()
    disc=PatchDiscriminator3D(**cfg.get("discriminator",{})).to(device)
    data=CodecImageDataset(root,"train");val=CodecImageDataset(root,"val")
    settings=cfg["codec_train"];max_steps=settings["steps"]
    stop=max_steps if stop_after is None else min(max_steps,int(stop_after))
    opt=torch.optim.AdamW(model.parameters(),lr=settings["learning_rate"],betas=(.5,.9))
    dop=torch.optim.AdamW(disc.parameters(),lr=settings["learning_rate"],betas=(.5,.9))
    contract={"inventory_identity":file_identity(Path(root)/"inventory.json"),"config":cfg}
    if (output/"contract.json").exists() and read_json(output/"contract.json")!=contract:
        raise ValueError("Existing codec output contract differs; use a new directory")
    if (output/"last.pt").exists() and resume is None:
        raise ValueError("Codec output already contains training; use --resume or a new directory")
    write_json(output/"contract.json",contract)
    if stop<=0:
        raise ValueError("Codec stop-after/steps must be positive")
    start=0;best=None
    if resume:
        s=torch.load(resume,map_location="cpu",weights_only=True)
        if s["contract"]!=contract:raise ValueError("Codec resume contract differs")
        model.load_state_dict(s["codec_state"]);disc.load_state_dict(s["discriminator_state"])
        opt.load_state_dict(s["optimizer_state"]);dop.load_state_dict(s["discriminator_optimizer"])
        start=s["step"];best=s["best"];restore_rng(s["rng"])
    for step in range(start,stop):
        items=[data[i] for i in _indices(len(data),step,0,settings.get("batch_size",1),cfg.get("seed",2026))]
        x=torch.stack([v[0] for v in items]).to(device);roi=torch.stack([v[1] for v in items]).to(device)
        model.train();opt.zero_grad(set_to_none=True)
        disc.requires_grad_(False).eval()
        recon,info=model(x)
        rec,multiscale=reconstruction_terms(x,recon,roi)
        gan=-disc(recon).mean() if step>=settings.get("gan_start",1000) else x.sum()*0
        loss=rec+settings.get("multiscale_weight",.1)*multiscale+info["commitment_loss"]+settings.get("gan_weight",.01)*gan
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError("VQ gradients are nonfinite")
        opt.step()
        dloss=x.sum()*0
        if step>=settings.get("gan_start",1000):
            disc.requires_grad_(True).train();dop.zero_grad(set_to_none=True)
            dloss=.5*(F.relu(1-disc(x)).mean()+F.relu(1+disc(recon.detach())).mean())
            dloss.backward();torch.nn.utils.clip_grad_norm_(disc.parameters(),1.);dop.step()
        row={"step":step+1,"rec":float(rec.detach()),"gan":float(gan.detach()),"discriminator":float(dloss.detach()),
             "perplexity":float(info["perplexity"]),"loss":float(loss.detach())}
        is_best=False
        if (step+1)%settings.get("validation_every",100)==0 or step+1==stop:
            model.eval();losses=[]
            with torch.no_grad():
                for i in range(min(len(val),settings.get("validation_batches",4))):
                    vx,vr=val[i];vx=vx[None].to(device);vr=vr[None].to(device)
                    pred,_=model(vx);vl,_=reconstruction_terms(vx,pred,vr);losses.append(float(vl))
            score=sum(losses)/len(losses);row["val_rec"]=score
            if best is None or score<best:best=score;is_best=True
        with (output/"metrics.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
        if (step+1)%settings.get("checkpoint_every",100)==0 or step+1==stop or is_best:
            checkpoint={"schema":"trjepa_codec_v1","model_config":asdict(model.config),"codec_state":model.state_dict(),
                "discriminator_state":disc.state_dict(),"optimizer_state":opt.state_dict(),"discriminator_optimizer":dop.state_dict(),
                "step":step+1,"best":best,"rng":rng_state(),"contract":contract}
            atomic_save(output/"last.pt",checkpoint)
            if is_best:atomic_save(output/"best.pt",checkpoint)
        if step==start or (step+1)%settings.get("log_every",10)==0:print(json.dumps(row),flush=True)
    return {"codec":str(output/"last.pt"),"step":stop,"existing_weights_compatible_interface":True}
