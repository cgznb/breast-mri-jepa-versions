"""Fixed-patient source-only MRI forecasts from selected and final checkpoints."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import argparse
import gc
import json
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import torch
from trjepa.data import (CacheReader,ensure_assets,file_identity,read_json,source_only,
                         validate_inference_inventory,verify_identity,write_json,atomic_npz)
from trjepa.evaluate import load_world
from trjepa.losses import balanced_loss
from trjepa.validation import pair_seed


def select_cases(inventory):
    pairs=[p for p in inventory["pairs"] if p["split"]=="val"]
    rng=np.random.default_rng(20260922);used=set();chosen=[]
    for transition in sorted({p["transition"] for p in pairs}):
        group=sorted((p for p in pairs if p["transition"]==transition),key=lambda p:p["pair_id"])
        order=rng.permutation(len(group));count=0
        for index in order:
            p=group[int(index)]
            if p["patient_id"] in used:continue
            chosen.append(p);used.add(p["patient_id"]);count+=1
            if count==5:break
        if count!=5:raise ValueError("Insufficient distinct patients for balanced gallery")
    assert len(chosen)==len(used)==30
    return chosen


def score_arrays(image,target,support,roi):
    image,target,support,roi=(torch.from_numpy(np.asarray(x,np.float32))[None] for x in (image,target,support,roi))
    mask=support>0.5;region=mask*(roi>0.5)
    error=(image-target).abs()
    return {"supported_mae":float(error[mask].mean()),
            "roi_mae":float(error[region].mean()) if region.any() else None,
            "selection_score":float(balanced_loss(image,target,support,roi=roi,roi_weight=.25,kind="l1")),
            "phase_mae":[float(error[:,i][mask[:,i]].mean()) for i in range(3)]}


def training_audit(run,model):
    root=read_json(run/"progress.json")
    modules={name:sum(p.numel() for p in module.parameters()) for name,module in model.named_children()}
    total=sum(p.numel() for p in model.parameters())
    stages={}
    for stage in "BCD":
        rows=[json.loads(line) for line in (run/stage/"metrics.jsonl").read_text().splitlines() if line.strip()]
        cfg=read_json(run/stage/"effective_config.json")
        valid=[r for r in rows if "selection_score" in r]
        choice=min(valid,key=lambda x:x["selection_score"])
        stages[stage]={"reference_updates":cfg["train"]["steps"][stage],
            "actual_updates":rows[-1]["step"],"sampled_items":rows[-1]["sampled_examples"],
            "budget":root["stages"][stage]["budget"],"early_stopped":root["stages"][stage]["early_stopped"],
            "best_update":choice["step"],"best_score":choice["selection_score"],
            "update_seconds_sum":sum(r["seconds"] for r in rows),
            "median_update_seconds":float(np.median([r["seconds"] for r in rows])),
            "validations":[{k:r[k] for k in ("step","validation_loss","selection_score","validation_generation", "validation_copy_source") if k in r} for r in valid]}
    events=[json.loads(line) for line in (run/"controller.log").read_text().splitlines() if line.strip()]
    for stage,record in stages.items():
        a=next(x["time"] for x in events if x.get("event")=="stage_start" and x["stage"]==stage)
        b=next(x["time"] for x in events if x.get("event")=="stage_complete" and x["stage"]==stage)
        record["wall_minutes"]=(datetime.fromisoformat(b)-datetime.fromisoformat(a)).total_seconds()/60
    profile=read_json(run/"preflight/selection.json")
    return {"all_world_parameters_including_frozen_codec_and_teacher":total,"module_parameters":modules,
            "world_parameters_excluding_frozen_teacher_and_codec":total-modules["codec"]-modules["target_encoder"],
            "trainable_parameters_by_stage":{s:profile["stages"][s]["selected"]["trainable_parameters"] for s in "BCD"},
            "wall_minutes":(datetime.fromisoformat(root["updated"])-datetime.fromisoformat(root["started"])).total_seconds()/60,
            "semantic":model.cfg["semantic"],"flow":model.cfg["flow"],
            "image_shape":[3,32,128,128],"latent_shape":[24,8,32,32],
            "semantic_spatial_tokens":256,"flow_spatial_tokens":256,"stages":stages,
            "sample_budget_is_not_optimizer_equivalence":True,"training_restarted":False}


@torch.inference_mode()
def main(args):
    torch.set_num_threads(6)
    run=Path(args.run).resolve();data=Path(args.data).resolve();out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    inventory=read_json(data/"inventory.json");ensure_assets(data,inventory)
    pairs=select_cases(inventory)
    paths=[data/"inventory.json",data/"statistics.json",Path(args.codec),
           run/"D/best.pt",run/"D/last.pt",run/"selected_config.json",Path(__file__)]
    paths+=sorted((REPO/"trjepa").glob("*.py"))
    bindings=[file_identity(p) for p in paths]
    protocol={"selection_seed":20260922,"case_count":30,"patient_count":30,
        "selection":"Five randomly ordered distinct patients per transition, no prediction/outcome selection",
        "steps":32,"solver":"heun","precision":"bf16_network_fp32_decoder","best_draws":4,"last_draws":1,
        "common_first_noise_best_last":True,"independent_test":False,"optimizer_updates":0,
        "cases":[{"case":f"C{i+1:02d}","pair_id":p["pair_id"],"patient_id":p["patient_id"],
            "source_file":p["source_file"],"target_file":p["target_file"],"transition":p["transition"],
            "delta_days":p["conditions"]["delta_days"],"spatial_verified":p["spatial_correspondence_verified"],
            "seed":pair_seed(p["pair_id"],20260999)} for i,p in enumerate(pairs)],"bindings":bindings}
    if (out/"protocol.json").exists() and read_json(out/"protocol.json")!=protocol:
        raise ValueError("Visualization protocol changed; use another output directory")
    write_json(out/"protocol.json",protocol)
    started=time.perf_counter()
    for tag,draws in (("best",4),("last",1)):
        model,saved=load_world(run/"D"/(tag+".pt"),args.codec,"cuda:0")
        model.eval().requires_grad_(False)
        validate_inference_inventory(saved["contract"],data)
        if tag=="best":write_json(out/"training_audit.json",training_audit(run,model))
        for ordinal,(case,pair) in enumerate(zip(protocol["cases"],pairs),1):
            path=out/"volumes"/(case["case"]+"_"+tag+".npz")
            receipt=path.with_suffix(".json")
            if receipt.exists():
                verify_identity(read_json(receipt)["output"]);continue
            src,audit=source_only(data,{k:pair[k] for k in ("pair_id","source_file","conditions")})
            observed=src.to("cuda:0")
            with torch.autocast("cuda",dtype=torch.bfloat16):
                forecast=model.sample(observed,samples=draws,steps=32,solver="heun",seed=case["seed"])
            images=forecast["images"][:,0].float().cpu().numpy()
            latents=forecast["latents"][:,0].float().cpu().numpy()
            if images.shape!=(draws,3,32,128,128) or not np.isfinite(images).all():raise ValueError("Invalid generated MRI")
            atomic_npz(path,images=images,latents=latents)
            assert audit["target_reads"]==0
            if tag=="best":
                target=CacheReader(data).read(pair["target_file"])
                decoded=model.codec.decode(target["latent"][None].to("cuda:0"))[0].cpu().numpy()
                values={k:v.numpy() for k,v in target.items() if k in ("image","roi","support")}
                pixels=src.image[0].numpy();mask=src.support[0].numpy()>.5
                window_values=pixels[mask&(pixels!=0)]
                low,high=np.percentile(window_values,[1,99.5]);low=min(float(low),0.)
                high=max(float(high),low+1e-3)
                center=int(src.roi[0,0].sum((-1,-2)).argmax())
                atomic_npz(out/"volumes"/(case["case"]+"_reference.npz"),source=pixels,target=values["image"],
                    target_codec=decoded,roi=values["roi"],support=values["support"],source_roi=src.roi[0].numpy(),
                    source_support=src.support[0].numpy(),window=np.array([low,high]),center=np.array(center))
            write_json(receipt,{"case":case["case"],"checkpoint_step":saved["step"],"draws":draws,
                "noise_seed":case["seed"],"target_reads_before_prediction":0,"output":file_identity(path)})
            progress={"phase":"inference","checkpoint":tag,"case":case["case"],"complete":ordinal,
                      "total":30,"seconds":round(time.perf_counter()-started,1)}
            write_json(out/"progress.json",progress);print(json.dumps(progress),flush=True)
            del observed,forecast,images,latents
        del model,saved
        gc.collect();torch.cuda.empty_cache()
    cases=[]
    for case in protocol["cases"]:
        with np.load(out/"volumes"/(case["case"]+"_reference.npz")) as f:ref={k:f[k] for k in f.files}
        with np.load(out/"volumes"/(case["case"]+"_best.npz")) as f:best=f["images"]
        with np.load(out/"volumes"/(case["case"]+"_last.npz")) as f:last=f["images"][0]
        predictions={"copy_source":ref["source"],"best_sample":best[0],"best_mean":best.mean(0),
                     "last_sample":last,"target_codec":ref["target_codec"]}
        metrics={k:score_arrays(v,ref["target"],ref["support"],ref["roi"]) for k,v in predictions.items()}
        per_draw=[score_arrays(x,ref["target"],ref["support"],ref["roi"]) for x in best]
        metrics["best_expected_sample"]={key:float(np.mean([m[key] for m in per_draw]))
            for key in ("supported_mae","roi_mae","selection_score")}
        case_public={k:case[k] for k in ("case","transition","delta_days","spatial_verified","seed")}
        cases.append({**case_public,"center":int(ref["center"]),"window":ref["window"].tolist(),"metrics":metrics,
            "mean_draw_sd":float(best.std(0).mean())})
    methods=cases[0]["metrics"].keys()
    summary={name:{key:float(np.mean([c["metrics"][name][key] for c in cases]))
                  for key in ("supported_mae","roi_mae","selection_score")} for name in methods}
    for identity in bindings:verify_identity(identity)
    report={"cases":cases,"summary":summary,"case_count":30,"patient_count":30,
        "spatial_verified_cases":sum(c["spatial_verified"] for c in cases),
        "transitions":dict(Counter(c["transition"] for c in cases)),"optimizer_updates":0,
        "checkpoint_steps":{"best":144,"last":576},"source_only_prediction":True,"bindings_unchanged":True,
        "independent_test":False,"seconds":time.perf_counter()-started}
    write_json(out/"report.json",report)
    write_json(out/"progress.json",{"phase":"complete","cases":30,"seconds":report["seconds"]})
    print(json.dumps({"phase":"complete","summary":summary,"seconds":report["seconds"]}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("run","data","codec","output"):p.add_argument("--"+key,required=True)
    main(p.parse_args())
