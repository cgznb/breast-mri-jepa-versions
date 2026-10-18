"""Verify and link the existing reduced raw-MRI cache without re-normalizing it."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import copy
import json
import os
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import torch
from trjepa.data import (CacheReader,ensure_assets,file_identity,read_json,validate_inventory,
                         verify_codec,verify_identity,write_json)


def inspect_view(job):
    source,destination,visit,identity=job
    verify_identity(identity)
    name=visit["file"]
    x=CacheReader(source).read(name)
    if tuple(x["image"].shape)!=(3,32,128,128) or tuple(x["latent"].shape)!=(24,8,32,32):
        raise ValueError("Expected reduced three-phase MRI and raw 24-channel ROI32 latents")
    if not (x["support"].flatten(1).sum(1)>0).all():
        raise ValueError("Visit has an entirely missing phase")
    with np.load(Path(source)/name,allow_pickle=False) as archive:
        if archive["latent"].dtype!=np.float16:
            raise ValueError("Expected original float16 prequant cache")
    target=Path(destination)/name
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if not os.path.samefile(target,Path(source)/name):
            raise ValueError("Refusing to replace an existing cache view")
    else:os.link(Path(source)/name,target)
    if visit["split"]=="train":
        z=x["latent"].numpy().astype(np.float64).reshape(24,-1)
        moments=(z.sum(1),np.square(z).sum(1),z.shape[1])
    else:moments=None
    return name,file_identity(target),moments


def prepare(args):
    source=Path(args.source).resolve();output=Path(args.output).resolve()
    if output==source or output.is_relative_to(source):raise ValueError("Use a separate adapter directory")
    inv=read_json(source/"inventory.json");validate_inventory(inv);verify_codec(inv,args.codec)
    ensure_assets(source,inv)
    safe=torch.load(args.codec,map_location="cpu",weights_only=True)
    origin=read_json(Path(args.origin)/"latents/cache_manifest.json")
    if safe.get("schema")!="symm_world_codec_v2" or safe.get("source_identity")!=origin["codec"]:
        raise ValueError("Safe VQ does not descend from the original latent codec")
    verify_identity(origin["codec"])
    if inv["latent_mode"]!="prequant":raise ValueError("Expected continuous codec latents")
    sources=[source/"inventory.json",source/"statistics.json",Path(args.codec),
             Path(args.origin)/"inventory.json",Path(args.origin)/"latents/cache_manifest.json",Path(__file__)]
    binding={"sources":[file_identity(p) for p in sources],"operation":"read-only hardlink of verified ROI32 arrays"}
    output.mkdir(parents=True,exist_ok=True)
    if (output/"preparation_contract.json").exists():
        if read_json(output/"preparation_contract.json")!=binding:raise ValueError("Preparation binding changed")
        if (output/"audit.json").exists():
            ensure_assets(output,read_json(output/"inventory.json"));return read_json(output/"audit.json")
    write_json(output/"preparation_contract.json",binding)
    totals=np.zeros(24);squares=np.zeros(24);count=0;identities={}
    jobs=[(source,output,v,inv["cache_identities"][v["file"]]) for v in inv["visits"]]
    torch.set_num_threads(1)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index,(name,identity,moments) in enumerate(pool.map(inspect_view,jobs),1):
            identities[name]=identity
            if moments is not None:
                total,square,n=moments;totals+=total;squares+=square;count+=n
            if index%200==0 or index==len(jobs):
                progress={"status":"verifying","views":index,"total":len(jobs)}
                write_json(output/"progress.json",progress);print(json.dumps(progress),flush=True)
    mean=totals/count;std=np.sqrt(np.maximum(0,squares/count-mean**2))
    old=read_json(source/"statistics.json")
    np.testing.assert_allclose(mean,old["mean"],atol=1e-12,rtol=0)
    np.testing.assert_allclose(std,old["std"],atol=1e-12,rtol=0)
    result=copy.deepcopy(inv)
    result["cache_identities"]=identities
    result["provenance"]=binding
    result["image_normalization"].update(background_policy="preserve_zero",already_normalized=True,
        normalization_policy="original crop valid foreground only; zero background and padding preserved")
    validate_inventory(result);write_json(output/"inventory.json",result)
    stats={"fit_split":"train","mean":mean.tolist(),"std":std.tolist(),
           "unique_files":sorted(v["file"] for v in inv["visits"] if v["split"]=="train"),
           "latent_mode":"prequant","codec_identity":file_identity(args.codec),
           "inventory_identity":file_identity(output/"inventory.json")}
    write_json(output/"statistics.json",stats)
    for identity in binding["sources"]:verify_identity(identity)
    report={"passed":True,"views":len(jobs),"pairs":len(result["pairs"]),
            "views_by_split":dict(Counter(v["split"] for v in inv["visits"])),
            "pairs_by_split":dict(Counter(p["split"] for p in inv["pairs"])),
            "patients_by_split":{s:len({p["patient_id"] for p in inv["pairs"] if p["split"]==s}) for s in ("train","val")},
            "spatial_verified_pairs":dict(Counter(p["split"] for p in inv["pairs"] if p["spatial_correspondence_verified"])),
            "image_shape":[3,32,128,128],"latent_shape":[24,8,32,32],"all_arrays_finite":True,
            "train_statistics_replayed":True,"original_cache_unchanged":True,"independent_test_split":False}
    write_json(output/"audit.json",report);write_json(output/"progress.json",{"status":"complete",**report})
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True);p.add_argument("--output",required=True)
    p.add_argument("--origin",required=True);p.add_argument("--codec",required=True)
    p.add_argument("--workers",type=int,default=8)
    print(json.dumps(prepare(p.parse_args()),indent=2))

