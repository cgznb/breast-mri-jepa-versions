"""Explicit cache contracts, train-only fitting and source-only inference inputs."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import random
import numpy as np
import torch
from torch import Tensor
from . import PHASES, UPSTREAM_COMMIT
from .conditions import check_condition, fit_schema

SCHEMA="trjepa_three_phase_source_grid_v1"


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(2**20),b""):
            h.update(chunk)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def file_identity(path):
    path=Path(path).resolve(); stat=path.stat()
    return {"path":str(path),"size_bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns}


def verify_identity(identity):
    if file_identity(identity["path"])!=identity:
        raise ValueError("Bound asset changed: "+identity["path"])


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


def atomic_npz(path,**arrays):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f".{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f,**arrays)
    tmp.replace(path)


@dataclass
class SourceBatch:
    image: Tensor
    latent: Tensor                   # raw, codec-space latent; normalize in model
    roi: Tensor
    conditions: list[dict]
    ids: list[str]
    def to(self,device):
        return SourceBatch(self.image.to(device),self.latent.to(device),self.roi.to(device),self.conditions,self.ids)


@dataclass
class TargetBatch:
    image: Tensor
    latent: Tensor
    roi: Tensor
    support: Tensor
    labels: list[dict]
    spatial_valid: Tensor
    def to(self,device):
        return TargetBatch(self.image.to(device),self.latent.to(device),self.roi.to(device),self.support.to(device),self.labels,self.spatial_valid.to(device))


class CacheReader:
    """Centralized cache read audit. Target reads are forbidden in source-only use.

    This is a deterministic API/file-access audit, not an OS security sandbox.
    The model's deployment API has no TargetBatch parameter at all.
    """
    def __init__(self,root,allowed=None,preload=False):
        self.root=Path(root).resolve()
        self.allowed=None if allowed is None else {str(self.resolve(p)) for p in allowed}
        self.opened=[]
        self.preload=preload
        self.cached={}
    def resolve(self,name):
        p=(self.root/name).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError("Cache path escapes the explicitly declared root")
        return p
    def read(self,name):
        path=self.resolve(name)
        if self.allowed is not None and str(path) not in self.allowed:
            raise PermissionError("Source-only inference attempted an undeclared cache read")
        if str(path) in self.cached:
            return self.cached[str(path)]
        with np.load(path,allow_pickle=False) as f:
            if set(f.files)!={"image","latent","roi","support"}:
                raise ValueError("Cache requires separate raw image, latent, ROI, and support arrays")
            arrays={k:np.asarray(f[k],dtype=np.float32) for k in f.files}
        image,latent,roi,support=(arrays[k] for k in ("image","latent","roi","support"))
        if image.ndim!=4 or image.shape[0]!=3 or latent.ndim!=4 or latent.shape[0]%3:
            raise ValueError("Corrupt three-phase cache shape")
        if roi.shape!=(1,*image.shape[-3:]) or support.shape!=image.shape:
            raise ValueError("Mask or support geometry differs from image")
        if tuple(s//4 for s in image.shape[-3:])!=latent.shape[-3:]:
            raise ValueError("Image/latent 4x geometry contract differs")
        if any(not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError("Nonfinite cached input")
        if (roi<0).any() or (roi>1).any() or (support<0).any() or (support>1).any():
            raise ValueError("ROI/support must lie in [0,1]")
        if self.allowed is not None:
            self.opened.append(str(path))
        result={k:torch.from_numpy(np.array(v,copy=True)) for k,v in arrays.items()}
        if self.preload:
            self.cached[str(path)]=result
        return result


def validate_inventory(inventory,*,require_spatial=False):
    if inventory.get("schema")!=SCHEMA or tuple(inventory.get("phase_order",()))!=PHASES:
        raise ValueError("This is not a TR-JEPA source-grid three-phase cache")
    if inventory.get("latent_mode") not in ("prequant","quantized"):
        raise ValueError("Latent coordinate type must be declared explicitly")
    if inventory.get("spatial_policy")!="source_defined_grid":
        raise ValueError("Legacy target-centered cache cannot be silently treated as source-grid residual data")
    assignment={}; pair_ids=set(); files={}
    for p in inventory["pairs"]:
        if p["split"] not in ("train","val","test"):
            raise ValueError("Unknown patient split")
        if assignment.setdefault(p["patient_id"],p["split"])!=p["split"]:
            raise ValueError("Patient leakage between dataset splits")
        if p["pair_id"] in pair_ids:
            raise ValueError("Duplicate pair ID")
        pair_ids.add(p["pair_id"])
        check_condition(p["conditions"])
        if p.get("source_day") is not None and p.get("target_day") is not None and p["target_day"]<=p["source_day"]:
            raise ValueError("Longitudinal targets must be later than the source")
        for key in ("source_file","target_file"):
            old=files.setdefault(p[key],(p["patient_id"],p["split"]))
            if old!=(p["patient_id"],p["split"]):
                raise ValueError("Cache reused across different patients or splits")
        if require_spatial and not p.get("spatial_correspondence_verified",False):
            raise ValueError("Spatial residual loss requires audited longitudinal correspondence; resampling alone is insufficient")
        labels=p.get("labels",{})
        survival=labels.get("survival")
        if survival is not None:
            if survival.get("landmark_day")!=p.get("source_day"):
                raise ValueError("Survival landmark must equal source observation time")
            if survival["duration_days"]<=0 or survival["event"] not in (0,1):
                raise ValueError("Survival labels require positive landmark-relative follow-up and event=0/1")
    if not pair_ids:
        raise ValueError("Empty cache inventory")


def fit_statistics(root,inventory):
    validate_inventory(inventory)
    reader=CacheReader(root)
    filenames=sorted({p[k] for p in inventory["pairs"] if p["split"]=="train" for k in ("source_file","target_file")})
    count=0; total=square=None
    for name in filenames:
        x=reader.read(name)["latent"].double().flatten(1)
        total=x.sum(1) if total is None else total+x.sum(1)
        square=x.square().sum(1) if square is None else square+x.square().sum(1)
        count+=x.shape[1]
    if count==0:
        raise ValueError("No training latents for normalization")
    mean=total/count
    std=(square/count-mean.square()).clamp_min(0).sqrt()
    if (std<1e-8).any():
        raise ValueError("Degenerate latent channels; inspect VQ codebook and data")
    return {"fit_split":"train","mean":mean.tolist(),"std":std.tolist(),"unique_files":filenames,
            "latent_mode":inventory["latent_mode"],"codec_identity":inventory.get("codec_identity")}


class PairDataset(torch.utils.data.Dataset):
    def __init__(self,root,split,*,require_spatial=False,preload=False):
        self.root=Path(root)
        self.inventory=read_json(self.root/"inventory.json")
        validate_inventory(self.inventory,require_spatial=require_spatial)
        self.records=[p for p in self.inventory["pairs"] if p["split"]==split]
        if not self.records:
            raise ValueError(f"No pairs for split {split}")
        self.reader=CacheReader(root,preload=preload)
    def __len__(self):
        return len(self.records)
    def __getitem__(self,index):
        p=self.records[index]
        return self.reader.read(p["source_file"]),self.reader.read(p["target_file"]),p


def collate(items):
    s,t,p=zip(*items)
    def stack(values,key):
        return torch.stack([v[key] for v in values])
    source=SourceBatch(stack(s,"image"),stack(s,"latent"),stack(s,"roi"),[r["conditions"] for r in p],[r["pair_id"] for r in p])
    target=TargetBatch(stack(t,"image"),stack(t,"latent"),stack(t,"roi"),stack(t,"support"),[r.get("labels",{}) for r in p],
        torch.tensor([r.get("spatial_correspondence_verified",False) for r in p]))
    return source,target


class RepresentationDataset(torch.utils.data.Dataset):
    """Each distinct training image view once, without pairs, treatment, future targets."""
    def __init__(self,root,preload=False):
        self.root=Path(root); self.reader=CacheReader(root,preload=preload)
        inv=read_json(self.root/"inventory.json"); validate_inventory(inv)
        # Prefer canonical per-visit self-grid views when available, preventing pair count weighting.
        self.files=inv.get("representation_train_files") or sorted({p[k] for p in inv["pairs"] if p["split"]=="train"
                                                                  for k in ("source_file","target_file")})
        all_train={p[k] for p in inv["pairs"] if p["split"]=="train" for k in ("source_file","target_file")}
        all_train.update(v["file"] for v in inv.get("visits",[]) if v["split"]=="train")
        if not self.files or not set(self.files)<=all_train:
            raise ValueError("Representation training cannot include held-out visits")
    def __len__(self):
        return len(self.files)
    def __getitem__(self,i):
        x=self.reader.read(self.files[i])
        return x["image"],x["roi"]


def source_only(root,case):
    allowed={"source_file","conditions","pair_id"}
    if set(case)-allowed:
        raise ValueError("Inference case must not contain target paths, target ROI, outcomes or future images")
    check_condition(case["conditions"])
    reader=CacheReader(root,allowed=[case["source_file"]])
    x=reader.read(case["source_file"])
    b=SourceBatch(x["image"][None],x["latent"][None],x["roi"][None],[case["conditions"]],[case.get("pair_id","query")])
    return b,{"opened":reader.opened,"target_reads":0,"audit":"declared-source-cache-only"}


def ensure_assets(root,inventory):
    """Expensive strict verification intended once on launch, not every minibatch."""
    identities=inventory.get("cache_identities",{})
    for name,identity in identities.items():
        if file_identity(Path(root)/name)!=identity:
            raise ValueError("Cache asset changed: "+name)
    expected=inventory.get("cache_sha256",{})
    for name,checksum in expected.items():
        if sha256(Path(root)/name)!=checksum:
            raise ValueError(f"Cache checksum differs: {name}")
    return {"files_verified":len(expected)+len(identities)}
