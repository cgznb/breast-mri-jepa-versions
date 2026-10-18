"""Training-only schema fitting and source-available structured tokenization."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import math
import re
import torch
from torch import Tensor, nn
from .blocks import ISAB, PMA, MLP, fourier

FORBIDDEN = re.compile(r"(^|_)(pcr|outcome|response|recurrence|survival|death|pathology|target|future|ypstage|rcb)(_|$)",re.I)


def numeric(x):
    if x is None or x == "":
        return None
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("Nonfinite condition value; use null for missing values")
    return x


def check_condition(record: dict) -> dict:
    allowed = {"clinical","treatment","delta_days","source_day","horizon_kind"}
    if set(record)-allowed:
        raise ValueError(f"Unknown condition keys: {sorted(set(record)-allowed)}")
    clinical = record.get("clinical",{})
    for key, value in clinical.items():
        if FORBIDDEN.search(key):
            raise ValueError(f"Forbidden outcome/future covariate: {key}")
        if isinstance(value,dict):
            if set(value)-{"value","available_day"}:
                raise ValueError("Timestamped covariate only supports value and available_day")
            if value.get("available_day") is None or record.get("source_day") is None:
                raise ValueError("Timestamped covariates require source_day and available_day")
            if numeric(value["available_day"]) > numeric(record["source_day"]):
                raise ValueError("Clinical covariate was unavailable at the source landmark")
    treatment = record.get("treatment") or {}
    if not isinstance(treatment,dict) or set(treatment)-{"regimen","drugs","known_at_source"}:
        raise ValueError("Treatment requires explicit regimen, drug set and optional known_at_source")
    if treatment.get("known_at_source") is False:
        raise ValueError("A realized future treatment cannot be a deployable source condition")
    drugs = treatment.get("drugs") or []
    names=[]
    for drug in drugs:
        if not isinstance(drug,dict) or set(drug)-{"name","dose"} or not drug.get("name"):
            raise ValueError("Drug must have a name and optional harmonized dose")
        numeric(drug.get("dose"))
        names.append(str(drug["name"]))
    if len(names)!=len(set(names)):
        raise ValueError("Duplicate drugs must be consolidated before set encoding")
    delta = numeric(record.get("delta_days"))
    if delta is not None and delta <= 0:
        raise ValueError("Query horizon must be positive elapsed DAYS, not visit index")
    if record.get("horizon_kind","query") not in ("query","planned","retrospective_observed"):
        raise ValueError("Unknown horizon availability contract")
    return record


def value_of(x):
    return x.get("value") if isinstance(x,dict) else x


def fit_scale(values):
    v = [numeric(x) for x in values if x is not None and x != ""]
    if not v:
        return {"mean":0.,"std":1.,"observed":0}
    mean = sum(v)/len(v)
    std = math.sqrt(sum((x-mean)**2 for x in v)/len(v))
    return {"mean":mean,"std":std if std>1e-8 else 1.,"observed":len(v)}


def fit_schema(records, *, clinical_numeric, clinical_categorical):
    """Caller passes canonical PAIR records; non-train rows are never fitted."""
    rows = [r["conditions"] for r in records if r["split"]=="train"]
    if not rows:
        raise ValueError("Schema fitting needs training patients")
    for row in rows:
        check_condition(row)
    keys = list(clinical_numeric)+list(clinical_categorical)
    if len(set(keys))!=len(keys) or any(FORBIDDEN.search(k) for k in keys):
        raise ValueError("Duplicate or forbidden clinical feature in schema")
    def vocab(values):
        return sorted({str(x) for x in values if x is not None and x != ""})
    scales = {k:fit_scale([value_of(r.get("clinical",{}).get(k)) for r in rows]) for k in clinical_numeric}
    cats = {k:vocab(value_of(r.get("clinical",{}).get(k)) for r in rows) for k in clinical_categorical}
    drugs = [d for r in rows for d in (r.get("treatment") or {}).get("drugs",[])]
    return {"version":"trjepa_conditions_v1","fit_split":"train","numeric":scales,"categorical":cats,
            "drug_vocab":vocab(d["name"] for d in drugs),"dose":fit_scale([d.get("dose") for d in drugs]),
            "regimen_vocab":vocab((r.get("treatment") or {}).get("regimen") for r in rows),
            "fields":keys,"n_training_pairs":len(rows)}


def cat_index(value, vocab):
    # 0 missing, 1 observed-but-unseen, >=2 fitted category. Padding is a separate mask.
    if value is None or value=="":
        return 0
    try:
        return vocab.index(str(value))+2
    except ValueError:
        return 1


class ConditionEncoder(nn.Module):
    def __init__(self, schema, dim=384, heads=6):
        super().__init__()
        self.schema,self.dim=schema,dim
        if schema["fit_split"]!="train":
            raise ValueError("Only train-fitted condition schemas can be loaded")
        self.numeric_names=list(schema["numeric"])
        self.cat_names=list(schema["categorical"])
        self.numeric_mlps=nn.ModuleList([nn.Sequential(nn.Linear(2,dim),nn.GELU(),MLP(dim)) for _ in self.numeric_names])
        self.categorical=nn.ModuleList([nn.Embedding(len(schema["categorical"][k])+2,dim) for k in self.cat_names])
        self.field=nn.Parameter(torch.randn(1,max(1,len(self.numeric_names)+len(self.cat_names)),dim)*.02)
        self.drug=nn.Embedding(len(schema["drug_vocab"])+2,dim)
        self.dose=nn.Sequential(nn.Linear(2,dim),nn.GELU(),MLP(dim))
        self.drug_blocks=nn.ModuleList([ISAB(dim,heads,16),ISAB(dim,heads,16)])
        self.drug_pool=PMA(dim,heads,4)
        self.regimen=nn.Embedding(len(schema["regimen_vocab"])+2,dim)
        self.time=nn.Sequential(nn.Linear(dim,dim),nn.SiLU(),nn.Linear(dim,dim))
        self.time_missing=nn.Embedding(2,dim)
        self.norm=nn.LayerNorm(dim)
    def forward(self, records):
        if not records:
            raise ValueError("Empty condition batch")
        for row in records:
            check_condition(row)
        device=self.field.device
        def floating(values):
            return torch.tensor(values,device=device,dtype=torch.float32)
        def indices(values):
            return torch.tensor(values,device=device,dtype=torch.long)
        tokens=[]
        for i,key in enumerate(self.numeric_names):
            scale=self.schema["numeric"][key]
            vals=[numeric(value_of(r.get("clinical",{}).get(key))) for r in records]
            x=floating([[0. if v is None else (v-scale["mean"])/scale["std"],float(v is None)] for v in vals])
            tokens.append(self.numeric_mlps[i](x)[:,None])
        for i,key in enumerate(self.cat_names):
            x=indices([cat_index(value_of(r.get("clinical",{}).get(key)),self.schema["categorical"][key]) for r in records])
            tokens.append(self.categorical[i](x)[:,None])
        clinical=torch.cat(tokens,1)+self.field[:,:len(tokens)] if tokens else self.field.expand(len(records),-1,-1)
        treatments=[r.get("treatment") or {} for r in records]
        sets=[r.get("drugs") or [{"name":None,"dose":None}] for r in treatments]
        width=max(map(len,sets))
        did=torch.zeros(len(records),width,device=device,dtype=torch.long)
        doses=torch.zeros(len(records),width,2,device=device)
        valid=torch.zeros(len(records),width,device=device,dtype=torch.bool)
        scale=self.schema["dose"]
        for b,drugs in enumerate(sets):
            for j,d in enumerate(drugs):
                did[b,j]=cat_index(d["name"],self.schema["drug_vocab"])
                v=numeric(d.get("dose"))
                doses[b,j]=floating([0. if v is None else (v-scale["mean"])/scale["std"],float(v is None)])
                valid[b,j]=True
        drug=self.drug(did)+self.dose(doses)
        for block in self.drug_blocks:
            drug=block(drug,valid)
        drug=self.drug_pool(drug,valid)
        regimen=self.regimen(indices([cat_index(r.get("regimen"),self.schema["regimen_vocab"]) for r in treatments]))[:,None]
        delta=[numeric(r.get("delta_days")) for r in records]
        days=floating([0 if d is None else d/365.25 for d in delta])
        time=self.time(fourier(days,self.dim))+self.time_missing(indices([int(d is None) for d in delta]))
        return self.norm(torch.cat((clinical,drug,regimen,time[:,None]),1))
