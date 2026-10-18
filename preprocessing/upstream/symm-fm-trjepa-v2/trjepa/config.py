from __future__ import annotations
from pathlib import Path
import copy
import yaml


def merge(base,overrides):
    base=copy.deepcopy(base)
    for k,v in overrides.items():
        base[k]=merge(base[k],v) if isinstance(v,dict) and isinstance(base.get(k),dict) else copy.deepcopy(v)
    return base


def load_config(path):
    path=Path(path).resolve()
    value=yaml.safe_load(path.read_text())
    if not isinstance(value,dict):raise ValueError("Config must be a YAML mapping")
    if "extends" in value:
        parent=path.parent/value.pop("extends")
        if parent.resolve()==path:raise ValueError("Recursive config extends")
        value=merge(load_config(parent),value)
    return value


def for_stage(cfg,stage):
    """Only optimization/loss settings vary across stages; architecture is fixed."""
    override=cfg.get("stage_overrides",{}).get(stage,{})
    if set(override)-{"train","loss","jepa_pretrain","augmentation"}:
        raise ValueError("Stage overrides may not silently change model/data coordinates")
    return merge(cfg,override)
