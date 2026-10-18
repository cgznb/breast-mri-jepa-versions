"""Record the final tested configuration and smoke-driven memory adjustment."""
from pathlib import Path
import json
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from trjepa.config import load_config
from trjepa.data import read_json,write_json,file_identity
from trjepa.train import stage_budget

run=REPO/"runs/registered_roi32_20260922"
cfg=load_config(run/"selected_config.json")
profile=read_json(run/"preflight/selection.json")
smoke=read_json(run/"gpu_smoke/verification.json")
audit=read_json(REPO/"data/registered_roi32/audit.json")
verification=read_json(REPO/"data/registered_roi32/verification.json")
assert smoke["passed"] and audit["passed"] and verification["passed"]
assert cfg["train"]["stage_batches"]=={"B":304,"C":368,"D":224}
choice=next(r for r in profile["stages"]["D"]["candidates"] if r["batch"]==224)
assert choice["passed"]
profile["stages"]["D"]["initial_selection"]=profile["stages"]["D"]["selected"]
profile["stages"]["D"]["selected"]=choice
profile["smoke_adjustment"]={"stage":"D","from_batch":240,"to_batch":224,
    "observed_smoke_peak_reserved_mib":smoke["stages"]["D"]["progress"]["peak_reserved_mib"],
    "reason":"Full pipeline smoke passed but exceeded the allocator headroom target"}
profile["budgets"]={s:stage_budget(cfg,s) for s in "BCD"}
profile["selected_config"]=file_identity(run/"selected_config.json")
write_json(run/"preflight/selection.json",profile)
report={"passed":True,"selected_config":file_identity(run/"selected_config.json"),
        "data_audit":file_identity(REPO/"data/registered_roi32/audit.json"),
        "data_verification":file_identity(REPO/"data/registered_roi32/verification.json"),
        "gpu_smoke":file_identity(run/"gpu_smoke/verification.json"),
        "selection":file_identity(run/"preflight/selection.json"),"budgets":profile["budgets"],
        "formal_training_started":False}
write_json(run/"readiness.json",report)
print(json.dumps({"passed":True,"budgets":profile["budgets"]}),flush=True)
