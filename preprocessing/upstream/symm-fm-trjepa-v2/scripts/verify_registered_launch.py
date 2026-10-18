"""Read back formal progress/checkpoints and confirm duplicate-launch protection."""
from pathlib import Path
import ast
import json
import math
import os
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from trjepa.data import file_identity,read_json,verify_identity,write_json

run=REPO/"runs/registered_roi32_20260922"
launch=read_json(run/"launch.json");os.kill(launch["pid"],0)
for identity in read_json(run/"runtime_binding.json")["assets"]:verify_identity(identity)
paths=sorted((REPO/"trjepa").glob("*.py"))+sorted((REPO/"scripts").glob("*.py"))+sorted((REPO/"tests").glob("*.py"))
for path in paths:ast.parse(path.read_text(),filename=str(path))
rows=[json.loads(line) for line in (run/"B/metrics.jsonl").read_text().splitlines() if line.strip()]
assert rows and [r["step"] for r in rows]==list(range(1,len(rows)+1))
for row in rows:
    assert all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float)))
validated=[r for r in rows if "validation_loss" in r]
assert validated,"Wait for the first formal validation"
saved=torch.load(run/"B/best.pt",map_location="cpu",weights_only=True)
assert saved["stage"]=="B" and saved["step"]>0 and saved["optimizer_state"]["state"]
assert all(torch.isfinite(v).all() for v in saved["model_state"].values())
assert saved["budget"]["batch"]==304 and saved["budget"]["examples"]==96000
assert saved["contract"]["statistics"]["fit_split"]=="train"
assert len(saved["contract"]["statistics"]["unique_files"])==2657
before=file_identity(run/"launch.json")
command=[sys.executable,"-B",str(REPO/"scripts/run_registered_roi32.py"),"--config",str(run/"selected_config.json"),
         "--data",str(REPO/"data/registered_roi32"),"--codec",saved["contract"]["codec_identity"]["path"],
         "--output",str(run),"--detach"]
duplicate=subprocess.run(command,cwd=REPO,text=True,capture_output=True)
assert duplicate.returncode!=0 and "already active" in duplicate.stderr
assert file_identity(run/"launch.json")==before
gpu=subprocess.run(["nvidia-smi","--query-gpu=memory.total,memory.used,utilization.gpu","--format=csv,noheader"],
                   check=True,text=True,capture_output=True).stdout.strip()
report={"passed":True,"pid":launch["pid"],"stage":"B","logged_updates":len(rows),
        "latest_logged_loss":rows[-1]["loss"],"validated_updates":[r["step"] for r in validated],
        "selected_checkpoint_update":saved["step"],"finite_checkpoint_tensors":True,
        "optimizer_state_present":True,"bound_runtime_unchanged":True,"duplicate_launch_rejected":True,
        "python_files_parsed":len(paths),"gpu_snapshot_total_used_utilization":gpu,
        "training_complete":False}
write_json(run/"launch_verification.json",report)
print(json.dumps(report),flush=True)
