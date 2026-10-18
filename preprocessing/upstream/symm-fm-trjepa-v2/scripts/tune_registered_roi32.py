"""Isolate OOM candidates in subprocesses and freeze the measured run config."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from trjepa.config import load_config
from trjepa.data import file_identity,read_json,write_json
from trjepa.train import stage_budget


def tune(args):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    cfg=load_config(args.config);reports={};selected={}
    for stage,initial in (("B",64),("C",128),("D",32)):
        trials={}
        def measure(batch):
            path=output/f"{stage}_batch{batch}.json"
            command=[sys.executable,"-B",str(REPO/"scripts/profile_registered_roi32.py"),
                     "--config",str(Path(args.config).resolve()),"--data",str(Path(args.data).resolve()),
                     "--codec",str(Path(args.codec).resolve()),"--stage",stage,"--batch",str(batch),"--output",str(path)]
            print(json.dumps({"event":"profiling","stage":stage,"batch":batch}),flush=True)
            with path.with_suffix(".log").open("w") as log:
                subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
            result=read_json(path);trials[batch]=result
            safe=result["passed"] and result["peak_reserved_mib"]<=args.memory_fraction*result["device_total_mib"]
            print(json.dumps({"stage":stage,"batch":batch,"safe":safe,
                **{k:result[k] for k in ("peak_reserved_mib","examples_per_second","reason") if k in result}}),flush=True)
            return safe
        low=0;high=initial
        while measure(high):
            low=high;high*=2
            if high>2048:break
        while high-low>16:
            mid=((low+high)//2//8)*8
            if mid<=low:break
            if measure(mid):low=mid
            else:high=mid
        safe=[r for r in trials.values() if r["passed"] and r["peak_reserved_mib"]<=args.memory_fraction*r["device_total_mib"]]
        if not safe:raise RuntimeError("No safe production batch for "+stage)
        speed=max(r["examples_per_second"] for r in safe)
        choice=max((r for r in safe if r["examples_per_second"]>=.9*speed),key=lambda r:r["batch"])
        reports[stage]={"selected":choice,"candidates":list(trials.values())};selected[stage]=choice["batch"]
        write_json(output/"selection.json",{"status":"profiling","stages":reports,"memory_fraction":args.memory_fraction})
    cfg["train"]["stage_batches"]=selected
    write_json(output.parent/"selected_config.json",cfg)
    result={"status":"complete","stages":reports,"memory_fraction":args.memory_fraction,
            "budgets":{s:stage_budget(cfg,s) for s in "BCD"},
            "source_config":file_identity(args.config),"selected_config":file_identity(output.parent/"selected_config.json")}
    write_json(output/"selection.json",result)
    print(json.dumps({"selected_batches":selected,"budgets":result["budgets"]}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("config","data","codec","output"):p.add_argument("--"+key,required=True)
    p.add_argument("--memory-fraction",type=float,default=.95)
    tune(p.parse_args())

