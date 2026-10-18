"""One locked sequential B->C->D->E run, with graceful checkpointed interruption."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import signal
import sys
import traceback

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from trjepa.config import load_config
from trjepa.data import file_identity,read_json,write_json
from trjepa.train import stage_budget,train_stage


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True);parser.add_argument("--data",required=True)
    parser.add_argument("--codec",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--resume",action="store_true");parser.add_argument("--device",default="cuda")
    args=parser.parse_args();root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/"training.lock").open("a")
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit("This training run is already active")
    cfg=load_config(args.config)
    sources=sorted((REPO/"trjepa").glob("*.py"))+[Path(__file__),Path(args.codec)]
    sources+=[Path(args.data)/p for p in ("inventory.json","statistics.json")]
    binding={"config":cfg,"assets":[file_identity(p) for p in sources]}
    bound=root/"runtime_binding.json"
    if bound.exists() and read_json(bound)!=binding:raise ValueError("Training runtime changed; use a new run")
    if (root/"launch.json").exists() and not args.resume:raise FileExistsError("Existing training: use --resume")
    if not bound.exists():write_json(bound,binding)
    write_json(root/"launch.json",{"pid":os.getpid(),"device":args.device,"budgets":{s:stage_budget(cfg,s) for s in "BCDE"}})
    stopped={"value":False}
    def stop(signum,frame):stopped["value"]=True
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    results={};init=None
    try:
        for stage in "BCDE":
            if stopped["value"]:break
            for identity in binding["assets"]:
                if file_identity(identity["path"])!=identity:raise ValueError("Bound training asset changed")
            output=root/stage
            status=read_json(output/"status.json") if (output/"status.json").exists() else None
            if status and status["complete"]:
                if not (output/"best.pt").is_file():raise FileNotFoundError("Completed stage lost its selected checkpoint")
                results[stage]=status;init=str(output/"best.pt");continue
            write_json(root/"progress.json",{"status":"running","stage":stage,"stages":results,"pid":os.getpid()})
            resume=str(output/"last.pt") if args.resume and (output/"last.pt").exists() else None
            report=train_stage(cfg,args.data,args.codec,output,stage,init=init,resume=resume,device=args.device,
                               should_stop=lambda:stopped["value"])
            results[stage]=report
            if not report["complete"]:break
            init=report["best_checkpoint"]
        status="complete" if len(results)==4 and all(v["complete"] for v in results.values()) else "stopped"
        write_json(root/"progress.json",{"status":status,"stages":results,"pid":os.getpid()})
    except BaseException as exc:
        write_json(root/"progress.json",{"status":"failed","stage":stage,"error":str(exc),"stages":results})
        traceback.print_exc();raise


if __name__=="__main__":main()
