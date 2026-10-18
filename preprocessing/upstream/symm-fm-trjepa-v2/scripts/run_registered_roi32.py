"""Detached, locked B->C->D training with checkpointed SIGTERM and exact resume."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import traceback

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))


def main():
    parser=argparse.ArgumentParser()
    for key in ("config","data","codec","output"):parser.add_argument("--"+key,required=True)
    parser.add_argument("--resume",action="store_true");parser.add_argument("--detach",action="store_true")
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--lock-fd",type=int,default=None,help=argparse.SUPPRESS)
    args=parser.parse_args();root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    lock=os.fdopen(args.lock_fd,"a") if args.lock_fd is not None else (root/"training.lock").open("a")
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit("This training run is already active")
    if args.detach:
        command=[sys.executable,"-B",str(Path(__file__).resolve()),*[a for a in sys.argv[1:] if a!="--detach"],
                 "--lock-fd",str(lock.fileno())]
        with (root/"controller.log").open("a",buffering=1) as log:
            child=subprocess.Popen(command,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,pass_fds=(lock.fileno(),))
        print(json.dumps({"pid":child.pid,"output":str(root),"log":str(root/"controller.log")}),flush=True)
        return
    from trjepa.config import load_config
    from trjepa.data import ensure_assets,file_identity,read_json,verify_identity,write_json
    from trjepa.train import stage_budget,train_stage
    cfg=load_config(args.config)
    paths=sorted((REPO/"trjepa").glob("*.py"))+[Path(__file__),Path(args.config),Path(args.codec)]
    paths += [Path(args.data)/p for p in ("inventory.json","statistics.json")]
    binding={"config":cfg,"assets":[file_identity(p) for p in paths]}
    bound=root/"runtime_binding.json"
    if bound.exists() and read_json(bound)!=binding:raise ValueError("Bound training runtime changed")
    if (root/"launch.json").exists() and not args.resume:raise FileExistsError("Existing training: use --resume")
    if not bound.exists():write_json(bound,binding)
    ensure_assets(args.data,read_json(Path(args.data)/"inventory.json"))
    started=datetime.now(timezone.utc).isoformat()
    write_json(root/"launch.json",{"pid":os.getpid(),"started":started,"device":args.device,
        "stages":"BCD","budgets":{s:stage_budget(cfg,s) for s in "BCD"}})
    stopped={"value":False}
    def stop(signum,frame):stopped["value"]=True
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    results={};init=None;stage=None
    try:
        for stage in "BCD":
            if stopped["value"]:break
            for identity in binding["assets"]:verify_identity(identity)
            output=root/stage
            status=read_json(output/"status.json") if (output/"status.json").exists() else None
            if status and status["complete"]:
                for name in ("best.pt","last.pt"):
                    if not (output/name).is_file():raise FileNotFoundError("Completed stage lost checkpoint")
                results[stage]=status;init=str(output/"best.pt");continue
            write_json(root/"progress.json",{"status":"running","stage":stage,"stages":results,
                "pid":os.getpid(),"started":started})
            print(json.dumps({"event":"stage_start","stage":stage,"time":datetime.now(timezone.utc).isoformat()}),flush=True)
            resume=str(output/"last.pt") if args.resume and (output/"last.pt").exists() else None
            report=train_stage(cfg,args.data,args.codec,output,stage,init=None if resume else init,resume=resume,
                device=args.device,should_stop=lambda:stopped["value"])
            results[stage]=report
            if not report["complete"]:break
            init=report["best_checkpoint"]
            print(json.dumps({"event":"stage_complete","stage":stage,"time":datetime.now(timezone.utc).isoformat()}),flush=True)
        status="complete" if len(results)==3 and all(v["complete"] for v in results.values()) else "stopped"
        write_json(root/"progress.json",{"status":status,"stages":results,"pid":os.getpid(),
            "started":started,"updated":datetime.now(timezone.utc).isoformat()})
    except BaseException as exc:
        write_json(root/"progress.json",{"status":"failed","stage":stage,"error":str(exc),"stages":results})
        traceback.print_exc();raise


if __name__=="__main__":main()

