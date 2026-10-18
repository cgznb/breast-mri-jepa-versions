"""Wait for the existing GPU run, measure stage batches, smoke-test, then train."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import copy
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from trjepa.config import load_config
from trjepa.data import file_identity,read_json,write_json,verify_identity
from trjepa.train import stage_budget


def process_identity(pid):
    try:
        fields=Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
        return None if fields[0]=="Z" else {"pid":pid,"start_ticks":fields[19]}
    except FileNotFoundError:return None


def safe_profile(report):
    return report.get("passed",False) and report["peak_reserved_mib"]<=.94*report["device_total_mib"]


def verify_smoke(root):
    import torch
    def load(stage):return torch.load(root/stage/"last.pt",map_location="cpu",weights_only=True)
    selected=torch.load(root/"B/best.pt",map_location="cpu",weights_only=True)
    encoder=selected["encoder_state"]
    reports={}
    for stage in "BCDE":
        saved=load(stage)
        if not saved["training_complete"] or not saved["optimizer_state"]["state"]:
            raise ValueError("Smoke optimizer/checkpoint incomplete")
        if not all(torch.isfinite(v).all() for v in saved["model_state"].values()):
            raise ValueError("Nonfinite smoke model")
        compared=0
        if stage!="B":
            for name,value in encoder.items():
                for prefix in ("encoder.","target_encoder."):
                    if not torch.equal(value,saved["model_state"][prefix+name]):
                        raise ValueError("Selected B EMA encoder was not preserved")
                    compared+=1
        rows=[json.loads(line) for line in (root/stage/"metrics.jsonl").read_text().splitlines()]
        if any(not math.isfinite(v) for row in rows for v in row.values() if isinstance(v,float)):
            raise ValueError("Nonfinite smoke metric")
        if stage=="E" and not any(row["semantic"]>0 for row in rows):
            raise ValueError("Joint smoke did not exercise the closed-loop objective")
        reports[stage]={"steps":saved["step"],"optimizer_states":len(saved["optimizer_state"]["state"]),
                        "frozen_encoder_tensors":compared,"validation_completed":any("validation_loss" in r for r in rows)}
        del saved
    report={"passed":True,"stages":reports,"real_data":True,"production_architecture":True}
    write_json(root/"verification.json",report)
    return report


def main():
    parser=argparse.ArgumentParser()
    for key in ("config","data","codec","output"):parser.add_argument("--"+key,required=True)
    parser.add_argument("--wait-for-pid",type=int);parser.add_argument("--detach",action="store_true")
    parser.add_argument("--resume",action="store_true");parser.add_argument("--lock-fd",type=int)
    args=parser.parse_args();root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    lock=os.fdopen(args.lock_fd,"a") if args.lock_fd is not None else (root/"pipeline.lock").open("a")
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit("This pipeline is already active")
    if (root/"pipeline_launch.json").exists() and not args.resume:raise FileExistsError("Existing pipeline: use --resume")
    if args.detach:
        command=[sys.executable,"-u","-B",str(Path(__file__).resolve()),"--lock-fd",str(lock.fileno())]
        for key in ("config","data","codec","output"):command.extend(["--"+key,str(Path(getattr(args,key)).resolve())])
        if args.wait_for_pid:command.extend(["--wait-for-pid",str(args.wait_for_pid)])
        if args.resume:command.append("--resume")
        env={**os.environ,"OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"1",
             "PYTORCH_ALLOC_CONF":"expandable_segments:True","CUDA_VISIBLE_DEVICES":"0"}
        with (root/"pipeline.log").open("a") as log:
            child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                   start_new_session=True,pass_fds=(lock.fileno(),))
        print(json.dumps({"pid":child.pid,"output":str(root),"log":str(root/"pipeline.log")}));return
    cfg=load_config(args.config)
    sources=sorted((REPO/"trjepa").glob("*.py"))+[REPO/"scripts"/p for p in
                ("launch_registered_roi32.py","run_registered_roi32.py","profile_registered_roi32.py")]
    sources+=[Path(args.data)/p for p in ("inventory.json","statistics.json","audit.json")]+[Path(args.codec)]
    binding={"config":cfg,"assets":[file_identity(p) for p in sources]}
    bound=root/"pipeline_binding.json"
    if bound.exists() and read_json(bound)!=binding:raise ValueError("Pipeline configuration or assets changed")
    if not bound.exists():write_json(bound,binding)
    stopped={"value":False};child=None
    def stop(signum,frame):
        stopped["value"]=True
        if child is not None and child.poll() is None:child.send_signal(signum)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    def status(state,**details):
        write_json(root/"pipeline_status.json",{"status":state,"pid":os.getpid(),
                   "updated_utc":datetime.now(timezone.utc).isoformat(),**details})
    def run(command,path):
        nonlocal child
        if stopped["value"]:raise InterruptedError("Pipeline stopped")
        for identity in binding["assets"]:verify_identity(identity)
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("a") as log:
            child=subprocess.Popen(command,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            if stopped["value"]:child.terminate()
            code=child.wait()
        child=None
        if stopped["value"]:raise InterruptedError("Pipeline stopped")
        if code:raise RuntimeError(f"Child exited {code}; inspect {path}")
    blocker=process_identity(args.wait_for_pid) if args.wait_for_pid else None
    write_json(root/"pipeline_launch.json",{"pid":os.getpid(),"blocker":blocker,"started_utc":datetime.now(timezone.utc).isoformat()})
    try:
        while True:
            if stopped["value"]:raise InterruptedError("Stopped while waiting")
            blocked=blocker is not None and process_identity(blocker["pid"])==blocker
            gpu_pids=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip().splitlines()
            if not blocked and not gpu_pids:break
            status("waiting_for_gpu",existing_run=blocker,gpu_pids=gpu_pids)
            time.sleep(10)
        preflight=root/"preflight";selected={};all_profiles={}
        for stage in "BCDE":
            reports={}
            def measure(batch):
                status("profiling",stage=stage,batch=batch)
                path=preflight/f"profile_{stage}_{batch}.json"
                if not path.exists():
                    run([sys.executable,"-u","-B",str(REPO/"scripts/profile_registered_roi32.py"),
                         "--config",str(Path(args.config).resolve()),"--data",str(Path(args.data).resolve()),
                         "--codec",str(Path(args.codec).resolve()),"--stage",stage,"--batch",str(batch),"--output",str(path)],path.with_suffix(".log"))
                reports[batch]=read_json(path);return reports[batch]
            low=0;high=1 if stage=="E" else 16
            while high<=1024:
                if not safe_profile(measure(high)):break
                low=high;high*=2
            if not low:raise RuntimeError(f"No safe starting batch for stage {stage}")
            # Refine the actual memory boundary, including non-power-of-two batches.
            for _ in range(4):
                quantum=1 if stage=="E" else 8
                candidate=((low+high)//(2*quantum))*quantum
                if candidate<=low or candidate>=high:break
                if safe_profile(measure(candidate)):low=candidate
                else:high=candidate
            safe=[r for r in reports.values() if safe_profile(r)]
            best_rate=max(r["examples_per_second"] for r in safe)
            efficient=[r for r in safe if r["examples_per_second"]>=.9*best_rate]
            selected[stage]=max(efficient,key=lambda r:r["batch"])
            all_profiles[stage]=list(reports.values())
        actual=copy.deepcopy(cfg);actual["train"]["stage_batches"]={s:r["batch"] for s,r in selected.items()}
        path=root/"selected_config.json"
        if path.exists() and read_json(path)!=actual:raise ValueError("Measured batch selection changed")
        if not path.exists():write_json(path,actual)
        write_json(preflight/"selection.json",{"selected":selected,"profiles":all_profiles,"allocator_limit_fraction":.94,
                   "throughput_floor_fraction":.9,"budgets":{s:stage_budget(actual,s) for s in "BCDE"}})
        smoke=copy.deepcopy(actual)
        smoke["train"].update(reference_batch_size=0,steps={s:3 for s in "BCDE"},preload=False,
                              validation_every=3,checkpoint_every=3,validation_batches=2,warmup_steps=1,log_every=1)
        smoke_root=preflight/"smoke";smoke_config=preflight/"smoke_config.json"
        if not (smoke_root/"verification.json").exists():
            write_json(smoke_config,smoke);status("smoke_training",batches=actual["train"]["stage_batches"])
            command=[sys.executable,"-u","-B",str(REPO/"scripts/run_registered_roi32.py"),"--config",str(smoke_config),
                     "--data",str(Path(args.data).resolve()),"--codec",str(Path(args.codec).resolve()),"--output",str(smoke_root)]
            if (smoke_root/"launch.json").exists():command.append("--resume")
            run(command,preflight/"smoke.log");verify_smoke(smoke_root)
        # The smoke weights are disposable; keep their verification and metrics.
        for checkpoint in smoke_root.glob("*/*.pt"):checkpoint.unlink()
        status("training",batches=actual["train"]["stage_batches"])
        command=[sys.executable,"-u","-B",str(REPO/"scripts/run_registered_roi32.py"),"--config",str(path),
                 "--data",str(Path(args.data).resolve()),"--codec",str(Path(args.codec).resolve()),"--output",str(root)]
        if (root/"launch.json").exists():command.append("--resume")
        run(command,root/"controller.log")
        status(read_json(root/"progress.json")["status"],batches=actual["train"]["stage_batches"])
    except InterruptedError as exc:status("stopped",reason=str(exc))
    except BaseException as exc:
        status("failed",error=str(exc));traceback.print_exc();raise


if __name__=="__main__":main()
