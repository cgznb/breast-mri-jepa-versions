"""Read-only live launch checks; write only an independent verification receipt."""
from pathlib import Path
import argparse
import json
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from trjepa.data import read_json,verify_identity,write_json


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True)
    args=parser.parse_args();root=Path(args.output).resolve()
    status=read_json(root/"pipeline_status.json");binding=read_json(root/"pipeline_binding.json")
    for identity in binding["assets"]:verify_identity(identity)
    pid=status["pid"]
    command=Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    if str(REPO/"scripts/launch_registered_roi32.py").encode() not in command:
        raise ValueError("Live process belongs to another pipeline")
    if str(root).encode() not in command:raise ValueError("Live pipeline output differs")
    progress={stage:read_json(root/stage/"progress.json") if (root/stage/"progress.json").exists() else None for stage in "BCDE"}
    report={"passed":True,"pid":pid,"status":status["status"],"bound_assets_checked":len(binding["assets"]),
            "formal_steps":{s:p["step"] if p else 0 for s,p in progress.items()},
            "GPU_profiling_completed":(root/"preflight/selection.json").exists(),
            "GPU_smoke_completed":(root/"preflight/smoke/verification.json").exists(),
            "status_utc":status["updated_utc"]}
    write_json(root/"launch_verification.json",report);print(json.dumps(report,indent=2))
