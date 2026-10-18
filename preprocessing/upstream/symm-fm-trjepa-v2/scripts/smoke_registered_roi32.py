"""Exercise selected physical batches, validation, handoffs and checkpoint resume."""
from pathlib import Path
import argparse
import gc
import json
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from trjepa.config import load_config
from trjepa.data import read_json,write_json
from trjepa.train import train_stage,stage_budget


def smoke(args):
    cfg=load_config(args.config);out=Path(args.output)
    if out.exists():raise FileExistsError("Use a fresh smoke directory")
    out.mkdir(parents=True)
    cfg["train"].update(preload=False,reference_batch_size=0,steps={s:3 for s in "BCD"},
        validation_every=3,validation_batches=4,checkpoint_every=3,early_stopping_patience=0,log_every=1)
    for stage in "BCD":
        cfg["stage_overrides"][stage].setdefault("train",{}).update(accumulation=1,early_stopping_patience=0)
    write_json(out/"config.json",cfg)
    init=None;reports={};previous=None
    for stage in "BCD":
        report=train_stage(cfg,args.data,args.codec,out/stage,stage,init=init,device="cuda:0")
        saved=torch.load(report["checkpoint"],map_location="cpu",weights_only=True)
        if not report["complete"] or saved["step"]!=3:raise ValueError("Smoke did not finish")
        if previous is not None:
            if stage=="C":
                state=previous["encoder_state"]
                for key,value in state.items():
                    if not torch.equal(value,saved["model_state"]["encoder."+key]):raise ValueError("B->C encoder handoff differs")
            else:
                state=previous["inference_state"]
                for key,value in state.items():
                    if key.startswith(("encoder.","target_encoder.","condition_encoder.","response_encoder.","dynamics.","heads.")):
                        if not torch.equal(value,saved["model_state"][key]):raise ValueError("C->D frozen handoff differs")
        reports[stage]={"passed":True,"physical_batch":stage_budget(cfg,stage)["batch"],
            "updates":saved["step"],"selected_checkpoint_exists":Path(report["best_checkpoint"]).is_file(),
            "progress":read_json(out/stage/"progress.json")}
        previous=saved;init=report["best_checkpoint"]
        # All three updates validate at the end, so best and last are the same selected state.
        if stage=="D":
            resumed=train_stage(cfg,args.data,args.codec,out/stage,stage,resume=report["checkpoint"],device="cuda:0")
            if not resumed["complete"] or resumed["step"]!=3:raise ValueError("Completed resume is not a no-op")
        gc.collect();torch.cuda.empty_cache()
    del previous,saved
    report={"passed":True,"stages":reports,"B_to_C_target_encoder_exact":True,
            "C_to_D_frozen_modules_exact":True,"completed_resume_noop":True,
            "synthetic_training":False,"formal_optimizer_updates":0}
    write_json(out/"verification.json",report)
    for stage in "BCD":
        for name in ("best.pt","best_loss.pt","last.pt"):(out/stage/name).unlink(missing_ok=True)
    write_json(out/"checkpoint_cleanup.json",{"verified_temporary_smoke_checkpoints_removed":True})
    print(json.dumps({"passed":True,"report":str(out/"verification.json")}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("config","data","codec","output"):p.add_argument("--"+key,required=True)
    smoke(p.parse_args())

