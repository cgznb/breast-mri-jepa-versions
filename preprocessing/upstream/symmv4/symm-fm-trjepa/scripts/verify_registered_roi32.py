"""Independent real-data checks and optional full-size CPU update checks."""
from pathlib import Path
import argparse
import json
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/"scripts"))
import numpy as np
import torch
from trjepa.codec import load_codec,ThreePhaseCodec
from trjepa.config import load_config
from trjepa.data import read_json,write_json,CacheReader,ensure_assets,file_identity
from trjepa.train import train_contract,stage_budget,validation_indices
from trjepa.data import PairDataset
from profile_registered_roi32 import profile


def main():
    parser=argparse.ArgumentParser()
    for key in ("data","codec","config","output"):parser.add_argument("--"+key,required=True)
    parser.add_argument("--production-updates",action="store_true")
    args=parser.parse_args();torch.set_num_threads(4)
    root=Path(args.data);cfg=load_config(args.config);inv=read_json(root/"inventory.json")
    assets=ensure_assets(root,inv);contract=train_contract(cfg,root,args.codec)
    assignments={}
    for row in inv["visits"]:
        if assignments.setdefault(row["patient_id"],row["split"])!=row["split"]:raise ValueError("Patient leakage")
    val=PairDataset(root,"val")
    indices=validation_indices(val,102,True)
    if len({val.records[i]["patient_id"] for i in indices})!=102:raise ValueError("Validation does not cover 102 patients")
    codec=ThreePhaseCodec(load_codec(args.codec)).eval()
    sys.path.insert(0,"/root/autodl-tmp/MAM/symm-fm-world-v2/src")
    from symm_world.codec import load_codec as load_reference
    reference=load_reference(args.codec)
    reader=CacheReader(root)
    sample=reader.read(inv["representation_train_files"][0])
    z=sample["latent"][None]
    with torch.no_grad():
        expected=reference.decode(z);actual=codec.decode(z)
    error=float((expected-actual).abs().max())
    if actual.shape!=(1,3,32,128,128) or error>1e-6:raise ValueError("Original VQ decoder differs")
    del reference,expected,actual,codec
    # Compare actual arrays against the original crop loader, including coverage and T0 ROI.
    sys.path.insert(0,"/root/autodl-tmp/MAM/MeWM-ISPY2")
    from mewm_ispy2.registered_three_phase_data import load_config as upstream_config,ThreePhaseCrops
    _,baseline=upstream_config("/root/autodl-tmp/MAM/MeWM-ISPY2/configs/registered_three_phase_roi32_v1.yaml")
    original=read_json("/root/autodl-tmp/MAM/MeWM-ISPY2/runs/registered_three_phase_roi32_v1/inventory.json")
    records=[original["visits"][i] for i in (0,len(original["visits"])//2,len(original["visits"])-1)]
    dataset=ThreePhaseCrops(baseline,original,records=records)
    comparison=[]
    for i,record in enumerate(records):
        old=dataset[i];new=reader.read("views/"+record["visit_id"].replace(":","_")+".npz")
        errors={key:float((old[a].float()-new[b].float()).abs().max()) for key,a,b in
                (("image","image","image"),("roi","mask","roi"),("support","coverage","support"))}
        if any(errors.values()):raise ValueError("Prepared MRI differs from the original reduced crop loader")
        comparison.append(errors)
    report={"passed":True,"patients":len(assignments),"assets":assets,"patient_disjoint_splits":True,
            "validation_patients":102,"decoder_max_error":error,"original_crop_comparisons":comparison,
            "budgets":{s:stage_budget(cfg,s) for s in "BCDE"},"production_cpu_profiles":{}}
    if args.production_updates:
        for stage in "BCDE":
            print(json.dumps({"event":"full_size_cpu_updates","stage":stage}),flush=True)
            result=profile(cfg,root,args.codec,stage,1,updates=2,device="cpu")
            if not result["passed"]:raise ValueError("Production CPU smoke failed")
            report["production_cpu_profiles"][stage]=result
            print(json.dumps(result),flush=True)
            write_json(args.output,report)
    write_json(args.output,report)
    print(json.dumps({"passed":True,"report":args.output}),flush=True)


if __name__=="__main__":main()
