"""Independent raw-crop and original-codec replay on actual reduced MRI."""
from pathlib import Path
import argparse
import json
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from trjepa.codec import load_codec,ThreePhaseCodec
from trjepa.config import load_config
from trjepa.data import read_json,write_json,CacheReader,ensure_assets,PairDataset
from trjepa.train import train_contract,stage_budget
from trjepa.validation import selection_indices


def main():
    p=argparse.ArgumentParser()
    for key in ("data","codec","config","output"):p.add_argument("--"+key,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    root=Path(a.data);cfg=load_config(a.config);inv=read_json(root/"inventory.json")
    assets=ensure_assets(root,inv);train_contract(cfg,root,a.codec)
    patients={}
    for row in inv["visits"]:
        if patients.setdefault(row["patient_id"],row["split"])!=row["split"]:raise ValueError("Patient leakage")
    ds=PairDataset(root,"val");ids=selection_indices(ds.records,102,cfg["seed"]+77)
    if len({ds.records[i]["patient_id"] for i in ids})!=102:raise ValueError("Validation must cover all 102 patients")
    codec=ThreePhaseCodec(load_codec(a.codec)).eval()
    sys.path.insert(0,"/root/autodl-tmp/MAM/symm-fm-world-v2/src")
    from symm_world.codec import load_codec as original_codec
    reference=original_codec(a.codec)
    reader=CacheReader(root);sample=reader.read(inv["representation_train_files"][0])
    with torch.no_grad():
        expected=reference.decode(sample["latent"][None]);actual=codec.decode(sample["latent"][None])
    error=float((expected-actual).abs().max())
    if actual.shape!=(1,3,32,128,128) or error>1e-6:raise ValueError("Original VQ decoding differs")
    del reference,expected,actual
    with torch.no_grad():reencoded=codec.encode(sample["image"][None])[0]
    relative=float((reencoded-sample["latent"]).square().mean().sqrt()/sample["latent"].square().mean().sqrt())
    if relative>.02:raise ValueError("Re-encoded image differs materially from cached coordinates")
    sys.path.insert(0,"/root/autodl-tmp/MAM/MeWM-ISPY2")
    from mewm_ispy2.registered_three_phase_data import load_config as upstream_config,ThreePhaseCrops
    _,baseline=upstream_config("/root/autodl-tmp/MAM/MeWM-ISPY2/configs/registered_three_phase_roi32_v1.yaml")
    original=read_json("/root/autodl-tmp/MAM/MeWM-ISPY2/runs/registered_three_phase_roi32_v1/inventory.json")
    records=[original["visits"][i] for i in (0,len(original["visits"])//2,len(original["visits"])-1)]
    dataset=ThreePhaseCrops(baseline,original,records=records);comparisons=[]
    for i,record in enumerate(records):
        before=dataset[i];after=reader.read("views/"+record["visit_id"].replace(":","_")+".npz")
        errors={key:float((before[old].float()-after[new]).abs().max()) for key,old,new in
                (("image","image","image"),("roi","mask","roi"),("support","coverage","support"))}
        if any(errors.values()):raise ValueError("Raw MRI, ROI or coverage differs from original crop loader")
        comparisons.append(errors)
    report={"passed":True,"patients":len(patients),"patient_disjoint_splits":True,"validation_patients":102,
            "assets":assets,"decoder_max_error":error,"reencoded_latent_relative_rmse":relative,
            "original_crop_comparisons":comparisons,"budgets":{s:stage_budget(cfg,s) for s in "BCD"}}
    write_json(a.output,report);print(json.dumps(report),flush=True)


if __name__=="__main__":main()

