"""Fixed-seed validation of the deployment path, separate from FM regression."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import math
import numpy as np
import torch
from .data import collate
from .losses import balanced_loss


def selection_indices(records,count,seed=2026):
    """Round-robin patients, randomized deterministically; no prefix-only cohort."""
    groups=defaultdict(list)
    for i,r in enumerate(records):groups[r["patient_id"]].append(i)
    rng=np.random.default_rng(seed)
    patients=sorted(groups)
    rng.shuffle(patients)
    for values in groups.values():rng.shuffle(values)
    chosen=[]
    while len(chosen)<min(count,len(records)):
        for patient in patients:
            if groups[patient] and len(chosen)<count:chosen.append(groups[patient].pop())
    return chosen


def pair_seed(pair_id,seed):
    return (int.from_bytes(hashlib.sha256(str(pair_id).encode()).digest()[:4],"little")+seed)%2**31


@torch.no_grad()
def validate_generation(model,dataset,device,*,seed=2026,max_pairs=8,steps=32,solver="heun",samples=1,roi_weight=.25):
    """Equal-patient mean of expected-sample supported MAE + ROI MAE.

    Uses target support/ROI ONLY after source-only sampling, for scoring.
    No best-of-K selection, mean-image shortcut, or future condition is used.
    """
    groups=defaultdict(list); rows=[]
    for index in selection_indices(dataset.records,max_pairs,seed):
        source,target=collate([dataset[index]])
        source,target=source.to(device),target.to(device)
        p=dataset.records[index]
        result=model.sample(source,steps=steps,solver=solver,samples=samples,seed=pair_seed(p["pair_id"],seed))
        def score(image):
            return float(balanced_loss(image,target.image,target.support,roi=target.roi,
                                      roi_weight=roi_weight,kind="l1"))
        generated=float(np.mean([score(x) for x in result["images"]]))
        copy=score(source.image)
        codec=score(model.codec.decode(target.latent))
        row={"pair_id":p["pair_id"],"patient_id":p["patient_id"],"generation":generated,
             "copy_source":copy,"target_codec":codec,"seed":pair_seed(p["pair_id"],seed)}
        groups[p["patient_id"]].append(row); rows.append(row)
    if not rows:raise ValueError("Empty generation validation")
    summary={key:float(np.mean([np.mean([r[key] for r in rs]) for rs in groups.values()]))
             for key in ("generation","copy_source","target_codec")}
    return {"score":summary["generation"],**summary,"pairs":len(rows),"patients":len(groups),
            "steps":steps,"solver":solver,"samples":samples,"rows":rows,
            "metric":"patient_macro_expected_sample_supported_L1_plus_ROI_L1",
            "beats_copy_source":summary["generation"]<summary["copy_source"]}
