"""Deterministic, resumable training sampling; optional patient balancing."""
from collections import defaultdict
import torch


def training_indices(dataset,step,micro,batch,seed,mode="pair_uniform"):
    g=torch.Generator().manual_seed(seed+1000003*step+1009*micro)
    if mode=="pair_uniform" or not (hasattr(dataset,"records") or hasattr(dataset,"patient_ids")):
        # Without replacement within a microbatch when possible, important for
        # cross-image JEPA variance. Accumulation is still independent draws.
        if batch<=len(dataset):return torch.randperm(len(dataset),generator=g)[:batch].tolist()
        return torch.randint(len(dataset),(batch,),generator=g).tolist()
    if mode!="patient_uniform":raise ValueError("Unknown training sampling policy")
    groups=defaultdict(list)
    ids=dataset.patient_ids if hasattr(dataset,"patient_ids") else [r["patient_id"] for r in dataset.records]
    for i,patient in enumerate(ids):groups[patient].append(i)
    patients=sorted(groups)
    picked=torch.randperm(len(patients),generator=g)[:batch].tolist() if batch<=len(patients) else torch.randint(len(patients),(batch,),generator=g).tolist()
    return [groups[patients[p]][int(torch.randint(len(groups[patients[p]]),(1,),generator=g))] for p in picked]


def augment_representation(image,roi,cfg):
    """Common spatial transforms across all DCE phases and ROI; no phase swap.

    Not applied to cached VQ latents: flipping a convolutional latent is not
    assumed equivalent to re-encoding the flipped raw image.
    """
    image,roi=image.clone(),roi.clone()
    for b in range(image.shape[0]):
        for axis in cfg.get("flip_axes",[]):
            if axis not in (0,1,2):raise ValueError("flip_axes must be ZYX spatial axes")
            if torch.rand((),device=image.device)<.5:
                image[b]=image[b].flip(axis+1);roi[b]=roi[b].flip(axis+1)
        # One shared scale preserves phase ordering and relative enhancement.
        strength=float(cfg.get("intensity_scale",0.))
        if strength:
            if not 0<=strength<1:raise ValueError("intensity_scale must lie in [0,1)")
            image[b]=image[b]*(1+strength*(2*torch.rand((),device=image.device)-1))
    return image,roi
