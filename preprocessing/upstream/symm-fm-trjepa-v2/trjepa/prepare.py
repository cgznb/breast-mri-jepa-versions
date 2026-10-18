"""Source-defined cache preparation and explicit legacy-manifest conversion.

No SSH access and no downloading of private patient data. Native NIfTI I/O is
optional (SimpleITK). Array-only preparation is fully supported and tested.
Resampling to a shared physical grid is NOT declared anatomical registration.
"""
from __future__ import annotations
from pathlib import Path
import copy
import json
import math
import numpy as np
import torch
from . import PHASES,UPSTREAM_COMMIT
from .data import SCHEMA,read_json,write_json,file_identity,atomic_npz,validate_inventory
from .conditions import check_condition
from .codec import load_codec,ThreePhaseCodec


def _path(base,name):
    p=Path(name)
    return p if p.is_absolute() else Path(base)/p


def reference_image(geometry):
    try:
        import SimpleITK as sitk
    except ImportError as e:
        raise ImportError("Native NIfTI preparation requires pip install '.[io]'") from e
    shape=geometry["shape_zyx"]
    spacing=np.asarray(geometry["spacing_xyz_mm"],dtype=float)
    direction=np.asarray(geometry["direction_lps"],dtype=float).reshape(3,3)
    if any(int(s)!=s or s<=0 for s in shape) or (spacing<=0).any() or not np.allclose(direction.T@direction,np.eye(3),atol=1e-4):
        raise ValueError("Invalid physical grid geometry")
    ref=sitk.Image(list(reversed(shape)),sitk.sitkFloat32)
    ref.SetSpacing(spacing.tolist());ref.SetOrigin(geometry["origin_lps_mm"]);ref.SetDirection(direction.ravel().tolist())
    return ref


def normalize_image(image, support, normalization):
    """Normalize raw intensities with an explicit background policy.

    Legacy symm-fm uses nonzero voxels only. Applying z-score to every acquired
    zero pixel changes its VQ input distribution. Never guess for a new dataset.
    """
    mean=float(normalization["mean"]); std=float(normalization["std"])
    if not np.isfinite(mean) or not np.isfinite(std) or std<=0:
        raise ValueError("Normalization requires finite mean and positive finite std")
    policy=normalization.get("background_policy")
    if policy not in ("preserve_zero", "all_supported"):
        raise ValueError("Raw normalization requires background_policy=preserve_zero or all_supported; "
                         "legacy symm-fm VQ requires preserve_zero. Do not guess from normalized arrays.")
    valid=np.asarray(support)>0
    if policy=="preserve_zero":valid=valid & (np.asarray(image)!=0)
    return np.where(valid,(image-normalization["mean"])/normalization["std"],0).astype(np.float32)


def load_visit(visit,geometry,base,normalization,transform_path=None):
    """Resample data into explicitly supplied SOURCE geometry, never target crop."""
    if visit.get("array_file"):
        if transform_path is not None or visit["crop_geometry"]!=geometry:
            raise ValueError("Array inputs must already occupy the exact declared source grid; use NIfTI to resample")
        path=_path(base,visit["array_file"])
        if path.suffix==".npy":
            raise ValueError("Use .npz with image/roi/support for explicit array geometry")
        with np.load(path,allow_pickle=False) as f:
            image=np.array(f["image"],dtype=np.float32)
            roi=np.array(f["roi"],dtype=np.float32) if "roi" in f else np.zeros((1,*image.shape[-3:]),np.float32)
            support=np.array(f["support"],dtype=np.float32) if "support" in f else np.ones_like(image)
        if not visit.get("already_normalized",False):
            image=normalize_image(image,support,normalization)
    else:
        import SimpleITK as sitk
        if len(visit.get("phase_paths",[]))!=3 or not visit.get("phase_order_verified",False):
            raise ValueError("Explicit ordered pre/first-post/metadata-late paths and phase_order_verified are required")
        ref=reference_image(geometry)
        transform=sitk.ReadTransform(str(_path(base,transform_path))) if transform_path else sitk.Transform(3,sitk.sitkIdentity)
        images=[];supports=[]
        for filename in visit["phase_paths"]:
            native=sitk.ReadImage(str(_path(base,filename)),sitk.sitkFloat32)
            if native.GetDimension()!=3:
                raise ValueError("Select three metadata-defined 3D phases from a 4D acquisition explicitly")
            support_native=sitk.Image(native.GetSize(),sitk.sitkUInt8)+1
            support_native.CopyInformation(native)
            res=sitk.Resample(native,ref,transform,sitk.sitkLinear,0.)
            valid=sitk.Resample(support_native,ref,transform,sitk.sitkNearestNeighbor,0)
            a=sitk.GetArrayFromImage(res).astype(np.float32)
            valid=sitk.GetArrayFromImage(valid).astype(np.float32)
            images.append(normalize_image(a,valid,normalization))
            supports.append(valid)
        image=np.stack(images).astype(np.float32);support=np.stack(supports).astype(np.float32)
        roi=np.zeros((1,*image.shape[-3:]),np.float32)
        if visit.get("tumor_mask_path"):
            mask=sitk.ReadImage(str(_path(base,visit["tumor_mask_path"])),sitk.sitkUInt8)
            mask=sitk.Resample(mask,ref,transform,sitk.sitkNearestNeighbor,0)
            roi=(sitk.GetArrayFromImage(mask)>0).astype(np.float32)[None]
    if image.shape!=(3,*geometry["shape_zyx"]) or roi.shape!=(1,*image.shape[-3:]) or support.shape!=image.shape:
        raise ValueError("Prepared shapes do not match the source grid")
    if not all(np.isfinite(a).all() for a in (image,roi,support)):
        raise ValueError("Nonfinite native/prepared image")
    return {"image":image,"roi":roi,"support":support}


def prepare(raw_path,output,*,codec_path=None,latent_mode="prequant",device="cpu",trusted_legacy=False):
    raw_path=Path(raw_path).resolve();base=raw_path.parent;output=Path(output)
    raw=read_json(raw_path)
    if tuple(raw["phase_order"])!=PHASES:
        raise ValueError("Three-phase ordering differs from the original repository")
    norm=raw["image_normalization"]
    if norm.get("fit_split")!="train" or norm["std"]<=0:
        raise ValueError("Use train-fitted image normalization, consistent with the frozen VQ checkpoint")
    visits={v["visit_id"]:v for v in raw["visits"]}
    if len(visits)!=len(raw["visits"]):
        raise ValueError("Duplicate visit IDs")
    assignments={}
    for v in visits.values():
        if assignments.setdefault(v["patient_id"],v["split"])!=v["split"]:
            raise ValueError("Patient crosses raw-image splits")
    model=None
    if codec_path:
        model=ThreePhaseCodec(load_codec(codec_path,trusted_legacy=trusted_legacy),latent_mode).to(device)
    contract={"raw_identity":file_identity(raw_path),"codec_identity":file_identity(codec_path) if codec_path else None,
              "latent_mode":latent_mode,"spatial_policy":"source_defined_grid"}
    output.mkdir(parents=True,exist_ok=True)
    if (output/"preparation_contract.json").exists() and read_json(output/"preparation_contract.json")!=contract:
        raise ValueError("Output already belongs to another raw manifest, codec or latent mode; use a new directory")
    write_json(output/"preparation_contract.json",contract)
    files={};canonical={};visit_rows=[];view_keys={}
    def cache(visit,geometry,transform=None):
        key=json.dumps({"visit_id":visit["visit_id"],"grid":geometry,"transform":transform},sort_keys=True)
        name=view_keys.setdefault(key,f"views/view_{len(view_keys):06d}.npz")
        if name not in files:
            arrays=load_visit(visit,geometry,base,norm,transform)
            if model is not None:
                x=torch.from_numpy(arrays["image"])[None].to(device)
                arrays["latent"]=model.encode(x)[0].float().cpu().numpy()
            atomic_npz(output/name,**arrays)
            files[name]=file_identity(output/name)
        return name
    for v in visits.values():
        name=cache(v,v["crop_geometry"])
        canonical[v["visit_id"]]=name
        visit_rows.append({"visit_id":v["visit_id"],"patient_id":v["patient_id"],"split":v["split"],"file":name})
    pairs=[]
    for p in raw["pairs"]:
        s,t=visits[p["source_visit_id"]],visits[p["target_visit_id"]]
        if s["patient_id"]!=t["patient_id"] or s["split"]!=t["split"]:
            raise ValueError("Pair endpoints differ in patient/split")
        check_condition(p["conditions"])
        if not s["day"]<t["day"]:
            raise ValueError("Pair must be longitudinally forward")
        transform=p.get("source_to_target_resampling_transform")
        if transform and p.get("transform_convention")!="output_source_to_input_target":
            raise ValueError("SimpleITK Resample expects output-source -> input-target coordinates")
        verified=bool(p.get("spatial_correspondence_verified",False))
        if verified and not p.get("registration_qc_note"):
            raise ValueError("Verified spatial correspondence requires an explicit QC record")
        target_file=cache(t,s["crop_geometry"],transform)
        pairs.append({"pair_id":p["pair_id"],"patient_id":s["patient_id"],"split":s["split"],
            "source_day":s["day"],"target_day":t["day"],"conditions":p["conditions"],"labels":p.get("labels",{}),
            "source_file":canonical[s["visit_id"]],"target_file":target_file,
            "spatial_correspondence_verified":verified,"registration_qc_note":p.get("registration_qc_note"),
            "source_geometry":s["crop_geometry"],"transition":p.get("transition",f"{s['visit_id']}->{t['visit_id']}")})
    inv={"schema":SCHEMA if model else "trjepa_images_only_v1","phase_order":list(PHASES),
        "spatial_policy":"source_defined_grid","latent_mode":latent_mode,"codec_identity":contract["codec_identity"],
        "image_normalization":norm,"source_manifest_identity":contract["raw_identity"],"pairs":pairs,"visits":visit_rows,
        "representation_train_files":[r["file"] for r in visit_rows if r["split"]=="train"],"cache_identities":files}
    if model:validate_inventory(inv)
    write_json(output/"inventory.json",inv)
    return {"cache":str(output),"views":len(files),"pairs":len(pairs),"has_latents":model is not None}


def encode_images_cache(root,codec_path,*,latent_mode="prequant",device="cpu",trusted_legacy=False):
    root=Path(root);inv=read_json(root/"inventory.json")
    if inv["schema"]!="trjepa_images_only_v1":
        raise ValueError("Only an image-only cache may be upgraded in place; changing latents needs a new cache")
    model=ThreePhaseCodec(load_codec(codec_path,trusted_legacy=trusted_legacy),latent_mode).to(device)
    identities={}
    for name in inv.get("cache_identities",inv.get("cache_sha256",{})):
        with np.load(root/name,allow_pickle=False) as f:
            arrays={k:np.array(f[k],dtype=np.float32) for k in ("image","roi","support")}
        x=torch.from_numpy(arrays["image"])[None].to(device)
        arrays["latent"]=model.encode(x)[0].float().cpu().numpy()
        atomic_npz(root/name,**arrays)
        identities[name]=file_identity(root/name)
    inv.pop("cache_sha256",None);inv.pop("codec_sha256",None)
    inv.update(schema=SCHEMA,codec_identity=file_identity(codec_path),latent_mode=latent_mode,cache_identities=identities)
    validate_inventory(inv);write_json(root/"inventory.json",inv)
    return {"encoded_views":len(identities),"codec_identity":inv["codec_identity"]}


def convert_legacy(inventory_path,assets_path,output):
    """Convert ACTUAL original admitted inventory; never relabel old latent arrays.

    Assets supplies local ordered NIfTI phases, actual day and explicit structured
    treatment mapping per pair, because old condition dictionaries are not assumed
    to match the report. Patient split and baseline_clinical are preserved.
    Source visit's complete geometry is used, rather than target-centered views.
    """
    legacy=read_json(inventory_path);assets=read_json(assets_path)
    if legacy.get("schema")!="three_phase_symmflow_all_pairs_v1":
        raise ValueError("Expected the repository's admitted_inventory.json")
    if tuple(legacy["phase_order"])!=PHASES:
        raise ValueError("Legacy phase ordering differs")
    visits=[]
    for v in legacy["visits"]:
        local=assets["visits"][v["visit_id"]]
        visits.append({"visit_id":v["visit_id"],"patient_id":v["canonical_patient_id"],"split":v["split"],
            "day":local["day"],"crop_geometry":v["crop_geometry"],**{k:local[k] for k in
            ("phase_paths","array_file","already_normalized","tumor_mask_path") if k in local},
            "phase_order_verified":local.get("phase_order_verified",False)})
    index={v["visit_id"]:v for v in visits}
    pairs=[]
    for p in legacy["pairs"]:
        extra=assets.get("pairs",{}).get(p["pair_id"],{})
        if "treatment" not in extra:
            raise ValueError(f"Explicit treatment normalization required for pair {p['pair_id']}; old value was {p.get('treatment')!r}")
        source=index[p["earlier_visit_id"]]
        condition={"clinical":extra.get("clinical",p.get("baseline_clinical") or {}),"treatment":extra["treatment"],
            "source_day":source["day"],"delta_days":None if p.get("interval_missing",False) else p["delta_days"],
            "horizon_kind":"retrospective_observed"}
        check_condition(condition)
        row={"pair_id":p["pair_id"],"source_visit_id":p["earlier_visit_id"],"target_visit_id":p["later_visit_id"],
             "conditions":condition,"transition":f"{p['earlier_stage']}->{p['later_stage']}",
             "labels":extra.get("labels",{}),"spatial_correspondence_verified":extra.get("spatial_correspondence_verified",False)}
        for k in ("registration_qc_note","source_to_target_resampling_transform","transform_convention"):
            if k in extra:row[k]=extra[k]
        pairs.append(row)
    norm=dict(legacy["image_normalization"]);norm["fit_split"]="train"
    norm["background_policy"]="preserve_zero"
    raw={"phase_order":list(PHASES),"image_normalization":norm,"visits":visits,"pairs":pairs,
         "legacy_inventory_identity":file_identity(inventory_path)}
    write_json(output,raw)
    return {"raw_manifest":str(output),"patients":len({v['patient_id'] for v in visits}),"pairs":len(pairs),
            "old_latents_reused":False,"reason":"raw-image semantics and source-grid targets require new caches"}
