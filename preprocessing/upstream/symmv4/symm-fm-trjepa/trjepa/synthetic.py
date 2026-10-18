"""Deterministic SYNTHETIC data for executable verification; not medical evidence."""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from . import PHASES
from .codec import MRILevelVQGAN,VQGANConfig,ThreePhaseCodec
from .data import SCHEMA,atomic_npz,write_json,file_identity
from .train import atomic_save


def make_synthetic(output,cfg,*,image_only=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(cfg.get("seed",2026));torch.set_num_threads(1)
    codec=MRILevelVQGAN(VQGANConfig(**cfg["codec"]))
    checkpoint=output/"synthetic_codec.pt"
    atomic_save(checkpoint,{"schema":"trjepa_codec_v1","model_config":asdict(codec.config),"codec_state":codec.state_dict(),
                            "synthetic_untrained":True})
    shared=ThreePhaseCodec(codec,cfg["latent_mode"])
    grid=torch.stack(torch.meshgrid(*(torch.linspace(-1,1,8) for _ in range(3)),indexing="ij"))
    lesion=(grid.square().sum(0)<.5).float()
    rows=[];visits=[];checks={};canonical=[]
    geom={"shape_zyx":[8,8,8],"spacing_xyz_mm":[1.,1.,1.],"origin_lps_mm":[0.,0.,0.],
          "direction_lps":[1.,0.,0.,0.,1.,0.,0.,0.,1.]}
    for i in range(6):
        split="train" if i<4 else "val"
        files=[]
        source=torch.randn(3,8,8,8)*.2+lesion[None]*(.8+i*.03)
        target=source*.9-lesion[None]*(.2 if i%2 else .4)
        for t,x in enumerate((source,target)):
            name=f"views/synthetic_{i}_{t}.npz"
            arrays={"image":x.numpy(),"roi":lesion[None].numpy(),"support":np.ones((3,8,8,8),np.float32)}
            if not image_only:arrays["latent"]=shared.encode(x[None])[0].numpy()
            atomic_npz(output/name,**arrays);checks[name]=file_identity(output/name);files.append(name)
            visits.append({"visit_id":f"s{i}_T{t}","patient_id":f"synthetic_{i}","split":split,"file":name})
            if split=="train":canonical.append(name)
        rows.append({"pair_id":f"synthetic_pair_{i}","patient_id":f"synthetic_{i}","split":split,"source_day":0.,"target_day":90.,
            "source_file":files[0],"target_file":files[1],"source_geometry":geom,"transition":"T0->T1",
            "spatial_correspondence_verified":True,"registration_qc_note":"synthetic arrays generated on the identical grid",
            "conditions":{"source_day":0.,"delta_days":90.,"horizon_kind":"query",
                "clinical":{"age":40+i,"sex":"female","er":"positive","subtype":"synthetic"},
                "treatment":{"regimen":"synthetic_A" if i%2 else "synthetic_B","known_at_source":True,
                    "drugs":[{"name":"synthetic_drug_1","dose":1.},{"name":"synthetic_drug_2","dose":2.}] }},
            "labels":{"pcr":i%2,"response_class":i%3,
                      "survival":{"duration_days":400+i*20.,"event":i%2,"landmark_day":0.}}})
    inv={"schema":"trjepa_images_only_v1" if image_only else SCHEMA,"phase_order":list(PHASES),
         "spatial_policy":"source_defined_grid","latent_mode":cfg["latent_mode"],"codec_identity":None if image_only else file_identity(checkpoint),
         "image_normalization":{"mean":0.,"std":1.,"fit_split":"train"},"pairs":rows,"visits":visits,
         "cache_identities":checks,"representation_train_files":canonical,"synthetic":True}
    write_json(output/"inventory.json",inv)
    write_json(output/"example_source_only.json",{k:rows[-1][k] for k in ("source_file","conditions","pair_id")})
    return {"root":str(output),"codec":str(checkpoint),"synthetic":True}
