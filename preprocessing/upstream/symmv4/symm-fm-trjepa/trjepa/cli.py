from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from .config import load_config
from .data import read_json,write_json,atomic_npz,source_only


def main():
    p=argparse.ArgumentParser(description="Three-phase TR-JEPA: explicit representation, dynamics and generative stages")
    sub=p.add_subparsers(dest="command",required=True)
    s=sub.add_parser("synthetic");s.add_argument("--config",required=True);s.add_argument("--output",required=True);s.add_argument("--image-only",action="store_true")
    s=sub.add_parser("prepare");s.add_argument("--raw",required=True);s.add_argument("--output",required=True);s.add_argument("--codec");s.add_argument("--latent-mode",default="prequant",choices=["prequant","quantized"]);s.add_argument("--device",default="cpu");s.add_argument("--trusted-legacy",action="store_true")
    s=sub.add_parser("encode-cache");s.add_argument("--root",required=True);s.add_argument("--codec",required=True);s.add_argument("--latent-mode",default="prequant",choices=["prequant","quantized"]);s.add_argument("--device",default="cpu");s.add_argument("--trusted-legacy",action="store_true")
    s=sub.add_parser("convert-legacy");s.add_argument("--inventory",required=True);s.add_argument("--assets",required=True);s.add_argument("--output",required=True)
    for name in ("train","train-codec"):
        s=sub.add_parser(name);s.add_argument("--config",required=True);s.add_argument("--root",required=True);s.add_argument("--output",required=True)
        s.add_argument("--device",default="cpu");s.add_argument("--init");s.add_argument("--resume");s.add_argument("--stop-after",type=int)
        if name=="train":s.add_argument("--stage",choices=list("BCDE"),required=True);s.add_argument("--codec",required=True);s.add_argument("--trusted-legacy",action="store_true")
    for name in ("sample","evaluate","export-features"):
        s=sub.add_parser(name);s.add_argument("--checkpoint",required=True);s.add_argument("--codec",required=True);s.add_argument("--root",required=True);s.add_argument("--output",required=True);s.add_argument("--device",default="cpu");s.add_argument("--trusted-legacy",action="store_true")
        if name=="sample":s.add_argument("--case",required=True)
        if name!="export-features":
            s.add_argument("--steps",type=int,default=20);s.add_argument("--samples",type=int,default=1);s.add_argument("--seed",type=int,default=2026);s.add_argument("--solver",choices=["euler","heun"],default="heun")
        if name=="evaluate":s.add_argument("--split",default="val",choices=["train","val","test"]);s.add_argument("--data-range",type=float,default=6.);s.add_argument("--limit",type=int)
        if name=="sample":s.add_argument("--guidance",type=float,default=1.)
    s=sub.add_parser("probe");s.add_argument("--features-file",required=True);s.add_argument("--features",default="future");s.add_argument("--output",required=True);s.add_argument("--epochs",type=int,default=300)
    s=sub.add_parser("inspect");s.add_argument("--config",required=True);s.add_argument("--root",required=True);s.add_argument("--codec",required=True)
    a=p.parse_args()
    if a.command=="synthetic":
        from .synthetic import make_synthetic
        result=make_synthetic(a.output,load_config(a.config),image_only=a.image_only)
    elif a.command=="prepare":
        from .prepare import prepare
        result=prepare(a.raw,a.output,codec_path=a.codec,latent_mode=a.latent_mode,device=a.device,trusted_legacy=a.trusted_legacy)
    elif a.command=="encode-cache":
        from .prepare import encode_images_cache
        result=encode_images_cache(a.root,a.codec,latent_mode=a.latent_mode,device=a.device,trusted_legacy=a.trusted_legacy)
    elif a.command=="convert-legacy":
        from .prepare import convert_legacy
        result=convert_legacy(a.inventory,a.assets,a.output)
    elif a.command in ("train","train-codec"):
        from .train import train_stage,train_codec
        cfg=load_config(a.config)
        if a.command=="train":
            result=train_stage(cfg,a.root,a.codec,a.output,a.stage,init=a.init,resume=a.resume,device=a.device,
                               trusted_legacy=a.trusted_legacy,stop_after=a.stop_after)
        else:result=train_codec(cfg,a.root,a.output,init=a.init,resume=a.resume,device=a.device,stop_after=a.stop_after)
    elif a.command in ("sample","evaluate","export-features"):
        from .evaluate import load_world,evaluate,export_features
        model,saved=load_world(a.checkpoint,a.codec,a.device,a.trusted_legacy)
        if a.command=="sample":
            if saved["stage"]=="C":raise ValueError("Stage C has no trained renderer; use D/E for image sampling")
            src,audit=source_only(a.root,read_json(a.case))
            value=model.sample(src.to(a.device),samples=a.samples,steps=a.steps,seed=a.seed,solver=a.solver,guidance=a.guidance)
            atomic_npz(a.output,**{k:v.detach().cpu().float().numpy() for k,v in value.items()})
            write_json(str(a.output)+".audit.json",audit)
            result={"file":a.output,"audit":audit}
        elif a.command=="evaluate":
            if saved["stage"]=="C":raise ValueError("Use D/E for image evaluation")
            report=evaluate(model,a.root,a.output,split=a.split,steps=a.steps,samples=a.samples,seed=a.seed,solver=a.solver,data_range=a.data_range,limit=a.limit)
            result={"report":str(Path(a.output)/"report.json"),"patients":report["patient_count"]}
        else:result=export_features(model,a.root,a.output)
    elif a.command=="probe":
        from .evaluate import linear_probe
        result=linear_probe(a.features_file,a.output,features=a.features,epochs=a.epochs)
    elif a.command=="inspect":
        from .train import train_contract,make_world
        cfg=load_config(a.config);contract=train_contract(cfg,a.root,a.codec)
        model=make_world(cfg,contract,a.codec,torch.device("cpu"))
        result={"modules":{n:sum(p.numel() for p in m.parameters()) for n,m in model.named_children()},
                "trainable_by_stage":{}}
        for stage in "CDE":
            model.configure_stage(stage);result["trainable_by_stage"][stage]=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=="__main__":main()
