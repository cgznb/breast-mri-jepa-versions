from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from .config import load_config
from .data import Store, collate_prefix, canonical_key
from .engine import (train, write_json, write_csv, load_for_inference, evaluate, load_checkpoint)
from .metrics import paired_bootstrap


def parser():
    p = argparse.ArgumentParser(description="Stage-conditioned MRI latent forecasting and pCR research runner")
    sub = p.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train", help="Train one explicitly configured phase")
    tr.add_argument("--config", required=True)
    tr.add_argument("--manifest", required=True)
    tr.add_argument("--output", required=True)
    tr.add_argument("--init", help="Trusted A/best.pt or B/best.pt warm start")
    tr.add_argument("--resume", action="store_true", help="Resume this output/last.pt, at an epoch boundary")
    tr.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    tr.add_argument("--stop-after-epoch", type=int, help="Testing interruption; does not change configured schedule")
    sm = sub.add_parser("smoke", help="CPU A->B->C synthetic integration test")
    sm.add_argument("--output", required=True)
    pf = sub.add_parser("preflight", help="Check declarations and paths; optionally inspect every pixel array")
    pf.add_argument("--manifest", required=True)
    pf.add_argument("--image-cache")
    pf.add_argument("--verify-pixels", action="store_true")
    pf.add_argument("--output", required=True)
    for name in ("evaluate", "predict", "diagnose"):
        e = sub.add_parser(name)
        e.add_argument("--checkpoint", required=True, help="Only load a trusted v2 checkpoint")
        e.add_argument("--manifest", required=True)
        e.add_argument("--device", default="cpu")
        e.add_argument("--image-cache")
        e.add_argument("--allow-synthetic", action="store_true")
        e.add_argument("--output", required=True)
        if name == "evaluate":
            e.add_argument("--split", choices=("val", "test", "train"), default="test")
            e.add_argument("--save-predictions", action="store_true", help="Write private patient-level CSV next to metrics")
            e.add_argument("--no-world", action="store_true")
        elif name == "predict":
            e.add_argument("--patient-key", required=True)
            e.add_argument("--landmark", type=int, choices=range(4), default=0)
            e.add_argument("--request", help="Comma-separated future stages, e.g. 1,2,3; output selection only")
            e.add_argument("--states-out", help="Optional NPZ of predicted latent states, NOT future MRI")
        else:
            e.add_argument("--patients", type=int, default=32)
    pa = sub.add_parser("paired", help="Patient-paired metric differences, B minus A")
    pa.add_argument("--a", required=True, help="Prediction CSV A")
    pa.add_argument("--b", required=True, help="Prediction CSV B; may be the same file for landmark comparisons")
    pa.add_argument("--stage-a", type=int, default=0, choices=range(4))
    pa.add_argument("--stage-b", type=int, default=0, choices=range(4))
    pa.add_argument("--branch-a", choices=("fused", "direct", "forecast"), default="fused")
    pa.add_argument("--branch-b", choices=("fused", "direct", "forecast"), default="fused")
    pa.add_argument("--repetitions", type=int, default=2000)
    pa.add_argument("--output", required=True)
    tb = sub.add_parser("tabular", help="T0 clinical logistic-regression baseline; no MRI file reads")
    tb.add_argument("--manifest", required=True)
    tb.add_argument("--include-arm", action="store_true")
    tb.add_argument("--output", required=True)
    tb.add_argument("--evaluate-test", action="store_true", help="Explicit final test evaluation; default uses train/val only")
    lp = sub.add_parser("linear-probe", help="Frozen MRI encoder + logistic probe at one real visit")
    lp.add_argument("--checkpoint", required=True)
    lp.add_argument("--manifest", required=True)
    lp.add_argument("--stage", type=int, choices=range(4), default=0)
    lp.add_argument("--device", default="cpu")
    lp.add_argument("--image-cache")
    lp.add_argument("--output", required=True)
    lp.add_argument("--evaluate-test", action="store_true", help="Explicit final test evaluation, not development model selection")
    info = sub.add_parser("checkpoint-info")
    info.add_argument("--checkpoint", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "train":
        cfg = load_config(args.config, args.set)
        result = train(cfg, args.manifest, args.output, init=args.init, resume=args.resume,
                       stop_after_epoch=args.stop_after_epoch)
        print(json.dumps({"output": result["output"], "best_epoch": result["best_epoch"]}, ensure_ascii=False))
    elif args.command == "smoke":
        from .synthetic import smoke
        print(json.dumps(smoke(args.output), ensure_ascii=False, indent=2))
    elif args.command == "preflight":
        store = Store(args.manifest, image_cache=args.image_cache)
        result = store.audit(args.verify_pixels)
        result["data_fingerprint"] = store.fingerprint()
        write_json(args.output, result)
        print(json.dumps(result["patients"], ensure_ascii=False))
    elif args.command == "paired":
        with open(args.a, newline="", encoding="utf-8") as f:
            a = list(csv.DictReader(f))
        with open(args.b, newline="", encoding="utf-8") as f:
            b = list(csv.DictReader(f))
        if args.repetitions < 1:
            raise ValueError("repetitions must be positive")
        write_json(args.output, paired_bootstrap(a, b, stage_a=args.stage_a, stage_b=args.stage_b,
                   branch_a=args.branch_a, branch_b=args.branch_b, repetitions=args.repetitions))
    elif args.command == "tabular":
        from .baselines import tabular
        tabular(args.manifest, output=args.output, include_arm=args.include_arm, evaluate_test=args.evaluate_test)
    elif args.command == "linear-probe":
        from .baselines import linear_probe
        linear_probe(args.checkpoint, args.manifest, stage=args.stage, output=args.output,
                     device=args.device, image_cache=args.image_cache, evaluate_test=args.evaluate_test)
    elif args.command == "checkpoint-info":
        s = load_checkpoint(args.checkpoint)
        print(json.dumps({k: s[k] for k in ("schema", "epoch", "best_score", "config", "target_coordinate_origin", "init")},
                         ensure_ascii=False, indent=2))
    else:
        model, store, cfg, state = load_for_inference(args.checkpoint, args.manifest, device=args.device,
                         image_cache=args.image_cache, allow_synthetic=args.allow_synthetic)
        if args.command == "evaluate":
            result, rows, wr = evaluate(model, store, cfg, args.split, world=False if args.no_world else None)
            result.update(split=args.split, checkpoint_epoch=state["epoch"],
                          interpretation="Development evaluation" if args.split != "test" else "Declared test split; independence needs provenance")
            write_json(args.output, result)
            if args.save_predictions:
                stem = Path(args.output).with_suffix("")
                if cfg.train.phase in {"C", "joint"}:
                    write_csv(str(stem)+"_predictions.csv", rows)
                write_csv(str(stem)+"_world.csv", wr)
        elif args.command == "diagnose":
            from .diagnostics import diagnose
            write_json(args.output, diagnose(model, store, cfg, patients=args.patients))
        elif args.command == "predict":
            if cfg.train.phase not in {"C", "joint"}:
                raise ValueError("pCR prediction requires a task-trained C or joint checkpoint")
            if cfg.train.profile in {"t0", "single_t0"} and args.landmark != 0:
                raise ValueError("A fixed-T0 checkpoint has not been trained for later observed prefixes")
            if cfg.train.profile == "single_t3" and args.landmark != 3:
                raise ValueError("The T3 reference checkpoint only supports T3")
            matches = [i for i, p in enumerate(store.patients)
                       if canonical_key(p["patient_key"]) == canonical_key(args.patient_key)]
            if len(matches) != 1:
                raise ValueError("Patient key does not identify exactly one manifest record")
            batch = collate_prefix([store.sample(matches[0], args.landmark, future=False, labels=False,
                                    single=cfg.train.profile.startswith("single_"), load_images=cfg.model.use_mri)])
            batch = batch.to(args.device)
            inp = batch.inputs
            if args.request is not None:
                requested = [int(v) for v in args.request.split(",") if v.strip()]
                if any(j <= args.landmark or j > 3 for j in requested):
                    raise ValueError("Requested stages must be strictly after landmark and <=3")
                mask = torch.zeros_like(inp.query_mask)
                mask[:, requested] = True
                inp = replace(inp, query_mask=mask)
            if args.states_out and (not cfg.model.use_mri or not model.world_trained):
                raise ValueError("This checkpoint has no trained world predictor; latent forecasting is unavailable")
            with torch.no_grad():
                out = model(inp, request_states=bool(args.states_out))
            result = {"landmark": args.landmark, "pcr_probability": float(out["logit"].float().sigmoid()[0]),
                      "direct_probability": float(out["direct_logit"].float().sigmoid()[0]),
                      "forecast_probability": float(out["forecast_logit"].float().sigmoid()[0])
                          if out["forecast_logit"] is not None else None,
                      "future_ground_truth_read": False, "readout_mode": cfg.model.readout, "output": "final pCR conditional prediction; not a causal treatment effect",
                      "target_coordinate_origin": state["target_coordinate_origin"]}
            if args.states_out:
                path = Path(args.states_out)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as f:
                    np.savez_compressed(f, states=out["requested_states"].float().cpu().numpy(),
                                        query_mask=inp.query_mask.cpu().numpy())
                path.chmod(0o600)
                result["latent_states_file"] = str(path)
            write_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
