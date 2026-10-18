#!/usr/bin/env python3
"""Run one real-data prediction with migrated reference weights, without training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(17, 43), default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--landmark", type=int, choices=range(4), default=0)
    parser.add_argument("--output-dir", type=Path,
                        help="A new directory; defaults to a timestamp under private_outputs/")
    args = parser.parse_args()

    import numpy as np
    import torch
    from mri_vla_jepa.training import predict

    manifest = ROOT / "data/registered_roi32/manifest.json"
    checkpoint = ROOT / f"data/reference_runs/encoder64_seed{args.seed}/best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError("The selected reference checkpoint has not been migrated")
    patients = json.loads(manifest.read_text())["patients"]
    candidates = [patient for patient in patients
                  if patient["split"] == "val" and patient["visits"][args.landmark] is not None
                  and all((manifest.parent / visit["image"]).is_file()
                          for visit in patient["visits"][:args.landmark + 1]
                          if visit is not None)]
    if not candidates:
        raise RuntimeError("No validation patient with the requested local MRI prefix is available")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.output_dir.expanduser().resolve() if args.output_dir else
              ROOT / "private_outputs/reference_inference" / stamp)
    if not output.is_relative_to((ROOT / "private_outputs").resolve()):
        raise ValueError("Keep per-patient inference results under this project's private_outputs directory")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    report = predict(checkpoint, manifest, candidates[0]["patient_key"],
                     output / "prediction.json", landmark=args.landmark,
                     device=args.device, forecast_states=True)
    with np.load(report["forecast_states"], allow_pickle=False) as forecast:
        finite_states = bool(np.isfinite(forecast["state_prediction"]).all())
    passed = (math.isfinite(report["pcr_probability"])
              and 0 <= report["pcr_probability"] <= 1 and finite_states)
    summary = {
        "passed": passed, "engineering_validation_only": True,
        "reference_checkpoint_seed": args.seed, "device": args.device,
        "patients_processed": 1, "landmark": args.landmark,
        "forecast_shape": report["forecast_shape"], "finite_forecast_states": finite_states,
        "finite_probability": math.isfinite(report["pcr_probability"]),
        "optimizer_steps": 0, "formal_training_started": False,
        "private_output_directory": str(output),
    }
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
        summary["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Per-patient values stay in the ignored output directory.
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
