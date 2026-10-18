from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mewm_ispy2.backend import load_transition_records, sha256_file
from mewm_ispy2.config import load_experiment_config
from mewm_ispy2.paper_diffusion import DDIM_ETA0_SAMPLER
from mewm_ispy2.contracts import ANCESTRAL_DDPM_SAMPLER, X0_PREDICTION_TYPE
from mewm_ispy2.registered_x0_checkpoint import peek_registered_x0_checkpoint_identity
from mewm_ispy2.vqgan import MRI_VQGAN_DEFAULTS
from mewm_ispy2.vqgan_visualization import orthogonal_planes, tumor_centroid
from mewm_ispy2.workflows import (
    _visit_loader,
    load_registered_paper_diffusion_for_inference,
)


PLANE_NAMES = ("axial", "coronal", "sagittal")
IMAGE_WINDOW = (-2.0, 3.0)
ERROR_MAX = 2.0


def _select_cases(
    records: list[Any],
    visit_loader: Any,
    transition_id: str | None = None,
    validation_indices: list[int] | None = None,
) -> list[tuple[Any, Any, int]]:
    validation = sorted(
        (record for record in records if record.fold == "val"),
        key=lambda record: record.transition_id,
    )
    if transition_id is not None and validation_indices is not None:
        raise ValueError("transition_id and validation_indices are mutually exclusive")
    if validation_indices is not None:
        if not validation_indices or len(set(validation_indices)) != len(validation_indices):
            raise ValueError("validation_indices must be nonempty and unique")
        selected = []
        for index in validation_indices:
            if not 0 <= index < len(validation):
                raise ValueError(
                    f"validation index {index} is outside [0, {len(validation) - 1}]"
                )
            record = validation[index]
            source = visit_loader(record.source_visit_id)
            if not bool((source.mask > 0).any()):
                raise ValueError(
                    f"validation transition has an empty source mask: {record.transition_id}"
                )
            selected.append((record, source, index))
        return selected
    if transition_id is not None:
        matches = [record for record in validation if record.transition_id == transition_id]
        if len(matches) != 1:
            raise ValueError(
                f"expected one validation transition {transition_id!r}, found {len(matches)}"
            )
        source = visit_loader(matches[0].source_visit_id)
        if not bool((source.mask > 0).any()):
            raise ValueError(f"validation transition has an empty source mask: {transition_id}")
        return [(matches[0], source, validation.index(matches[0]))]
    selected = []
    for stage_id in (1, 2, 3):
        for record in validation:
            if record.stage_id != stage_id:
                continue
            source = visit_loader(record.source_visit_id)
            if bool((source.mask > 0).any()):
                selected.append((record, source, validation.index(record)))
                break
        else:
            raise ValueError(f"validation split has no eligible stage-{stage_id} case")
    return selected


def _load_model(config: Any, loaded: Any, vqgan_path: Path, checkpoint: Path):
    vqgan_sha256 = sha256_file(vqgan_path)
    physical_batch_size = (
        peek_registered_x0_checkpoint_identity(checkpoint).physical_batch_size
        if config.diffusion.prediction_type == X0_PREDICTION_TYPE
        else 1
    )
    model, identity, global_step = load_registered_paper_diffusion_for_inference(
        runtime=config.diffusion,
        loaded=loaded,
        vqgan_checkpoint_path=vqgan_path,
        diffusion_checkpoint_path=checkpoint,
        physical_batch_size=physical_batch_size,
    )
    return model, identity, global_step, vqgan_sha256


def _plane(volume: torch.Tensor, centroid: tuple[int, int, int], name: str) -> np.ndarray:
    return (
        orthogonal_planes(volume, centroid)[name]
        .detach()
        .float()
        .cpu()
        .numpy()
    )


def _render_case(case: dict[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
    columns = ("Source", "Generated", "Target", "Absolute error")
    for row, plane_name in enumerate(PLANE_NAMES):
        source = _plane(case["source"], case["centroid"], plane_name)
        generated = _plane(case["generated"], case["centroid"], plane_name)
        target = _plane(case["target"], case["centroid"], plane_name)
        mask = _plane(case["mask"], case["centroid"], plane_name)
        error = np.abs(generated - target)
        arrays = (source, generated, target, error)
        for column, (axis, values) in enumerate(zip(axes[row], arrays, strict=True)):
            if column < 3:
                axis.imshow(
                    values,
                    cmap="gray",
                    vmin=IMAGE_WINDOW[0],
                    vmax=IMAGE_WINDOW[1],
                    origin="lower",
                )
                if np.any(mask > 0):
                    axis.contour(
                        mask,
                        levels=[0.5],
                        colors=["#00FFFF"],
                        linewidths=0.8,
                        origin="lower",
                    )
            else:
                axis.imshow(
                    values,
                    cmap="magma",
                    vmin=0.0,
                    vmax=ERROR_MAX,
                    origin="lower",
                )
            if row == 0:
                axis.set_title(columns[column])
            axis.axis("off")
        axes[row, 0].text(
            -0.08,
            0.5,
            plane_name.capitalize(),
            rotation=90,
            va="center",
            ha="center",
            transform=axes[row, 0].transAxes,
        )
    record = case["record"]
    figure.suptitle(
        f"{case['sampler_label']} | Stage {record.stage_id} | "
        f"{record.transition_type} | seed {case['seed']} | "
        f"MAE {case['mae']:.4f} | {case['sampling_seconds']:.1f} s",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _render_overview(cases: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(
        len(cases),
        4,
        figsize=(15, 2.65 * len(cases) + 0.35),
        constrained_layout=True,
        squeeze=False,
    )
    columns = ("Source", "Generated", "Target", "Absolute error")
    for row, case in enumerate(cases):
        source = _plane(case["source"], case["centroid"], "axial")
        generated = _plane(case["generated"], case["centroid"], "axial")
        target = _plane(case["target"], case["centroid"], "axial")
        mask = _plane(case["mask"], case["centroid"], "axial")
        arrays = (source, generated, target, np.abs(generated - target))
        for column, (axis, values) in enumerate(zip(axes[row], arrays, strict=True)):
            if column < 3:
                axis.imshow(
                    values,
                    cmap="gray",
                    vmin=IMAGE_WINDOW[0],
                    vmax=IMAGE_WINDOW[1],
                    origin="lower",
                )
                if np.any(mask > 0):
                    axis.contour(
                        mask,
                        levels=[0.5],
                        colors=["#00FFFF"],
                        linewidths=0.8,
                        origin="lower",
                    )
            else:
                axis.imshow(
                    values,
                    cmap="magma",
                    vmin=0.0,
                    vmax=ERROR_MAX,
                    origin="lower",
                )
            if row == 0:
                axis.set_title(columns[column])
            axis.axis("off")
        record = case["record"]
        axes[row, 0].text(
            0.02,
            0.98,
            f"Stage {record.stage_id} | {record.transition_type} | "
            f"seed {case['seed']} | MAE {case['mae']:.4f}",
            transform=axes[row, 0].transAxes,
            va="top",
            ha="left",
            color="white",
            fontsize=9,
            fontweight="bold",
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 3, "edgecolor": "none"},
        )
    figure.suptitle(
        f"Registered-large Diffusion | {cases[0]['sampler_label']}",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize registered-large diffusion")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vqgan-checkpoint", type=Path, required=True)
    parser.add_argument("--diffusion-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--sampler",
        choices=(ANCESTRAL_DDPM_SAMPLER, DDIM_ETA0_SAMPLER),
        default=ANCESTRAL_DDPM_SAMPLER,
    )
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--transition-id")
    parser.add_argument("--validation-indices", nargs="+", type=int)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--cpu-bf16", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_experiment_config(args.config)
    loaded = load_transition_records(
        config.data.bundle_json,
        config.data.phase_manifest_csv,
        backend=config.data.backend,
    )
    visit_loader = _visit_loader(config, loaded)
    selected = _select_cases(
        list(loaded.records),
        visit_loader,
        transition_id=args.transition_id,
        validation_indices=args.validation_indices,
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    model, identity, global_step, vqgan_sha256 = _load_model(
        config,
        loaded,
        args.vqgan_checkpoint.resolve(),
        args.diffusion_checkpoint.resolve(),
    )
    model = model.to(device).eval()
    sampling_steps = (
        model.config.timesteps
        if args.sampling_steps is None
        else args.sampling_steps
    )
    model._resolve_sampling_override(args.sampler, sampling_steps)
    sampler_label = (
        f"DDPM-{sampling_steps}"
        if args.sampler == ANCESTRAL_DDPM_SAMPLER
        else f"DDIM-{sampling_steps} eta=0"
    )

    cases: list[dict[str, Any]] = []
    for ordinal, (record, source, validation_index) in enumerate(selected, start=1):
        seed = (
            args.seed + validation_index
            if args.validation_indices is not None
            else args.seed + record.stage_id - 1
        )
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        source_image = source.image.unsqueeze(0).to(device=device, dtype=torch.float32)
        source_mask = source.mask.unsqueeze(0).to(device=device, dtype=torch.float32)
        initial_noise = torch.randn(
            1,
            *MRI_VQGAN_DEFAULTS.latent_shape(tuple(source_image.shape[-3:])),
            device=device,
            dtype=source_image.dtype,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        autocast_context = (
            torch.autocast(device_type="cpu", dtype=torch.bfloat16)
            if args.cpu_bf16 and device.type == "cpu"
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            result = model.sample_with_diagnostics(
                source_image,
                source_mask,
                [record.action_text],
                [record.clinical_text],
                torch.tensor([float(record.delta_days)], device=device),
                torch.tensor([record.stage_id], dtype=torch.long, device=device),
                noise=initial_noise,
                sampler=args.sampler,
                sampling_steps=sampling_steps,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        sampling_seconds = time.perf_counter() - started
        generated = result.image.detach().float().cpu()[0]
        target = visit_loader(record.target_visit_id)
        if source.image.shape != target.image.shape or generated.shape != source.image.shape:
            raise ValueError("source, generated, and target shapes do not match")
        centroid = tumor_centroid(source.mask)
        target_error = (generated - target.image).abs()
        source_error = (source.image - target.image).abs()
        source_mask_bool = source.mask > 0
        if not bool(source_mask_bool.any()):
            raise ValueError("selected source mask unexpectedly became empty")
        mae = float(target_error.mean())
        case = {
            "record": record,
            "validation_index": validation_index,
            "source": source.image,
            "generated": generated,
            "target": target.image,
            "mask": source.mask,
            "centroid": centroid,
            "seed": seed,
            "sampler_label": sampler_label,
            "sampling_seconds": sampling_seconds,
            "mae": mae,
            "mse": float(target_error.square().mean()),
            "tumor_mae": float(target_error[source_mask_bool].mean()),
            "copy_source_mae": float(source_error.mean()),
            "copy_source_tumor_mae": float(source_error[source_mask_bool].mean()),
            "unique_code_count": int(result.unique_code_count),
            "effective_code_count": float(result.effective_code_count),
        }
        artifact_stem = (
            f"case-{ordinal:02d}"
            if args.validation_indices is not None
            else f"stage-{record.stage_id}"
        )
        case["tensor_path"] = f"{artifact_stem}-tensors.pt"
        case["detail_path"] = f"{artifact_stem}-detail.png"
        cases.append(case)
        torch.save(
            {
                "source_zscore": source.image,
                "generated_zscore": generated,
                "target_zscore": target.image,
                "source_mask": source.mask,
                "initial_noise": initial_noise.detach().float().cpu()[0],
                "generated_normalized_latent": result.normalized_latent.detach()
                .float()
                .cpu()[0],
                "generated_continuous_latent": result.continuous_latent.detach()
                .float()
                .cpu()[0],
                "generated_code_indices": result.code_indices.detach().cpu()[0],
            },
            args.output_dir / case["tensor_path"],
        )
        _render_case(case, args.output_dir / case["detail_path"])
        print(
            f"completed validation index {validation_index} stage {record.stage_id} "
            f"with seed {seed} "
            f"in {sampling_seconds:.1f}s",
            flush=True,
        )

    _render_overview(cases, args.output_dir / "overview.png")
    manifest = {
        "schema_version": "mewm_registered_large_visualization_v2",
        "config": str(args.config.resolve()),
        "vqgan_checkpoint": str(args.vqgan_checkpoint.resolve()),
        "vqgan_sha256": vqgan_sha256,
        "diffusion_checkpoint": str(args.diffusion_checkpoint.resolve()),
        "diffusion_sha256": sha256_file(args.diffusion_checkpoint),
        "checkpoint_global_step": global_step,
        "data_backend": loaded.backend,
        "data_contract_sha256": loaded.data_contract_sha256,
        "denoiser_architecture": identity.denoiser_architecture,
        "sampler": args.sampler,
        "sampling_steps": sampling_steps,
        "eta": 0.0 if args.sampler == DDIM_ETA0_SAMPLER else None,
        "selection": (
            f"sorted validation indices {args.validation_indices}"
            if args.validation_indices is not None
            else (
                f"validation transition {args.transition_id}"
                if args.transition_id is not None
                else "first nonempty-source validation transition per stage"
            )
        ),
        "display_image_window_zscore": list(IMAGE_WINDOW),
        "display_error_max_zscore": ERROR_MAX,
        "cases": [
            {
                "stage_id": case["record"].stage_id,
                "validation_index": case["validation_index"],
                "transition_id": case["record"].transition_id,
                "source_visit_id": case["record"].source_visit_id,
                "target_visit_id": case["record"].target_visit_id,
                "seed": case["seed"],
                "centroid_zyx": list(case["centroid"]),
                "zscore_mae": case["mae"],
                "zscore_mse": case["mse"],
                "whole_volume_target_mae": case["mae"],
                "source_mask_tumor_target_mae": case["tumor_mae"],
                "copy_source_whole_volume_target_mae": case["copy_source_mae"],
                "copy_source_source_mask_tumor_target_mae": case[
                    "copy_source_tumor_mae"
                ],
                "unique_code_count": case["unique_code_count"],
                "effective_code_count": case["effective_code_count"],
                "sampling_seconds": case["sampling_seconds"],
                "tensor_path": case["tensor_path"],
                "detail_path": case["detail_path"],
            }
            for case in cases
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
