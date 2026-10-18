from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mewm_ispy2.backend import sha256_file
from mewm_ispy2.paper_contracts import (
    registered_large_paper_contract_payload,
    registered_large_paper_runtime_contract,
)


def _vqgan_is_running(pid: int) -> bool:
    proc = Path(f"/proc/{pid}/cmdline")
    try:
        command = proc.read_bytes().replace(b"\0", b" ")
    except FileNotFoundError:
        return False
    return b"mewm_ispy2.cli train-vqgan" in command


def _atomic_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def _build_diffusion_config(
    vqgan_config: Path, vqgan_checkpoint: Path, output: Path
) -> None:
    payload = yaml.safe_load(vqgan_config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("VQGAN config root must be a mapping")

    contract = registered_large_paper_runtime_contract(
        vqgan_sha256=sha256_file(vqgan_checkpoint)
    )
    payload["diffusion"] = {
        "denoiser_architecture": "mewm_paper_registered_large_v1",
        "input_channels": 49,
        "semantic_channels": 32,
        "prediction_type": "epsilon",
        "samples_per_transition": 8,
        "timesteps": 200,
        "sampler": "ancestral_ddpm",
        "latent_contract": "continuous_codebook_minmax_v1",
        "ema_decay": 0.995,
        "paper": registered_large_paper_contract_payload(contract),
    }
    payload["diffusion_training"] = {"batch_size": 12}
    _atomic_yaml(output, payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqgan-pid", type=int, required=True)
    parser.add_argument("--vqgan-log", type=Path, required=True)
    parser.add_argument("--vqgan-config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--diffusion-log", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=200)
    args = parser.parse_args()

    print(f"waiting for VQGAN pid {args.vqgan_pid}", flush=True)
    while _vqgan_is_running(args.vqgan_pid):
        time.sleep(60)

    log_text = args.vqgan_log.read_text(encoding="utf-8", errors="replace")
    failure_markers = ("Traceback (most recent call last)", "CUDA out of memory")
    if any(marker in log_text for marker in failure_markers):
        raise RuntimeError("VQGAN ended with an error; Diffusion will not start")

    checkpoints = sorted(args.checkpoint_dir.glob("best-composite-*.ckpt"))
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"expected one best VQGAN composite checkpoint, found {len(checkpoints)}"
        )
    vqgan_checkpoint = checkpoints[0].resolve()
    _build_diffusion_config(
        args.vqgan_config.resolve(), vqgan_checkpoint, args.resolved_config.resolve()
    )

    args.diffusion_log.parent.mkdir(parents=True, exist_ok=True)
    log_handle = args.diffusion_log.open("ab", buffering=0)
    os.dup2(log_handle.fileno(), sys.stdout.fileno())
    os.dup2(log_handle.fileno(), sys.stderr.fileno())
    command = [
        sys.executable,
        "-m",
        "mewm_ispy2.cli",
        "train-diffusion",
        "--config",
        str(args.resolved_config.resolve()),
        "--max-epochs",
        str(args.max_epochs),
        "--devices",
        "1",
        "--vqgan-checkpoint",
        str(vqgan_checkpoint),
    ]
    os.execv(sys.executable, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
