"""Managed registered three-phase ROI32 preparation, training and evaluation."""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mewm_ispy2 import registered_roi32_runtime as runtime
from mewm_ispy2.registered_roi32_data import check_disk, file_identity, read_json, timestamp, write_json
from mewm_ispy2.registered_three_phase_data import inventory, load_config


def progress(config, status, **values):
    payload = {"status": status, "updated_at": timestamp(), "pid": os.getpid(), **values}
    write_json(Path(config["output_dir"]) / "progress.json", payload)
    print(json.dumps(payload, allow_nan=False), flush=True)


def check_prerequisites(config, baseline, manifest, stage):
    if stage in ("all", "latents", "smoke-fm", "fm", "evaluate") and manifest["missing_sources"]:
        raise ValueError("Restore the missing registered sources before full-cohort generation")
    required = {"all": ("vq_audit/COMPLETE.json", "vq_audit/summary.json"),
                "fm": ("latents/COMPLETE.json", "vq_audit/COMPLETE.json", "vq_audit/summary.json", "smoke/fm-gpu.json"),
                "evaluate": ("fm/COMPLETE.json",)}
    root = Path(config["output_dir"])
    for name in required.get(stage, ()):
        if not (root / name).is_file():
            raise ValueError(f"Complete the prerequisite stage before {stage}: {name}")
    if stage in ("all", "fm"):
        audit = read_json(root / "vq_audit/summary.json")
        if not audit["codebook_unchanged"] or audit["codec"] != file_identity(Path(baseline["output_dir"]) / "vq/best.pt"):
            raise ValueError("The completed transfer audit does not match the selected frozen codec")


def select_gpu_batch(config, baseline, manifest, device):
    import torch
    from mewm_ispy2.registered_three_phase_batching import benchmark
    from mewm_ispy2.registered_three_phase_training import smoke, source_contract

    selection = benchmark(config, baseline, manifest, device)
    if selection is None:
        return None
    path = Path(config["output_dir"]) / "smoke/fm-gpu.json"
    contract = source_contract(config, baseline)
    options = list(dict.fromkeys((selection["batch_size"], selection["fallback_batch_size"])))
    if path.is_file():
        report = read_json(path)
        if (report["status"] == "passed" and report["contract"] == contract
                and report["device"].startswith("cuda") and report["batch_size"] in options):
            return report["batch_size"]
    for batch_size in options:
        progress(config, "gpu_smoke", batch_size=batch_size)
        try:
            report = smoke(config, baseline, manifest, device, batch_size=batch_size)
            return report["batch_size"]
        except torch.cuda.OutOfMemoryError:
            progress(config, "gpu_smoke_batch_oom", batch_size=batch_size)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError("Both configured three-phase physical batches exceed GPU memory")


def run_stages(config, baseline, manifest, stage, device, pins):
    import torch
    from mewm_ispy2 import registered_three_phase_audit as audit
    from mewm_ispy2 import registered_three_phase_evaluation as evaluation
    from mewm_ispy2 import registered_three_phase_latents as latents
    from mewm_ispy2 import registered_three_phase_training as training

    stages = ("latents", "smoke-fm", "fm", "evaluate") if stage == "all" else (stage,)
    actions = {"audit-vq": audit.evaluate, "latents": latents.extract,
               "fm": training.train, "evaluate": evaluation.evaluate}
    for current in stages:
        if runtime.STOP_REQUESTED:
            return False
        runtime.verify_contract_files(pins)
        progress(config, "running_stage", stage=current, device=str(device))
        if current == "smoke-fm":
            if device.type == "cpu":
                training.smoke(config, baseline, manifest, device, batch_size=1)
            else:
                select_gpu_batch(config, baseline, manifest, device)
            finished = True
        else:
            finished = actions[current](config, baseline, manifest, device)
        if finished is False or runtime.STOP_REQUESTED:
            return False
        progress(config, "stage_complete", stage=current, device=str(device))
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return True


def execute(config, baseline, config_path, stage, cpu=False, shared_gpu=None, memory_fraction=0.45):
    if cpu and stage == "all":
        raise ValueError("The managed training sequence requires a GPU")
    manifest = inventory(config, baseline)
    progress(config, "inventory_complete", counts=manifest["counts"])
    if stage == "inventory":
        return
    check_prerequisites(config, baseline, manifest, stage)
    import torch
    from run_registered_roi32 import reserve_gpu
    from mewm_ispy2 import registered_three_phase_evaluation as evaluation
    from mewm_ispy2 import registered_three_phase_latents as latents
    from mewm_ispy2 import registered_three_phase_training as training

    paths = [__file__, runtime.__file__, evaluation.__file__, config_path,
             config["three_phase"]["baseline_config"], sys.modules[reserve_gpu.__module__].__file__]
    pins = {"sources": [file_identity(path) for path in paths],
            "latent_contract": latents.contract_for(config, baseline),
            "flow_sources": training.source_contract(config, baseline),
            "execution": {"shared_gpu": shared_gpu, "gpu_memory_fraction": memory_fraction if shared_gpu is not None else None}}
    write_json(Path(config["output_dir"]) / "controller_binding.json", pins)
    check_disk(config, 8 * 1024**3)
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    lock = None
    if cpu:
        device = torch.device("cpu")
    else:
        if shared_gpu is None:
            index, lock = reserve_gpu(config)
            if index is None:
                return
        else:
            index = shared_gpu
            lock = Path(f"/tmp/registered_three_phase_roi32_shared_gpu_{index}.lock").open("a+")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                raise RuntimeError("Another three-phase process already shares this GPU") from None
        if torch.cuda.is_initialized():
            lock.close()
            raise RuntimeError("CUDA initialized before GPU reservation")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        device = torch.device("cuda:0")
        progress(config, "shared_gpu_selected" if shared_gpu is not None else "gpu_reserved",
                 physical_gpu=index, stage=stage)
    try:
        runtime.verify_contract_files(pins)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            if shared_gpu is not None:
                torch.cuda.set_per_process_memory_fraction(memory_fraction, device)
                free, total = torch.cuda.mem_get_info(device)
                if free < 12 * 1024**3:
                    raise RuntimeError("Shared GPU has less than 12 GiB free before starting")
                progress(config, "shared_gpu_ready", physical_gpu=index, allocator_fraction=memory_fraction,
                         allocator_limit_mib=total * memory_fraction / 1024**2, free_mib=free / 1024**2)
            torch.use_deterministic_algorithms(config["runtime"]["deterministic_algorithms"])
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
        if run_stages(config, baseline, manifest, stage, device, pins):
            if stage == "all":
                write_json(Path(config["output_dir"]) / "COMPLETE.json",
                           {"status": "passed", "configuration": config, "updated_at": timestamp()})
            progress(config, "complete", stage=stage)
    finally:
        if lock is not None:
            lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO / "configs/registered_three_phase_roi32_v1.yaml"))
    parser.add_argument("--stage", choices=("all", "inventory", "audit-vq", "latents", "smoke-fm", "fm", "evaluate"), default="inventory")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--shared-gpu", type=int, help="Use this GPU alongside its existing jobs")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cpu and args.stage == "all":
        parser.error("--stage all requires the managed GPU sequence")
    if args.cpu and args.shared_gpu is not None:
        parser.error("--cpu and --shared-gpu cannot be combined")
    if not 0 < args.gpu_memory_fraction <= 1 or (args.shared_gpu is not None and args.shared_gpu < 0):
        parser.error("GPU memory fraction must be in (0,1] and GPU index must be nonnegative")
    config_path = Path(args.config).resolve()
    config, baseline = load_config(config_path)
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with runtime.output_lock(root, args.lock_fd) as lock:
        if args.detach and not args.worker:
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--config", str(config_path),
                       "--stage", args.stage, "--worker", "--lock-fd", str(lock.fileno())]
            if args.cpu:
                command.append("--cpu")
            if args.shared_gpu is not None:
                command.extend(("--shared-gpu", str(args.shared_gpu), "--gpu-memory-fraction", str(args.gpu_memory_fraction)))
            environment = dict(os.environ, PYTHONUNBUFFERED="1", OPENBLAS_NUM_THREADS="1",
                               OMP_NUM_THREADS=str(config["runtime"]["cpu_threads"]),
                               MKL_NUM_THREADS=str(config["runtime"]["cpu_threads"]),
                               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", CUBLAS_WORKSPACE_CONFIG=":4096:8")
            with (root / "controller.log").open("a") as handle:
                child = subprocess.Popen(command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                                         pass_fds=(lock.fileno(),))
            write_json(root / "launch.json", {"pid": child.pid, "configuration": file_identity(config_path),
                                               "stage": args.stage, "shared_gpu": args.shared_gpu,
                                               "gpu_memory_fraction": args.gpu_memory_fraction if args.shared_gpu is not None else None,
                                               "started_at": timestamp()})
            print(json.dumps({"pid": child.pid, "log": str(root / "controller.log"),
                              "progress": str(root / "progress.json")}))
            return
        runtime.install_signals()
        try:
            execute(config, baseline, config_path, args.stage, args.cpu, args.shared_gpu, args.gpu_memory_fraction)
            if runtime.STOP_REQUESTED:
                progress(config, "paused", stage=args.stage)
        except Exception as error:
            progress(config, "failed", stage=args.stage, error=str(error))
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
