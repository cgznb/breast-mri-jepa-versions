"""Predeclared ablations and dependency-aware execution; never selects on test data."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import argparse
import json
import shlex
import subprocess
import sys
import yaml
from .config import Config, from_dict
from .data import Store
from .engine import load_checkpoint, write_json
from .provenance import source_snapshot, file_identity

# C recipe, B variant (None means no B), whether A initialization is needed.
EXPERIMENTS = {
    "clinical": ("clinical", None, False),
    "clinical_arm": ("clinical_arm", None, False),
    "mri_direct": ("mri_direct", None, False),
    "legacy_direct": ("legacy_direct", None, False),
    "legacy_large_grid": ("legacy_large_grid", None, False),
    "multiscale_direct": ("multiscale_direct", None, False),
    "pretrained_direct_ft": ("pretrained_direct_ft", None, True),
    "pretrained_direct": ("pretrained_direct", None, True),
    "tf_aux": ("tf_aux", "tf_only", True),
    "rollout_aux": ("rollout_aux", "default", True),
    "full": ("full", "default", True),
    "copy_readout": ("copy_readout", "default", True),
    "stopgrad_forecast": ("stopgrad_forecast", "default", True),
    "no_residual": ("no_residual", "absolute", True),
    "no_tf": ("no_tf", "free_only", True),
    "no_arm": ("no_arm", "no_arm", True),
    "linear_world": ("linear_world", "linear", True),
    "full_encoder_ft": ("full_encoder_ft", "default", True),
}


def base_config(phase, profile="t0"):
    c = Config()
    c.train.phase = phase
    c.train.profile = profile
    c.train.augment = False
    if phase == "A":
        c.train.profile = "dynamic"  # A is same-visit SSL, shared across later protocols
        c.train.epochs = 80
        c.train.freeze_encoder = False
        c.train.augment = True
        c.train.world_eval = False
        c.model.activation_checkpoint = True
        c.model.readout = "direct"
        c.loss.tf = c.loss.rollout = 0
    elif phase == "B":
        c.train.epochs = 80
        c.train.freeze_encoder = True
        c.model.readout = "direct"  # task heads are not trained in B
        c.train.world_eval = True
    else:
        c.train.epochs = 60
    return c.validate()


def world_config(variant, profile):
    c = base_config("B", profile)
    if variant == "tf_only":
        c.loss.rollout = 0
    elif variant == "absolute":
        c.model.residual_world = False
    elif variant == "free_only":
        c.loss.tf = 0
    elif variant == "no_arm":
        c.model.use_arm = False
    elif variant == "linear":
        c.model.world_arch = "linear"
    elif variant != "default":
        raise ValueError(f"Unknown world variant {variant}")
    return c.validate()


def task_config(name, profile):
    c = base_config("C", profile)
    direct = {"clinical", "clinical_arm", "mri_direct", "legacy_direct", "legacy_large_grid",
              "multiscale_direct", "pretrained_direct", "pretrained_direct_ft"}
    if name in direct:
        c.model.readout = "direct"
        c.loss.tf = c.loss.rollout = 0
        c.train.world_eval = False
        c.train.freeze_encoder = name in {"clinical", "clinical_arm", "pretrained_direct"}
    if name in {"clinical", "clinical_arm"}:
        c.model.use_mri = False
        c.model.use_arm = name == "clinical_arm"
    elif name == "mri_direct":
        c.model.use_clinical = c.model.use_arm = False
    elif name in {"legacy_direct", "legacy_large_grid"}:
        c.model.encoder = "legacy_small"
        if name == "legacy_direct":
            c.model.token_grid = [2, 4, 4]
    elif name == "tf_aux":
        c.model.readout = "direct"
        c.loss.rollout = 0
    elif name == "rollout_aux":
        c.model.readout = "direct"
    elif name == "copy_readout":
        # Identical B init, heads, coefficients and world losses to full.
        # ONLY terminal readout receives the anchor copy instead of the predicted S3.
        c.model.readout = "copy_gated"
    elif name == "stopgrad_forecast":
        c.model.forecast_detach = True
    elif name == "no_residual":
        c.model.residual_world = False
    elif name == "no_tf":
        c.loss.tf = 0
    elif name == "no_arm":
        c.model.use_arm = False
    elif name == "linear_world":
        c.model.world_arch = "linear"
    elif name == "full_encoder_ft":
        c.train.freeze_encoder = False
        c.loss.align = 1.0
        c.model.activation_checkpoint = True
    elif name not in direct and name != "full":
        raise ValueError(f"Unknown experiment {name}")
    if not c.train.freeze_encoder and c.model.encoder == "multiscale" and c.model.use_mri:
        c.model.activation_checkpoint = True
    return c.validate()


def generate_templates(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    with (root/"A_representation.yaml").open("w") as f:
        yaml.safe_dump(base_config("A").to_dict(), f, sort_keys=False)
    for profile in ("t0", "dynamic"):
        folder = root/profile
        folder.mkdir(exist_ok=True)
        for variant in ("default", "tf_only", "absolute", "free_only", "no_arm", "linear"):
            with (folder/f"B_{variant}.yaml").open("w") as f:
                yaml.safe_dump(world_config(variant, profile).to_dict(), f, sort_keys=False)
        for name in EXPERIMENTS:
            with (folder/f"C_{name}.yaml").open("w") as f:
                yaml.safe_dump(task_config(name, profile).to_dict(), f, sort_keys=False)
    for stage in (0, 3):
        cfg = task_config("pretrained_direct", "t0")
        cfg.train.profile = f"single_t{stage}"
        cfg.validate()
        with (root/f"C_single_T{stage}_reference.yaml").open("w") as f:
            yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)


def plan_runs(manifest, workdir, profile, names, seeds, *, device="cuda", cache=None, overrides=()):
    root = Path(workdir).resolve()
    config_dir = root/"resolved_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    nodes = {}
    def add(key, cfg, init=None):
        if key in nodes:
            return nodes[key]
        cfg.train.device = device
        cfg.train.image_cache = cache
        d = cfg.to_dict()
        for item in overrides:
            key_arg, sep, value = item.partition("=")
            if not sep or key_arg.count(".") != 1:
                raise ValueError("Override format is section.key=value")
            section, field = key_arg.split(".")
            if section not in {"model", "train", "loss"} or field not in d[section]:
                raise ValueError(f"Unknown override {key_arg}")
            # Recipes are scientific conditions, not a hidden mutable sweep.
            if key_arg not in {"train.batch_size", "train.accumulation", "train.workers", "train.precision",
                               "train.eval_batch_size", "train.threads"}:
                raise ValueError("Suite --set only permits hardware/batch settings. Edit individual YAMLs for new scientific conditions.")
            d[section][field] = yaml.safe_load(value)
        cfg = from_dict(d)
        config_file = config_dir/(key.replace("/", "__")+".yaml")
        with config_file.open("w") as f:
            yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
        output = root/key
        command = [sys.executable, str(Path(__file__).parents[2]/"train_response.py"), "train",
                   "--config", str(config_file), "--manifest", str(Path(manifest).resolve()), "--output", str(output)]
        if init:
            command += ["--init", init]
        node = {"id": key, "phase": cfg.train.phase, "config": cfg.to_dict(), "output": str(output),
                "init": init, "command": command}
        nodes[key] = node
        return node
    for seed in seeds:
        for name in names:
            if name not in EXPERIMENTS:
                raise ValueError(f"Unknown experiment {name}; available: {','.join(EXPERIMENTS)}")
            recipe, variant, needs_a = EXPERIMENTS[name]
            initial = None
            if needs_a:
                ac = base_config("A")
                ac.train.seed = seed
                a = add(f"A_shared/seed{seed}", ac)
                initial = str(Path(a["output"])/"best.pt")
            if variant:
                bc = world_config(variant, profile)
                bc.train.seed = seed
                b = add(f"{profile}/B_{variant}/seed{seed}", bc, initial)
                initial = str(Path(b["output"])/"best.pt")
            cc = task_config(recipe, profile)
            cc.train.seed = seed
            add(f"{profile}/{name}/seed{seed}", cc, initial)
    return list(nodes.values())


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate an explicit, dependency-aware ablation plan")
    p.add_argument("--manifest", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--profile", choices=("t0", "dynamic"), default="t0")
    p.add_argument("--experiments", default="full", help="Comma-separated IDs or all")
    p.add_argument("--seeds", default="17,43")
    p.add_argument("--device", default="cuda")
    p.add_argument("--image-cache")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--execute", action="store_true", help="Run sequentially in this process; default only prints a plan")
    p.add_argument("--resume", action="store_true", help="Validate/reuse completed runs or resume partial runs")
    p.add_argument("--require-positive-world-skill", action="store_true",
                   help="Stop before C if B validation ratio>=1 or undefined (not a significance test)")
    args = p.parse_args(argv)
    names = list(EXPERIMENTS) if args.experiments == "all" else args.experiments.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]
    plan = plan_runs(args.manifest, args.workdir, args.profile, names, seeds,
                     device=args.device, cache=args.image_cache, overrides=args.set)
    write_json(Path(args.workdir)/f"plan_{args.profile}.json", plan)
    print(f"Planned {len(plan)} training runs; shared A/B dependencies appear only once.", flush=True)
    for node in plan:
        print(shlex.join(node["command"]), flush=True)
        if not args.execute:
            continue
        command = list(node["command"])
        last = Path(node["output"])/"last.pt"
        completed = False
        if last.exists() and args.resume:
            state = load_checkpoint(last)
            cfg = from_dict(node["config"])
            store = Store(args.manifest, cfg.train.image_shape, allow_synthetic=cfg.train.allow_synthetic,
                          image_cache=cfg.train.image_cache)
            if (state["config"] != node["config"] or state["source_identity"] != source_snapshot()
                    or state["data_fingerprint"] != store.fingerprint()):
                raise ValueError(f"Existing run differs from planned code/config/data: {node['id']}")
            if node["init"] and state["init"]["checkpoint_identity"] != file_identity(node["init"]):
                raise ValueError("Upstream initialization checkpoint has changed")
            completed = state["epoch"] >= cfg.train.epochs or state["bad_epochs"] >= cfg.train.patience
            if not completed:
                if "--init" in command:
                    index = command.index("--init")
                    del command[index:index+2]
                command += ["--resume"]
        if not completed:
            subprocess.run(command, check=True)
        else:
            print(f"Verified completed run: {node['id']}", flush=True)
        if node["phase"] == "B" and args.require_positive_world_skill:
            with (Path(node["output"])/"metrics_best.json").open() as f:
                ratio = json.load(f)["world"]["all"]["ratio"]
            if ratio is None or ratio >= 1:
                raise RuntimeError(f"B has not beaten copy (ratio={ratio}). Inspect B before continuing to C.")

if __name__ == "__main__":
    main()
