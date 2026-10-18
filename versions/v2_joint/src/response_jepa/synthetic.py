"""Small, explicitly synthetic fixtures for integration tests; NOT clinical evidence."""
from __future__ import annotations
from pathlib import Path
import copy
import numpy as np
from .config import Config
from .data import CLINICAL, PHASES, ARMS
from .engine import write_json, train, load_for_inference, evaluate


def make_cohort(directory, seed=20261008, shape=(8, 16, 16), counts=(16, 8, 4)):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/"manifest.json").exists():
        raise FileExistsError("Synthetic directory already exists; use a fresh directory")
    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(*(np.linspace(-1, 1, s) for s in shape), indexing="ij")
    support = x*x + y*y + .5*z*z < .95
    patients = []
    counter = 0
    for split, n in zip(("train", "val", "test"), counts):
        for i in range(n):
            key = f"SYNTHETIC-{counter:04d}"
            label = i % 2
            geometry = {"source_only": True, "source_stage": 0, "roi_source": "T0",
                        "orientation": "RAS", "resampling": "source_grid", "grid_id": key+"-T0"}
            visits = []
            offset = rng.uniform(-.2, .2, 2)
            for j in range(4):
                if i % 7 == 6 and j == 1:   # a missing intermediate, with later target retained
                    visits.append(None)
                    continue
                radius = .40 - j*(.055 if label else .01)
                lesion = np.exp(-((x-offset[0])**2 + (y-offset[1])**2 + z*z)/(2*radius*radius))
                anatomy = .15 + .1*np.sin(3*x)*np.cos(2*y)
                image = np.stack([anatomy + .10*lesion,
                                  anatomy + (.9-.15*j*label)*lesion,
                                  anatomy + (.65-.10*j*label)*lesion])
                image = (image + rng.normal(0, .02, image.shape))*support[None]
                filename = f"{key}_T{j}.npy"
                np.save(directory/filename, image.astype(np.float32), allow_pickle=False)
                visits.append({"stage": j, "available_at": j, "grid_id": geometry["grid_id"], "image": filename})
            patients.append({"patient_key": key, "split": split, "geometry": geometry, "visits": visits,
                             "clinical": {"values": [float(40+i), float(i%2), float((i//2)%2), 1.0],
                                          "known_at": [0, 0, 0, 0]},
                             "arm": {"label": ARMS[i%2], "known_at_stage": 0, "semantics": "assigned_arm_scenario"},
                             "target": {"pcr": label}})
            counter += 1
    manifest = {"schema": "responsewm_raw_mri_v1", "synthetic": True, "time_basis": "stage_index",
                "canonical_stages": [0, 1, 2, 3], "phase_order": PHASES, "clinical_features": CLINICAL,
                "image_preprocessing": {"mode": "synthetic_preprocessed"}, "patients": patients}
    write_json(directory/"manifest.json", manifest)
    return directory/"manifest.json"


def small_config(phase="C"):
    cfg = Config()
    m, t, l = cfg.model, cfg.train, cfg.loss
    m.dim, m.widths, m.depths, m.token_grid = 32, [8, 16, 24, 32], [1, 1, 1, 1], [1, 2, 2]
    m.spatial_depth = m.fusion_depth = m.predictor_depth = 1
    m.context_queries, m.state_queries, m.pcr_queries = 4, 2, 2
    m.dropout = 0.0
    t.phase, t.device, t.precision, t.allow_synthetic = phase, "cpu", "fp32", True
    t.epochs, t.patience, t.warmup_epochs = 2, 2, 0
    t.batch_size, t.eval_batch_size, t.accumulation, t.threads = 4, 4, 1, 2
    t.workers, t.pin_memory = 0, False
    t.image_shape = [8, 16, 16]
    t.profile = "dynamic"
    t.augment = phase == "A"
    t.freeze_encoder = phase not in {"A", "joint"}
    if phase == "A":
        m.readout = "direct"
        l.tf = l.rollout = 0
        t.world_eval = False
    elif phase == "B":
        m.readout = "direct"
    return cfg.validate()


def smoke(output):
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = make_cohort(out/"synthetic_data")
    results = {}
    initial = None
    for phase in ("A", "B", "C"):
        cfg = small_config(phase)
        directory = out/phase
        results[phase] = train(cfg, manifest, directory, init=initial)
        initial = directory/"best.pt"
    model, store, cfg, state = load_for_inference(initial, manifest, allow_synthetic=True)
    metrics, _, _ = evaluate(model, store, cfg, "test")
    report = {"scope": "synthetic integration only; no real MRI or clinical performance validation",
              "phases_completed": ["A", "B", "C"],
              "best_epochs": {k: v["best_epoch"] for k, v in results.items()},
              "test_selection": metrics["selection"],
              "target_coordinate_origin": metrics["target_coordinate_origin"]}
    write_json(out/"smoke_report.json", report)
    return report
