"""Standalone adapter for the audited upstream preprocessed_roi32 manifest/cache.

No re-z-scoring, resizing, future-driven cropping, automatic patient dropping,
or fabricated visit dates. Raw NIfTI/DICOM preparation remains upstream.
"""
from __future__ import annotations
from dataclasses import fields
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import math
import re
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from .contracts import Inputs, Targets, Batch
from .provenance import file_identity, record_metadata

PHASES = ["pre_aqc0", "first_post_aqc1", "metadata_late"]
CLINICAL = ["age_at_screening", "hr_positive", "her2_positive", "mammaprint_binary"]
# Category order matches the audited upstream arm.py. Not a drug/dose model.
ARMS = (
    "Paclitaxel", "Paclitaxel + ABT 888 + Carboplatin", "Paclitaxel + AMG 386",
    "Paclitaxel + AMG 386 + Trastuzumab", "Paclitaxel + Ganetespib",
    "Paclitaxel + Ganitumab", "Paclitaxel + MK-2206", "Paclitaxel + MK-2206 + Trastuzumab",
    "Paclitaxel + Neratinib", "Paclitaxel + Pembrolizumab",
    "Paclitaxel + Pertuzumab + Trastuzumab", "Paclitaxel + Trastuzumab", "T-DM1 + Pertuzumab",
)

def canonical_key(x):
    x = str(x).strip()
    if x.startswith("ISPY2-"):
        x = x[6:]
    if re.fullmatch(r"\d+\.0", x):
        x = x[:-2]
    if not x or x.lower() in {"nan", "none"}:
        raise ValueError("Missing patient key")
    return x

def json_hash(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def stage(x):
    if type(x) is not int or not 0 <= x <= 3:
        raise ValueError("Stages must be explicit integers 0..3")
    return x

class Store:
    def __init__(self, path, image_shape=(32, 128, 128), *, allow_synthetic=False,
                 normalization=None, image_cache=None):
        self.path = Path(path).resolve()
        self.manifest = load_json(self.path)
        self.image_shape = tuple(image_shape)
        m = self.manifest
        if (m.get("schema") != "responsewm_raw_mri_v1" or m.get("canonical_stages") != [0, 1, 2, 3]
                or m.get("phase_order") != PHASES or m.get("clinical_features") != CLINICAL
                or m.get("time_basis") != "stage_index"):
            raise ValueError("Expected upstream responsewm_raw_mri_v1 stage-index manifest")
        synthetic = bool(m.get("synthetic", False))
        mode = m.get("image_preprocessing", {}).get("mode")
        if synthetic and not allow_synthetic:
            raise ValueError("Synthetic data require explicit train.allow_synthetic=true")
        if mode == "preprocessed_roi32":
            if self.image_shape != (32, 128, 128):
                raise ValueError("Real ROI32 must remain [32,128,128]; resizing is forbidden")
            n = m.get("image_normalization", {})
            required = {"mean", "std", "fit_split", "scope", "background_policy",
                        "already_normalized", "normalization_policy"}
            if (not required.issubset(n) or n["fit_split"] != "train"
                    or n["background_policy"] != "preserve_zero" or n["already_normalized"] is not True
                    or type(n["std"]) not in {float, int} or not math.isfinite(n["std"]) or n["std"] <= 0
                    or type(n["mean"]) not in {float, int} or not math.isfinite(n["mean"])
                    or any(not isinstance(n[k], str) or not n[k].strip() for k in ("scope", "normalization_policy"))):
                raise ValueError("Missing/invalid train-only image normalization provenance")
        elif not (synthetic and mode == "synthetic_preprocessed"):
            raise ValueError("v2 accepts preprocessed_roi32 NPZ (or explicit synthetic fixtures). "
                             "Use upstream preparation for raw NIfTI/DICOM; no silent preprocessing changes.")
        self.patients = m.get("patients", [])
        if not self.patients:
            raise ValueError("Empty cohort")
        self.splits = defaultdict(list)
        keys, assets = set(), set()
        for i, p in enumerate(self.patients):
            key = canonical_key(p["patient_key"])
            if key in keys:
                raise ValueError("Duplicate patient / cross-split leakage")
            keys.add(key)
            if p.get("split") not in {"train", "val", "test"}:
                raise ValueError("Unknown split")
            self.splits[p["split"]].append(i)
            g = p.get("geometry", {})
            if (g.get("source_only") is not True or g.get("source_stage") != 0
                    or g.get("roi_source") != "T0" or g.get("orientation") != "RAS"
                    or g.get("resampling") != "source_grid" or not g.get("grid_id")):
                raise ValueError("A T0-only ROI/reference-grid declaration is required")
            visits = p.get("visits")
            if not isinstance(visits, list) or len(visits) != 4 or visits[0] is None:
                raise ValueError("Keep four visit slots, with real T0; missing visits remain null")
            for j, v in enumerate(visits):
                if v is None:
                    continue
                if stage(v.get("stage")) != j or not j <= stage(v.get("available_at")) <= 3:
                    raise ValueError("Invalid visit chronology")
                if v.get("grid_id") != g["grid_id"] or (j == 0 and v["available_at"] != 0):
                    raise ValueError("Grid or baseline availability mismatch")
                if not isinstance(v.get("image"), str) or "phases" in v:
                    raise ValueError("Use one stacked image path per visit")
                f = self.resolve(v["image"])
                if f.suffix not in ({".npy", ".npz"} if synthetic else {".npz"}):
                    raise ValueError("Real ROI32 visits must reference stacked NPZ files")
                if str(f) in assets:
                    raise ValueError("An image asset is reused across patients/stages/splits")
                assets.add(str(f))
            self.clinical_raw(p, 3)
            self.arm(p, 3)
        self.normalization = normalization or self.fit_normalization()
        n = self.normalization
        if (n.get("schema") != "response_jepa_clinical_norm_v1" or n.get("features") != CLINICAL
                or len(n.get("mean", [])) != 4 or len(n.get("std", [])) != 4
                or not np.isfinite(n["mean"]).all() or not np.isfinite(n["std"]).all()
                or np.any(np.asarray(n["std"]) <= 0)):
            raise ValueError("Invalid checkpoint normalization")
        if n["mean"][1:] != [0.0, 0.0, 0.0] or n["std"][1:] != [1.0, 1.0, 1.0]:
            raise ValueError("Binary clinical variables must retain 0/1 coding")
        self.cache = None
        if image_cache:
            self.cache = self._read_cache(image_cache, assets)

    def resolve(self, path):
        p = Path(path)
        return (p if p.is_absolute() else self.path.parent / p).resolve()

    @staticmethod
    def clinical_raw(p, landmark):
        c = p.get("clinical", {})
        vals, known = c.get("values", []), c.get("known_at", [])
        if len(vals) != 4 or len(known) != 4:
            raise ValueError("Four clinical values and known_at entries are required")
        mask = []
        for j, (value, when) in enumerate(zip(vals, known)):
            if value is None:
                if when is not None:
                    raise ValueError("Missing clinical value must have null known_at")
                mask.append(False)
            else:
                if type(value) not in {int, float} or not math.isfinite(value) or (j and value not in (0, 1)):
                    raise ValueError("Clinical value invalid")
                mask.append(stage(when) <= landmark)
        return np.array([v if v is not None else 0 for v in vals], dtype=np.float32), np.array(mask, dtype=bool)

    @staticmethod
    def arm(p, landmark):
        a = p.get("arm", {})
        if (set(a) != {"label", "known_at_stage", "semantics"}
                or a["semantics"] not in {"prospective_verified", "assigned_arm_scenario"}):
            raise ValueError("Declare Arm label, known_at_stage and semantics")
        if a["label"] is None:
            if a["known_at_stage"] is not None:
                raise ValueError("Unknown Arm must have null timing")
            return 0, -1
        if a["label"] not in ARMS:
            raise ValueError("Arm label is not one of the upstream 13 official categories")
        when = stage(a["known_at_stage"])
        return (ARMS.index(a["label"]) + 1, when) if when <= landmark else (0, -1)

    def fit_normalization(self):
        pairs = [self.clinical_raw(self.patients[i], 0) for i in self.splits["train"]]
        if not pairs:
            raise ValueError("Inference-only cohort requires checkpoint normalization")
        values, masks = np.stack([x[0] for x in pairs]), np.stack([x[1] for x in pairs])
        ages = values[masks[:, 0], 0]
        mean = float(ages.mean()) if len(ages) else 0.0
        std = float(ages.std()) if len(ages) else 1.0
        if std < 1e-6:
            std = 1.0
        return {"schema": "response_jepa_clinical_norm_v1", "features": CLINICAL,
                "mean": [mean, 0.0, 0.0, 0.0], "std": [std, 1.0, 1.0, 1.0],
                "fit_split": "train", "count": int(len(ages)),
                "fit_patient_count": len(self.splits["train"]),
                "fit_patient_keys": sorted(canonical_key(self.patients[i]["patient_key"])
                                           for i in self.splits["train"])}

    def _read_cache(self, directory, assets):
        directory = self.resolve(directory)
        index = load_json(directory / "index.json")
        if (index.get("schema") != "registered_roi32_image_cache_v1"
                or index.get("shape") != [3, 32, 128, 128] or index.get("dtype") != "float32"
                or index.get("phase_order") != PHASES
                or index.get("image_normalization") != self.manifest.get("image_normalization")
                or index.get("image_preprocessing") != {"mode": "preprocessed_roi32"}):
            raise ValueError("Original cache metadata does not match this manifest")
        result = {}
        for e in index["images"]:
            source = str(Path(e["source"]["path"]).resolve())
            cached = Path(e["cached"]["path"]).resolve()
            if source in result or not cached.is_relative_to(directory) or cached.suffix != ".npy":
                raise ValueError("Invalid original cache entry")
            if file_identity(source) != e["source"] or file_identity(cached) != e["cached"]:
                raise ValueError("Stale cache: source/cached file identity changed; regenerate upstream")
            result[source] = e
        if not assets.issubset(result):
            raise ValueError("Manifest images are absent from the original cache")
        # A subset is permitted for external inference, unlike changing cached values.
        return result

    def read_image(self, v):
        source = self.resolve(v["image"])
        path = source
        if self.cache is not None:
            e = self.cache[str(source)]
            if file_identity(source) != e["source"] or file_identity(e["cached"]["path"]) != e["cached"]:
                raise ValueError("Image/cache changed during execution")
            path = Path(e["cached"]["path"])
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as z:
                if "image" not in z:
                    raise ValueError("NPZ is missing the image array")
                x = z["image"]
        else:
            x = np.load(path, allow_pickle=False, mmap_mode="r")
        if x.shape != (3, *self.image_shape) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError(f"Image must be finite float32 [3,{','.join(map(str, self.image_shape))}]")
        return torch.from_numpy(np.array(x, copy=True))

    def allowed(self, i, profile="dynamic"):
        visits = self.patients[i]["visits"]
        if profile in {"t0", "single_t0"}:
            return [0]
        if profile == "single_t3":
            return [3] if visits[3] is not None and visits[3]["available_at"] <= 3 else []
        return sorted({v["available_at"] for v in visits if v is not None})

    def observed_stage(self, i, landmark):
        legal = [j for j, visit in enumerate(self.patients[i]["visits"])
                 if visit is not None and j <= landmark and visit["available_at"] <= landmark]
        if not legal:
            raise ValueError("Requested prefix has no legally observed MRI visit")
        return max(legal)

    def sample(self, i, landmark, *, future=True, labels=True, single=False, load_images=True):
        p = self.patients[i]
        observed = torch.tensor([v is not None and j <= landmark and v["available_at"] <= landmark
                                 for j, v in enumerate(p["visits"])], dtype=torch.bool)
        if single:
            observed &= torch.arange(4) == landmark
        if not observed.any():
            raise ValueError("Requested reference visit is unavailable")
        shape = self.image_shape if load_images else (1, 1, 1)
        images = torch.zeros(4, 3, *shape)
        target_images = torch.zeros_like(images)
        fm = torch.zeros(4, dtype=torch.bool)
        for j, v in enumerate(p["visits"]):
            if v is None:
                continue
            if observed[j] and load_images:
                images[j] = self.read_image(v)
            elif future and j > landmark and load_images:
                target_images[j] = self.read_image(v)
                fm[j] = True
        raw, cm = self.clinical_raw(p, landmark)
        norm = self.normalization
        clinical = ((raw - np.array(norm["mean"], dtype=np.float32)) /
                    np.array(norm["std"], dtype=np.float32))
        clinical[~cm] = 0
        aid, when = self.arm(p, landmark)
        inp = Inputs(images, observed, torch.tensor(landmark), torch.from_numpy(clinical),
                     torch.from_numpy(cm), torch.tensor(aid), torch.tensor(when), torch.arange(4) > landmark)
        y = p.get("target", {}).get("pcr") if labels else None
        if y is not None and (type(y) not in {int, float} or y not in (0, 1)):
            raise ValueError("pCR label must be binary or null")
        sup = Targets(target_images, fm, torch.tensor(float(y or 0)), torch.tensor(y is not None))
        return inp, sup, canonical_key(p["patient_key"])

    def fingerprint(self):
        assets = [file_identity(self.resolve(v["image"])) for p in self.patients
                  for v in p["visits"] if v is not None]
        return {"schema": "response_jepa_data_identity_v1",
                "manifest": record_metadata(self.manifest), "assets": assets,
                "image_normalization": record_metadata(self.manifest.get("image_normalization")),
                "partitions": {s: sorted(canonical_key(self.patients[i]["patient_key"]) for i in self.splits[s])
                               for s in ("train", "val", "test")}}

    def audit(self, verify_pixels=False):
        result = {"patients": dict((s, len(self.splits[s])) for s in ("train", "val", "test")),
                  "visits": {}, "image_normalization": record_metadata(self.manifest.get("image_normalization")),
                  "clinical_normalization": record_metadata(self.normalization), "verified_pixels": bool(verify_pixels),
                  "limitations": ["JSON declarations cannot establish actual ROI provenance or physical alignment",
                                  "Stage indices cannot verify that T3 precedes pathology",
                                  "Preprocessed image normalization must be re-fitted for a new CV split"]}
        result["labels"] = {}
        for s in ("train", "val", "test"):
            labels = [self.patients[i].get("target", {}).get("pcr") for i in self.splits[s]]
            if any(y is not None and (type(y) not in {int, float} or y not in (0, 1)) for y in labels):
                raise ValueError("Invalid pCR label in data audit")
            result["labels"][s] = {"positive": sum(y == 1 for y in labels),
                                    "negative": sum(y == 0 for y in labels),
                                    "missing": sum(y is None for y in labels)}
            result["visits"][s] = [sum(self.patients[i]["visits"][j] is not None for i in self.splits[s])
                                     for j in range(4)]
        for p in self.patients:
            for v in p["visits"]:
                if v is not None:
                    if not self.resolve(v["image"]).is_file():
                        raise FileNotFoundError("A declared MRI source file is missing")
                    if verify_pixels:
                        self.read_image(v)
        return result

class PrefixDataset(Dataset):
    def __init__(self, store, tasks, *, future=True, labels=True, single=False, load_images=True,
                 sample_weighting="sample", profile="dynamic", phase="joint"):
        self.store, self.tasks = store, list(tasks)
        self.options = dict(future=future, labels=labels, single=single, load_images=load_images)
        if sample_weighting not in {"sample", "patient_equal"}:
            raise ValueError("Unknown prefix sample weighting")
        self.sample_weights = None
        if sample_weighting == "patient_equal":
            if phase == "A":
                raise ValueError("Patient-equal prefix weights are not visit weights")
            if not self.tasks or len(set(self.tasks)) != len(self.tasks):
                raise ValueError("Patient-equal tasks must be nonempty and contain no duplicate prefixes")
            splits = {store.patients[i]["split"] for i, _ in self.tasks}
            if len(splits) != 1:
                raise ValueError("Patient-equal tasks must belong to one split")
            # Reference the complete legal population, not an epoch subset or a
            # resumed suffix. E[weight * loss] over all M rows is patient-equal.
            eligible = {i: _prefix_stages(store, i, profile, phase)
                        for i in store.splits[next(iter(splits))]}
            eligible = {i: stages for i, stages in eligible.items() if stages}
            total = sum(len(stages) for stages in eligible.values())
            count = len(eligible)
            if any(i not in eligible or j not in eligible[i] for i, j in self.tasks):
                raise ValueError("Task is not a legal prefix for its profile and phase")
            self.sample_weights = tuple(total / (count * len(eligible[i])) for i, _ in self.tasks)
    def __len__(self):
        return len(self.tasks)
    def __getitem__(self, index):
        sample = self.store.sample(*self.tasks[index], **self.options)
        return sample if self.sample_weights is None else (*sample, self.sample_weights[index])

class VisitDataset(Dataset):
    def __init__(self, store, tasks):
        self.store, self.tasks = store, list(tasks)
    def __len__(self):
        return len(self.tasks)
    def __getitem__(self, index):
        i, j = self.tasks[index]
        p = self.store.patients[i]
        y = p.get("target", {}).get("pcr")
        if y is not None and y not in (0, 1):
            raise ValueError("Invalid pCR label")
        return {"image": self.store.read_image(p["visits"][j]), "stage": torch.tensor(j),
                "label": torch.tensor(float(y or 0)), "label_mask": torch.tensor(y is not None),
                "key": canonical_key(p["patient_key"])}

def collate_prefix(samples):
    widths = {len(sample) for sample in samples}
    if widths not in ({3}, {4}):
        raise ValueError("Prefix samples must consistently contain inputs, targets, key and optional weight")
    inp = Inputs(**{f.name: torch.stack([getattr(x[0], f.name) for x in samples]) for f in fields(Inputs)})
    sup = Targets(**{f.name: torch.stack([getattr(x[1], f.name) for x in samples]) for f in fields(Targets)})
    weight = torch.as_tensor([x[3] for x in samples], dtype=torch.float32) if widths == {4} else None
    if weight is not None and (not torch.isfinite(weight).all() or (weight <= 0).any()):
        raise ValueError("Prefix sample weights must be finite and positive")
    return Batch(inp, sup, tuple(x[2] for x in samples), weight)

def _prefix_stages(store, i, profile, phase):
    stages = store.allowed(i, profile)
    if phase == "B":
        # Match world-only task eligibility by excluding prefixes with no future
        # MRI. This helper is shared by planning and population weighting.
        stages = [j for j in stages if any(v is not None for v in store.patients[i]["visits"][j+1:])]
    return stages

def planned_tasks(store, cfg, split, epoch=0):
    """A: one visit/patient, grouped by stage. B/C/joint: sampled or all prefixes.
    All-prefix training shuffles every eligible (patient, landmark) once per epoch.
    Validation is fixed and covers every eligible prefix, then patient-averages.
    """
    t = cfg.train
    ids = list(store.splits[split])
    rng = random.Random(t.seed + 100003 * (epoch + 1))
    if split == "train" and t.prefix_sampling != "all_prefixes":
        rng.shuffle(ids)
    tasks = []
    for i in ids:
        p = store.patients[i]
        if t.phase == "A":
            stages = [j for j, v in enumerate(p["visits"]) if v is not None]
            # Stable val visit, rotating training visits; never clinical future input.
            offset = int(hashlib.sha256(canonical_key(p["patient_key"]).encode()).hexdigest()[:8], 16)
            j = stages[((epoch if split == "train" else 0) + offset) % len(stages)]
            tasks.append((i, j))
            continue
        stages = _prefix_stages(store, i, t.profile, t.phase)
        # B needs targets; this selection is NOT used for the task model's inference.
        if not stages:
            continue
        if split == "train":
            if t.prefix_sampling == "all_prefixes":
                tasks.extend((i, j) for j in stages)
            else:
                offset = int(hashlib.sha256(canonical_key(p["patient_key"]).encode()).hexdigest()[:8], 16)
                j = rng.choice(stages) if t.prefix_sampling == "random" else stages[(epoch + offset) % len(stages)]
                tasks.append((i, j))
        else:
            tasks.extend((i, j) for j in stages)
    if not tasks:
        raise ValueError(f"No eligible {t.phase}/{t.profile} tasks in {split}")
    if t.phase == "A":
        # Same-stage physical batches are necessary for patient-axis variance.
        tasks.sort(key=lambda x: x[1])
    elif split == "train" and t.prefix_sampling == "all_prefixes":
        rng.shuffle(tasks)
    return tasks
