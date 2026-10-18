"""Small-data controls: regularized logistic regression and a frozen-encoder linear probe."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from .data import Store, collate_prefix, canonical_key
from .engine import load_for_inference, write_json, write_csv
from .metrics import classification


def fit_probe(x, y, splits, *, output, c_values=(.001, .01, .1, 1., 10.), keys=None, landmark=0, evaluate_test=False):
    x, y = np.asarray(x), np.asarray(y)
    split = np.asarray(splits)
    tr, va, te = split == "train", split == "val", (split == "test") & evaluate_test
    if not tr.any() or not va.any() or len(np.unique(y[tr])) != 2:
        raise ValueError("Logistic controls need both classes in train and a nonempty val split")
    candidates = []
    best, selected = float("inf"), None
    for c in c_values:
        model = Pipeline([("scale", StandardScaler()),
                          ("logistic", LogisticRegression(C=c, solver="lbfgs", max_iter=3000))])
        model.fit(x[tr], y[tr])
        score = classification(y[va], model.predict_proba(x[va])[:, 1])
        candidates.append({"C": c, "validation": score})
        if score["nll"] < best:
            best, selected = score["nll"], model
    report = {"selection": "val NLL; scaling fitted on train only", "candidates": candidates,
              "selected_C": selected.named_steps["logistic"].C,
              "val": classification(y[va], selected.predict_proba(x[va])[:, 1]),
              "test": classification(y[te], selected.predict_proba(x[te])[:, 1]) if te.any() else None,
              "input_dimension": x.shape[1], "train_patients": int(tr.sum()),
              "coefficients_standardized": selected.named_steps["logistic"].coef_.tolist(),
              "intercept": selected.named_steps["logistic"].intercept_.tolist(),
              "scaler_mean": selected.named_steps["scale"].mean_.tolist(),
              "scaler_scale": selected.named_steps["scale"].scale_.tolist()}
    output = Path(output)
    write_json(output, report)
    if keys is not None:
        prediction = selected.predict_proba(x)[:, 1]
        for s in (("val", "test") if evaluate_test else ("val",)):
            rows = [{"patient_key": keys[i], "landmark": landmark, "label": float(y[i]),
                     "fused": float(prediction[i]), "direct": float(prediction[i]), "forecast": None}
                    for i in range(len(y)) if splits[i] == s]
            write_csv(output.with_name(output.stem + f"_{s}_predictions.csv"), rows)
    return report


def tabular(manifest, *, output, include_arm=False, allow_synthetic=False, image_shape=(32, 128, 128), evaluate_test=False):
    store = Store(manifest, image_shape, allow_synthetic=allow_synthetic)
    values, labels, splits, keys = [], [], [], []
    for i, p in enumerate(store.patients):
        if p["split"] == "test" and not evaluate_test:
            continue
        inp, sup, key = store.sample(i, 0, future=False, load_images=False)
        if not sup.label_mask:
            continue
        features = torch.cat((inp.clinical, inp.clinical_mask.float())).numpy()
        if include_arm:
            features = np.r_[features, np.eye(14, dtype=np.float32)[int(inp.arm_id)]]
        values.append(features)
        labels.append(float(sup.label))
        splits.append(p["split"])
        keys.append(key)
    return fit_probe(values, labels, splits, output=output, keys=keys, evaluate_test=evaluate_test)


@torch.no_grad()
def linear_probe(checkpoint, manifest, *, stage=0, output, device="cpu", image_cache=None,
                 allow_synthetic=False, evaluate_test=False):
    model, store, cfg, state = load_for_inference(checkpoint, manifest, device=device,
                                                image_cache=image_cache, allow_synthetic=allow_synthetic)
    features, labels, splits, keys = [], [], [], []
    for i, p in enumerate(store.patients):
        if p["split"] == "test" and not evaluate_test:
            continue
        if p["visits"][stage] is None or p["visits"][stage]["available_at"] > stage:
            continue
        inp, sup, key = store.sample(i, stage, future=False, single=True)
        if not sup.label_mask:
            continue
        batch = collate_prefix([(inp, sup, key)]).to(device)
        values = model.encode_selected(batch.inputs.images, batch.inputs.observed, teacher=True)
        features.append(values[0, stage].mean(0).float().cpu().numpy())
        labels.append(float(sup.label))
        splits.append(p["split"])
        keys.append(key)
    report = fit_probe(features, labels, splits, output=output, keys=keys, landmark=stage, evaluate_test=evaluate_test)
    report.update(stage=stage, feature_source="spatial mean of fixed target MRI encoder, no clinical or Arm",
                  target_coordinate_origin=state["target_coordinate_origin"],
                  note="T3 is a later-information reference, not a T0 competitor or guaranteed upper bound")
    write_json(output, report)
    return report
