"""Replay original-volume metrics and all offline image packets independently."""

from __future__ import annotations

import base64
import io
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
from PIL import Image

from mewm_ispy2.registered_roi32_data import read_json, verify_identity, write_json
from mewm_ispy2.registered_three_phase_data import PHASES

ROOT = REPO / "runs/registered_three_phase_roi32_v1/fm_review_30cases_20260917"


def read_packet(path):
    source = path.read_text()
    prefix = "window.registerReviewData("
    assert source.startswith(prefix) and source.endswith(");\n")
    decoder = json.JSONDecoder()
    body = source[len(prefix):-3]
    key, index = decoder.raw_decode(body)
    assert body[index] == ","
    value = decoder.decode(body[index + 1:])
    return key, value


def replay_packet(path, key, originals, mask=None):
    recorded_key, packet = read_packet(path)
    assert recorded_key == key and set(packet["volumes"]) == set(originals)
    error, checked = 0.0, 0
    for name, original in originals.items():
        item = packet["volumes"][name]
        with Image.open(io.BytesIO(base64.b64decode(item["png"].split(",", 1)[1]))) as image:
            image.load()
            assert image.size == (384, 4096) and image.mode == "RGB"
            rgb = np.asarray(image, dtype=np.uint16)
        integers = rgb[:, :, 0] * 256 + rgb[:, :, 1]
        decoded = (integers.astype(np.float64) * item["scale"] + item["lower"]).reshape(3, 32, 128, 128)
        delta = float(np.max(np.abs(decoded - original)))
        assert delta <= item["scale"] / 2 + 1e-7
        assert math.isclose(delta, item["max_error"], abs_tol=1e-10)
        error = max(error, delta)
        checked += 1
    if mask is not None:
        with Image.open(io.BytesIO(base64.b64decode(packet["mask"].split(",", 1)[1]))) as image:
            assert np.array_equal(np.asarray(image).reshape(32, 128, 128) > 127, mask)
    return checked, error


def main():
    results = read_json(ROOT / "results.json")
    complete = read_json(ROOT / "COMPLETE.json")
    verify_identity(complete["results"])
    verify_identity(complete["protocol"])
    for field in ("checkpoint", "training_completion", "source"):
        verify_identity(results["protocol"][field])
    cases = results["cases"]
    assert len(cases) == len({case["record"]["patient_id"] for case in cases}) == 30
    assert all(case["record"]["split"] == "val" for case in cases)
    counts = Counter((case["record"]["earlier_stage"], case["record"]["later_stage"]) for case in cases)
    assert len(counts) == 6 and set(counts.values()) == {5}
    checked_metrics, encoded_volumes, max_error = 0, 0, 0.0
    for case in cases:
        name = case["record"]["pair_id"].replace(":", "_").replace("->", "_to_")
        directory, web = ROOT / "cases" / name, ROOT / "web/data" / name
        for identity in case["inference"]["artifacts"] + [case["reference"]]:
            verify_identity(identity)
        with np.load(directory / "reference.npz", allow_pickle=False) as data:
            originals = {key: data[key].copy() for key in ("source", "target", "source_vq", "target_vq")}
            mask, valid = data["mask"].copy(), data["valid"].copy()
        count, delta = replay_packet(web / "reference.js", name + "/reference", originals, mask)
        encoded_volumes += count; max_error = max(max_error, delta)
        assert 10 <= case["anchor_z"] <= 21
        for step in (10, 20, 30, 50):
            with np.load(directory / f"steps_{step}.npz", allow_pickle=False) as data:
                samples, mean, latent = data["samples"].astype(np.float32), data["mean"].copy(), data["latent"]
            assert samples.shape == (4, 3, 32, 128, 128) and np.isfinite(samples).all()
            assert latent.shape == (4, 24, 8, 32, 32) and np.isfinite(latent).all()
            assert np.array_equal(mean, samples.mean(0))
            for method, sample, value in (("sample", 0, samples[0]), ("mean", -1, mean)):
                for phase, phase_name in enumerate(PHASES):
                    recorded = next(row for row in case["image_metrics"] if row["steps"] == step
                                    and row["method"] == method and row["sample"] == sample and row["phase"] == phase_name)
                    for region, keep in (("foreground", valid[phase]), ("t0", mask)):
                        delta = float(np.abs(value[phase][keep].astype(np.float64) - originals["target"][phase][keep]).mean())
                        assert math.isclose(delta, recorded[region + "_target_mae"], abs_tol=1e-10)
                        checked_metrics += 1
            pack = {f"sample{i + 1}": samples[i] for i in range(4)}
            pack["mean"] = mean
            count, delta = replay_packet(web / f"step{step}.js", name + f"/step{step}", pack)
            encoded_volumes += count; max_error = max(max_error, delta)
    summary = read_json(ROOT / "web/summary.json")
    aggregate_checks = 0
    for name, keys in (("image_metrics", ("steps", "method", "phase")),
                       ("diversity", ("steps", "phase")), ("step_differences", ("steps", "phase"))):
        groups = defaultdict(list)
        for case in cases:
            for row in case[name]:
                groups[tuple(row[key] for key in keys)].append(row)
        for row in summary[name]:
            values = groups[tuple(row[key] for key in keys)]
            assert len(values) == row["observations"]
            for field, expected in row.items():
                if field in keys or field == "observations":
                    continue
                finite = [value[field] for value in values if value[field] is not None]
                if expected is None:
                    assert not finite
                else:
                    assert math.isclose(math.fsum(finite) / len(finite), expected, abs_tol=1e-10)
                aggregate_checks += 1
    assert encoded_volumes == 720
    report = {"status": "passed", "patients": 30, "main_forecast_volumes": 480,
              "original_volume_mae_checks": checked_metrics, "aggregate_checks": aggregate_checks,
              "decoded_display_volumes": encoded_volumes, "mask_packets": 30,
              "maximum_display_quantization_error": max_error, "raw_sample_mean_replays": 120}
    write_json(ROOT / "web/checks/data_verification.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
