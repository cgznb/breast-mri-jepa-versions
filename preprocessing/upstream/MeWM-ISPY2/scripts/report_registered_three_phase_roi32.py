"""Independently replay aggregate metrics and summarize the integration study."""

from __future__ import annotations

import csv
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "runs/registered_three_phase_roi32_v1"
PHASES = ("pre_aqc0", "first_post_aqc1", "metadata_late")
METRICS = ("l1", "valid_l1", "t0_roi_l1", "ssim_clamped_neg1_pos3", "lpips_three_slices_sum")


def read(path):
    return json.loads(path.read_text())


def identity(path):
    path = Path(path).resolve()
    state = path.stat()
    return {"path": str(path), "size_bytes": state.st_size, "mtime_ns": state.st_mtime_ns}


def replay(rows, group, keys, metrics):
    for metric in metrics:
        patients = defaultdict(list)
        for row in rows:
            if all(row[key] == group[key] for key in keys) and row[metric] not in (None, ""):
                patients[row["patient_id"]].append(float(row[metric]))
        result = statistics.mean(statistics.mean(values) for values in patients.values()) if patients else None
        expected = group[metric]
        if result is None or expected is None:
            assert result is expected
        else:
            assert abs(result - expected) < 1e-12, (group, metric, result, expected)


def main():
    audit, probe = read(ROOT / "vq_audit/summary.json"), read(ROOT / "vq_mixed_probe/summary.json")
    manifest, smoke = read(ROOT / "inventory.json"), read(ROOT / "smoke/fm-cpu.json")
    protocol = probe["protocol"]
    assert identity(protocol["base_codec"]["path"]) == protocol["base_codec"] == audit["codec"]
    assert not manifest["missing_sources"]
    assert smoke["status"] == "passed" and smoke["recovery_max_error"] == smoke["ema_recovery_max_error"] == 0
    visits = {row["visit_id"]: row for row in manifest["visits"]}
    train = {visits[visit]["patient_id"] for visit in protocol["train_visits"]}
    val = {visits[visit]["patient_id"] for visit in protocol["val_visits"]}
    assert len(train) == 32 and len(val) == 16 and train.isdisjoint(val)
    assert all(visits[visit]["fold"] == "train" for visit in protocol["train_visits"])
    assert all(visits[visit]["fold"] == "val" for visit in protocol["val_visits"])
    tables = {}
    for stage in ("vq_audit", "vq_mixed_probe"):
        for name in ("phases", "enhancement"):
            with (ROOT / stage / (name + ".csv")).open() as handle:
                tables[stage, name] = list(csv.DictReader(handle))
    checks = 0
    for row in audit["phase_metrics"]:
        replay(tables["vq_audit", "phases"], row, ["phase"], METRICS)
        checks += len(METRICS)
    for name, stage in (("frozen", "vq_audit"), ("fine_tuned", "vq_mixed_probe")):
        chosen = [row for row in tables[stage, "phases"] if row["visit_id"] in protocol["val_visits"]]
        assert len(chosen) == 3 * len(protocol["val_visits"])
        for row in probe[name]:
            replay(chosen, row, ["phase"], METRICS)
            checks += len(METRICS)
        chosen = [row for row in tables[stage, "enhancement"] if row["visit_id"] in protocol["val_visits"]]
        for row in probe[name + "_enhancement"]:
            fields = ("mae", "relative_mae", "signed_bias", "spatial_correlation")
            replay(chosen, row, ["enhancement", "region"], fields)
            checks += len(fields)
    original_source = protocol["runtime_source"]
    archive = ROOT / "vq_mixed_probe/source_snapshot.py"
    if not archive.exists():
        assert identity(original_source["path"]) == original_source
        shutil.copy2(original_source["path"], archive)
    assert archive.stat().st_size == original_source["size_bytes"] and archive.stat().st_mtime_ns == original_source["mtime_ns"]
    frozen = {row["phase"]: row for row in probe["frozen"]}
    tuned = {row["phase"]: row for row in probe["fine_tuned"]}
    if (ROOT / "fm_evaluation/COMPLETE.json").exists():
        finished = read(ROOT / "fm_evaluation/COMPLETE.json")
        trained = read(ROOT / "fm/COMPLETE.json")
        assert finished["status"] == trained["status"] == "passed"
        training_status = (f"Formal FM completed {trained['updates']} optimizer updates and "
                           f"{trained['sampled_pairs']} sampled pairs; final evaluation completed at "
                           f"{finished['updated_at']} across all 538 validation pairs / 102 patients. "
                           "See [forecast results and limitations](fm_evaluation/review/report.md) and "
                           "[forecast comparisons](fm_evaluation/review/index.html).")
    else:
        training_status = ("Current training state is recorded in [controller progress](progress.json) "
                           "and [FM progress](fm/progress.json); final forecast evaluation remains pending.")
    lines = ["# Registered three-phase ROI32 integration and VQ transfer", "",
             training_status, "",
             "The small registered pipeline now supports pre / first-post / metadata-selected late phases "
             "on one fixed T0 crop. MRI shape is 3 x 32 x 128 x 128, continuous latent shape is "
             "24 x 8 x 32 x 32, and the symmetric noise/source state has 48 channels. "
             "The existing single-channel codec and the original small velocity feature widths are retained.", "",
             "All 101 missing registered first-post files were restored into the new experiment after 202 "
             "existing pre/late anchor files matched the remote registered source. No cohort exclusions were added: "
             "764/102 patients, 2657/382 visits and 3462/538 train/validation pairs.", "",
             "## Full frozen-codec audit", "",
             "102 validation patients / 382 visits / 1146 phase reconstructions; no codec update. "
             "Patient-macro metrics use one mean per patient before averaging patients.", "",
             "| Phase | Foreground MAE | T0-mask MAE | SSIM [-1,3] |",
             "|---|---:|---:|---:|"]
    full = {row["phase"]: row for row in audit["phase_metrics"]}
    for phase in PHASES:
        row = full[phase]
        lines.append(f"| {phase} | {row['valid_l1']:.6f} | {row['t0_roi_l1']:.6f} | {row['ssim_clamped_neg1_pos3']:.6f} |")
    lines += ["", "## Matched 100-update mixed-phase probe", "",
              "A copy of the selected VQ, codebook and discriminators was fine-tuned with the original full GAN "
              "objective, fixed shared intensity normalization and fresh optimizers at 1e-5. "
              "Training used 32 patients / 64 visits, one pre/first/late triplet per update. "
              "The fixed 100-update checkpoint was evaluated on 16 other validation patients / 61 visits. "
              "Both sides below use that same subset and CPU FP32. The original codec is unchanged.", "",
              "| Phase | Frozen foreground MAE | Tuned foreground MAE | Frozen T0-mask MAE | Tuned T0-mask MAE | ROI change |",
              "|---|---:|---:|---:|---:|---:|"]
    for phase in PHASES:
        a, b = frozen[phase], tuned[phase]
        delta = 100 * (b["t0_roi_l1"] / a["t0_roi_l1"] - 1)
        lines.append(f"| {phase} | {a['valid_l1']:.6f} | {b['valid_l1']:.6f} | {a['t0_roi_l1']:.6f} | {b['t0_roi_l1']:.6f} | {delta:+.2f}% |")
    lines += ["", "| Enhancement in T0 mask | Frozen MAE | Tuned MAE | Change |", "|---|---:|---:|---:|"]
    for first in probe["frozen_enhancement"]:
        if first["region"] != "t0_roi":
            continue
        second = next(row for row in probe["fine_tuned_enhancement"] if row["region"] == "t0_roi" and row["enhancement"] == first["enhancement"])
        change = 100 * (second["mae"] / first["mae"] - 1)
        lines.append(f"| {first['enhancement']} | {first['mae']:.6f} | {second['mae']:.6f} | {change:+.2f}% |")
    lines += ["", "## Initial codec decision", "",
              "Retain the original frozen DCE0 codec for the first registered three-phase generation baseline. "
              "This short comparison shows only small changes and no consistent improvement of T0-region error "
              "across all three phases. It does not justify replacing the original weights with the probe checkpoint. "
              "The single-channel architecture can be shared across phases; separate per-phase codecs or "
              "from-scratch training are not required to connect this pipeline. A longer, full-cohort mixed-phase "
              "fine-tuning study could still be useful and has not been ruled out.", "",
              "## Interpretation limits", "",
              "This short mixed-phase comparison informs the next codec experiment; it is not a full training "
              "study, a comparison against from-scratch training or proof that three separate codecs are needed. "
              "The evaluation cohort remains development data. T0 masks are model predictions and can be clipped.", "",
              "SSIM clips to [-1,3] and the inherited LPIPS input clips to [-1,1]. Strong enhancement can exceed "
              "those windows, so unclipped regional and enhancement errors are needed to assess signal preservation.", "",
              "The real-data generator smoke passed forward/backward, exact model/EMA recovery, Heun25 and finite "
              "three-phase decoding. This is an interface check using a fresh generator, not a forecast-quality result. "
              "Those checks describe the initial integration audit. Full-cohort latent extraction and the batch64 GPU gate "
              "subsequently passed, and formal FM started on 2026-09-16. " + training_status, "",
              f"Independent CSV replay passed {checks} aggregate metric checks and verified split separation, "
              "the unchanged original codec, complete input inventory and exact smoke recovery.", "",
              "[Paired frozen/mixed figures](vq_mixed_probe/figures/index.html) | "
              "[Frozen reconstruction gallery](vq_audit/index.html) | [Full audit](vq_audit/report.md) | "
              "[Mixed-phase results](vq_mixed_probe/summary.json)", ""]
    (ROOT / "report.md").write_text("\n".join(lines))
    verification = {"status": "passed", "aggregate_metric_checks": checks, "original_codec_unchanged": True,
                    "patient_split_overlap": 0, "remaining_missing_sources": 0, "smoke_recovery_error": 0.0,
                    "probe_source_archive": identity(archive), "probe_original_source": original_source}
    (ROOT / "integration_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification))


if __name__ == "__main__":
    main()
