# Symm-FM TR-JEPA v0.2 — audited recovery release

This is the complete standalone three-phase workflow, not a replacement of every
historical experiment in `cgznb/symm-fm`. The original VQ interface, independent
medical JEPA, treatment/dynamics modules and multi-layer 3D DiT are retained.

**中文入口：[故障审计与修复报告](docs/AUDIT_AND_FIXES_ZH.md) · [参数与恢复运行手册](docs/TRAINING_RECOVERY_ZH.md)**

The previous release's functional tests did not establish medical image quality.
This release fixes demonstrated preprocessing/support/validation issues and adds
failure localization. There is still **no validation on the user's MRI, clinical
labels, actual checkpoint, or production GPU**.

```bash
pip install -e '.[test]'
python -m pytest -q
bash scripts/run_smoke.sh /absolute/path/to/new_smoke_run
python -m trjepa.cli --help
```

Start with `diagnose` on an existing cache/checkpoint. A normalization mismatch
requires rebuilding inputs, not increasing training duration. Do not overwrite
old checkpoints or relabel an old cache as a new coordinate system.

Configuration choices:

| File | Intended use |
|---|---|
| `configs/optimized.yaml` | Fresh B→C→D; same 384-wide / 12-layer main architecture |
| `configs/warmstart_existing.yaml` | Explicit old-weight initialization, same correct cache, original time coordinates |
| `configs/renderer_only.yaml` | Fresh D, structured-condition direct CFM, no JEPA features |
| `configs/renderer_only_residual.yaml` | Same isolation arm, requires correspondence QC |
| `configs/renderer_only_matched.yaml` | Matches full model’s D transport mode and update budget; report B/C costs separately |
| `configs/overfit.yaml` | Deliberate eight-training-pair fit check, not a performance experiment |
| `configs/small_data.yaml` | 192-wide, 8-layer encoder/DiT variant; **not old-weight compatible** |
| `configs/microbatch1.yaml` | Explicitly disables cross-image variance regularizer for microbatch 1 |
| `configs/joint_ode.yaml` | Optional E with actual short differentiable ODE regularizer |
| `configs/formal_validation.yaml` | Full validation cohort instead of a small monitoring subset |
| `configs/smoke_v2.yaml` | CPU functional tests only; never real MRI training |

`best.pt` uses sampled-image validation in D/E for the new recipes.
`best_loss.pt` separately records the auxiliary-loss minimum. Exact resume and
weights-only warmstart have different contracts. E is **off by default** in the
real-data pipeline script.

The optional installer creates `extensions/trjepa_v2` and `run_trjepa_v2.py` beside
an existing v1 extension; it refuses overwrite. It does not push to GitHub.

Evidence is under `validation/`. Results in `validation/legacy_v1/` and documents
in `docs/legacy_v1/` are historical, not v2 validation. Reproduction and controlled
latent-fitting scripts are included. No patient data or pretrained medical weights
are distributed. Licenses and attribution remain in `LICENSE`, `licenses/` and
`docs/SOURCES.md`; this mixed-license release includes non-commercial restrictions.
