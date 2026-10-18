# Longitudinal 3D Foundation Modeling for Neoadjuvant Breast Cancer Response Prediction from Serial DCE-MRI

Code for the paper *"Longitudinal 3D Foundation Modeling for Neoadjuvant Breast Cancer Response Prediction from Serial DCE-MRI"* (AMAI 2026, MICCAI workshop).

We pair a frozen Pillar-0 3D medical-imaging foundation encoder with a trainable Temporal Dynamics Network (TDN) to predict pathologic complete response (pCR) from serial DCE-MRI acquired at
four clinical timepoints.

![method](assets/method.png)

## Overview

- **Encoders (frozen).** Each timepoint's 3D DCE-MRI volume is encoded independently into a 1152-D
  embedding by Pillar-0 (`src/pillar.py`). A 2D ViT-MAE baseline (`src/vit2d.py`) produces 768-D
  embeddings from tumor-centered slices.
- **Temporal Dynamics Network** (`src/tdn.py`). Projects each timepoint's embedding into a compact
  disease signature, adds an elapsed-time positional encoding, and reads the sequence with a
  Transformer over `[query, T0, T1, T2, T3, clinical]` (a key-padding mask excludes missing visits).
  The query output is a residual logit fused with a fixed logistic-regression clinical prior.
- **Only the TDN and its prediction head are trained; the encoders remain frozen.**

## Installation

```bash
git clone https://github.com/omarftt/longitudinal_temporal_pillar.git && cd longitudinal_temporal_pillar
pip install -e . 
```

Pillar-0 weights are gated: request access to `YalaLab/Pillar0-BreastMRI` on Hugging Face and run
`hf auth login` before running `extract_pillar.py`.

## Data

The tabular metadata (`data/metadata_enriched.csv`) and the predefined train/val/test split
(`data/splits/`) are committed, so only the imaging and the Pillar-0 weights need to be downloaded —
see [`data/README.md`](data/README.md) for links and the expected folder layout.

## Usage


**1 — Metadata and splits (optional; already committed)**

`data/metadata_enriched.csv` and `data/splits/` ship with the repo, so you can go straight to step 2.
To rebuild them from source instead:

```bash
# add real elapsed-days (days_T0..T3) from the TCIA image manifests
python scripts/add_elapsed_days.py --csv data/BreastDCEDL_ISPY2_metadata_longitudinal.csv \
    --manifests data/ISPY2-719-Image-manifest.xlsx data/ACRIN-6698-Image-manifest.xlsx \
    --out data/BreastDCEDL_ISPY2_metadata_longitudinal.csv

# build the segmentation-free enriched tabular (17-D clinical/treatment + phase + days columns)
python scripts/build_metadata.py --csv data/BreastDCEDL_ISPY2_metadata_longitudinal.csv \
    --ispy2-xlsx data/ISPY2-metadata.xlsx --out data/metadata_enriched.csv

python scripts/make_splits.py --csv data/metadata_enriched.csv --out data/splits
```

**2 — Extract frozen embeddings**

```bash
python scripts/extract_pillar.py --nifti data/BreastDCEDL_ISPY2 \
    --metadata-csv data/metadata_enriched.csv --splits-dir data/splits --out data/emb_global      # 3D
python scripts/extract_vit2d.py  --zip data/BreastDCEDL_ISPY2.zip \
    --csv data/metadata_enriched.csv --splits-dir data/splits --out-emb data/emb_vit2d            # 2D
```

**3 — Train and evaluate**

```bash
python scripts/run_experiments.py --global-dir data/emb_global \
    --metadata-csv data/metadata_enriched.csv --splits-dir data/splits --output-dir results/exp_3d
python scripts/run_experiments.py --global-dir data/emb_vit2d \
    --metadata-csv data/metadata_enriched.csv --splits-dir data/splits --output-dir results/exp_2d
python scripts/clinical_baseline.py --metadata-csv data/metadata_enriched.csv \
    --splits-dir data/splits --output-dir results/clinical
```

Reported numbers are the mean ± std over multiple random seeds; rerun `run_experiments.py` with
different `--seed` values into the same `--output-dir`.

**4 — Aggregate**

```bash
python scripts/aggregate.py --dir results/exp_3d
python scripts/aggregate.py --dir results/exp_2d
python scripts/aggregate.py --dir results/clinical
```

## Local MeWM I-SPY2 reproduction

This checkout includes an adapter for the registered Strict-A data already used by
`/root/autodl-tmp/MAM/MeWM-ISPY2-DCE0-BiFlowNet-RF`. The primary path recovers real
pre/early/late registered DCE phases and preserves the established 690-patient local
pCR split (`529/77/84`). It does not pass the BiFlowNet `[DCE0,SER]` tensor directly
to Pillar-0 or silently treat SER as a DCE phase.

Prepare and preflight the local cohort with:

```bash
python scripts/prepare_mewm_ispy2.py --config configs/mewm_ispy2_registered.yaml
python scripts/extract_pillar_mewm.py --config configs/mewm_ispy2_registered.yaml \
    --dry-run --limit 1 --device cpu
```

See [`docs/mewm_ispy2_reproduction.md`](docs/mewm_ispy2_reproduction.md) for the
data contract, gated Pillar-0 setup, full extraction/training commands, expected
counts, and the explicitly separated DCE0-only ablation.

The fixed `778/98/102` full978 experiment also supports two audited temporal
training strategies: four independent depth-specific models and one model trained
on every unique contiguous prefix. See
[`docs/full978_dual_temporal_strategies.md`](docs/full978_dual_temporal_strategies.md)
for the exact missing-visit policy, validation-only regularization selection,
commands, and real/generated test results.


<!-- ## Citation

```bibtex
@inproceedings{titocruz2026longitudinal,
  title     = {Longitudinal 3D Foundation Modeling for Neoadjuvant Breast Cancer Response Prediction from Serial DCE-MRI},
  author    = {Tito Cruz, Fidel Omar and Ghafouri, Neda and Wang, Zengyan and Khosravi, Pegah and Tian, Yu and Chen, Chen},
  booktitle = {Applications of Medical AI (AMAI) Workshop, MICCAI},
  year      = {2026},
}
``` -->

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
