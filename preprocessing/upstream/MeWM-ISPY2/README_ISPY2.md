# MeWM-derived I-SPY2 DCE0 Forecasting

This worktree adapts the MeWM/DiffTumor latent world-model stack to longitudinal
I-SPY2 breast DCE MRI. It predicts the precontrast AQC DCE phase zero of a
future visit from phase zero of the source visit.

The implementation is intentionally separate from MeWM's GPT policy,
inverse-dynamics, HCC survival, and source-Mask feathering paths.

## Data Contract

Model inputs are exactly:

- `source_dce0`: robust-normalized single-channel `[1,128,128,128]` volume in
  `[-1,1]`
- `source_mask`: binary source tumor Mask on the same grid
- `action_text`: treatment-arm description
- `clinical_text`: age, HR, HER2, MP, and menopausal status only
- `delta_days`: Fourier-encoded source-to-future interval
- `stage_id`: `1`, `2`, or `3` for T0->T1, T1->T2, or T2->T3

Phase zero is parsed from the ordered manifest as `dce_paths[0]`; the complete
phase list must be contiguous and match `n_times`. Each visit uses its own
nonzero 0.5/99.5 percentiles before mapping to `[-1,1]`. This deliberately
removes scanner gain from the precontrast anatomy channel and must not be
interpreted as quantitative contrast enhancement. SER, source FTV, pCR, patient
ID inside clinical text, future Mask, target FTV, and target phase metadata are
not model conditions.

`infer-source` uses the source-only manifest loader. Target volumes are loaded
only by `evaluate`, after the complete prediction artifact set already exists.

## Model Stages

1. MRI VQGAN: single-channel DCE0 in `[-1,1]`, 8D codes, 16,384-entry codebook,
   and 4x spatial compression (`128^3 -> 8x32^3`). The pinned DiffTumor CT
   autoencoder initializes all 71 MRI autoencoder/codebook states, then must be
   fine-tuned on I-SPY2 MRI. The decoder returns its final convolution without
   `tanh` or clamp, matching the original CT architecture. VGG-LPIPS receives
   these values through an unchecked `normalize=False` path. Reconstructions
   outside `[-1,1]` remain computable, but their perceptual scores are outside
   LPIPS's calibrated image domain. Training uses CUDA FP16 mixed precision,
   LR `1.875e-5`, norm clipping at 1.0, the checkpoint's 60k/120k schedulers,
   replicate padding, GroupNorm `eps=1e-6`, low-frequency code restart, random
   three-axis slices, and summed LPIPS. The original BatchNorm 2D/3D PatchGAN
   structure and hinge loss are retained, but both discriminators are randomly
   initialized and learned only from MRI; CT discriminator weights are never
   loaded.
2. Conditional latent diffusion: 8 noisy future-latent channels plus 8 masked
   source-latent channels and one `{0,1}` Mask channel. Both image paths use the
   VQGAN's continuous pre-quantization encoder output, normalized without clamp
   by the frozen codebook's global minimum and maximum. The 17-channel denoiser
   predicts epsilon with L1 loss over 200 cosine steps and EMA 0.995. Inference
   runs all 200 ancestral DDPM reverse steps, clamps only each estimated clean
   normalized latent to `[-1,1]`, inverse-scales the final latent, requantizes it
   to the nearest codebook entries, and returns the raw linear-decoder tensor.
3. DynUNet: trained on real DCE0/Mask pairs and applied independently to each of
   eight generated future DCE0 samples.

Action and clinical text are encoded separately by one shared revision-locked
NF4 MedGemma q/v-LoRA tower. Their projected 768D vectors are fused with 128D
day features and a 64D stage embedding using the fixed 1728->1024->512 FiLM MLP.

## Commands

Run commands from this worktree with `PYTHONPATH=.`:

```bash
python -m mewm_ispy2 preflight \
  --config configs/ispy2_current.yaml \
  --prepare-sample

python -m mewm_ispy2 train-vqgan \
  --config configs/ispy2_current_ct_aligned.yaml \
  --ct-checkpoint /root/.cache/mewm_ispy2/difftumor/089ba0f3f94a7858603a55106791d7d977d7bc0b/AutoencoderModel.ckpt \
  --max-epochs 100 --devices 1

python -m mewm_ispy2 train-diffusion \
  --config configs/ispy2_current_ct_aligned_weak_gan_deep_b1.yaml \
  --vqgan-checkpoint /absolute/path/to/vqgan/checkpoints/composite/best-composite.ckpt \
  --max-epochs 200 --devices 1

python -m mewm_ispy2 train-segmenter \
  --config configs/ispy2_current.yaml \
  --max-epochs 200 --devices 1

python -m mewm_ispy2 infer-source \
  --config configs/ispy2_current.yaml \
  --transition-id ISPY2-000101:T0->T1 \
  --vqgan-checkpoint /path/to/mri-vqgan.ckpt \
  --diffusion-checkpoint /path/to/diffusion.pt \
  --segmenter-checkpoint /path/to/segmenter.ckpt \
  --output-dir runs/predictions

python -m mewm_ispy2 evaluate \
  --config configs/ispy2_current.yaml \
  --prediction-dir runs/predictions/ISPY2-000101_T0_to_T1 \
  --output-dir runs/evaluation/ISPY2-000101_T0_to_T1

python -m mewm_ispy2 smoke --output-dir runs/smoke/manual
```

The registered configuration never falls back to current data. Preflight exits
nonzero until every locked visit referenced by the transition split has one of
`fixed_reference`, `rigid_fallback`, or `deformable`.

DCE0 uses new data/cache schemas and fresh `runs/*_dce0` output roots. A
last-phase cache or MRI checkpoint cannot be resumed into this pipeline. MRI
VQGAN checkpoints bind the complete CT-aligned training contract, including the
linear decoder, unchecked LPIPS, and fresh MRI discriminators. Checkpoints
created with either the previous bounded decoder or the earlier simplified
training behavior are rejected. Diffusion training and inference additionally
require the VQGAN checkpoint's recorded data-contract SHA256 and backend to
match the transition records loaded by the selected diffusion configuration.
The check runs before MedGemma or the denoiser is allocated.

## Checkpoints and Outputs

Diffusion checkpoints contain the denoiser, EMA denoiser, MedGemma LoRA weights,
shared text projection, stage embedding, and FiLM fusion. They exclude the 4B
base model and frozen VQGAN, and bind MedGemma revision, MRI VQGAN SHA256, data
contract SHA256, backend, `continuous_codebook_minmax_v1` latent contract,
`ancestral_ddpm` sampler, diffusion timesteps, and EMA decay. Schema v3 rejects
the previous raw-quantized/DDIM identity. Training resume uses Lightning's
standard checkpoint path, restoring optimizer, global step, and callback state
while rejecting any identity or schedule mismatch.

Each prediction contains eight DCE volumes, eight Masks, eight Mask probability
volumes, mean DCE, mean Mask probability, entropy, and `prediction.json`.
Evaluation writes per-sample CSV and summary JSON with whole-ROI and tumor-local
image metrics, Dice, HD95, surface Dice, volume/growth errors, best-of-eight, and
uncertainty-error correlation.
