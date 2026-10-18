# Source provenance and implementation boundaries

Reviewed on 2026-09-20. “Adapted” means code and architecture were inspected and translated to the 3D medical setting, not that upstream weights or results were reproduced. No claim is made that all implementations mentioned in the input report were reviewed or copied.

## 1. User repository: direct compatibility extraction

Repository: https://github.com/cgznb/symm-fm
Pinned commit: `92265d3b1749ae3b686f2089843c49da129fd4d2`.

Inspected implementation files:

- `workflows/first_post_three_phase/mewm_ispy2/three_phase_symmflow.py`: phase ordering, 3×8 latent channels, frozen codec, original SymmFlow sampling.
- `three_phase_all_pairs_data.py`: target-centered geometry, train-only statistics, latent cache and validation-only reference images.
- `three_phase_all_pairs_training.py`: source-read guard, fixed-seed sampling, copy-source/target-codec comparisons, checkpoint contracts.
- `three_phase_pilot_data.py`: exact `PAIR_FIELDS`, baseline clinical/treatment containers, true day intervals, target localization is supervision-only.
- `first_post_world_data.py`: original VQ checkpoint format and numerical contracts.
- `vqgan.py`: `VQGANConfig`, same-padding convolutions, `MRIEncoder`, `MRIDecoder`, EMA codebook, pre/post quantization.
- `src/ispy2_symmflow/models/velocity.py`: direct official MONAI conditional 3D U-Net use.
- `src/ispy2_symmflow/flow/path.py`: two-branch sigma=0 interpolation and analytic velocities.

`trjepa/codec.py` keeps original parameter names and geometry. This was checked with a synthetic checkpoint in the original marker/state_dict format; **no user's real checkpoint was available for binary or numerical comparison**. The optional MONAI backend reuses the architecture family, not the old input-dependent weights.

## 2. I-JEPA: adapted encoder/masked-predictor pattern

Official repository: https://github.com/facebookresearch/ijepa
Implementation: https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py
Inspected blob SHA: `2349fa40bf8fcfaceef9f828990fff7019709d11`.
Official CVPR 2023 codebase; license CC BY-NC 4.0.

Reused/adapted: QKV self-attention, pre-LN residual Transformer, mask-token predictor, position-aware target queries and EMA target concept. Changed: anisotropic 3D patches, 3D coordinates, medical masking, PyTorch SDPA, optional checkpointing. This is not the original 2D weight layout and no ImageNet checkpoint is silently inflated into 3D.

The implemented block mask sampler is a medical adaptation, not a verbatim copy of I-JEPA's complete multiblock collator. The weak variance regularizer is an explicit implementation addition; it is not attributed to the original I-JEPA objective.

## 3. DiT: adapted substantial generative backbone

Official repository: https://github.com/facebookresearch/DiT
Inspected file: https://github.com/facebookresearch/DiT/blob/main/models.py
License: https://github.com/facebookresearch/DiT/blob/main/LICENSE.txt (CC BY-NC 4.0).

Reused/adapted: Fourier time MLP, adaLN-Zero shift/scale/gates, multihead attention, ratio-4 feedforward block, zero initialized final prediction and non-affine final normalization. Changed: 3D patchification/unpatchification, semantic cross-attention, continuous clinical global modulation, immutable source anchor, velocity instead of diffusion-noise output. No claim of exact DiT checkpoint compatibility.

The core main configuration contains 12 DiT blocks, not a pair of convolution layers. Attention is implemented with PyTorch SDPA rather than requiring timm at runtime.

## 4. Set Transformer: concrete official MAB/ISAB/PMA adaptation

Official repository: https://github.com/juho-lee/set_transformer
Inspected file: https://github.com/juho-lee/set_transformer/blob/master/modules.py
Blob SHA: `3b7e698e14ef7b8e1ea0c419fe51d93e3ee04bf5`.
MIT copyright: Juho Lee, 2020. Full license retained in `licenses/SET_TRANSFORMER_MIT.txt`.

MAB projection/residual equations and sqrt(dim_V) scaling are retained; padding-validity masks are added. Two ISAB layers and PMA with four seeds form the drug-set encoder. Permutation invariance is covered by executable tests. The treatment-response query stack is a separate Perceiver-style cross-attention adaptation, not an assertion that Set Transformer itself models causal drug effects.

## 5. VLA-JEPA: design reference, not transplanted robotics code

Official repository: https://github.com/ginwind/VLA-JEPA
Paper identified by official repository: https://arxiv.org/abs/2602.10098

The relevant design influence is a source-available transition representation trained by predicting future state targets. The package does not import Qwen3-VL, RGB video patch stems, robot action trajectories or the full starVLA runtime. The report's medical adaptation is implemented separately. The official README lists Qwen3-VL-2B and V-JEPA2 checkpoints, illustrating why those inputs/weights cannot simply be renamed as a breast-MRI encoder.

## 6. Optional MONAI backend: actual dependency reuse, not locally validated

Official repository: https://github.com/Project-MONAI/MONAI
Pinned package API target: `monai==1.5.1`, `monai.networks.nets.DiffusionModelUNet`.
Public reference family: https://github.com/Project-MONAI/GenerativeModels
MONAI is used only when `flow.backbone: monai_unet3d` is explicitly selected.

The offline execution host could not install MONAI, so this path is **not included among the tested real-backbone executions**. It never silently falls back to a small fake network. The default, locally tested path is the substantial adapted 3D DiT.

## 7. What is not claimed

RIVER, D-JEPA, JEPA-T and other papers appear in the user's input report. They are not claimed as code donors in this release, because their actual source implementations were not sufficiently verified in this execution. No third-party trained medical segmentation model or medical perceptual metric is bundled. Licensing and publication status of a referenced repository does not establish novelty, clinical validity or checkpoint compatibility of this implementation.
