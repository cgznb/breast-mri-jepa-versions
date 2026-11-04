> **v4范围说明（2026-11-03）**：下文为最初单seed44稿对冻结实现或队列的历史来源核查，保留用于定位来源事实，不定义本版结果。当前仅使用seed42、44、2026：各seed内先逐患者平均五折概率，再计算指标及Macro，最后等权平均三个seed的指标。主表为均值±样本SD（ddof=1）；患者区间和检验共享所有seed的患者抽样并固定已拟合模型。当前来源与配置/runtime一致性由`notes/source_audit.json`实时验证，60个run/180个登记报告。现行结果以`data/extended_provenance.json`、`data/subgroup_provenance.json`及最终构建记录为准。训练probe先在各snapshot坐标内算诊断，再折内/seed内汇总，不集成潜在状态。历史来源事实不能证明外部独立性或所有seed稳定性。

# Methods evidence and unresolved items

Scope: original Medium976 seed44 suite, read-only source audit. No new training or patient-data processing was performed. Only this note and `sections/methods.tex` were written by the methods agent.

## Frozen evidence root

`/projects/xts0030/public/cxx/planned_tasks/medium976_t0_t2_no_tf_cv5_multiseed_v1/`

- `input_configs/{direct_no_jepa,direct_readout,forecast_attention}_seed44_fold1.yaml`, and matching folds 2--5.
- `runtime/src/mri_vla_jepa/model.py`
- `runtime/src/mri_vla_jepa/encoder.py`
- `runtime/src/mri_vla_jepa/vq_encoder.py`
- `runtime/src/mri_vla_jepa/attention.py`
- `runtime/src/mri_vla_jepa/data.py`
- `runtime/src/mri_vla_jepa/arm.py`
- `runtime/src/mri_vla_jepa/training.py`
- `runtime/src/mri_vla_jepa/landmark_profiles.py`
- `runtime/src/mri_vla_jepa/constants.py`

Current manifest inspected: `/personal_fast_storage/VLA-PCRV1/data/registered_roi32_medium976_v1/manifest.json`. Only aggregate metadata were printed; no patient-specific content was copied into the manuscript.

## Verified model facts

1. All three primary models have `enable_world_model: true`, `fixed_transition_queries: true`, `history_delta: true`, and `world_residual: true`. NoJEPA does **not** remove transition queries, the world-predictor module, or the EMA teacher from the instantiated architecture. Its direct pCR queries can attend to transition-query tokens through fusion. The auxiliary objective is zero; the world predictor is not trained by a forecasting objective or used as an explicit pCR input.
2. `direct_no_jepa`: direct readout, mean spatial setting, `rollout_weight: 0`. `direct_readout`: identical direct-readout settings with `rollout_weight: 0.5`. `forecast_attention`: forecast readout, attention pooling, `rollout_weight: 0.5`. All three have `jepa_weight: 0`, `flow_weight: 0`, `reconstruction_weight: 0`, `variance_weight: 0`, and `prediction_training: hybrid`.
3. `compute_loss` branches on `prediction_training == teacher_forcing`. These runs take `_backbone(... classify=True)`, then autonomous rollout. Because `jepa_weight=0`, no teacher-forcing prediction/objective is computed even though the implementation option is named hybrid. The draft therefore describes autonomous representation supervision, not a positive JEPA next-state term.
4. In `model.py::_backbone`, all MRI context comes from online `encoder`; `_observed_teacher_states` obtains detached observed states for forecast initialization and current-reference subtraction. `_teacher_states` encodes available future scans only for target computation. `teacher_encoder.eval()` is retained during model training; all teacher parameters have no gradients.
5. `_autonomous_states` begins at latest actual observed stage and predicts through T3. It does not accept a supervision object. Future states replace future slots only, and previously predicted states are reused. Missing true future scans do not stop intermediate predictions.
6. `autonomous_rollout_loss` computes channel/spatial mean absolute error for each available future target, averages over each patient's target count, then averages over patients with at least one future target. Empty-target batches have a differentiable zero. No missing visit is imputed as a target. Unlike the disabled teacher-forced objective, this loss does not require an observed adjacent predecessor for each target.
7. `_source_history` supplies the latest available predecessor state for each source, with explicit validity. In autonomous rollout, this includes only observed states or previous predictions. Inactive predictor blocks are masked. Predictor attention is causal between transition blocks and unrestricted within blocks; a learned root token remains accessible.
8. EMA decay is 0.999 and `update_teacher` is called after the optimizer step. All online encoder components, including the continuous convolutional front end, are trainable with `encoder_training: finetune`.
9. MRI frontend is `vq_encoder`: a shared single-channel convolutional encoder returns **64 channels per phase**, bypassing `pre_quant` compression. Three phases concatenate to 192 channels, pool to [2,4,4], and project to 768. No quantizer/codebook, decoder, discriminator, codebook update, or downstream VQ reconstruction loss is present. `vq_embedding_dim: 8` does **not** mean these runs use eight latent channels per phase.
10. Each spatial/fusion/predictor Transformer has eight layers; d=768; 16 attention heads each have 64 channels, so Q/K/V internal width is 1024. Feed-forward width3072, dropout0.3, pre-LN, GELU. Avoid incorrectly computing 768/16=48 as the per-head width.
11. Four pCR queries are mean pooled. Eight transition queries per nonzero stage are retained. Full pCR features concatenate direct query mean, attention-pooled terminal T3 prediction, and attention-pooled `terminal - current observed EMA state`. `pcr_delta_reference: current`, not baseline T0. The same terminal-derived attention weights pool state and change. Scalar score is LN(d)+Linear(d,1,bias=False), zero-initialized, so pooling starts uniform.
12. Direct pCR head input width is768; full forecast head input width2304. Full vs Direct is a bundled forecast-readout comparison, **not** an attention-only ablation. Mean vs Attention would isolate pooling if included.
13. Full classification gradients propagate through the forecast terminal state into the predictor and transition-conditioning features. EMA reference and target states remain detached. Transformer-mixed attention does not constitute validated voxel/anatomic lesion localization.

## Verified clinical/time semantics

- Exact clinical order from `arm.py::CLINICAL_FEATURES` and manifest: `age_at_screening`, `hr_positive`, `her2_positive`, `mammaprint_binary`.
- `_fit_clinical` fits age mean/std using training partition; binary features retain0/1. Masked clinical inputs become zero and are concatenated with their availability indicators.
- Manifest aggregate:973 patients have all four clinical availability stages0;3 patients have unavailable age (None) and other fields at0.
- All976 arm records have `semantics=assigned_arm_scenario` and `known_at_stage=0`. This is an assumed baseline known-arm scenario, **not** prospectively verified availability or actual treatment exposure/dose history.
- There are13 released arm names and one unknown ID. `arm` embedding plus known-stage embedding is used; `arm_mask` gates the known-stage contribution, and unknown arm uses its unknown ID.
- MRI phases are `pre_aqc0`, `first_post_aqc1`, `metadata_late`. Do not call the late phase universally DCE2.
- `landmark_profiles.py` restricts sampled and evaluated prediction sources to actual T0/T1/T2 visits; T3 remains available as a future supervision target.
- The epoch loop visits each shuffled training patient once and samples one eligible landmark uniformly from that patient's actual T0--T2 visits. Validation visits all eligible prefixes; checkpoint criterion is patient-equal-prefix F1@0.5.

## Items deliberately unresolved in the draft

1. Original codec pretraining cohort, patient overlap and exact initialization lineage were not established from the frozen model/runtime alone. The paper has a visible TODO. Checkpoint code asserts `symm_world_codec_v2`, numeric contract `registered_ROI32_shared_training_DCE0_normalization`, and an exact architecture. Those checks do not prove pretraining independence or the original training data.
2. The continuous codec component header attributes architecture/checkpoint naming to MeWM via `cgznb/symm-fm` and local `trjepa/codec.py`, and states CC BY-NC4.0 terms. This is a code-provenance statement, not a verified identification of the supplied checkpoint's pretraining dataset. Bibliographic credit and permitted release contents must be finalized before submission.
3. Parent/root agent is handling cohort geometry, registration, exclusion accounting, optimization protocol, seed44 selection disclosure, statistics, and results. Methods does not claim external independent validation, a causal treatment model, image-generation efficacy, or anatomical interpretability.

## Source functions for targeted verification

`model.py`: `TimeCausalWorldPredictor`, `RawVLAJEPA.__init__`, `_backbone`, `_pcr_state_readout`, `_observed_teacher_states`, `_teacher_states`, `_source_history`, `_autonomous_states`, `autonomous_rollout_loss`, `compute_loss`, `update_teacher`.

`encoder.py`: `VQVisitMRIEncoder.forward`, `set_pretrained_trainable`.

`vq_encoder.py`: `SinglePhaseVQEncoder`, `VQContinuousFrontend.forward`, `initialize_pretrained`.

`attention.py`: `_WideSelfAttention`, `_WideTransformerLayer`, `build_transformer`.

`data.py`: `_clinical_values`, `_fit_clinical`, `batch`.

`training.py`: `_configure`, `train`, epoch sampler and optimizer/EMA update loop.
