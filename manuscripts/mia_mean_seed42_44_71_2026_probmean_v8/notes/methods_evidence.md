# v8 Methods evidence and unresolved items

Current scope: NoJEPA, DirectReadout, and ForecastMean; training seeds 42, 44, 71, 2026, five folds per model/seed. ForecastMean is the overall model. All 60 fitted models are principal comparisons; there is no additional model arm. Model and seed choices were changed after earlier holdout results had been available. Sources, config identity, runtime consistency, completed evaluations, and patient-level metric reconstruction are audited in `notes/source_audit.json` and `data/extended_provenance.json`. No training or preprocessing is performed for this manuscript revision.

## Aggregation and inference

Within each model/seed/landmark, average five fold probabilities per patient before computing metrics. Macro first averages T0/T1/T2 metrics within each seed; the final result averages four seed metrics. Sample seed SD uses ddof1 and is calculated from seed-specific Macro values for Macro. Do not average seed probabilities, fold metrics, or landmark SDs. The 60 runs share the same 175-patient/56-positive internal holdout; 180 registered landmark report identities do not enlarge the patient count.

The 2,000 shared-patient bootstrap intervals and 99,999 null-centered approximate bootstrap tests keep fitted models and seeds fixed. Principal family 9: three model pairs by Macro AUROC/AP/F1. Secondary family 69: landmark 27, other Macro 33, longitudinal 9. Separate Holm 9/Holm 69 and all-78 Holm sensitivity are reported. A separate numerical sensitivity repeats only the Mean-minus-NoJEPA T2 AUROC hypothesis with 999,999 draws and RNG 20261104; recomputing each seed AUROC before averaging and holding the other 77 p values fixed gives all-78 p=0.054872 rather than the uniform-table 0.04408. The uniform 78-test table is retained; the global T2 decision is numerically sensitive. No other model-comparison family is added. Retrospective model/seed selection and retraining uncertainty are outside these conditional calculations.

## Verified architecture and supervision contract

1. All three models retain the continuous image encoder, fusion network, transition-query slots, baseline clinical variables, and assigned-arm scenario. NoJEPA removes the future-state loss; it does not remove all transition-related modules.
2. NoJEPA uses direct readout and rollout weight 0. DirectReadout uses the same direct readout and rollout weight 0.5. ForecastMean uses forecast readout, arithmetic mean spatial pooling, and rollout weight 0.5. The single-step teacher-forced JEPA, reconstruction, flow, and variance-regularization weights are zero. The implemented future supervision is autonomous latent-rollout L1.
3. `_autonomous_states` begins at the latest actual observed stage and predicts through T3 without accepting true future states as predecessor inputs. Missing future scans remove their targets but do not stop autonomous intermediate predictions. Future EMA features are loss targets only.
4. `autonomous_rollout_loss` averages channel/spatial absolute error within available future targets, averages those targets within each patient, then averages patients with at least one target. Empty-target batches have a differentiable zero. It does not require continuous observed chains, unlike the archived diagnostic.
5. The target encoder is detached and in evaluation mode; EMA decay is 0.999 and updates follow optimizer steps. The observed current EMA state initializes forecasting and supplies the detached change reference. The online continuous encoder is fine-tuned.
6. The shared single-channel codec encoder returns 64 channels per DCE phase before compression. Three phases concatenate to 192 channels, pool to 2×4×4 spatial tokens, and project to width 768. No quantizer, codebook assignment, decoder, discriminator, or downstream image-reconstruction loss is used. `vq_embedding_dim:8` is not the number of active continuous channels per phase.
7. Spatial, fusion, and predictor Transformers each have eight layers, 16 attention heads of 64 channels, internal attention width 1024 projected back to residual width 768, feed-forward width 3072, dropout 0.3, pre-LN, and GELU. Shared self-attention remains present in this mean-readout experiment.
8. Four pCR-query features are averaged to the direct history vector q. For ForecastMean, each of the 32 terminal-state spatial tokens has pooling weight 1/32. The terminal-state vector is its arithmetic spatial mean; the change vector is the arithmetic mean of terminal minus current observed EMA state. No learned spatial pooling scorer weights these two summaries.
9. ForecastMean concatenates q, the terminal-state mean, and the change mean, giving head input width 2304. Direct heads have width 768. ForecastMean versus DirectReadout jointly changes forecast-feature access and head input width. It is not a classifier-width-matched or single-feature ablation.
10. The pCR gradient in ForecastMean reaches generated terminal states, predictor, and transition-conditioning features. Current EMA references and future EMA targets remain detached. Gradient routing does not establish biological interpretation or anatomical localization.

## Clinical and time semantics

- Clinical order: age at screening, HR positivity, HER2 positivity, and binary MammaPrint. Age mean/std come from current training patients. Binary fields retain0/1, and availability indicators accompany masked values.
- There are 13 released assigned-arm categories plus an unknown category. Assigned arm at T0 is a retrospective conditional scenario, not verified pre-randomization availability or delivered treatment exposure.
- MRI channels are precontrast, first postcontrast, and metadata-selected late phase; the late phase is not universally a fixed DCE acquisition index.
- Training samples one available T0/T1/T2 prefix per patient per epoch. Validation assesses available prefixes with patient-equal weighting. The best checkpoint uses patient-equal-prefix F1 at 0.5.
- T3 is never a reported classifier-input landmark; an actual T3 scan in training may supply a future target.
- Cohort split seed 2026 is fixed and distinct from the training-seed comparison. Current training/validation/holdout identities are disjoint within each fold, while different training folds overlap.

## Provenance and interpretation still unresolved

The supplied codec checkpoint and localization model have not been fully audited for patient overlap with the current internal holdout. Architecture, shape, and numeric-contract assertions do not establish pretraining independence. Applicable initialization credit and release permissions must be finalized by the authors. The present manuscript makes no external independent-validation, causal-treatment, image-generation, or anatomical-interpretability claim.

The original source-function navigation remains `model.py` (`_backbone`, `_pcr_state_readout`, `_observed_teacher_states`, `_teacher_states`, `_source_history`, `_autonomous_states`, `autonomous_rollout_loss`, `compute_loss`, `update_teacher`), `encoder.py`, `vq_encoder.py`, `attention.py`, `data.py`, `arm.py`, and `training.py`. Current exact source paths and hashes are resolved through the audit/provenance files rather than inferred from a historical seed44 directory.
