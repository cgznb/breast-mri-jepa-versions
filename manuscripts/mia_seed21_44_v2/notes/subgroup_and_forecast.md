# Exploratory subgroup and archived forecast diagnostics

This revision follows the user's updated scope: **seed21 and seed44 are weighted equally**, after averaging the five fold metrics separately within each seed. The manuscript-facing CSVs and tables show only the combined `mean21_44` group. Seed-specific fold/aggregate records are retained only for audit. No model was rerun and no checkpoint was loaded.

## Clinical and acquisition-related subgroup analysis

`scripts/subgroup_analysis.py` reads the current Medium976 manifest and receipt-backed best-test predictions for the three principal models. It verifies 175 matched patients/56 positive labels at T0/T1/T2 for each of the 30 trained runs. Each result is calculated per fold, averaged within a seed, and then averaged over the two seeds; Macro additionally averages T0/T1/T2.

The subgroup assignments use the manifest's released HR/HER2 values and unstandardized age. Age 50 is an **exploratory reporting cutpoint**, not a claimed clinical decision boundary. Source namespace comes from the frozen geometry metadata, and `largest_clipped` comes from its saved baseline mask QC. The latter refers to the predicted baseline component, not pathology-verified lesion coverage.

| Subgroup | Patients | pCR positive |
|---|---:|---:|
| HR+/HER2- | 68 | 11 |
| HR+/HER2+ | 31 | 12 |
| HR-/HER2+ | 14 | 7 |
| HR-/HER2- | 62 | 26 |
| Age <50 | 90 | 29 |
| Age >=50 | 85 | 27 |
| ISPY2 native namespace | 125 | 39 |
| ACRIN-6698 native namespace | 50 | 17 |
| Largest predicted component contained | 164 | 55 |
| Largest predicted component partly clipped | 11 | 1 |

The namespace groups belong to the same clinical release; they are not independent external validation cohorts. The HR-/HER2+ group is small even though both outcome classes are represented. AUROC/AP and their uncertainty are suppressed when a group has fewer than five outcomes in either class or fewer than ten patients. Consequently, discrimination is not reported for the 11 clipped cases with only one positive outcome. Their thresholded and probability metrics remain descriptive only.

Point estimates include AUROC, AP, balanced accuracy, sensitivity, specificity, F1, NLL, and Brier score. Thresholded quantities use 0.5. NLL clips probabilities at 1e-7 for numerical stability. A 2,000-resample within-subgroup patient bootstrap uses the **same patient weights across both seeds, all models, all five folds, and all three landmarks**; each replicate preserves the complete fold -> seed -> two-seed mean aggregation. A replicate missing one outcome class is omitted for AUROC intervals, with its valid-replicate count recorded. Intervals condition on fixed trained models and do not address seed/model selection or multiple subgroup comparisons.

Combined Macro AUROC differences, ForecastAttention minus NoJEPA:

| Subgroup | Difference | Exploratory 95% interval |
|---|---:|---|
| HR+/HER2- | +0.0522 | +0.0095 to +0.0995 |
| HR+/HER2+ | -0.0154 | -0.0780 to +0.0473 |
| HR-/HER2+ | +0.0041 | -0.1133 to +0.0993 |
| HR-/HER2- | +0.0219 | -0.0094 to +0.0538 |
| Age <50 | +0.0274 | +0.0057 to +0.0493 |
| Age >=50 | +0.0187 | -0.0070 to +0.0485 |
| ISPY2 namespace | +0.0178 | -0.0029 to +0.0419 |
| ACRIN-6698 namespace | +0.0253 | -0.0008 to +0.0568 |
| Component contained | +0.0220 | +0.0045 to +0.0410 |

These intervals are unadjusted exploratory summaries. Their crossing or not crossing zero does not establish an interaction or subtype-specific efficacy. A difference between subgroups would require an interaction analysis with appropriate power and a prespecified question. In particular, the negative HR+/HER2+ estimate must remain visible.

Audit-only seed-direction check: the HR+/HER2- and HR-/HER2- AUROC differences are positive in both seeds. The HR+/HER2+ difference changes direction between the two seeds, while HR-/HER2+ is zero in one seed and slightly positive in the other. Separate seed values are retained in `subgroup_seed_metrics.csv`, but are not presented as additional manuscript result groups.

The full model's HR+/HER2- sensitivity/specificity at 0.5 is 0.2030/0.9170, compared with 0.0970/0.9690 for NoJEPA. In HR+/HER2+, the corresponding full-model quantities are 0.6611/0.4684 despite its lower AUROC; the NoJEPA quantities are 0.5306/0.5719. In HR-/HER2-, full-model sensitivity/specificity is 0.7846/0.3944. These results show why pooled discrimination should be supplemented by subgroup operating characteristics. The high AUROC in the 14-patient HR-/HER2+ group coexists with specificity 0.1667 and a wide interval, so it should not be described as a robust subtype-specific result.

## Existing forecast and gradient evidence

The source runs contain `diagnostics.json`, sampled every five epochs on a **fixed 32-patient training probe** with each patient's fixed sampled prefix. They are explicitly neither full validation nor checkpoint-selection evaluations. There are no separate `world_best_test.json` or `world_best_val.json` files for this suite. We therefore do not describe these data as best-checkpoint holdout forecast accuracy.

Within each diagnostic snapshot, forecast and copy targets are in the same snapshot's EMA encoder coordinates. Autonomous copy means holding the real landmark state constant. Continuous-chain eligibility additionally requires the real target and intermediate stages. The saved ratio `autonomous_relative_to_copy` and skill `1 - ratio` provide an internal reference that avoids comparing raw latent L1 values between learned teacher spaces. Even normalized skills are descriptive and do not establish that the learned spaces capture the same anatomy or biology.

For the illustrative common epoch20 probe, averaged over folds within each seed and then the two seeds:

| Model | Horizon1 skill | Horizon2 skill | Horizon3 skill |
|---|---:|---:|---:|
| DirectReadout | +0.0227 | +0.0368 | +0.1584 |
| ForecastAttention | -0.0295 | -0.0271 | +0.0174 |

NoJEPA's archive explicitly marks its world diagnostic unavailable because no world objective or differentiable forecast classifier is enabled. It is shown as unavailable, not assigned zero forecast error or zero skill. Horizon3 eligibility is only 5--9 patients per training probe, so long-horizon values are especially uncertain. The common epoch20 is a descriptive reference, not a new checkpoint selection or a claim about the best pCR checkpoint. Curves at common epochs 5--50 are also exported.

The negative early-horizon skill for the full classifier is relevant counter-evidence: a better pCR ranking cannot simply be explained as more accurate latent forecasts. Classification may shape the forecast representation in ways useful to the readout but less faithful to the fixed-copy latent criterion. This is a hypothesis, not proof of the mechanism.

Audit-only seed-direction check: DirectReadout's horizon1--3 mean skills are positive in both seeds at epoch20. ForecastAttention's horizon1--2 mean skills are negative in both seeds and its horizon3 mean skill is positive in both. The combined diagnostic therefore does not hide a reversal between the two selected seeds, but remains a fixed-training-probe analysis with model-specific teacher coordinates.

At epoch20, the mean task-loss gradient norm in the world predictor is 0 for DirectReadout and 0.8205 for ForecastAttention; corresponding weighted rollout-gradient means are 0.1364 and 0.1743. These values verify the expected classification-gradient route in the full architecture. Absolute gradient norms depend on parameterization and training state, and are not clinical effect sizes. Raw probe rows retain the seed and fold but no patient identifiers.

## Files

- `data/subgroup_sizes.csv`: aggregate subgroup denominators and suppression status.
- `data/subgroup_metrics.csv`: combined eight-metric landmark/Macro results and AUROC intervals.
- `data/subgroup_paired_differences.csv`: combined paired Full--NoJEPA AUROC intervals.
- `data/subgroup_fold_metrics.csv`, `data/subgroup_seed_metrics.csv`: seed/fold audit aggregates only.
- `data/subgroup_world_probe.csv`, `data/subgroup_gradient_probe.csv`: archived seed/fold training diagnostics.
- `data/subgroup_world_summary.csv`: combined common-epoch curves; descriptive run spread is not an independent-sample standard error.
- `data/subgroup_provenance.json`: manifest and source-report/diagnostic hashes; no patient-level rows.
- `tables/subgroup_results.tex`, `tables/subgroup_world_probe.tex`: combined manuscript tables.

All outputs contain only aggregated data. MRI, masks, geometry assets, manifests, training outputs, and checkpoints were not changed.

Numerical verification: all eight point metrics and the tied-score weighted AUROC implementation agreed with scikit-learn within 1e-12 on independent numerical cases. Every combined metric was checked against the equal mean of the corresponding two seed aggregates, with maximum discrepancy zero. Aggregate CSV scans found no numeric patient identifiers or individual prediction rows.
