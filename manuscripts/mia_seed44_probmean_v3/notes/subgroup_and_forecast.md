# Seed44 probability-ensemble subgroups and archived forecast diagnostics

This revision uses **seed44 only**. For pCR, each patient's five fold-model probabilities are averaged separately at each landmark, then all metrics and patient-bootstrap intervals are calculated on those 175 averaged probabilities. This is a fivefold probability ensemble, not a mean of fold metrics. Macro is the equal mean of the T0, T1, and T2 metrics. Earlier v2 results are preserved in their original directory and are not the results of this revision. No model was rerun and no checkpoint was loaded.

## Clinical and acquisition-related subgroup analysis

`scripts/subgroup_analysis.py` reads the current Medium976 manifest and receipt-backed best-test predictions for the three principal models. It verifies 175 matched patients and 56 positive labels at T0/T1/T2 in each of the 15 seed44 runs. Each source receipt and registered report identity is checked before reading the predictions. Only group aggregates are exported; patient identifiers and individual probabilities remain in the source reports.

Subgroup assignments use the manifest's released HR/HER2 values and unstandardized age. Age 50 is an exploratory reporting cutpoint, not an established clinical decision boundary. Source namespace comes from the frozen geometry metadata, and `largest_clipped` comes from its saved baseline mask QC. The latter refers to the predicted baseline component, not pathology-verified lesion coverage.

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

The namespace groups belong to the same clinical release; they are not independent external validation cohorts. AUROC/AP and their uncertainty are suppressed when a group has fewer than ten patients or fewer than five outcomes in either class. Consequently, discrimination is not reported for the 11 clipped cases with only one positive outcome. Their thresholded and probability metrics remain descriptive only.

Point estimates include AUROC, AP, balanced accuracy, sensitivity, specificity, F1, NLL, and Brier score. Thresholded quantities use 0.5, applied **after averaging fold probabilities**. NLL clips averaged probabilities at 1e-7 for numerical stability. A 2,000-resample within-subgroup patient bootstrap uses the same patient weights across models and landmarks, with RNG seed 20261102. The fixed probability ensemble is resampled; probabilities and labels remain paired. A replicate missing one outcome class is omitted for AUROC intervals, with valid-replicate counts recorded. Intervals condition on the fitted seed44 ensemble and exclude retraining, model/seed-selection uncertainty, and multiplicity correction.

Macro AUROC differences, ForecastAttention minus NoJEPA:

| Subgroup | Difference | Exploratory 95% interval |
|---|---:|---|
| HR+/HER2- | +0.0287 | -0.0318 to +0.0914 |
| HR+/HER2+ | +0.0307 | -0.0750 to +0.1318 |
| HR-/HER2+ | +0.0748 | -0.0251 to +0.2502 |
| HR-/HER2- | +0.0331 | -0.0128 to +0.0840 |
| Age <50 | +0.0141 | -0.0154 to +0.0422 |
| Age >=50 | +0.0313 | -0.0044 to +0.0731 |
| ISPY2 namespace | +0.0217 | -0.0083 to +0.0539 |
| ACRIN-6698 namespace | +0.0309 | -0.0145 to +0.0820 |
| Component contained | +0.0231 | +0.0012 to +0.0464 |

All four HR/HER2 differences are positive but their intervals include zero. These unadjusted exploratory intervals do not establish an interaction or subtype-specific efficacy. Unlike the older two-seed mean-of-fold-metrics analysis, this seed44 probability-ensemble analysis has a positive HR+/HER2+ point difference; the old negative estimate must not be carried into the current manuscript.

The full model's HR+/HER2- sensitivity/specificity at 0.5 is 0.2424/0.9474, compared with 0.0909/0.9825 for NoJEPA. This low sensitivity persists despite better subgroup ranking. Low pCR prevalence provides context for predictive values, but does not itself explain sensitivity. In HR+/HER2+, Full and NoJEPA sensitivity/specificity are 0.5556/0.4737 and 0.5833/0.4737. In HR-/HER2-, Full sensitivity/specificity is 0.8718/0.3148. The 14-patient HR-/HER2+ group has Full macro AUROC 0.9116 but specificity only 0.0476 and a wide difference interval. Its high discrimination must not be described as a robust subtype-specific result.

## Archived forecast and gradient evidence: a different aggregation

The source runs contain `diagnostics.json`, sampled every five epochs on a fixed 32-patient **training probe**, with each patient's fixed sampled prefix. These are neither full validation nor checkpoint-selection evaluations. No separate best-checkpoint holdout forecast-evaluation file exists for this suite.

Only seed44 diagnostics are included. Mechanism summaries are descriptive **five-fold means of per-snapshot copy-normalized skills or gradient norms**. They are separate from the patient-wise probability ensemble used for pCR. Latent predictions cannot be averaged across fold models as though their learned EMA coordinates were shared.

Within each snapshot, the forecast and copy target share that snapshot's EMA encoder coordinates. Autonomous copy holds the real landmark state constant. Unlike the missing-target-masked training objective, diagnostic eligibility requires a continuous actually observed visit chain: the real landmark, target, and every intervening canonical visit must be present. Missing intermediate visits exclude the chain, and denominators vary by horizon. The saved skill `1 - autonomous_relative_to_copy` compares each forecast to its own snapshot's copy reference. Even normalized skills remain descriptive and do not establish that the learned spaces capture equivalent anatomy or biology.

At the illustrative common epoch 20, averaging the five seed44 fold-specific skills:

| Model | Horizon1 skill | Horizon2 skill | Horizon3 skill |
|---|---:|---:|---:|
| DirectReadout | +0.0258 | +0.0500 | +0.1935 |
| ForecastAttention | -0.0368 | -0.0154 | +0.0242 |

NoJEPA's archive marks world diagnostics unavailable because no world objective or differentiable forecast classifier is enabled. It is shown as unavailable rather than assigned zero forecast error or zero skill. Horizon3 eligibility is only 5--9 training patients per fold probe. Epoch20 is a descriptive reference, not new checkpoint selection or a claim about the best pCR checkpoint. Common-epoch trajectories from 5--50 are also exported.

The negative early-horizon skill for the full classifier remains counter-evidence to explaining its pCR ranking by uniformly more accurate forecasts. Classification may shape the forecast representation in ways useful to the endpoint but less faithful to its own latent copy criterion. This remains a hypothesis rather than a demonstrated mechanism.

At epoch20, the mean task-loss gradient norm in the world predictor is 0 for DirectReadout and 0.7148 for ForecastAttention; corresponding weighted rollout-gradient means are 0.1213 and 0.1308. These values check the expected classification-gradient route. Absolute gradient norms depend on parameterization and training state and are not clinical effect sizes. Archived rows retain seed and fold but no patient identifiers.

## Files and scope

- `data/subgroup_sizes.csv`: aggregate subgroup denominators and suppression status.
- `data/subgroup_metrics.csv`: seed44 probability-ensemble metrics and AUROC intervals.
- `data/subgroup_paired_differences.csv`: paired Full--NoJEPA AUROC intervals.
- `data/subgroup_fold_metrics.csv`: per-fold metrics for audit only; not the manuscript estimator.
- `data/subgroup_seed_metrics.csv`: same seed44 ensemble point estimates in the legacy seed-audit schema.
- `data/subgroup_world_probe.csv`, `data/subgroup_gradient_probe.csv`: archived seed44/fold training diagnostics.
- `data/subgroup_world_summary.csv`: seed44 diagnostic fold means; descriptive fold spread is not a patient standard error.
- `data/subgroup_provenance.json`: source hashes and separate pCR/diagnostic aggregation descriptions.
- `tables/subgroup_results.tex`, `tables/subgroup_world_probe.tex`: manuscript tables with the distinct aggregation rules stated explicitly.

MRI, masks, geometry assets, manifests, source training outputs, and checkpoints were not changed. Only this revision's subgroup script and aggregate outputs were written.

Independent numerical verification used scikit-learn 1.9.1: all 936 reported, unsuppressed subgroup metric values agreed within 3.4e-16; the 120 legacy seed-audit rows exactly match ensemble point estimates. Weighted AUROC also agreed within 1e-12 on 486 valid resamples with tied scores; all 14 missing-class resamples were non-finite and excluded as intended. Independent whole-holdout macro AUROC/AP/F1 reproduced NoJEPA 0.7279/0.5968/0.5416, DirectReadout 0.7399/0.6075/0.5678, and ForecastAttention 0.7507/0.6283/0.5604.
