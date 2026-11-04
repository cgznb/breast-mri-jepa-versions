# Seed44 subgroup and training-probe analysis: fold metrics calculated separately

This revision uses seed44 only. Each fold model predicts the same 175 holdout patients (56 positives) at T0/T1/T2, and each fold metric is calculated independently. We then report the arithmetic mean and sample SD (ddof=1) across five fold metrics. Macro first averages T0/T1/T2 metric values within each fold. Prediction probabilities are never averaged; this report does not describe a probability ensemble.

The five models use overlapping development patients and the same holdout, so their metric SD is descriptive variation among fitted fold models. It is not an independent-test SD or uncertainty over training seeds. Folds and repeated landmarks do not increase the number of independent holdout patients.

Subgroup confidence intervals use 2,000 shared patient-bootstrap draws (RNG 20261103). Each draw recomputes each fixed fold model AUROC before averaging five fold metrics and T0/T1/T2 for Macro. Patient draws are shared across all folds, models, and landmarks; fold models are held fixed and not resampled. Single-class subgroup bootstrap draws are omitted from discrimination intervals, with valid counts exported. AUROC/AP are suppressed unless n>=10, positives>=5, negatives>=5. All subgroup intervals are unadjusted and do not establish interactions.

## Macro ForecastAttention minus NoJEPA

| Subgroup | n / positives | Difference of mean fold AUROCs | Conditional 95% patient interval |
|---|---:|---:|---:|
| HR+/HER2- | 68 / 11 | +0.0324 | [-0.0153, +0.0811] |
| HR+/HER2+ | 31 / 12 | -0.0415 | [-0.1353, +0.0457] |
| HR-/HER2+ | 14 / 7 | +0.0082 | [-0.1483, +0.1483] |
| HR-/HER2- | 62 / 26 | +0.0217 | [-0.0245, +0.0667] |
| Age <50 | 90 / 29 | +0.0175 | [-0.0080, +0.0449] |
| Age >=50 | 85 / 27 | +0.0183 | [-0.0181, +0.0576] |
| ISPY2 namespace | 125 / 39 | +0.0117 | [-0.0164, +0.0422] |
| ACRIN-6698 namespace | 50 / 17 | +0.0221 | [-0.0135, +0.0631] |
| Largest component contained | 164 / 55 | +0.0152 | [-0.0069, +0.0377] |
| Largest component clipped | 11 / 1 | suppressed | suppressed |

The HR+/HER2+ difference is negative; the other three HR/HER2 point differences are positive. All four unadjusted intervals include zero. The clipped subgroup contains 11 patients and one positive outcome, so discrimination is suppressed.

## Archived training probes, common epoch 20

Each diagnostic is calculated in its own fold snapshot and EMA coordinates, followed by the mean and descriptive sample SD across five seed44 fold statistics. Learned latent states are never averaged. An epoch/horizon is retained only when all five fold snapshots are available. NoJEPA has no trained-world diagnostic and is unavailable rather than zero.

Forecast skill is 1 - forecast L1 / copy L1. The copy baseline holds the observed landmark state constant within the same checkpoint EMA coordinates. Unlike the training loss mask, probes require a complete real observed landmark-to-target chain; missing intermediate visits exclude the query. These are fixed up-to-32-patient training probes, not best-checkpoint holdout forecast assessments.

| Model | Horizon | Mean skill | Sample fold SD | Probe patient range |
|---|---|---:|---:|---:|
| DirectReadout | horizon_1 | +0.0258 | 0.0555 | 23--32 |
| DirectReadout | horizon_2 | +0.0500 | 0.1423 | 17--20 |
| DirectReadout | horizon_3 | +0.1935 | 0.1336 | 5--9 |
| DirectReadout | horizon_all | +0.0451 | 0.0833 | 23--32 |
| ForecastAttention | horizon_1 | -0.0368 | 0.0163 | 23--32 |
| ForecastAttention | horizon_2 | -0.0154 | 0.0714 | 17--20 |
| ForecastAttention | horizon_3 | +0.0242 | 0.0277 | 5--9 |
| ForecastAttention | horizon_all | -0.0271 | 0.0252 | 23--32 |

The direct-readout probes show higher copy-normalized skill than forecast attention at each listed horizon. The attention model retains nonzero pCR-to-predictor gradients. These descriptive diagnostics check gradient routing but cannot identify a clinical mechanism or establish generalization.

## Figure aggregation

- ROC/PR: construct each fold curve separately and plot mean coordinates on a common grid. Shared patient bootstrap draws are evaluated for every fixed fold model before averaging curve coordinates. Text labels give mean exact fold AUROC/AP; the interpolated curve area need not equal the mean exact metric.
- Landmark plots: points are mean fold metrics, error bars are sample SD across five fitted fold metrics, and faint shading shows conditional patient-bootstrap intervals. These summaries measure different quantities.
- Reliability: each fold model has five separate equal-count probability rank bins of 35 patients. Plotted coordinates average the five fold-specific summaries. Membership can differ across folds; this is a descriptive mean plot rather than a curve for averaged patient probabilities.
- DCA: compute each fold net benefit separately at every threshold, then plot the mean.
- Outcome score changes: compute patient T2-T0 scores within each fold; average the five fold-specific histogram proportions. This is not the histogram of averaged probabilities. Denominators remain 119 non-pCR and 56 pCR patients.
- Mechanism: use mean and sample SD of fold-specific statistics, retaining each snapshot coordinate system.

These scripts read retained predictions, diagnostics, and validated NPZs. They do not train models, load model checkpoints, or recombine MRI metadata and masks.


## Independent numerical verification

A separate scikit-learn recomputation read the original manifest and retained predictions, evaluated every seed44 fold model separately, and independently calculated subgroup AUROC, AP, balanced accuracy, sensitivity, specificity, F1, NLL, and Brier scores. All 6,552 finite per-fold/aggregate mean/SD scalar comparisons passed; maximum absolute error was 4.44e-16. Discrimination was suppressed under the stated denominator rule. All figure exports retain seed44 only and explicitly distinguish fold1--5 summaries from their mean. The independent holdout count remains 175.
