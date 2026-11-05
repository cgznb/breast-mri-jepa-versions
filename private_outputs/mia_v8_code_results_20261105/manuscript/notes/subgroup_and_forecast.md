# Four-seed subgroup and training-probe analysis: seeds 42, 44, 71, 2026

The three principal variants are NoJEPA, DirectReadout, and ForecastMean. ForecastMean is the complete model and pools terminal-state and change tokens with uniform spatial means. The spatial encoder and fusion Transformers retain their internal attention operations.

Within each training seed/model/landmark, each patient probability is the arithmetic mean of five fold probabilities. Metrics are evaluated separately in each seed; the four seed metrics then supply their equal mean and sample SD (ddof=1, variance denominator three). Probabilities are never averaged across seeds. The same 175 holdout patients (56 positives) occur at every seed/landmark, so repeated predictions do not create 700 independent patients.

Subgroup confidence intervals use 2,000 shared within-subgroup patient-bootstrap draws (RNG 20261103). Each draw recomputes four separate seed AUROCs before averaging seed metrics and T0/T1/T2 metrics for Macro. The seed set and fitted fold models are fixed, with no seed or fold resampling. Single-class draws are omitted from discrimination intervals, with valid counts retained. AUROC/AP are suppressed unless n>=10, positives>=5, negatives>=5. These unadjusted intervals do not establish interactions or corrected subgroup significance.

## Macro ForecastMean minus NoJEPA

| Subgroup | n / positives | Difference of mean seed AUROCs | Conditional unadjusted 95% patient interval |
|---|---:|---:|---:|
| HR+/HER2- | 68 / 11 | +0.0385 | [+0.0089, +0.0743] |
| HR+/HER2+ | 31 / 12 | +0.0143 | [-0.0349, +0.0629] |
| HR-/HER2+ | 14 / 7 | +0.0272 | [-0.0272, +0.1136] |
| HR-/HER2- | 62 / 26 | +0.0380 | [+0.0083, +0.0705] |
| Age <50 | 90 / 29 | +0.0269 | [+0.0069, +0.0473] |
| Age >=50 | 85 / 27 | +0.0165 | [+0.0003, +0.0352] |
| ISPY2 namespace | 125 / 39 | +0.0253 | [+0.0094, +0.0422] |
| ACRIN-6698 namespace | 50 / 17 | +0.0183 | [-0.0050, +0.0424] |
| Largest component contained | 164 / 55 | +0.0227 | [+0.0107, +0.0359] |
| Largest component clipped | 11 / 1 | suppressed | suppressed |

All four HR/HER2 differences are positive. The unadjusted paired intervals exclude zero in the HR+/HER2- and HR-/HER2- groups; the other two subtype intervals include zero. Both exploratory age intervals exclude zero, with the older group's lower bound only about +0.000344. No age or subtype interaction was tested, and no multiplicity adjustment was applied to these intervals. The clipped subgroup contains 11 patients with one positive outcome, so discrimination remains suppressed.

In HR+/HER2- patients, Macro sensitivity/specificity/AUROC is 0.1061/0.9868/0.6812 for NoJEPA, 0.0985/0.9912/0.7064 for DirectReadout, and 0.1515/0.9620/0.7197 for ForecastMean. Sensitivity at threshold 0.5 remains low despite the positive AUROC difference.

## Archived training probes, common epoch 20

Each diagnostic is evaluated within its own fold snapshot and EMA coordinates. Five fold statistics are averaged within each seed, and the four seed means supply the overall mean and sample SD. Latent states are never averaged across models or seeds. A summary epoch/horizon requires all 20 fold snapshots, with all five folds in each of the four seeds. NoJEPA has no trained-world diagnostic and is unavailable rather than zero.

Forecast skill is 1 - forecast L1 / copy L1. The copy forecast holds the observed landmark state constant in the same snapshot EMA coordinates. Unlike the training loss mask, probes require a complete real observed landmark-to-target chain; missing intermediate visits exclude the query. They are fixed up-to-32-patient training probes, not best-checkpoint holdout forecast evaluations. Both DirectReadout and ForecastMean have complete 20-snapshot diagnostics at epochs 5, 10, ..., 60.

| Model | Horizon | Mean skill | Sample SD of four seed means | Probe patient range |
|---|---|---:|---:|---:|
| DirectReadout | horizon_1 | +0.0129 | 0.0216 | 23--32 |
| DirectReadout | horizon_2 | +0.0621 | 0.0195 | 17--20 |
| DirectReadout | horizon_3 | +0.1635 | 0.0698 | 5--9 |
| DirectReadout | horizon_all | +0.0398 | 0.0213 | 23--32 |
| ForecastMean | horizon_1 | -0.0469 | 0.0257 | 23--32 |
| ForecastMean | horizon_2 | -0.0479 | 0.0498 | 17--20 |
| ForecastMean | horizon_3 | +0.0283 | 0.0521 | 5--9 |
| ForecastMean | horizon_all | -0.0447 | 0.0348 | 23--32 |

DirectReadout has higher copy-normalized skill than ForecastMean at each listed horizon. The pCR-to-predictor gradient means are 0 and 0.9867, and weighted rollout-gradient means are 0.1474 and 0.1715 for DirectReadout and ForecastMean, respectively. These descriptive diagnostics check gradient routing; they do not establish a clinical mechanism, forecast fidelity on independent patients, or a causal explanation for ranking differences.

## Figure aggregation

- ROC/PR: evaluate each seed predictor separately, interpolate on common grids, and plot mean curve coordinates. Patient draws are shared across all seed/model/landmark vectors before averaging coordinates. Text labels give mean exact AUROC/AP, which need not equal the area under an interpolated mean curve.
- Landmark plots: points are mean seed metrics; error bars show sample SD across four seed metrics, while faint shading gives separate conditional patient-bootstrap intervals.
- Reliability: every seed uses its own five equal-count probability rank bins of 35 patients. Plot coordinates average seed-specific bin summaries. Membership may differ across seeds; this is a descriptive plot rather than calibration of a twenty-model predictor.
- DCA: calculate seed net benefit separately at every threshold, then plot the four-seed mean.
- Score changes: calculate patient T2-T0 score changes separately within each seed, construct seed histograms, and average bin proportions. No probability is averaged across seeds before histogramming. Denominators remain 119 non-pCR and 56 pCR patients.
- Mechanism: summarize within-seed five-fold diagnostic means with the four-seed mean/SD, keeping snapshot coordinate systems separate.

Figure generation reads retained predictions, diagnostics, and validated NPZs only. It does not train models, load model checkpoints, or recombine native metadata and masks. The anonymous method framework uses the same development example and display windows as prior versions.

## Independent verification

A separate scikit-learn calculation rebuilt each seed-specific five-fold probability predictor directly from retained source reports and checked subgroup AUROC, AP, balanced accuracy, sensitivity, specificity, F1, negative log likelihood, and Brier score. All 5,616 scalar comparisons agreed to floating-point precision (maximum absolute difference 4.44e-16). A further 576 independent mechanism mean/SD checks recomputed skills and patient-weighted gradient summaries from source diagnostics; every summarized epoch/horizon had all 20 snapshots. The independent patient count remains 175, with 56 positives. Only the four designated training seeds and three principal variants are exported. All eight generated figure plates were visually inspected; the subgroup forest plot uses a short numeric axis format to prevent crowded tick labels.
