# Three-seed subgroup and training-probe analysis: seeds 44, 2026, 3407

Training seeds: 44, 2026, and 3407. Within each seed/model/landmark, each patient probability is the arithmetic mean of five fold probabilities. Metrics are evaluated separately in each seed; the three seed metrics then supply their equal mean and sample SD (ddof=1, variance denominator two). Probabilities are never averaged across seeds. The same 175 holdout patients (56 positives) occur at every seed/landmark, so repeated predictions do not create 525 independent patients.

Subgroup confidence intervals use 2,000 shared within-subgroup patient-bootstrap draws (RNG 20261103). Each draw recomputes three separate seed AUROCs before averaging seed metrics and T0/T1/T2 metrics for Macro. The seed set and fitted fold models are fixed, with no seed or fold resampling. Single-class draws are omitted from discrimination intervals, with valid counts retained. AUROC/AP are suppressed unless n>=10, positives>=5, negatives>=5. These unadjusted intervals do not establish interactions or corrected subgroup significance.

## Macro ForecastAttention minus NoJEPA

| Subgroup | n / positives | Difference of mean seed AUROCs | Conditional 95% patient interval |
|---|---:|---:|---:|
| HR+/HER2- | 68 / 11 | +0.0138 | [-0.0236, +0.0502] |
| HR+/HER2+ | 31 / 12 | +0.0034 | [-0.0672, +0.0741] |
| HR-/HER2+ | 14 / 7 | +0.0567 | [-0.0249, +0.1729] |
| HR-/HER2- | 62 / 26 | +0.0296 | [-0.0019, +0.0635] |
| Age <50 | 90 / 29 | +0.0123 | [-0.0057, +0.0302] |
| Age >=50 | 85 / 27 | +0.0223 | [+0.0001, +0.0490] |
| ISPY2 namespace | 125 / 39 | +0.0170 | [-0.0019, +0.0364] |
| ACRIN-6698 namespace | 50 / 17 | +0.0160 | [-0.0020, +0.0357] |
| Largest component contained | 164 / 55 | +0.0178 | [+0.0046, +0.0324] |
| Largest component clipped | 11 / 1 | suppressed | suppressed |

All four HR/HER2 differences are positive and all their unadjusted paired intervals include zero. The younger age-group interval also includes zero; the older group lower bound is only about +0.000068. No age or subtype interaction was tested, and no multiplicity adjustment was applied to these intervals. The clipped subgroup contains 11 patients with one positive outcome, so discrimination remains suppressed.

## Archived training probes, common epoch 20

Each diagnostic is evaluated within its own fold snapshot and EMA coordinates. Five fold statistics are averaged within each seed, and the three seed means supply the overall mean and sample SD. Latent states are never averaged across models or seeds. A summary epoch/horizon requires all 15 fold snapshots, with all five folds in each of the three seeds. NoJEPA has no trained-world diagnostic and is unavailable rather than zero.

Forecast skill is 1 - forecast L1 / copy L1. The copy forecast holds the observed landmark state constant in the same snapshot EMA coordinates. Unlike the training loss mask, probes require a complete real observed landmark-to-target chain; missing intermediate visits exclude the query. They are fixed up-to-32-patient training probes, not best-checkpoint holdout forecast evaluations.

| Model | Horizon | Mean skill | Sample SD of three seed means | Probe patient range |
|---|---|---:|---:|---:|
| DirectReadout | horizon_1 | +0.0081 | 0.0224 | 23--32 |
| DirectReadout | horizon_2 | +0.0454 | 0.0043 | 17--20 |
| DirectReadout | horizon_3 | +0.1312 | 0.0571 | 5--9 |
| DirectReadout | horizon_all | +0.0260 | 0.0183 | 23--32 |
| ForecastAttention | horizon_1 | -0.0462 | 0.0215 | 23--32 |
| ForecastAttention | horizon_2 | -0.0474 | 0.0426 | 17--20 |
| ForecastAttention | horizon_3 | +0.0107 | 0.0215 | 5--9 |
| ForecastAttention | horizon_all | -0.0445 | 0.0272 | 23--32 |

DirectReadout has higher copy-normalized skill than ForecastAttention at each listed horizon. ForecastAttention retains a nonzero pCR-to-predictor gradient. These descriptive diagnostics check gradient routing; they do not establish a clinical mechanism or generalization.

## Figure aggregation

- ROC/PR: evaluate each seed predictor separately, interpolate on common grids, and plot mean curve coordinates. Patient draws are shared across all seed/model/landmark vectors before averaging coordinates. Text labels give mean exact AUROC/AP, which need not equal the area under an interpolated mean curve.
- Landmark plots: points are mean seed metrics; error bars show sample SD across three seed metrics, while faint shading gives separate conditional patient-bootstrap intervals.
- Reliability: every seed uses its own five equal-count probability rank bins of 35 patients. Plot coordinates average seed-specific bin summaries. Membership may differ across seeds; this is a descriptive plot rather than calibration of a fifteen-model predictor.
- DCA: calculate seed net benefit separately at every threshold, then plot the three-seed mean.
- Score changes: calculate patient T2-T0 score changes separately within each seed, construct seed histograms, and average bin proportions. No probability is averaged across seeds before histogramming. Denominators remain 119 non-pCR and 56 pCR patients.
- Mechanism: summarize within-seed five-fold diagnostic means with the three-seed mean/SD, keeping snapshot coordinate systems separate.

Figure generation reads retained predictions, diagnostics, and validated NPZs only. It does not train models, load model checkpoints, or recombine native metadata and masks. The anonymous method framework uses the same development example and display windows as prior versions.

## Independent verification

A separate scikit-learn calculation rebuilt each seed-specific five-fold probability predictor directly from the retained source reports and checked subgroup AUROC, AP, balanced accuracy, sensitivity, specificity, F1, negative log likelihood, and Brier score. All 4,680 scalar comparisons agreed to floating-point precision (maximum absolute difference 4.44e-16). A further 192 mechanism aggregation checks confirmed complete 15-snapshot coverage and the means and sample SDs of within-seed fold means. The independent patient count remains 175. Curve, reliability, and score-change exports contain only seeds 44, 2026, and 3407, or their explicit aggregate tag; seed 42 is absent. All eight generated figure plates were visually checked together for clipping and layout errors.
