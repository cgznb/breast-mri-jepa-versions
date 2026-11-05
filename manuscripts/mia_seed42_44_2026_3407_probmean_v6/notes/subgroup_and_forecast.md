# Four-seed subgroup and training-probe analysis

Training seeds: 42, 44, 2026, and 3407. Within each seed/model/landmark, each patient probability is the arithmetic mean of five fold probabilities. Metrics are evaluated independently for every seed, then the four seed metrics supply the mean and sample SD (ddof=1). Probabilities are never averaged across seeds. The same 175 holdout patients (56 positives) occur at every landmark and seed; repeated predictions do not create 700 independent patients.

Subgroup confidence intervals use 2,000 shared within-subgroup patient-bootstrap draws (RNG 20261103). Every draw recomputes four separate seed AUROCs before averaging seed metrics and T0/T1/T2 metrics for Macro. The fitted seed set is held fixed; no seeds or fold models are resampled. Single-class draws are omitted from discrimination intervals, with valid counts exported. AUROC/AP are suppressed unless n>=10, positives>=5, and negatives>=5. These unadjusted intervals do not establish subgroup interactions or corrected significance.

## Macro ForecastAttention minus NoJEPA

| Subgroup | n / positives | Difference of mean seed AUROCs | Conditional 95% patient interval |
|---|---:|---:|---:|
| HR+/HER2- | 68 / 11 | +0.0257 | [-0.0010, +0.0556] |
| HR+/HER2+ | 31 / 12 | +0.0113 | [-0.0457, +0.0730] |
| HR-/HER2+ | 14 / 7 | +0.0425 | [-0.0187, +0.1320] |
| HR-/HER2- | 62 / 26 | +0.0219 | [-0.0057, +0.0516] |
| Age <50 | 90 / 29 | +0.0160 | [+0.0004, +0.0318] |
| Age >=50 | 85 / 27 | +0.0192 | [+0.0017, +0.0410] |
| ISPY2 namespace | 125 / 39 | +0.0182 | [+0.0032, +0.0348] |
| ACRIN-6698 namespace | 50 / 17 | +0.0160 | [+0.0001, +0.0336] |
| Largest component contained | 164 / 55 | +0.0186 | [+0.0077, +0.0310] |
| Largest component clipped | 11 / 1 | suppressed | suppressed |

All four HR/HER2 point differences are positive, and all four unadjusted paired intervals include zero. The HR+/HER2- interval now includes zero after adding seed3407; the previous three-seed interval excluded zero. Both age-group intervals exclude zero, but neither is multiplicity-adjusted, and no interaction is tested. The clipped subgroup contains 11 patients and one positive outcome, so discrimination remains suppressed.

## Archived training probes, common epoch 20

Each diagnostic is calculated within its fold snapshot and EMA coordinates. Five fold statistics are averaged within each seed, and the four seed means supply the overall mean and sample SD. Learned latent states are never averaged across models or seeds. A summary epoch/horizon is retained only when all 20 fold snapshots are available, with all five folds in each of the four seeds. NoJEPA has no trained-world diagnostic and is unavailable rather than zero.

Forecast skill is 1 - forecast L1 / copy L1. The copy baseline holds the observed landmark state constant in the same snapshot EMA coordinates. Unlike the training loss mask, probes require a complete real observed landmark-to-target chain; missing intermediate visits exclude the query. These are fixed up-to-32-patient training probes, not best-checkpoint holdout forecast evaluations.

| Model | Horizon | Mean skill | Sample SD of four seed means | Probe patient range |
|---|---|---:|---:|---:|
| DirectReadout | horizon_1 | +0.0090 | 0.0183 | 23--32 |
| DirectReadout | horizon_2 | +0.0524 | 0.0146 | 17--20 |
| DirectReadout | horizon_3 | +0.1590 | 0.0726 | 5--9 |
| DirectReadout | horizon_all | +0.0336 | 0.0213 | 23--32 |
| ForecastAttention | horizon_1 | -0.0509 | 0.0199 | 23--32 |
| ForecastAttention | horizon_2 | -0.0516 | 0.0358 | 17--20 |
| ForecastAttention | horizon_3 | +0.0122 | 0.0178 | 5--9 |
| ForecastAttention | horizon_all | -0.0485 | 0.0236 | 23--32 |

DirectReadout has higher copy-normalized skill than ForecastAttention at each listed horizon. ForecastAttention retains a nonzero pCR-to-predictor gradient. These descriptive diagnostics check gradient routing; they cannot identify a clinical mechanism or establish generalization.

## Figure aggregation

- ROC/PR: evaluate each seed predictor separately and plot the arithmetic mean of curve coordinates on common grids. Patient-bootstrap draws are shared across all seed/model/landmark predictions before averaging curve coordinates. Text labels are means of exact seed AUROC/AP; the area under an interpolated mean curve need not equal the mean exact metric.
- Landmark plots: points are mean seed metrics; error bars are sample SD across four seed metrics, and faint shading shows separate conditional patient-bootstrap intervals. Seed variation and patient uncertainty are different quantities.
- Reliability: every seed has five independent equal-count probability rank bins of 35 patients. Plot coordinates average seed-specific bin summaries. Membership can differ across seeds; this is a descriptive mean plot, not a reliability curve for a twenty-model probability predictor.
- DCA: calculate each seed net benefit separately at every threshold and plot its mean across four seeds.
- Outcome score changes: compute T2-T0 score changes within each seed/patient, construct each seed histogram, then average bin proportions. Do not plot a histogram of cross-seed-mean probabilities. Denominators remain 119 non-pCR and 56 pCR patients.
- Mechanism: average within-seed fold diagnostic means, with SD across four seed means; snapshot coordinate systems remain separate.

Figure generation reads retained report predictions and existing validated NPZs. It does not train models, load model checkpoints, or recombine native metadata with masks. The anonymous framework uses the same development example and display windows as before.


## Independent numerical verification

A separate scikit-learn recomputation read the source manifests and retained reports, rebuilt all four seed-specific five-fold predictors, and independently recalculated subgroup AUROC, AP, balanced accuracy, sensitivity, specificity, F1, NLL, and Brier scores. All 5,616 finite per-seed/aggregate mean/SD scalar comparisons passed; maximum absolute error was 4.44e-16. Discrimination suppression was retained. Another 192 mechanism aggregation checks independently verified that each summary retained all 20 fold diagnostics and used the mean/SD of four within-seed five-fold means. All curve, reliability, and outcome-score exports include seed3407 as applicable; no repeated seed prediction increases the independent patient count above 175.
