# Three-seed subgroup and training-probe analysis

Training seeds: 42, 44, and 2026. Within each seed and model, each patient landmark probability is the mean of five fold predictions. Metrics are computed independently per seed, and their arithmetic mean and sample SD (ddof=1) are reported. No probability is averaged across seeds. The same 175 holdout patients (56 positives) occur at each landmark and seed; repeated predictions do not increase the number of independent patients.

Subgroup confidence intervals use 2,000 shared within-subgroup patient-bootstrap draws (RNG 20261103), recompute all three seed AUROCs, average seed AUROCs, and average landmark AUROCs for Macro. The fitted seeds are held fixed; there is no seed resampling. Single-class bootstrap draws are omitted from discrimination intervals, with valid counts exported. AUROC/AP are suppressed unless n>=10, positives>=5, and negatives>=5. These intervals are unadjusted and do not establish subgroup interactions.

## Macro ForecastAttention minus NoJEPA

| Subgroup | n / positives | Delta mean AUROC | Conditional 95% patient interval |
|---|---:|---:|---:|
| HR+/HER2- | 68 / 11 | +0.0333 | [+0.0072, +0.0653] |
| HR+/HER2+ | 31 / 12 | +0.0171 | [-0.0466, +0.0864] |
| HR-/HER2+ | 14 / 7 | +0.0363 | [-0.0408, +0.1474] |
| HR-/HER2- | 62 / 26 | +0.0224 | [-0.0060, +0.0551] |
| Age <50 | 90 / 29 | +0.0189 | [+0.0016, +0.0371] |
| Age >=50 | 85 / 27 | +0.0226 | [+0.0024, +0.0480] |
| ISPY2 namespace | 125 / 39 | +0.0226 | [+0.0056, +0.0419] |
| ACRIN-6698 namespace | 50 / 17 | +0.0156 | [-0.0035, +0.0353] |
| Largest component contained | 164 / 55 | +0.0215 | [+0.0089, +0.0357] |
| Largest component clipped | 11 / 1 | suppressed | suppressed |

All four HR/HER2 point differences are positive, but only the HR+/HER2- unadjusted interval excludes zero. This is not multiplicity-adjusted or evidence of a subtype interaction. The clipped subgroup contains 11 patients and one positive outcome, so discrimination is suppressed.

## Archived training probes, common epoch 20

Each diagnostic is calculated within its own fold snapshot and EMA coordinates. Five fold statistics are averaged within each seed; the three seed means then supply the overall mean and sample SD. Learned latent states are never averaged across models or seeds. A summary epoch/horizon is retained only when all 15 fold snapshots are available. NoJEPA has no trained-world diagnostic and is unavailable, not zero.

Forecast skill is 1 - forecast L1 / copy L1. The copy baseline holds the observed landmark state constant within the same checkpoint EMA coordinates. Unlike the training loss mask, probes require a complete real observed landmark-to-target chain; missing intermediate visits exclude that query. These are up-to-32-patient fixed training probes, not best-checkpoint holdout forecast assessments.

| Model | Horizon | Skill mean | SD of seed means | Probe patient range |
|---|---|---:|---:|---:|
| DirectReadout | horizon_1 | +0.0068 | 0.0218 | 23--32 |
| DirectReadout | horizon_2 | +0.0551 | 0.0167 | 17--20 |
| DirectReadout | horizon_3 | +0.1724 | 0.0826 | 5--9 |
| DirectReadout | horizon_all | +0.0367 | 0.0250 | 23--32 |
| ForecastAttention | horizon_1 | -0.0576 | 0.0182 | 23--32 |
| ForecastAttention | horizon_2 | -0.0584 | 0.0405 | 17--20 |
| ForecastAttention | horizon_3 | +0.0089 | 0.0203 | 5--9 |
| ForecastAttention | horizon_all | -0.0545 | 0.0249 | 23--32 |

The direct-readout probes show higher copy-normalized skill than forecast attention at each listed horizon. The attention model retains nonzero pCR-to-predictor gradients. These descriptive diagnostics support checking gradient routing; they cannot identify a clinical mechanism or establish generalization.

## Figure aggregation

- ROC/PR: compute every seed curve independently and plot the mean curve coordinates. The shaded band resamples patients jointly across all seed/model/landmark predictions, then averages curve coordinates. Text labels are mean exact AUROC/AP; the area under an interpolated mean curve need not equal the mean exact metric.
- Landmark plots: points are mean seed metrics; error bars are sample SD of three seed metrics, while faint shading is conditional patient-bootstrap intervals. These are distinct uncertainty summaries.
- Reliability: each seed has its own five equal-count rank bins of 35 patients. The plotted coordinates average seed-specific bin summaries. Bin membership can differ across seeds; the graph is descriptive, not a reliability curve for a 15-model predictor.
- DCA: plot the arithmetic mean of independently computed seed net benefits at each threshold.
- Outcome score changes: compute T2-T0 scores within each seed/patient; create each seed histogram and average bin proportions. Do not compute a histogram of cross-seed-mean probabilities. Histogram denominators remain 119 non-pCR and 56 pCR patients, not three times those counts.
- Mechanism: use the mean of within-seed fold means and seed SD; each snapshot retains its own coordinates.

Figure generation reads retained report predictions and validated NPZs only. It does not launch training or load model checkpoints.


## Independent numerical verification

A separate scikit-learn recomputation read the source manifests and retained predictions, rebuilt all three seed-specific five-fold predictors, and independently recalculated subgroup AUROC, AP, balanced accuracy, sensitivity, specificity, F1, NLL, and Brier scores. All 4,680 finite per-seed/aggregate mean/SD scalar comparisons passed; maximum absolute error was 4.44e-16. Suppressed discrimination remained suppressed. The curve export uses only the aggregate seed label, while histogram/reliability exports contain the three individual seeds plus their explicitly marked metric/curve/distribution mean. Independent patient count remains 175.
