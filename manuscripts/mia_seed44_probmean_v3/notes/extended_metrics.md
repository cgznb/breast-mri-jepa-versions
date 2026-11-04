# Seed44 patient probability ensemble analysis

All 20 completed receipts, 60 registered report byte/mtime identities, and 20 manifest references passed. Seed44 configs and all landmark/fold test label and ID alignments passed. Every validation prediction was checked against its source manifest: its patient belongs to that model fold validation split, not its training or test split. All test predictions belong to the fixed manifest test split and labels match. Raw IDs, prediction rows, and probability arrays are never exported.
For every model, landmark, and patient, five fold probabilities are arithmetically averaged before metrics, patient bootstrap contrasts, temporal differences, or decision curves. This is not a mean of fold metrics. Macro averages three landmark metric values, not probabilities across landmarks.
The 18 quantities comprise 14 metrics and TP/FP/FN/TN. Landmark confusion counts are integer counts from one 175-patient ensemble prediction vector; Macro counts are the arithmetic mean of landmark counts. Individual fold metrics are audit-only.
All landmark vectors use exactly the same 175 holdout patients and 56 positive labels (prevalence 0.32). Every model has disjoint OOF development coverage 801/760/700 at T0/T1/T2 and no development/test patient overlap.
There are 2,000 shared nonparametric patient bootstrap draws (RNG 20261102), reused across models and landmarks. Percentile 95% intervals condition on the fixed trained models and chosen thresholds; they exclude retraining, seed-selection, and threshold-selection uncertainty. They do not quantify variation across training seeds.
The OOF Youden analysis concatenates exactly one held-out-fold score per available development patient and selects one threshold per model/landmark. OOF scores come from single fold models, while test scores are five-model ensemble means. Their score distributions differ: this is an operating-point sensitivity analysis, not validated ensemble calibration.
DCA uses a hypothetical pCR-positive decision and uncalibrated model scores; its net benefit is exploratory and does not establish clinical usefulness. No clinical intervention or benefit was evaluated.

## Fixed 0.5 ensemble results (Macro)

| Model | AUROC | AP | F1 | Accuracy | Sensitivity | Specificity | PPV | NPV | BA | MCC | Brier | NLL | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NoJEPA | 0.7279 | 0.5968 | 0.5416 | 0.6648 | 0.6190 | 0.6863 | 0.4815 | 0.7929 | 0.6527 | 0.2894 | 0.2010 | 0.5822 | 0.1337 |
| DirectReadout | 0.7399 | 0.6075 | 0.5678 | 0.6781 | 0.6607 | 0.6863 | 0.4981 | 0.8113 | 0.6735 | 0.3277 | 0.2001 | 0.5819 | 0.1272 |
| ForecastAttention | 0.7507 | 0.6283 | 0.5604 | 0.6495 | 0.6964 | 0.6275 | 0.4705 | 0.8145 | 0.6619 | 0.3038 | 0.2061 | 0.5990 | 0.1687 |
| ForecastMean | 0.7465 | 0.6139 | 0.5742 | 0.6838 | 0.6667 | 0.6919 | 0.5044 | 0.8154 | 0.6793 | 0.3386 | 0.1958 | 0.5724 | 0.1255 |

## OOF validation-Youden ensemble results (Macro)

| Model | F1 | Sensitivity | Specificity | MCC |
|---|---|---|---|---|
| NoJEPA | 0.5471 | 0.6548 | 0.6527 | 0.2885 |
| DirectReadout | 0.5713 | 0.6548 | 0.7003 | 0.3364 |
| ForecastAttention | 0.5631 | 0.7202 | 0.6050 | 0.3042 |
| ForecastMean | 0.5563 | 0.6012 | 0.7367 | 0.3268 |

## Paired Macro contrasts (fixed 0.5)

- DirectReadout minus NoJEPA, auroc: +0.0121 [-0.0038, +0.0287].
- DirectReadout minus NoJEPA, auprc: +0.0107 [-0.0114, +0.0321].
- DirectReadout minus NoJEPA, f1: +0.0261 [-0.0039, +0.0621].
- DirectReadout minus NoJEPA, brier: -0.0010 [-0.0064, +0.0045].
- DirectReadout minus NoJEPA, nll: -0.0002 [-0.0132, +0.0123].
- DirectReadout minus NoJEPA, ece_10_equal_width: -0.0064 [-0.0460, +0.0224].
- ForecastAttention minus NoJEPA, auroc: +0.0228 [+0.0012, +0.0465].
- ForecastAttention minus NoJEPA, auprc: +0.0315 [-0.0030, +0.0698].
- ForecastAttention minus NoJEPA, f1: +0.0188 [-0.0264, +0.0735].
- ForecastAttention minus NoJEPA, brier: +0.0050 [-0.0072, +0.0173].
- ForecastAttention minus NoJEPA, nll: +0.0169 [-0.0128, +0.0469].
- ForecastAttention minus NoJEPA, ece_10_equal_width: +0.0350 [-0.0220, +0.0582].
- ForecastAttention minus DirectReadout, auroc: +0.0108 [-0.0101, +0.0327].
- ForecastAttention minus DirectReadout, auprc: +0.0208 [-0.0108, +0.0566].
- ForecastAttention minus DirectReadout, f1: -0.0074 [-0.0424, +0.0352].
- ForecastAttention minus DirectReadout, brier: +0.0060 [-0.0056, +0.0177].
- ForecastAttention minus DirectReadout, nll: +0.0171 [-0.0103, +0.0444].
- ForecastAttention minus DirectReadout, ece_10_equal_width: +0.0415 [-0.0072, +0.0570].
- ForecastAttention minus ForecastMean, auroc: +0.0042 [-0.0140, +0.0240].
- ForecastAttention minus ForecastMean, auprc: +0.0144 [-0.0190, +0.0476].
- ForecastAttention minus ForecastMean, f1: -0.0138 [-0.0460, +0.0233].
- ForecastAttention minus ForecastMean, brier: +0.0103 [+0.0012, +0.0194].
- ForecastAttention minus ForecastMean, nll: +0.0266 [+0.0051, +0.0483].
- ForecastAttention minus ForecastMean, ece_10_equal_width: +0.0432 [+0.0075, +0.0607].

## T2 minus T0 paired intervals (fixed 0.5)

- NoJEPA auroc: +0.0006 [-0.0062, +0.0070].
- NoJEPA auprc: +0.0017 [-0.0100, +0.0125].
- NoJEPA f1: +0.0114 [+0.0000, +0.0396].
- DirectReadout auroc: +0.0038 [-0.0064, +0.0140].
- DirectReadout auprc: +0.0061 [-0.0109, +0.0248].
- DirectReadout f1: +0.0002 [-0.0336, +0.0395].
- ForecastAttention auroc: +0.0099 [+0.0013, +0.0187].
- ForecastAttention auprc: +0.0059 [-0.0098, +0.0203].
- ForecastAttention f1: -0.0103 [-0.0535, +0.0340].
- ForecastMean auroc: +0.0041 [-0.0047, +0.0134].
- ForecastMean auprc: +0.0121 [-0.0054, +0.0321].
- ForecastMean f1: +0.0111 [-0.0137, +0.0425].

## Exploratory Macro decision-curve point estimates

| Hypothetical threshold | NoJEPA | DirectReadout | ForecastAttention | ForecastMean | Treat all |
|---|---|---|---|---|---|
| 0.10 | 0.2436 | 0.2455 | 0.2444 | 0.2447 | 0.2444 |
| 0.20 | 0.1833 | 0.1771 | 0.1557 | 0.1748 | 0.1500 |
| 0.30 | 0.1015 | 0.1222 | 0.0890 | 0.1205 | 0.0286 |
| 0.40 | 0.0571 | 0.0616 | 0.0254 | 0.0540 | -0.1333 |
| 0.50 | -0.0152 | -0.0019 | -0.0305 | 0.0038 | -0.3600 |
| 0.60 | -0.0448 | -0.0571 | -0.0057 | -0.0352 | -0.7000 |
| 0.70 | -0.0463 | -0.0413 | 0.0279 | -0.0178 | -1.2667 |
| 0.80 | 0.0362 | 0.0171 | 0.0000 | 0.0210 | -2.4000 |

## Interpretation boundaries

- ForecastAttention has the largest Macro AUROC and AP among the principal variants, but DirectReadout has higher F1 at 0.5; ForecastMean has the largest F1 among all four variants.
- Predictive ranking, classification at 0.5, calibration, and hypothetical utility can disagree. A ranking advantage does not establish calibration or clinical decision benefit.
- ForecastAttention has lower Macro net-benefit point estimates than DirectReadout at hypothetical thresholds 0.20, 0.30, and 0.40; its net benefit at 0.50 is negative (-0.0305). It has higher point estimates at some higher thresholds, so there is no uniform utility advantage. These thresholds were not prespecified clinical decision rules.
- These data use one selected training seed and an internal downstream holdout. Patient bootstrap precision does not support a claim of robustness across seeds or external clinical validation.
- The temporal comparison conditions on 175 patients with all three landmarks. It is descriptive on the common complete-prefix cohort and cannot establish benefit across all development patients.
