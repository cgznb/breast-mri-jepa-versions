# Extended seed21/44 analysis

All 40 model/seed/fold receipts and their 120 report identities passed. Test predictions remain internal; only aggregate values are exported.
The 18 reported quantities comprise 14 metrics plus TP/FP/FN/TN counts. Counts and metrics are averaged across separately evaluated folds.
Alternative thresholds use per-landmark validation patients only, so missing validation prefixes do not require pooling repeated patient rows.
Bootstrap intervals condition on the trained models and fixed validation-selected thresholds, and exclude retraining, threshold-selection and seed-selection uncertainty.
Decision curves use a hypothetical risk-threshold utility with uncalibrated probabilities; their net benefit does not establish clinical usefulness.

## Two-seed three-landmark fixed-threshold means

| Model | Accuracy | Sensitivity | Specificity | PPV | NPV | BA | MCC |
|---|---|---|---|---|---|---|---|
| NoJEPA | 0.6743 | 0.5875 | 0.7151 | 0.4954 | 0.7888 | 0.6513 | 0.2931 |
| DirectReadout | 0.6790 | 0.5970 | 0.7176 | 0.5011 | 0.7922 | 0.6573 | 0.3037 |
| ForecastAttention | 0.6509 | 0.6673 | 0.6431 | 0.4796 | 0.8143 | 0.6552 | 0.3013 |
| ForecastMean | 0.6594 | 0.6196 | 0.6782 | 0.4900 | 0.7999 | 0.6489 | 0.2934 |

## Two-seed validation-Youden three-landmark means

| Model | F1 | Sensitivity | Specificity | MCC |
|---|---|---|---|---|
| NoJEPA | 0.5349 | 0.6143 | 0.6838 | 0.2851 |
| DirectReadout | 0.5371 | 0.6006 | 0.7059 | 0.2976 |
| ForecastAttention | 0.5418 | 0.6405 | 0.6605 | 0.2885 |
| ForecastMean | 0.5293 | 0.5964 | 0.7045 | 0.2973 |

## T2 minus T0 paired intervals (fixed 0.5)

- NoJEPA auroc: +0.0003 [-0.0059, +0.0067].
- NoJEPA auprc: -0.0002 [-0.0166, +0.0128].
- NoJEPA f1: -0.0018 [-0.0152, +0.0113].
- DirectReadout auroc: +0.0016 [-0.0038, +0.0064].
- DirectReadout auprc: +0.0073 [-0.0015, +0.0147].
- DirectReadout f1: +0.0218 [+0.0049, +0.0405].
- ForecastAttention auroc: +0.0071 [+0.0006, +0.0138].
- ForecastAttention auprc: +0.0065 [-0.0038, +0.0168].
- ForecastAttention f1: +0.0165 [-0.0010, +0.0341].
- ForecastMean auroc: +0.0001 [-0.0061, +0.0061].
- ForecastMean auprc: +0.0003 [-0.0117, +0.0118].
- ForecastMean f1: -0.0081 [-0.0265, +0.0092].
