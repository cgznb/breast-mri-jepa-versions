# Numerical verification of v3 seed44 probability ensembles

This audit reconstructs patient-level five-fold mean probabilities directly from the registered JSON reports in memory. It exports aggregate comparisons only.

- Independently compared all 18 landmark/policy quantities at 12 model/landmark combinations and two policies against scikit-learn, with explicit NPV, specificity, ECE and confusion-count formulas where appropriate.
- Independently selected all 12 pooled OOF thresholds with `sklearn.metrics.roc_curve(..., drop_intermediate=False)` and the largest-threshold tie rule.
- Checked 300 weighted patient-bootstrap replicates against scikit-learn AUROC, AP, F1, accuracy and sensitivity.
- The resulting 1,932 scalar metric comparisons agree within 7.78e-16.
- Checked all 1,216 landmark/Macro DCA points from independently counted TP/FP values; maximum absolute discrepancy is 7.22e-16.
- The main script additionally verifies all published four-decimal AUROC/AP/F1 targets, 20 completed receipts, 60 registered report size/mtime identities, source manifest membership and labels, disjoint OOF validation coverage, and common 175-patient test alignment.

Environment: Python 3.12.1, NumPy 1.26.4, scikit-learn 1.9.1.
Analysis script SHA256: `c74695888dd66fb12d8432e32b14db31ba18908c7631e53ea6d16351e6a8bcb8`.

The numerical audit establishes agreement of calculations. It does not add external validation, training-seed replication, clinical utility, or ensemble calibration evidence.
