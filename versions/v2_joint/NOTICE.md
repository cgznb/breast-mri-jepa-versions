# Provenance and scope

This independent v2 implementation was prepared against the public source/data contracts of
`cgznb/breast-mri-vla-jepa`, commit `[commit reference omitted]`.
The upstream MIT attribution is retained for architecture/data-contract adaptations.
It is not an archive of the entire upstream repository, and does not claim to reproduce its
historical checkpoints or experimental scores.

VLA-JEPA query-based task/state learning and VICReg-style variance/covariance objectives
are conceptual references. These components are independently implemented here with PyTorch;
no third-party foundation-model source tree or pretrained weights are bundled.

Synthetic fixtures are explicitly fictitious and are only for software tests. Real medical
images, clinical records, patient-level predictions and trained checkpoints are not bundled.
Code licensing does not replace I-SPY2 or any other medical data use agreement.

Full checkpoints store optimizer/RNG state using Python serialization. Load only checkpoints
from a source you trust. Do not upload patient-level outputs or source data to public repositories.
