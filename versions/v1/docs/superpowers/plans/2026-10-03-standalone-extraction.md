# Standalone MRI VLA-JEPA Implementation Plan

**Goal:** Extract the fixed-T0 and dynamic-T0–T3 VLA-style B models into an independently installable source package and ZIP.

**Architecture:** Use the `mri_vla_jepa` namespace. Extract only the required MRI encoder, optional flow module, data contracts, training helpers and new VLA-style model; keep the existing raw manifest and checkpoint schemas compatible. Keep both versions as configurations of one shared implementation.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, PyYAML, scikit-learn; optional nibabel/pandas/openpyxl; pytest.

- [x] Extract model classes into `model.py`, `model_config.py`, `encoder.py`, and `flow.py`; migrate model and boundary tests.
- [x] Extract dataset, contracts, Arm definitions, preparation script and synthetic examples; migrate data tests.
- [x] Extract independent training configuration and helpers, trainer and CLI; migrate trainer tests and four configs.
- [x] Add package metadata, entry points, IO/metrics, Chinese usage documentation and retained source license.
- [x] Run all standalone tests with `/private/tmp/breast-v3-validation-venv/bin/python -m pytest -q` from this directory.
- [x] Compare extracted model weights and deterministic outputs against the original new model.
- [x] Build and install the standalone wheel, run both CLI smoke profiles from another working directory, and verify no `responsewm` module is imported.
- [x] Archive only source/configuration/tests/docs; extract the ZIP into a temporary directory and verify its entry point and tests there.
- [x] Record exact validation commands and deliver absolute links to the directory and ZIP.

- [x] Correct formal/default MRI dimensions to the original 32×128×128 contract, validate a full-resolution backward pass and refresh the ZIP. Synthetic smoke dimensions remain 8×16×16.
