This is a demo for data processing and modeling pipeline for building MRI-informed breast cancer response prediction model

Please refers to these papers for more details on the methodology and associated image processing pipeline:

Jarrett AM, Kazerouni AS, Wu C, Virostko J, Sorace AG, DiCarlo JC, Hormuth DA, Ekrut DA, Patt D, Goodgame B, Avery S. Quantitative magnetic resonance imaging and tumor forecasting of breast cancer patients in the community setting. Nature protocols. 2021;16(11):5309-38.

Wu C, Jarrett AM, Zhou Z, Elshafeey N, Adrada BE, Candelaria RP, Mohamed RM, Boge M, Huo L, White JB, Tripathy D. MRI-based digital models forecast patient-specific treatment responses to neoadjuvant chemotherapy in triple-negative breast cancer. Cancer research. 2022;82(18):3394-404.

Further questions and discussion are welcome through the repository, or via email (cwu19@mdanderson.org, cw35926@utexas.edu, ajarrett@utexas.edu).

## Python longitudinal registration to T0

The `ispy2-registration` package converts the repository's F4 inter-visit
registration sequence to Python for the current ISPY2 NIfTI/JSON dataset. It
does not convert the modeling pipeline.

The registration contract is:

- T0 is fixed; each available T1, T2, and T3 is registered independently.
- Within every visit, each actual DCE phase after phase 0 is rigidly registered
  to phase 0. Phase counts and IDs come from the visit JSON and are not fixed.
- The voxelwise mean of the internally aligned DCE phases is used only to
  estimate the visit-to-T0 transform.
- Each original DCE phase is resampled once using its phase-to-phase rigid
  transform composed after the selected visit-to-T0 transform.
- Existing SER and PE maps are not recomputed. They receive only the selected
  visit-to-T0 transform, as do the FTV and raw analysis Masks.
- Raw analysis Mask voxels outside the moving field of view are filled with
  label 128. All other registered modalities retain zero padding.
- A whole-breast Mask and ADC/DWI are not registration inputs.
- Empty future FTV Masks use intensity-only B-spline registration.
- Source NIfTI files are never modified.

### Environment

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/pytest -q
```

The locked registration dependencies are ITK 5.4.6 and itk-elastix 0.25.3.
Runtime defaults are loaded from `config/ispy2_registration.toml`. Explicit
`--manifest`, `--output-root`, `--workers`, `--elastix-threads`, and `--seed`
values override the TOML values. The immutable Python dataclasses provide
fallback defaults for omitted TOML fields.

Rigid registration uses Mattes mutual information with a three-level
`(4, 2, 1)` shrink schedule, `(2, 1, 0)` mm smoothing, and
`RegularStepGradientDescent` with learning rate 2.0, minimum step 0.0001, and
relaxation factor 0.5.

### Commands

```bash
.venv/bin/ispy2-register audit --config config/ispy2_registration.toml

.venv/bin/ispy2-register register-patient ISPY2-000101 \
  --config config/ispy2_registration.toml

.venv/bin/ispy2-register register-cohort \
  --config config/ispy2_registration.toml \
  --patients ISPY2-000101,ISPY2-000102,ISPY2-000103 \
  --resume

.venv/bin/ispy2-register register-cohort \
  --config config/ispy2_registration.toml \
  --resume
```

The default output root is:

```text
/root/autodl-tmp/Datasets/manifest-1781750940386/ISPY2/processed_ispy2_world_phase3_scan_phase_aligned_multires_registered_to_T0
```

This is a fresh output tree. The older
`processed_ispy2_world_phase3_scan_phase_aligned_registered_to_T0` tree is not
used as a resume source and must remain unchanged.

Each published future visit contains registered modalities, updated metadata,
per-phase rigid transforms and QC, Elastix parameter files, a
combined-transform descriptor, the selected longitudinal displacement field,
the Jacobian determinant, an orthogonal QC image, and `registration.json`.
Pair directories are published atomically. With `--resume`, complete reference
and moving visits are skipped before phase alignment or other preprocessing. A
pair is complete only when its status is terminal and every file declared by
`registration.json` exists; incomplete visits are recomputed.

Valid output states are `fixed_reference`, `deformable`, and
`rigid_fallback`. An intra-visit phase failure is recorded as
`failed_intra_visit`. Patients without T0 receive `skipped_no_T0`. A failed
pair does not stop other pairs; cohort mode returns a nonzero status and writes
`manifest/registration_failures.json` when failures remain.

Deformable QC protects the union of fixed and transformed FTV Masks plus a
10 mm physical margin. Folding in that protected region, more than 10 percent
FTV volume change, or similarity regression causes rigid fallback. Folding
outside the protected region is accepted up to 2 percent of outside-region
voxels and is explicitly marked with
`qc_status=warning_outside_protected_roi_folding`. Candidate and accepted
folding flags, protected/outside/global fold fractions, and warning codes are
stored in the visit JSON, `registration.json`, and output manifest.

### Qingyuan ACRIN plus ISPY2 workflow

The multimodal preparation commands identify patients and series from DICOM
headers. Literal quote characters in TCIA directory names are not interpreted
or renamed. Original DICOM roots are read-only; retry downloads and all derived
files live below `/data1/cxx/ispy2_acrin_t0_registration_v1`.

```bash
ispy2-register audit-dicom \
  --dicom-root ORIGINAL_ROOT RETRY_ROOT \
  --output-root /data1/cxx/ispy2_acrin_t0_registration_v1 \
  --collection ACRIN-6698 --deep

ispy2-register prepare-dicom \
  --dicom-root ORIGINAL_ROOT RETRY_ROOT \
  --output-root /data1/cxx/ispy2_acrin_t0_registration_v1/preprocessed \
  --collection ACRIN-6698 --workers 4 --resume --skip-dwi

ispy2-register combine-manifests \
  --manifests ACRIN_PREPROCESSED.csv ISPY2_PREPROCESSED.csv \
  --destination /data1/cxx/ispy2_acrin_t0_registration_v1/manifest/ispy2_acrin_preprocessed_manifest.csv

ispy2-register register-cohort \
  --config config/qingyuan_ispy2_acrin_registration.toml --resume --skip-dwi
```

`--skip-dwi` runs the DCE-first path and records complete DWI candidates as
`deferred`, not missing. A later preparation and registration run without
`--skip-dwi` reopens only deferred visits while retaining completed DCE work.

DWI never contributes to the DCE longitudinal transform estimate. When DWI
and DCE share a `FrameOfReferenceUID`, their DICOM origins define the grid
mapping. Otherwise a Mattes-MI `b0 -> DCE0` rigid fallback is estimated. DWI,
ADC, and SEG are composed with the DCE-to-T0 transform and resampled once;
DWI/ADC use linear interpolation and SEG uses nearest neighbor. DWI failures
are recorded without failing the visit's DCE registration.
