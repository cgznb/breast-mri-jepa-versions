# Medium976 cohort and preprocessing evidence

Read-only audit for the seed44 MIA draft, 2026-11-02. No MRI, mask, geometry,
manifest, checkpoint, or experiment result was changed. Only manuscript files
were written. Patient identifiers are deliberately omitted from this note.

## Current authoritative inputs

- Dataset root: `/personal_fast_storage/VLA-PCRV1/data/registered_roi32_medium976_v1`.
- `README.md`: full protocol, version-matched geometry, exclusions, normalisation,
  clipping and independence limits.
- `manifest.json`: fold1 copy, 976 patient records. `clinical_features` is
  `[age_at_screening, hr_positive, her2_positive, mammaprint_binary]`.
- `manifest_summary.json`: final per-stage, split and fold counts.
- `completion_receipt.json`: completed final export; 976 patients, 3641 usable
  visits, five folds, fixed175, 85 largest-component clips, final verification.
- `normalization/fold1_clinical_state.json`: training-only age mean/std;
  binary fields mean0/std1 (retained as released), three missing ages in fold1
  training; explicit clinical counts and intensity contract.
- `geometry/*.json`: 976 frozen sampling plans and source namespaces.

## Independently recomputed aggregates

Computed using Python standard library `json`, `collections`, and `statistics`
directly from the final `manifest.json`, grouping records by current split.
Development is `split != test`, so each development patient is counted once.
Sample age SD uses `statistics.stdev`; model normalisation instead uses its
own saved population scale and is not estimated from this summary table.

| Field | All | Development | Holdout |
|---|---:|---:|---:|
| Patients | 976 | 801 | 175 |
| pCR positive | 313 | 257 | 56 |
| pCR negative | 663 | 544 | 119 |
| Observed age n | 973 | 798 | 175 |
| Age mean | 48.7266187050 | 48.6854636591 | 48.9142857143 |
| Age sample SD | 10.5212542368 | 10.6062202919 | 10.1522078131 |
| Age median | 49 | 49 | 49 |
| Age minimum | 23 | 23 | 25 |
| Age maximum | 77 | 77 | 71 |
| HR positive | 533 | 434 | 99 |
| HER2 positive | 240 | 195 | 45 |
| MammaPrint released indicator=1 | 477 | 393 | 84 |
| T0 examinations | 976 | 801 | 175 |
| T1 examinations | 935 | 760 | 175 |
| T2 examinations | 875 | 700 | 175 |
| T3 examinations | 855 | 692 | 163 |

Clinical HR/HER2/MammaPrint have no missing values in this manifest. Exactly
three age values are missing, all in the development population. Assigned-arm
records are present for all 976 patients, use 13 distinct official labels, and
all have `semantics=assigned_arm_scenario` with baseline `known_at_stage=0`.
Do not relabel this as prospectively verified treatment timing or infer doses,
cycle schedules, treatment changes, or personalised treatment effects.

HR/HER2 joint counts, all/development/holdout:

- HR-/HER2-: 356 / 294 / 62.
- HR+/HER2-: 380 / 312 / 68.
- HR+/HER2+: 153 / 122 / 31.
- HR-/HER2+: 87 / 73 / 14.

The draft does not label these as detailed pathological subtypes because the
release's ER/PR definitions and clinical coding need source confirmation.

## Source collection identity

`docs/DATA_PROVENANCE_ZH.md` states the official mapping is
`Full-Collection-Ancillary-Patient-Information-file.xlsx`, with labels from
`ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx`. Legacy directory names are storage
history, not independent clinical cohort labels.

Across all 976 frozen `geometry/*.json`, the `native_patient_id` prefixes
identify 712 ISPY2 and 264 ACRIN-6698 source identifiers. Fold1's fixed holdout
has 125 ISPY2 and 50 ACRIN-6698 identifiers. All final patient records are
linked to the same I-SPY2 clinical release. Do not claim 264 constitutes an
external validation cohort. Official source pages were independently obtained
by the references agent: `sources/official/tcia_ispy2.html` and
`sources/official/tcia_acrin6698.html`. Use `tciaispy2_2022` (Li et al., 2022,
Version 1, DOI `10.7937/TCIA.D8Z0-9T85`) and `tciaacrin6698_2021` (Newitt et al.,
2021, DOI `10.7937/TCIA.KK02-6D95`). The official ISPY2 page explicitly describes
the shared Imaging Cohort 1 and a download manifest covering both source
collections. Licence, recruitment centres, acquisition ranges and canonical
visit timing still require source-specific confirmation as applicable.

## Patient and examination eligibility

Current source policy and audit:

- `private_outputs/medium_reselection_20261028/README.md`.
- `private_outputs/medium_reselection_20261028/selection_summary.json`.
- `private_outputs/medium_reselection_20261028/selection_index.json`.
- `private_outputs/medium_reselection_20261028/selection_policy.json`.

980 clinically matched source candidates (historical source batches600+58+322)
were considered. Exclusions were 3 without actual T0 and 1 empty/unavailable
T0 MAMA-MIA localisation, producing976 patients and3642 candidate visits.
Historical source batches are not described as separate independent datasets.
All old866 patients and3039 visits were retained, but this final976 cohort
uses a different patient/visit admission rule: low-NCC/FOV/FTV/adjacent-pair
rules were removed. Do not transfer the old Strict-A866 exclusion ledger to
this cohort.

The final dataset's `availability_overrides.json` records one unusable T3 due
to saved intravisit transforms shifting postcontrast images outside the fixed
T0 crop. The source acquisition itself is nonempty. T0--T2 and the patient are
retained; finalT3 availability855, not candidate856. Thus finalvisit3641, not
3642. Twenty-three T0-only patients remain for pCR supervision with all future
MRI target masks false. Eight availability patterns are represented.

## Split construction and independence

Evidence:

- `private_outputs/medium976_split_audit_20261028/README.md`.
- `private_outputs/medium976_split_audit_20261028/split_policy.json`.
- `private_outputs/medium976_split_audit_20261028/split_summary.json`.
- `private_outputs/medium976_split_audit_20261028/packaged_data_verification.json`.
- `tools/medium976/build_manifests.py`, especially `fit_image_scale` and manifest
  `provenance` fields.

All175 holdoutpatients have realT0/T1/T2; T3available163. The139 previously
complete historical holdout patients are retained and36 added. The36 selection
uses pCRquotas25negative/11positive, T3-missing strata, baseline categorical
balancing, and RNGseed2026. It samples10000 candidate sets, then deterministic
improving within-stratum swaps. `read_model_predictions=false` and
`performance_inputs=[]`. Balancing inputs include HR/HER2/MP/Arm/ageband/
menopause/sourcebatch/missingness; these split-design inputs are not all model
input features.

Of36 added holdoutpatients,29 were olddevelopment,2 were oldholdout outside the
139 and5 newly admitted. Current downstream runs use the new fold splits, but
prior models cannot be called heldout on new175. Pretrained codec and
MAMA-MIA checkpoint training-patient overlap with new175 has not been audited
in this task. The current model receipt explicitly says
`downstream_holdout=true`, `independent_test=false`.

Development uses patient-level stratified fivefold CV with seed2026 and joint
pCR/visitpattern strata, rare strata merged within pCR. Fold1 train640/val161;
folds2--5 train641/val160. Every developmentpatient is validation exactly once.
This is five overlapping training sets evaluated on the same175 people, not
five independent test cohorts.

## Preprocessing and geometry

Current and historical implementation/evidence:

- `tools/medium976/geometry.py`: explicit version-matched metadata/mask/phase
  mapping, fixedlargest-component crop, sourceidentities, no mutation.
- `tools/medium976/export_roi32.py`: MRIorder1 interpolation, masks/supportorder0,
  finite arrays, zero background, compression/readback check, legacyreplay.
- `docs/DATA_PROVENANCE_ZH.md`: historical intravisit Euler3D to precontrast,
  longitudinal rigid then Elastix B-spline with rigidfallback, MAMA-MIA on
  nativefirstpost and largest26connectedcomponent crop.
- `docs/PREPROCESSING_SOURCES_ZH.md`: archivedsourcecode, exactrevision limits,
  segmentation and codec lineage.

Finalimagefloat32 shape[3,32,128,128]; phases precontrast, firstpost, and explicit
metadata-defined late (not fixedlastphase). SpacingZYX[2.0,.7032,.7032]mm, common
prescribedresampling spacing. Nativeacquisition metadata and oldmaskheader must
stay paired; currentpublicJSON cannot be substituted for historicalnativeJSON
alone. Registeredoutputgeometry is centredregistration coordinates, not native
maskgeometry. MRI sampled via `crop_to_registered_index_zyx`, mask via
`crop_to_native_mask_index_zyx`; placeholderNIfTI affine is not authoritative.

Eachpatient uses T0 nativefirstpostMAMA-MIA prediction to locate onecrop; no
followupsegmentation/recentering. TheNPZ `roi` is sameT0sampledmask at allvisits,
not futuregroundtruth. Mainclassifier does not use segmentation as a separate
input (see model agent's runtime audit); masks localise the inputcrop.

Largestpredictedcomponent partlyclipped85patients; anycomponent incompletely
covered268patients. Thelatter counts multicompartment masks and should not be
conflated with85. These are retainedflags, not exclusions. Geometrynumerical
validation does not prove per-caseanatomicalregistration or segmentation
accuracy; geometryplan `segmentation_review_status=pending_visual_review`.

## Normalisation

RawNPZs shared across folds, no fittednormalisation on disk. Perfold training
patients' uniqueactualvisits only, DCE0explicitnonzero foreground voxels,
populationmean/std (ddof0); a singlecommonscalar applied once toforeground of
all3phases/allstages/allsplits, preservingzerobackground. Neitherperpatient nor
perphasezscore. No validation/test fit identities overlap; independentpackaged
verification recomputes statistics. Clinicalnormalisationfits trainingonly;
age standardized, officialbinarycodes unchanged withmask handling.

## Remaining author/source confirmations

1. Data download and reuse licence terms (collection citations now verified).
2. Original ethics/consent and current institutional waiver statements.
3. Release-specific pCR pathological definition; binarycolumn alone does not
   establish nodal/in-situ criteria.
4. CanonicalT1/T2/T3 treatmenttiming, centres, scanners, sequenceacquisition.
5. Codec and segmentation pretraining patientmembership relative to final175.
6. Quantitative/anatomical registration and localisation quality beyond array
   and geometryconsistency, including cropped-lesion sensitivity.
7. Exact oldupstreamregistration coderevision when cachedimages were generated;
   archivedcode describes method but cannot retrospectively prove revision.

No statements of externalvalidation, independently pretrainedholdout,
prospectivearmavailability, exactpCRpathologicalcriteria, or universal
anatomicalregistrationquality were invented.
