from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.image
import numpy as np
import pytest

from ispy2_registration.fallback_visualization import (
    Category,
    DiagnosticBundle,
    ElastixMetricFailure,
    ExampleSlot,
    generate_rigid_fallback_examples,
    abnormal_slice_indices,
    load_parameter_object,
    load_status_records,
    parse_elastix_metric_failure,
    physical_aspect,
    rank_category_records,
    render_example,
    select_example_slots,
    similarity_slice_indices,
    validate_bundle_shapes,
    validate_output_directory,
)
from ispy2_registration.config import RegistrationConfig
from ispy2_registration.models import Geometry


def _status(
    root: Path,
    patient: str,
    visit: str,
    reasons: list[str],
    **qc_values: float | str | bool | None,
) -> Path:
    pair = root / patient / visit
    pair.mkdir(parents=True)
    payload = {
        "patient_id": patient,
        "visit": visit,
        "registration_status": "rigid_fallback",
        "qc": {"reasons": reasons, **qc_values},
    }
    path = pair / "registration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_example_slot_filename_includes_secondary_view() -> None:
    slot = ExampleSlot(
        ordinal=18,
        category=Category.JACOBIAN_NONFINITE,
        patient_id="ISPY2-000104",
        visit="T3",
        status_path=Path("registration.json"),
        reasons=("jacobian_nonfinite",),
        secondary_view=True,
    )

    assert slot.filename == (
        "18_jacobian_nonfinite_ISPY2-000104_T3_secondary_view.png"
    )


def test_selection_uses_metric_then_reason_count_then_pair_id(tmp_path: Path) -> None:
    reason = "jacobian_nonpositive_protected_roi"
    _status(
        tmp_path,
        "ISPY2-3",
        "T1",
        [reason, "ftv_volume_change"],
        protected_roi_fold_fraction=0.4,
    )
    _status(
        tmp_path,
        "ISPY2-2",
        "T1",
        [reason],
        protected_roi_fold_fraction=0.4,
    )
    _status(
        tmp_path,
        "ISPY2-1",
        "T2",
        [reason],
        protected_roi_fold_fraction=0.5,
    )
    _status(
        tmp_path,
        "ISPY2-4",
        "T3",
        [reason],
        protected_roi_fold_fraction=0.3,
    )

    records = load_status_records(tmp_path)
    selected = rank_category_records(records, Category.PROTECTED_FOLD, count=3)

    assert [(record.patient_id, record.visit) for record in selected] == [
        ("ISPY2-1", "T2"),
        ("ISPY2-2", "T1"),
        ("ISPY2-3", "T1"),
    ]
    assert selected[-1].reasons == (reason, "ftv_volume_change")


def test_elastix_error_parser_reads_resolution_and_sample_counts() -> None:
    text = """Installing components at resolution 3 ...
Stopping condition: Error in metric.
Description: ITK ERROR: AdvancedMattesMutualInformationMetric: Too many samples map outside moving image buffer: 1135 / 5000
"""

    assert parse_elastix_metric_failure(text) == ElastixMetricFailure(
        resolution=3,
        outside_samples=1135,
        total_samples=5000,
    )


def test_selection_rejects_missing_category(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="protected_roi_folding has 0 eligible pairs",
    ):
        select_example_slots(tmp_path)


def _bundle(shape: tuple[int, int, int] = (5, 6, 7)) -> DiagnosticBundle:
    geometry = Geometry(
        shape_zyx=shape,
        spacing_xyz=(1.0, 1.0, 2.0),
        direction=tuple(np.eye(3).ravel()),
    )
    z, y, x = np.mgrid[: shape[0], : shape[1], : shape[2]]
    fixed = (z + y + x).astype(np.float32)
    rigid = fixed + 0.1
    candidate = fixed + 0.2
    mask = np.zeros(shape, dtype=np.uint8)
    mask[2, 3, 4] = 1
    protected = np.zeros(shape, dtype=bool)
    protected[1:4, 2:5, 3:6] = True
    jacobian = np.ones(shape, dtype=np.float32)
    jacobian[2, 3, 4] = -0.2
    jacobian[4, 0, 0] = -0.3
    jacobian[0, 0, 0] = np.nan
    return DiagnosticBundle(
        fixed=fixed,
        rigid=rigid,
        candidate=candidate,
        fixed_ftv=mask,
        rigid_ftv=mask.copy(),
        candidate_ftv=mask.copy(),
        candidate_jacobian=jacobian,
        protected_roi=protected,
        valid_fov=np.ones(shape, dtype=bool),
        fixed_geometry=geometry,
    )


def test_abnormal_slice_indices_intersect_requested_region() -> None:
    region = np.zeros((5, 6, 7), dtype=bool)
    region[3, 1, 5] = True

    assert abnormal_slice_indices(region) == (3, 1, 5)


def test_secondary_indices_use_a_different_abnormal_location() -> None:
    region = np.zeros((7, 8, 9), dtype=bool)
    region[1, 2, 3] = True
    region[5, 6, 7] = True

    primary = abnormal_slice_indices(region)
    secondary = abnormal_slice_indices(region, exclude=primary)

    assert secondary != primary
    assert region[secondary]


def test_similarity_slices_maximize_regional_residual_increase() -> None:
    fixed = np.zeros((4, 5, 6), dtype=np.float32)
    rigid = np.zeros_like(fixed)
    candidate = np.zeros_like(fixed)
    region = np.zeros_like(fixed, dtype=bool)
    region[2, 3, 4] = True
    candidate[2, 3, 4] = 9

    assert similarity_slice_indices(fixed, rigid, candidate, region) == (2, 3, 4)


def test_validate_bundle_shapes_rejects_non_t0_shapes() -> None:
    broken = replace(_bundle(), candidate=np.zeros((1, 1, 1), np.float32))

    with pytest.raises(ValueError, match="candidate shape differs from T0"):
        validate_bundle_shapes(broken)


def test_load_parameter_object_requires_retained_maps(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing retained transform files"):
        load_parameter_object(tmp_path)


def test_physical_aspect_uses_plane_spacing() -> None:
    geometry = _bundle().fixed_geometry

    assert physical_aspect(geometry, 0) == pytest.approx(1.0)
    assert physical_aspect(geometry, 1) == pytest.approx(2.0)
    assert physical_aspect(geometry, 2) == pytest.approx(2.0)


_REASON_FOR_CATEGORY = {
    Category.PROTECTED_FOLD: "jacobian_nonpositive_protected_roi",
    Category.OUTSIDE_FOLD: "outside_protected_roi_fold_fraction",
    Category.PROTECTED_NCC: "protected_roi_similarity_regressed",
    Category.NON_TUMOR_NCC: "non_tumor_similarity_regressed",
    Category.BSPLINE_FAILED: "bspline_failed",
    Category.JACOBIAN_NONFINITE: "jacobian_nonfinite",
}


@pytest.mark.parametrize("category", list(Category))
def test_render_example_writes_common_nonblank_png(
    tmp_path: Path,
    category: Category,
) -> None:
    slot = ExampleSlot(
        ordinal=1,
        category=category,
        patient_id="ISPY2-1",
        visit="T1",
        status_path=tmp_path / "registration.json",
        reasons=(_REASON_FOR_CATEGORY[category],),
    )
    bundle = _bundle()
    if category is Category.BSPLINE_FAILED:
        bundle = replace(
            bundle,
            candidate=None,
            candidate_ftv=None,
            candidate_jacobian=None,
        )
    destination = tmp_path / slot.filename
    qc = {
        "protected_roi_fold_fraction": 0.01,
        "outside_protected_roi_fold_fraction": 0.03,
        "rigid_protected_ncc": 0.8,
        "deformable_protected_ncc": 0.6,
        "rigid_non_tumor_ncc": 0.7,
        "deformable_non_tumor_ncc": 0.5,
    }
    failure = (
        ElastixMetricFailure(3, 100, 5000)
        if category is Category.BSPLINE_FAILED
        else None
    )

    render_example(
        slot,
        bundle,
        destination,
        qc=qc,
        elastix_failure=failure,
    )

    image = matplotlib.image.imread(destination)
    assert image.shape[:2] == (1800, 2400)
    assert float(np.std(image)) > 0.01


def test_bspline_failure_renderer_requires_diagnostic(tmp_path: Path) -> None:
    slot = ExampleSlot(
        ordinal=1,
        category=Category.BSPLINE_FAILED,
        patient_id="ISPY2-1",
        visit="T1",
        status_path=tmp_path / "registration.json",
        reasons=("bspline_failed",),
    )
    bundle = replace(
        _bundle(),
        candidate=None,
        candidate_ftv=None,
        candidate_jacobian=None,
    )

    with pytest.raises(ValueError, match="requires elastix failure diagnostics"):
        render_example(
            slot,
            bundle,
            tmp_path / "failure.png",
            qc={},
            elastix_failure=None,
        )


def test_generation_refuses_nonempty_destination(tmp_path: Path) -> None:
    destination = tmp_path / "examples"
    destination.mkdir()
    (destination / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="destination is nonempty"):
        generate_rigid_fallback_examples(
            (),
            tmp_path,
            RegistrationConfig(),
            destination,
        )


def test_validate_output_requires_exact_file_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected 18 PNG files"):
        validate_output_directory(tmp_path)
