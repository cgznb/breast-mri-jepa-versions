from __future__ import annotations

import numpy as np
import pytest

from ispy2_registration.models import Geometry
from ispy2_registration.qc import (
    QCInputs,
    QCThresholds,
    evaluate_deformable_qc,
    masked_ncc,
)


GEOMETRY = Geometry(
    shape_zyx=(2, 3, 4),
    spacing_xyz=(1.0, 1.0, 2.0),
    direction=tuple(np.eye(3).ravel()),
)


def _inputs() -> QCInputs:
    fixed = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    empty = np.zeros_like(fixed, dtype=np.uint8)
    moving_ftv = empty.copy()
    moving_ftv[0, 1, 1:3] = 1
    registered_ftv = moving_ftv.copy()
    return QCInputs(
        fixed_image=fixed,
        rigid_image=fixed.copy(),
        deformable_image=fixed.copy(),
        fixed_ftv=empty,
        rigid_ftv=registered_ftv,
        deformable_ftv=registered_ftv,
        source_moving_ftv=moving_ftv,
        source_geometry=GEOMETRY,
        target_geometry=GEOMETRY,
        jacobian=np.ones_like(fixed, dtype=np.float32),
    )


def test_masked_ncc_ignores_excluded_region() -> None:
    fixed = np.arange(8, dtype=np.float32)
    moving = fixed.copy()
    moving[-1] = -100
    excluded = np.zeros(8, dtype=bool)
    excluded[-1] = True

    assert masked_ncc(fixed, moving, excluded) == pytest.approx(1.0)


def test_qc_accepts_finite_nonfolding_similarity_improvement() -> None:
    result = evaluate_deformable_qc(_inputs(), QCThresholds())

    assert result.accepted
    assert result.status == "deformable"
    assert result.reasons == ()


def _large_inputs() -> QCInputs:
    shape = (10, 20, 20)
    geometry = Geometry(
        shape_zyx=shape,
        spacing_xyz=(1.0, 1.0, 1.0),
        direction=tuple(np.eye(3).ravel()),
    )
    fixed = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    ftv = np.zeros(shape, dtype=np.uint8)
    ftv[5, 10, 10] = 1
    return QCInputs(
        fixed_image=fixed,
        rigid_image=fixed.copy(),
        deformable_image=fixed.copy(),
        fixed_ftv=ftv.copy(),
        rigid_ftv=ftv.copy(),
        deformable_ftv=ftv.copy(),
        source_moving_ftv=ftv.copy(),
        source_geometry=geometry,
        target_geometry=geometry,
        jacobian=np.ones(shape, dtype=np.float32),
    )


def test_qc_accepts_limited_outside_roi_folding_with_warning() -> None:
    inputs = _large_inputs()
    jacobian = inputs.jacobian.copy()
    jacobian[0, 0, :20] = 0
    inputs = QCInputs(**{**inputs.__dict__, "jacobian": jacobian})

    result = evaluate_deformable_qc(
        inputs,
        QCThresholds(protected_margin_mm=1.0, max_outside_fold_fraction=0.02),
    )

    assert result.accepted
    assert result.status == "deformable"
    assert result.qc_status == "warning_outside_protected_roi_folding"
    assert result.protected_roi_fold_fraction == 0.0
    assert 0.0 < result.outside_protected_roi_fold_fraction < 0.02
    assert result.candidate_outside_protected_roi_folding
    assert result.outside_protected_roi_folding_accepted
    assert result.qc_warning_codes == (
        "candidate_outside_protected_roi_folding",
        "outside_protected_roi_folding_accepted",
    )


def test_qc_rejects_outside_roi_folding_over_two_percent() -> None:
    inputs = _large_inputs()
    jacobian = inputs.jacobian.copy()
    jacobian.reshape(-1)[:100] = -0.1
    inputs = QCInputs(**{**inputs.__dict__, "jacobian": jacobian})

    result = evaluate_deformable_qc(
        inputs,
        QCThresholds(protected_margin_mm=1.0, max_outside_fold_fraction=0.02),
    )

    assert not result.accepted
    assert "outside_protected_roi_fold_fraction" in result.reasons
    assert result.outside_protected_roi_fold_fraction > 0.02


def test_qc_rejects_any_folding_inside_protected_roi() -> None:
    inputs = _large_inputs()
    jacobian = inputs.jacobian.copy()
    jacobian[5, 10, 10] = 0
    inputs = QCInputs(**{**inputs.__dict__, "jacobian": jacobian})

    result = evaluate_deformable_qc(inputs, QCThresholds(protected_margin_mm=1.0))

    assert not result.accepted
    assert "jacobian_nonpositive_protected_roi" in result.reasons
    assert result.protected_roi_fold_fraction > 0.0


def test_qc_rejects_non_tumor_similarity_below_rigid() -> None:
    inputs = _inputs()
    worse = np.flip(inputs.fixed_image, axis=2).copy()
    inputs = QCInputs(**{**inputs.__dict__, "deformable_image": worse})

    result = evaluate_deformable_qc(inputs, QCThresholds())

    assert "non_tumor_similarity_regressed" in result.reasons


def test_qc_rejects_similarity_regression_in_peritumoral_shell() -> None:
    inputs = _large_inputs()
    worse = inputs.deformable_image.copy()
    worse[4:7, 9:12, 9:12] = np.flip(worse[4:7, 9:12, 9:12], axis=2)
    inputs = QCInputs(**{**inputs.__dict__, "deformable_image": worse})

    result = evaluate_deformable_qc(
        inputs,
        QCThresholds(protected_margin_mm=2.0),
    )

    assert "protected_roi_similarity_regressed" in result.reasons
    assert result.deformable_protected_ncc < result.rigid_protected_ncc


def test_qc_rejects_ftv_physical_volume_change_over_ten_percent() -> None:
    inputs = _inputs()
    larger = inputs.deformable_ftv.copy()
    larger[1, 1, 1] = 1
    inputs = QCInputs(**{**inputs.__dict__, "deformable_ftv": larger})

    result = evaluate_deformable_qc(inputs, QCThresholds(max_ftv_volume_change=0.10))

    assert result.ftv_volume_change == pytest.approx(0.5)
    assert "ftv_volume_change" in result.reasons


def test_fallback_keeps_candidate_outside_roi_folding_marker() -> None:
    inputs = _large_inputs()
    larger = inputs.deformable_ftv.copy()
    larger[5, 10, 11] = 1
    jacobian = inputs.jacobian.copy()
    jacobian[0, 0, 0] = 0
    inputs = QCInputs(
        **{**inputs.__dict__, "deformable_ftv": larger, "jacobian": jacobian}
    )

    result = evaluate_deformable_qc(
        inputs,
        QCThresholds(
            max_ftv_volume_change=0.10,
            protected_margin_mm=1.0,
            max_outside_fold_fraction=0.02,
        ),
    )

    assert result.status == "rigid_fallback"
    assert "ftv_volume_change" in result.reasons
    assert result.candidate_outside_protected_roi_folding
    assert not result.outside_protected_roi_folding_accepted
    assert result.qc_warning_codes == ("candidate_outside_protected_roi_folding",)


def test_empty_source_ftv_skips_volume_rule() -> None:
    inputs = _inputs()
    empty = np.zeros_like(inputs.source_moving_ftv)
    inputs = QCInputs(
        **{
            **inputs.__dict__,
            "source_moving_ftv": empty,
            "deformable_ftv": np.ones_like(empty),
        }
    )

    result = evaluate_deformable_qc(inputs, QCThresholds())

    assert result.ftv_volume_change is None
    assert "ftv_volume_change" not in result.reasons
