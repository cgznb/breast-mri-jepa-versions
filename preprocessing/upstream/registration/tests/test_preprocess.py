from __future__ import annotations

import numpy as np
import pytest

from ispy2_registration.preprocess import (
    average_dce,
    bspline_driver,
    pad_z,
    prepare_ftv_mask,
    prepare_pair_drivers,
    rigid_driver,
    slice_clahe,
)


def test_average_dce_is_voxelwise_phase_mean() -> None:
    phases = [
        np.zeros((2, 3, 4), dtype=np.float32),
        np.full((2, 3, 4), 2.0, dtype=np.float32),
    ]

    result = average_dce(phases)

    assert result.dtype == np.float32
    assert np.array_equal(result, np.ones((2, 3, 4), dtype=np.float32))


def test_average_dce_replaces_nonfinite_mean_and_rejects_shape_mismatch() -> None:
    first = np.ones((1, 2, 3), dtype=np.float32)
    second = np.ones_like(first)
    second[0, 0, 0] = np.nan

    assert average_dce((first, second))[0, 0, 0] == 0
    with pytest.raises(ValueError, match="same shape"):
        average_dce((first, np.ones((2, 2, 3), dtype=np.float32)))


def test_slice_clahe_is_finite_bounded_and_keeps_blank_slice_blank() -> None:
    volume = np.zeros((2, 8, 8), dtype=np.float32)
    volume[1] = np.arange(64, dtype=np.float32).reshape(8, 8)
    volume[1, 0, 0] = np.inf

    result = slice_clahe(volume)

    assert result.dtype == np.float32
    assert np.count_nonzero(result[0]) == 0
    assert np.isfinite(result).all()
    assert result.min() >= 0
    assert result.max() <= 350


def test_ftv_preparation_dilates_only_in_plane_and_fills_holes() -> None:
    mask = np.zeros((3, 9, 9), dtype=np.uint8)
    mask[1, 4, 4] = 1
    mask[2, 2:7, 2:7] = 1
    mask[2, 4, 4] = 0

    result = prepare_ftv_mask(mask)

    assert result.dtype == np.uint8
    assert result[1, 4, 6] == 1
    assert result[1, 4, 7] == 0
    assert result[0].sum() == 0
    assert result[2, 4, 4] == 1


def test_pad_z_adds_one_blank_slice_at_each_end() -> None:
    array = np.ones((2, 3, 4), dtype=np.float32)

    result = pad_z(array)

    assert result.shape == (4, 3, 4)
    assert np.count_nonzero(result[[0, -1]]) == 0
    assert np.array_equal(result[1:-1], array)


def test_rigid_driver_uses_original_tumor_enhancement_formula() -> None:
    image = np.array([[[1.0, 2.0]]], dtype=np.float32)
    mask = np.array([[[0, 1]]], dtype=np.uint8)

    result = rigid_driver(image, mask, use_tumor=True)

    assert result[0, 0, 1] == pytest.approx(255.0)
    assert result[0, 0, 0] == pytest.approx(255.0 / 12.0)


def test_bspline_driver_without_tumor_is_plain_normalized_image() -> None:
    image = np.array([[[1.0, 2.0]]], dtype=np.float32)
    mask = np.zeros_like(image, dtype=np.uint8)

    result = bspline_driver(image, mask, weight=0.5, use_tumor=False)

    assert result == pytest.approx(np.array([[[127.5, 255.0]]], dtype=np.float32))


def test_empty_moving_ftv_disables_both_pair_enhancements() -> None:
    fixed_phases = (np.arange(64, dtype=np.float32).reshape(1, 8, 8),)
    moving_phases = (np.flip(fixed_phases[0], axis=2).copy(),)
    fixed_ftv = np.zeros((1, 8, 8), dtype=np.uint8)
    fixed_ftv[0, 3:5, 3:5] = 1
    moving_ftv = np.zeros_like(fixed_ftv)

    drivers = prepare_pair_drivers(fixed_phases, moving_phases, fixed_ftv, moving_ftv)

    assert not drivers.rigidity_applied
    assert np.count_nonzero(drivers.moving_rigidity) == 0
    assert drivers.fixed_rigid == pytest.approx(
        rigid_driver(drivers.fixed_anatomical, drivers.fixed_mask, use_tumor=False)
    )
    assert drivers.moving_bspline == pytest.approx(
        bspline_driver(
            drivers.moving_anatomical,
            drivers.moving_mask,
            weight=0.5,
            use_tumor=False,
        )
    )
