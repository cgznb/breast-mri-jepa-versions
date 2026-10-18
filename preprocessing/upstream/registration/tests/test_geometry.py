from __future__ import annotations

import numpy as np
import pytest

from ispy2_registration.geometry import (
    array_from_sitk_image,
    centered_origin,
    direction_from_iop,
    sitk_image_from_array,
)
from ispy2_registration.models import Geometry


def test_direction_from_axial_iop_uses_itk_xyz_axes() -> None:
    direction = direction_from_iop((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0))

    assert np.asarray(direction).reshape(3, 3) == pytest.approx(
        np.diag((-1.0, -1.0, 1.0))
    )


def test_direction_rejects_non_orthogonal_iop() -> None:
    with pytest.raises(ValueError, match="orthogonal"):
        direction_from_iop((1.0, 0.0, 0.0, 1.0, 0.0, 0.0))


def test_centered_origin_places_continuous_grid_center_at_zero() -> None:
    geometry = Geometry(
        shape_zyx=(5, 7, 9),
        spacing_xyz=(2.0, 1.0, 3.0),
        direction=tuple(np.eye(3).ravel()),
    )
    image = sitk_image_from_array(np.zeros(geometry.shape_zyx), geometry)

    point = image.TransformContinuousIndexToPhysicalPoint((4.0, 3.0, 2.0))

    assert point == pytest.approx((0.0, 0.0, 0.0))
    assert centered_origin(geometry) == pytest.approx((-8.0, -3.0, -6.0))


def test_zyx_array_round_trip_preserves_values_and_geometry() -> None:
    array = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    geometry = Geometry(
        shape_zyx=array.shape,
        spacing_xyz=(0.7, 0.8, 2.5),
        direction=direction_from_iop((0.0, 1.0, 0.0, -1.0, 0.0, 0.0)),
    )

    image = sitk_image_from_array(array, geometry)

    assert image.GetSize() == (5, 4, 3)
    assert image.GetSpacing() == pytest.approx(geometry.spacing_xyz)
    assert image.GetDirection() == pytest.approx(geometry.direction)
    assert np.array_equal(array_from_sitk_image(image), array)


def test_geometry_rejects_nonpositive_spacing() -> None:
    with pytest.raises(ValueError, match="spacing"):
        Geometry(
            shape_zyx=(3, 4, 5),
            spacing_xyz=(1.0, 0.0, 2.0),
            direction=tuple(np.eye(3).ravel()),
        )

