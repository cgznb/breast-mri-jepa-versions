from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from ispy2_registration.geometry import sitk_image_from_array
from ispy2_registration.models import Geometry
from ispy2_registration.rigid import RigidRegistrationError, estimate_rigid, sampling_percentage


def _gaussian_image() -> sitk.Image:
    z, y, x = np.mgrid[:12, :16, :18]
    array = np.exp(-((x - 8.0) ** 2 + (y - 7.0) ** 2 + (z - 5.0) ** 2) / 12.0)
    geometry = Geometry(
        shape_zyx=array.shape,
        spacing_xyz=(0.8, 0.9, 2.0),
        direction=tuple(np.eye(3).ravel()),
    )
    return sitk_image_from_array(array.astype(np.float32), geometry)


def _asymmetric_image() -> sitk.Image:
    shape = (32, 40, 44)
    z, y, x = np.mgrid[: shape[0], : shape[1], : shape[2]]
    array = (
        np.exp(-((x - 15) ** 2 / 20 + (y - 19) ** 2 / 30 + (z - 14) ** 2 / 16))
        + 0.7
        * np.exp(-((x - 27) ** 2 / 8 + (y - 11) ** 2 / 13 + (z - 21) ** 2 / 10))
        + 0.35
        * np.exp(-((x - 8) ** 2 / 5 + (y - 30) ** 2 / 9 + (z - 8) ** 2 / 7))
    ).astype(np.float32)
    geometry = Geometry(
        shape_zyx=shape,
        spacing_xyz=(1.0, 1.0, 2.0),
        direction=tuple(np.eye(3).ravel()),
    )
    return sitk_image_from_array(array, geometry)


def test_sampling_percentage_targets_5000_without_exceeding_full_image() -> None:
    assert sampling_percentage(1000, samples=5000) == 1.0
    assert sampling_percentage(8000, samples=5000) == pytest.approx(0.625)


def test_rigid_registration_is_finite_and_reproducible() -> None:
    fixed = _gaussian_image()
    moving = sitk.Image(fixed)

    first = estimate_rigid(fixed, moving, samples=500, iterations=8, seed=20260719)
    second = estimate_rigid(fixed, moving, samples=500, iterations=8, seed=20260719)

    assert first.sampling_percentage == pytest.approx(500 / fixed.GetNumberOfPixels())
    assert np.isfinite(first.metric_value)
    assert np.isfinite(first.transform.GetParameters()).all()
    assert first.transform.GetParameters() == pytest.approx(second.transform.GetParameters())


def test_rigid_registration_rejects_blank_driver() -> None:
    blank = sitk.Image([8, 8, 8], sitk.sitkFloat32)

    with pytest.raises(RigidRegistrationError, match="blank"):
        estimate_rigid(blank, blank, samples=50, iterations=2)


@pytest.mark.parametrize(
    ("angles", "translation"),
    [
        ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)),
        (
            (np.deg2rad(2.0), np.deg2rad(-3.0), np.deg2rad(4.0)),
            (4.0, -2.0, 2.0),
        ),
    ],
)
def test_rigid_registration_recovers_known_transform(
    angles: tuple[float, float, float],
    translation: tuple[float, float, float],
) -> None:
    fixed = _asymmetric_image()
    truth = sitk.Euler3DTransform()
    truth.SetCenter((0.0, 0.0, 0.0))
    truth.SetRotation(*angles)
    truth.SetTranslation(translation)
    moving = sitk.Resample(
        fixed,
        fixed,
        truth.GetInverse(),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )

    result = estimate_rigid(
        fixed,
        moving,
        samples=5000,
        iterations=500,
        seed=20260719,
    )

    parameters = np.asarray(result.transform.GetParameters())
    assert parameters[:3] == pytest.approx(angles, abs=0.01)
    assert parameters[3:] == pytest.approx(translation, abs=0.25)
