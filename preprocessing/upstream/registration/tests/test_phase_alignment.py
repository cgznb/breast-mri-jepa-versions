from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from ispy2_registration.models import Geometry
from ispy2_registration.phase_alignment import (
    IntraVisitRegistrationError,
    align_dce_phases,
    registration_region,
)
from ispy2_registration.rigid import RigidRegistrationError, RigidResult


GEOMETRY = Geometry(
    shape_zyx=(20, 40, 50),
    spacing_xyz=(0.8, 0.9, 2.0),
    direction=tuple(np.eye(3).ravel()),
)


def test_registration_region_matches_matlab_ftv_bbox_margins() -> None:
    ftv = np.zeros(GEOMETRY.shape_zyx, dtype=np.uint8)
    ftv[8:10, 15:18, 20:24] = 1

    region = registration_region(ftv)

    assert region.slices_zyx == (slice(3, 15), slice(5, 28), slice(10, 34))
    assert not region.used_full_fov


def test_registration_region_uses_full_fov_for_empty_ftv() -> None:
    ftv = np.zeros(GEOMETRY.shape_zyx, dtype=np.uint8)

    region = registration_region(ftv)

    assert region.slices_zyx == (slice(0, 20), slice(0, 40), slice(0, 50))
    assert region.used_full_fov


def _rigid_result() -> RigidResult:
    return RigidResult(
        transform=sitk.Euler3DTransform(),
        metric_value=-0.5,
        iterations=3,
        stop_condition="synthetic",
        sampling_percentage=0.25,
    )


def test_align_dce_phases_uses_actual_phase_ids_and_preserves_phase_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (8, 12, 14)
    geometry = Geometry(
        shape_zyx=shape,
        spacing_xyz=(1.0, 1.0, 2.0),
        direction=tuple(np.eye(3).ravel()),
    )
    z, y, x = np.mgrid[: shape[0], : shape[1], : shape[2]]
    reference = np.exp(-((x - 7) ** 2 + (y - 6) ** 2 + (z - 4) ** 2) / 8).astype(
        np.float32
    )
    phases = tuple(reference * float(index + 1) for index in range(4))
    phase_ids = (0, 2, 5, 7)
    calls: list[tuple[tuple[int, ...], int]] = []

    def fake_estimate(fixed, moving, **kwargs):
        calls.append((tuple(reversed(fixed.GetSize())), kwargs["seed"]))
        return _rigid_result()

    monkeypatch.setattr(
        "ispy2_registration.phase_alignment.estimate_rigid", fake_estimate
    )

    result = align_dce_phases(
        phases,
        phase_ids,
        np.zeros(shape, dtype=np.uint8),
        geometry,
        samples=50,
        iterations=3,
        seed=20260719,
    )

    assert result.phase_ids == phase_ids
    assert len(result.aligned_phases) == 4
    assert np.array_equal(result.aligned_phases[0], phases[0])
    assert [item.phase_id for item in result.registrations] == [2, 5, 7]
    assert calls == [(shape, 20260721), (shape, 20260724), (shape, 20260726)]
    assert result.region.used_full_fov


def test_align_dce_phases_reports_the_failed_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (5, 6, 7)
    geometry = Geometry(
        shape_zyx=shape,
        spacing_xyz=(1.0, 1.0, 1.0),
        direction=tuple(np.eye(3).ravel()),
    )
    phases = (np.ones(shape, np.float32), np.ones(shape, np.float32) * 2)

    def fail(*args, **kwargs):
        raise RigidRegistrationError("synthetic failure")

    monkeypatch.setattr("ispy2_registration.phase_alignment.estimate_rigid", fail)

    with pytest.raises(IntraVisitRegistrationError, match="phase 4"):
        align_dce_phases(
            phases,
            (0, 4),
            np.zeros(shape, dtype=np.uint8),
            geometry,
            samples=20,
            iterations=2,
        )


def test_align_dce_phases_retries_full_fov_after_cropped_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (12, 30, 40)
    geometry = Geometry(
        shape_zyx=shape,
        spacing_xyz=(1.0, 1.0, 2.0),
        direction=tuple(np.eye(3).ravel()),
    )
    z, y, x = np.mgrid[: shape[0], : shape[1], : shape[2]]
    reference = np.exp(
        -((x - 20) ** 2 + (y - 15) ** 2 + (z - 6) ** 2) / 20
    ).astype(np.float32)
    phases = (reference, reference * 1.2, reference * 1.4)
    ftv = np.zeros(shape, dtype=np.uint8)
    ftv[5:6, 14:16, 20:22] = 1
    calls: list[tuple[int, ...]] = []

    def fail_cropped_registration(fixed, moving, **kwargs):
        fixed_shape = tuple(reversed(fixed.GetSize()))
        calls.append(fixed_shape)
        if fixed_shape != shape:
            raise RigidRegistrationError("cropped optimizer diverged")
        return _rigid_result()

    monkeypatch.setattr(
        "ispy2_registration.phase_alignment.estimate_rigid",
        fail_cropped_registration,
    )

    result = align_dce_phases(
        phases,
        (0, 1, 2),
        ftv,
        geometry,
        samples=50,
        iterations=3,
    )

    assert result.region.used_full_fov
    assert [item.phase_id for item in result.registrations] == [1, 2]
    assert calls == [(11, 22, 22), shape, shape]


@pytest.mark.parametrize("phase_ids", [(1, 2), (0, 0), (0,)])
def test_align_dce_phases_rejects_invalid_phase_ids(
    phase_ids: tuple[int, ...],
) -> None:
    shape = (3, 4, 5)
    phases = (np.ones(shape, np.float32), np.ones(shape, np.float32))
    geometry = Geometry(
        shape_zyx=shape,
        spacing_xyz=(1.0, 1.0, 1.0),
        direction=tuple(np.eye(3).ravel()),
    )

    with pytest.raises(ValueError, match="phase IDs"):
        align_dce_phases(
            phases,
            phase_ids,
            np.zeros(shape, dtype=np.uint8),
            geometry,
            samples=20,
            iterations=2,
        )
