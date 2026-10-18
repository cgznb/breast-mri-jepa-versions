from __future__ import annotations

import itk
import numpy as np
import pytest
import SimpleITK as sitk

from ispy2_registration.models import Geometry
from ispy2_registration.transforms import (
    apply_transform,
    compose_source_grid_with_longitudinal,
    compose_phase_with_longitudinal,
    configure_transform_parameters,
    dicom_grid_transform,
    displacement_and_jacobian,
    resample_rigid,
    resample_dce_phases_once,
    resample_multimodal_once,
)


def _geometry(shape: tuple[int, int, int] = (9, 10, 11)) -> Geometry:
    return Geometry(
        shape_zyx=shape,
        spacing_xyz=(1.0, 1.0, 1.0),
        direction=tuple(np.eye(3).ravel()),
    )


def _translation_parameters(geometry: Geometry, offset=(2.0, 0.0, 0.0)):
    size = [str(value) for value in geometry.size_xyz]
    half = [-(value - 1) / 2 for value in geometry.size_xyz]
    parameter_map = {
        "Transform": ["TranslationTransform"],
        "NumberOfParameters": ["3"],
        "TransformParameters": [str(value) for value in offset],
        "InitialTransformParametersFileName": ["NoInitialTransform"],
        "HowToCombineTransforms": ["Compose"],
        "FixedImageDimension": ["3"],
        "MovingImageDimension": ["3"],
        "FixedInternalImagePixelType": ["float"],
        "MovingInternalImagePixelType": ["float"],
        "Size": size,
        "Index": ["0", "0", "0"],
        "Spacing": ["1", "1", "1"],
        "Origin": [str(value) for value in half],
        "Direction": [str(value) for value in np.eye(3).ravel()],
        "UseDirectionCosines": ["true"],
        "ResampleInterpolator": ["FinalBSplineInterpolator"],
        "FinalBSplineInterpolationOrder": ["1"],
        "Resampler": ["DefaultResampler"],
        "DefaultPixelValue": ["0"],
        "ResultImagePixelType": ["float"],
        "ResultImageFormat": ["nii"],
        "CompressResultImage": ["true"],
    }
    parameter_object = itk.ParameterObject.New()
    parameter_object.SetParameterMap(parameter_map)
    return parameter_object


def test_configure_transform_uses_target_grid_without_mutating_source() -> None:
    source_geometry = _geometry()
    target_geometry = Geometry(
        shape_zyx=(7, 8, 9),
        spacing_xyz=(0.7, 0.8, 2.0),
        direction=(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0),
    )
    original = _translation_parameters(source_geometry)

    configured = configure_transform_parameters(original, target_geometry, interpolation="nearest")

    original_map = original.GetParameterMap(0)
    result_map = configured.GetParameterMap(0)
    assert list(original_map["Size"]) == ["11", "10", "9"]
    assert list(result_map["Size"]) == ["9", "8", "7"]
    assert list(result_map["Spacing"]) == ["0.7", "0.8", "2"]
    assert list(result_map["FinalBSplineInterpolationOrder"]) == ["0"]


def test_transformix_translation_is_applied_once_to_intensity() -> None:
    geometry = _geometry()
    moving = np.zeros(geometry.shape_zyx, dtype=np.float32)
    moving[4, 5, 7] = 1.0

    result = apply_transform(
        moving,
        geometry,
        geometry,
        _translation_parameters(geometry),
        interpolation="linear",
    )

    assert np.unravel_index(np.argmax(result), result.shape) == (4, 5, 5)
    assert result.dtype == np.float32


def test_nearest_transform_preserves_raw_mask_labels() -> None:
    geometry = _geometry()
    moving = np.zeros(geometry.shape_zyx, dtype=np.uint8)
    moving[4, 5, 7] = 49
    moving[4, 5, 8] = 33

    result = apply_transform(
        moving,
        geometry,
        geometry,
        _translation_parameters(geometry),
        interpolation="nearest",
    )

    assert result.dtype == np.uint8
    assert set(np.unique(result)) <= {0, 33, 49}
    assert result[4, 5, 5] == 49


def test_transformix_nearest_uses_requested_padding_value() -> None:
    geometry = _geometry()
    moving = np.full(geometry.shape_zyx, 49, dtype=np.uint8)

    result = apply_transform(
        moving,
        geometry,
        geometry,
        _translation_parameters(geometry),
        interpolation="nearest",
        default_value=128,
    )

    assert np.all(result[..., :-2] == 49)
    assert np.all(result[..., -2:] == 128)


def test_rigid_resampling_uses_fixed_to_moving_direction() -> None:
    geometry = _geometry()
    moving = np.zeros(geometry.shape_zyx, dtype=np.uint8)
    moving[4, 5, 7] = 1
    transform = sitk.TranslationTransform(3, (2.0, 0.0, 0.0))

    result = resample_rigid(
        moving,
        geometry,
        geometry,
        transform,
        interpolation="nearest",
    )

    assert result[4, 5, 5] == 1


def test_rigid_nearest_uses_requested_padding_value() -> None:
    geometry = _geometry()
    moving = np.full(geometry.shape_zyx, 49, dtype=np.uint8)
    transform = sitk.TranslationTransform(3, (2.0, 0.0, 0.0))

    result = resample_rigid(
        moving,
        geometry,
        geometry,
        transform,
        interpolation="nearest",
        default_value=128,
    )

    assert np.all(result[..., :-2] == 49)
    assert np.all(result[..., -2:] == 128)


def test_translation_field_has_positive_unit_jacobian(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geometry = _geometry((7, 8, 9))
    monkeypatch.chdir(tmp_path)

    field, jacobian = displacement_and_jacobian(
        geometry,
        geometry,
        _translation_parameters(geometry),
    )

    assert field.shape == (*geometry.shape_zyx, 3)
    assert field[..., 0] == pytest.approx(2.0)
    assert field[..., 1:] == pytest.approx(0.0)
    assert jacobian == pytest.approx(np.ones(geometry.shape_zyx, dtype=np.float32))
    assert not (tmp_path / "deformationField.nii").exists()


def test_phase_transform_is_applied_after_longitudinal_displacement() -> None:
    geometry = _geometry((5, 6, 7))
    field = np.zeros((*geometry.shape_zyx, 3), dtype=np.float32)
    field[..., 0] = 1.0
    phase = sitk.Euler3DTransform()
    phase.SetCenter((0.0, 0.0, 0.0))
    phase.SetRotation(0.0, 0.0, np.pi / 2.0)

    combined = compose_phase_with_longitudinal(phase, field, geometry)

    point = (2.0, 0.0, 0.0)
    expected = phase.TransformPoint((3.0, 0.0, 0.0))
    assert combined.TransformPoint(point) == pytest.approx(expected)


def test_dce_phases_are_resampled_once_with_phase_specific_composition() -> None:
    geometry = _geometry()
    phase_zero = np.zeros(geometry.shape_zyx, dtype=np.float32)
    phase_five = np.zeros_like(phase_zero)
    phase_zero[4, 5, 7] = 1.0
    phase_five[4, 5, 7] = 2.0
    longitudinal = np.zeros((*geometry.shape_zyx, 3), dtype=np.float32)
    longitudinal[..., 0] = 1.0
    phase_transform = sitk.TranslationTransform(3, (2.0, 0.0, 0.0))

    outputs = resample_dce_phases_once(
        (phase_zero, phase_five),
        (0, 5),
        {5: phase_transform},
        geometry,
        geometry,
        longitudinal,
    )

    assert np.unravel_index(np.argmax(outputs[0]), outputs[0].shape) == (4, 5, 6)
    assert np.unravel_index(np.argmax(outputs[1]), outputs[1].shape) == (4, 5, 4)
    assert outputs[0].dtype == np.float32
    assert outputs[1].dtype == np.float32


def test_dicom_grid_transform_maps_centered_dce_to_native_dwi_coordinates() -> None:
    dce = Geometry((5, 7, 9), (1.0, 2.0, 3.0), tuple(np.eye(3).ravel()))
    dwi = Geometry((3, 5, 7), (2.0, 2.0, 4.0), tuple(np.eye(3).ravel()))
    transform = dicom_grid_transform(dce, (10.0, 20.0, 30.0), dwi, (4.0, 5.0, 6.0))

    dce_centered_origin = np.asarray((-(9 - 1) / 2, -(7 - 1), -(5 - 1) * 3 / 2))
    dwi_centered_origin = np.asarray((-(7 - 1), -(5 - 1), -(3 - 1) * 4 / 2))
    expected_offset = (
        np.asarray((10.0, 20.0, 30.0))
        - dce_centered_origin
        - np.asarray((4.0, 5.0, 6.0))
        + dwi_centered_origin
    )
    assert transform.GetOffset() == pytest.approx(expected_offset)


def test_dwi_grid_mapping_is_composed_after_longitudinal_mapping() -> None:
    geometry = _geometry((5, 6, 7))
    field = np.zeros((*geometry.shape_zyx, 3), dtype=np.float32)
    field[..., 0] = 2.0
    grid = sitk.TranslationTransform(3, (3.0, 0.0, 0.0))

    combined = compose_source_grid_with_longitudinal(grid, field, geometry)

    assert combined.TransformPoint((0.0, 0.0, 0.0)) == pytest.approx((5.0, 0.0, 0.0))


def test_multimodal_one_pass_uses_linear_for_image_and_nearest_for_seg() -> None:
    geometry = _geometry((5, 6, 7))
    image = np.zeros(geometry.shape_zyx, dtype=np.float32)
    image[2, 3, 4] = 1.0
    segmentation = image.astype(np.uint8)
    field = np.zeros((*geometry.shape_zyx, 3), dtype=np.float32)
    transform = sitk.TranslationTransform(3, (1.0, 0.0, 0.0))

    registered_image = resample_multimodal_once(
        image, geometry, geometry, transform, field, interpolation="linear"
    )
    registered_seg = resample_multimodal_once(
        segmentation, geometry, geometry, transform, field, interpolation="nearest"
    )

    assert np.unravel_index(np.argmax(registered_image), registered_image.shape) == (2, 3, 3)
    assert registered_seg.dtype == np.uint8
    assert set(np.unique(registered_seg)) <= {0, 1}
    assert registered_seg[2, 3, 3] == 1
