from __future__ import annotations

from pathlib import Path

import itk
import numpy as np
import pytest
import SimpleITK as sitk

from ispy2_registration.elastix import (
    bspline_parameter_map,
    estimate_bspline,
    itk_transform_from_sitk,
)
from ispy2_registration.geometry import array_from_itk_image, itk_image_from_array
from ispy2_registration.models import Geometry


def _identity_geometry(shape: tuple[int, int, int]) -> Geometry:
    return Geometry(
        shape_zyx=shape,
        spacing_xyz=(1.0, 1.0, 1.0),
        direction=tuple(np.eye(3).ravel()),
    )


def test_bspline_schedule_matches_matlab_parameters(tmp_path: Path) -> None:
    params = bspline_parameter_map(tmp_path / "moving_rigidity.mha")

    assert params["Registration"] == ["MultiMetricMultiResolutionRegistration"]
    assert params["NumberOfResolutions"] == ["4"]
    assert params["FinalGridSpacingInPhysicalUnits"] == ["8.0"]
    assert params["MaximumNumberOfIterations"] == ["300", "300", "300", "1200"]
    assert params["NumberOfSpatialSamples"] == ["5000", "5000", "5000", "5000"]
    assert params["Metric"] == [
        "AdvancedMattesMutualInformation",
        "TransformRigidityPenalty",
    ]
    assert params["Metric0Weight"] == ["1.0"]
    assert params["Metric1Weight"] == ["0.1"]
    assert params["LinearityConditionWeight"] == ["50.0"]
    assert params["OrthonormalityConditionWeight"] == ["1.0"]
    assert params["PropernessConditionWeight"] == ["2.0"]
    assert params["MovingRigidityImageName"] == [str(tmp_path / "moving_rigidity.mha")]
    assert params["UseMultiThreadingForMetrics"] == ["false"]


def test_bspline_map_is_mi_only_without_rigidity() -> None:
    params = bspline_parameter_map(None)

    assert params["Metric"] == ["AdvancedMattesMutualInformation"]
    assert "Metric1Weight" not in params
    assert "MovingRigidityImageName" not in params
    assert "UseMovingRigidityImage" not in params
    assert params["UseMultiThreadingForMetrics"] == ["false"]


def test_itk_image_round_trip_preserves_zyx_geometry() -> None:
    array = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    geometry = Geometry(
        shape_zyx=array.shape,
        spacing_xyz=(0.7, 0.8, 2.5),
        direction=(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0),
    )

    image = itk_image_from_array(array, geometry)

    assert tuple(image.GetSpacing()) == pytest.approx(geometry.spacing_xyz)
    assert np.asarray(image.GetDirection()) == pytest.approx(
        np.asarray(geometry.direction).reshape(3, 3)
    )
    assert np.array_equal(array_from_itk_image(image), array)


def test_sitk_translation_converts_to_equivalent_itk_affine() -> None:
    transform = sitk.TranslationTransform(3, (2.0, -1.0, 0.5))

    converted = itk_transform_from_sitk(transform)

    assert tuple(converted.TransformPoint((1.0, 2.0, 3.0))) == pytest.approx(
        (3.0, 1.0, 3.5)
    )


def test_zero_iteration_bspline_composes_rigid_once(tmp_path: Path) -> None:
    fixed_array = np.zeros((16, 16, 16), dtype=np.float32)
    moving_array = np.zeros_like(fixed_array)
    fixed_array[8, 8, 7] = 1.0
    moving_array[8, 8, 9] = 1.0
    geometry = _identity_geometry(fixed_array.shape)
    fixed = itk_image_from_array(fixed_array, geometry)
    moving = itk_image_from_array(moving_array, geometry)
    rigid = sitk.TranslationTransform(3, (2.0, 0.0, 0.0))
    params = bspline_parameter_map(
        None,
        resolutions=1,
        iterations=(0,),
        spatial_samples=100,
    )

    result = estimate_bspline(
        fixed,
        moving,
        rigid,
        output_dir=tmp_path,
        threads=1,
        parameter_map=params,
    )
    transformed = itk.transformix_filter(
        moving,
        transform_parameter_object=result.transform_parameters,
        log_to_console=False,
    )
    transformed_array = array_from_itk_image(transformed)

    assert result.transform_parameters.GetNumberOfParameterMaps() == 2
    assert np.unravel_index(np.argmax(transformed_array), transformed_array.shape) == (
        8,
        8,
        7,
    )
