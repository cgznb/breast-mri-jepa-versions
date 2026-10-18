from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import itk
import numpy as np
import SimpleITK as sitk


class ElastixRegistrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ElastixResult:
    registered_image: object
    transform_parameters: object
    rigidity_applied: bool
    output_dir: Path


def bspline_parameter_map(
    moving_rigidity_path: Path | None,
    *,
    resolutions: int = 4,
    iterations: Sequence[int] = (300, 300, 300, 1200),
    spatial_samples: int = 5000,
    seed: int = 20260719,
) -> dict[str, list[str]]:
    if resolutions < 1 or len(iterations) != resolutions:
        raise ValueError("iterations must provide one value per resolution")
    if spatial_samples < 1:
        raise ValueError("spatial_samples must be positive")
    params: dict[str, list[str]] = {
        "FixedInternalImagePixelType": ["float"],
        "MovingInternalImagePixelType": ["float"],
        "Registration": ["MultiMetricMultiResolutionRegistration"],
        "FixedImagePyramid": ["FixedRecursiveImagePyramid"],
        "MovingImagePyramid": ["MovingRecursiveImagePyramid"],
        "Interpolator": ["BSplineInterpolator"],
        "ResampleInterpolator": ["FinalBSplineInterpolator"],
        "Resampler": ["DefaultResampler"],
        "Metric": ["AdvancedMattesMutualInformation"],
        "Optimizer": ["StandardGradientDescent"],
        "Transform": ["BSplineTransform"],
        "FinalGridSpacingInPhysicalUnits": ["8.0"],
        "UpsampleGridOption": ["true"],
        "NumberOfResolutions": [str(resolutions)],
        "ImageSampler": ["Random"],
        "NewSamplesEveryIteration": ["true"],
        "NumberOfSpatialSamples": [str(spatial_samples)] * resolutions,
        "RandomSeed": [str(seed)],
        "ErodeMask": ["true"],
        "BSplineInterpolationOrder": ["1"],
        "FinalBSplineInterpolationOrder": ["3"],
        "DefaultPixelValue": ["0"],
        "WriteTransformParametersEachIteration": ["false"] * resolutions,
        "WriteTransformParametersEachResolution": ["true"] * resolutions,
        "ShowExactMetricValue": ["false"] * resolutions,
        "WriteResultImageAfterEachResolution": ["false"],
        "WriteResultImage": ["true"],
        "ResultImagePixelType": ["float"],
        "ResultImageFormat": ["nii"],
        "MaximumNumberOfIterations": [str(value) for value in iterations],
        "SP_a": ["150000.0", "120000.0", "70000.0", "20000.0"][:resolutions],
        "SP_alpha": ["0.602"] * resolutions,
        "SP_A": ["100.0"] * resolutions,
        "NumberOfHistogramBins": ["32"] * resolutions,
        "Metric0Weight": ["1.0"],
        "AutomaticTransformInitialization": ["false"],
        "UseDirectionCosines": ["true"],
        "UseMultiThreadingForMetrics": ["false"],
    }
    if resolutions > 4:
        params["SP_a"] = ["20000.0"] * resolutions
    if moving_rigidity_path is not None:
        params.update(
            {
                "Metric": [
                    "AdvancedMattesMutualInformation",
                    "TransformRigidityPenalty",
                ],
                "Metric1Weight": ["0.1"],
                "LinearityConditionWeight": ["50.0"],
                "OrthonormalityConditionWeight": ["1.0"],
                "PropernessConditionWeight": ["2.0"],
                "UseLinearityCondition": ["true"],
                "UseOrthonormalityCondition": ["true"],
                "UsePropernessCondition": ["true"],
                "CalculateLinearityCondition": ["true"],
                "CalculateOrthonormalityCondition": ["true"],
                "CalculatePropernessCondition": ["true"],
                "DilateRigidityImages": ["false"] * (resolutions - 1) + ["true"],
                "DilationRadiusMultiplier": ["2.0"],
                "UseFixedRigidityImage": ["false"],
                "UseMovingRigidityImage": ["true"],
                "MovingRigidityImageName": [str(moving_rigidity_path)],
            }
        )
    return params


def _single_sitk_transform(transform: sitk.Transform) -> sitk.Transform:
    if transform.GetName() != "CompositeTransform":
        return transform
    composite = sitk.CompositeTransform(transform)
    if composite.GetNumberOfTransforms() != 1:
        raise ValueError("only a single rigid transform can initialize Elastix")
    return composite.GetNthTransform(0)


def itk_transform_from_sitk(transform: sitk.Transform):
    source = _single_sitk_transform(transform)
    dimension = source.GetDimension()
    if dimension != 3:
        raise ValueError("only 3D rigid transforms are supported")
    if hasattr(source, "GetMatrix"):
        matrix = np.asarray(source.GetMatrix(), dtype=np.float64).reshape(3, 3)
    else:
        matrix = np.eye(3, dtype=np.float64)
    center = tuple(float(value) for value in getattr(source, "GetCenter", lambda: (0, 0, 0))())
    if hasattr(source, "GetTranslation"):
        translation = tuple(float(value) for value in source.GetTranslation())
    elif hasattr(source, "GetOffset"):
        translation = tuple(float(value) for value in source.GetOffset())
    else:
        raise ValueError(f"unsupported SimpleITK transform: {source.GetName()}")
    converted = itk.AffineTransform[itk.D, 3].New()
    converted.SetMatrix(itk.matrix_from_array(matrix))
    converted.SetCenter(center)
    converted.SetTranslation(translation)
    return converted


def estimate_bspline(
    fixed_image,
    moving_image,
    rigid_transform: sitk.Transform,
    *,
    output_dir: Path,
    threads: int,
    moving_rigidity=None,
    parameter_map: dict[str, list[str]] | None = None,
) -> ElastixResult:
    if threads < 1:
        raise ValueError("threads must be positive")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rigidity_path: Path | None = None
    if moving_rigidity is not None:
        rigidity_array = itk.array_from_image(moving_rigidity)
        if np.any(rigidity_array):
            rigidity_path = output_path / "moving_rigidity.mha"
            itk.imwrite(moving_rigidity, str(rigidity_path), compression=True)
    params = (
        bspline_parameter_map(rigidity_path)
        if parameter_map is None
        else {key: list(value) for key, value in parameter_map.items()}
    )
    if rigidity_path is not None:
        params["MovingRigidityImageName"] = [str(rigidity_path)]
    initial_transform = itk_transform_from_sitk(rigid_transform)
    try:
        result_image, transform_parameters = itk.elastix_registration_method(
            fixed_image=fixed_image,
            moving_image=moving_image,
            parameter_object=params,
            initial_transform=initial_transform,
            output_directory=str(output_path),
            log_file_name="elastix.log",
            log_to_console=False,
            log_to_file=True,
            number_of_threads=threads,
        )
    except RuntimeError as exc:
        raise ElastixRegistrationError(f"B-spline registration failed: {exc}") from exc
    registered = itk.array_from_image(result_image)
    if not np.isfinite(registered).all() or not np.any(registered):
        raise ElastixRegistrationError("B-spline registration returned an empty or nonfinite image")
    if transform_parameters.GetNumberOfParameterMaps() < 2:
        raise ElastixRegistrationError("Elastix did not preserve the rigid initial transform")
    return ElastixResult(
        registered_image=result_image,
        transform_parameters=transform_parameters,
        rigidity_applied=rigidity_path is not None,
        output_dir=output_path,
    )
