from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk


class RigidRegistrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class RigidResult:
    transform: sitk.Euler3DTransform
    metric_value: float
    iterations: int
    stop_condition: str
    sampling_percentage: float


def sampling_percentage(number_of_pixels: int, *, samples: int = 5000) -> float:
    if number_of_pixels < 1 or samples < 1:
        raise ValueError("pixel and sample counts must be positive")
    return min(1.0, float(samples) / float(number_of_pixels))


def _validate_driver(image: sitk.Image, name: str) -> None:
    if image.GetDimension() != 3:
        raise RigidRegistrationError(f"{name} driver must be 3D")
    statistics = sitk.StatisticsImageFilter()
    statistics.Execute(image)
    if not np.isfinite((statistics.GetMinimum(), statistics.GetMaximum())).all():
        raise RigidRegistrationError(f"{name} driver contains nonfinite values")
    if statistics.GetMaximum() <= statistics.GetMinimum():
        raise RigidRegistrationError(f"{name} driver is blank or constant")


def estimate_rigid(
    fixed: sitk.Image,
    moving: sitk.Image,
    *,
    samples: int = 5000,
    iterations: int = 500,
    seed: int = 20260719,
    learning_rate: float = 2.0,
    min_step: float = 1e-4,
    relaxation_factor: float = 0.5,
    shrink_factors: tuple[int, ...] = (4, 2, 1),
    smoothing_sigmas: tuple[float, ...] = (2.0, 1.0, 0.0),
) -> RigidResult:
    _validate_driver(fixed, "fixed")
    _validate_driver(moving, "moving")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if learning_rate <= 0 or min_step <= 0:
        raise ValueError("learning rate and minimum step must be positive")
    if not 0.0 < relaxation_factor < 1.0:
        raise ValueError("relaxation factor must be between zero and one")
    if (
        not shrink_factors
        or len(shrink_factors) != len(smoothing_sigmas)
        or any(value < 1 for value in shrink_factors)
        or any(value < 0 for value in smoothing_sigmas)
    ):
        raise ValueError(
            "shrink factors and smoothing sigmas must be valid equal-length schedules"
        )
    percentage = sampling_percentage(fixed.GetNumberOfPixels(), samples=samples)
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(percentage, seed)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetShrinkFactorsPerLevel(list(shrink_factors))
    registration.SetSmoothingSigmasPerLevel(list(smoothing_sigmas))
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=min_step,
        numberOfIterations=iterations,
        relaxationFactor=relaxation_factor,
        gradientMagnitudeTolerance=1e-8,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    initial = sitk.Euler3DTransform()
    initial.SetCenter((0.0, 0.0, 0.0))
    registration.SetInitialTransform(initial, inPlace=True)
    try:
        registration.Execute(
            sitk.Cast(fixed, sitk.sitkFloat32),
            sitk.Cast(moving, sitk.sitkFloat32),
        )
    except RuntimeError as exc:
        raise RigidRegistrationError(f"rigid optimizer failed: {exc}") from exc
    parameters = np.asarray(initial.GetParameters(), dtype=np.float64)
    metric_value = float(registration.GetMetricValue())
    if not np.isfinite(parameters).all() or not np.isfinite(metric_value):
        raise RigidRegistrationError("rigid optimizer returned nonfinite parameters")
    registered = sitk.Resample(
        moving,
        fixed,
        initial,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    registered_array = sitk.GetArrayViewFromImage(registered)
    if not np.isfinite(registered_array).all() or not np.any(registered_array):
        raise RigidRegistrationError("rigid optimizer returned an empty or nonfinite image")
    return RigidResult(
        transform=sitk.Euler3DTransform(initial),
        metric_value=metric_value,
        iterations=int(registration.GetOptimizerIteration()),
        stop_condition=registration.GetOptimizerStopConditionDescription(),
        sampling_percentage=percentage,
    )
