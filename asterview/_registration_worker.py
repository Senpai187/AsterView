"""CPU-only SimpleITK registration worker; no Qt, VTK or Torch imports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import SimpleITK as sitk


def progress(text):
    print("ASTER_PROGRESS:" + text, flush=True)


def working_image(job, name, config):
    """Preserve physical source geometry while bounding the optimization grid."""
    geometry = config[name]
    raw = np.load(job / (name + ".npy"), allow_pickle=False, mmap_mode="r")
    if raw.dtype != np.float32 or list(raw.shape) != geometry["shape"]:
        raise ValueError("Invalid registration source array.")
    image = sitk.GetImageFromArray(raw)
    image.SetSpacing(geometry["spacing"])
    image.SetOrigin(geometry["origin"])
    image.SetDirection(np.asarray(geometry["direction"]).reshape(-1).tolist())
    size = np.asarray(image.GetSize())
    extent = (size - 1) * image.GetSpacing()
    pitch = max(1.0, float(extent.max() / 224), float((np.prod(extent) / 8_000_000) ** (1 / 3)))
    target_size = np.minimum(size, np.maximum(8, np.floor(extent / pitch).astype(int) + 1))
    target_spacing = extent / (target_size - 1)
    work = sitk.Resample(image, target_size.tolist(), sitk.Transform(), sitk.sitkLinear, image.GetOrigin(), target_spacing.tolist(), image.GetDirection(),
                         config[name + "_data_min"], sitk.sitkFloat32)
    del image, raw
    mask_file = job / (name + "-valid.npy")
    if mask_file.is_file():
        raw_mask = np.load(mask_file, allow_pickle=False, mmap_mode="r")
        if list(raw_mask.shape) != geometry["shape"] or raw_mask.dtype != np.bool_:
            raise ValueError("Invalid registration padding mask.")
        mask = sitk.GetImageFromArray(np.asarray(raw_mask, dtype=np.uint8))
        mask.SetSpacing(geometry["spacing"])
        mask.SetOrigin(geometry["origin"])
        mask.SetDirection(np.asarray(geometry["direction"]).reshape(-1).tolist())
        support = sitk.Resample(sitk.Cast(mask, sitk.sitkFloat32), work, sitk.Transform(), sitk.sitkLinear, 0, sitk.sitkFloat32)
        valid = support > .999
        del raw_mask, mask, support
    else:
        valid = sitk.Image(work.GetSize(), sitk.sitkUInt8) + 1
        valid.CopyInformation(work)
    values = sitk.GetArrayViewFromImage(work)[sitk.GetArrayViewFromImage(valid) > 0]
    if values.size < 256 or not np.isfinite(values).all():
        raise ValueError("Too few valid registration voxels.")
    low, high = np.percentile(values, [.5, 99.5])
    if high - low < 1e-5:
        raise ValueError("A source has insufficient intensity variation for registration.")
    if name == "fixed" and config["bone_focus"]:
        threshold = 300. if config[name + "_is_hu"] else float(np.percentile(values, 75))
        bone = sitk.And(work > threshold, valid)
        if int(sitk.GetArrayViewFromImage(bone).sum()) < 512:
            raise ValueError("The selected high-intensity focus has too few valid voxels.")
        valid = bone
    # Display intensities and source arrays are untouched. Only the metric's
    # bounded working images use independent robust clipping/normalization.
    normalized = (sitk.Clamp(work, lowerBound=float(low), upperBound=float(high)) - float(low)) / float(high - low)
    normalized = sitk.Cast(normalized, sitk.sitkFloat32)
    return normalized, sitk.Cast(valid, sitk.sitkUInt8), {"shape_zyx": target_size[::-1].tolist(), "spacing_xyz": target_spacing.tolist()}


def center(geometry):
    index = (np.asarray(geometry["shape"])[::-1] - 1) / 2
    return np.asarray(geometry["origin"]) + np.asarray(geometry["direction"]) @ (index * geometry["spacing"])


def matrix_from_sitk(transform):
    """Works for Euler/composite transforms without assuming parameter order."""
    origin = np.asarray(transform.TransformPoint((0., 0., 0.)))
    matrix = np.eye(4)
    matrix[:3, 3] = origin
    for axis in range(3):
        point = np.eye(3)[axis]
        matrix[:3, axis] = np.asarray(transform.TransformPoint(point.tolist())) - origin
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job = Path(args.job)
    config = json.loads((job / "config.json").read_text(encoding="utf-8"))
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    started = time.monotonic()
    progress("Building bounded physical-space registration images…")
    fixed, fixed_mask, fixed_grid = working_image(job, "fixed", config)
    moving, moving_mask, moving_grid = working_image(job, "moving", config)
    fixed_center, moving_center = center(config["fixed"]), center(config["moving"])
    transform = sitk.Euler3DTransform()
    transform.SetComputeZYX(True)
    transform.SetCenter(fixed_center.tolist())
    if config["initial_moving_to_fixed"] is not None:
        inverse = np.linalg.inv(np.asarray(config["initial_moving_to_fixed"], dtype=float))
        transform.SetMatrix(inverse[:3, :3].reshape(-1).tolist(), tolerance=1e-6)
        transform.SetTranslation((inverse[:3, 3] - fixed_center + inverse[:3, :3] @ fixed_center).tolist())
    elif config["initialization"] == "centers":
        transform.SetTranslation((moving_center - fixed_center).tolist())
    initial_matrix = matrix_from_sitk(transform)
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=48)
    count = np.prod(fixed.GetSize())
    if count <= 150_000:
        registration.SetMetricSamplingStrategy(registration.NONE)
    else:
        registration.SetMetricSamplingStrategy(registration.REGULAR)
        registration.SetMetricSamplingPercentage(min(.5, 150_000 / count), seed=config["seed"])
    registration.SetMetricFixedMask(fixed_mask)
    registration.SetMetricMovingMask(moving_mask)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=2., minStep=.001, numberOfIterations=200,
                                                          relaxationFactor=.5, gradientMagnitudeTolerance=1e-6)
    registration.SetOptimizerScalesFromPhysicalShift()
    minimum = min(*fixed.GetSize(), *moving.GetSize())
    factors = [x for x in (4, 2, 1) if minimum / x >= 4]
    registration.SetShrinkFactorsPerLevel(factors)
    registration.SetSmoothingSigmasPerLevel([float(x / 2) if x > 1 else 0. for x in factors])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(transform, inPlace=True)
    def level():
        progress(f"Rigid mutual-information registration: level {registration.GetCurrentLevel() + 1}/{len(factors)}…")
    registration.AddCommand(sitk.sitkMultiResolutionIterationEvent, level)
    registration.Execute(fixed, moving)
    # SimpleITK's optimized transform maps fixed physical points to moving.
    # The public AsterView contract is the inverse: moving patient LPS→fixed.
    optimized_matrix = matrix_from_sitk(transform)
    valid_points = int(registration.GetMetricNumberOfValidPoints())
    if not np.isfinite(registration.GetMetricValue()) or valid_points < 128:
        raise RuntimeError("Insufficient valid overlap for rigid registration.")
    warnings = ["Review alignment on serial orthogonal slices; the metric is not an anatomical accuracy score."]
    # Compare both candidates on the same full working grid, independently of
    # sampled pyramid objectives. A local optimizer must not silently replace
    # a better user/center initialization with a worse final candidate.
    progress("Checking initial and optimized alignment on the full working grid…")
    evaluator = sitk.ImageRegistrationMethod()
    evaluator.SetMetricAsMattesMutualInformation(numberOfHistogramBins=48)
    evaluator.SetMetricFixedMask(fixed_mask)
    evaluator.SetMetricMovingMask(moving_mask)
    evaluator.SetInterpolator(sitk.sitkLinear)
    metrics = []
    for candidate in (initial_matrix, optimized_matrix):
        affine = sitk.AffineTransform(3)
        affine.SetMatrix(candidate[:3, :3].reshape(-1).tolist())
        affine.SetTranslation(candidate[:3, 3].tolist())
        evaluator.SetInitialTransform(affine)
        metrics.append(float(evaluator.MetricEvaluate(fixed, moving)))
    if not np.isfinite(metrics).all():
        raise RuntimeError("The alignment quality comparison was not finite.")
    retained_initial = metrics[0] < metrics[1]
    selected = initial_matrix if retained_initial else optimized_matrix
    metric = metrics[0] if retained_initial else metrics[1]
    moving_to_fixed = np.linalg.inv(selected)
    if retained_initial:
        warnings.append("Optimization did not improve the full-grid metric; the initial alignment was retained. Review or refine it.")
    if config["bone_focus"]:
        warnings.append("High-intensity focus is an intensity mask, not a validated stable-bone segmentation.")
    angle = np.rad2deg(np.arccos(np.clip((np.trace(moving_to_fixed[:3, :3]) - 1) / 2, -1, 1)))
    displacement = np.linalg.norm((moving_to_fixed[:3, :3] @ moving_center + moving_to_fixed[:3, 3]) - moving_center)
    if angle > 45 or displacement > 80:
        warnings.append("Large estimated motion requires particular care; refine or reject an incorrect local optimum.")
    initialization = "provided moving-to-fixed transform" if config["initial_moving_to_fixed"] is not None else config["initialization"]
    result = {"moving_to_fixed": moving_to_fixed.tolist(), "metric": metric, "warnings": warnings,
              "method": "SimpleITK2.5.2 rigid6DOF, MattesMI48, physical multiresolution, fixed seed20261020, one CPU thread; " + initialization + " initialization; full-grid initial/final check",
              "fixed_working_grid": fixed_grid, "moving_working_grid": moving_grid,
              "valid_metric_points": valid_points, "initial_full_grid_metric": metrics[0], "optimized_full_grid_metric": metrics[1],
              "retained_initial": bool(retained_initial), "optimizer_stop": registration.GetOptimizerStopConditionDescription(),
              "elapsed_seconds": time.monotonic() - started}
    (job / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    progress("Rigid alignment is ready for review and manual refinement.")


if __name__ == "__main__":
    main()
