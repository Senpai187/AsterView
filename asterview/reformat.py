"""Physical-coordinate multiplanar and curved-planar reconstruction.

All positions and directions use DICOM patient LPS coordinates in millimeters.
Volume voxels have NumPy order ``(z, y, x)``; transforms take ``(..., x, y, z)``.
Images have their first row at the positive ``up`` edge. Sampling a plane never
resamples or copies the complete source volume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import map_coordinates


class VolumeLike(Protocol):
    data: np.ndarray
    spacing: Sequence[float]
    data_min: float
    valid_mask: np.ndarray | None
    padding_value: float | None

    def world_to_index(self, points: np.ndarray) -> np.ndarray: ...


_MAX_TILE_PIXELS = 262_144
_EPS = 1e-10


class ReconstructionCancelled(RuntimeError):
    """A reconstruction stopped at an explicit cancellation checkpoint."""


def _check_cancel(cancel: Callable[[], bool] | None) -> None:
    if cancel is not None and cancel():
        raise ReconstructionCancelled("Reconstruction canceled.")


def _vector(value: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite coordinates")
    return result


def _unit(value: Sequence[float], name: str) -> np.ndarray:
    result = _vector(value, name)
    length = np.linalg.norm(result)
    if length < _EPS:
        raise ValueError(f"{name} must have nonzero length")
    return result / length


def _frame(right: Sequence[float], up: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    right_axis = _unit(right, "right")
    up_axis = _unit(up, "up")
    up_axis = _unit(up_axis - np.dot(up_axis, right_axis) * right_axis, "orthogonal up")
    return right_axis, up_axis, np.cross(right_axis, up_axis)


def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'} and finite")
    return result


def _count(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if result != value or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _coordinates(extent_mm: float, count: int) -> np.ndarray:
    if count == 1:
        return np.zeros(1, dtype=np.float64)
    return np.linspace(-extent_mm / 2, extent_mm / 2, count, dtype=np.float64)


def _source(volume: VolumeLike) -> tuple[np.ndarray, float, np.ndarray, np.ndarray | None, float | None]:
    data = np.asarray(volume.data)
    if data.ndim != 3 or min(data.shape) < 1 or data.dtype.kind not in "fiu":
        raise ValueError("volume.data must be a nonempty, real numeric 3-D array")
    spacing = np.asarray(volume.spacing, dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("volume.spacing must contain three positive finite values")
    # The production Volume validates finite voxels and caches its range once.
    # This fallback keeps the functions usable with small custom Volume objects.
    minimum = getattr(volume, "data_min", None)
    if minimum is None:
        minimum, maximum = float(np.min(data)), float(np.max(data))
        if not np.isfinite(maximum):
            raise ValueError("volume contains nonfinite voxel values")
    minimum = float(minimum)
    if not np.isfinite(minimum):
        raise ValueError("volume contains nonfinite voxel values")
    mask = getattr(volume, "valid_mask", None)
    padding_value = getattr(volume, "padding_value", None)
    if mask is not None:
        if not isinstance(mask, np.ndarray) or mask.shape != data.shape or mask.dtype != np.bool_:
            raise ValueError("volume.valid_mask must be a Boolean array matching volume.data")
        if padding_value is None or not np.isfinite(padding_value) or padding_value >= minimum:
            raise ValueError("masked volume requires a finite padding value below valid intensities")
        padding_value = float(padding_value)
    return data, minimum, spacing, mask, padding_value


def _index(volume: VolumeLike, points: np.ndarray) -> np.ndarray:
    result = np.asarray(volume.world_to_index(points), dtype=np.float64)
    if result.shape != points.shape or not np.all(np.isfinite(result)):
        raise ValueError("volume.world_to_index returned invalid coordinates")
    return result


def _slab_offsets(thickness_mm: float, spacing: np.ndarray) -> np.ndarray:
    thickness = _positive(thickness_mm, "slab thickness", allow_zero=True)
    if thickness == 0:
        return np.zeros(1, dtype=np.float64)
    # Odd sample count includes the center; physical step never exceeds the
    # smallest voxel pitch, including in an oblique plane.
    half_steps = max(1, int(np.ceil(thickness / (2 * float(np.min(spacing))))))
    return np.linspace(-thickness / 2, thickness / 2, 2 * half_steps + 1)


def _mode(mode: str) -> str:
    normalized = str(mode).lower()
    if normalized not in {"mean", "max", "min"}:
        raise ValueError("mode must be 'mean', 'max', or 'min'")
    return normalized


def _reduce_samples(data: np.ndarray, base_zyx: np.ndarray, normal_zyx: np.ndarray,
                    offsets: np.ndarray, mode: str, outside: float,
                    valid_mask: np.ndarray | None = None,
                    padding_value: float | None = None,
                    cancel: Callable[[], bool] | None = None) -> np.ndarray:
    if valid_mask is not None:
        return _reduce_masked_samples(data, valid_mask, padding_value, base_zyx,
                                      normal_zyx, offsets, mode, outside, cancel)
    accumulated = None
    for offset in offsets:
        _check_cancel(cancel)
        coordinates = base_zyx if offset == 0 else base_zyx + offset * normal_zyx
        sampled = map_coordinates(data, coordinates, order=1, mode="constant",
                                  cval=outside, prefilter=False, output=np.float32)
        if not np.all(np.isfinite(sampled)):
            raise ValueError("sampled volume contains nonfinite voxel values")
        if accumulated is None:
            accumulated = sampled.astype(np.float64) if mode == "mean" else sampled
        elif mode == "mean":
            accumulated += sampled
        elif mode == "max":
            np.maximum(accumulated, sampled, out=accumulated)
        else:
            np.minimum(accumulated, sampled, out=accumulated)
    if mode == "mean":
        accumulated /= len(offsets)
    return np.asarray(accumulated, dtype=np.float32)


def _reduce_masked_samples(data: np.ndarray, valid_mask: np.ndarray, padding_value: float,
                           base_zyx: np.ndarray, normal_zyx: np.ndarray,
                           offsets: np.ndarray, mode: str, outside: float,
                           cancel: Callable[[], bool] | None = None) -> np.ndarray:
    """Interpolate only valid anatomy, without copying the source scalar array.

    Invalid source values are the loader's common finite padding sentinel.
    Interpolating the scalar and Boolean mask separately gives the weighted
    numerator after subtracting the known sentinel contribution. Mean slabs
    combine these numerators and weights; extrema ignore invalid samples.
    """
    shape = base_zyx.shape[1:]
    if mode == "mean":
        total = np.zeros(shape, dtype=np.float64)
        total_weight = np.zeros(shape, dtype=np.float64)
    else:
        best = np.full(shape, -np.inf if mode == "max" else np.inf, dtype=np.float64)
        has_valid = np.zeros(shape, dtype=bool)
    for offset in offsets:
        _check_cancel(cancel)
        coordinates = base_zyx if offset == 0 else base_zyx + offset * normal_zyx
        weights = map_coordinates(valid_mask, coordinates, order=1, mode="constant",
                                  cval=0, prefilter=False, output=np.float64)
        values = map_coordinates(data, coordinates, order=1, mode="constant",
                                 cval=padding_value, prefilter=False, output=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("sampled volume contains nonfinite voxel values")
        valid = weights > 1e-7
        numerator = values - padding_value * (1 - weights)
        if mode == "mean":
            # Zero-weight interpolation contributes neither background nor a
            # denominator term, including samples outside the source extent.
            np.add(total, numerator, out=total, where=valid)
            np.add(total_weight, weights, out=total_weight, where=valid)
        else:
            sampled = np.divide(numerator, weights, out=np.zeros(shape), where=valid)
            if mode == "max":
                np.maximum(best, sampled, out=best, where=valid)
            else:
                np.minimum(best, sampled, out=best, where=valid)
            has_valid |= valid
    output = np.full(shape, outside, dtype=np.float64)
    if mode == "mean":
        np.divide(total, total_weight, out=output, where=total_weight > 1e-7)
    else:
        np.copyto(output, best, where=has_valid)
    return output.astype(np.float32)


def rotation_matrix(rx: float = 0, ry: float = 0, rz: float = 0) -> np.ndarray:
    """Return active LPS Euler rotation ``Rz @ Ry @ Rx``; angles are degrees."""
    angles = _vector((rx, ry, rz), "rotation angles")
    x, y, z = np.deg2rad(angles)
    cx, cy, cz = np.cos((x, y, z))
    sx, sy, sz = np.sin((x, y, z))
    rotate_x = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    rotate_y = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    rotate_z = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    return rotate_z @ rotate_y @ rotate_x


def base_frame(name: str, rotation: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return radiological display ``(right, up)`` unit vectors in LPS.

    Axial: screen right = patient left; screen up = anterior.
    Coronal: screen right = patient left; screen up = superior.
    Sagittal: screen right = posterior; screen up = superior.
    An optional orthonormal 3 x 3 active LPS rotation corrects all sections.
    """
    frames = {
        "axial": ((1, 0, 0), (0, -1, 0)),
        "coronal": ((1, 0, 0), (0, 0, 1)),
        "sagittal": ((0, 1, 0), (0, 0, 1)),
    }
    try:
        right, up = (np.asarray(axis, dtype=np.float64) for axis in frames[name.lower()])
    except (KeyError, AttributeError) as exc:
        raise ValueError("plane must be 'axial', 'coronal', or 'sagittal'") from exc
    if rotation is not None:
        matrix = np.asarray(rotation, dtype=np.float64)
        if (matrix.shape != (3, 3) or not np.all(np.isfinite(matrix))
                or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(matrix), 1, atol=1e-6)):
            raise ValueError("rotation must be a finite orthonormal 3 x 3 rotation")
        right, up = matrix @ right, matrix @ up
    return right, up


def sample_plane(volume: VolumeLike, center: Sequence[float], right: Sequence[float],
                 up: Sequence[float], size_mm: tuple[float, float] = (150, 150),
                 shape: tuple[int, int] = (512, 512), slab_mm: float = 0,
                 mode: str = "mean", cancel: Callable[[], bool] | None = None) -> np.ndarray:
    """Sample a physical plane or slab with trilinear interpolation.

    ``size_mm`` is (width, height), ``shape`` is (rows, columns). For dimensions
    larger than one pixel the outer pixel centers span exactly the requested
    size, so measurement pitch is ``size_mm / (pixel_count - 1)``. A one-pixel
    dimension samples the center. Slab modes are mean, maximum and minimum;
    samples outside the volume use its minimum value. When a validity mask is
    present, interpolation and slab statistics exclude DICOM padding; an
    entirely invalid result uses the minimum valid anatomy value.
    A true ``cancel()`` callback raises :class:`ReconstructionCancelled` between
    row tiles and slab samples.
    """
    _check_cancel(cancel)
    data, outside, spacing, valid_mask, padding_value = _source(volume)
    center = _vector(center, "center")
    right, up, normal = _frame(right, up)
    if len(size_mm) != 2 or len(shape) != 2:
        raise ValueError("size_mm and shape must have two entries")
    width, height = (_positive(value, "plane size") for value in size_mm)
    rows, columns = (_count(value, "image dimension") for value in shape)
    mode = _mode(mode)
    offsets = _slab_offsets(slab_mm, spacing)
    reference = _index(volume, np.stack((center, center + right, center + up, center + normal)))
    origin_xyz = reference[0]
    right_xyz, up_xyz, normal_xyz = reference[1:] - origin_xyz
    horizontal = _coordinates(width, columns)
    vertical = -_coordinates(height, rows)
    output = np.empty((rows, columns), dtype=np.float32)
    tile_rows = max(1, _MAX_TILE_PIXELS // columns)
    for first in range(0, rows, tile_rows):
        _check_cancel(cancel)
        last = min(rows, first + tile_rows)
        coordinates = (origin_xyz[:, None, None]
                       + right_xyz[:, None, None] * horizontal[None, None, :]
                       + up_xyz[:, None, None] * vertical[None, first:last, None])
        output[first:last] = _reduce_samples(
            data, coordinates[::-1], normal_xyz[::-1, None, None], offsets, mode, outside,
            valid_mask, padding_value, cancel)
    _check_cancel(cancel)
    return output


def resample_path(points: Sequence[Sequence[float]], samples: int = 512) -> np.ndarray:
    """Interpolate an open 3-D arch and return approximately uniform arc steps.

    A natural cubic spline passes through at least three distinct control
    points. Consecutive duplicate control points are removed. The spline is
    integrated numerically on a dense grid and inverted by arc length.
    """
    points = np.asarray(points, dtype=np.float64)
    samples = _count(samples, "path samples", minimum=3)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("arch points must be a finite N x 3 array")
    if len(points) < 3:
        raise ValueError("at least three distinct arch points are required")
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > _EPS]
    points = points[keep]
    if len(points) < 3:
        raise ValueError("at least three distinct arch points are required")
    chord = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    spline = CubicSpline(chord, points, axis=0, bc_type="natural")
    dense_parameter = np.linspace(0, chord[-1], max(4096, samples * 8, len(points) * 64))
    dense_points = spline(dense_parameter)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(dense_points, axis=0), axis=1))]
    if arc[-1] < _EPS or not np.isfinite(arc[-1]):
        raise ValueError("arch has no finite physical length")
    target_parameter = np.interp(np.linspace(0, arc[-1], samples), arc, dense_parameter)
    return np.asarray(spline(target_parameter), dtype=np.float64)


def _path_axes(path: np.ndarray, up: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    up_axis = _unit(up, "up")
    tangent = np.gradient(path, axis=0)
    tangent -= (tangent @ up_axis)[:, None] * up_axis
    lengths = np.linalg.norm(tangent, axis=1)
    if np.any(lengths < _EPS):
        raise ValueError("arch tangent must not be parallel to up or stationary")
    tangent /= lengths[:, None]
    return up_axis, np.cross(up_axis, tangent)


def path_frame(path: Sequence[Sequence[float]], index: int,
               up: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(center, right, up)`` for a cross section at a path column.

    Cross-section normal follows the arch tangent, and the right axis is
    ``cross(up, tangent)``. ``path`` is normally the panoramic result's .path.
    """
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 or not np.all(np.isfinite(path)):
        raise ValueError("path must be a finite N x 3 array with at least two points")
    index = _count(index, "path index", minimum=0)
    if index >= len(path):
        raise ValueError("path index is outside the path")
    up_axis, right = _path_axes(path, up)
    return path[index].copy(), right[index].copy(), up_axis.copy()


@dataclass(frozen=True)
class PanoramicResult:
    """Panoramic intensities, sampled LPS arch and physical measurement pitch."""

    image: np.ndarray
    path: np.ndarray
    column_pitch_mm: float
    row_pitch_mm: float
    vertical_bounds_mm: tuple[float, float] | None = None

    @property
    def vertical_limits_mm(self):
        height = (self.image.shape[0] - 1) * self.row_pitch_mm
        return self.vertical_bounds_mm or (-height / 2, height / 2)

    def offset_at_row(self, row):
        if self.image.shape[0] == 1:
            return np.asarray(row) * 0 + sum(self.vertical_limits_mm) / 2
        return self.vertical_limits_mm[1] - np.asarray(row) * self.row_pitch_mm

    def row_at_offset(self, offset):
        return 0. if self.row_pitch_mm == 0 else (self.vertical_limits_mm[1] - np.asarray(offset)) / self.row_pitch_mm

    def __iter__(self):
        # Supports the convenient ``image, path = curved_planar(...)`` form.
        yield self.image
        yield self.path


def curved_planar(volume: VolumeLike, points: Sequence[Sequence[float]], up: Sequence[float],
                  height_mm: float = 80, thickness_mm: float = 5, samples: int = 512,
                  rows: int = 300, mode: str = "mean",
                  cancel: Callable[[], bool] | None = None, *,
                  focal_offset_mm: float = 0,
                  vertical_bounds_mm: tuple[float, float] | None = None) -> PanoramicResult:
    """Reconstruct an open dental arch with a symmetric physical slab.

    Columns follow the arc-length spline. Rows run from +up to -up. At each
    column slab samples lie perpendicular to the local arch tangent and up.
    Control points normally share an axial plane; arbitrary 3-D arches retain
    their supplied heights. The function does not infer or detect an arch.
    ``focal_offset_mm`` shifts the focal trough along ``cross(up, tangent)``.
    The offset curve is resampled at uniform arc steps before reconstruction;
    result.path and its physical pitch describe that displayed trough.
    A true cancellation callback raises :class:`ReconstructionCancelled`.
    """
    _check_cancel(cancel)
    data, outside, spacing, valid_mask, padding_value = _source(volume)
    height = _positive(height_mm, "panoramic height")
    if vertical_bounds_mm is not None:
        from .panoramic_range import validate_bounds
        lower, upper = validate_bounds(vertical_bounds_mm)
        height = upper - lower
    else:
        lower, upper = -height / 2, height / 2
    rows = _count(rows, "panoramic rows")
    mode = _mode(mode)
    path = resample_path(points, samples)
    up_axis, normals = _path_axes(path, up)
    focal_offset = float(focal_offset_mm)
    if not np.isfinite(focal_offset):
        raise ValueError("Panoramic focal offset must be finite")
    if focal_offset != 0:
        _check_cancel(cancel)
        path = resample_path(path + focal_offset * normals, samples)
        up_axis, normals = _path_axes(path, up)
    offsets = _slab_offsets(thickness_mm, spacing)
    path_index = _index(volume, path)
    up_index = _index(volume, path[:1] + up_axis)[0] - path_index[0]
    normal_index = _index(volume, path + normals) - path_index
    vertical = np.linspace(upper, lower, rows) if rows > 1 else np.array([(upper + lower) / 2])
    output = np.empty((rows, len(path)), dtype=np.float32)
    tile_rows = max(1, _MAX_TILE_PIXELS // len(path))
    for first in range(0, rows, tile_rows):
        _check_cancel(cancel)
        last = min(rows, first + tile_rows)
        coordinates = (path_index.T[:, None, :]
                       + up_index[:, None, None] * vertical[None, first:last, None])
        output[first:last] = _reduce_samples(
            data, coordinates[::-1], normal_index.T[::-1, None, :], offsets, mode, outside,
            valid_mask, padding_value, cancel)
    _check_cancel(cancel)
    arc_length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    return PanoramicResult(output, path, arc_length / (len(path) - 1),
                           height / (rows - 1) if rows > 1 else 0.0, (lower, upper))
