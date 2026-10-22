"""Reusable, thread-confined native reslicing with reference geometry semantics.

VTK borrows the existing contiguous float32 source allocation; only the sampled
plane is copied for delivery. Patient LPS geometry is converted to continuous
voxel coordinates before entering VTK, avoiding direction/origin conventions
in the image pipeline. Slabs reuse the same native plane pipeline and physical
sample locations. Padding-aware interpolation uses the reference implementation.
"""

from __future__ import annotations

import os
from threading import get_ident
from typing import Callable, Sequence

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import VTK_FLOAT
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkImagingCore import vtkImageReslice

from .reformat import (
    VolumeLike,
    _check_cancel,
    _coordinates,
    _count,
    _frame,
    _index,
    _mode,
    _positive,
    _source,
    _slab_offsets,
    _vector,
    sample_plane as reference_sample_plane,
)


class FastReslicer:
    """Bind one immutable volume and reuse its native sampling pipeline.

    ``sample_plane`` has the reference sampler's arguments except ``volume``.
    Returned arrays own their pixels and survive subsequent calls. The native
    objects are created on the first sampling thread and may only be used on
    that thread. Create a separate instance for each persistent worker.

    Changing/replacing source data, its validity mask, or geometry requires a
    new instance. No source-sized allocation or copy occurs during binding or
    sampling. An optional ``threads`` cap affects only this VTK algorithm and
    does not alter global VTK thread settings.
    """

    def __init__(self, volume: VolumeLike, *, threads: int | None = None) -> None:
        self.volume = volume
        self._data, self._outside, self._spacing, mask, _ = _source(volume)
        self._native = (mask is None and self._data.dtype == np.float32
                        and self._data.flags.c_contiguous)
        self._threads = (_count(threads, "threads") if threads is not None
                         else min(8, os.cpu_count() or 1))
        self._owner_thread: int | None = None
        self._image = self._scalars = self._reslice = self._axes = None
        self._coordinate = self._valid = self._comparison = None

    @property
    def backend(self) -> str:
        """Sampling backend; validity-mask sources use the reference sampler."""
        return "vtk" if self._native else "reference"

    def _claim_thread(self) -> None:
        current = get_ident()
        if self._owner_thread is None:
            self._owner_thread = current
        elif self._owner_thread != current:
            raise RuntimeError("FastReslicer must remain on its first sampling thread.")

    def _ensure_pipeline(self) -> None:
        if self._reslice is not None:
            return
        self._image = vtkImageData()
        self._image.SetDimensions(*self._data.shape[::-1])
        # Keep both NumPy data and VTK scalars alive for the complete binding.
        # ravel() is a view because native eligibility requires C-contiguity.
        self._scalars = numpy_to_vtk(self._data.ravel(), deep=False)
        self._image.GetPointData().SetScalars(self._scalars)
        self._axes = vtkMatrix4x4()
        self._reslice = vtkImageReslice()
        self._reslice.SetInputData(self._image)
        self._reslice.SetResliceAxes(self._axes)
        self._reslice.SetOutputDimensionality(2)
        self._reslice.SetOutputOrigin(0, 0, 0)
        self._reslice.SetOutputScalarType(VTK_FLOAT)
        self._reslice.SetInterpolationModeToLinear()
        self._reslice.SetBackgroundLevel(self._outside)
        # The reference uses constant padding outside voxel-center extents,
        # not VTK's default half-voxel replicated edge.
        self._reslice.BorderOff()
        self._reslice.WrapOff()
        self._reslice.MirrorOff()
        self._reslice.SetEnableSMP(False)

    def _boundary_mask(self, origin: np.ndarray, right: np.ndarray, up: np.ndarray,
                       horizontal: np.ndarray, vertical: np.ndarray,
                       shape: tuple[int, int],
                       cancel: Callable[[], bool] | None,
                       displacement: np.ndarray | None = None) -> np.ndarray | None:
        """Enforce SciPy's strict center extent despite VTK border tolerance.

        Fully interior planes allocate no coordinate grid. Edge-crossing planes
        reuse one float64 plane and two Boolean planes, rather than a 3-D grid.
        Arithmetic order matches the reference at exact/near boundary points.
        """
        corners = (origin[:, None]
                   + right[:, None] * horizontal[[0, -1, 0, -1]]
                   + up[:, None] * vertical[[0, 0, -1, -1]])
        if displacement is not None:
            corners += displacement[:, None]
        upper = np.asarray(self._data.shape[::-1], dtype=np.float64) - 1
        if np.all(corners.min(axis=1) >= 0) and np.all(corners.max(axis=1) <= upper):
            return None
        if (self._coordinate is None or self._coordinate.shape != shape):
            self._coordinate = np.empty(shape, dtype=np.float64)
            self._valid = np.empty(shape, dtype=bool)
            self._comparison = np.empty(shape, dtype=bool)
        self._valid.fill(True)
        for axis in range(3):
            _check_cancel(cancel)
            self._coordinate[:] = origin[axis] + right[axis] * horizontal[None, :]
            self._coordinate += up[axis] * vertical[:, None]
            if displacement is not None:
                self._coordinate += displacement[axis]
            np.greater_equal(self._coordinate, 0, out=self._comparison)
            self._valid &= self._comparison
            np.less_equal(self._coordinate, upper[axis], out=self._comparison)
            self._valid &= self._comparison
        return self._valid

    def sample_plane(self, center: Sequence[float], right: Sequence[float],
                     up: Sequence[float], size_mm: tuple[float, float] = (150, 150),
                     shape: tuple[int, int] = (512, 512), slab_mm: float = 0,
                     mode: str = "mean",
                     cancel: Callable[[], bool] | None = None) -> np.ndarray:
        """Sample a plane; top row is +up and outer centers span ``size_mm``.

        Geometry, mode, cancellation exception and background semantics match
        ``reformat.sample_plane``. VTK linear interpolation has finite fractional
        precision, so small sub-voxel intensity-rounding differences are expected
        (the tests exercise analytical and textured phantoms).
        """
        _check_cancel(cancel)
        self._claim_thread()
        thickness = _positive(slab_mm, "slab thickness", allow_zero=True)
        if not self._native:
            return reference_sample_plane(self.volume, center, right, up, size_mm,
                                          shape, slab_mm, mode, cancel)

        center = _vector(center, "center")
        right, up, normal = _frame(right, up)
        if len(size_mm) != 2 or len(shape) != 2:
            raise ValueError("size_mm and shape must have two entries")
        width, height = (_positive(value, "plane size") for value in size_mm)
        rows, columns = (_count(value, "image dimension") for value in shape)
        mode = _mode(mode)
        offsets = _slab_offsets(thickness, self._spacing)
        reference = _index(self.volume, np.stack((center, center + right,
                                                center + up, center + normal)))
        origin = reference[0]
        right_index, up_index, normal_index = reference[1:] - origin
        horizontal = _coordinates(width, columns)
        vertical = -_coordinates(height, rows)
        self._ensure_pipeline()
        # The image pipeline works in unit voxel indices. Its columns are the
        # physical right/down/normal vectors transformed through world_to_index.
        axes = np.column_stack((right_index, -up_index, normal_index))
        first = origin + right_index * horizontal[0] + up_index * vertical[0]
        for row in range(3):
            for column in range(3):
                self._axes.SetElement(row, column, float(axes[row, column]))
        self._reslice.SetOutputExtent(0, columns - 1, 0, rows - 1, 0, 0)
        self._reslice.SetOutputSpacing(width / (columns - 1) if columns > 1 else 1,
                                       height / (rows - 1) if rows > 1 else 1, 1)
        # Small planes avoid native thread-launch overhead; a 512-square plane
        # uses at most eight threads. This is local to the worker's algorithm.
        self._reslice.SetNumberOfThreads(min(self._threads,
                                              max(1, rows * columns // 32768)))
        accumulated = None
        for offset in offsets:
            _check_cancel(cancel)
            displacement = normal_index * offset
            valid = self._boundary_mask(origin, right_index, up_index, horizontal,
                                        vertical, (rows, columns), cancel, displacement)
            if valid is not None and not np.any(valid):
                sampled = np.full((rows, columns), self._outside, dtype=np.float32)
            else:
                for row in range(3):
                    self._axes.SetElement(row, 3, float(first[row] + displacement[row]))
                _check_cancel(cancel)
                self._reslice.Update()
                _check_cancel(cancel)
                sampled = vtk_to_numpy(self._reslice.GetOutput().GetPointData().GetScalars()).reshape((rows, columns))
                if valid is not None:
                    np.copyto(sampled, self._outside, where=~valid)
                if not np.all(np.isfinite(sampled)):
                    raise ValueError("sampled volume contains nonfinite voxel values")
            # Consume VTK's borrowed output before the next Update. Only one
            # retained accumulation plane is needed regardless of thickness;
            # mean uses float64 like the reference to limit accumulation error.
            if accumulated is None:
                accumulated = sampled.astype(np.float64) if mode == "mean" and len(offsets) > 1 else sampled.copy()
            elif mode == "mean":
                accumulated += sampled
            elif mode == "max":
                np.maximum(accumulated, sampled, out=accumulated)
            else:
                np.minimum(accumulated, sampled, out=accumulated)
        if mode == "mean" and len(offsets) > 1:
            accumulated /= len(offsets)
        output = np.asarray(accumulated, dtype=np.float32)
        _check_cancel(cancel)
        return output
