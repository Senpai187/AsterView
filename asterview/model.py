"""Volume geometry in DICOM's patient LPS coordinate system.

Array order is z, y, x; all public coordinates and spacing are x, y, z.
The direction matrix columns are the increasing x/y/z voxel axes in LPS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np


@dataclass
class Volume:
    data: np.ndarray
    spacing: tuple[float, float, float]
    origin: np.ndarray
    direction: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    window: tuple[float, float] = (400.0, 1800.0)
    is_hu: bool = False
    warnings: list[str] = field(default_factory=list)
    data_min: float = field(init=False)
    data_max: float = field(init=False)
    # Appended to preserve the positional constructor used by existing clients.
    valid_mask: np.ndarray | None = None
    padding_value: float | None = None

    def __post_init__(self) -> None:
        self.data = np.ascontiguousarray(self.data, dtype=np.float32)
        if self.data.ndim != 3 or any(n < 1 for n in self.data.shape):
            raise ValueError("A volume must be a nonempty z, y, x array.")
        if self.valid_mask is not None:
            mask = np.asarray(self.valid_mask)
            if mask.dtype != np.bool_ or mask.shape != self.data.shape:
                raise ValueError("A validity mask must be boolean and match the z, y, x voxel shape.")
            self.valid_mask = np.ascontiguousarray(mask)
        elif self.padding_value is not None:
            raise ValueError("A padding sentinel requires an explicit validity mask.")
        # Bounded scratch space even for large studies; one validation pass.
        # Padding is not anatomy and never contributes to intensity statistics.
        low, high = np.inf, -np.inf
        flat = self.data.reshape(-1)
        mask_flat = self.valid_mask.reshape(-1) if self.valid_mask is not None else None
        for start in range(0, flat.size, 1_048_576):
            block = flat[start:start + 1_048_576]
            valid = block if mask_flat is None else block[mask_flat[start:start + block.size]]
            if not valid.size:
                continue
            if not np.isfinite(valid).all():
                raise ValueError("Voxel data must contain only finite values.")
            low, high = min(low, float(valid.min())), max(high, float(valid.max()))
        if not np.isfinite(low):
            raise ValueError("The volume contains no valid non-padding voxels.")
        self.data_min, self.data_max = float(low), float(high)
        if mask_flat is not None:
            if self.padding_value is None:
                # A nearby finite float32 sentinel is numerically stable for
                # GPU interpolation; validity-aware 2D sampling ignores it.
                candidate = max(-np.finfo(np.float32).max, low - max(1., (high - low) * 0.01))
                sentinel = np.float32(candidate)
                if not sentinel < low:
                    with np.errstate(over="ignore", invalid="ignore"):
                        sentinel = np.nextafter(np.float32(low), np.float32(-np.inf))
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    sentinel = np.float32(self.padding_value)
            if not np.isfinite(sentinel) or not sentinel < low:
                raise ValueError("Padding requires a finite float32 sentinel below the valid intensity range.")
            self.padding_value = float(sentinel)
            for start in range(0, flat.size, 1_048_576):
                block = flat[start:start + 1_048_576]
                block[~mask_flat[start:start + block.size]] = sentinel
        self.spacing = tuple(float(v) for v in self.spacing)
        if len(self.spacing) != 3 or not np.all(np.isfinite(self.spacing)) or min(self.spacing) <= 0:
            raise ValueError("Spacing must contain three positive finite millimeter values.")
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(3).copy()
        self.direction = np.asarray(self.direction, dtype=np.float64).reshape(3, 3).copy()
        if not np.all(np.isfinite(self.origin)) or not np.all(np.isfinite(self.direction)):
            raise ValueError("Geometry must be finite.")
        if not np.allclose(self.direction.T @ self.direction, np.eye(3), atol=1e-4):
            raise ValueError("Direction axes must be orthonormal.")
        self.window = tuple(float(v) for v in self.window)
        if len(self.window) != 2 or not np.all(np.isfinite(self.window)) or self.window[1] <= 0:
            raise ValueError("Window must be a finite center and positive width.")

    @property
    def dimensions(self) -> tuple[int, int, int]:
        """Voxel counts in x, y, z order."""
        return tuple(int(v) for v in self.data.shape[::-1])

    @property
    def center(self) -> np.ndarray:
        """Center of the voxel-center extent, in patient LPS millimeters."""
        return self.index_to_world((np.asarray(self.dimensions) - 1) / 2)

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """Axis-aligned LPS bounds of voxel centers (VTK image convention)."""
        corners = np.array(list(product(*[(0, n - 1) for n in self.dimensions])))
        world = self.index_to_world(corners)
        low, high = world.min(axis=0), world.max(axis=0)
        return tuple(float(v) for pair in zip(low, high) for v in pair)

    def index_to_world(self, points: np.ndarray | tuple) -> np.ndarray:
        """Convert (..., 3) floating voxel x/y/z indices to patient LPS mm."""
        points = np.asarray(points, dtype=np.float64)
        if points.shape[-1:] != (3,):
            raise ValueError("Coordinates must have a final dimension of three.")
        return (points * self.spacing) @ self.direction.T + self.origin

    def world_to_index(self, points: np.ndarray | tuple) -> np.ndarray:
        """Convert (..., 3) patient LPS mm to floating voxel x/y/z indices."""
        points = np.asarray(points, dtype=np.float64)
        if points.shape[-1:] != (3,):
            raise ValueError("Coordinates must have a final dimension of three.")
        return ((points - self.origin) @ self.direction) / self.spacing
