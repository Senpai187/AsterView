"""Physical, continuous section positions along an editable panoramic arch."""
from __future__ import annotations

import numpy as np


class ArchSectionGeometry:
    """Small cached path geometry; never owns or resamples source voxels."""

    def __init__(self, path, up):
        path = np.asarray(path, dtype=np.float64)
        up = np.asarray(up, dtype=np.float64)
        if (path.ndim != 2 or path.shape[1] != 3 or len(path) < 2
                or not np.all(np.isfinite(path)) or up.shape != (3,)
                or not np.all(np.isfinite(up)) or np.linalg.norm(up) < 1e-8):
            raise ValueError("Arch section geometry requires a finite path and vertical axis")
        self.path = path.copy()
        self.up = up / np.linalg.norm(up)
        self.delta = np.diff(self.path, axis=0)
        self.lengths = np.linalg.norm(self.delta, axis=1)
        if np.any(self.lengths < 1e-8):
            raise ValueError("Consecutive arch positions must be distinct")
        self.arc = np.r_[0., np.cumsum(self.lengths)]
        tangents = np.gradient(self.path, axis=0)
        tangents -= (tangents @ self.up)[:, None] * self.up
        magnitude = np.linalg.norm(tangents, axis=1)
        if np.any(magnitude < 1e-8):
            raise ValueError("An arch tangent cannot be parallel to the vertical axis")
        self.right = np.cross(self.up, tangents / magnitude[:, None])

    @property
    def length(self):
        return float(self.arc[-1])

    def nearest_distance(self, point):
        """Project onto segments rather than snapping the range to pixel columns."""
        point = np.asarray(point, dtype=np.float64)
        fraction = np.clip(np.einsum("ij,ij->i", point - self.path[:-1], self.delta)
                           / self.lengths ** 2, 0., 1.)
        positions = self.path[:-1] + fraction[:, None] * self.delta
        index = int(np.argmin(np.sum((positions - point) ** 2, axis=1)))
        return float(self.arc[index] + fraction[index] * self.lengths[index])

    def frame(self, distance_mm):
        """Interpolate a section, or return None beyond the traced anatomy."""
        distance = float(distance_mm)
        if not np.isfinite(distance) or distance < -1e-8 or distance > self.length + 1e-8:
            return None
        distance = min(self.length, max(0., distance))
        index = min(len(self.path) - 2, max(0, int(np.searchsorted(self.arc, distance, side="right") - 1)))
        fraction = (distance - self.arc[index]) / self.lengths[index]
        center = self.path[index] + fraction * self.delta[index]
        right = self.right[index] + fraction * (self.right[index + 1] - self.right[index])
        right /= np.linalg.norm(right)
        return center, right, self.up

    def offset_limits(self, anchor_mm, count, spacing_mm):
        """Keep a complete range inside the arch where its physical span fits."""
        half = (count - 1) * float(spacing_mm) / 2
        if 2 * half <= self.length:
            return half - anchor_mm, self.length - half - anchor_mm
        # A large matrix may exceed a short arch. Keep the range centered;
        # the out-of-arch cells stay empty, never duplicate terminal sections.
        midpoint = self.length / 2 - anchor_mm
        return midpoint, midpoint
