"""Physical panoramic row bounds relative to an editable arch plane."""
from __future__ import annotations

from itertools import product
import numpy as np


def validate_bounds(bounds):
    values = np.asarray(bounds, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all() or values[1] <= values[0]:
        raise ValueError("Panoramic lower and upper bounds must be finite and increasing")
    return float(values[0]), float(values[1])


def acquisition_bounds(volume, points, up):
    points, up = np.asarray(points, float), np.asarray(up, float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1 or not np.isfinite(points).all():
        raise ValueError("A finite arch is required for panoramic range controls")
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-8:
        raise ValueError("A finite vertical direction is required")
    up = up / np.linalg.norm(up)
    corners = volume.index_to_world(np.asarray(list(product(*[(0, n - 1) for n in volume.dimensions]))))
    origin = points.mean(axis=0)
    levels = (corners - origin) @ up
    return validate_bounds((levels.min(), levels.max()))


def range_presets(volume, points, up):
    """Geometric presets, deliberately not unverified maxilla/mandible labels."""
    lower, upper = acquisition_bounds(volume, points, up)
    middle = float(np.clip(0., lower + (upper - lower) * .1, upper - (upper - lower) * .1))
    return {"Full scan": (lower, upper), "Above arch": (middle, upper),
            "Below arch": (lower, middle)}
