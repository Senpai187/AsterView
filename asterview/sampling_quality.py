"""Display-aware sampling budgets without filtering or source-volume copies."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _positive_pair(values: Sequence[float], name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} requires two values")
    pair = tuple(float(value) for value in values)
    if not all(math.isfinite(value) and value > 0 for value in pair):
        raise ValueError(f"{name} must be finite and positive")
    return pair


def bounded_sample_shape(size_mm: Sequence[float], pixels_per_mm: float, *,
                         maximum: int = 2048, pixel_budget: int = 4_194_304,
                         minimum: int = 32) -> tuple[int, int]:
    """Return (rows, columns), preserving aspect within an explicit RAM budget."""
    width, height = _positive_pair(size_mm, "Physical size")
    if not math.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
        raise ValueError("Sampling density must be finite and positive")
    if (not isinstance(maximum, int) or not isinstance(pixel_budget, int)
            or not isinstance(minimum, int) or minimum < 2
            or maximum < minimum or pixel_budget < minimum * minimum):
        raise ValueError("Invalid image sampling budget")
    density = min(float(pixels_per_mm), (maximum - 1) / max(width, height),
                  math.sqrt(pixel_budget / (width * height)))
    rows = max(minimum, min(maximum, math.ceil(height * density) + 1))
    columns = max(minimum, min(maximum, math.ceil(width * density) + 1))
    # Ceil/+1 and the minimum dimension can exceed a strict caller budget.
    while rows * columns > pixel_budget:
        if rows >= columns and rows > minimum:
            rows -= 1
        elif columns > minimum:
            columns -= 1
        else:
            break
    return rows, columns


def display_sample_shape(size_mm: Sequence[float], viewport_pixels: Sequence[float], *,
                         device_ratio: float = 1., zoom: float = 1.,
                         quality: float = 1., maximum: int = 2048,
                         pixel_budget: int = 4_194_304) -> tuple[int, int]:
    """Sample at display density, including device pixels and magnification.

    Pixel values still come from the original trilinear reconstruction. This
    removes an avoidable low-resolution intermediate, not source limitations.
    The cap bounds work and memory even under extreme magnification.
    """
    width, height = _positive_pair(size_mm, "Physical size")
    screen_width, screen_height = _positive_pair(viewport_pixels, "Viewport size")
    if not all(math.isfinite(value) and value > 0 for value in (device_ratio, zoom, quality)):
        raise ValueError("Display scale must be finite and positive")
    density = min(screen_width / width, screen_height / height) * device_ratio * zoom * quality
    return bounded_sample_shape((width, height), density, maximum=maximum,
                                pixel_budget=pixel_budget)


def panorama_sampling_plan(points: Sequence[Sequence[float]], height_mm: float,
                           spacing: Sequence[float], *,
                           display_shape: tuple[int, int] | None = None,
                           maximum: int = 2048,
                           pixel_budget: int = 4_194_304) -> tuple[tuple[float, float], tuple[int, int]]:
    """Plan CPR pixels using its physical arch length, source pitch and display.

    Returns ``((arc_length_mm, height_mm), (rows, columns))``. At least one
    sample per smallest source voxel is requested, with higher sampling when
    required by the display. No fixed 512-column/360-row intermediate remains.
    """
    from .reformat import resample_path

    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Spacing must contain three finite positive values")
    path = resample_path(points, samples=256)
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    size = _positive_pair((length, height_mm), "Panoramic size")
    density = 1. / float(spacing.min())
    if display_shape is not None:
        rows, columns = _positive_pair(display_shape, "Display shape")
        density = max(density, (rows - 1) / height_mm, (columns - 1) / length)
    return size, bounded_sample_shape(size, density, maximum=maximum, pixel_budget=pixel_budget)
