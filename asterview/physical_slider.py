"""A bounded physical slider; only lightweight coordinates live in the widget."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import QSlider


class PhysicalSlider(QSlider):
    position_changed = Signal(float)

    def __init__(self, orientation=Qt.Orientation.Vertical, parent=None):
        super().__init__(orientation, parent)
        self.physical_limits = (0., 1.)
        self.physical_value = 0.
        self.setRange(0, 100_000)
        self.setAccessibleName("Section position in millimeters")
        self.setStyleSheet(
            "QSlider::groove:vertical { background:#333943; width:4px; border-radius:2px; }"
            "QSlider::handle:vertical { background:#8cbaf2; height:18px; margin:0 -4px; border-radius:4px; }"
            "QSlider::groove:horizontal { background:#333943; height:4px; border-radius:2px; }"
            "QSlider::handle:horizontal { background:#8cbaf2; width:18px; margin:-4px 0; border-radius:4px; }")
        self.valueChanged.connect(self._changed)

    def set_position(self, minimum, maximum, value, step=.1):
        minimum, maximum, value, step = map(float, (minimum, maximum, value, step))
        if not np.isfinite((minimum, maximum, value, step)).all() or maximum < minimum or step <= 0:
            raise ValueError("Physical slider requires finite limits and a positive step")
        self.physical_limits = minimum, maximum
        self.physical_value = float(np.clip(value, minimum, maximum))
        span = maximum - minimum
        with QSignalBlocker(self):
            self.setValue(0 if span <= 0 else round((self.physical_value - minimum) / span * 100_000))
            increment = 1 if span <= 0 else max(1, round(step / span * 100_000))
            self.setSingleStep(increment)
            self.setPageStep(increment * 5)
        self.setEnabled(span > 1e-8)
        self.setToolTip(f"{self.physical_value:.2f} mm · range {minimum:.2f} to {maximum:.2f} mm")

    def _changed(self, value):
        low, high = self.physical_limits
        self.physical_value = low + value / 100_000 * (high - low)
        self.setToolTip(f"{self.physical_value:.2f} mm")
        self.position_changed.emit(self.physical_value)


def line_scroll_limits(volume, position, normal):
    """Signed LPS positions along a line through the acquired voxel-center box."""
    position, normal = np.asarray(position, float), np.asarray(normal, float)
    normal = normal / np.linalg.norm(normal)
    index = volume.world_to_index(position)
    direction = volume.world_to_index(position + normal) - index
    lower, upper = -np.inf, np.inf
    for coordinate, delta, count in zip(index, direction, volume.dimensions):
        if abs(delta) < 1e-12:
            if coordinate < -1e-6 or coordinate > count - 1 + 1e-6:
                return float(position @ normal), float(position @ normal)
            continue
        bounds = sorted((-coordinate / delta, (count - 1 - coordinate) / delta))
        lower, upper = max(lower, bounds[0]), min(upper, bounds[1])
    base = float(position @ normal)
    if lower > upper or not np.isfinite((lower, upper)).all():
        return base, base
    return base + lower, base + upper
