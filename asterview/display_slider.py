"""Continuous brightness/contrast controls backed by exact window values."""
import math
from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtWidgets import QWidget, QHBoxLayout, QSlider


class DisplaySlider(QWidget):
    def __init__(self, value_control, name, *, logarithmic=False, inverted=False):
        super().__init__()
        self.control = value_control
        self.logarithmic, self.inverted = logarithmic, inverted
        self.low, self.high = (1., 6000.) if logarithmic else (-1500., 3000.)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 2000)
        self.slider.setAccessibleName(name)
        self.slider.setToolTip(name + " · drag or use arrow keys; enter an exact value at the right")
        value_control.setFixedWidth(94)
        # Styled NoButtons produces a one-pixel edit field on the bundled Qt
        # style, regardless of the spin box width. Keep normal edit allocation
        # and hide only its stepping buttons; keyboard/value editing remains.
        value_control.setButtonSymbols(value_control.ButtonSymbols.UpDownArrows)
        value_control.setStyleSheet(
            "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {"
            " width: 0; height: 0; border: none; }"
        )
        layout.addWidget(self.slider, 1)
        layout.addWidget(value_control)
        self.slider.valueChanged.connect(self.from_slider)
        value_control.valueChanged.connect(self.from_value)
        self.from_value()

    def set_limits(self, low, high):
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError("Display slider requires a finite increasing range")
        if self.logarithmic and low <= 0:
            raise ValueError("Contrast range must be positive")
        self.low, self.high = float(low), float(high)
        self.from_value()

    def from_value(self, *_):
        value = min(self.high, max(self.low, self.control.value()))
        if self.logarithmic:
            fraction = math.log(value / self.low) / math.log(self.high / self.low)
        else:
            fraction = (value - self.low) / (self.high - self.low)
        if self.inverted:
            fraction = 1 - fraction
        with QSignalBlocker(self.slider):
            self.slider.setValue(round(fraction * 2000))

    def from_slider(self, value):
        fraction = value / 2000
        if self.inverted:
            fraction = 1 - fraction
        if self.logarithmic:
            result = self.low * (self.high / self.low) ** fraction
        else:
            result = self.low + (self.high - self.low) * fraction
        self.control.setValue(result)
