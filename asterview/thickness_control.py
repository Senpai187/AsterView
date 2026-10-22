"""Preset-first physical slab control with the existing numeric widget contract."""
from __future__ import annotations

import math
import re

from PySide6.QtCore import QSignalBlocker, Signal, Qt
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QWidget


# Practical AsterView choices, not a claim about a vendor's complete preset list.
THICKNESS_PRESETS_MM = (.2, .4, .6, .8, 1., 1.2, 1.5, 2., 3., 4., 5., 8., 10., 15., 20., 30.)


class ThicknessControl(QWidget):
    """Physical slab choices, bounded below by the source voxel pitch."""
    valueChanged = Signal(float)
    editingFinished = Signal()

    def __init__(self, value=0., parent=None):
        super().__init__(parent)
        self._value = .2
        self._minimum, self._maximum = .2, 30.
        self._decimals = 3
        self._suffix = " mm"
        self._special = ""
        self._single_step = .2
        self._native_pitch = None
        self.combo = QComboBox(self)
        self.combo.setEditable(True)
        self.combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.combo.setMaxVisibleItems(12)
        self.combo.setMinimumWidth(82)
        self.combo.setToolTip("Choose a thickness or type a value in millimeters")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.combo)
        self.setFocusProxy(self.combo)
        self._rebuild()
        self.combo.currentIndexChanged.connect(self._selected)
        self.combo.lineEdit().editingFinished.connect(self._commit_text)
        self.setValue(value)

    def value(self):
        return self._value

    def _text(self, value):
        return self._special if value == 0 and self._special else f"{value:.{self._decimals}f}{self._suffix}"

    def _sync(self):
        with QSignalBlocker(self.combo):
            index = next((i for i in range(self.combo.count())
                          if abs(float(self.combo.itemData(i)) - self._value) < 10 ** (-self._decimals - 1)), -1)
            self.combo.setCurrentIndex(index)
            self.combo.setEditText(self._text(self._value))

    def setValue(self, value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Slice thickness must be finite")
        value = max(self._minimum, round(min(self._maximum, max(self._minimum, value)), self._decimals))
        changed = value != self._value
        self._value = value
        self._sync()
        if changed:
            self.valueChanged.emit(value)

    def _selected(self, index):
        if index >= 0:
            self.setValue(self.combo.itemData(index))

    def _commit_text(self):
        text = self.combo.currentText().strip()
        if self._special and text.casefold() == self._special.casefold():
            self.setValue(0.)
        else:
            match = re.fullmatch(r"([+]?(?:\d+(?:[.,]\d*)?|[.,]\d+))\s*(?:mm)?", text, re.IGNORECASE)
            if match:
                value = float(match.group(1).replace(",", "."))
                if self._minimum <= value <= self._maximum:
                    self.setValue(value)
                else:
                    self._sync()
            else:
                self._sync()
        self.editingFinished.emit()

    def _rebuild(self):
        with QSignalBlocker(self.combo):
            self.combo.clear()
            if self._native_pitch is None:
                presets = THICKNESS_PRESETS_MM
            else:
                count = min(1024, int(self._maximum / self._native_pitch + 1e-8))
                presets = tuple(index * self._native_pitch for index in range(1, count + 1))
            for value in dict.fromkeys(presets):
                if self._minimum <= value <= self._maximum:
                    self.combo.addItem(self._text(value), value)
        self._sync()

    def configure_for_spacing(self, spacing):
        values = (spacing,) if isinstance(spacing, (int, float)) else tuple(spacing)
        values = tuple(float(value) for value in values)
        if not values or not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("Thickness presets require positive source spacing")
        pitch = min(values)
        self._native_pitch = pitch
        self._minimum = pitch
        self._maximum = max(self._maximum, pitch)
        self._single_step = pitch
        self._decimals = next((places for places in range(3, 7)
                               if abs(round(pitch, places) - pitch) < 1e-9), 6)
        self._rebuild()
        self.setValue(self._value)
        self.combo.setItemData(0, f"Smallest source voxel pitch: {pitch:.{self._decimals}f} mm",
                               role=Qt.ItemDataRole.ToolTipRole)

    def setRange(self, minimum, maximum):
        minimum, maximum = float(minimum), float(maximum)
        if not all(map(math.isfinite, (minimum, maximum))) or minimum < 0 or maximum < minimum:
            raise ValueError("Invalid thickness range")
        self._minimum = max(minimum, self._native_pitch or .2)
        self._maximum = max(maximum, self._minimum)
        self._rebuild()
        self.setValue(self._value)

    def minimum(self):
        return self._minimum

    def maximum(self):
        return self._maximum

    def setDecimals(self, decimals):
        self._decimals = min(6, max(3, int(decimals)))
        self._rebuild()
        self.setValue(self._value)

    def decimals(self):
        return self._decimals

    def setSuffix(self, suffix):
        self._suffix = str(suffix)
        self._rebuild()

    def setSpecialValueText(self, text):
        # Retain the former numeric-control API without reintroducing a
        # zero-thickness/Native choice into the physical slab dropdown.
        self._special = ""

    def setSingleStep(self, value):
        self._single_step = float(value)

    def singleStep(self):
        return self._single_step

    def setAccessibleName(self, value):
        super().setAccessibleName(value)
        self.combo.setAccessibleName(value)

    def setToolTip(self, value):
        super().setToolTip(value)
        self.combo.setToolTip(value)
