"""A scan-backed physical-range editor with a lightweight sagittal preview."""
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                               QDoubleSpinBox, QPushButton, QLabel, QDialogButtonBox)

from .panoramic_range import range_presets
from .slice_view import SliceView


class PanoramicRangeDialog(QDialog):
    def __init__(self, viewer):
        super().__init__(viewer)
        self.viewer = viewer
        self.setWindowTitle("Panoramic data range")
        self.setMinimumSize(500, 520)
        layout = QVBoxLayout(self)
        self.origin, self.up, self.limits = viewer.panorama_range_geometry()
        self.step = float(min(viewer.volume.spacing))
        self.preview = SliceView("Drag upper and lower boundaries", "#d8b78a")
        self.preview.setMinimumHeight(280)
        self.preview.settings_button.hide()
        self.preview.snapshot_button.hide()
        source = viewer.pano_sagittal_view
        self.preview.set_window(source.center, source.window_width, source.invert)
        if source.data is not None:
            self.preview.set_image(source.data, source.frame)
        layout.addWidget(self.preview, 1)
        presets = QHBoxLayout()
        self.presets = range_presets(viewer.volume, viewer.arch_points, self.up)
        for name, bounds in self.presets.items():
            button = QPushButton(name)
            button.clicked.connect(lambda checked=False, values=bounds: self.set_bounds(values))
            presets.addWidget(button)
        layout.addLayout(presets)
        self.lower, self.upper = QDoubleSpinBox(), QDoubleSpinBox()
        for control, name in ((self.lower, "panoRangeLower"), (self.upper, "panoRangeUpper")):
            control.setObjectName(name)
            control.setDecimals(3)
            control.setRange(*self.limits)
            control.setSuffix(" mm")
            control.setSingleStep(self.step)
            control.setKeyboardTracking(False)
            control.valueChanged.connect(self.update_preview)
        form = QFormLayout()
        form.addRow("Lower bound", self.lower)
        form.addRow("Upper bound", self.upper)
        layout.addLayout(form)
        self.message = QLabel("Offsets follow the arch plane. Presets describe acquired positions.")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply range")
        self.buttons.accepted.connect(self.apply_range)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.preview.range_bounds_changed.connect(lambda lower, upper: self.set_bounds((lower, upper)))
        self.set_bounds(viewer.current_panorama_bounds())

    def set_bounds(self, values):
        with QSignalBlocker(self.lower), QSignalBlocker(self.upper):
            self.lower.setValue(values[0])
            self.upper.setValue(values[1])
        self.update_preview()

    def selected_bounds(self):
        # Spin boxes round their limits for display. Keep the actual geometry
        # inside the source when an oblique acquisition boundary is rounded out.
        return (max(self.limits[0], self.lower.value()),
                min(self.limits[1], self.upper.value()))

    def update_preview(self, *_):
        bounds = self.selected_bounds()
        valid = bounds[1] - bounds[0] >= self.step - 1e-6
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(valid)
        if valid:
            self.preview.set_data_range(self.origin, self.up, bounds, self.limits, self.step)
            self.message.setText(f"Included height: {bounds[1] - bounds[0]:.2f} mm · offsets from the arch plane")
        else:
            self.message.setText(f"Keep the upper bound at least {self.step:.3f} mm above the lower bound.")

    def apply_range(self):
        try:
            self.viewer.apply_panorama_range(*self.selected_bounds())
        except ValueError as error:
            self.message.setText(str(error))
            return
        self.accept()
