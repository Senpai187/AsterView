"""Comparison controls and linked orthogonal review window."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QWidget, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
                              QPushButton, QComboBox, QCheckBox, QDoubleSpinBox, QSlider, QFileDialog)

from .slice_view import SliceView
from .physical_slider import line_scroll_limits
from .mpr_geometry import plane_frame, plane_references, rotate_about_view, translate_reference


MODES = ("Registered moving", "Blend", "Checkerboard", "Wipe", "Visual difference")


def comparison_image(fixed, moving, valid, *, mode="Blend", alpha=.5, fixed_window=(500, 2000), moving_window=(500, 2000), tile=32):
    """Return windowed display intensities, never quantitative tissue change.

    Only common full-support pixels participate. Different CBCT intensity
    calibration is handled by separate display windows, not invented HU.
    """
    fixed, moving, valid = np.asarray(fixed), np.asarray(moving), np.asarray(valid)
    if fixed.ndim != 2 or moving.shape != fixed.shape or valid.shape != fixed.shape or valid.dtype != np.bool_:
        raise ValueError("Comparison planes and boolean overlap must share a two-dimensional shape.")
    if mode not in MODES or not np.isfinite(alpha) or not 0 <= alpha <= 1 or not isinstance(tile, int) or isinstance(tile, bool) or not 2 <= tile <= 512:
        raise ValueError("Choose a valid comparison mode, blend fraction and checker size.")
    def validate_window(image, setting):
        level, width = [float(x) for x in setting]
        if not np.isfinite([level, width]).all() or width <= 0 or not np.isfinite(image).all(where=valid):
            raise ValueError("Comparison windows and overlapping source values must be finite.")
        return level, width
    def window(image, setting):
        level, width = setting
        return np.clip((image - level) / width + .5, 0, 1)
    fixed_window = validate_window(fixed, fixed_window)
    moving_window = validate_window(moving, moving_window)
    second = window(moving, moving_window)
    if mode == "Registered moving":
        output = second
    else:
        first = window(fixed, fixed_window)
        if mode == "Blend":
            output = (1 - alpha) * first + alpha * second
        elif mode == "Visual difference":
            output = np.abs(first - second)
        else:
            yy, xx = np.ogrid[:fixed.shape[0], :fixed.shape[1]]
            select = ((xx // tile + yy // tile) % 2 == 0) if mode == "Checkerboard" else xx >= round(alpha * fixed.shape[1])
            output = np.where(select, second, first)
    output *= 255
    output[~valid] = 0
    return output.astype(np.float32, copy=False)


class ComparisonPanel(QWidget):
    def __init__(self, controls):
        super().__init__()
        self.controls = controls
        layout = QVBoxLayout(self)
        title = QLabel("Before / after superimposition")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self.info = QLabel("Keep the reference scan open, then load a second DICOM volume.")
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.info)
        self.open_button = QPushButton("Choose comparison DICOM folder…")
        self.open_button.clicked.connect(self.choose_source)
        layout.addWidget(self.open_button)
        self.fit_mode = QComboBox()
        self.fit_mode.addItems(["Manual", "Automatic"])
        self.fit_mode.setToolTip("Choose the initial fitting method before loading the second series.")
        layout.addWidget(self.fit_mode)
        self.series = QComboBox()
        self.series.setToolTip("Explicitly choose the second source series. The reference scan stays open.")
        layout.addWidget(self.series)
        self.load_button = QPushButton("Load selected comparison series")
        self.load_button.clicked.connect(controls.load_selected)
        layout.addWidget(self.load_button)
        self.auto_button = QPushButton("Align scans automatically · rigid 6DOF")
        self.auto_button.setObjectName("primary")
        self.auto_button.clicked.connect(controls.align)
        layout.addWidget(self.auto_button)
        self.bone_focus = QCheckBox("Focus on high-intensity reference structures")
        self.bone_focus.setToolTip("An intensity mask; not a stable-bone segmentation. Exclude changed anatomy by review.")
        layout.addWidget(self.bone_focus)
        self.review_button = QPushButton("Open linked comparison views")
        self.review_button.clicked.connect(controls.open_comparison)
        layout.addWidget(self.review_button)
        self.status = QLabel("No comparison scan loaded.")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        row = QHBoxLayout()
        self.accept_button, self.reject_button = QPushButton("Accept alignment"), QPushButton("Reject alignment")
        self.accept_button.clicked.connect(lambda: controls.review("accepted"))
        self.reject_button.clicked.connect(lambda: controls.review("rejected"))
        row.addWidget(self.accept_button)
        row.addWidget(self.reject_button)
        layout.addLayout(row)
        label = QLabel("Manual correction in reference LPS")
        label.setObjectName("sectionTitle")
        layout.addWidget(label)
        grid = QGridLayout()
        self.manual = []
        for row_index, (name, unit, limit, step) in enumerate((('Left', ' mm', 200, .5), ('Posterior', ' mm', 200, .5), ('Superior', ' mm', 200, .5),
                                                            ('Rotate L axis', '°', 180, .5), ('Rotate P axis', '°', 180, .5), ('Rotate S axis', '°', 180, .5))):
            item = QDoubleSpinBox()
            item.setRange(-limit, limit)
            item.setSingleStep(step)
            item.setDecimals(1)
            item.setSuffix(unit)
            item.setKeyboardTracking(False)
            self.manual.append(item)
            grid.addWidget(QLabel(name), row_index, 0)
            grid.addWidget(item, row_index, 1)
        layout.addLayout(grid)
        self.apply_button = QPushButton("Apply correction around reference center")
        self.apply_button.clicked.connect(controls.apply_manual)
        layout.addWidget(self.apply_button)
        row = QHBoxLayout()
        self.save_button, self.restore_button = QPushButton("Save alignment…"), QPushButton("Open alignment…")
        self.save_button.clicked.connect(controls.save)
        self.restore_button.clicked.connect(controls.restore)
        row.addWidget(self.save_button)
        row.addWidget(self.restore_button)
        layout.addLayout(row)
        note = QLabel("Review all planes before accepting. Visual intensity differences do not establish growth or treatment change.")
        note.setWordWrap(True)
        note.setObjectName("subtitle")
        layout.addWidget(note)
        layout.addStretch()

    def choose_source(self):
        folder = QFileDialog.getExistingDirectory(self.controls.workstation, "Choose the second DICOM study folder")
        if folder:
            self.controls.begin_workflow(folder)

    def refresh(self, busy=False):
        ready = self.controls.moving is not None and self.controls.result is not None
        self.open_button.setEnabled(not busy and self.controls.workstation.volume is not None)
        self.load_button.setEnabled(not busy and self.series.count() > 0)
        for item in (self.auto_button, self.review_button, self.accept_button, self.reject_button, self.apply_button, self.save_button, self.restore_button):
            item.setEnabled(not busy and ready)
        if ready:
            result = self.controls.result
            metric = "" if result.metric is None else f" · MI {result.metric:.4f} (not accuracy)"
            self.status.setText(f"Alignment {result.review}{metric}\n" + "\n".join(result.warnings))


class ComparisonWorkspace(QWidget):
    """Persistent tab host; fitting stays in its own temporary window."""
    def __init__(self, controls):
        super().__init__()
        self.controls = controls
        self.body = QVBoxLayout(self)
        self.empty = QWidget(self)
        layout = QVBoxLayout(self.empty)
        layout.addStretch()
        label = QLabel("Choose a second scan, align it, then accept the fit to compare both volumes here.")
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        button = QPushButton("Choose comparison scan…")
        button.clicked.connect(lambda: controls.begin_workflow())
        layout.addWidget(button)
        layout.addStretch()
        self.body.addWidget(self.empty)

    def install(self, view):
        self.empty.hide()
        self.body.addWidget(view)

    def clear(self):
        self.empty.show()


class ComparisonDialog(QDialog):
    def __init__(self, controls, *, fitting=True, parent=None):
        super().__init__(controls.workstation if parent is None else parent)
        self.controls = controls
        self.fitting = bool(fitting)
        if not fitting:
            self.setWindowFlags(Qt.WindowType.Widget)
        self.setWindowTitle("AsterView · Align scans" if fitting else "AsterView · Superimposition")
        self.resize(1320, 900)
        layout = QVBoxLayout(self)
        self.notice = QLabel("")
        self.notice.setWordWrap(True)
        layout.addWidget(self.notice)
        row = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItems(MODES)
        self.mode.setCurrentText("Blend" if fitting else "Registered moving")
        row.addWidget(self.mode)
        row.addWidget(QLabel("Blend / wipe"))
        self.alpha = QSlider(Qt.Orientation.Horizontal)
        self.alpha.setRange(0, 100)
        self.alpha.setValue(50)
        row.addWidget(self.alpha, 1)
        self.link_main = QCheckBox("Link Explorer")
        self.link_main.setChecked(True)
        row.addWidget(self.link_main)
        self.tool = QComboBox()
        self.tool.addItems(["Navigate", "Correct planes", "Pan", "Measure"])
        row.addWidget(self.tool)
        row.addWidget(QLabel("Thickness"))
        from .thickness_control import ThicknessControl
        self.thickness = ThicknessControl()
        self.thickness.setRange(0, 30)
        self.thickness.configure_for_spacing(controls.workstation.volume.spacing)
        self.thickness.setSingleStep(.2)
        self.thickness.setSuffix(" mm")
        self.thickness.setValue(getattr(controls.workstation, "slab", self.thickness).value())
        row.addWidget(self.thickness)
        self.projection = QComboBox()
        self.projection.addItems(["mean", "max", "min"])
        row.addWidget(self.projection)
        layout.addLayout(row)
        row = QGridLayout()
        self.windows = []
        for source, (title, volume) in enumerate((("Reference", controls.workstation.volume), ("Comparison", controls.moving))):
            row.addWidget(QLabel(title), source, 0)
            for index, value in enumerate(volume.window):
                item = QSlider(Qt.Orientation.Horizontal)
                low, high = volume.data_min, volume.data_max
                item.setRange(int(np.floor(min(low, value))) if index == 0 else 1,
                              int(np.ceil(max(high, value))) if index == 0 else int(max(100, 2 * (high - low), 3 * value)))
                item.setValue(round(value))
                item.setAccessibleName(title + (" brightness" if index == 0 else " contrast"))
                row.addWidget(QLabel("Brightness" if index == 0 else "Contrast"), source, 1 + index * 3)
                row.addWidget(item, source, 2 + index * 3)
                label = QLabel(str(item.value()))
                label.setMinimumWidth(48)
                item.valueChanged.connect(lambda number, text=label: text.setText(str(number)))
                row.addWidget(label, source, 3 + index * 3)
                self.windows.append(item)
        layout.addLayout(row)
        self.fitting_controls = QWidget(self)
        adjustment = QHBoxLayout(self.fitting_controls)
        adjustment.setContentsMargins(0, 0, 0, 0)
        adjustment.addWidget(QLabel("Fit comparison"))
        self.auto_fit = QPushButton("Fit automatically")
        self.auto_fit.clicked.connect(controls.align)
        adjustment.addWidget(self.auto_fit)
        self.fit_values = []
        for title, source in zip(("L", "P", "S", "L°", "P°", "S°"), controls.panel.manual):
            adjustment.addWidget(QLabel(title))
            item = QDoubleSpinBox()
            item.setRange(source.minimum(), source.maximum())
            item.setDecimals(source.decimals())
            item.setSuffix(source.suffix())
            item.setSingleStep(source.singleStep())
            item.setMaximumWidth(92)
            self.fit_values.append(item)
            item.valueChanged.connect(source.setValue)
            source.valueChanged.connect(item.setValue)
            adjustment.addWidget(item)
        self.apply = QPushButton("Apply fit")
        self.apply.clicked.connect(controls.apply_manual)
        adjustment.addWidget(self.apply)
        self.accept = QPushButton("Accept alignment")
        self.accept.clicked.connect(lambda: controls.review("accepted"))
        adjustment.addWidget(self.accept)
        layout.addWidget(self.fitting_controls)
        self.fitting_controls.setVisible(fitting)
        self.refine_button = QPushButton("Adjust alignment…")
        self.refine_button.clicked.connect(controls.open_alignment)
        self.refine_button.setVisible(not fitting)
        layout.addWidget(self.refine_button)
        grid = QGridLayout()
        self.views = []
        for col, plane in enumerate(("Axial", "Coronal", "Sagittal")):
            top, bottom = SliceView("Reference · " + plane), SliceView("Registered · " + plane, "#e6b77a")
            for view in (top, bottom):
                view.picked.connect(self.pick)
                view.scrolled.connect(lambda delta, p=plane: self.scroll(p, delta))
                view.position_changed.connect(lambda value, p=plane: self.set_plane_position(p, value))
                view.reference_translated.connect(lambda payload, p=plane: self.translate(p, payload))
                view.reference_rotated.connect(lambda angle, p=plane: self.rotate(p, angle))
                view.section_defaults = lambda: (self.thickness.value(), self.projection.currentText())
                view.section_follow_label = "Follow comparison thickness"
                view.section_settings_changed.connect(lambda v=view, p=plane: self.section_changed(p, v))
                view.snapshot_requested.connect(lambda v=view: self.snapshot(v))
                view.maximize_requested.connect(lambda v=view: self.maximize(v))
                view.sampling_changed.connect(lambda: self.resize_timer.start() if hasattr(self, "resize_timer") else None)
                view.max_button.show()
                grid.addWidget(view, 0 if view is top else 1, col)
            self.views.append((plane, top, bottom))
        layout.addLayout(grid, 1)
        self.center = np.asarray(controls.workstation.crosshair, float).copy()
        self.rotation = self.main_rotation()
        self.displayed_rotation = self.rotation.copy()
        self.cached_planes = None
        self._maximized = None
        self.mode.currentTextChanged.connect(self.display_cached)
        self.alpha.valueChanged.connect(self.display_cached)
        for item in self.windows:
            item.valueChanged.connect(self.display_cached)
        self.link_main.toggled.connect(self.link_changed)
        self.tool.currentTextChanged.connect(self.set_tool)
        self.thickness.valueChanged.connect(lambda _: controls.request_planes())
        self.projection.currentTextChanged.connect(lambda _: controls.request_planes())
        self.sync_timer = QTimer(self)
        self.sync_timer.setInterval(33)
        self.sync_timer.timeout.connect(self.sync_position)
        self.sync_timer.start()
        self.resize_timer = QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.setInterval(40)
        self.resize_timer.timeout.connect(controls.request_planes)

    def sync_position(self):
        if self.link_main.isChecked():
            value = np.asarray(self.controls.workstation.crosshair, float)
            rotation = self.main_rotation()
            if not np.allclose(value, self.center) or not np.allclose(rotation, self.rotation):
                self.center = value.copy()
                self.rotation = rotation
                self.refresh_references()
                self.controls.request_planes()

    def main_rotation(self):
        getter = getattr(self.controls.workstation, "rotation", None)
        return np.asarray(getter() if callable(getter) else np.eye(3), float).copy()

    def set_fitting_busy(self, busy):
        for item in [self.apply, self.accept, self.auto_fit, self.refine_button, *self.fit_values]:
            item.setEnabled(not busy)

    def link_changed(self, linked):
        if linked:
            self.center = np.asarray(self.controls.workstation.crosshair, float).copy()
            self.rotation = self.main_rotation()
        self.refresh_references()
        self.controls.request_planes()

    def set_tool(self, tool):
        for _, first, second in self.views:
            first.mode = second.mode = tool

    def section_changed(self, plane, source):
        # Compare identical physical slabs. One pane's thickness applies to
        # its partner; it must not silently compare different slab geometries.
        for name, first, second in self.views:
            if name == plane:
                partner = second if source is first else first
                partner.slab_mm, partner.slab_mode = source.slab_mm, source.slab_mode
        self.controls.request_planes()

    def snapshot(self, view):
        path, _ = QFileDialog.getSaveFileName(self, "Save comparison section", "AsterView-comparison.png", "PNG (*.png)")
        if path:
            view.grab().save(path)

    def maximize(self, view):
        self._maximized = None if self._maximized is view else view
        for _, first, second in self.views:
            for item in (first, second):
                item.setVisible(self._maximized is None or item is self._maximized)
        self.resize_timer.start()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "resize_timer"):
            self.resize_timer.start()

    def refresh_references(self):
        for plane, first, second in self.views:
            right, up = plane_frame(plane, self.displayed_rotation)
            normal = np.cross(right, up)
            volume = self.controls.workstation.volume
            limits = line_scroll_limits(volume, self.center, normal)
            position = float(self.center @ normal)
            for view in (first, second):
                view.crosshair = self.center.copy()
                view.reference_lines = plane_references(plane, self.displayed_rotation)
                view.set_scroll_position(*limits, position, min(volume.spacing))
                view.update()

    def set_plane_position(self, plane, position):
        if not np.isfinite(position):
            return
        # The slider follows the currently displayed axes, just like the
        # visible reference lines, even while a corrected frame is in flight.
        right, up = plane_frame(plane, self.displayed_rotation)
        normal = np.cross(right, up)
        self._publish_position(self.center + normal * (float(position) - float(self.center @ normal)))

    def _constrain(self, point):
        volume = self.controls.workstation.volume
        start = volume.world_to_index(self.center)
        end = volume.world_to_index(point)
        movement = end - start
        fraction = 1.
        for position, delta, count in zip(start, movement, volume.dimensions):
            if delta > 0:
                fraction = min(fraction, (count - 1 - position) / delta)
            elif delta < 0:
                fraction = min(fraction, -position / delta)
        return self.center + (point - self.center) * max(0., fraction)

    def _publish_position(self, point):
        self.center = self._constrain(np.asarray(point, float))
        if self.link_main.isChecked():
            workstation = self.controls.workstation
            workstation.crosshair = self.center.copy()
            for name in ("refresh_references", "schedule_render"):
                callback = getattr(workstation, name, None)
                if callable(callback):
                    callback()
        self.refresh_references()
        self.controls.request_planes()

    def pick(self, point):
        # SliceView.picked carries navigation metadata, not a bare vector.
        if isinstance(point, dict):
            point = point.get("world")
        if point is None:
            return
        point = np.asarray(point, float)
        if point.shape != (3,) or not np.isfinite(point).all():
            return
        self._publish_position(point)

    def scroll(self, plane, delta):
        if not np.isfinite(delta):
            return
        right, up = plane_frame(plane, self.rotation)
        self._publish_position(self.center + np.cross(right, up) * float(delta) * min(self.controls.workstation.volume.spacing))

    def translate(self, plane, payload):
        point = translate_reference(self.center, payload["delta"], plane, payload["target"], self.rotation)
        self._publish_position(point)

    def rotate(self, plane, angle):
        workstation = self.controls.workstation
        if self.link_main.isChecked() and callable(getattr(workstation, "rotate_planes", None)):
            workstation.rotate_planes(plane, angle)
            self.rotation = self.main_rotation()
        else:
            self.rotation = rotate_about_view(self.rotation, plane, angle)
        self.refresh_references()
        self.controls.request_planes()

    def display_cached(self, *_):
        if self.cached_planes is None:
            return
        first_window = tuple(item.value() for item in self.windows[:2])
        second_window = tuple(item.value() for item in self.windows[2:])
        mode, alpha = self.mode.currentText(), self.alpha.value() / 100
        for (_, first, second), (frame, fixed, moving, valid) in zip(self.views, self.cached_planes):
            geometry = tuple(np.round(np.concatenate((*frame[:3], np.asarray(frame[3]))), 9))
            first_key = (geometry, first_window)
            first_state = getattr(first, "_comparison_pixels", None)
            if first_state is None or first_state[0] is not fixed or first_state[1] != first_key:
                first.center, first.window_width = first_window
                first.set_image(fixed, frame, self.center, "Reference source · patient LPS")
                first._comparison_pixels = (fixed, first_key)
            # Orthogonal scrolling changes only one plane pair. Retain the
            # other two displayed images; their crosshairs still refresh below.
            # Strong references prevent stale identity reuse after cache eviction.
            second_key = (geometry, mode, second_window,
                          first_window if mode != "Registered moving" else None,
                          alpha if mode in ("Blend", "Wipe") else None)
            second_state = getattr(second, "_comparison_pixels", None)
            same_pixels = (second_state is not None and second_state[0] is fixed
                           and second_state[1] is moving and second_state[2] is valid
                           and second_state[3] == second_key)
            if not same_pixels:
                image = comparison_image(fixed, moving, valid, mode=mode, alpha=alpha,
                                         fixed_window=first_window, moving_window=second_window)
                second.center, second.window_width = 127.5, 255
                second.set_image(image, frame, self.center, f"{mode} · common overlap {valid.mean():.0%}")
                second._comparison_pixels = (fixed, moving, valid, second_key)
        result = self.controls.result
        self.refresh_references()
        self.notice.setText(f"Alignment {result.review} · Top: reference. Bottom: registered comparison in the same physical planes. "
                            "Scroll, move the lines or correct a plane in either row to update both scans. Visual difference is not a calibrated change measurement.")

    def closeEvent(self, event):
        self.sync_timer.stop()
        self.resize_timer.stop()
        self.controls.cancel_planes()
        super().closeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, "sync_timer"):
            self.sync_timer.start()
            self.resize_timer.start()

    def hideEvent(self, event):
        if hasattr(self, "sync_timer"):
            self.sync_timer.stop()
            self.resize_timer.stop()
            if self.controls.dialog is self:
                self.controls.cancel_planes()
        super().hideEvent(event)
