"""Qt image viewport with physical-coordinate navigation and editable overlays."""
from __future__ import annotations

import numpy as np
import qtawesome as qta
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QFont
from PySide6.QtWidgets import (QWidget, QToolButton, QMenu, QDialog, QVBoxLayout,
                               QFormLayout, QCheckBox, QDoubleSpinBox, QComboBox,
                               QDialogButtonBox, QLabel)


def direction_label(vector):
    names = (("R", "L"), ("A", "P"), ("I", "S"))
    return "".join(names[i][int(vector[i] > 0)] for i in np.argsort(-np.abs(vector))
                   if abs(vector[i]) > 0.2)


class SliceView(QWidget):
    picked = Signal(object)
    scrolled = Signal(float)
    window_dragged = Signal(float, float)
    measured = Signal(str)
    reference_translated = Signal(object)
    reference_rotated = Signal(float)
    interaction_started = Signal()
    interaction_finished = Signal()
    point_moved = Signal(int, object)
    point_removed = Signal(int)
    path_finished = Signal()
    mask_stroke = Signal(object)
    maximize_requested = Signal()
    path_node_dragged = Signal(object)
    path_node_removed = Signal(object)
    implant_dragged = Signal(object)
    snapshot_requested = Signal()
    section_settings_changed = Signal()
    grid_settings_requested = Signal()
    sampling_changed = Signal()
    position_changed = Signal(float)
    zoom_changed = Signal(float)
    range_bounds_changed = Signal(float, float)

    def __init__(self, title, accent="#42b8c4", parent=None):
        super().__init__(parent)
        self.title, self.accent = title, accent
        self.setMinimumSize(220, 180)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.max_button = QToolButton(self)
        self.max_button.setIcon(qta.icon("fa6s.expand", color="#bdc1cb"))
        self.max_button.setAccessibleName("Maximize or restore section")
        self.max_button.setToolTip("Maximize / restore this section")
        self.max_button.setStyleSheet("QToolButton { background:transparent; padding:2px; border:0; } QToolButton:hover { background:#363a44; }")
        self.max_button.clicked.connect(self.maximize_requested)
        self.max_button.hide()
        self.snapshot_button = QToolButton(self)
        self.snapshot_button.setIcon(qta.icon("fa6s.camera", color="#bdc1cb"))
        self.snapshot_button.setAccessibleName("Save section quick shot")
        self.snapshot_button.setToolTip("Save this section with overlays")
        self.snapshot_button.setStyleSheet(self.max_button.styleSheet())
        self.snapshot_button.clicked.connect(self.snapshot_requested)
        self.slab_mm = None
        self.slab_mode = None
        self.section_defaults = lambda: (0., "mean")
        self.section_grid_enabled = False
        self._settings_dialog = None
        self.settings_button = QToolButton(self)
        self.settings_button.setIcon(qta.icon("fa6s.wrench", color="#bdc1cb"))
        self.settings_button.setAccessibleName("Section controls")
        self.settings_button.setToolTip("Section thickness, projection and layout")
        self.settings_button.setStyleSheet(self.max_button.styleSheet())
        self.settings_menu = QMenu(self.settings_button)
        self.settings_menu.addAction("Section thickness and projection…", self.open_section_settings)
        self.grid_action = self.settings_menu.addAction("Section grid…", self.grid_settings_requested.emit)
        self.grid_action.setVisible(False)
        self.settings_menu.addSeparator()
        self.settings_menu.addAction("Reset pan and zoom", self.reset_view)
        self.settings_menu.addAction("Maximize / restore section", self.maximize_requested.emit)
        self.settings_menu.addAction("Save section quick shot…", self.snapshot_requested.emit)
        self.settings_button.setMenu(self.settings_menu)
        self.settings_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._compact_header = False
        self._restore_max_button = False
        self.grid_cell_index = None
        self.grid_caption = ""
        self.data = self.gray = self.image = None
        self.frame = None
        self.physical_size_mm = None
        self.calibrated = True
        self.crosshair = None
        self.points = []
        self.path_points = []
        self.nerve_paths = []
        self.selected_nerve_path = self.selected_nerve_node = None
        self.path_edit_enabled = False
        self.path_slice_half_mm = .5
        self._path_drag = None
        self.landmarks = []
        self.selected_point = None
        self.segmentation_image = self._segmentation_rgba = None
        self.tooth_annotations = []
        self.segmentation_preview = False
        self.brush_radius_mm = 2.
        self.brush_color = "#e5ba70"
        self._brush_hover = None
        self.implants = []
        self.selected_implant_id = None
        self.clearance_measurement = None
        self._implant_drag = None
        self.source_text = ""
        self.measurements = []
        self.pending_measure = None
        self.mode = "Navigate"
        self.center, self.window_width, self.invert = 500.0, 2000.0, False
        self.zoom, self.pan = 1.0, QPointF(0, 0)
        self.last_pos = None
        self.note = ""
        self.reference_lines = []
        self.show_references = True
        self._drag_kind = None
        self._drag_target = None
        self._drag_world = None
        self._window_dirty = False
        self._window_buffer = None
        self.image_key = None
        self.fraction_navigation_enabled = False
        self.cursor_fraction = None
        self._fraction_drag = None
        from .physical_slider import PhysicalSlider
        self.position_slider = PhysicalSlider(parent=self)
        self.position_slider.hide()
        self.position_slider.position_changed.connect(self.position_changed)
        self.position_slider.sliderPressed.connect(self.interaction_started)
        self.position_slider.sliderReleased.connect(self.interaction_finished)
        self._position_visible = False
        self.section_markers = []
        self.section_columns = []
        self.data_range = None

    def set_scroll_position(self, minimum, maximum, value, step=.1):
        self.position_slider.set_position(minimum, maximum, value, step)
        self._position_visible = True
        self.position_slider.show()
        self._position_header_controls()

    def set_data_range(self, origin, axis, bounds, limits=None, minimum_step=.1):
        from .panoramic_range import validate_bounds
        bounds = validate_bounds(bounds)
        limits = bounds if limits is None else validate_bounds(limits)
        if not np.isfinite(minimum_step) or minimum_step <= 0 or bounds[0] < limits[0] or bounds[1] > limits[1]:
            raise ValueError("Panoramic bounds must lie in the source range with a positive minimum step")
        origin, axis = np.asarray(origin, float), np.asarray(axis, float)
        if origin.shape != (3,) or axis.shape != (3,) or not np.isfinite((origin, axis)).all() or np.linalg.norm(axis) < 1e-8:
            raise ValueError("Panoramic range needs a finite origin and vertical axis")
        self.data_range = {"origin": origin.copy(), "axis": axis / np.linalg.norm(axis),
                           "bounds": bounds, "limits": limits, "step": float(minimum_step)}
        self.update()

    def _data_range_lines(self):
        if self.data_range is None or self.frame is None or self.data is None:
            return []
        center, right, up, _ = self.frame
        setting = self.data_range
        axis = setting["axis"]
        a, b = float(axis @ right), float(axis @ up)
        if a * a + b * b < 1e-8:
            return []
        direction = np.cross(np.cross(right, up), axis)
        direction /= np.linalg.norm(direction)
        result = []
        for index, value in enumerate(setting["bounds"]):
            point = center + (axis @ setting["origin"] + value - axis @ center) * (a * right + b * up) / (a * a + b * b)
            q = self.world_to_screen(point)
            vector = self.world_to_screen(point + direction) - q
            vector /= max(1e-8, np.hypot(vector.x(), vector.y()))
            result.append((index, value, q, vector))
        return result

    def capture_content_image(self, include_annotations=False):
        """Export currently displayed scan pixels without widget controls."""
        if self.image is None:
            return QImage()
        canvas = self.image_canvas_rect().toAlignedRect().intersected(self.rect())
        if include_annotations:
            return self.grab(canvas).toImage()
        ratio = self.devicePixelRatioF()
        output = QImage(max(1, round(canvas.width() * ratio)),
                        max(1, round(canvas.height() * ratio)), QImage.Format.Format_ARGB32)
        output.setDevicePixelRatio(ratio)
        output.fill(QColor("#08090b"))
        painter = QPainter(output)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        target = self._rect().translated(-canvas.left(), -canvas.top())
        painter.drawImage(target, self.image)
        painter.end()
        return output

    def enable_section_grid(self, enabled=True):
        self.section_grid_enabled = bool(enabled)
        self.grid_action.setVisible(self.section_grid_enabled)

    def set_section_settings(self, slab_mm=None, mode=None):
        """Set this pane's physical slab; None follows the workstation setting."""
        if slab_mm is not None and (not np.isfinite(slab_mm) or not 0 <= slab_mm <= 30):
            raise ValueError("Section thickness must be between 0 and 30 mm")
        if mode not in (None, "mean", "max", "min"):
            raise ValueError("Unknown section projection")
        slab_mm = None if slab_mm is None else float(slab_mm)
        if self.slab_mm != slab_mm or self.slab_mode != mode:
            self.slab_mm, self.slab_mode = slab_mm, mode
            self.image_key = None
            self.section_settings_changed.emit()

    def open_section_settings(self):
        if self._settings_dialog is not None:
            self._settings_dialog.raise_()
            self._settings_dialog.activateWindow()
            return self._settings_dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(self.title + " · Section controls")
        dialog.setMinimumWidth(350)
        layout = QVBoxLayout(dialog)
        note = QLabel("Thickness and projection apply to this pane. Navigation remains linked to the source volume.")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        inherited = QCheckBox(getattr(self, "section_follow_label", "Follow Image inspector settings"))
        inherited.setChecked(self.slab_mm is None and self.slab_mode is None)
        from .thickness_control import ThicknessControl
        thickness = ThicknessControl()
        if getattr(self, "thickness_presets_spacing", None) is not None:
            thickness.configure_for_spacing(self.thickness_presets_spacing)
        thickness.setObjectName("sectionThickness")
        thickness.setRange(0, 30)
        thickness.setDecimals(3)
        thickness.setSingleStep(.2)
        thickness.setSuffix(" mm")
        default_mm, default_mode = self.section_defaults()
        thickness.setValue(default_mm if self.slab_mm is None else self.slab_mm)
        projection = QComboBox()
        projection.setObjectName("sectionProjection")
        projection.addItems(["mean", "max", "min"])
        projection.setCurrentText(self.slab_mode or default_mode)
        for item in (thickness, projection):
            item.setEnabled(not inherited.isChecked())
            inherited.toggled.connect(lambda checked, control=item: control.setEnabled(not checked))
        form.addRow(inherited)
        form.addRow("Slab thickness", thickness)
        form.addRow("Projection", projection)
        layout.addLayout(form)
        hint = QLabel("The lowest thickness follows the source voxel pitch. A thicker slab combines samples along its normal; cursor and measurements remain on the center plane.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
            lambda: self.set_section_settings(None if inherited.isChecked() else thickness.value(),
                                              None if inherited.isChecked() else projection.currentText()))
        buttons.rejected.connect(dialog.close)
        layout.addWidget(buttons)
        self._settings_dialog = dialog
        dialog.finished.connect(lambda *_: setattr(self, "_settings_dialog", None))
        dialog.open()
        return dialog

    def set_image(self, array, frame=None, crosshair=None, note="", physical_size_mm=None):
        if self.frame is not None and frame is not None:
            old_center, old_right, old_up, _ = self.frame
            center, right, up, _ = frame
            old_normal = np.cross(old_right, old_up)
            if not np.allclose(old_right, right) or not np.allclose(old_up, up) or abs(np.dot(np.asarray(center) - old_center, old_normal)) > 1e-5:
                self.pending_measure = None
        self.data, self.frame, self.crosshair, self.note = array, frame, crosshair, note
        self.physical_size_mm = physical_size_mm
        if self.grid_cell_index is not None:
            self.setToolTip(f"{self.title}\n{note}\nScroll to move the section range · Wrench for thickness, maximize and quick shot")
        self._window_image()

    def set_window(self, center, width, invert=False):
        self.center, self.window_width, self.invert = center, max(float(width), 1e-6), invert
        if self.isVisible():
            self._window_image()
        else:
            self._window_dirty = True

    def set_segmentation(self, plane, labels=None, visible_ids=None, opacity=.35, *, preview=False):
        self.segmentation_preview = preview
        self.tooth_annotations = []
        if plane is None:
            self.segmentation_image = self._segmentation_rgba = None
        else:
            lut = np.zeros((65536, 4), dtype=np.uint8)
            for key, label in (labels or {}).items():
                if key not in (visible_ids or set()) or label["review"] == "rejected":
                    continue
                color = QColor(label.get("color", "#d8b78a"))
                lut[key] = [color.red(), color.green(), color.blue(), round(255 * opacity)]
            self._segmentation_rgba = np.ascontiguousarray(lut[plane])
            rows, columns = plane.shape
            self.segmentation_image = QImage(self._segmentation_rgba.data, columns, rows, columns * 4, QImage.Format.Format_RGBA8888)
            for key, label in (labels or {}).items():
                if "fdi" in label and key in (visible_ids or set()) and label["review"] != "rejected":
                    yy, xx = np.nonzero(plane == key)
                    if len(xx) >= 3:
                        self.tooth_annotations.append(((float(xx.mean()) + .5) / columns, (float(yy.mean()) + .5) / rows, str(label["fdi"])))
        self.update()

    def set_nerve_paths(self, paths, selected_path_id=None, selected_node=None, edit_enabled=False):
        self.nerve_paths = paths
        self.selected_nerve_path, self.selected_nerve_node = selected_path_id, selected_node
        self.path_edit_enabled = bool(edit_enabled)
        self.update()

    def _nerve_nodes(self):
        if self.frame is None:
            return []
        center, right, up, _ = self.frame
        normal = np.cross(right, up)
        result = []
        for path in self.nerve_paths:
            if path["id"] != self.selected_nerve_path or path.get("review") == "rejected" or not path.get("visible", True):
                continue
            tolerance = max(self.path_slice_half_mm, path["diameter_mm"] / 2)
            for index, position in enumerate(path["points_lps_mm"]):
                position = np.asarray(position)
                distance = float(np.dot(position - center, normal))
                result.append((path, index, position, distance, abs(distance) <= tolerance))
        return result

    def nerve_node_hit(self, point):
        if not self.path_edit_enabled or not self._rect().contains(point):
            return None
        hits = []
        for path, index, position, _, near in self._nerve_nodes():
            screen = self.world_to_screen(position)
            distance = np.hypot((screen - point).x(), (screen - point).y())
            if near and distance <= 11:
                hits.append((distance, path["id"], index, position))
        return min(hits, key=lambda hit: hit[0])[1:] if hits else None

    def _draw_nerve_paths(self, painter):
        center, right, up, _ = self.frame
        normal = np.cross(right, up)
        for path in self.nerve_paths:
            if path.get("review") == "rejected" or not path.get("visible", True):
                continue
            points = np.asarray(path.get("sampled_points_lps_mm", path["points_lps_mm"]))
            selected = path["id"] == self.selected_nerve_path
            color = QColor(path["color"])
            tolerance = max(self.path_slice_half_mm, path["diameter_mm"] / 2)
            depths = (points - center) @ normal
            # A faint dashed projection provides context; solid portions alone
            # lie within the center section's physical path neighborhood.
            if selected:
                color.setAlpha(65)
                painter.setPen(QPen(color, 1, Qt.PenStyle.DashLine))
                for a, b in zip(points, points[1:]):
                    painter.drawLine(self.world_to_screen(a), self.world_to_screen(b))
            color.setAlpha(240)
            painter.setPen(QPen(color, 2 if selected else 1.5))
            for a, b, da, db in zip(points, points[1:], depths, depths[1:]):
                lo, hi = 0., 1.
                if abs(db - da) < 1e-10:
                    if abs(da) > tolerance:
                        continue
                else:
                    limits = sorted(((-tolerance - da) / (db - da), (tolerance - da) / (db - da)))
                    lo, hi = max(lo, limits[0]), min(hi, limits[1])
                    if lo > hi:
                        continue
                painter.drawLine(self.world_to_screen(a + lo * (b - a)), self.world_to_screen(a + hi * (b - a)))
        for path, index, position, _, near in self._nerve_nodes():
            if not self.path_edit_enabled and index != self.selected_nerve_node:
                continue
            point = self.world_to_screen(position)
            color = QColor("#ffe0a3" if index == self.selected_nerve_node else path["color"])
            color.setAlpha(255 if near else 70)
            painter.setPen(QPen(color, 1.7, Qt.PenStyle.SolidLine if near else Qt.PenStyle.DashLine))
            painter.setBrush(QColor("#17191d") if near else Qt.BrushStyle.NoBrush)
            painter.drawEllipse(point, 5 if index == self.selected_nerve_node else 3.5, 5 if index == self.selected_nerve_node else 3.5)
            if near and (self.path_edit_enabled or index == self.selected_nerve_node):
                painter.drawText(point + QPointF(7, -6), f"N{index + 1}")
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _window_image(self):
        self._window_dirty = False
        if self.data is not None:
            # Contrast sliders and linked navigation frequently recolor the
            # same-size pane. Reuse two plane-sized allocations; the source
            # pixels remain immutable, and QImage still borrows owned gray.
            if self._window_buffer is None or self._window_buffer.shape != self.data.shape:
                self._window_buffer = np.empty(self.data.shape, dtype=np.float32)
                self.gray = np.empty(self.data.shape, dtype=np.uint8)
            np.subtract(self.data, self.center - self.window_width / 2, out=self._window_buffer)
            np.multiply(self._window_buffer, 255 / self.window_width, out=self._window_buffer)
            np.clip(self._window_buffer, 0, 255, out=self._window_buffer)
            np.copyto(self.gray, self._window_buffer, casting="unsafe")
            if self.invert:
                np.subtract(255, self.gray, out=self.gray)
            h, w = self.gray.shape
            self.image = QImage(self.gray.data, w, h, w, QImage.Format.Format_Grayscale8)
        self.update()

    def showEvent(self, event):
        if self._window_dirty:
            self._window_image()
        super().showEvent(event)

    def reset_view(self):
        self.zoom, self.pan = 1.0, QPointF(0, 0)
        self.update()
        self.sampling_changed.emit()

    def preferred_sample_shape(self, size_mm, *, quality=1., maximum=2048, pixel_budget=4_194_304):
        from .sampling_quality import display_sample_shape

        area = self.image_area_rect()
        return display_sample_shape(size_mm, (area.width(), area.height()),
                                    device_ratio=self.devicePixelRatioF(), zoom=self.zoom,
                                    quality=quality, maximum=maximum, pixel_budget=pixel_budget)

    def set_grid_cell(self, index):
        """Give a serial cell a compact identity while keeping full-size detail."""
        self.grid_cell_index = int(index)
        self.setMinimumSize(64, 64)
        self._position_header_controls()
        self.update()

    def compact_grid(self):
        return self.grid_cell_index is not None and (self.width_pixels() < 220 or self.height() < 210)

    def image_area_rect(self):
        gutter = 15 if getattr(self, "_position_visible", False) else 0
        if self.compact_grid():
            return QRectF(2, 22, max(1, self.width_pixels() - 4 - gutter), max(1, self.height() - 24))
        return QRectF(8, 35, max(1, self.width_pixels() - 16 - gutter), max(1, self.height() - 67))

    def image_canvas_rect(self):
        gutter = 15 if getattr(self, "_position_visible", False) else 0
        if self.compact_grid():
            return QRectF(1, 22, max(1, self.width_pixels() - 2 - gutter), max(1, self.height() - 23))
        return QRectF(1, 32, max(1, self.width_pixels() - 2 - gutter), max(1, self.height() - 58))

    def _rect(self):
        area = self.image_area_rect()
        if self.image is None:
            return area
        if self.frame is not None:
            size = self.edge_size_mm()
            aspect = size[0] / size[1]
        elif self.physical_size_mm is not None:
            aspect = self.physical_size_mm[0] / self.physical_size_mm[1]
        else:
            aspect = self.image.width() / self.image.height()
        w, h = area.width(), area.width() / aspect
        if h > area.height():
            h, w = area.height(), area.height() * aspect
        w, h = w * self.zoom, h * self.zoom
        return QRectF(area.center().x() - w / 2 + self.pan.x(),
                      area.center().y() - h / 2 + self.pan.y(), w, h)

    def width_pixels(self):
        return QWidget.width(self)

    def edge_size_mm(self):
        """QImage occupies pixel edges; sampler FOV spans pixel centers."""
        size = self.frame[3]
        rows, cols = self.data.shape
        return (size[0] * cols / (cols - 1) if cols > 1 else size[0],
                size[1] * rows / (rows - 1) if rows > 1 else size[1])

    def world_to_screen(self, world):
        if self.frame is None or self.data is None:
            return None
        center, right, up, size = self.frame
        size = self.edge_size_mm()
        delta = np.asarray(world) - center
        rect = self._rect()
        return QPointF(rect.center().x() + np.dot(delta, right) / size[0] * rect.width(),
                       rect.center().y() - np.dot(delta, up) / size[1] * rect.height())

    def screen_to_world(self, point, *, allow_outside=False):
        rect = self._rect()
        if (self.frame is None or self.data is None or rect.width() <= 0 or rect.height() <= 0
                or (not allow_outside and not rect.contains(point))):
            return None
        center, right, up, size = self.frame
        size = self.edge_size_mm()
        return center + right * ((point.x() - rect.center().x()) / rect.width() * size[0]) - up * ((point.y() - rect.center().y()) / rect.height() * size[1])

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#08090b"))
        compact = self.compact_grid()
        header_height = 22 if compact else 32
        p.fillRect(QRectF(0, 0, self.width_pixels(), header_height), QColor("#202329"))
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QColor(self.accent))
        p.setFont(QFont("Segoe UI Variable", 8 if compact else 10, QFont.Weight.DemiBold))
        wide = self.width_pixels() >= 400 and not compact
        title_width = (self.width_pixels() - 31 if compact else
                       self.width_pixels() * .55 - 15 if wide else self.width_pixels() - (43 if self.width_pixels() < 180 else 103))
        title = ((self.grid_caption or str(self.grid_cell_index + 1))
                 if compact else self.title)
        p.drawText(QRectF(6 if compact else 12, 0 if compact else 4,
                         max(20, title_width), 21 if compact else 25), Qt.AlignmentFlag.AlignVCenter,
                   p.fontMetrics().elidedText(title, Qt.TextElideMode.ElideRight, max(20, int(title_width))))
        p.setPen(QColor("#8f97a7"))
        p.setFont(QFont("Segoe UI Variable", 8))
        if wide:
            mode_width = self.width_pixels() * .45 - 101
            p.drawText(QRectF(self.width_pixels() * .55, 5, mode_width, 25), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(self.mode, Qt.TextElideMode.ElideRight, int(mode_width)))
        if self.image is None:
            p.setPen(QColor("#71869d"))
            p.drawText(self.image_canvas_rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                       self.note or "Load a DICOM series or the synthetic demo")
            return
        rect = self._rect()
        p.setClipRect(self.image_canvas_rect())
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(rect, self.image)
        if self.frame is None:
            p.setPen(QPen(QColor("#e5858b"), 1))
            for fraction, label in self.section_columns:
                x = rect.left() + float(fraction) * rect.width()
                p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
                p.drawText(QPointF(x + 3, rect.top() + 15), str(label))
        else:
            p.setPen(QPen(QColor("#e5858b"), 1.1))
            marker_labels = []
            for start, finish, label in self.section_markers:
                a, b = self.world_to_screen(start), self.world_to_screen(finish)
                p.drawLine(a, b)
                anchor = a + QPointF(3, -3)
                bounds = p.fontMetrics().boundingRect(str(label))
                label_rect = QRectF(anchor.x(), anchor.y() - bounds.height(),
                                    bounds.width() + 5, bounds.height() + 3)
                if not any(label_rect.intersects(previous) for previous in marker_labels):
                    p.drawText(anchor, str(label))
                    marker_labels.append(label_rect)
            for index, value, q, vector in self._data_range_lines():
                p.setPen(QPen(QColor("#d8b78a"), 1.5, Qt.PenStyle.DashLine))
                span = max(self.width_pixels(), self.height()) * 2
                p.drawLine(q - vector * span, q + vector * span)
                p.drawText(q + QPointF(8, -5), f"{'Lower' if index == 0 else 'Upper'} {value:+.1f} mm")
        if self.segmentation_image is not None:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            p.drawImage(rect, self.segmentation_image)
            p.setFont(QFont("Segoe UI Variable", 8, QFont.Weight.DemiBold))
            occupied = []
            for x, y, text in ([] if compact else self.tooth_annotations):
                point = QPointF(rect.left() + rect.width() * x, rect.top() + rect.height() * y)
                bounds = QRectF(point.x() - 11, point.y() - 9, 22, 18)
                if any(bounds.intersects(previous) for previous in occupied):
                    continue
                occupied.append(bounds)
                p.fillRect(bounds, QColor(15, 17, 21, 185))
                p.setPen(QColor("#f0e4d3"))
                p.drawText(bounds, Qt.AlignmentFlag.AlignCenter, text)
        if self.frame is None and self.cursor_fraction is not None and self.show_references:
            x, y = self.cursor_fraction
            point = QPointF(rect.left() + float(x) * rect.width(), rect.top() + float(y) * rect.height())
            p.setPen(QPen(QColor("#e5858b"), 1.2))
            p.drawLine(QPointF(point.x(), rect.top()), QPointF(point.x(), rect.bottom()))
            p.setPen(QPen(QColor("#71a9f7"), 1.2))
            p.drawLine(QPointF(rect.left(), point.y()), QPointF(rect.right(), point.y()))
            p.setPen(QPen(QColor("#e8e6e1"), 1))
            p.drawEllipse(point, 3, 3)
        if self.frame is not None:
            center, right, up, size = self.frame
            size = self.edge_size_mm()
            p.setPen(QColor("#94a8bf"))
            if not compact:
                p.drawText(QRectF(14, 39, 50, 20), direction_label(-right))
                p.drawText(QRectF(self.width_pixels() - 60, 39, 46, 20), Qt.AlignmentFlag.AlignRight, direction_label(right))
                p.drawText(QRectF(self.width_pixels() / 2 - 25, 39, 50, 20), Qt.AlignmentFlag.AlignCenter, direction_label(up))
            if self.crosshair is not None and self.reference_lines and self.show_references:
                q = self.world_to_screen(self.crosshair)
                span = max(rect.width(), rect.height()) * 2
                for reference, direction, handle in self._reference_geometry():
                    color = QColor(reference["color"])
                    color.setAlpha(210)
                    p.setPen(QPen(color, 1.2))
                    gap = direction * 7
                    p.drawLine(q - direction * span, q - gap)
                    p.drawLine(q + gap, q + direction * span)
                    p.setBrush(QColor("#17191d"))
                    p.setPen(QPen(color, 1.6))
                    p.drawEllipse(handle, 3.8, 3.8)
                    p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(QColor("#e8e6e1"), 1))
                p.drawEllipse(q, 3, 3)
            elif self.crosshair is not None and self.show_references:
                q = self.world_to_screen(self.crosshair)
                p.setPen(QPen(QColor(self.accent), 1, Qt.PenStyle.DashLine))
                p.drawLine(QPointF(rect.left(), q.y()), QPointF(rect.right(), q.y()))
                p.drawLine(QPointF(q.x(), rect.top()), QPointF(q.x(), rect.bottom()))
                p.setPen(QPen(QColor(self.accent), 1.5))
                p.drawEllipse(q, 4, 4)
            if self.points:
                p.setPen(QPen(QColor("#f6bd60"), 2))
                screen_points = [self.world_to_screen(x) for x in self.points]
                for a, b in zip(screen_points, screen_points[1:]):
                    p.drawLine(a, b)
                for i, q in enumerate(screen_points):
                    p.drawEllipse(q, 4, 4)
                    if len(self.points) <= 5 or i == self.selected_point or i % 5 == 0 or i == len(self.points) - 1:
                        p.drawText(q + QPointF(7, -7), str(i + 1))
                    if i == self.selected_point:
                        p.drawEllipse(q, 7, 7)
            if len(self.path_points) > 1:
                p.setPen(QPen(QColor("#ffdc8f"), 1.5))
                for a, b in zip(self.path_points, self.path_points[1:]):
                    p.drawLine(self.world_to_screen(a), self.world_to_screen(b))
            normal = np.cross(right, up)
            self._draw_nerve_paths(p)
            for landmark in self.landmarks:
                position = np.asarray(landmark["position_lps_mm"])
                if abs(np.dot(position - center, normal)) > .5:
                    continue
                q = self.world_to_screen(position)
                color = "#6bc7a3" if landmark["review"] == "accepted" else "#e5ba70"
                p.setPen(QPen(QColor(color), 1.7))
                p.drawEllipse(q, 5, 5)
                p.drawText(q + QPointF(8, 14), landmark["name"] + " · " + landmark["review"])
            implant_label_row = 0
            for implant in self.implants:
                a = np.asarray(implant["center"]) - np.asarray(implant["axis"]) * implant["length"] / 2
                b = np.asarray(implant["center"]) + np.asarray(implant["axis"]) * implant["length"] / 2
                normal = np.cross(right, up)
                half_extent = implant["length"] / 2 * abs(np.dot(implant["axis"], normal)) + implant["diameter"] / 2
                if abs(np.dot(np.asarray(implant["center"]) - center, normal)) <= half_extent:
                    qa, qb = self.world_to_screen(a), self.world_to_screen(b)
                    p.setPen(QPen(QColor("#48d7b1"), 1.5, Qt.PenStyle.DashLine))
                    radius = implant["diameter"] / size[0] * rect.width() / 2
                    delta = qb - qa
                    length = np.hypot(delta.x(), delta.y())
                    if length > 1:
                        edge = QPointF(-delta.y() / length * radius, delta.x() / length * radius)
                        p.drawLine(qa + edge, qb + edge)
                        p.drawLine(qa - edge, qb - edge)
                        p.drawLine(qa + edge, qa - edge)
                        p.drawLine(qb + edge, qb - edge)
                    else:
                        p.drawEllipse(qa, radius, radius)
                    p.setPen(QColor("#48d7b1"))
                    if self.width_pixels() >= 350 and (self.mode != "Move implant" or implant.get("id") == self.selected_implant_id):
                        label_box = QRectF(12, 86 + implant_label_row * 20, self.width_pixels() - 24, 19)
                        p.fillRect(label_box, QColor(15, 17, 21, 200))
                        p.drawText(label_box, Qt.AlignmentFlag.AlignVCenter,
                                   p.fontMetrics().elidedText(implant["name"] + " · projection", Qt.TextElideMode.ElideRight, int(label_box.width())))
                        implant_label_row += 1
            if self.mode == "Move implant":
                for implant, kind, position in self.implant_handles():
                    point = self.world_to_screen(position)
                    p.setPen(QPen(QColor("#f6d09b"), 1.5))
                    p.setBrush(QColor("#202329"))
                    p.drawEllipse(point, 6, 6)
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    if implant.get("id") == self.selected_implant_id:
                        p.drawText(point + QPointF(8, 15), "Move" if kind == "center" else "Axis")
            result = self.clearance_measurement
            if result is not None and result.get("implant_point_lps") is not None:
                a, b = np.asarray(result["implant_point_lps"]), np.asarray(result["canal_point_lps"])
                projected = max(abs((a - center) @ normal), abs((b - center) @ normal)) > self.path_slice_half_mm
                color = "#ef8487" if result.get("geometry_status") in ("overlap", "within_margin") else "#e5ba70"
                p.setPen(QPen(QColor(color), 2, Qt.PenStyle.DashLine if projected else Qt.PenStyle.SolidLine))
                qa, qb = self.world_to_screen(a), self.world_to_screen(b)
                p.drawLine(qa, qb)
                p.drawEllipse(qa, 3, 3)
                p.drawEllipse(qb, 3, 3)
                label = f"{result['distance_mm']:.2f} mm"
                if result.get("status") == "unreviewed":
                    label += " · unreviewed"
                if projected:
                    label += " · projection"
                label_box = QRectF(12, 64, self.width_pixels() - 24, 20)
                p.fillRect(label_box, QColor(15, 17, 21, 210))
                p.drawText(label_box, Qt.AlignmentFlag.AlignVCenter,
                           p.fontMetrics().elidedText(label, Qt.TextElideMode.ElideRight, int(label_box.width())))
            for a, b in self.measurements:
                normal = np.cross(right, up)
                if max(abs(np.dot(a - center, normal)), abs(np.dot(b - center, normal))) > 0.5:
                    continue
                qa, qb = self.world_to_screen(a), self.world_to_screen(b)
                p.setPen(QPen(QColor("#f6bd60"), 2))
                p.drawLine(qa, qb)
                p.drawEllipse(qa, 3, 3)
                p.drawEllipse(qb, 3, 3)
                p.drawText((qa + qb) / 2 + QPointF(5, -6), f"{np.linalg.norm(b - a):.2f} mm")
            if self.mode in ("Paint mask", "Erase mask") and self._brush_hover is not None:
                p.setPen(QPen(QColor(self.brush_color), 1.4))
                p.drawEllipse(self._brush_hover, self.brush_radius_mm / size[0] * rect.width(), self.brush_radius_mm / size[1] * rect.height())
            # A physical scale bar is shown only when DICOM spacing is calibrated.
            length = 10 if size[0] / self.zoom < 150 else 20
            pixels = length / size[0] * rect.width()
            p.setPen(QPen(QColor("#b5c5d5"), 2))
            q = QPointF(18, self.height() - 43)
            if self.calibrated and not compact:
                p.drawLine(q, q + QPointF(pixels, 0))
                p.drawText(q + QPointF(0, -5), f"{length} mm")
            elif not self.calibrated and not compact:
                p.drawText(q + QPointF(0, -5), "Uncalibrated geometry · display coordinates")
        p.setClipping(False)
        if compact:
            return
        if self.source_text:
            p.setFont(QFont("Segoe UI Variable", 8))
            p.setPen(QColor("#a8acb5"))
            p.drawText(QRectF(12, self.height() - 66, self.width_pixels() - 24, 18), Qt.AlignmentFlag.AlignRight,
                       p.fontMetrics().elidedText(self.source_text, Qt.TextElideMode.ElideRight, self.width_pixels() - 24))
        p.fillRect(QRectF(0, self.height() - 25, self.width_pixels(), 25), QColor("#17191f"))
        p.setPen(QColor("#9a9fac"))
        note = "Mask preview · full detail on release" if self.segmentation_preview else ("Drag solid canal nodes · dashed paths are projections" if self.mode == "Edit canal path" else self.note)
        p.drawText(QRectF(12, self.height() - 24, self.width_pixels() - 24, 22), Qt.AlignmentFlag.AlignVCenter,
                   p.fontMetrics().elidedText(note or f"W {self.window_width:.0f}   L {self.center:.0f}   ·   Scroll to browse", Qt.TextElideMode.ElideRight, self.width_pixels() - 24))

    def _reference_geometry(self):
        if self.frame is None or self.data is None or self.crosshair is None or not self.show_references:
            return []
        center, right, up, _ = self.frame
        size = self.edge_size_mm()
        rect = self._rect()
        q = self.world_to_screen(self.crosshair)
        result = []
        radius = min(rect.width(), rect.height()) * .32
        for reference in self.reference_lines:
            world = np.asarray(reference["direction"])
            direction = QPointF(np.dot(world, right) / size[0] * rect.width(), -np.dot(world, up) / size[1] * rect.height())
            length = np.hypot(direction.x(), direction.y())
            if length < 1e-8:
                continue
            direction /= length
            result.append((reference, direction, q + direction * radius))
        return result

    def reference_hit(self, point):
        if self.frame is None or self.data is None or self.crosshair is None or not self.show_references or not self.reference_lines or not self._rect().contains(point):
            return None, None
        q = self.world_to_screen(self.crosshair)
        delta = point - q
        if np.hypot(delta.x(), delta.y()) < 11:
            return "center", None
        for reference, direction, handle in self._reference_geometry():
            if np.hypot((point - handle).x(), (point - handle).y()) < 10:
                return "rotate", reference["name"]
        for reference, direction, handle in self._reference_geometry():
            distance = abs(delta.x() * direction.y() - delta.y() * direction.x())
            if distance < 7:
                return "line", reference["name"]
        return None, None

    def mousePressEvent(self, event):
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self.last_pos = event.position()
        self._drag_kind = None
        world = self.screen_to_world(event.position())
        self._drag_world = world
        if event.button() == Qt.MouseButton.LeftButton and self.image_canvas_rect().contains(event.position()):
            for index, _, q, vector in self._data_range_lines():
                delta = event.position() - q
                if abs(delta.x() * vector.y() - delta.y() * vector.x()) <= 7:
                    self._drag_kind, self._drag_target = "data-range", index
                    self.interaction_started.emit()
                    event.accept()
                    return
        if (self.fraction_navigation_enabled and self.frame is None and self.image is not None
                and self.mode in ("Navigate", "Correct planes")
                and self.image_canvas_rect().contains(event.position())
                and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton)):
            rect = self._rect()
            fraction = ((event.position().x() - rect.left()) / rect.width(),
                        (event.position().y() - rect.top()) / rect.height())
            if event.button() == Qt.MouseButton.LeftButton and not rect.contains(event.position()):
                return
            origin = fraction if event.button() == Qt.MouseButton.LeftButton else (
                (.5, .5) if self.cursor_fraction is None else self.cursor_fraction)
            self._fraction_drag = (QPointF(event.position()), tuple(origin))
            self._drag_kind = "fraction"
            self.interaction_started.emit()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            if event.button() == Qt.MouseButton.LeftButton:
                self.picked.emit({"world": None, "fraction": tuple(np.clip(origin, 0, 1))})
            event.accept()
            return
        # A reference plane extends beyond the fitted image rectangle. Right
        # drag on its surrounding canvas must use that same physical mapping,
        # rather than unexpectedly switching to window/level.
        canvas = self.image_canvas_rect()
        if (self.mode == "Add arch point" and event.button() == Qt.MouseButton.RightButton
                and canvas.contains(event.position())):
            self.path_finished.emit()
            event.accept()
            return
        if (event.button() == Qt.MouseButton.RightButton and canvas.contains(event.position())
                and self.reference_lines and self.show_references and self.crosshair is not None):
            world = self.screen_to_world(event.position(), allow_outside=True)
            if world is not None:
                self._drag_world = world
                self._drag_kind = "rotate"
                self.interaction_started.emit()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
        if self.mode == "Move implant" and event.button() == Qt.MouseButton.LeftButton and world is not None:
            hit = self.implant_hit(event.position())
            if hit is not None:
                implant, kind, position = hit
                rect, size = self._rect(), self.edge_size_mm()
                self._implant_drag = (implant["id"], kind, np.asarray(implant["center"]).copy(), np.asarray(implant["axis"]).copy(),
                    float(implant["length"]), QPointF(event.position()), self.frame[1] * size[0] / rect.width(), -self.frame[2] * size[1] / rect.height())
                self._drag_kind = "implant"
                self.selected_implant_id = implant["id"]
                self._emit_implant_drag(event.position(), "start")
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                return
        if self.mode == "Edit canal path" and event.button() == Qt.MouseButton.LeftButton and world is not None:
            hit = self.nerve_node_hit(event.position())
            if hit is not None:
                path_id, index, position = hit
                self._drag_kind = "nerve"
                rect, size = self._rect(), self.edge_size_mm()
                self._path_drag = (path_id, index, position.copy(), QPointF(event.position()),
                    self.frame[1] * size[0] / rect.width(), -self.frame[2] * size[1] / rect.height(),
                    np.cross(self.frame[1], self.frame[2]), self.frame[0].copy())
                self.path_node_dragged.emit({"phase": "start", "path_id": path_id, "node_index": index, "position_lps_mm": position.copy(), "plane_center": self._path_drag[-1], "plane_normal": self._path_drag[-2]})
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                return
        if self.mode in ("Paint mask", "Erase mask") and event.button() == Qt.MouseButton.LeftButton and world is not None:
            self._drag_kind = "mask"
            self.mask_stroke.emit({"phase": "start", "world": world, "frame": self.frame, "erase": self.mode == "Erase mask"})
            return
        if self.mode in ("Add arch point", "Edit arch") and world is not None:
            if event.button() == Qt.MouseButton.RightButton:
                self.path_finished.emit()
                return
            if event.button() == Qt.MouseButton.LeftButton:
                for i, position in enumerate(self.points):
                    q = self.world_to_screen(position)
                    if np.hypot((q - event.position()).x(), (q - event.position()).y()) < 10:
                        self.selected_point = i
                        self._drag_kind = "point"
                        self.interaction_started.emit()
                        self.update()
                        return
                self.selected_point = None
        kind, target = self.reference_hit(event.position())
        if world is not None and self.reference_lines and self.show_references and self.mode in ("Navigate", "Correct planes"):
            rotate = self.mode == "Correct planes" or kind == "rotate" or event.button() == Qt.MouseButton.RightButton
            if rotate:
                self._drag_kind = "rotate"
                self.interaction_started.emit()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                return
        if event.button() == Qt.MouseButton.LeftButton:
            if self.mode == "Pan":
                self._drag_kind = "pan"
                return
            if self.mode == "Window / level":
                return
            if self.mode == "Measure" and world is not None and self.calibrated:
                if self.pending_measure is None:
                    self.pending_measure = world
                    self.measured.emit("Choose the second point")
                else:
                    self.measurements.append((self.pending_measure, world))
                    self.measured.emit(f"Distance: {np.linalg.norm(world - self.pending_measure):.2f} mm")
                    self.pending_measure = None
                    self.update()
            elif self.mode != "Measure":
                rect = self._rect()
                if rect.contains(event.position()):
                    if self.mode == "Navigate" and self.reference_lines and kind in ("center", "line"):
                        self._drag_kind, self._drag_target = "reference", target
                        self.interaction_started.emit()
                    else:
                        if self.mode == "Navigate" and self.reference_lines:
                            self._drag_kind, self._drag_target = "reference", None
                            self.interaction_started.emit()
                        self.picked.emit({"world": world, "fraction": ((event.position().x() - rect.left()) / rect.width(), (event.position().y() - rect.top()) / rect.height())})

    def mouseMoveEvent(self, event):
        self._brush_hover = event.position()
        if self.mode in ("Paint mask", "Erase mask"):
            self.update()
        if self.last_pos is None:
            if self.mode == "Edit canal path":
                self.setCursor(Qt.CursorShape.OpenHandCursor if self.nerve_node_hit(event.position()) else Qt.CursorShape.CrossCursor)
                return
            kind, target = self.reference_hit(event.position())
            if self.mode == "Correct planes" or kind == "rotate":
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            elif kind in ("center", "line"):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            elif self.mode == "Pan":
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.CrossCursor if self.mode in ("Measure", "Place implant") else Qt.CursorShape.ArrowCursor)
            return
        delta = event.position() - self.last_pos
        if self._drag_kind == "data-range" and self.data_range is not None:
            world = self.screen_to_world(event.position(), allow_outside=True)
            if world is not None and self.image_canvas_rect().contains(event.position()):
                setting = self.data_range
                lower, upper = setting["bounds"]
                value = float((world - setting["origin"]) @ setting["axis"])
                if self._drag_target == 0:
                    lower = float(np.clip(value, setting["limits"][0], upper - setting["step"]))
                else:
                    upper = float(np.clip(value, lower + setting["step"], setting["limits"][1]))
                setting["bounds"] = lower, upper
                self.range_bounds_changed.emit(lower, upper)
                self.update()
            self.last_pos = event.position()
            event.accept()
            return
        if self._drag_kind == "fraction" and self._fraction_drag is not None:
            mouse, origin = self._fraction_drag
            rect = self._rect()
            motion = event.position() - mouse
            fraction = (origin[0] + motion.x() / rect.width(), origin[1] + motion.y() / rect.height())
            self.picked.emit({"world": None, "fraction": tuple(np.clip(fraction, 0, 1))})
            self.last_pos = event.position()
            event.accept()
            return
        world = self.screen_to_world(event.position(),
                                     allow_outside=self._drag_kind in ("rotate", "reference"))
        if (self._drag_kind in ("rotate", "reference")
                and not QRectF(1, 32, self.width_pixels() - 2, self.height() - 58).contains(event.position())):
            world = self._drag_world = None
        if self._drag_kind == "implant" and self._implant_drag is not None:
            self._emit_implant_drag(event.position(), "move")
        elif self._drag_kind == "nerve" and self._path_drag is not None:
            path_id, index, start, mouse, across, down, normal, center = self._path_drag
            pixels = event.position() - mouse
            position = start + pixels.x() * across + pixels.y() * down
            self.path_node_dragged.emit({"phase": "move", "path_id": path_id, "node_index": index, "position_lps_mm": position, "plane_center": center, "plane_normal": normal})
        elif self._drag_kind == "mask" and world is not None:
            self.mask_stroke.emit({"phase": "move", "world": world, "frame": self.frame, "erase": self.mode == "Erase mask"})
        elif self._drag_kind == "point" and world is not None and self._drag_world is not None:
            self.point_moved.emit(self.selected_point, world - self._drag_world)
        elif self._drag_kind == "rotate" and world is not None and self._drag_world is not None and self.crosshair is not None:
            _, right, up, _ = self.frame
            previous = self._drag_world - self.crosshair
            current = world - self.crosshair
            if min(np.linalg.norm(previous), np.linalg.norm(current)) > 1:
                before = np.arctan2(np.dot(previous, up), np.dot(previous, right))
                after = np.arctan2(np.dot(current, up), np.dot(current, right))
                angle = (after - before + np.pi) % (2 * np.pi) - np.pi
                self.reference_rotated.emit(float(np.rad2deg(angle)))
        elif self._drag_kind == "reference" and world is not None and self._drag_world is not None:
            self.reference_translated.emit({"delta": world - self._drag_world, "target": self._drag_target})
        elif event.buttons() & Qt.MouseButton.MiddleButton or self._drag_kind == "pan":
            self.pan += delta
            self.update()
        if world is not None:
            self._drag_world = world
        self.last_pos = event.position()

    def mouseReleaseEvent(self, event):
        if self._drag_kind == "implant":
            self._emit_implant_drag(event.position(), "end")
            self._implant_drag = None
        if self._drag_kind == "nerve":
            self.path_node_dragged.emit({"phase": "end"})
            self._path_drag = None
        if self._drag_kind == "mask":
            self.mask_stroke.emit({"phase": "end"})
        if self._drag_kind in ("rotate", "reference", "point", "fraction", "data-range"):
            self.interaction_finished.emit()
        self._fraction_drag = None
        self._drag_kind = None
        self.last_pos = None
        self._drag_world = None
        self.unsetCursor()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.mode == "Edit canal path" and self.selected_nerve_node is not None:
            self.path_node_removed.emit({"path_id": self.selected_nerve_path, "node_index": self.selected_nerve_node})
            event.accept()
        elif event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.mode in ("Add arch point", "Edit arch") and self.selected_point is not None:
            self.point_removed.emit(self.selected_point)
            self.selected_point = None
            event.accept()
        else:
            super().keyPressEvent(event)

    def _position_header_controls(self):
        dense = self.compact_grid()
        compact = self.width_pixels() < 180 or dense
        if compact:
            self._restore_max_button = self._restore_max_button or not self.max_button.isHidden()
            self.max_button.hide()
            self.snapshot_button.hide()
        else:
            self.snapshot_button.show()
            if self._compact_header and self._restore_max_button:
                self.max_button.show()
        self._compact_header = compact
        self.max_button.setGeometry(self.width_pixels() - 28, 4, 23, 23)
        self.snapshot_button.setGeometry(self.width_pixels() - 54, 4, 23, 23)
        self.settings_button.setGeometry(self.width_pixels() - (21 if dense else 28 if compact else 80),
                                         1 if dense else 4, 20 if dense else 23, 20 if dense else 23)
        if hasattr(self, "position_slider"):
            top = 22 if dense else 34
            bottom = 2 if dense else 27
            self.position_slider.setGeometry(self.width_pixels() - 15, top, 14,
                                             max(20, self.height() - top - bottom))

    def resizeEvent(self, event):
        self._position_header_controls()
        super().resizeEvent(event)
        self.sampling_changed.emit()

    def leaveEvent(self, event):
        self._brush_hover = None
        self.update()
        super().leaveEvent(event)

    def wheelEvent(self, event):
        if self._drag_kind in ("mask", "nerve", "implant"):
            event.accept()
            return
        steps = event.angleDelta().y() / 120
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom = min(12, max(0.2, self.zoom * 1.15 ** steps))
            self.zoom_changed.emit(self.zoom)
            self.update()
            self.sampling_changed.emit()
        else:
            self.scrolled.emit(steps)
        event.accept()

    def implant_handles(self):
        if self.frame is None or self.data is None:
            return []
        center, right, up, _ = self.frame
        normal = np.cross(right, up)
        handles = []
        for implant in self.implants:
            position, axis = np.asarray(implant["center"]), np.asarray(implant["axis"])
            extent = implant["length"] / 2 * abs(axis @ normal) + implant["diameter"] / 2
            if abs((position - center) @ normal) > extent:
                continue
            handles.append((implant, "center", position))
            tip = position + axis * implant["length"] / 2
            delta = self.world_to_screen(tip) - self.world_to_screen(position)
            if np.hypot(delta.x(), delta.y()) > 18:
                handles.append((implant, "axis", tip))
        return handles

    def implant_hit(self, point):
        hits = []
        for item in self.implant_handles():
            q = self.world_to_screen(item[2]) - point
            distance = np.hypot(q.x(), q.y())
            if distance <= 10:
                hits.append((distance, item))
        return min(hits, key=lambda value: value[0])[1] if hits else None

    def _emit_implant_drag(self, point, phase):
        key, kind, center, axis, length, mouse, across, down = self._implant_drag
        delta = point - mouse
        motion = delta.x() * across + delta.y() * down
        candidate = center + motion if kind == "center" else center.copy()
        direction = axis * length / 2 + motion
        if kind == "axis" and np.linalg.norm(direction) > 1e-8:
            axis = direction / np.linalg.norm(direction)
        self.implant_dragged.emit({"phase": phase, "implant_id": key, "center_lps": candidate, "axis_lps": axis})
