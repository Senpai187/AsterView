from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal, QSignalBlocker, QBuffer, QIODevice
from PySide6.QtGui import QAction, QKeySequence, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFileDialog, QSplitter, QTabWidget, QComboBox,
    QDoubleSpinBox, QSpinBox, QFormLayout, QGroupBox, QScrollArea, QListWidget,
    QProgressBar, QMessageBox, QCheckBox, QSlider,
)

from . import __version__
from .dicom_io import discover_series, load_series
from .demo import make_demo
from .reformat import sample_plane, base_frame, rotation_matrix, curved_planar, path_frame
from .render3d import VolumeView, PRESETS
from .slice_view import SliceView
from .mpr_geometry import plane_frame, plane_references, rotate_about_view, to_euler, translate_reference
from .fast_reslice import FastReslicer
from .dental_controls import DentalControls
from .planning_controls import PlanningControls
from .workflow_controls import WorkflowControls
from .presentation import patient_display_name
from .slice_scheduler import LatestSliceQueue


from .ui_shell import STYLE


class Job(QThread):
    result = Signal(object)
    error = Signal(str)
    progress = Signal(int, str)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function
        self.cancelled = False

    def run(self):
        try:
            result = self.function(self.progress.emit, lambda: self.cancelled)
            if not self.cancelled:
                self.result.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


def spin(value, low, high, step=1, decimals=1):
    result = QDoubleSpinBox()
    result.setRange(low, high)
    result.setDecimals(decimals)
    result.setSingleStep(step)
    result.setValue(value)
    result.setKeyboardTracking(False)
    return result


def button(text, callback, primary=False):
    result = QPushButton(text)
    result.clicked.connect(callback)
    if primary:
        result.setObjectName("primary")
    return result


class MainWindow(PlanningControls, WorkflowControls, DentalControls, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"AsterView {__version__}")
        self.setWindowIcon(QIcon(str(Path(__file__).with_name("assets") / "asterview.ico")))
        self.resize(1520, 1000)
        self.volume = None
        self.segmentation = None
        self.nerve_paths = None
        self._path_gesture_active = False
        self.path_timer = QTimer(self)
        self.path_timer.setSingleShot(True)
        self.path_timer.setInterval(16)
        self.path_timer.timeout.connect(self.refresh_nerve_paths)
        self._mask_previous = None
        self._mask_changed = False
        self.mask_timer = QTimer(self)
        self.mask_timer.setSingleShot(True)
        self.mask_timer.setInterval(16)
        self.mask_timer.timeout.connect(lambda: self.refresh_segmentation(update_3d=False))
        self.crosshair = None
        self.series = []
        self.arch_points = []
        self._arch_up = None
        self.pano_result = None
        self.pano_revision = 0
        self._pano_bounds = (-45., 45.)
        self.slice_job = None
        self.slice_revision = 0
        self.slice_pending = None
        self.slice_queue = LatestSliceQueue()
        self._plane_requests = None
        self._closing = False
        self._report_analysis_active = False
        self._report_analysis_completed = False
        self.pano_up = np.array([0., 0., 1.])
        self.implants = []
        self.implant_margins = {}
        self.clearance_results = []
        self._clearance_display = None
        self._implant_drag_active = False
        self.clearance_timer = QTimer(self)
        self.clearance_timer.setSingleShot(True)
        self.clearance_timer.setInterval(16)
        self.clearance_timer.timeout.connect(self.update_clearances)
        self.tmj_centers = {"Right": None, "Left": None}
        self.job = None
        self.mode = "Navigate"
        self.last_frame_ms = 0
        self._interaction_active = False
        self._rotation_override = None
        self._mpr_sizes = {}
        self._wheel_settle = QTimer(self)
        self._wheel_settle.setSingleShot(True)
        self._wheel_settle.setInterval(120)
        self._wheel_settle.timeout.connect(self.end_interaction)
        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.setInterval(16)
        self.render_timer.timeout.connect(self.render_views)
        self.pano_refresh_timer = QTimer(self)
        self.pano_refresh_timer.setSingleShot(True)
        self.pano_refresh_timer.setInterval(350)
        self.pano_refresh_timer.timeout.connect(self.refresh_panorama_after_edit)
        self._build_ui()

    def _build_ui(self):
        from .ui_shell import build
        build(self)

    def edit_clinician_profile(self):
        from .clinician_profile import ensure_clinician_profile
        profile = ensure_clinician_profile(self, force=True)
        wizard = getattr(self.reporting_panel, "report_wizard", None)
        if wizard is not None:
            wizard.load_report()
        return profile

    def show_superimposition(self):
        self.tabs.setCurrentIndex(self.superimposition_index)
        self.inspector.show()

    def restore_section_views(self):
        if self.volume is None:
            return
        self.reset_orientation()
        for view in self.all_slices:
            if view.isVisible():
                view.reset_view()
        self.schedule_render()
        self.statusBar().showMessage("Section orientation, zoom and pan restored")

    def restore_loaded_scan(self):
        """The explicit top-level Restore starts a fresh workspace for this data."""
        if self.volume is None:
            return
        volume = self.volume
        if self.job is not None:
            self.job.cancelled = True
            self.job.finished.connect(
                lambda: QTimer.singleShot(0, self.restore_loaded_scan)
                if self.volume is volume and not self._closing else None)
            self.statusBar().showMessage("Stopping the active operation before restoring this scan…")
            return
        self.reset_workspace()
        self.set_volume(volume)
        self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage("DICOM restored to a newly loaded workspace")

    def capture_report_views(self, view_keys):
        if self.volume is None:
            raise ValueError("Open a scan before attaching views.")
        allowed = {"Axial", "Coronal", "Sagittal", "3D", "Panoramic"}
        if not view_keys or len(set(view_keys)) != len(view_keys) or any(key not in allowed for key in view_keys):
            raise ValueError("Choose distinct available scan views.")
        records = []
        for key in view_keys:
            state = {"crosshair_lps_mm": self.crosshair.tolist(),
                     "rotation": self.rotation().tolist(),
                     "window": [self.window_level.value(), self.window_width.value()]}
            if key == "3D":
                image = self.volume_view.capture_image(clean=True)
            elif key == "Panoramic":
                if self.pano_result is None or self._pano_offset_dirty or self.pano_refresh_timer.isActive():
                    raise ValueError("Wait for the updated panoramic image before attaching it.")
                image = self.pano_view.capture_content_image()
                import hashlib
                state["path_sha256"] = hashlib.sha256(self.pano_result.path.tobytes()).hexdigest()
                state["vertical_axis_lps"] = self.pano_up.tolist()
                state["focal_offset_mm"] = self.pano_offset.value()
                state["vertical_bounds_mm"] = list(self.pano_result.vertical_limits_mm)
            else:
                view = self.mpr_views[key]
                right, up = plane_frame(key, self.rotation())
                normal = np.cross(right, up)
                center = self.volume.center + normal * np.dot(self.crosshair - self.volume.center, normal)
                size = self._mpr_sizes[key]
                slab = self.slab.value() if view.slab_mm is None else view.slab_mm
                data = sample_plane(self.volume, center, right, up, size,
                                    shape=view.preferred_sample_shape(size),
                                    slab_mm=slab, mode=view.slab_mode or self.slab_mode.currentText())
                view.set_image(data, (center, right, up, size), self.crosshair)
                image = view.capture_content_image()
                state.update(slab_mm=slab, zoom=float(view.zoom), pan=[view.pan.x(), view.pan.y()])
            if image.isNull():
                raise ValueError(f"{key} has no completed image to attach.")
            buffer = QBuffer()
            buffer.open(QIODevice.OpenModeFlag.WriteOnly)
            if not image.save(buffer, "PNG"):
                raise ValueError("The selected image could not be captured.")
            records.append({"title": key, "png": bytes(buffer.data()),
                            "source": self.volume_fingerprint(), "view_state": state})
        return records

    def analyze_for_report(self):
        """Prepare a local review inventory and supported, pending image candidates."""
        import copy
        from .local_review import analyze_local
        panel = self.reporting_panel
        if self.volume is None:
            panel.set_analysis_status("Open a DICOM volume before preparing a report.")
            return False
        if self.job is not None:
            panel.set_analysis_status("Wait for the current operation or cancel it first.")
            return False
        panel.document = panel._sync(commit_editor=True)
        volume, fingerprint, segmentation = self.volume, self.volume_fingerprint(), self.segmentation
        context = (volume, panel._epoch, panel._revision, panel._form_revision,
                   copy.deepcopy(panel.document), segmentation)
        self._report_analysis_completed = False

        def compute(progress, cancel):
            return analyze_local(volume, fingerprint, segmentation, progress=progress, cancel=cancel)

        def completed(result):
            if self.volume is not volume or panel._epoch != context[1]:
                panel.set_analysis_status("The scan changed; analysis was discarded.")
                return
            try:
                installed = True
                if result["batch"] is not None:
                    installed = panel.install_candidates(result["batch"], context)
                panel.install_analysis_summary(result["summary"], source=fingerprint,
                                               coverage=result["coverage"])
                message = ("Local review preparation complete. Review pending image candidates and enter the diagnostic interpretation."
                           if installed else "Scan inventory completed. Findings changed during processing; rerun to add candidates.")
                panel.set_analysis_status(message, progress=100, busy=False)
                self._report_analysis_completed = True
            except (ValueError, RuntimeError) as exc:
                panel.set_analysis_status(str(exc), busy=False)

        if self.start_job(compute, completed):
            self._report_analysis_active = True
            panel.set_busy(True)
            panel.set_analysis_status("Checking the complete acquired data locally…", progress=0, busy=True)
            return True
        return False

    def new_slice(self, title, color="#65c6db"):
        view = SliceView(title, color)
        view.window_dragged.connect(self.drag_window)
        view.measured.connect(self.statusBar().showMessage)
        view.mask_stroke.connect(self.paint_mask)
        view.path_node_dragged.connect(self.path_node_dragged)
        view.path_node_removed.connect(self.delete_path_node)
        view.implant_dragged.connect(self.implant_dragged)
        view.snapshot_requested.connect(lambda v=view: self.snapshot_section(v))
        view.sampling_changed.connect(lambda v=view: self.slice_sampling_changed(v))
        from .ui_shell import wire_section_view
        wire_section_view(self, view)
        self.all_slices.append(view)
        return view

    def _build_panorama(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        bar = QHBoxLayout()
        self.arch_toggle = QCheckBox("Add arch points on axial view")
        self.arch_toggle.setChecked(True)
        self.arch_toggle.toggled.connect(self.sync_arch_mode)
        bar.addWidget(self.arch_toggle)
        bar.addWidget(button("Auto trace arch", self.auto_arch))
        bar.addWidget(button("Undo point", self.undo_arch))
        bar.addWidget(button("Clear arch", self.clear_arch))
        bar.addStretch()
        bar.addWidget(button("Reformat", self.generate_panorama, True))
        layout.addLayout(bar)
        options = QHBoxLayout()
        options.addWidget(QLabel("Thickness"))
        from .thickness_control import ThicknessControl
        self.pano_thickness = ThicknessControl(5)
        self.pano_thickness.setSuffix(" mm")
        self.pano_thickness.setMaximumWidth(90)
        options.addWidget(self.pano_thickness)
        self.pano_projection = QComboBox()
        self.pano_projection.addItems(["mean", "max", "min"])
        self.pano_thickness.valueChanged.connect(self.invalidate_panorama)
        self.pano_projection.currentTextChanged.connect(self.invalidate_panorama)
        self.pano_projection.setMaximumWidth(100)
        options.addWidget(self.pano_projection)
        options.addWidget(QLabel("Arch depth"))
        self.pano_offset = spin(0, -20, 20, .4, 2)
        self.pano_offset.setSuffix(" mm")
        self.pano_offset.setToolTip("Move the panoramic focal trough inward or outward; wheel over the panorama does the same.")
        self.pano_offset.valueChanged.connect(self.panorama_offset_changed)
        options.addWidget(self.pano_offset)
        from .ui_shell import icon
        range_button = button("Data range", self.define_data_range)
        range_button.setIcon(icon("crop-simple"))
        range_button.setToolTip("Choose the panoramic upper and lower bounds")
        options.addWidget(range_button)
        options.addStretch()
        layout.addLayout(options)
        tip = QLabel("Auto trace prepares the arch. Drag points to correct it; right-click to finish drawing. The panorama updates automatically. Use Implants for serial cross-sections.")
        tip.setWordWrap(True)
        tip.setObjectName("subtitle")
        layout.addWidget(tip)
        grid = QGridLayout()
        self.panorama_grid = grid
        self.arch_view = self.new_slice("Axial · Arch path")
        self.arch_view.picked.connect(self.arch_picked)
        self.arch_view.point_moved.connect(self.move_arch_point)
        self.arch_view.point_removed.connect(self.remove_arch_point)
        self.arch_view.path_finished.connect(self.finish_arch)
        self.arch_view.interaction_started.connect(self.begin_interaction)
        self.arch_view.interaction_finished.connect(self.end_interaction)
        self.arch_view.scrolled.connect(lambda steps: self.scroll_plane("Axial", steps))
        self.arch_view.reference_translated.connect(lambda info: self.translate_planes("Axial", info))
        self.arch_view.reference_rotated.connect(lambda angle: self.rotate_planes("Axial", angle))
        self.pano_view = self.new_slice("Panoramic reformat", "#d8b78a")
        self.pano_view.picked.connect(self.panorama_picked)
        self.configure_panorama_navigation(self.pano_view)
        self.pano_sagittal_view = self.new_slice("Sagittal · Arch locator", "#e5858b")
        self.pano_sagittal_view.picked.connect(lambda info: self.slice_picked(info, "Sagittal"))
        self.pano_sagittal_view.scrolled.connect(lambda steps: self.scroll_plane("Sagittal", steps))
        self.pano_sagittal_view.reference_translated.connect(lambda payload: self.translate_planes("Sagittal", payload))
        self.pano_sagittal_view.reference_rotated.connect(lambda degrees: self.rotate_planes("Sagittal", degrees))
        self.pano_sagittal_view.interaction_started.connect(self.begin_interaction)
        self.pano_sagittal_view.interaction_finished.connect(self.end_interaction)
        self.pano_sagittal_view.range_bounds_changed.connect(self.apply_panorama_range)
        grid.addWidget(self.pano_sagittal_view, 0, 0)
        grid.addWidget(self.arch_view, 0, 1)
        grid.addWidget(self.pano_view, 1, 0)
        self._volume_slots[1] = (grid, (1, 1, 1, 1))
        for i in range(2):
            grid.setRowStretch(i, 1)
            grid.setColumnStretch(i, 1)
        layout.addLayout(grid, 1)
        self.pano_index = QSlider(Qt.Orientation.Horizontal)
        self.pano_index.setRange(0, 511)
        self.pano_index.setValue(256)
        self.pano_index.valueChanged.connect(self.panorama_index_changed)
        layout.addWidget(self.pano_index)
        self.tabs.addTab(page, "Panoramic")

    def _build_implants(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QGroupBox("Generic implant")
        controls_layout = QVBoxLayout(controls)
        tip = QLabel("Place an implant on a section, then drag its center or end handles with Move implant. Review canal tracing before interpreting clearance.")
        tip.setWordWrap(True)
        tip.setObjectName("subtitle")
        controls_layout.addWidget(tip)
        bar = QFormLayout()
        self.implant_diameter = spin(4, 1, 10, 0.1)
        self.implant_length = spin(10, 3, 30, 0.5)
        self.implant_tilt = spin(0, 0, 180, 1)
        self.implant_yaw = spin(0, -180, 180, 1)
        for item in (self.implant_diameter, self.implant_length):
            item.editingFinished.connect(self.implant_dimensions_edited)
        for item in (self.implant_tilt, self.implant_yaw):
            item.editingFinished.connect(lambda: self.implant_dimensions_edited(orientation=True))
        self.implant_diameter.setSuffix(" mm")
        self.implant_length.setSuffix(" mm")
        for label, item in (("Diameter", self.implant_diameter), ("Length", self.implant_length), ("Tilt °", self.implant_tilt), ("Azimuth °", self.implant_yaw)):
            bar.addRow(label, item)
        controls_layout.addLayout(bar)
        actions = QGridLayout()
        for index, (text, callback) in enumerate((
            ("Place on image", self.begin_implant_placement),
            ("At crosshair", self.add_implant),
            ("Update selected", self.edit_implant),
            ("Remove", self.delete_implant),
            ("Move implant", lambda: self.tool_combo.setCurrentText("Move implant")),
        )):
            actions.addWidget(button(text, callback, index == 0), index // 2, index % 2)
        controls_layout.addLayout(actions)
        self.implant_list = QListWidget()
        self.implant_list.setMaximumHeight(100)
        self.implant_list.currentRowChanged.connect(self.select_implant)
        controls_layout.addWidget(self.implant_list)
        self.planning_panel.layout().insertWidget(0, controls)
        section_options = QHBoxLayout()
        from .ui_shell import build_section_grid_controls
        build_section_grid_controls(self, section_options)
        section_options.addWidget(button("Auto arch", self.auto_arch))
        section_options.addWidget(button("Plan implant", self.open_planning, True))
        layout.addLayout(section_options)
        section_navigation = QHBoxLayout()
        self.build_implant_arch_slider(section_navigation)
        layout.addLayout(section_navigation)
        self.implant_display_tabs = QTabWidget()
        axes_page = QWidget()
        views = QHBoxLayout(axes_page)
        self.implant_views = [self.new_slice("Longitudinal", "#6bc7a3"), self.new_slice("Orthogonal", "#6bc7a3"), self.new_slice("Transverse", "#6bc7a3")]
        for view in self.implant_views:
            view.picked.connect(lambda info: self.slice_picked(info, "Axial"))
            views.addWidget(view)
        self.implant_display_tabs.addTab(axes_page, "Implant axes")
        serial_page = QScrollArea()
        serial_page.setWidgetResizable(True)
        serial_content = QWidget()
        serial_page.setWidget(serial_content)
        self.implant_serial_layout = QGridLayout(serial_content)
        self.implant_serial_layout.setContentsMargins(0, 0, 0, 0)
        self.implant_serial_views = []
        self.implant_display_tabs.addTab(serial_page, "Serial sections")
        self.implant_display_tabs.setCurrentIndex(1)
        self.implant_display_tabs.currentChanged.connect(self.schedule_render)
        self.rebuild_implant_sections()
        implant_grid = QGridLayout()
        self.implant_arch_view = self.new_slice("Axial · Implant arch", "#71a9f7")
        self.implant_arch_view.picked.connect(self.implant_locator_picked)
        self.implant_arch_view.point_moved.connect(self.move_arch_point)
        self.implant_arch_view.point_removed.connect(self.remove_arch_point)
        self.implant_arch_view.scrolled.connect(lambda steps: self.scroll_plane("Axial", steps))
        self.implant_arch_view.reference_translated.connect(lambda info: self.translate_planes("Axial", info))
        self.implant_arch_view.reference_rotated.connect(lambda angle: self.rotate_planes("Axial", angle))
        self.implant_arch_view.interaction_started.connect(self.begin_interaction)
        self.implant_arch_view.interaction_finished.connect(self.end_interaction)
        self.implant_pano_view = self.new_slice("Panoramic · Implant location", "#d8b78a")
        self.implant_pano_view.picked.connect(self.panorama_picked)
        self.configure_panorama_navigation(self.implant_pano_view)
        implant_grid.addWidget(self.implant_arch_view, 0, 0)
        implant_grid.addWidget(self.implant_pano_view, 0, 1)
        implant_grid.addWidget(self.implant_display_tabs, 1, 0)
        self._volume_slots[2] = (implant_grid, (1, 1, 1, 1))
        for index in range(2):
            implant_grid.setRowStretch(index, 1)
            implant_grid.setColumnStretch(index, 1)
        layout.addLayout(implant_grid, 1)
        self.tabs.addTab(page, "Implant view")

    def _build_tmj(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        tip = QLabel("Mark each condylar center on the axial locator. Drag its center to move; drag the round handle or right-drag its axis to rotate. Centers are projected onto the locator; each retains its own depth.")
        tip.setWordWrap(True)
        tip.setObjectName("subtitle")
        layout.addWidget(tip)
        bar = QHBoxLayout()
        self.tmj_angles = {}
        bar.addWidget(button("Auto locate TMJ", self.auto_tmj))
        for name in ("Right", "Left"):
            bar.addWidget(button(f"Mark {name.lower()} center", lambda checked=False, n=name: self.begin_tmj(n)))
            bar.addWidget(QLabel(f"{name} angle °"))
            item = spin(0, -180, 180, 1)
            item.valueChanged.connect(self.schedule_render)
            self.tmj_angles[name] = item
            bar.addWidget(item)
        self.tmj_offset = spin(0, -30, 30, 0.5)
        self.tmj_offset.valueChanged.connect(self.schedule_render)
        bar.addWidget(QLabel("Section offset mm"))
        bar.addWidget(self.tmj_offset)
        layout.addLayout(bar)
        grid = QGridLayout()
        from .tmj_locator import TMJLocatorView
        self.tmj_locator = TMJLocatorView()
        self.all_slices.append(self.tmj_locator)
        self.tmj_locator.window_dragged.connect(self.drag_window)
        self.tmj_locator.measured.connect(self.statusBar().showMessage)
        self.tmj_locator.snapshot_requested.connect(lambda: self.snapshot_section(self.tmj_locator))
        self.tmj_locator.picked.connect(lambda info: self.slice_picked(info, "Axial"))
        self.tmj_locator.scrolled.connect(lambda steps: self.scroll_plane("Axial", steps))
        self.tmj_locator.center_changed.connect(self.move_tmj_center)
        self.tmj_locator.angle_changed.connect(lambda side, angle: self.tmj_angles[side].setValue(angle))
        self.tmj_locator.interaction_started.connect(self.begin_interaction)
        self.tmj_locator.interaction_finished.connect(self.end_interaction)
        grid.addWidget(self.tmj_locator, 0, 1)
        self._volume_slots[3] = (grid, (1, 1, 1, 1))
        self.tmj_views = {}
        for col, name in enumerate(("Right", "Left")):
            for row, plane in enumerate(("Sagittal", "Coronal")):
                view = self.new_slice(f"{name} TMJ · {plane}", "#b9a5d6")
                self.tmj_views[(name, plane)] = view
                self.wire_tmj_section(view, name, plane)
                grid.addWidget(view, row, col * 2)
        for i in range(2):
            grid.setRowStretch(i, 1)
        for i in range(3):
            grid.setColumnStretch(i, 1)
        layout.addLayout(grid, 1)
        self.tabs.addTab(page, "Bilateral TMJ")

    def renderer_status(self, text):
        self.gpu_label.setText(text)

    def set_tool(self, text):
        physical_tools = {"Measure", "Correct planes", "Place implant", "Move implant", "Mark right TMJ", "Mark left TMJ", "Paint mask", "Erase mask", "Edit canal path"}
        if self.volume is not None and not self.volume.metadata.get("geometry_calibrated", True) and text in physical_tools:
            self.tool_combo.setCurrentText("Navigate")
            self.statusBar().showMessage("Physical tools require calibrated DICOM geometry")
            return
        if self.mode in ("Paint mask", "Erase mask") and text != self.mode:
            self.finish_mask_stroke(reset_pointer=True)
        if self.mode == "Edit canal path" and text != self.mode:
            self.finish_path_gesture(reset_pointer=True)
        if self.mode == "Move implant" and text != self.mode:
            self.finish_implant_gesture()
        self.mode = text
        if hasattr(self.volume_view, "set_implant_editing"):
            index = self.implant_list.currentRow()
            key = self.implants[index]["id"] if 0 <= index < len(self.implants) else None
            self.volume_view.set_implant_editing(text == "Move implant", key)
        if hasattr(self, "rail_group"):
            self.rail_group.setExclusive(False)
            for item in self.rail_group.buttons():
                item.setChecked(item.property("mode") == text)
            self.rail_group.setExclusive(True)
        for view in self.all_slices:
            view.mode = text
            view.update()
        self.sync_arch_mode()
        if hasattr(self, "segmentation_panel"):
            self.segmentation_panel.sync_tool(text)
        self.refresh_nerve_paths()

    def sync_arch_mode(self, *_):
        if not hasattr(self, "arch_view"):
            return
        self.arch_view.mode = self.mode if self.mode in ("Pan", "Window / level", "Measure") else ("Add arch point" if self.arch_toggle.isChecked() else "Edit arch")
        self.arch_view.update()

    def open_folder(self):
        if self.job is not None:
            self.statusBar().showMessage("Wait for the current operation or cancel it first")
            return
        folder = QFileDialog.getExistingDirectory(self, "Open DICOM study folder")
        if folder:
            self.scan_folder(folder)

    def scan_folder(self, folder):
        self.inspector.show()
        self.inspector_tabs.setCurrentIndex(2)
        self.start_job(lambda progress, cancel: discover_series(folder, progress=progress, cancel=cancel), self.discovered)

    def start_job(self, function, callback):
        if self.job is not None or self._closing:
            return False
        self.job = Job(function, self)
        self.job.result_callback = callback
        self.job.result.connect(self.job_result)
        self.job.error.connect(self.job_error)
        self.job.progress.connect(self.job_progress)
        self.job.finished.connect(self.job_finished)
        self.load_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.cancel_button.show()
        self.job.start()
        return True

    def job_result(self, result):
        worker = self.sender()
        if worker is self.job and not worker.cancelled and not self._closing:
            worker.result_callback(result)

    def job_progress(self, percent, message):
        self.progress_bar.setRange(0, 100 if percent > 0 else 0)
        self.progress_bar.setValue(percent)
        self.statusBar().showMessage(message)
        if self._report_analysis_active:
            self.reporting_panel.set_analysis_status(message, progress=percent, busy=True)
        if self.segmentation_panel._busy:
            self.segmentation_panel.activity.setText(message)
        if self.nerve_panel._busy:
            self.nerve_panel.activity.setText(message)

    def job_error(self, message):
        self.pano_refresh_timer.stop()
        if self._report_analysis_active:
            self.reporting_panel.set_analysis_status(
                "Local review canceled." if self.job and self.job.cancelled else message, busy=False)
            return
        if self.job and self.job.cancelled:
            self.statusBar().showMessage("Operation canceled")
            return
        QMessageBox.warning(self, "Unable to complete operation", message)

    def job_finished(self):
        self.progress_bar.hide()
        self.cancel_button.hide()
        self.segmentation_panel.set_busy(False)
        self.nerve_panel.set_busy(False)
        self.reporting_panel.set_busy(False)
        self.load_button.setEnabled(bool(self.series))
        old = self.sender()
        if self._report_analysis_active:
            if old and old.cancelled:
                self.reporting_panel.set_analysis_status("Local review canceled; no partial result was installed.", busy=False)
            elif not self._report_analysis_completed:
                self.reporting_panel.set_analysis_status("Local review did not complete. Existing findings were preserved.", busy=False)
            self._report_analysis_active = False
        if old is self.job:
            self.job = None
        if old:
            cancelled = old.cancelled
            old.result_callback = None
            old.deleteLater()
            if not cancelled:
                QTimer.singleShot(0, self.finish_pending_auto_panorama)
            else:
                self._auto_pano_pending = None
                self.clear_pending_automatic_setup()

    def cancel_job(self):
        self.pano_refresh_timer.stop()
        self.clear_pending_automatic_setup()
        if self.job:
            self.job.cancelled = True
            self.statusBar().showMessage("Cancel requested; waiting for the current operation to stop")

    def discovered(self, series):
        self.series = series
        self.series_list.clear()
        for item in series:
            self.series_list.addItem(item.label)
        if series:
            self.series_list.setCurrentRow(0)
            self.statusBar().showMessage(f"Found {len(series)} series. Select one to load.")
            if len(series) == 1:
                QTimer.singleShot(150, self.load_selected)
        else:
            self.statusBar().showMessage("No supported image series found in this folder")

    def load_selected(self, *_):
        row = self.series_list.currentRow()
        if 0 <= row < len(self.series):
            item = self.series[row]
            self.start_job(lambda progress, cancel: load_series(item, progress=progress, cancel=cancel), self.set_volume)

    def load_demo(self):
        if self.job is not None:
            self.statusBar().showMessage("Wait for the current operation or cancel it first")
            return
        self.set_volume(make_demo())

    def set_volume(self, volume):
        if self._maximized_view is not None:
            self.maximize_view(self._maximized_view)
        self.reset_workspace_splitters()
        self.close_section_grids()
        self.finish_implant_gesture()
        self.finish_mask_stroke()
        self.finish_path_gesture(reset_pointer=True)
        self.nerve_paths = None
        self.nerve_panel.clear()
        self.segmentation = None
        self.segmentation_panel.clear()
        self.slice_revision += 1
        self.slice_queue.invalidate()
        self.slice_pending = None
        if self.slice_job:
            self.slice_job.cancelled = True
        self.volume = volume
        for control in (self.slab, self.header_thickness, self.pano_thickness):
            control.configure_for_spacing(volume.spacing)
            control.setValue(min(volume.spacing))
        with QSignalBlocker(self.pano_offset):
            self.pano_offset.setValue(0)
        self.pano_offset.setSingleStep(float(min(volume.spacing)))
        self.segmentation_panel.refresh_runtime()
        self.landmark_panel.clear()
        self._rotation_override = None
        self._mpr_sizes = {name: self.plane_size(*plane_frame(name, np.eye(3)))
                           for name in ("Axial", "Coronal", "Sagittal")}
        self.tool_combo.setCurrentText("Navigate")
        self.crosshair = np.array(volume.center, dtype=float)
        self.implants.clear()
        self.implant_margins.clear()
        self.schedule_clearance()
        self.reset_workflow_state()
        self.implant_list.clear()
        self.clear_arch()
        self.tmj_centers = {"Right": None, "Left": None}
        self._tmj_section_corrections = {}
        self.slab.setValue(0)
        self.slab_mode.setCurrentText("mean")
        self.tmj_offset.setValue(0)
        self.implant_section_offset.setRange(-1000, 1000)
        self.implant_section_offset.setValue(0)
        for item in self.tmj_angles.values():
            item.setValue(0)
        for item in self.rotation_spins:
            item.blockSignals(True)
            item.setValue(0)
            item.blockSignals(False)
        for view in self.all_slices:
            view.calibrated = volume.metadata.get("geometry_calibrated", True)
            view.measurements.clear()
            view.pending_measure = None
            view.points = []
            view.path_points = []
            view.set_nerve_paths([])
            view.landmarks = []
            view.set_segmentation(None)
            view.selected_point = None
            view.implants = []
            view.reset_view()
            view.slab_mm = view.slab_mode = None
            view.thickness_presets_spacing = float(min(volume.spacing))
            if hasattr(view, "serial_grid_width"):
                del view.serial_grid_width
            view.serial_grid_settings = (3, 3, 1.)
            view._grid_source_frame = None
            view.data = view.image = None
            view.frame = None
            view.note = ""
            view.image_key = None
            view.update()
        self.window_level.blockSignals(True)
        self.window_width.blockSignals(True)
        self.window_level.setValue(volume.window[0])
        self.window_width.setValue(max(1, volume.window[1]))
        self.window_level.blockSignals(False)
        self.window_width.blockSignals(False)
        self.window_preset.blockSignals(True)
        self.window_preset.setCurrentIndex(0)
        # HU presets remain visible but are disabled if no HU calibration is declared.
        for i in (1, 2, 3):
            self.window_preset.model().item(i).setEnabled(volume.is_hu)
        self.window_preset.blockSignals(False)
        self.threshold.blockSignals(True)
        self.threshold.setValue(0)
        self.threshold.blockSignals(False)
        self.volume_view.set_volume(volume)
        self.volume_view.set_preset(self.render_preset.currentText())
        self.volume_view.set_quality(self.render_quality.currentText())
        self.volume_view.set_view_orientation("Isometric")
        self.volume_view.set_implants([])
        low, high = float(volume.data_min), float(volume.data_max)
        level, width = self.window_level.value(), self.window_width.value()
        self.brightness.set_limits(min(low, level - 1), max(high, level + 1))
        self.contrast.set_limits(1, max(2, high - low, width * 2))
        self.refresh_references()
        self.update_window()
        meta = volume.metadata
        description = meta.get("series_description", meta.get("SeriesDescription", "DICOM volume"))
        synthetic = meta.get("synthetic", False) or "synthetic" in str(description).lower()
        if synthetic:
            description = "SYNTHETIC DEMO · geometric phantom, not a patient scan"
        sx, sy, sz = volume.spacing
        z, y, x = volume.data.shape
        spacing_text = f"{sx:g} × {sy:g} × {sz:g} mm" if volume.metadata.get("geometry_calibrated", True) else "Uncalibrated geometry"
        identity = " · ".join(value for value in (
            patient_display_name(meta.get("patient_name")), str(meta.get("patient_id") or "")
        ) if value)
        if identity:
            description = identity + "  /  " + str(description)
        full_info = f"{description}  |  {x} × {y} × {z}  |  {spacing_text}  |  {volume.data.nbytes / 2**20:.0f} MiB  |  {'HU declared' if volume.is_hu else 'Uncalibrated intensity'}"
        title = "SYNTHETIC DEMO · Dental phantom" if synthetic else str(description)
        self.study_label.setText(title)
        self.study_label.setToolTip(full_info)
        self.update_acquisition_info()
        self.tabs.setCurrentIndex(0)
        self._initial_view = self.capture_view_defaults()
        self.schedule_render()
        warning = " • ".join(volume.warnings)
        self.statusBar().showMessage(warning or "Volume loaded · linked patient-coordinate sections ready")

    def rotation(self):
        if self._rotation_override is not None:
            return self._rotation_override.copy()
        return rotation_matrix(*(item.value() for item in self.rotation_spins))

    def open_segmentation(self):
        self.inspector.show()
        self.inspector_tabs.setCurrentWidget(self.review_page)
        self.review_tabs.setCurrentIndex(0)
        self.segmentation_panel.refresh_runtime()

    def maximize_view(self, target):
        if target not in self.mpr_layout:
            return
        self.finish_mask_stroke(reset_pointer=True)
        self._maximized_view = None if self._maximized_view is target else target
        if self.apply_explorer_maximize(self._maximized_view):
            self.sync_volume_geometry()
            self._volume_geometry_timer.start(0)
            self.refresh_segmentation()
            self.schedule_render()
            return
        for widget in self.mpr_layout:
            self.mpr_grid.removeWidget(self._volume_placeholders[0] if widget is self.volume_card else self.section_layout_widget(widget))
        for widget, (row, column) in self.mpr_layout.items():
            show = self._maximized_view is None or widget is self._maximized_view
            layout_widget = self._volume_placeholders[0] if widget is self.volume_card else self.section_layout_widget(widget)
            widget.setVisible(show and (widget is self.volume_card or layout_widget is widget))
            layout_widget.setVisible(show)
            if self._maximized_view is widget:
                self.mpr_grid.addWidget(layout_widget, 0, 0, 2, 2)
            else:
                self.mpr_grid.addWidget(layout_widget, row, column)
        self.sync_volume_geometry()
        self._volume_geometry_timer.start(0)
        self.refresh_segmentation()
        self.schedule_render()

    def segment_scan(self):
        if self.volume is None or self.job is not None:
            self.statusBar().showMessage("Load a scan and wait for the current operation first")
            return
        if not self.volume.metadata.get("geometry_calibrated", True) or min(self.volume.dimensions) < 2:
            self.statusBar().showMessage("Automatic segmentation requires calibrated 3D CT/CBCT geometry")
            return
        if self.volume.metadata.get("modality", "").upper() not in ("CT", "CBCT"):
            self.statusBar().showMessage("Open a CT/CBCT study for this model; the synthetic demo is a navigation example")
            return
        from .inference import run_segmentation
        self.open_segmentation()
        self.finish_mask_stroke()
        self.tool_combo.setCurrentText("Navigate")
        volume, fingerprint = self.volume, self.volume_fingerprint()
        def infer(progress, cancel):
            result = run_segmentation(volume, fingerprint, progress=lambda message: progress(0, str(message)), cancel=cancel)
            return id(volume), result
        if self.start_job(infer, self.segmentation_ready):
            self.segmentation_panel.set_busy(True)
            self.segmentation_panel.activity.setText("Preparing source volume for local inference…")

    def segmentation_ready(self, payload):
        identity, result = payload
        if self.volume is None or id(self.volume) != identity or self._closing:
            return
        result.validate_source(self.volume, self.volume_fingerprint())
        self.finish_mask_stroke()
        self.segmentation = result
        self.tool_combo.setCurrentText("Navigate")
        self.segmentation_panel.install()
        self.open_segmentation()
        self.tabs.setCurrentIndex(0)
        self.volume_visible.setChecked(False)
        self.refresh_segmentation()
        self.statusBar().showMessage("Segmentation ready · review masks and correct them in the linked sections")

    def refresh_segmentation(self, *, update_3d=True):
        if not hasattr(self, "mpr_views"):
            return
        result = self.segmentation
        panel = self.segmentation_panel
        for view in self.all_slices:
            if self._maximized_view is not None and view in self.mpr_views.values() and view is not self._maximized_view:
                continue
            if (result is None or view.frame is None or view.data is None
                    or not panel.show_2d.isChecked() or not panel.visible_ids):
                view.set_segmentation(None)
            elif view.isVisible() or view in self.mpr_views.values():
                frame, shape = view.frame, view.data.shape
                preview = self._mask_previous is not None
                if preview:
                    scale = min(1., 224 / max(shape))
                    reduced = tuple(max(2, round(count * scale)) for count in shape)
                    # Both displays occupy identical pixel-edge extents, even
                    # when preview and anatomy have different sample counts.
                    width, height = view.edge_size_mm()
                    size = (width * (reduced[1] - 1) / reduced[1], height * (reduced[0] - 1) / reduced[0])
                    frame, shape = (*frame[:3], size), reduced
                plane = result.sample_plane(self.volume, frame, shape)
                view.set_segmentation(plane, result.labels, panel.visible_ids, panel.opacity.value(), preview=preview)
        if update_3d:
            if result is None:
                self.volume_view.set_segmentation(None)
            else:
                self.volume_view.set_segmentation(result.data, result.labels, panel.visible_ids, visible=panel.show_3d.isChecked())

    def paint_mask(self, payload):
        phase = payload["phase"]
        if phase == "end":
            self.finish_mask_stroke()
            return
        result = self.segmentation
        key = self.segmentation_panel.selected_label
        if result is None or key is None or self.tabs.currentIndex() != 0 or self.mode not in ("Paint mask", "Erase mask"):
            if phase == "start":
                self.statusBar().showMessage("Select a segmentation object before painting")
            return
        point = np.asarray(payload["world"])
        if phase == "start":
            self.finish_mask_stroke()
            result.begin_stroke()
            self._mask_previous = point.copy()
        if self._mask_previous is None:
            return
        radius = self.segmentation_panel.radius.value()
        distance = float(np.linalg.norm(point - self._mask_previous))
        steps = max(1, int(np.ceil(distance / max(radius * .5, .1))))
        _, right, up, _ = payload["frame"]
        for fraction in np.linspace(0, 1, steps + 1)[1:]:
            center = self._mask_previous + fraction * (point - self._mask_previous)
            changed = result.paint(self.volume, center, right, up, radius, key,
                erase=payload["erase"], overwrite=self.segmentation_panel.overwrite.isChecked())
            self._mask_changed |= bool(changed)
        self._mask_previous = point.copy()
        if not self.mask_timer.isActive():
            self.mask_timer.start()

    def finish_mask_stroke(self, *, reset_pointer=False):
        self.mask_timer.stop()
        had_stroke = self._mask_previous is not None
        # Settle at full overlay detail before handing changed data to VTK.
        self._mask_previous = None
        if self.segmentation is not None:
            self.segmentation.end_stroke()
            if self._mask_changed:
                self.segmentation_panel.refresh()
                self.volume_view.segmentation_modified()
                self.statusBar().showMessage("Mask corrected · affected objects returned to pending review")
            if had_stroke:
                self.refresh_segmentation(update_3d=self._mask_changed)
        self._mask_changed = False
        if reset_pointer:
            for view in getattr(self, "all_slices", []):
                if view._drag_kind == "mask":
                    view._drag_kind = view.last_pos = view._drag_world = None

    def undo_mask(self):
        if self.segmentation is None:
            return
        self.finish_mask_stroke()
        self.segmentation.undo()
        self.segmentation_panel.refresh()
        self.volume_view.segmentation_modified()
        self.refresh_segmentation()

    def focus_segment(self):
        result, key = self.segmentation, self.segmentation_panel.selected_label
        if result is None or key is None or self.job is not None:
            return
        self.finish_mask_stroke()
        self.tool_combo.setCurrentText("Navigate")
        volume = self.volume
        def locate(progress, cancel):
            return id(result), result.label_centroid(volume, key)
        def located(payload):
            identity, center = payload
            if self.segmentation is not result or identity != id(result):
                return
            if center is None:
                self.statusBar().showMessage("No voxels were found for this object")
                return
            self.crosshair = np.asarray(center)
            self.tabs.setCurrentIndex(0)
            self.refresh_references()
            self.schedule_render()
        self.start_job(locate, located)

    def workflow_changed(self, index):
        if index != 0 and self.mode in ("Paint mask", "Erase mask", "Edit canal path"):
            self.tool_combo.setCurrentText("Navigate")
        self.refresh_nerve_paths()
        self.place_shared_volume(index)
        if index == getattr(self, "superimposition_index", -1):
            if self.comparison.moving is None and not self.comparison._busy:
                QTimer.singleShot(0, self.comparison.begin_workflow)
            return
        self.request_automatic_workflow(index)
        if index in (1, 2) and len(self.arch_points) >= 3 and self.pano_result is None:
            self.pano_refresh_timer.start()

    def refresh_landmarks(self):
        if not hasattr(self, "mpr_views"):
            return
        panel = getattr(self, "landmark_panel", None)
        result = panel.result if panel is not None else None
        points = [] if result is None else [item for item in result["landmarks"] if item["review"] != "rejected"]
        for view in self.mpr_views.values():
            view.landmarks = points
            view.update()

    def orientation_changed(self, *_, preserve=False):
        self.finish_path_gesture(reset_pointer=True)
        if self._mask_previous is not None:
            self.finish_mask_stroke(reset_pointer=True)
        if not preserve:
            self._rotation_override = None
        if self.arch_points or self.pano_result is not None:
            old_up = self._arch_up if self._arch_up is not None else self.pano_up
            if not np.allclose(self.rotation()[:, 2], old_up, atol=1e-8, rtol=0):
                self.clear_arch()
                self.statusBar().showMessage("Axial plane changed. Redraw the arch on the corrected plane.")
        self.schedule_render()
        self.refresh_references()

    def begin_interaction(self):
        self._wheel_settle.stop()
        self._interaction_active = True
        for view in self.all_slices:
            view.pending_measure = None

    def end_interaction(self):
        self._interaction_active = False
        self.schedule_render()

    def refresh_references(self, *_):
        if self.volume is None or self.crosshair is None or not hasattr(self, "volume_view"):
            return
        rotation = self.rotation()
        spatial = self.volume.metadata.get("geometry_calibrated", True) and min(self.volume.dimensions) > 1
        planar_views = [*self.mpr_views.items(), ("Sagittal", self.pano_sagittal_view),
                        ("Axial", self.arch_view), ("Axial", self.implant_arch_view)]
        for name, view in planar_views:
            view.crosshair = self.crosshair.copy()
            view.reference_lines = plane_references(name, rotation) if spatial else []
            view.show_references = self.lines_visible.isChecked()
            view.update()
        self.volume_view.set_reference_planes(self.crosshair, rotation,
            visible=self.volume.metadata.get("geometry_calibrated", True),
            enabled_planes={name: item.isChecked() for name, item in self.plane_toggles.items()})
        if self.pano_result is not None:
            self.update_panorama_cursor()

    def translate_planes(self, view_name, payload):
        if self.volume is None:
            return
        candidate = translate_reference(self.crosshair, payload["delta"], view_name, payload["target"], self.rotation())
        self.crosshair = self.constrain_position(self.crosshair, candidate)
        self.refresh_references()
        self.schedule_render()

    def constrain_position(self, position, candidate):
        """Stop at the acquired box along the requested physical motion."""
        index = self.volume.world_to_index(candidate)
        # Limit the motion along its own vector rather than independently
        # clipping coordinates, which would move the viewed plane at obliquity.
        start = self.volume.world_to_index(position)
        delta = index - start
        fraction = 1.0
        for coordinate, change, count in zip(start, delta, self.volume.dimensions):
            if change > 0:
                fraction = min(fraction, (count - 1 - coordinate) / change)
            elif change < 0:
                fraction = min(fraction, -coordinate / change)
        return position + (candidate - position) * max(0., fraction)

    def rotate_planes(self, view_name, degrees):
        if self.volume is None:
            return
        corrected = rotate_about_view(self.rotation(), view_name, degrees)
        self._rotation_override = corrected
        for item, value in zip(self.rotation_spins, to_euler(corrected)):
            item.blockSignals(True)
            item.setValue(value)
            item.blockSignals(False)
        self.orientation_changed(preserve=True)

    def reset_orientation(self):
        if self.volume is None:
            return
        self._rotation_override = None
        for item in self.rotation_spins:
            item.blockSignals(True)
            item.setValue(0)
            item.blockSignals(False)
        self.orientation_changed(preserve=True)

    def reset_workspace(self):
        if self.volume is None:
            return
        self.slice_queue.invalidate()
        self.slice_pending = None
        if self.slice_job:
            self.slice_job.cancelled = True
        self.finish_implant_gesture()
        self.finish_mask_stroke(reset_pointer=True)
        self.finish_path_gesture(reset_pointer=True)
        self._wheel_settle.stop()
        self._interaction_active = False
        self._auto_setup_epoch = getattr(self, "_auto_setup_epoch", 0) + 1
        self.clear_pending_automatic_setup()
        self.close_section_grids()
        if getattr(self, "_section_dialog", None) is not None:
            self._section_dialog.close()
        if self._maximized_view is not None:
            self.maximize_view(self._maximized_view)
        self._volume_maximized = False
        self.reset_workspace_splitters()
        self.tabs.setCurrentIndex(0)
        self.implant_section_offset.setRange(-1000, 1000)
        for name, value in getattr(self, "_initial_view", {}).items():
            control = getattr(self, name)
            with QSignalBlocker(control):
                if isinstance(control, QComboBox):
                    control.setCurrentIndex(value)
                elif isinstance(control, QCheckBox):
                    control.setChecked(value)
                else:
                    control.setValue(value)
        self.header_thickness.setValue(self.slab.value())
        self.rebuild_implant_sections()
        self.tool_combo.setCurrentText("Navigate")
        self.reset_orientation()
        self.clear_arch()
        self.auto_setup_provenance = {}
        self.tmj_centers = {"Right": None, "Left": None}
        self._tmj_section_corrections = {}
        for item in self.tmj_angles.values():
            with QSignalBlocker(item):
                item.setValue(0)
        self.crosshair = np.array(self.volume.center)
        for view in self.all_slices:
            view.reset_view()
            view.slab_mm = view.slab_mode = None
            if hasattr(view, "serial_grid_width"):
                del view.serial_grid_width
            view.serial_grid_settings = (3, 3, 1.)
            view._grid_source_frame = None
            view.image_key = None
        for toggle in self.plane_toggles.values():
            toggle.setChecked(True)
        self.volume_view.set_preset(self.render_preset.currentText())
        self.volume_view.set_quality(self.render_quality.currentText())
        self.volume_view.set_threshold(self.threshold.value())
        self.volume_view.set_volume_visibility(self.volume_visible.isChecked())
        self.volume_view.reset_camera()
        self.update_window()
        self.sync_volume_geometry()
        self.refresh_references()
        self.schedule_render()
        self.statusBar().showMessage("Initial scan view restored · findings, masks, nerve paths and implants retained")

    def capture_view_defaults(self):
        """Small viewing-state snapshot; never copies the source or review objects."""
        names = ("window_level", "window_width", "window_preset", "slab", "slab_mode",
                 "resolution", "render_preset", "render_quality", "threshold",
                 "volume_visible", "lines_visible", "pano_thickness", "pano_projection",
                 "implant_section_rows", "implant_section_columns", "implant_section_count",
                 "implant_section_spacing", "implant_section_width", "implant_section_offset",
                 "pano_offset", "tmj_offset", "arch_toggle")
        result = {}
        for name in names:
            control = getattr(self, name)
            result[name] = (control.currentIndex() if isinstance(control, QComboBox)
                            else control.isChecked() if isinstance(control, QCheckBox)
                            else control.value())
        return result

    def schedule_render(self, *_):
        if not self._closing and not self.render_timer.isActive():
            self.render_timer.start()

    def frame_size(self):
        dims = np.array(self.volume.data.shape[::-1]) - 1
        extent = dims * self.volume.spacing
        return float(max(np.linalg.norm(extent) * 1.02, 25))

    def plane_size(self, right, up):
        from itertools import product
        corners = self.volume.index_to_world(np.array(list(product(*[(0, n - 1) for n in self.volume.dimensions]))))
        width = max(float(np.ptp(corners @ right)) * 1.04, 20)
        height = max(float(np.ptp(corners @ up)) * 1.04, 20)
        return width, height

    def draw_plane(self, view, center, right, up, size, note="", shape=None, crosshair=None):
        # Reconstruct to the actual display, including DPI and magnification.
        # Explicit legacy dimensions must not limit a maximized or zoomed pane.
        quality = (.75, 1., 1.5)[self.resolution.currentIndex()]
        shape = view.preferred_sample_shape(size, quality=quality)
        frame = (np.array(center), np.array(right), np.array(up), size)
        if self.section_layout_widget(view) is not view:
            view._grid_source_frame = frame
            return
        if getattr(self, "_plane_targets", None) is not None:
            self._plane_targets.append(id(view))
        cursor = self.crosshair.copy() if crosshair is None else np.array(crosshair)
        self.sync_section_position(view, frame, cursor)
        slab = self.slab.value() if view.slab_mm is None else view.slab_mm
        mode = view.slab_mode or self.slab_mode.currentText()
        note = f"{note + ' · ' if note else ''}Slab {slab:g} mm · {mode}"
        key = (id(self.volume), tuple(np.round(np.r_[frame[0], frame[1], frame[2], size], 9)), shape, slab, mode)
        if view.image_key == key:
            view.crosshair = cursor
            view.note = note
            view.update()
            return
        self._plane_requests.append((view, frame, cursor, note, shape, key, slab, mode))

    def render_views(self):
        if self.volume is None or self._closing:
            return
        self._plane_requests = []
        self._plane_targets = []
        rotation = self.rotation()
        size = self.frame_size()
        tab = self.tabs.currentIndex()
        try:
            if tab == 0:
                for name, view in self.mpr_views.items():
                    if self._maximized_view is not None and self._maximized_view is not view:
                        continue
                    right, up = plane_frame(name, rotation)
                    normal = np.cross(right, up)
                    center = self.volume.center + normal * np.dot(self.crosshair - self.volume.center, normal)
                    self.draw_plane(view, center, right, up, self._mpr_sizes[name])
            elif tab == 1:
                right, up = plane_frame("Sagittal", rotation)
                normal = np.cross(right, up)
                center = self.volume.center + normal * np.dot(self.crosshair - self.volume.center, normal)
                self.draw_plane(self.pano_sagittal_view, center, right, up, self._mpr_sizes["Sagittal"])
                right, up = plane_frame("Axial", rotation)
                normal = np.cross(right, up)
                center = self.volume.center + normal * np.dot(self.crosshair - self.volume.center, normal)
                self.arch_view.points = self.arch_points
                self.arch_view.path_points = [] if self.pano_result is None else self.pano_result.path
                self.sync_arch_mode()
                self.draw_plane(self.arch_view, center, right, up, self._mpr_sizes["Axial"], f"{len(self.arch_points)} arch points · scroll to choose axial level")
            elif tab == 2:
                self.render_implant_locator()
                self.render_implant_views()
            elif tab == 3:
                self.render_tmj_views()
        except (ValueError, RuntimeError) as exc:
            self.statusBar().showMessage(str(exc))
            return
        self.render_section_grids()
        self.slice_revision += 1
        self.refresh_references()
        context = (id(self.volume), tab, tuple(self._plane_targets))
        if self._plane_requests:
            self.slice_queue.request((self.volume, self._plane_requests), context=context)
        else:
            self.slice_queue.invalidate(context)
        self.slice_pending = self.slice_queue.pending
        self._plane_requests = None
        self._plane_targets = None
        if self.slice_job and not self.slice_queue.active_is_current:
            self.slice_job.cancelled = True
        self.start_slice_job()

    def start_slice_job(self):
        if self.slice_job is not None or self.slice_pending is None or self._closing:
            return
        request = self.slice_queue.take()
        if request is None:
            return
        ticket, (volume, requests) = request.ticket, request.payload
        self.slice_pending = self.slice_queue.pending
        def sample(progress, cancelled):
            start = time.perf_counter()
            frames = []
            engine = FastReslicer(volume)
            for view, frame, cursor, note, shape, key, slab, mode in requests:
                if cancelled():
                    return ticket, [], 0
                center, right, up, size = frame
                image = engine.sample_plane(center, right, up, size_mm=size, shape=shape, slab_mm=slab, mode=mode, cancel=cancelled)
                frames.append((view, image, frame, cursor, note, key))
            return ticket, frames, (time.perf_counter() - start) * 1000
        self.slice_job = Job(sample, self)
        self.slice_job.ticket = ticket
        self.slice_job.result.connect(self.frames_ready)
        self.slice_job.error.connect(self.slice_error)
        self.slice_job.finished.connect(self.slice_finished)
        self.slice_job.start()

    def frames_ready(self, result):
        ticket, frames, elapsed = result
        if self._closing or not self.slice_queue.accepts(ticket, interacting=self._interaction_active):
            return
        for view, image, frame, cursor, note, key in frames:
            view.implants = self.implants
            view.set_image(image, frame, cursor, note)
            view.image_key = key
        self.last_frame_ms = elapsed
        self.refresh_segmentation(update_3d=False)
        self.refresh_nerve_paths()

    def slice_error(self, message):
        if self.slice_job and self.slice_job.cancelled:
            return
        self.statusBar().showMessage(message)

    def slice_finished(self):
        old = self.sender()
        if old is self.slice_job:
            self.slice_job = None
        if old:
            self.slice_queue.complete(old.ticket)
            old.deleteLater()
        self.slice_pending = self.slice_queue.pending
        self.start_slice_job()

    def slice_picked(self, info, plane):
        world = info.get("world")
        if world is None or self.volume is None:
            return
        index = self.volume.world_to_index(world)
        if np.any(index < -1e-4) or np.any(index > np.array(self.volume.dimensions) - 1 + 1e-4):
            self.statusBar().showMessage("Choose a point inside the acquired image volume")
            return
        if self.mode == "Place implant":
            self.crosshair = np.asarray(world)
            self.add_implant()
        elif self.mode.startswith("Mark "):
            side = "Right" if "right" in self.mode else "Left"
            self.tmj_centers[side] = np.asarray(world)
            self.tool_combo.setCurrentText("Navigate")
            self.statusBar().showMessage(f"{side} TMJ center marked · adjust the angle in Bilateral TMJ")
        else:
            self.crosshair = np.asarray(world)
        self.refresh_references()
        self.schedule_render()

    def scroll_plane(self, name, steps):
        if self.volume is None:
            return
        self.begin_interaction()
        self._wheel_settle.start()
        right, up = plane_frame(name, self.rotation())
        normal = np.cross(right, up)
        candidate = self.crosshair + normal * steps * min(self.volume.spacing)
        self.crosshair = self.constrain_position(self.crosshair, candidate)
        self.refresh_references()
        self.schedule_render()

    def update_window(self, *_):
        self.brightness.from_value()
        self.contrast.from_value()
        invert = False
        if self.volume:
            invert = bool(self.volume.metadata.get("photometric_interpretation", "") == "MONOCHROME1" or self.volume.metadata.get("invert", False))
        for view in self.all_slices:
            view.set_window(self.window_level.value(), self.window_width.value(), invert)

    def drag_window(self, dx, dy):
        width = self.window_width.value()
        self.window_width.blockSignals(True)
        self.window_level.blockSignals(True)
        self.window_width.setValue(max(1, width + dx * max(width * 0.006, 1)))
        self.window_level.setValue(self.window_level.value() + dy * max(width * 0.004, 1))
        self.window_width.blockSignals(False)
        self.window_level.blockSignals(False)
        self.update_window()

    def apply_window_preset(self, name):
        if self.volume is None:
            return
        if name == "Scan default":
            level, width = self.volume.window
        elif name == "Full intensity range":
            low, high = self.volume.data_min, self.volume.data_max
            level, width = (low + high) / 2, max(high - low, 1)
        elif not self.volume.is_hu:
            return
        else:
            level, width = {"Bone (HU)": (500, 2500), "Soft tissue (HU)": (40, 400), "Lung (HU)": (-600, 1500)}[name]
        self.window_level.setValue(level)
        self.window_width.setValue(width)

    def clear_measurements(self):
        for view in self.all_slices:
            view.measurements.clear()
            view.pending_measure = None
            view.update()

    def arch_picked(self, info):
        if self.arch_toggle.isChecked() and info.get("world") is not None:
            if len(self.arch_points) >= 2048:
                self.statusBar().showMessage("Arch point limit reached")
                return
            if not self.arch_points:
                self._arch_up = self.rotation()[:, 2].copy()
            self.arch_points.append(np.asarray(info["world"]))
            self.invalidate_panorama()
            self.schedule_render()
        elif self.arch_view.mode not in ("Add arch point", "Edit arch"):
            self.slice_picked(info, "Axial")

    def move_arch_point(self, index, delta):
        if not 0 <= index < len(self.arch_points):
            return
        # Keep the point on its original plane, including after axial scrolling.
        candidate = self.arch_points[index] + np.asarray(delta)
        voxel = self.volume.world_to_index(candidate)
        if np.any(voxel < 0) or np.any(voxel > np.asarray(self.volume.dimensions) - 1):
            return
        self.arch_points[index] = candidate
        self.arch_view.points = self.arch_points
        self.invalidate_panorama()
        self.arch_view.update()

    def remove_arch_point(self, index):
        if 0 <= index < len(self.arch_points):
            self.arch_points.pop(index)
            self.arch_view.points = self.arch_points
            self.invalidate_panorama()
            self.arch_view.update()

    def finish_arch(self):
        if len(self.arch_points) < 3:
            self.statusBar().showMessage("Place at least three arch points before finishing")
            return
        self.arch_toggle.setChecked(False)
        self.sync_arch_mode()
        self.pano_refresh_timer.start()
        self.statusBar().showMessage("Arch ready · updating the panoramic reformat")

    def section_settings_changed(self, view):
        panoramas = (getattr(self, "pano_view", None), getattr(self, "implant_pano_view", None))
        if view in panoramas:
            # These panes are two presentations of the same curved reformat.
            if view.slab_mm is not None:
                self.pano_thickness.setValue(view.slab_mm)
            if view.slab_mode is not None:
                self.pano_projection.setCurrentText(view.slab_mode)
            for pane in panoramas:
                if pane is not None:
                    pane.slab_mm = pane.slab_mode = None
            self.invalidate_panorama()
        self.schedule_render()

    def refresh_panorama_after_edit(self):
        if self._closing or self.volume is None or len(self.arch_points) < 3:
            return
        if self.pano_result is not None and not getattr(self, "_pano_offset_dirty", False) and not self.panorama_needs_detail():
            return
        if self.arch_toggle.isChecked() or self.tabs.currentIndex() not in (1, 2):
            return
        if self.job is not None or (self._interaction_active and not getattr(self, "_pano_offset_dirty", False)):
            self.pano_refresh_timer.start()
            return
        self.generate_panorama()

    def invalidate_panorama(self, *_):
        self.pano_revision += 1
        self.pano_result = None
        self.arch_view.path_points = []
        for view in [self.pano_view, self.implant_pano_view]:
            view.data = view.image = None
            view.note = "Updating panoramic reformat…" if len(self.arch_points) >= 3 else "Auto trace an arch or draw at least three points"
            view.update()
        if len(self.arch_points) >= 3:
            self.pano_refresh_timer.start()
        else:
            self.pano_refresh_timer.stop()

    def undo_arch(self):
        if self.arch_points:
            self.arch_points.pop()
            self.invalidate_panorama()
            self.schedule_render()

    def clear_arch(self):
        if getattr(self, "_pano_range_dialog", None) is not None:
            self._pano_range_dialog.close()
        self.arch_points = []
        self._arch_up = None
        self._pano_bounds = (-45., 45.)
        self.arch_view.points = []
        self.arch_view.selected_point = None
        self._pano_offset_dirty = False
        if hasattr(self, "pano_offset"):
            with QSignalBlocker(self.pano_offset):
                self.pano_offset.setValue(0)
        self.invalidate_panorama()
        self.sync_panorama_range_overlay()
        self.schedule_render()

    def panorama_range_geometry(self):
        from .panoramic_range import acquisition_bounds
        if self.volume is None or len(self.arch_points) < 3:
            raise ValueError("Trace an arch before setting the panoramic data range.")
        up = self.rotation()[:, 2]
        origin = np.asarray(self.arch_points).mean(axis=0)
        return origin, up, acquisition_bounds(self.volume, self.arch_points, up)

    def current_panorama_bounds(self):
        from .panoramic_range import validate_bounds
        lower, upper = validate_bounds(getattr(self, "_pano_bounds", (-45., 45.)))
        _, _, limits = self.panorama_range_geometry()
        lower, upper = max(lower, limits[0]), min(upper, limits[1])
        return (lower, upper) if upper > lower else limits

    def sync_panorama_range_overlay(self):
        if self.volume is None or len(self.arch_points) < 3:
            self.pano_sagittal_view.data_range = None
        else:
            origin, up, limits = self.panorama_range_geometry()
            self.pano_sagittal_view.set_data_range(origin, up, self.current_panorama_bounds(),
                                                  limits, min(self.volume.spacing))
        self.pano_sagittal_view.update()

    def define_data_range(self):
        if self.volume is None or len(self.arch_points) < 3:
            self.statusBar().showMessage("Trace an arch before setting the panoramic data range")
            return
        existing = getattr(self, "_pano_range_dialog", None)
        if existing is not None:
            existing.raise_()
            return existing
        from .panoramic_range_dialog import PanoramicRangeDialog
        dialog = PanoramicRangeDialog(self)
        self._pano_range_dialog = dialog
        dialog.finished.connect(lambda *_: setattr(self, "_pano_range_dialog", None))
        dialog.open()
        self.sync_panorama_range_overlay()
        return dialog

    def apply_panorama_range(self, lower, upper):
        from .panoramic_range import validate_bounds
        lower, upper = validate_bounds((lower, upper))
        _, _, limits = self.panorama_range_geometry()
        lower, upper = max(lower, limits[0]), min(upper, limits[1])
        if upper - lower < min(self.volume.spacing) - 1e-6:
            raise ValueError("The panoramic range must include at least one source voxel.")
        self._pano_bounds = (lower, upper)
        self.sync_panorama_range_overlay()
        self.pano_revision += 1
        self._pano_offset_dirty = True
        self.pano_refresh_timer.start()
        self.schedule_render()
        self.statusBar().showMessage(f"Panoramic range {lower:+.2f} to {upper:+.2f} mm · updating")

    def generate_panorama(self):
        if self.volume is None or len(self.arch_points) < 3:
            self.statusBar().showMessage("Place at least three arch points first")
            return
        if not self.volume.metadata.get("geometry_calibrated", True):
            self.statusBar().showMessage("Panoramic reconstruction requires calibrated DICOM geometry")
            return
        if self.job is None:
            self.pano_refresh_timer.stop()
        volume, points = self.volume, np.array(self.arch_points)
        up = self.rotation() @ np.array([0., 0., 1.])
        revision = self.pano_revision
        thickness, mode = self.pano_thickness.value(), self.pano_projection.currentText()
        focal_offset = self.pano_offset.value()
        bounds = self.current_panorama_bounds()
        self._pano_bounds = bounds
        self.sync_panorama_range_overlay()
        rows, columns = self.panorama_sample_shape()
        def compute(progress, cancel):
            progress(15, "Computing curved panoramic reformat…")
            result = curved_planar(volume, points, up, height_mm=bounds[1] - bounds[0], thickness_mm=thickness,
                                   samples=columns, rows=rows, mode=mode, cancel=cancel,
                                   focal_offset_mm=focal_offset, vertical_bounds_mm=bounds)
            progress(100, "Panoramic reformat ready")
            return revision, id(volume), up, thickness, mode, result
        self.start_job(compute, self.panorama_ready)

    def panorama_ready(self, payload):
        revision, volume_identity, up, thickness, mode, result = payload
        if revision != self.pano_revision or volume_identity != id(self.volume):
            self.statusBar().showMessage("Arch or orientation changed during reconstruction. Generate the updated panorama.")
            return
        self.pano_up = up
        self._pano_offset_dirty = False
        fraction = self.pano_index.value() / max(1, self.pano_index.maximum())
        self.pano_result = result
        self.pano_index.blockSignals(True)
        self.pano_index.setRange(0, len(result.path) - 1)
        self.pano_index.setValue(round(fraction * (len(result.path) - 1)))
        self.pano_index.blockSignals(False)
        size = ((result.image.shape[1] - 1) * result.column_pitch_mm, (result.image.shape[0] - 1) * result.row_pitch_mm)
        self.pano_view.set_image(result.image, note=f"Curved reformat · {mode} · {thickness:g} mm slab · click to locate", physical_size_mm=size)
        self.update_panorama_cursor()
        self.sync_panorama_position()
        self.schedule_render()
        if self.panorama_needs_detail():
            self.pano_refresh_timer.start()

    def configure_panorama_navigation(self, view):
        view.fraction_navigation_enabled = True
        view.scrolled.connect(self.scroll_panorama)
        view.interaction_started.connect(self.begin_interaction)
        view.interaction_finished.connect(self.end_interaction)

    def slice_sampling_changed(self, view):
        if self._closing or self.volume is None:
            return
        self.schedule_render()
        if view in (getattr(self, "pano_view", None), getattr(self, "implant_pano_view", None)):
            if self.pano_result is not None and self.panorama_needs_detail():
                self.pano_refresh_timer.start()

    def display_quality_changed(self, *_):
        self.schedule_render()
        if self.pano_result is not None and self.panorama_needs_detail():
            self.pano_refresh_timer.start()

    def panorama_sample_shape(self):
        from .sampling_quality import panorama_sampling_plan
        lower, upper = self.current_panorama_bounds()
        height = upper - lower
        size, shape = panorama_sampling_plan(self.arch_points, height, self.volume.spacing)
        quality = (.75, 1., 1.5)[self.resolution.currentIndex()]
        for view in (self.pano_view, self.implant_pano_view):
            if view.isVisible():
                target = view.preferred_sample_shape(size, quality=quality)
                shape = tuple(max(a, b) for a, b in zip(shape, target))
        return panorama_sampling_plan(self.arch_points, height, self.volume.spacing,
                                      display_shape=shape)[1]

    def panorama_needs_detail(self):
        if self.pano_result is None or self.volume is None or len(self.arch_points) < 3:
            return False
        required = self.panorama_sample_shape()
        return any(wanted > actual + 2 for wanted, actual in zip(required, self.pano_result.image.shape))

    def update_panorama_cursor(self):
        if self.pano_result is None or self.volume is None:
            return
        result = self.pano_result
        index = int(np.clip(self.pano_index.value(), 0, len(result.path) - 1))
        up = np.asarray(self.pano_up, dtype=float)
        up = up / np.linalg.norm(up)
        vertical = float(np.dot(self.crosshair - result.path[index], up))
        rows, columns = result.image.shape
        row = result.row_at_offset(vertical)
        fraction = ((index + .5) / columns, float(np.clip((row + .5) / rows, 0, 1)))
        for view in (self.pano_view, self.implant_pano_view):
            view.cursor_fraction = fraction
            view.update()
        self.sync_panorama_position()

    def scroll_panorama(self, steps):
        if self.pano_result is None or self.volume is None:
            return
        self.begin_interaction()
        self._wheel_settle.start()
        self.pano_offset.setValue(self.pano_offset.value() + float(steps) * min(self.volume.spacing))

    def panorama_offset_changed(self, *_):
        if self.volume is None or len(self.arch_points) < 3:
            return
        self.pano_revision += 1
        self._pano_offset_dirty = True
        self.pano_refresh_timer.start()
        self.sync_panorama_position()
        self.statusBar().showMessage(f"Panoramic arch depth {self.pano_offset.value():+.2f} mm · updating")

    def panorama_picked(self, info):
        result = self.pano_result
        if result is None:
            return
        try:
            fraction = np.asarray(info.get("fraction", ()), dtype=float)
        except (TypeError, ValueError):
            return
        if fraction.shape != (2,) or not np.all(np.isfinite(fraction)):
            return
        rows, columns = result.image.shape
        # Fractions describe the displayed image edges; voxels occupy centers.
        column = int(np.clip(round(fraction[0] * columns - .5), 0, columns - 1))
        row = float(np.clip(fraction[1] * rows - .5, 0, rows - 1))
        vertical_mm = float(result.offset_at_row(row))
        if self.locate_panorama_column(column, vertical_mm):
            self.pano_index.blockSignals(True)
            self.pano_index.setValue(column)
            self.pano_index.blockSignals(False)
            self.update_panorama_cursor()

    def panorama_index_changed(self, index):
        self.locate_panorama_column(index)
        self.update_panorama_cursor()

    def locate_panorama_column(self, index, vertical_mm=None):
        if self.volume is None or self.pano_result is None:
            return False
        path = self.pano_result.path
        try:
            value = float(index)
            if not np.isfinite(value):
                return False
            index = int(np.clip(value, 0, len(path) - 1))
            if vertical_mm is not None:
                vertical_mm = float(vertical_mm)
                if not np.isfinite(vertical_mm):
                    return False
        except (TypeError, ValueError):
            return False
        center = np.asarray(path[index], dtype=float)
        up = np.asarray(self.pano_up, dtype=float)
        up = up / np.linalg.norm(up)
        if vertical_mm is None:
            vertical_mm = float(np.dot(self.crosshair - center, up))
        candidate = center + up * vertical_mm
        source_index = self.volume.world_to_index(candidate)
        if (not np.all(np.isfinite(source_index)) or np.any(source_index < 0)
                or np.any(source_index > np.asarray(self.volume.dimensions) - 1)):
            self.statusBar().showMessage("That panoramic position lies outside the acquired volume")
            return False
        self.crosshair = candidate
        self.refresh_references()
        self.schedule_render()
        return True

    def implant_axis(self):
        tilt, yaw = np.deg2rad([self.implant_tilt.value(), self.implant_yaw.value()])
        return self.rotation() @ np.array([np.sin(tilt) * np.cos(yaw), np.sin(tilt) * np.sin(yaw), np.cos(tilt)])

    def add_implant(self):
        if self.volume is None:
            return
        if not self.volume.metadata.get("geometry_calibrated", True):
            self.statusBar().showMessage("Implant geometry requires calibrated DICOM spacing")
            return
        if len(self.implants) >= 200:
            self.statusBar().showMessage("Implant limit reached")
            return
        implant = {"id": uuid.uuid4().hex, "name": f"Implant {len(self.implants) + 1}", "center": self.crosshair.tolist(), "axis": self.implant_axis().tolist(), "diameter": self.implant_diameter.value(), "length": self.implant_length.value()}
        self.remember_implants()
        self.implants.append(implant)
        self.refresh_implants(len(self.implants) - 1)

    def begin_implant_placement(self):
        self.tool_combo.setCurrentText("Place implant")
        self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage("Click a section to place a generic implant at that position")

    def edit_implant(self):
        index = self.implant_list.currentRow()
        if 0 <= index < len(self.implants):
            self.remember_implants()
            self.implants[index].update(center=self.crosshair.tolist(), axis=self.implant_axis().tolist(), diameter=self.implant_diameter.value(), length=self.implant_length.value())
            self.refresh_implants(index)

    def delete_implant(self):
        index = self.implant_list.currentRow()
        if 0 <= index < len(self.implants):
            self.remember_implants()
            removed = self.implants.pop(index)
            self.implant_margins.pop(removed["id"], None)
            self.refresh_implants(min(index, len(self.implants) - 1))

    def refresh_implants(self, index):
        self.implant_list.blockSignals(True)
        self.implant_list.clear()
        for item in self.implants:
            self.implant_list.addItem(f"{item['name']}  ·  Ø {item['diameter']:g} × {item['length']:g} mm")
        self.implant_list.setCurrentRow(index)
        self.implant_list.blockSignals(False)
        for item in self.implants:
            item.setdefault("id", uuid.uuid4().hex)
        self.volume_view.set_implants(self.implants)
        self.select_implant(index)
        self.schedule_clearance()
        self.schedule_render()

    def select_implant(self, index):
        if 0 <= index < len(self.implants):
            item = self.implants[index]
            self.crosshair = np.asarray(item["center"])
            self.implant_diameter.setValue(item["diameter"])
            self.implant_length.setValue(item["length"])
            axis = self.rotation().T @ np.asarray(item["axis"])
            self.implant_tilt.setValue(np.rad2deg(np.arccos(np.clip(axis[2], -1, 1))))
            self.implant_yaw.setValue(np.rad2deg(np.arctan2(axis[1], axis[0])))
            self.planning_panel.margin.blockSignals(True)
            self.planning_panel.margin.setValue(self.implant_margins.get(item["id"], 0))
            self.planning_panel.margin.blockSignals(False)
            for view in self.all_slices:
                view.selected_implant_id = item["id"]
            if hasattr(self.volume_view, "set_implant_editing"):
                self.volume_view.set_implant_editing(self.mode == "Move implant", item["id"])
            self.schedule_clearance()
            self.schedule_render()

    def render_implant_views(self):
        index = self.implant_list.currentRow()
        if not (0 <= index < len(self.implants)):
            # Cross-sections are anatomy navigation and must work before an
            # implant is placed. Keep the same path frame used by the panorama.
            if self.pano_result is not None:
                from .reformat import path_frame
                column = min(len(self.pano_result.path) - 1, self.pano_index.value())
                center, right, axis = path_frame(self.pano_result.path, column, self.pano_up)
                center = center + axis * np.dot(self.crosshair - center, axis)
                across = np.cross(axis, right)
                note = "Dental arch section"
            else:
                center = self.crosshair.copy()
                right, axis = self.rotation()[:, 0], self.rotation()[:, 2]
                across = np.cross(axis, right)
                note = "Planar section · automatic arch preparation pending"
            for view, r, u in zip(self.implant_views, (right, across, right), (axis, axis, across)):
                self.draw_plane(view, center, r, u, (40, 50), note, (500, 400))
            self.render_serial_sections(center, axis, right, across)
            return
        item = self.implants[index]
        center, axis = np.asarray(item["center"]), np.asarray(item["axis"])
        basis = np.array([1., 0., 0.]) if abs(axis[0]) < 0.9 else np.array([0., 1., 0.])
        right = np.cross(basis, axis)
        right /= np.linalg.norm(right)
        across = np.cross(axis, right)
        for view, r, u in zip(self.implant_views, (right, across, right), (axis, axis, across)):
            self.draw_plane(view, center, r, u, (40, 50), f"{item['name']} · generic cylinder projection", (500, 400), center)
        self.render_serial_sections(center, axis, right, across)

    def begin_tmj(self, side):
        self.tool_combo.setCurrentText(f"Mark {side.lower()} TMJ")
        self.tabs.setCurrentIndex(3)
        self.statusBar().showMessage(f"Click the {side.lower()} condylar center on the axial locator")

    def move_tmj_center(self, side, center):
        index = self.volume.world_to_index(center)
        if np.any(index < 0) or np.any(index > np.asarray(self.volume.dimensions) - 1):
            self.tmj_locator.set_tmj_geometry(self.tmj_centers,
                {name: item.value() for name, item in self.tmj_angles.items()}, self.rotation(),
                {name: self.tmj_section_rotation(name)[:, 0] for name in ("Right", "Left")})
            return
        self.tmj_centers[side] = np.asarray(center).copy()
        self.schedule_render()

    def render_tmj_views(self):
        rotation = self.rotation()
        right, vertical = plane_frame("Axial", rotation)
        normal = np.cross(right, vertical)
        center = self.volume.center + normal * np.dot(self.crosshair - self.volume.center, normal)
        self.tmj_locator.set_tmj_geometry(self.tmj_centers,
            {name: item.value() for name, item in self.tmj_angles.items()}, rotation,
            {name: self.tmj_section_rotation(name)[:, 0] for name in ("Right", "Left")})
        self.draw_plane(self.tmj_locator, center, right, vertical, self._mpr_sizes["Axial"], "Projected centers · scroll axial level · drag a handle to rotate")
        self.render_tmj_section_planes()

    def snapshot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save current workspace snapshot", "AsterView.png", "PNG (*.png)")
        if path:
            if self.workspace_snapshot().save(path):
                self.statusBar().showMessage("Snapshot saved with visible study information and overlays.")

    def workspace_snapshot(self):
        from PySide6.QtGui import QPainter
        from PySide6.QtCore import QPoint
        widget = self
        pixmap = widget.grab()
        if (self.volume is not None
                and min(self.volume.dimensions) > 1 and self.volume_view.isVisible()):
            capture = self.volume_view.capture_image()
            target = self.volume_view.vtk_widget
            position = target.mapTo(widget, QPoint(0, 0))
            painter = QPainter(pixmap)
            painter.drawImage(position.x(), position.y(), capture.scaled(target.size()))
            painter.end()
        return pixmap

    def volume_fingerprint(self):
        # Session identity is checked without storing patient name or raw image data.
        meta = self.volume.metadata
        return {"series_uid": str(meta.get("series_uid", "")), "source_group": str(meta.get("source_group", "")), "sop_instance_digest": str(meta.get("sop_instance_digest", "")), "frame_of_reference_uid": str(meta.get("frame_of_reference_uid", "")), "shape": list(self.volume.data.shape), "spacing": list(self.volume.spacing), "origin": np.asarray(self.volume.origin).tolist(), "direction": np.asarray(self.volume.direction).tolist(), "synthetic": bool(meta.get("synthetic", False))}

    def save_session(self):
        if self.volume is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save viewer session", "session.aster.json", "AsterView session (*.json)")
        if not path:
            return
        payload = {"schema": 1, "volume": self.volume_fingerprint(), "crosshair": self.crosshair.tolist(), "rotation": list(to_euler(self.rotation())), "window": [self.window_level.value(), self.window_width.value()], "arch": [x.tolist() for x in self.arch_points], "implants": self.implants, "tmj_centers": {k: v.tolist() if v is not None else None for k, v in self.tmj_centers.items()}, "tmj_angles": {k: v.value() for k, v in self.tmj_angles.items()}}
        payload["implant_margins"] = {item["id"]: self.implant_margins.get(item["id"], 0) for item in self.implants}
        payload["viewing"] = {
            "pano_offset": self.pano_offset.value(),
            "pano_bounds": list(self._pano_bounds),
            "implant_section_width": self.implant_section_width.value(),
            "tmj_corrections": {
                side: getattr(self, "_tmj_section_corrections", {}).get(side, np.eye(3)).tolist()
                for side in ("Right", "Left")
            },
        }
        try:
            from .session import validate_session
            validate_session(payload, self.volume_fingerprint())
            Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.statusBar().showMessage("Session saved locally · reopen the matching DICOM series before restoring")
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Session could not be saved", str(exc))

    def open_session(self):
        if self.volume is None:
            self.statusBar().showMessage("Load the matching DICOM series first")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Restore session for current series", "", "AsterView session (*.json)")
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            self.restore_session(payload)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, "Session could not be restored", str(exc))

    def restore_session(self, payload):
        from .session import validate_session
        state = validate_session(payload, self.volume_fingerprint())
        self.clear_arch()
        self.crosshair = np.array(state["crosshair"])
        for widget, value in zip(self.rotation_spins, state["rotation"]):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self._rotation_override = rotation_matrix(*state["rotation"])
        self.window_level.setValue(state["window"][0])
        self.window_width.setValue(state["window"][1])
        self.arch_points = [np.array(x) for x in state["arch"]]
        self._arch_up = self.rotation()[:, 2].copy() if self.arch_points else None
        viewing = state["viewing"]
        self._pano_bounds = tuple(viewing["pano_bounds"])
        self.sync_panorama_range_overlay()
        for control, value in ((self.pano_offset, viewing["pano_offset"]),
                               (self.implant_section_width, viewing["implant_section_width"])):
            with QSignalBlocker(control):
                control.setValue(value)
        self._tmj_section_corrections = {
            side: np.array(matrix) for side, matrix in viewing["tmj_corrections"].items()
        }
        self.implants = state["implants"]
        self.implant_margins = state.get("implant_margins", {})
        self.refresh_implants(0 if self.implants else -1)
        self.tmj_centers = {k: np.array(v) if v is not None else None for k, v in state["tmj_centers"].items()}
        for key, value in state["tmj_angles"].items():
            self.tmj_angles[key].setValue(value)
        self.schedule_render()
        self.refresh_references()
        self.statusBar().showMessage("Matching session restored · regenerate the panorama from its saved arch")

    def closeEvent(self, event):
        self.close_section_grids()
        self.finish_implant_gesture()
        self.finish_mask_stroke()
        self.finish_path_gesture(reset_pointer=True)
        self.path_timer.stop()
        self.clearance_timer.stop()
        self.comparison.shutdown()
        self._closing = True
        self.pano_refresh_timer.stop()
        self.render_timer.stop()
        self._wheel_settle.stop()
        self.slice_pending = None
        self.slice_queue.invalidate()
        if self.slice_job and self.slice_job.isRunning():
            self.slice_job.cancelled = True
            self.slice_job.finished.connect(self.close)
            event.ignore()
            return
        if self.job and self.job.isRunning():
            self.job.cancelled = True
            self.statusBar().showMessage("Finishing the active decoding step before closing…")
            self.job.finished.connect(self.close)
            event.ignore()
            return
        self.volume_view.cleanup()
        event.accept()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AsterView local DICOM workstation")
    parser.add_argument("folder", nargs="?", help="DICOM study folder")
    parser.add_argument("--empty", action="store_true", help="Start without the synthetic demo")
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    from .clinician_profile import ensure_clinician_profile
    QTimer.singleShot(0, lambda: ensure_clinician_profile(window))
    if args.folder:
        QTimer.singleShot(200, lambda: window.scan_folder(args.folder))
    elif not args.empty:
        QTimer.singleShot(200, window.load_demo)
    return app.exec()
