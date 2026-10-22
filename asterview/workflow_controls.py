"""Reviewable dental setup and per-section workspace actions."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer, QPoint, QEvent, QRect, QSignalBlocker, Signal, QPointF
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QWidget, QFileDialog, QFormLayout,
                               QSpinBox, QDoubleSpinBox, QDialogButtonBox, QLabel,
                               QGridLayout, QScrollArea, QHBoxLayout, QPushButton, QCheckBox)

from .mpr_geometry import plane_frame
from .tmj_sections import TMJSectionControls
from .section_navigation import SectionNavigationControls


class SectionGridContainer(QWidget):
    """Embedded section group with an explicit restore-to-single lifecycle."""
    closed = Signal()

    def closeEvent(self, event):
        self.closed.emit()
        event.accept()


class WorkflowControls(TMJSectionControls, SectionNavigationControls):
    def reset_workspace_splitters(self):
        from .workspace_layout import reset_workspace_splitters
        reset_workspace_splitters(self)

    def apply_explorer_maximize(self, target):
        from .workspace_layout import apply_explorer_maximize
        return apply_explorer_maximize(self, target)

    def _invalidate_section_sampling(self):
        """Reject already-emitted frame batches before retiring their widgets."""
        self.slice_revision += 1
        queue = getattr(self, "slice_queue", None)
        if queue is not None:
            queue.invalidate()
        self.slice_pending = None
        if self.slice_job is not None:
            self.slice_job.cancelled = True

    def section_layout_widget(self, source):
        for record in getattr(self, "_section_grids", []):
            if record["source"] is source:
                return record["dialog"]
        return source

    @staticmethod
    def _section_parent_layout(source):
        parent = source.parentWidget()
        def containing(layout):
            if layout is None:
                return None
            for index in range(layout.count()):
                item = layout.itemAt(index)
                if item.widget() is source:
                    return layout
                nested = containing(item.layout())
                if nested is not None:
                    return nested
            return None
        return containing(parent.layout() if parent is not None else None)

    @staticmethod
    def _clear_section_pixels(view, note=""):
        view.image = view.gray = view.data = None
        view.segmentation_image = view._segmentation_rgba = None
        view.frame = view.image_key = None
        view.tooth_annotations = []
        view.note = note
        if hasattr(view, "position_slider"):
            view.position_slider.hide()
            view._position_visible = False
            view._position_context = None
        view.update()

    @staticmethod
    def _section_sample_shape(view, size, maximum=2048):
        """Sample displayed image pixels, not fixed large arrays per tiny cell."""
        return view.preferred_sample_shape(size, maximum=maximum)

    def _arch_section_geometry(self):
        if self.pano_result is None:
            self._section_arch_cache = None
            return None
        from .section_geometry import ArchSectionGeometry
        key = (id(self.pano_result), tuple(np.asarray(self.pano_up, dtype=float)))
        cache = getattr(self, "_section_arch_cache", None)
        if cache is None or cache[0] != key:
            cache = (key, ArchSectionGeometry(self.pano_result.path, self.pano_up))
            self._section_arch_cache = cache
        return cache[1]

    def reset_workflow_state(self):
        self._auto_setup_epoch = getattr(self, "_auto_setup_epoch", 0) + 1
        self.clear_pending_automatic_setup()
        self._auto_tmj_attempt = None
        self._auto_arch_attempt = None
        self.auto_setup_provenance = {}
        self._tmj_section_corrections = {}
        self._auto_pano_pending = None
        self._section_arch_cache = None
        self._implant_history = []
        if hasattr(self, "comparison"):
            self.comparison.reset_comparison()
        if hasattr(self, "reporting_panel"):
            self.reporting_panel.reset_source()

    def clear_pending_automatic_setup(self):
        """Cancel continuations without changing existing masks/manual geometry."""
        self._auto_setup_request = None
        self._auto_anatomy_request = None
        self._auto_pano_pending = None
        self._auto_arch_attempt = None
        self._auto_tmj_attempt = None

    def request_automatic_workflow(self, index):
        """Initialize an empty dental workflow after an explicit tab selection."""
        if self.volume is None or index not in (1, 2, 3):
            return
        kind = "tmj" if index == 3 else "arch"
        if kind == "arch" and self.arch_points:
            return
        if kind == "tmj" and any(value is not None for value in self.tmj_centers.values()):
            return
        segmentation = self.segmentation
        key = (id(self.volume), id(segmentation),
               None if segmentation is None else segmentation.revision)
        attribute = "_auto_tmj_attempt" if kind == "tmj" else "_auto_arch_attempt"
        if getattr(self, attribute, None) == key:
            return
        setattr(self, attribute, key)
        volume, epoch = self.volume, self._auto_setup_epoch
        def invoke():
            if self.volume is volume and self._auto_setup_epoch == epoch and not self._closing:
                (self.auto_tmj if kind == "tmj" else self.auto_arch)()
        QTimer.singleShot(0, invoke)

    def _automatic_setup_message(self, message):
        self.statusBar().showMessage(message)
        self.segmentation_panel.activity.setText(message)

    def _ensure_automatic_anatomy(self, kind):
        """Reuse current masks or infer them locally and queue this workflow."""
        from .automatic_setup import AutomaticSetupRequest
        if self.volume is None:
            self._automatic_setup_message("Open a calibrated CT/CBCT scan before automatic dental setup")
            return False
        if self.job is not None:
            self._auto_setup_request = AutomaticSetupRequest.capture(self, kind)
            name = "TMJ setup" if kind == "tmj" else "panoramic/implant arch"
            self._automatic_setup_message(f"Queued {name} · the current operation will finish first; Cancel stops the queued setup")
            return False
        if self.segmentation is not None:
            try:
                self.segmentation.validate_source(self.volume, self.volume_fingerprint())
            except ValueError as exc:
                self._automatic_setup_message("Automatic setup unavailable · current masks do not match this source: " + str(exc))
                return False
            return True
        from .inference import detect_model_status, run_segmentation, InferenceCancelled
        volume, fingerprint = self.volume, self.volume_fingerprint()
        if (not volume.metadata.get("geometry_calibrated", True)
                or min(volume.dimensions) < 2
                or volume.metadata.get("modality", "").upper() not in ("CT", "CBCT")):
            self._automatic_setup_message("Automatic dental setup requires calibrated three-dimensional CT/CBCT geometry; the demo does not run this model")
            return False
        status = detect_model_status()
        if not status.get("ready"):
            self.open_segmentation()
            setattr(self, "_auto_tmj_attempt" if kind == "tmj" else "_auto_arch_attempt", None)
            self._automatic_setup_message("Automatic anatomy unavailable · " + status.get("reason", "Run Setup AI.ps1 to install the local model"))
            return False
        request = AutomaticSetupRequest.capture(self, kind)
        self._auto_setup_request = self._auto_anatomy_request = request
        def infer(progress, cancel):
            try:
                return run_segmentation(volume, fingerprint,
                    progress=lambda message: progress(0, str(message)), cancel=cancel)
            except InferenceCancelled:
                raise
            except (ValueError, RuntimeError, OSError) as exc:
                return {"unavailable": str(exc)}
        def install(result):
            if (not request.source_matches(self) or self.segmentation is not request.segmentation
                    or self._auto_anatomy_request is not request):
                self.clear_pending_automatic_setup()
                self._automatic_setup_message("Scan or masks changed during anatomy inference; automatic setup was discarded")
                return
            if isinstance(result, dict) and "unavailable" in result:
                self.clear_pending_automatic_setup()
                self._automatic_setup_message("Automatic anatomy failed · " + result["unavailable"])
                return
            try:
                result.validate_source(volume, fingerprint)
            except ValueError as exc:
                self.clear_pending_automatic_setup()
                self._automatic_setup_message("Automatic anatomy result rejected · " + str(exc))
                return
            self.finish_mask_stroke()
            self.segmentation = result
            self.segmentation_panel.install()
            self.refresh_segmentation()
            pending = self._auto_setup_request
            if pending is not None and pending.source_matches(self) and pending.segmentation is None:
                self._auto_setup_request = pending.with_segmentation(result)
            self._auto_anatomy_request = None
            self._automatic_setup_message("Local anatomy ready · completing the requested dental setup; masks remain pending review")
        if self.start_job(infer, install):
            self.segmentation_panel.set_busy(True)
            self._automatic_setup_message("Preparing local dental anatomy automatically · the requested view will continue when ready")
        else:
            self.clear_pending_automatic_setup()
        return False

    def auto_arch(self):
        if not self._ensure_automatic_anatomy("arch"):
            return
        from .auto_dental import propose_arch
        volume, segmentation = self.volume, self.segmentation
        epoch = getattr(self, "_auto_setup_epoch", 0)
        fingerprint, rotation = self.volume_fingerprint(), self.rotation().copy()
        points_before = np.array(self.arch_points).copy()
        right, up = plane_frame("Axial", rotation)
        normal = np.cross(right, up)
        center = volume.center + normal * np.dot(self.crosshair - volume.center, normal)
        revision = segmentation.revision
        def propose(progress, cancel):
            progress(10, "Tracing an editable arch from dental anatomy…")
            try:
                return propose_arch(volume, segmentation, fingerprint, plane_center_lps=center,
                                    right_lps=right, up_lps=up, cancel=cancel, depth_mode="anatomy")
            except ValueError as exc:
                return {"unavailable": str(exc)}
        def install(result):
            if epoch != getattr(self, "_auto_setup_epoch", 0) or self.volume is not volume or self.segmentation is not segmentation or segmentation.revision != revision or not np.array_equal(points_before, np.array(self.arch_points)) or not np.allclose(rotation, self.rotation()) or abs(np.dot(self.crosshair - center, normal)) > 1e-5:
                self.statusBar().showMessage("Dental setup changed during tracing; run Auto trace arch again")
                return
            if isinstance(result, dict) and "unavailable" in result:
                self.statusBar().showMessage("Automatic arch unavailable · " + result["unavailable"])
                return
            self.arch_points = [point.copy() for point in result.points_lps_mm]
            self._arch_up = rotation[:, 2].copy()
            anatomy_center = np.mean(result.points_lps_mm, axis=0)
            candidate = self.crosshair + normal * np.dot(anatomy_center - self.crosshair, normal)
            index = volume.world_to_index(candidate)
            self.crosshair = (candidate if np.all((index >= 0) & (index <= np.asarray(volume.dimensions) - 1))
                              else anatomy_center)
            self.arch_view.points = self.arch_points
            self.arch_toggle.setChecked(False)
            self.invalidate_panorama()
            self.auto_setup_provenance["arch"] = result
            self.refresh_references()
            self.statusBar().showMessage("Proposed arch · review and drag its points. " + " · ".join(result.warnings))
            self.schedule_render()
            self._auto_pano_pending = (id(volume), self.pano_revision)
        self.start_job(propose, install)

    def finish_pending_auto_panorama(self):
        if self._closing:
            self.clear_pending_automatic_setup()
            return
        if self.job is not None:
            return
        pending = getattr(self, "_auto_pano_pending", None)
        self._auto_pano_pending = None
        if pending == (id(self.volume), self.pano_revision):
            self.generate_panorama()
            if self.job is not None:
                return
        request = getattr(self, "_auto_setup_request", None)
        self._auto_setup_request = None
        if getattr(self, "_auto_anatomy_request", None) is not None:
            # Inference ended without installing a usable result (error/cancel).
            self._auto_anatomy_request = None
            return
        if request is not None:
            if request.matches(self):
                (self.auto_tmj if request.kind == "tmj" else self.auto_arch)()
            else:
                self._automatic_setup_message("Dental setup changed while waiting; the queued setup was discarded. Run Auto setup again to use the current view")

    def auto_tmj(self):
        if not self._ensure_automatic_anatomy("tmj"):
            return
        from .auto_dental import propose_tmj
        volume, segmentation = self.volume, self.segmentation
        epoch = getattr(self, "_auto_setup_epoch", 0)
        fingerprint, rotation = self.volume_fingerprint(), self.rotation().copy()
        before = {key: None if value is None else value.copy() for key, value in self.tmj_centers.items()}
        angles = {key: item.value() for key, item in self.tmj_angles.items()}
        corrections = {side: getattr(self, "_tmj_section_corrections", {}).get(side, np.eye(3)).copy()
                       for side in ("Right", "Left")}
        revision = segmentation.revision
        def propose(progress, cancel):
            progress(10, "Finding condylar setup candidates…")
            try:
                return propose_tmj(volume, segmentation, fingerprint, cancel=cancel)
            except ValueError as exc:
                return {"unavailable": str(exc)}
        def install(result):
            same_centers = all((value is None and self.tmj_centers[key] is None) or (value is not None and self.tmj_centers[key] is not None and np.array_equal(value, self.tmj_centers[key])) for key, value in before.items())
            if epoch != getattr(self, "_auto_setup_epoch", 0) or self.volume is not volume or self.segmentation is not segmentation or segmentation.revision != revision or not same_centers or not np.allclose(rotation, self.rotation()) or any(angles[key] != item.value() for key, item in self.tmj_angles.items()):
                self.statusBar().showMessage("TMJ setup changed during proposal; run Auto locate TMJ again")
                return
            if any(not np.allclose(corrections[side], getattr(self, "_tmj_section_corrections", {}).get(side, np.eye(3)))
                   for side in corrections):
                self.statusBar().showMessage("TMJ correction changed during proposal; run Auto locate TMJ again")
                return
            if isinstance(result, dict) and "unavailable" in result:
                self.statusBar().showMessage("Automatic TMJ setup unavailable · " + result["unavailable"])
                return
            for key, center in result.centers_lps_mm.items():
                side = key.capitalize()
                self.tmj_centers[side] = np.asarray(center).copy()
                getattr(self, "_tmj_section_corrections", {}).pop(side, None)
                axis = rotation.T @ result.axes_lps[key]
                if np.linalg.norm(axis[:2]) < 1e-6:
                    result.warnings.append(f"{side} axis is normal to the corrected locator; retained its prior angle for manual review")
                else:
                    self.tmj_angles[side].setValue(np.rad2deg(np.arctan2(axis[1], axis[0])))
            if result.centers_lps_mm:
                anatomy_center = np.mean(list(result.centers_lps_mm.values()), axis=0)
                normal = rotation[:, 2]
                candidate = self.crosshair + normal * np.dot(anatomy_center - self.crosshair, normal)
                index = volume.world_to_index(candidate)
                self.crosshair = (candidate if np.all((index >= 0) & (index <= np.asarray(volume.dimensions) - 1))
                                  else anatomy_center)
                self.refresh_references()
            self.auto_setup_provenance["tmj"] = result
            self.schedule_render()
            self.statusBar().showMessage("Provisional TMJ setup · review both centers and axes. " + " · ".join(result.warnings))
        self.start_job(propose, install)

    def rebuild_implant_sections(self, *_):
        if not hasattr(self, "implant_serial_layout"):
            return
        self._invalidate_section_sampling()
        if getattr(self, "_section_dialog", None) is not None:
            self._section_dialog.close()
        count = self.implant_section_count.value()
        columns = self.implant_section_columns.value()
        while len(self.implant_serial_views) < count:
            view = self.new_slice(f"Cross-section {len(self.implant_serial_views) + 1}", "#e5858b")
            view.set_grid_cell(len(self.implant_serial_views))
            if self.implant_serial_views:
                view.slab_mm = self.implant_serial_views[0].slab_mm
                view.slab_mode = self.implant_serial_views[0].slab_mode
            from .ui_shell import wire_section_view
            wire_section_view(self, view, grid=True)
            view.picked.connect(lambda info: self.slice_picked(info, "Axial"))
            view.scrolled.connect(lambda steps: self.implant_section_offset.setValue(
                self.implant_section_offset.value() + steps * self.implant_section_spacing.value()))
            view.zoom = (self.implant_serial_views[0].zoom if self.implant_serial_views
                         else getattr(self, "_implant_grid_zoom", 1.))
            view.zoom_changed.connect(self.link_implant_section_zoom)
            view.interaction_started.connect(self.begin_interaction)
            view.interaction_finished.connect(self.end_interaction)
            view.max_button.show()
            view.maximize_requested.connect(lambda v=view: self.maximize_section(v))
            self.implant_serial_views.append(view)
        for index, view in enumerate(self.implant_serial_views):
            self.implant_serial_layout.removeWidget(view)
            if index < count:
                self.implant_serial_layout.addWidget(view, index // columns, index % columns)
            else:
                self._clear_section_pixels(view)
            view.setVisible(index < count)
        for row in range(8):
            self.implant_serial_layout.setRowStretch(row, int(row < self.implant_section_rows.value()))
        for column in range(12):
            self.implant_serial_layout.setColumnStretch(column, int(column < columns))
        self.implant_section_offset.setSingleStep(self.implant_section_spacing.value())
        self.schedule_render()

    def render_serial_sections(self, center, axis, right, across):
        count = self.implant_section_count.value()
        spacing = self.implant_section_spacing.value()
        geometry = self._arch_section_geometry()
        selected_index = self.implant_list.currentRow() if hasattr(self, "implant_list") else -1
        implant_selected = 0 <= selected_index < len(getattr(self, "implants", []))
        section_height = 50.
        if geometry is not None and hasattr(self.pano_result, "image"):
            height = (self.pano_result.image.shape[0] - 1) * self.pano_result.row_pitch_mm
            if np.isfinite(height) and height > 0:
                section_height = float(height)
        vertical_middle = (sum(self.pano_result.vertical_limits_mm) / 2
                           if geometry is not None and hasattr(self.pano_result, "vertical_limits_mm") else 0.)
        offset = self.implant_section_offset.value()
        if geometry is not None:
            selected = geometry.nearest_distance(center)
            lower, upper = geometry.offset_limits(selected, count, spacing)
            offset = float(np.clip(offset, lower, upper))
            with QSignalBlocker(self.implant_section_offset):
                self.implant_section_offset.setRange(np.floor(lower * 100) / 100,
                                                     np.ceil(upper * 100) / 100)
                self.implant_section_offset.setValue(offset)
            self._implant_arch_scroll_base = selected
            slider = getattr(self, "implant_arch_position", None)
            if slider is not None:
                slider.set_position(selected + lower, selected + upper, selected + offset, min(self.volume.spacing))
        else:
            self._implant_arch_scroll_base = None
            slider = getattr(self, "implant_arch_position", None)
            if slider is not None:
                slider.setEnabled(False)
            with QSignalBlocker(self.implant_section_offset):
                self.implant_section_offset.setRange(-1000, 1000)
        self.implant_section_offset.setSingleStep(spacing)
        markers = []
        for index, view in enumerate(self.implant_serial_views[:count]):
            if getattr(self, "_section_dialog_view", None) in self.implant_serial_views and self._section_dialog_view is not view:
                continue
            distance = offset + (index - (count - 1) / 2) * spacing
            if geometry is not None:
                frame = geometry.frame(selected + distance)
                if frame is None:
                    self._clear_section_pixels(view, "Outside traced arch")
                    continue
                position, horizontal, vertical = frame
                # Anatomy navigation shows the same height as the CPR; a
                # selected implant retains its own planning depth.
                if implant_selected:
                    position = position + vertical * np.dot(center - position, vertical)
                else:
                    position = position + vertical * vertical_middle
                view._arch_scroll_base = selected + (index - (count - 1) / 2) * spacing
                markers.append((selected + distance, index + 1))
                note = f"Arch {selected + distance:.1f} mm · {distance:+.1f} mm"
                view.grid_caption = f"{index + 1} · {selected + distance:.1f}"
            else:
                view._arch_scroll_base = (index - (count - 1) / 2) * spacing
                position, horizontal, vertical = center + across * distance, right, axis
                note = f"Implant-normal offset {distance:+.1f} mm · trace an arch for dental sections"
                view.grid_caption = f"{index + 1} · {distance:+.1f}"
            size = (self.implant_section_width.value(), section_height)
            self.draw_plane(view, position, horizontal, vertical, size, note,
                            self._section_sample_shape(view, size), self.crosshair)
        self.update_implant_section_markers(geometry, markers, self.implant_section_width.value())

    def configure_section_grid(self, view=None):
        """Choose a grid for implant serial views, or a linked grid around any plane."""
        from .ui_shell import SectionGridPicker
        for record in getattr(self, "_section_grids", []):
            if view in record["views"]:
                view = record["source"]
                break
        implant = view is None or view in self.implant_serial_views
        if not implant and (self.volume is None or view.frame is None):
            self.statusBar().showMessage("Open a planar section before creating a linked section grid")
            return
        existing = getattr(self, "_grid_settings_dialog", None)
        if existing is not None:
            existing.close()
        dialog = QDialog(self)
        dialog.setWindowTitle("Implant section grid" if implant else view.title + " · Linked section grid")
        layout = QVBoxLayout(dialog)
        rows, columns = QSpinBox(), QSpinBox()
        rows.setObjectName("sectionGridRows")
        columns.setObjectName("sectionGridColumns")
        rows.setRange(1, 8)
        columns.setRange(1, 12)
        previous = getattr(view, "serial_grid_settings", (3, 3, 1.)) if not implant else (
            self.implant_section_rows.value(), self.implant_section_columns.value(), self.implant_section_spacing.value())
        rows.setValue(previous[0])
        columns.setValue(previous[1])
        picker = SectionGridPicker(rows.value(), columns.value())
        layout.addWidget(picker, 0, Qt.AlignmentFlag.AlignHCenter)
        chosen = QLabel()
        def update_label(*_):
            picker.set_grid(rows.value(), columns.value())
            chosen.setText(f"{rows.value()} rows × {columns.value()} columns · {rows.value() * columns.value()} linked sections")
        def matrix_changed(nrows, ncolumns):
            rows.setValue(nrows)
            columns.setValue(ncolumns)
        picker.grid_changed.connect(matrix_changed)
        rows.valueChanged.connect(update_label)
        columns.valueChanged.connect(update_label)
        update_label()
        layout.addWidget(chosen)
        form = QFormLayout()
        form.addRow("Rows", rows)
        form.addRow("Columns", columns)
        spacing = QDoubleSpinBox()
        spacing.setObjectName("sectionGridSpacing")
        spacing.setRange(.2, 20)
        spacing.setDecimals(2)
        spacing.setSingleStep(.2)
        spacing.setSuffix(" mm")
        spacing.setValue(previous[2])
        form.addRow("Center-to-center spacing", spacing)
        width = QDoubleSpinBox()
        width.setObjectName("sectionGridWidth")
        width.setRange(5, 150 if implant else 500)
        width.setDecimals(1)
        width.setSuffix(" mm")
        width.setValue(self.implant_section_width.value() if implant else
                       getattr(view, "serial_grid_width", view.frame[3][0]))
        form.addRow("Section image width", width)
        from .thickness_control import ThicknessControl
        thickness = ThicknessControl()
        thickness.setObjectName("sectionGridThickness")
        if self.volume is not None:
            thickness.configure_for_spacing(self.volume.spacing)
        sample = (self.implant_serial_views[0] if self.implant_serial_views else None) if implant else view
        slab = None if sample is None else sample.slab_mm
        active_group = next((record for record in getattr(self, "_section_grids", [])
                             if record["source"] is view), None)
        if active_group is not None:
            slab = active_group.get("group_slab_mm", slab)
        thickness.setValue(self.slab.value() if slab is None else slab)
        follow = QCheckBox("Follow shared thickness")
        follow.setChecked(slab is None)
        thickness.setEnabled(not follow.isChecked())
        thickness_edited = [False]
        def thickness_changed(*_):
            thickness_edited[0] = True
            thickness.setEnabled(not follow.isChecked())
        thickness.valueChanged.connect(thickness_changed)
        follow.toggled.connect(thickness_changed)
        form.addRow(follow)
        form.addRow("Section thickness", thickness)
        layout.addLayout(form)
        hint = QLabel("Sections run left to right, then top to bottom. Scroll any section to move the linked range. Each wrench controls its own slab thickness and projection.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Done")
        def apply():
            if implant:
                with QSignalBlocker(self.implant_section_rows), QSignalBlocker(self.implant_section_columns), QSignalBlocker(self.implant_section_count):
                    self.implant_section_rows.setValue(rows.value())
                    self.implant_section_columns.setValue(columns.value())
                    self.implant_section_count.setValue(rows.value() * columns.value())
                self.implant_section_spacing.setValue(min(spacing.value(), self.implant_section_spacing.maximum()))
                self.implant_section_width.setValue(width.value())
                self.rebuild_implant_sections()
                if thickness_edited[0]:
                    for child in self.implant_serial_views:
                        child.set_section_settings(None if follow.isChecked() else thickness.value(), child.slab_mode)
                self.implant_display_tabs.setCurrentIndex(1)
            else:
                view.serial_grid_settings = (rows.value(), columns.value(), spacing.value())
                record = self.open_section_grid(view, rows.value(), columns.value(), spacing.value(), width_mm=width.value())
                if thickness_edited[0]:
                    value = None if follow.isChecked() else thickness.value()
                    if record is None:
                        view.set_section_settings(value, view.slab_mode)
                    else:
                        record["group_slab_mm"] = value
                        record["pane_settings"] = {
                            index: (value, *settings[1:])
                            for index, settings in record["pane_settings"].items()
                        }
                        for child in record["views"]:
                            child.set_section_settings(value, child.slab_mode)
            dialog.accept()
        buttons.accepted.connect(apply)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        self._grid_settings_dialog = dialog
        dialog.finished.connect(lambda *_: setattr(self, "_grid_settings_dialog", None))
        dialog.open()
        return dialog

    def open_section_grid(self, source, rows, columns, spacing_mm, *, width_mm=None):
        """Replace one planar pane with a linked, embedded section matrix."""
        if (not isinstance(rows, int) or isinstance(rows, bool)
                or not isinstance(columns, int) or isinstance(columns, bool)
                or not 1 <= rows <= 8 or not 1 <= columns <= 12
                or not np.isfinite(spacing_mm) or not .2 <= spacing_mm <= 20):
            raise ValueError("Section grid requires 1–8 rows, 1–12 columns and 0.2–20 mm spacing")
        if self.volume is None or source not in self.all_slices or source.frame is None:
            raise ValueError("Section grid requires a current planar source view")
        if width_mm is not None and (not np.isfinite(width_mm) or not 5 <= width_mm <= 500):
            raise ValueError("Section image width must be between 5 and 500 mm")
        grids = getattr(self, "_section_grids", None)
        if grids is None:
            self._section_grids = grids = []
        source.serial_grid_settings = (rows, columns, float(spacing_mm))
        if width_mm is not None:
            source.serial_grid_width = float(width_mm)
        previous = next((item for item in grids if item["source"] is source), None)
        if rows == columns == 1:
            if previous is not None:
                previous["dialog"].close()
            return None
        if previous is not None:
            if getattr(self, "_section_dialog_view", None) in previous["views"] or getattr(self, "_section_dialog_view", None) is previous["dialog"]:
                self._section_dialog.close()
            previous.update(rows=rows, columns=columns, spacing_mm=float(spacing_mm))
            if width_mm is not None:
                previous["width_mm"] = float(width_mm)
            previous["offset"].setSingleStep(spacing_mm)
            previous["caption"].setText(f"{source.title} · {rows} × {columns}")
            previous["dialog"].setWindowTitle(source.title + f" · {rows} × {columns} linked sections")
            self._resize_section_grid(previous)
            self.schedule_render()
            return previous
        if len(grids) >= 4:
            self.statusBar().showMessage("Restore a section grid to a single pane before opening another")
            return None
        parent = source.parentWidget()
        original = self._section_parent_layout(source)
        if original is None:
            raise ValueError("The section source must belong to a workspace layout")
        if getattr(self, "_section_dialog_view", None) is source:
            self._section_dialog.close()
            parent, original = source.parentWidget(), self._section_parent_layout(source)
        container = SectionGridContainer(parent)
        container.title = source.title + " · Sections"
        container.setObjectName("linkedSectionGrid")
        container.setWindowTitle(source.title + f" · {rows} × {columns} linked sections")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        options = QHBoxLayout()
        options.setContentsMargins(5, 2, 5, 2)
        caption = QLabel(f"{source.title} · {rows} × {columns}")
        options.addWidget(caption)
        options.addStretch()
        offset = QDoubleSpinBox()
        offset.setRange(-1000, 1000)
        offset.setDecimals(2)
        offset.setSingleStep(spacing_mm)
        offset.setSuffix(" mm")
        offset.setMaximumWidth(94)
        offset.setAccessibleName("Linked section grid offset")
        offset.setToolTip("Scroll any section to move this whole range")
        offset.valueChanged.connect(self.schedule_render)
        options.addWidget(offset)
        edit = QPushButton("Grid…")
        edit.setToolTip("Change rows, columns or physical section spacing")
        edit.clicked.connect(lambda: self.configure_section_grid(source))
        options.addWidget(edit)
        single = QPushButton("Single")
        single.setToolTip("Restore the original single section")
        single.clicked.connect(container.close)
        options.addWidget(single)
        layout.addLayout(options)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        grid = QGridLayout(content)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(3)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        scroll.verticalScrollBar().valueChanged.connect(self.schedule_render)
        scroll.horizontalScrollBar().valueChanged.connect(self.schedule_render)
        record = {"source": source, "volume": self.volume, "dialog": container,
                  "views": [], "rows": rows, "columns": columns,
                  "spacing_mm": float(spacing_mm), "offset": offset,
                  "width_mm": float(width_mm) if width_mm is not None else getattr(source, "serial_grid_width", source.frame[3][0]),
                  "acquisition_label": None, "caption": caption, "grid": grid,
                  "scroll": scroll, "parent": parent, "original_layout": original,
                  "pane_settings": {}}
        grids.append(record)
        original.replaceWidget(source, container)
        source.hide()
        self._resize_section_grid(record)
        def closed():
            if record not in self._section_grids:
                return
            self._invalidate_section_sampling()
            if getattr(self, "_section_dialog_view", None) in record["views"] or getattr(self, "_section_dialog_view", None) is container:
                self._section_dialog.close()
            self._section_grids.remove(record)
            original.replaceWidget(container, source)
            source.show()
            for child in record["views"]:
                self._clear_section_pixels(child)
                if child in self.all_slices:
                    self.all_slices.remove(child)
            container.deleteLater()
            if not self._closing:
                self.schedule_render()
        container.closed.connect(closed)
        container.show()
        self.schedule_render()
        return record

    def _resize_section_grid(self, record):
        """Reuse cells and controls across matrix changes; release removed pixels."""
        self._invalidate_section_sampling()
        source, grid = record["source"], record["grid"]
        count = record["rows"] * record["columns"]
        for index, child in enumerate(record["views"]):
            grid.removeWidget(child)
            if index >= count:
                record["pane_settings"][index] = (child.slab_mm, child.slab_mode, child.zoom, QPointF(child.pan))
                self._clear_section_pixels(child)
                if child in self.all_slices:
                    self.all_slices.remove(child)
                child.deleteLater()
        del record["views"][count:]
        from .ui_shell import wire_section_view
        while len(record["views"]) < count:
            index = len(record["views"])
            child = self.new_slice(f"{source.title} · {index + 1}", source.accent)
            child.set_grid_cell(index)
            child.slab_mm, child.slab_mode = record.get("group_slab_mm", source.slab_mm), source.slab_mode
            if index in record["pane_settings"]:
                child.slab_mm, child.slab_mode, child.zoom, child.pan = record["pane_settings"][index]
            child.mode = self.mode
            child.set_window(source.center, source.window_width, source.invert)
            wire_section_view(self, child, grid=True)
            child.picked.connect(lambda info, group=record: self.section_grid_picked(group, info))
            child.scrolled.connect(lambda steps, group=record: self.scroll_section_grid(group, steps))
            child.reference_translated.connect(source.reference_translated.emit)
            child.reference_rotated.connect(source.reference_rotated.emit)
            child.interaction_started.connect(self.begin_interaction)
            child.interaction_finished.connect(self.end_interaction)
            child.max_button.show()
            child.maximize_requested.connect(lambda v=child: self.maximize_section(v))
            record["views"].append(child)
        for index, child in enumerate(record["views"]):
            grid.addWidget(child, index // record["columns"], index % record["columns"])
            child.show()
        for row in range(8):
            grid.setRowStretch(row, int(row < record["rows"]))
        for column in range(12):
            grid.setColumnStretch(column, int(column < record["columns"]))

    def scroll_section_grid(self, record, steps):
        if record not in getattr(self, "_section_grids", []) or record["volume"] is not self.volume:
            return
        if not np.isfinite(steps):
            return
        control = record["offset"]
        control.setValue(control.value() + float(steps) * record["spacing_mm"])

    def section_grid_picked(self, record, info):
        """Locate anatomy without double-applying the selected cell's offset."""
        before = self.section_source_frame(record["source"])
        tmj = next((key for key, view in getattr(self, "tmj_views", {}).items()
                    if view is record["source"]), None)
        if tmj is None:
            self.slice_picked(info, "Axial")
        else:
            self.pick_tmj_section(tmj[0], info)
        after = self.section_source_frame(record["source"])
        if before is not None and after is not None:
            shift = float(np.dot(np.asarray(after[0]) - before[0], np.cross(before[1], before[2])))
            if abs(shift) > 1e-8:
                with QSignalBlocker(record["offset"]):
                    record["offset"].setValue(record["offset"].value() - shift)

    def section_source_frame(self, source):
        """Resolve source geometry even when its workflow is currently hidden."""
        name = next((key for key, view in getattr(self, "mpr_views", {}).items() if view is source), None)
        for attribute, plane in (("pano_sagittal_view", "Sagittal"), ("arch_view", "Axial"),
                                 ("implant_arch_view", "Axial"), ("tmj_locator", "Axial")):
            if source is getattr(self, attribute, None):
                name = plane
        if name is not None:
            right, up = plane_frame(name, self.rotation())
            normal = np.cross(right, up)
            center = self.volume.center + normal * np.dot(self.crosshair - self.volume.center, normal)
            size = getattr(self, "_mpr_sizes", {}).get(name, source.frame[3])
            return center, right, up, size
        tmj = next((key for key, view in getattr(self, "tmj_views", {}).items() if view is source), None)
        if tmj is not None:
            return self.tmj_section_frame(*tmj)
        request = next((item[1] for item in (getattr(self, "_plane_requests", None) or []) if item[0] is source), None)
        return request if request is not None else getattr(source, "_grid_source_frame", source.frame)

    def render_section_grids(self):
        for record in list(getattr(self, "_section_grids", [])):
            if record["volume"] is not self.volume:
                record["dialog"].close()
                continue
            if (self.isVisible() and not record["dialog"].isVisible()
                    and getattr(self, "_section_dialog_view", None) not in record["views"]):
                # A grid in a hidden workflow needs no pixel work. Its source
                # frame is recomputed from current geometry when shown again.
                continue
            frame = self.section_source_frame(record["source"])
            if frame is None:
                for child in record["views"]:
                    self._clear_section_pixels(child, "Choose a source section")
                continue
            center, right, up, size = frame
            size = (record["width_mm"], size[1])
            tmj = next((key for key, view in getattr(self, "tmj_views", {}).items()
                        if view is record["source"]), None)
            cursor = self.tmj_section_cursor(tmj[0]) if tmj is not None else self.crosshair
            normal = np.cross(right, up)
            count = len(record["views"])
            for index, child in enumerate(record["views"]):
                if getattr(self, "_section_dialog_view", None) in record["views"] and self._section_dialog_view is not child:
                    continue
                offset = record["offset"].value() + (index - (count - 1) / 2) * record["spacing_mm"]
                position = np.asarray(center) + normal * offset
                child.grid_caption = f"{index + 1} · {offset:+.1f}"
                child.reference_lines = record["source"].reference_lines
                child.show_references = record["source"].show_references
                self.draw_plane(child, position, right, up, size,
                                f"{offset:+.2f} mm · spacing {record['spacing_mm']:g} mm",
                                self._section_sample_shape(child, size), cursor)

    def close_section_grids(self):
        dialog = getattr(self, "_grid_settings_dialog", None)
        if dialog is not None:
            dialog.close()
        for record in list(getattr(self, "_section_grids", [])):
            record["dialog"].close()

    def snapshot_section(self, view):
        if view.image is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save section with visible overlays", "section.png", "PNG image (*.png)")
        if path:
            if not view.grab().save(path, "PNG"):
                self.statusBar().showMessage("Unable to save section image")
            else:
                self.statusBar().showMessage("Section image saved with its current overlays")

    def snapshot_volume(self):
        if self.volume is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save 3D quick shot", "volume.png", "PNG image (*.png)")
        if path:
            self.volume_view.capture_image().save(path, "PNG")

    def place_shared_volume(self, index):
        previous = self._volume_workflow
        if index == previous:
            return
        if self._maximized_view is not None:
            self.maximize_view(self._maximized_view)
        if getattr(self, "_section_dialog", None) is not None:
            self._section_dialog.close()
        self._volume_maximized = False
        self._volume_workflow = index
        self.sync_volume_geometry()
        self._volume_geometry_timer.start(0)
        self.volume_view._queue_render()

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Move, QEvent.Type.Show, QEvent.Type.LayoutRequest):
            timer = getattr(self, "_volume_geometry_timer", None)
            if timer is not None and not self._closing:
                timer.start(0)
        return super().eventFilter(watched, event)

    def sync_volume_geometry(self):
        if not hasattr(self, "_volume_placeholders") or self._closing:
            return
        index = self.tabs.currentIndex()
        if index not in self._volume_placeholders:
            self.volume_card.hide()
            return
        placeholder = self._volume_placeholders[index]
        show = not (index == 0 and self._maximized_view is not None and self._maximized_view is not self.volume_card)
        if self._volume_maximized:
            geometry = self.tabs.contentsRect()
        else:
            geometry = QRect(placeholder.mapTo(self.tabs, QPoint(0, 0)), placeholder.size())
        if self.volume_card.geometry() != geometry:
            self.volume_card.setGeometry(geometry)
        self.volume_card.setVisible(show)
        if show:
            self.volume_card.raise_()

    def maximize_volume(self):
        if self.tabs.currentIndex() == 0:
            self.maximize_view(self.volume_card)
        else:
            self._volume_maximized = not self._volume_maximized
            self.sync_volume_geometry()

    def update_acquisition_info(self):
        from .acquisition import acquisition_text, acquisition_details
        text = acquisition_text(self.volume.metadata, self.volume.dimensions,
                                self.volume.spacing, self.volume.direction)
        self.acquisition_label.setText(text or "Acquisition parameters not supplied in this source")
        for view in self.all_slices:
            view.source_text = ""
        self.acquisition_label.setToolTip(acquisition_details(self.volume.metadata))

    def reviewed_measurements_for_report(self):
        result = []
        for view in self.all_slices:
            for index, (start, end) in enumerate(view.measurements):
                result.append({"label": f"Clinician ruler · {view.title} · {index + 1}",
                               "value": float(np.linalg.norm(np.asarray(end) - start)), "unit": "mm", "reviewed": True})
        return result

    def implant_locator_picked(self, info):
        if self.mode == "Edit arch" and info.get("world") is not None:
            return
        self.slice_picked(info, "Axial")

    def render_implant_locator(self):
        rotation = self.rotation()
        right, up = plane_frame("Axial", rotation)
        normal = np.cross(right, up)
        center = self.volume.center + normal * np.dot(self.crosshair - self.volume.center, normal)
        self.implant_arch_view.points = self.arch_points
        self.implant_arch_view.path_points = [] if self.pano_result is None else self.pano_result.path
        self.implant_arch_view.mode = self.mode if self.mode in ("Move implant", "Place implant", "Pan", "Window / level", "Measure") else "Edit arch"
        self.draw_plane(self.implant_arch_view, center, right, up, self._mpr_sizes["Axial"], "Editable arch · choose axial level")
        if self.pano_result is not None:
            source = self.pano_view
            self.implant_pano_view.set_image(source.data, frame=None, note=source.note, physical_size_mm=source.physical_size_mm)
        else:
            self.implant_pano_view.data = self.implant_pano_view.image = None
            self.implant_pano_view.note = "Auto trace arch or draw one in Panoramic, then Reformat"
            self.implant_pano_view.update()

    def maximize_section(self, view):
        if view is self.volume_card:
            self.maximize_volume()
            return
        if view in self.mpr_layout and self.tabs.currentIndex() == 0:
            self.maximize_view(view)
            return
        if getattr(self, "_section_dialog", None) is not None:
            self._section_dialog.close()
            return
        view = self.section_layout_widget(view)
        parent = view.parentWidget()
        original = parent.layout() if parent else None
        if original is None:
            return
        placeholder = QWidget()
        original.replaceWidget(view, placeholder)
        dialog = QDialog(self)
        dialog.setWindowTitle(getattr(view, "title", "3D volume") + " · AsterView")
        dialog.resize(1100, 850)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        if hasattr(self, "acquisition_label"):
            acquisition = QLabel(self.acquisition_label.text())
            acquisition.setTextFormat(Qt.TextFormat.PlainText)
            acquisition.setContentsMargins(10, 5, 10, 5)
            acquisition.setToolTip(self.acquisition_label.toolTip())
            layout.addWidget(acquisition)
        layout.addWidget(view)
        self._section_dialog = dialog
        self._section_dialog_view = view
        def restore(*_):
            layout.removeWidget(view)
            original.replaceWidget(placeholder, view)
            view.setParent(parent)
            placeholder.deleteLater()
            view.show()
            self._section_dialog = None
            self._section_dialog_view = None
            self.schedule_render()
        dialog.finished.connect(restore)
        dialog.show()
        self.schedule_render()
