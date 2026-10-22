"""Link physical section sliders to existing navigation state."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from .physical_slider import PhysicalSlider, line_scroll_limits


class SectionNavigationControls:
    def sync_section_position(self, view, frame, cursor):
        if self.volume is None:
            return
        if not getattr(view, "_position_bound", False):
            view._position_bound = True
            view.position_changed.connect(lambda value, pane=view: self.section_position_changed(pane, value))
        center, right, up, _ = frame
        normal = np.cross(right, up).astype(float)
        normal /= np.linalg.norm(normal)
        step = float(min(self.volume.spacing))
        tmj = next((key for key, pane in getattr(self, "tmj_views", {}).items() if pane is view), None)
        group = next((record for record in getattr(self, "_section_grids", [])
                      if view in record["views"]), None)
        if tmj is not None:
            control = self.tmj_offset
            context = ("control", control, 0.)
            limits, value = (control.minimum(), control.maximum()), control.value()
        elif view in getattr(self, "implant_serial_views", []) and hasattr(view, "_arch_scroll_base"):
            control, base = self.implant_section_offset, view._arch_scroll_base
            context = ("control", control, base)
            limits = control.minimum() + base, control.maximum() + base
            value = control.value() + base
        elif group is not None:
            value = float(np.asarray(center) @ normal)
            control = group["offset"]
            context = ("control", control, value - control.value())
            limits = line_scroll_limits(self.volume, center, normal)
        else:
            position = np.asarray(cursor)
            value = float(position @ normal)
            context = ("plane", normal)
            limits = line_scroll_limits(self.volume, position, normal)
        view._position_context = context
        view.set_scroll_position(*limits, value, step)
        if (view in getattr(self, "implant_views", []) and not getattr(view, "_position_wheel_bound", False)):
            view._position_wheel_bound = True
            view.scrolled.connect(lambda steps, pane=view: self.section_position_changed(
                pane, pane.position_slider.physical_value + float(steps) * min(self.volume.spacing)))

    def section_position_changed(self, view, value):
        if self.volume is None or not np.isfinite(value):
            return
        context = getattr(view, "_position_context", None)
        if context is None:
            return
        if context[0] == "control":
            context[1].setValue(value - context[2])
        else:
            normal = context[1]
            candidate = self.crosshair + normal * (value - float(self.crosshair @ normal))
            self.crosshair = self.constrain_position(self.crosshair, candidate) if hasattr(self, "constrain_position") else candidate
            if hasattr(self, "refresh_references"):
                self.refresh_references()
        self.schedule_render()

    def sync_panorama_position(self):
        if self.volume is None:
            return
        for view in (getattr(self, "pano_view", None), getattr(self, "implant_pano_view", None)):
            if view is None:
                continue
            if not getattr(view, "_panorama_position_bound", False):
                view._panorama_position_bound = True
                view.position_changed.connect(self.pano_offset.setValue)
            view.set_scroll_position(self.pano_offset.minimum(), self.pano_offset.maximum(),
                                     self.pano_offset.value(), min(self.volume.spacing))

    def link_implant_section_zoom(self, zoom):
        zoom = float(np.clip(zoom, .2, 12))
        self._implant_grid_zoom = zoom
        for view in self.implant_serial_views:
            view.zoom = zoom
            view.update()
        self.schedule_render()

    def build_implant_arch_slider(self, layout):
        label = QLabel("Arch position")
        self.implant_arch_position = PhysicalSlider(Qt.Orientation.Horizontal)
        self.implant_arch_position.setMinimumWidth(100)
        self.implant_arch_position.setAccessibleName("Linked implant section arch position")
        self.implant_arch_position.position_changed.connect(self.set_implant_arch_position)
        self.implant_arch_position.sliderPressed.connect(self.begin_interaction)
        self.implant_arch_position.sliderReleased.connect(self.end_interaction)
        layout.addWidget(label)
        layout.addWidget(self.implant_arch_position, 1)
        self.implant_arch_position.setEnabled(False)

    def set_implant_arch_position(self, value):
        base = getattr(self, "_implant_arch_scroll_base", None)
        if base is not None:
            self.implant_section_offset.setValue(float(value) - base)

    def update_implant_section_markers(self, geometry, positions, width):
        markers, columns = [], []
        if geometry is not None:
            sample_count = len(self.pano_result.path)
            for distance, label in positions:
                frame = geometry.frame(distance)
                if frame is None:
                    continue
                center, horizontal, _ = frame
                markers.append((center - horizontal * width / 2, center + horizontal * width / 2, label))
                column = np.interp(distance, geometry.arc, np.arange(sample_count))
                fraction = (column + .5) / sample_count
                columns.append((fraction, label))
        for view in (getattr(self, "arch_view", None), getattr(self, "implant_arch_view", None)):
            if view is not None:
                view.section_markers = markers
                view.update()
        for view in (getattr(self, "pano_view", None), getattr(self, "implant_pano_view", None)):
            if view is not None:
                view.section_columns = columns
                view.update()
