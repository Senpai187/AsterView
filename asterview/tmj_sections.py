"""Direct manipulation of independently corrected bilateral TMJ sections."""
from __future__ import annotations

import numpy as np

from .mpr_geometry import plane_frame, plane_references, rotate_about_view, translate_reference
from .reformat import rotation_matrix


class TMJSectionControls:
    def tmj_section_rotation(self, side):
        base = self.rotation() @ rotation_matrix(0, 0, self.tmj_angles[side].value())
        correction = getattr(self, "_tmj_section_corrections", {}).get(side, np.eye(3))
        return base @ correction

    def tmj_section_frame(self, side, plane):
        center = self.tmj_centers[side]
        if center is None:
            return None
        right, up = plane_frame(plane, self.tmj_section_rotation(side))
        return (np.asarray(center) + np.cross(right, up) * self.tmj_offset.value(),
                right, up, (50., 50.))

    def tmj_section_cursor(self, side):
        rotation = self.tmj_section_rotation(side)
        normals = [np.cross(*plane_frame(plane, rotation)) for plane in ("Sagittal", "Coronal")]
        return np.asarray(self.tmj_centers[side]) + sum(normals) * self.tmj_offset.value()

    def wire_tmj_section(self, view, side, plane):
        if getattr(view, "_tmj_section_wired", False):
            return
        view._tmj_section_wired = True
        view.scrolled.connect(self.scroll_tmj_sections)
        view.picked.connect(lambda payload: self.pick_tmj_section(side, payload))
        view.reference_translated.connect(lambda payload: self.translate_tmj_section(side, plane, payload))
        view.reference_rotated.connect(lambda degrees: self.rotate_tmj_section(side, plane, degrees))
        view.interaction_started.connect(self.begin_interaction)
        view.interaction_finished.connect(self.end_interaction)

    def scroll_tmj_sections(self, steps):
        if self.volume is None or not np.isfinite(steps):
            return
        self.begin_interaction()
        timer = getattr(self, "_wheel_settle", None)
        if timer is not None:
            timer.start()
        self.tmj_offset.setValue(self.tmj_offset.value() + float(steps) * min(self.volume.spacing))

    def pick_tmj_section(self, side, payload):
        world = payload.get("world")
        if world is None or self.tmj_centers[side] is None:
            return
        world = np.asarray(world, dtype=float)
        if world.shape != (3,) or not np.isfinite(world).all():
            return
        # Keep the clicked point at the linked intersections even with an offset.
        delta = np.asarray(world) - self.tmj_section_cursor(side)
        self.move_tmj_center(side, np.asarray(self.tmj_centers[side]) + delta)

    def translate_tmj_section(self, side, plane, payload):
        center = self.tmj_centers[side]
        if center is None:
            return
        candidate = translate_reference(center, payload["delta"], plane,
                                        payload.get("target"), self.tmj_section_rotation(side))
        self.move_tmj_center(side, candidate)

    def rotate_tmj_section(self, side, plane, degrees):
        if self.tmj_centers[side] is None or not np.isfinite(degrees):
            return
        rotation = self.tmj_section_rotation(side)
        cursor = self.tmj_section_cursor(side)
        corrected = rotate_about_view(rotation, plane, float(degrees))
        normals = [np.cross(*plane_frame(name, corrected)) for name in ("Sagittal", "Coronal")]
        center = cursor - sum(normals) * self.tmj_offset.value()
        index = self.volume.world_to_index(center)
        if np.any(index < 0) or np.any(index > np.asarray(self.volume.dimensions) - 1):
            return
        base = self.rotation() @ rotation_matrix(0, 0, self.tmj_angles[side].value())
        if not hasattr(self, "_tmj_section_corrections"):
            self._tmj_section_corrections = {}
        self._tmj_section_corrections[side] = base.T @ corrected
        self.tmj_centers[side] = center
        self.schedule_render()

    def render_tmj_section_planes(self):
        for side in ("Right", "Left"):
            for plane in ("Sagittal", "Coronal"):
                view = self.tmj_views[(side, plane)]
                frame = self.tmj_section_frame(side, plane)
                if frame is None:
                    self._clear_section_pixels(view, f"Mark the {side.lower()} center on the axial locator")
                    view.reference_lines = []
                    continue
                view.reference_lines = [
                    reference for reference in plane_references(plane, self.tmj_section_rotation(side))
                    if reference["name"] != "Axial"
                ]
                view.show_references = self.lines_visible.isChecked()
                setup = "Proposed / editable center" if "tmj" in getattr(self, "auto_setup_provenance", {}) else "Manual center"
                self.draw_plane(view, *frame, f"{setup} · offset {self.tmj_offset.value():+.2f} mm",
                                crosshair=self.tmj_section_cursor(side))
