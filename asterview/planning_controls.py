"""Coalesced live clearance and linked implant manipulation."""
import numpy as np
from copy import deepcopy
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidgetItem


class PlanningControls:
    def schedule_clearance(self, *_):
        if hasattr(self, "clearance_timer") and not self.clearance_timer.isActive():
            self.clearance_timer.start()

    def update_clearances(self):
        from .implant_planning import evaluate_clearances
        paths = [] if self.nerve_paths is None else self.nerve_paths.paths
        curves = {} if self.nerve_paths is None else {path["id"]: self.nerve_paths.samples(path["id"]) for path in paths}
        self.clearance_results = evaluate_clearances(self.implants, paths, margins_mm=self.implant_margins, curve_samples=curves)
        panel = self.planning_panel
        panel.results.blockSignals(True)
        panel.results.clear()
        for index, result in enumerate(self.clearance_results):
            distance = result.get("distance_mm")
            text = f"{self.implants[index]['name']} · "
            status = result["status"].replace("_", " ")
            if result["status"] == "unreviewed":
                status += " · " + result["geometry_status"].replace("_", " ")
            text += "No canal path" if distance is None else f"{distance:.2f} mm · {status}\nMargin {self.implant_margins.get(self.implants[index]['id'], 0):.2f} mm"
            panel.results.addItem(QListWidgetItem(text))
        index = self.implant_list.currentRow()
        panel.results.setCurrentRow(index)
        panel.results.blockSignals(False)
        panel.margin.setEnabled(0 <= index < len(self.implants))
        selected = self.clearance_results[index] if 0 <= index < len(self.clearance_results) else None
        panel.summary.setText("Select an implant to inspect its closest canal clearance." if selected is None else panel.results.item(index).text())
        self._clearance_display = selected if panel.show_measurement.isChecked() else None
        for view in self.all_slices:
            view.clearance_measurement = self._clearance_display
            view.update()
        if hasattr(self.volume_view, "set_clearance_measurement"):
            self.volume_view.set_clearance_measurement(self._clearance_display)

    def change_implant_margin(self, value):
        index = self.implant_list.currentRow()
        if 0 <= index < len(self.implants):
            self.implant_margins[self.implants[index]["id"]] = float(value)
            self.schedule_clearance()

    def open_planning(self):
        self.inspector.show()
        self.inspector_tabs.setCurrentWidget(self.planning_scroll)
        self.schedule_clearance()

    def focus_clearance(self):
        result = getattr(self, "_clearance_display", None)
        if result is not None and result.get("implant_point_lps") is not None:
            self.crosshair = np.asarray(result["implant_point_lps"]).copy()
            self.refresh_references()
            self.schedule_render()

    def implant_dragged(self, payload):
        phase = payload["phase"]
        if phase == "end":
            if self._implant_drag_active and "center_lps" in payload:
                self.implant_dragged(dict(payload, phase="move"))
            self._implant_drag_active = False
            self.end_interaction()
            self.select_implant(self.implant_list.currentRow())
            self.schedule_clearance()
            return
        if self.mode != "Move implant" or self.volume is None:
            return
        index = next((i for i, item in enumerate(self.implants) if item["id"] == payload["implant_id"]), -1)
        if index < 0:
            return
        if phase == "start":
            self.remember_implants()
            self._implant_drag_active = True
            self.begin_interaction()
            self.implant_list.setCurrentRow(index)
        if not self._implant_drag_active:
            return
        implant = self.implants[index]
        previous = np.asarray(implant["center"])
        implant["center"] = self.constrain_position(previous, np.asarray(payload["center_lps"])).tolist()
        if "axis_lps" in payload:
            axis = np.asarray(payload["axis_lps"], dtype=float)
            norm = float(np.linalg.norm(axis))
            if np.isfinite(norm) and norm > 1e-8:
                implant["axis"] = (axis / norm).tolist()
        self.crosshair = np.asarray(implant["center"])
        self.volume_view.set_implants(self.implants)
        self.refresh_references()
        self.schedule_clearance()
        self.schedule_render()

    def remember_implants(self):
        if not hasattr(self, "_implant_history"):
            self._implant_history = []
        self._implant_history.append((deepcopy(self.implants), dict(self.implant_margins)))
        del self._implant_history[:-30]

    def implant_dimensions_edited(self, orientation=False):
        index = self.implant_list.currentRow()
        if not 0 <= index < len(self.implants):
            return
        self.remember_implants()
        item = self.implants[index]
        item.update(diameter=self.implant_diameter.value(), length=self.implant_length.value())
        if orientation:
            item["axis"] = self.implant_axis().tolist()
        self.refresh_implants(index)

    def finish_implant_gesture(self):
        if self._implant_drag_active:
            self._implant_drag_active = False
            self.end_interaction()
        for view in self.all_slices:
            if view._drag_kind == "implant":
                view._implant_drag = None
                view._drag_kind = view.last_pos = None
                view.unsetCursor()

    def undo_implant(self):
        if getattr(self, "_implant_history", []):
            self.implants, self.implant_margins = self._implant_history.pop()
            self.refresh_implants(min(self.implant_list.currentRow(), len(self.implants) - 1))
