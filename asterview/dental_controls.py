"""Application orchestration for tooth instances and patient-space canal paths."""
from __future__ import annotations

import copy
import uuid

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox


def combine_teeth(volume, fingerprint, teeth, base=None):
    """Replace dentition labels while retaining other reviewed source objects."""
    from .segmentation import create_result
    teeth.validate_source(volume, fingerprint)
    data, labels = teeth.data.copy(), copy.deepcopy(teeth.labels)
    if base is not None:
        base.validate_source(volume, fingerprint)
        original = base.snapshot()
        for key, value in base.labels.items():
            if value["kind"] == "tooth":
                continue
            new_key = key
            if new_key in labels:
                new_key = next(candidate for candidate in range(1, 65536) if candidate not in labels)
            labels[new_key] = copy.deepcopy(value)
            if np.any((original == key) & (data != 0)):
                labels[new_key]["review"] = "pending"
            data[(original == key) & (data == 0)] = new_key
    provenance = copy.deepcopy(teeth.provenance)
    if base is not None:
        provenance["method"] = (provenance.get("method", "") + f"; non-tooth objects retained from {base.provenance['name']} {base.provenance['version']}; existing review decisions retained")[:1024]
        provenance["license"] = (provenance.get("license", "") + "; retained anatomy: " + base.provenance.get("license", "unspecified"))[:1024]
    return create_result(volume, fingerprint, data.astype(np.uint16, copy=False), labels, provenance)


class DentalControls:
    def segment_teeth(self):
        if self.volume is None or self.job is not None:
            self.statusBar().showMessage("Load a CT/CBCT scan and wait for the current operation first")
            return
        from .tooth_inference import run_tooth_segmentation
        self.finish_mask_stroke(reset_pointer=True)
        self.finish_path_gesture(reset_pointer=True)
        self.tool_combo.setCurrentText("Navigate")
        volume, fingerprint, base = self.volume, self.volume_fingerprint(), self.segmentation
        revision = None if base is None else base.revision
        def infer(progress, cancel):
            teeth = run_tooth_segmentation(volume, fingerprint, base_result=base,
                progress=lambda message: progress(0, str(message)), cancel=cancel)
            if base is not None and base.revision != revision:
                raise ValueError("The source masks changed during tooth inference; run it again using the current masks.")
            return id(volume), base, revision, combine_teeth(volume, fingerprint, teeth, base)
        if self.start_job(infer, self.teeth_ready):
            self.segmentation_panel.set_busy(True)
            self.segmentation_panel.activity.setText("Preparing individual tooth and FDI inference…")

    def teeth_ready(self, payload):
        identity, base, revision, result = payload
        if self.volume is None or id(self.volume) != identity or self._closing:
            return
        if self.segmentation is not base or (base is not None and base.revision != revision):
            self.statusBar().showMessage("Tooth result discarded because the source masks changed")
            return
        self.segmentation_ready((identity, result))
        self.segmentation_panel.activity.setText("Individual teeth and suggested FDI labels · pending review")

    def open_nerve_paths(self):
        self.inspector.show()
        self.inspector_tabs.setCurrentWidget(self.review_page)
        self.review_tabs.setCurrentWidget(self.nerve_scroll)

    def extract_nerve_paths(self):
        if self.volume is None or self.segmentation is None or self.job is not None:
            self.statusBar().showMessage("Create or open a canal mask and wait for the current operation first")
            return
        from .nerve_paths import extract_canal_paths
        self.finish_mask_stroke(reset_pointer=True)
        self.finish_path_gesture(reset_pointer=True)
        self.tool_combo.setCurrentText("Navigate")
        volume, fingerprint, result = self.volume, self.volume_fingerprint(), self.segmentation
        revision = result.revision
        canal_ids = [key for key, item in result.labels.items() if item["kind"] == "canal"]
        if not canal_ids:
            self.statusBar().showMessage("The current result has no mandibular canal label")
            return
        def extract(progress, cancel):
            data = result.snapshot()
            canal = np.isin(data, canal_ids).astype(np.uint8)
            collection = extract_canal_paths(volume, fingerprint, canal, label_id=1,
                cancel=cancel, progress=lambda *args: progress(0, str(args[-1])), provenance=result.provenance)
            return id(volume), id(result), revision, collection
        if self.start_job(extract, self.nerve_paths_ready):
            self.nerve_panel.set_busy(True)
            self.nerve_panel.activity.setText("Extracting independent canal components…")
            self.open_nerve_paths()

    def nerve_paths_ready(self, payload):
        volume_id, result_id, revision, collection = payload
        if (self.volume is None or id(self.volume) != volume_id or self.segmentation is None
                or id(self.segmentation) != result_id or self.segmentation.revision != revision):
            self.statusBar().showMessage("Canal paths discarded because the source or mask changed")
            return
        self.install_nerve_paths(collection)
        warnings = getattr(collection, "warnings", [])
        if warnings:
            self.nerve_panel.activity.setText("Review required · " + "; ".join(warnings))

    def install_nerve_paths(self, collection):
        collection.validate_source(self.volume, self.volume_fingerprint())
        self.finish_path_gesture(reset_pointer=True)
        self.nerve_paths = collection
        self.nerve_panel.install()
        self.open_nerve_paths()
        self.refresh_nerve_paths()
        if collection.paths:
            self.focus_path_node()
        else:
            self.nerve_panel.activity.setText("No usable canal path was found. Inspect the mask or trace a manual path.")

    def refresh_nerve_paths(self, *_):
        if not hasattr(self, "nerve_panel") or not hasattr(self, "volume_view"):
            return
        panel = self.nerve_panel
        paths = []
        if self.nerve_paths is not None:
            from .nerve_paths import sample_path
            for path in self.nerve_paths.paths:
                paths.append(dict(path, sampled_points_lps_mm=self.nerve_paths.samples(path["id"]),
                                  visible=path["id"] in panel.visible_ids))
            if panel.show_original.isChecked():
                for path in self.nerve_paths.original_paths:
                    paths.append(dict(path, id="original:" + path["id"], name="Original " + path["name"],
                        color="#929aa6", review="pending", visible=True,
                        sampled_points_lps_mm=sample_path(path["points_lps_mm"])))
        editable = self.mode == "Edit canal path" and self.tabs.currentIndex() == 0
        for view in self.all_slices:
            if self.volume is not None and view.frame is not None:
                normal = np.cross(view.frame[1], view.frame[2])
                view.path_slice_half_mm = float(np.sum(np.abs(self.volume.direction.T @ normal) * self.volume.spacing) / 2)
            view.set_nerve_paths(paths, panel.selected_path_id, panel.selected_node, editable)
        self.volume_view.set_nerve_paths(paths, panel.selected_path_id, panel.selected_node, editable)
        self.schedule_clearance()
        panel.edit_button.setChecked(editable)

    def path_node_dragged(self, payload):
        phase = payload["phase"]
        if phase == "end":
            if "position_lps_mm" in payload and self._path_gesture_active:
                self.path_node_dragged(dict(payload, phase="move"))
            self.finish_path_gesture()
            return
        if self.nerve_paths is None or self.nerve_panel._busy or self.mode != "Edit canal path" or self.tabs.currentIndex() != 0:
            return
        panel = self.nerve_panel
        if phase == "start":
            self.finish_path_gesture()
            self.nerve_paths.begin_gesture()
            self._path_gesture_active = True
            self.begin_interaction()
            panel.paths.blockSignals(True)
            for row in range(panel.paths.count()):
                if panel.paths.item(row).data(Qt.ItemDataRole.UserRole) == payload["path_id"]:
                    panel.paths.setCurrentRow(row)
                    break
            panel.paths.blockSignals(False)
            panel.selected_node = int(payload["node_index"])
            panel.node.blockSignals(True)
            panel.node.setValue(panel.selected_node + 1)
            panel.node.blockSignals(False)
        if not self._path_gesture_active:
            return
        position = np.asarray(payload["position_lps_mm"], dtype=float)
        previous = np.asarray(self.nerve_paths.get_path(payload["path_id"])["points_lps_mm"][int(payload["node_index"])])
        position = self.constrain_position(previous, position)
        try:
            self.nerve_paths.move_node(self.volume, payload["path_id"], int(payload["node_index"]), position)
        except ValueError as exc:
            self.statusBar().showMessage(str(exc))
            return
        if "plane_normal" in payload:
            normal = np.asarray(payload["plane_normal"])
            self.crosshair = position - np.dot(position - payload["plane_center"], normal) * normal
        else:
            self.crosshair = position.copy()
        self.refresh_references()
        self.schedule_render()
        if not self.path_timer.isActive():
            self.path_timer.start()

    def finish_path_gesture(self, *, reset_pointer=False):
        if not getattr(self, "_path_gesture_active", False):
            return
        self._path_gesture_active = False
        self.path_timer.stop()
        if self.nerve_paths is not None:
            self.nerve_paths.end_gesture()
        if reset_pointer:
            for view in getattr(self, "all_slices", []):
                if view._drag_kind == "nerve":
                    view._drag_kind = view.last_pos = view._drag_world = view._path_drag = None
        self.end_interaction()
        self.nerve_panel.refresh()
        self.statusBar().showMessage("Canal path corrected · pending review")

    def select_path_node(self, value):
        self.finish_path_gesture()
        self.nerve_panel.selected_node = max(0, int(value) - 1)
        self.focus_path_node()

    def focus_path_node(self):
        path = self.nerve_panel.current_path()
        if path is None:
            return
        self.finish_path_gesture()
        node = min(self.nerve_panel.selected_node, len(path["points_lps_mm"]) - 1)
        self.nerve_panel.node.blockSignals(True)
        self.nerve_panel.node.setValue(node + 1)
        self.nerve_panel.node.blockSignals(False)
        self.crosshair = np.asarray(path["points_lps_mm"][node]).copy()
        self.tabs.setCurrentIndex(0)
        self.refresh_references()
        self.refresh_nerve_paths()
        self.schedule_render()

    def frame_selected_path(self):
        path = self.nerve_panel.current_path()
        if path is not None:
            self.volume_view.focus_nerve_path(path["id"])

    def new_nerve_path(self):
        if self.volume is None or self.nerve_panel._busy or not self.volume.metadata.get("geometry_calibrated", True):
            return
        from .nerve_paths import create_paths
        self.finish_path_gesture(reset_pointer=True)
        if self.nerve_paths is None:
            self.nerve_paths = create_paths(self.volume, self.volume_fingerprint(), [])
        center = self.volume.world_to_index(self.crosshair)
        first, second = center.copy(), center.copy()
        first[0] -= 3 / self.volume.spacing[0]
        second[0] += 3 / self.volume.spacing[0]
        points = [self.volume.index_to_world(np.clip(index, 0, np.asarray(self.volume.dimensions) - 1)).tolist()
                  for index in (first, second)]
        path = {"id": uuid.uuid4().hex, "name": "Manual canal path", "side": "unknown",
            "points_lps_mm": points, "diameter_mm": 2., "color": "#ee8487", "review": "pending",
            "provenance": {"name": "AsterView manual tracing", "version": "1", "method": "Two-point starter at crosshair; manual correction required"}}
        key = self.nerve_paths.add_path(self.volume, path)
        self.nerve_panel.visible_ids.add(key)
        self.nerve_panel.selected_node = 0
        self.nerve_panel.refresh(key)
        self.open_nerve_paths()
        self.tool_combo.setCurrentText("Edit canal path")
        self.focus_path_node()

    def insert_path_node(self):
        if self.nerve_panel._busy:
            return
        path = self.nerve_panel.current_path()
        if path is None:
            return
        self.finish_path_gesture(reset_pointer=True)
        index = min(self.nerve_panel.selected_node + 1, len(path["points_lps_mm"]))
        try:
            self.nerve_paths.insert_node(self.volume, path["id"], index, self.crosshair)
            self.nerve_panel.selected_node = index
            self.nerve_panel.refresh()
        except ValueError as exc:
            self.statusBar().showMessage(str(exc))

    def delete_path_node(self, payload=None):
        if self.nerve_panel._busy:
            return
        path = self.nerve_panel.current_path()
        if path is None:
            return
        self.finish_path_gesture(reset_pointer=True)
        index = payload["node_index"] if isinstance(payload, dict) else self.nerve_panel.selected_node
        try:
            self.nerve_paths.delete_node(path["id"], int(index))
            self.nerve_panel.refresh()
        except ValueError as exc:
            self.statusBar().showMessage(str(exc))

    def update_path_properties(self):
        if self.nerve_panel._busy:
            return
        panel = self.nerve_panel
        path = panel.current_path()
        if path is None:
            return
        self.finish_path_gesture(reset_pointer=True)
        self.nerve_paths.begin_gesture()
        try:
            self.nerve_paths.set_name(path["id"], panel.name.text())
            self.nerve_paths.set_side(path["id"], panel.side.currentText())
            self.nerve_paths.set_diameter(path["id"], panel.diameter.value())
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid path properties", str(exc))
        finally:
            self.nerve_paths.end_gesture()
        panel.refresh()

    def undo_path(self):
        if self.nerve_panel._busy:
            return
        self.finish_path_gesture(reset_pointer=True)
        if self.nerve_paths is not None:
            self.nerve_paths.undo()
            self.nerve_panel.visible_ids |= {path["id"] for path in self.nerve_paths.paths}
            self.nerve_panel.refresh()

    def undo_edit(self):
        if self.mode == "Move implant":
            self.undo_implant()
        elif self.mode == "Edit canal path" or self.review_tabs.currentWidget() is self.nerve_scroll:
            self.undo_path()
        else:
            self.undo_mask()
