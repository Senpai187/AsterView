"""Reviewable canal centerlines, manual tracing and reversible point editing."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QCheckBox, QDoubleSpinBox, QSpinBox, QComboBox,
    QLineEdit, QFileDialog, QMessageBox)


class NervePathPanel(QWidget):
    def __init__(self, workstation):
        super().__init__()
        self.workstation = workstation
        self.selected_node = 0
        self.visible_ids = set()
        self._refreshing = self._busy = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 8, 2, 2)
        title = QLabel("Editable canal paths")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self.extract_button = QPushButton("Create paths from canal mask")
        self.extract_button.setObjectName("primary")
        self.extract_button.clicked.connect(workstation.extract_nerve_paths)
        layout.addWidget(self.extract_button)
        self.cancel_button = QPushButton("Cancel path extraction")
        self.cancel_button.clicked.connect(workstation.cancel_job)
        self.cancel_button.hide()
        layout.addWidget(self.cancel_button)
        manual = QPushButton("New manual path at crosshair")
        manual.clicked.connect(workstation.new_nerve_path)
        layout.addWidget(manual)
        self.activity = QLabel("")
        self.activity.setWordWrap(True)
        self.activity.setTextFormat(Qt.TextFormat.PlainText)
        self.activity.setObjectName("subtitle")
        layout.addWidget(self.activity)
        self.paths = QListWidget()
        self.paths.setMinimumHeight(115)
        self.paths.setWordWrap(True)
        self.paths.currentRowChanged.connect(self.selection_changed)
        self.paths.itemChanged.connect(self.visibility_changed)
        layout.addWidget(self.paths)
        self.edit_button = QPushButton("Drag control points")
        self.edit_button.setCheckable(True)
        self.edit_button.clicked.connect(lambda: workstation.tool_combo.setCurrentText("Edit canal path"))
        layout.addWidget(self.edit_button)
        row = QHBoxLayout()
        row.addWidget(QLabel("Node"))
        self.node = QSpinBox()
        self.node.setMinimum(1)
        self.node.valueChanged.connect(workstation.select_path_node)
        row.addWidget(self.node)
        focus = QPushButton("Locate")
        focus.clicked.connect(workstation.focus_path_node)
        row.addWidget(focus)
        layout.addLayout(row)
        self.frame_button = QPushButton("Frame path in 3D")
        self.frame_button.clicked.connect(workstation.frame_selected_path)
        layout.addWidget(self.frame_button)
        for left, right in ((("Add at crosshair", workstation.insert_path_node), ("Delete node", workstation.delete_path_node)),
                            (("Undo path edit", workstation.undo_path), ("Navigate", lambda: workstation.tool_combo.setCurrentText("Navigate"))),
                            (("Accept", lambda: self.review("accepted")), ("Reject", lambda: self.review("rejected")))):
            row = QHBoxLayout()
            for text, callback in (left, right):
                action = QPushButton(text)
                action.clicked.connect(callback)
                row.addWidget(action)
            layout.addLayout(row)
        self.name = QLineEdit()
        self.name.setPlaceholderText("Path name")
        layout.addWidget(self.name)
        row = QHBoxLayout()
        self.side = QComboBox()
        self.side.addItems(["unknown", "right", "left"])
        self.diameter = QDoubleSpinBox()
        self.diameter.setRange(.2, 20)
        self.diameter.setSingleStep(.2)
        self.diameter.setSuffix(" mm")
        self.diameter.setToolTip("Display tube diameter, not a safety margin")
        row.addWidget(self.side)
        row.addWidget(self.diameter)
        layout.addLayout(row)
        properties = QPushButton("Update path properties")
        properties.clicked.connect(workstation.update_path_properties)
        layout.addWidget(properties)
        self.show_original = QCheckBox("Compare original candidate")
        self.show_original.toggled.connect(workstation.refresh_nerve_paths)
        layout.addWidget(self.show_original)
        for text, callback in (("Save paths…", self.save_file), ("Open paths…", self.open_file)):
            action = QPushButton(text)
            action.clicked.connect(callback)
            layout.addWidget(action)
        hint = QLabel("Paths are separate from voxel masks. Solid nodes lie near the section; dashed projections provide context. Drag in 3D along the camera plane. New edits return the path to pending. Side is unassigned until reviewed. Tube diameter is a display property.")
        hint.setWordWrap(True)
        hint.setObjectName("subtitle")
        layout.addWidget(hint)
        layout.addStretch()

    @property
    def selected_path_id(self):
        item = self.paths.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def current_path(self):
        collection = self.workstation.nerve_paths
        return next((path for path in collection.paths if path["id"] == self.selected_path_id), None) if collection is not None else None

    def install(self):
        self.visible_ids = {path["id"] for path in self.workstation.nerve_paths.paths}
        self.selected_node = 0
        self.refresh()
        self.activity.setText("Pending paths · review each visible segment and assign its side")

    def set_busy(self, busy):
        self._busy = bool(busy)
        self.cancel_button.setVisible(self._busy)
        self.extract_button.setEnabled(not self._busy)
        self.edit_button.setEnabled(not self._busy and self.current_path() is not None)

    def clear(self):
        self.visible_ids.clear()
        self.selected_node = 0
        self.activity.clear()
        self.refresh()

    def refresh(self, selected=None):
        selected = self.selected_path_id if selected is None else selected
        self._refreshing = True
        self.paths.clear()
        collection = self.workstation.nerve_paths
        if collection is not None:
            for path in collection.paths:
                item = QListWidgetItem(f"{path['name']} · {path['side']} · {path['review']}\n{len(path['points_lps_mm'])} nodes")
                item.setData(Qt.ItemDataRole.UserRole, path["id"])
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if path["id"] in self.visible_ids else Qt.CheckState.Unchecked)
                self.paths.addItem(item)
                if selected == path["id"]:
                    self.paths.setCurrentItem(item)
            if self.paths.currentRow() < 0 and self.paths.count():
                self.paths.setCurrentRow(0)
        self._refreshing = False
        self.selection_changed()

    def selection_changed(self, *_):
        if self._refreshing:
            return
        self.workstation.finish_path_gesture(reset_pointer=True)
        path = self.current_path()
        self.node.blockSignals(True)
        if path is not None:
            self.selected_node = min(self.selected_node, len(path["points_lps_mm"]) - 1)
            self.node.setRange(1, len(path["points_lps_mm"]))
            self.node.setValue(self.selected_node + 1)
            self.node.setSuffix(f" / {len(path['points_lps_mm'])}")
            self.name.setText(path["name"])
            self.side.setCurrentText(path["side"])
            self.diameter.setValue(path["diameter_mm"])
        self.node.blockSignals(False)
        self.edit_button.setEnabled(path is not None and not self._busy)
        self.workstation.refresh_nerve_paths()

    def visibility_changed(self, item):
        if self._refreshing:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        if item.checkState() == Qt.CheckState.Checked:
            self.visible_ids.add(key)
        else:
            self.visible_ids.discard(key)
        self.workstation.refresh_nerve_paths()

    def review(self, status):
        if self.current_path() is not None and not self._busy:
            self.workstation.finish_path_gesture()
            self.workstation.nerve_paths.set_review(self.selected_path_id, status)
            self.refresh()

    def save_file(self):
        w = self.workstation
        if w.nerve_paths is None or w.job is not None:
            return
        w.finish_path_gesture()
        path, _ = QFileDialog.getSaveFileName(self, "Save source-bound canal paths", "canals.aster-paths.json", "AsterView canal paths (*.json)")
        if path:
            from .nerve_paths import export_nerve_paths
            try:
                export_nerve_paths(path, w.nerve_paths, w.volume, w.volume_fingerprint())
                w.statusBar().showMessage("Canal paths and original candidates saved locally")
            except (ValueError, OSError) as exc:
                QMessageBox.warning(self, "Unable to save paths", str(exc))

    def open_file(self):
        w = self.workstation
        if w.volume is None or w.job is not None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open canal paths for current source", "", "AsterView canal paths (*.json)")
        if path:
            from .nerve_paths import read_nerve_paths
            try:
                result = read_nerve_paths(path, w.volume, w.volume_fingerprint())
                w.install_nerve_paths(result)
            except (ValueError, OSError) as exc:
                QMessageBox.warning(self, "Unable to open paths", str(exc))
