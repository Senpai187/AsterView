"""Object browser, local inference and explicit review/correction controls."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QCheckBox, QDoubleSpinBox, QComboBox, QFileDialog, QMessageBox,
)


class SegmentationPanel(QWidget):
    def __init__(self, workstation):
        super().__init__()
        self.workstation = workstation
        self.visible_ids = set()
        self._refreshing = False
        self._busy = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 8, 2, 2)
        title = QLabel("Teeth & mandibular canals")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self.runtime = QLabel("")
        self.runtime.setTextFormat(Qt.TextFormat.PlainText)
        self.runtime.setWordWrap(True)
        self.runtime.setObjectName("subtitle")
        layout.addWidget(self.runtime)
        self.run_button = QPushButton("Segment scan automatically")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(workstation.segment_scan)
        layout.addWidget(self.run_button)
        self.tooth_runtime = QLabel("")
        self.tooth_runtime.setWordWrap(True)
        self.tooth_runtime.setObjectName("subtitle")
        layout.addWidget(self.tooth_runtime)
        self.tooth_button = QPushButton("Individual teeth & FDI")
        self.tooth_button.clicked.connect(workstation.segment_teeth)
        layout.addWidget(self.tooth_button)
        self.cancel_button = QPushButton("Cancel segmentation")
        self.cancel_button.clicked.connect(workstation.cancel_job)
        self.cancel_button.hide()
        layout.addWidget(self.cancel_button)
        self.activity = QLabel("")
        self.activity.setWordWrap(True)
        self.activity.setTextFormat(Qt.TextFormat.PlainText)
        self.activity.setObjectName("subtitle")
        layout.addWidget(self.activity)
        row = QHBoxLayout()
        for text, callback in (("Open masks…", self.import_file), ("Save masks…", self.export_file)):
            action = QPushButton(text)
            action.clicked.connect(callback)
            row.addWidget(action)
        layout.addLayout(row)
        label = QLabel("Object browser · visibility / review")
        label.setObjectName("sectionTitle")
        layout.addWidget(label)
        self.objects = QListWidget()
        self.objects.setWordWrap(True)
        self.objects.setMinimumHeight(155)
        self.objects.currentRowChanged.connect(self.selection_changed)
        self.objects.itemChanged.connect(self.visibility_changed)
        self.objects.itemDoubleClicked.connect(lambda _: workstation.focus_segment())
        layout.addWidget(self.objects, 1)
        self.focus_button = QPushButton("Go to object center")
        self.focus_button.clicked.connect(workstation.focus_segment)
        layout.addWidget(self.focus_button)
        row = QHBoxLayout()
        self.accept_button, self.reject_button = QPushButton("Accept"), QPushButton("Reject")
        self.accept_button.clicked.connect(lambda: self.review("accepted"))
        self.reject_button.clicked.connect(lambda: self.review("rejected"))
        row.addWidget(self.accept_button)
        row.addWidget(self.reject_button)
        layout.addLayout(row)
        label = QLabel("Review tooth number")
        label.setObjectName("sectionTitle")
        layout.addWidget(label)
        row = QHBoxLayout()
        self.fdi_combo = QComboBox()
        self.fdi_combo.addItem("Unassigned", None)
        for quadrant in range(1, 9):
            for number in range(1, 9 if quadrant <= 4 else 6):
                fdi = 10 * quadrant + number
                self.fdi_combo.addItem(str(fdi), fdi)
        row.addWidget(self.fdi_combo)
        self.assign_number = QPushButton("Assign FDI")
        self.assign_number.clicked.connect(self.assign_fdi)
        row.addWidget(self.assign_number)
        layout.addLayout(row)
        self.swap_number = QCheckBox("Swap if that number is occupied")
        layout.addWidget(self.swap_number)
        self.number_hint = QLabel("Select an individual tooth to review its number.")
        self.number_hint.setWordWrap(True)
        self.number_hint.setObjectName("subtitle")
        layout.addWidget(self.number_hint)
        self.undo_number = QPushButton("Undo numbering")
        self.undo_number.clicked.connect(self.undo_fdi)
        layout.addWidget(self.undo_number)
        label = QLabel("Correct on a section")
        label.setObjectName("sectionTitle")
        layout.addWidget(label)
        row = QHBoxLayout()
        self.paint_button, self.erase_button = QPushButton("Paint"), QPushButton("Erase")
        for action, mode in ((self.paint_button, "Paint mask"), (self.erase_button, "Erase mask")):
            action.setCheckable(True)
            action.clicked.connect(lambda checked=False, m=mode: self.choose_tool(m))
            row.addWidget(action)
        layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Radius"))
        self.radius = QDoubleSpinBox()
        self.radius.setRange(.3, 15)
        self.radius.setSingleStep(.5)
        self.radius.setValue(2)
        self.radius.setSuffix(" mm")
        self.radius.valueChanged.connect(self.selection_changed)
        row.addWidget(self.radius)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.undo_button = QPushButton("Undo stroke")
        self.undo_button.setToolTip("Undo the last mask stroke · Ctrl+Z")
        self.undo_button.clicked.connect(workstation.undo_mask)
        row.addWidget(self.undo_button)
        navigate = QPushButton("Navigate")
        navigate.clicked.connect(lambda: self.choose_tool("Navigate"))
        row.addWidget(navigate)
        layout.addLayout(row)
        self.overwrite = QCheckBox("Allow painting over other objects")
        layout.addWidget(self.overwrite)
        self.show_2d, self.show_3d = QCheckBox("Show section overlays"), QCheckBox("Show 3D objects")
        for action in (self.show_2d, self.show_3d):
            action.setChecked(True)
            action.toggled.connect(self.display_changed)
            layout.addWidget(action)
        row = QHBoxLayout()
        row.addWidget(QLabel("Overlay"))
        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(.05, .85)
        self.opacity.setSingleStep(.05)
        self.opacity.setValue(.35)
        self.opacity.valueChanged.connect(self.display_changed)
        row.addWidget(self.opacity)
        layout.addLayout(row)
        hint = QLabel("Automatic candidates and FDI labels need review. Anatomy inference supplies tooth groups; Individual teeth & FDI creates separate instances. Canals represent the bony nerve pathway. Overlays show the center slice. Edits return affected objects to pending.")
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.refresh_runtime()
        self.refresh()

    @property
    def selected_label(self):
        item = self.objects.currentItem()
        return int(item.data(Qt.ItemDataRole.UserRole)) if item is not None else None

    def refresh_runtime(self):
        try:
            from .inference import detect_model_status
            status = detect_model_status()
        except (ImportError, OSError, ValueError) as exc:
            status = {"ready": False, "reason": str(exc)}
        self._runtime_ready = bool(status.get("ready"))
        self.runtime.setText("DentalSegmentator · local GPU inference" if self._runtime_ready
                             else "AI runtime unavailable · " + status.get("reason", "Run Setup AI.ps1"))
        try:
            from .tooth_inference import detect_tooth_model_status
            tooth_status = detect_tooth_model_status()
        except (ImportError, OSError, ValueError) as exc:
            tooth_status = {"ready": False, "reason": str(exc)}
        self._tooth_ready = bool(tooth_status.get("ready"))
        self.tooth_runtime.setText("ToothSeg · individual teeth and FDI" if self._tooth_ready else "Tooth AI unavailable · run Setup Tooth AI.ps1")
        self._set_run_enabled()

    def _set_run_enabled(self):
        volume = self.workstation.volume
        eligible = volume is not None and volume.metadata.get("modality", "").upper() in ("CT", "CBCT")
        self.run_button.setEnabled(self._runtime_ready and not self._busy and eligible)
        self.run_button.setToolTip("Open a calibrated CT or CBCT study first" if not eligible else "Run the local DentalSegmentator model")
        result = self.workstation.segmentation
        has_teeth = result is not None and any(label["kind"] == "tooth" for label in result.labels.values())
        self.tooth_button.setEnabled(self._tooth_ready and not self._busy and eligible and has_teeth)
        self.tooth_button.setToolTip("Run anatomy segmentation first to define a bounded dental region" if not has_teeth else "Run independent tooth instances and learned FDI prediction locally")

    def set_busy(self, busy):
        self._busy = busy
        self.cancel_button.setVisible(busy)
        self._set_run_enabled()

    def install(self):
        result = self.workstation.segmentation
        self.visible_ids = {key for key, value in result.labels.items() if value["kind"] in ("tooth", "canal")}
        if not self.visible_ids:
            self.visible_ids = set(result.labels)
        self.refresh(selected=min(self.visible_ids) if self.visible_ids else None)
        self.activity.setText("Automatic/imported masks · pending review")

    def clear(self):
        self.visible_ids = set()
        self.activity.setText("")
        self.refresh()

    def refresh(self, selected=None):
        current = self.selected_label if selected is None else selected
        result = self.workstation.segmentation
        self._refreshing = True
        self.objects.clear()
        if result is not None:
            for key, value in result.labels.items():
                item = QListWidgetItem(f"{value['name']} · {value['review']}")
                item.setData(Qt.ItemDataRole.UserRole, key)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if key in self.visible_ids else Qt.CheckState.Unchecked)
                item.setToolTip(f"Object {key} · {result.provenance['name']} {result.provenance['version']}")
                self.objects.addItem(item)
                if current == key:
                    self.objects.setCurrentItem(item)
            if self.objects.currentRow() < 0 and self.objects.count():
                self.objects.setCurrentRow(0)
        self._refreshing = False
        self._set_run_enabled()
        self.selection_changed()

    def selection_changed(self, *_):
        if self._refreshing:
            return
        w = self.workstation
        selected = self.selected_label
        enabled = w.segmentation is not None and selected in w.segmentation.labels
        for action in (self.focus_button, self.accept_button, self.reject_button,
                       self.paint_button, self.erase_button, self.undo_button):
            action.setEnabled(enabled)
        color = "#e5ba70"
        if enabled:
            color = w.segmentation.labels[selected].get("color", color)
        label = w.segmentation.labels[selected] if enabled else {}
        instance = bool(label.get("instance", "fdi" in label))
        for action in (self.fdi_combo, self.assign_number, self.swap_number):
            action.setEnabled(instance)
        self.undo_number.setEnabled(w.segmentation is not None and bool(w.segmentation._number_undo))
        self.fdi_combo.setCurrentIndex(max(0, self.fdi_combo.findData(label.get("fdi"))))
        prediction = label.get("predicted_fdi")
        self.number_hint.setText((f"Original model suggestion: {prediction} · verify against the scan" if prediction is not None else "Unnumbered instance · assign after review") if instance else "Grouped dentitions cannot receive one FDI number.")
        for view in getattr(w, "all_slices", []):
            view.brush_radius_mm = self.radius.value()
            view.brush_color = color
            view.update()

    def assign_fdi(self):
        w = self.workstation
        if self.selected_label is None:
            return
        w.finish_mask_stroke(reset_pointer=True)
        try:
            w.segmentation.assign_fdi(self.selected_label, self.fdi_combo.currentData(), swap=self.swap_number.isChecked())
            self.refresh()
            w.refresh_segmentation()
        except ValueError as exc:
            QMessageBox.warning(self, "Unable to assign FDI number", str(exc))

    def undo_fdi(self):
        if self.workstation.segmentation is not None:
            self.workstation.segmentation.undo_fdi()
            self.refresh()
            self.workstation.refresh_segmentation()

    def visibility_changed(self, item):
        if self._refreshing:
            return
        key = int(item.data(Qt.ItemDataRole.UserRole))
        if item.checkState() == Qt.CheckState.Checked:
            self.visible_ids.add(key)
        else:
            self.visible_ids.discard(key)
        self.display_changed()

    def display_changed(self, *_):
        self.workstation.refresh_segmentation()

    def review(self, status):
        key = self.selected_label
        if key is not None:
            self.workstation.segmentation.set_review(key, status)
            self.refresh(key)
            self.display_changed()

    def choose_tool(self, mode):
        w = self.workstation
        w.tabs.setCurrentIndex(0)
        if mode != "Navigate" and self.selected_label is None:
            return
        if self.selected_label is not None and mode != "Navigate":
            self.visible_ids.add(self.selected_label)
            self.refresh(self.selected_label)
        w.tool_combo.setCurrentText(mode)

    def sync_tool(self, mode):
        self.paint_button.setChecked(mode == "Paint mask")
        self.erase_button.setChecked(mode == "Erase mask")

    def import_file(self):
        w = self.workstation
        if w.volume is None or w.job is not None:
            w.statusBar().showMessage("Load the matching scan and wait for the current operation first")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open masks for current source volume", "", "AsterView masks (*.npz)")
        if path:
            from .segmentation import read_segmentation
            volume, fingerprint = w.volume, w.volume_fingerprint()
            w.start_job(lambda progress, cancel: (id(volume), read_segmentation(path, volume, fingerprint)), w.segmentation_ready)

    def export_file(self):
        w = self.workstation
        if w.segmentation is None or w.job is not None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save reviewed source-grid masks", "masks.aster.npz", "AsterView masks (*.npz)")
        if path:
            from .segmentation import export_segmentation
            result, volume, fingerprint = w.segmentation, w.volume, w.volume_fingerprint()
            def save(progress, cancel):
                export_segmentation(path, result, volume, fingerprint)
                return "Segmentation masks and review saved locally"
            w.start_job(save, w.statusBar().showMessage)
