"""Explicit implant/canal geometry and user-selected planning margins."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QCheckBox, QListWidget, QFileDialog, QMessageBox)


class PlanningPanel(QWidget):
    def __init__(self, workstation):
        super().__init__()
        self.workstation = workstation
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 8, 2, 2)
        title = QLabel("Implant clearance")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self.summary = QLabel("Place an implant and create a reviewed canal path.")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.summary)
        row = QHBoxLayout()
        row.addWidget(QLabel("Safety margin"))
        self.margin = QDoubleSpinBox()
        self.margin.setRange(0, 100)
        self.margin.setDecimals(2)
        self.margin.setSingleStep(.1)
        self.margin.setSuffix(" mm")
        self.margin.valueChanged.connect(workstation.change_implant_margin)
        row.addWidget(self.margin)
        layout.addLayout(row)
        self.results = QListWidget()
        self.results.setWordWrap(True)
        self.results.setMinimumHeight(160)
        layout.addWidget(self.results)
        for text, callback in (("Move selected implant", lambda: workstation.tool_combo.setCurrentText("Move implant")),
                               ("Show implant workspace", lambda: workstation.tabs.setCurrentIndex(2)),
                               ("Locate closest pair", workstation.focus_clearance),
                               ("Save plan…", self.save_file), ("Open plan…", self.open_file)):
            action = QPushButton(text)
            action.clicked.connect(callback)
            layout.addWidget(action)
        self.show_measurement = QCheckBox("Show closest-distance ruler")
        self.show_measurement.setChecked(True)
        self.show_measurement.toggled.connect(workstation.schedule_clearance)
        layout.addWidget(self.show_measurement)
        hint = QLabel("Margin is your chosen threshold; 0 means no added margin. Distances use the finite implant surface and the displayed canal tube. A negative value indicates overlap. Pending paths remain unreviewed. A displayed gap does not establish surgical safety.")
        hint.setWordWrap(True)
        hint.setObjectName("subtitle")
        layout.addWidget(hint)
        layout.addStretch()

    def save_file(self):
        w = self.workstation
        if w.volume is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save source-bound implant plan", "implants.aster-plan.json", "AsterView plan (*.json)")
        if path:
            from .implant_planning import export_planning
            try:
                export_planning(path, w.implants, w.implant_margins, w.volume, w.volume_fingerprint())
                w.statusBar().showMessage("Implant plan saved; canal paths are stored separately")
            except (ValueError, OSError) as exc:
                QMessageBox.warning(self, "Unable to save plan", str(exc))

    def open_file(self):
        w = self.workstation
        if w.volume is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open plan for current source", "", "AsterView plan (*.json)")
        if path:
            from .implant_planning import read_planning
            try:
                payload = read_planning(path, w.volume, w.volume_fingerprint())
                w.implants = payload["implants"]
                w.implant_margins = payload["margins_mm"]
                w.refresh_implants(0 if w.implants else -1)
            except (ValueError, OSError) as exc:
                QMessageBox.warning(self, "Unable to open plan", str(exc))
