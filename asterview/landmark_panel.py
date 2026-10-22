"""Local landmark navigation and explicit candidate review in the inspector."""
from __future__ import annotations

import uuid

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QFileDialog, QMessageBox, QInputDialog,
)

from .landmarks import read_landmarks, export_landmarks, empty_result, MAX_LANDMARKS, validate_landmarks


class LandmarkPanel(QWidget):
    def __init__(self, workstation):
        super().__init__()
        self.workstation = workstation
        self.result = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 8, 2, 2)
        title = QLabel("Landmark review")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self.source = QLabel("Import local landmark candidates or bookmark the current position.")
        self.source.setTextFormat(Qt.TextFormat.PlainText)
        self.source.setWordWrap(True)
        self.source.setObjectName("subtitle")
        layout.addWidget(self.source)
        for text, callback in (("Import landmarks…", self.import_file),
                               ("Bookmark crosshair…", self.bookmark)):
            action = QPushButton(text)
            action.clicked.connect(callback)
            layout.addWidget(action)
        self.points = QListWidget()
        self.points.setWordWrap(True)
        self.points.itemDoubleClicked.connect(self.focus_selected)
        self.points.currentRowChanged.connect(self.selection_changed)
        layout.addWidget(self.points, 1)
        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setTextFormat(Qt.TextFormat.PlainText)
        self.detail.setObjectName("subtitle")
        layout.addWidget(self.detail)
        self.focus_button = QPushButton("Go to selected point")
        self.focus_button.clicked.connect(self.focus_selected)
        layout.addWidget(self.focus_button)
        row = QHBoxLayout()
        self.accept_button, self.reject_button = QPushButton("Accept"), QPushButton("Reject")
        self.accept_button.clicked.connect(lambda: self.review_selected("accepted"))
        self.reject_button.clicked.connect(lambda: self.review_selected("rejected"))
        row.addWidget(self.accept_button)
        row.addWidget(self.reject_button)
        layout.addLayout(row)
        self.export_button = QPushButton("Export review…")
        self.export_button.clicked.connect(self.export_file)
        layout.addWidget(self.export_button)
        hint = QLabel("Imported points start as pending. Confidence is the producer's score; acceptance records your review. No AI inference runs here.")
        hint.setWordWrap(True)
        hint.setObjectName("subtitle")
        layout.addWidget(hint)
        self.refresh()

    def clear(self):
        self.result = None
        self.refresh()

    def install(self, result):
        w = self.workstation
        self.result = validate_landmarks(result, w.volume, w.volume_fingerprint())
        self.refresh()

    def refresh(self, selected=-1):
        self.points.blockSignals(True)
        self.points.clear()
        if self.result is not None:
            self.source.setText(self.result["producer"])
            for item in self.result["landmarks"]:
                self.points.addItem(f"{item['name']} · {item['review']}")
            self.points.setCurrentRow(selected)
        else:
            self.source.setText("Import local landmark candidates or bookmark the current position.")
        self.points.blockSignals(False)
        self.export_button.setEnabled(bool(self.result and self.result["landmarks"]))
        self.selection_changed()
        self.workstation.refresh_landmarks()

    def selected(self):
        index = self.points.currentRow()
        items = self.result["landmarks"] if self.result else []
        return items[index] if 0 <= index < len(items) else None

    def selection_changed(self, *_):
        item = self.selected()
        for action in (self.focus_button, self.accept_button, self.reject_button):
            action.setEnabled(item is not None)
        if item is None:
            self.detail.setText("")
        else:
            coords = ", ".join(f"{value:.2f}" for value in item["position_lps_mm"])
            score = f"\nProducer confidence: {item['confidence']:.3f}" if "confidence" in item else ""
            self.detail.setText(f"LPS mm: {coords}{score}\nSource: {item['producer']}")

    def focus_selected(self, *_):
        item = self.selected()
        if item is None:
            return
        w = self.workstation
        w.crosshair = np.array(item["position_lps_mm"], dtype=float)
        w.tabs.setCurrentIndex(0)
        w.refresh_references()
        w.schedule_render()
        w.statusBar().showMessage(f"{item['name']} · {item['review']} landmark · linked location")

    def review_selected(self, state):
        item = self.selected()
        if item is not None:
            item["review"] = state
            self.refresh(self.points.currentRow())

    def bookmark(self):
        w = self.workstation
        if w.volume is None or not w.volume.metadata.get("geometry_calibrated", True) or min(w.volume.dimensions) < 2:
            w.statusBar().showMessage("Load a calibrated 3D volume before bookmarking landmarks")
            return
        name, accepted = QInputDialog.getText(self, "Bookmark crosshair", "Landmark or tooth label")
        if not accepted or not name.strip():
            return
        if len(name) > 96 or any(ord(c) < 32 for c in name):
            QMessageBox.warning(self, "Invalid label", "Use up to 96 printable characters.")
            return
        if self.result is None:
            self.result = empty_result(w.volume_fingerprint())
        if len(self.result["landmarks"]) >= MAX_LANDMARKS:
            w.statusBar().showMessage("The landmark limit for this volume has been reached")
            return
        self.result["landmarks"].append({"id": str(uuid.uuid4()), "name": name.strip(),
            "position_lps_mm": w.crosshair.tolist(), "review": "accepted",
            "producer": "AsterView manual landmark"})
        self.refresh(len(self.result["landmarks"]) - 1)

    def import_file(self):
        w = self.workstation
        if w.volume is None:
            w.statusBar().showMessage("Load the matching source volume first")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import landmarks for current series", "", "Landmark JSON (*.json)")
        if path:
            try:
                result = read_landmarks(path, w.volume, w.volume_fingerprint())
                self.install(result)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                QMessageBox.warning(self, "Landmarks could not be imported", str(exc))

    def export_file(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export landmark review", "landmarks.aster.json", "Landmark JSON (*.json)")
        if path:
            try:
                w = self.workstation
                export_landmarks(path, self.result, w.volume, w.volume_fingerprint())
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "Review could not be exported", str(exc))
