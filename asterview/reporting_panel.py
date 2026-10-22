"""Clinician findings, optional linked bookmarks, and editable local TXT drafts."""
from __future__ import annotations

import copy
import html
import hashlib
import json
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QPlainTextEdit, QCheckBox, QFileDialog, QMessageBox, QComboBox,
    QDoubleSpinBox, QTextBrowser, QListWidgetItem, QStackedWidget, QSizePolicy, QLayout)

from .reporting import (MAX_FINDINGS, build_report, make_finding, new_findings,
                        validate_findings, export_findings, read_findings, export_text)
from .differential_support import FEATURES, suggestions
from .university_report import (SECTIONS, FIELDS, create_report, report_basis,
                                check_current_report, export_docx)


class _SidebarStack(QStackedWidget):
    """Hidden finding controls must not make the compact wizard taller."""
    def __init__(self):
        super().__init__()
        self.layout().setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.currentChanged.connect(self._current_changed)

    def _current_changed(self, index):
        for i in range(self.count()):
            policy = QSizePolicy.Policy.Preferred if i == index else QSizePolicy.Policy.Ignored
            self.widget(i).setSizePolicy(policy, policy)
        self.setMinimumSize(0, 0)
        self.updateGeometry()

    def sizeHint(self):
        return self.currentWidget().sizeHint() if self.currentWidget() is not None else super().sizeHint()

    def minimumSizeHint(self):
        return self.currentWidget().minimumSizeHint() if self.currentWidget() is not None else super().minimumSizeHint()

    def hasHeightForWidth(self):
        return self.currentWidget().hasHeightForWidth() if self.currentWidget() is not None else False

    def heightForWidth(self, width):
        widget = self.currentWidget()
        return widget.heightForWidth(width) if widget is not None else -1


class ReportingPanel(QWidget):
    findings_changed = Signal()

    def __init__(self, workstation):
        super().__init__()
        self.workstation = workstation
        self.document = None
        self._editing_location = None
        self._editor_id = None
        self._refreshing = False
        self._editing_support_ids = []
        self._epoch = self._revision = self._form_revision = 0
        self._busy = False
        self._draft_context = None
        self._report_dialog = self.report_wizard = None
        self.analysis_status = {"message": "Analyze locally, then review each proposal on its source images.",
                                "progress": None, "busy": False}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._stack = _SidebarStack()
        self._stack.currentChanged.connect(lambda *_: self._stack.updateGeometry())
        outer.addWidget(self._stack)
        self._legacy_page = QWidget()
        self._stack.addWidget(self._legacy_page)
        layout = QVBoxLayout(self._legacy_page)
        layout.setContentsMargins(2, 8, 2, 2)
        title = QLabel("Findings & report")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        limitation = QLabel("Source-linked image proposals · review before accepting")
        limitation.setObjectName("subtitle")
        limitation.setTextFormat(Qt.TextFormat.PlainText)
        limitation.setWordWrap(True)
        layout.addWidget(limitation)
        self.search_scope = QComboBox()
        self.search_scope.addItems(("Mandibular mask", "Local box"))
        self.search_scope.setToolTip("Mandibular-mask envelope only, or an explicitly selected source-axis box centered at the current position.")
        layout.addWidget(self.search_scope)
        row = QHBoxLayout()
        row.addWidget(QLabel("Box side"))
        self.search_size = QDoubleSpinBox()
        self.search_size.setRange(15, 100)
        self.search_size.setValue(50)
        self.search_size.setSuffix(" mm")
        self.search_size.setKeyboardTracking(False)
        row.addWidget(self.search_size)
        layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Minimum contrast"))
        self.search_contrast = QDoubleSpinBox()
        self.search_contrast.setDecimals(2)
        self.search_contrast.setRange(.03, .5)
        self.search_contrast.setValue(.12)
        self.search_contrast.setSingleStep(.02)
        self.search_contrast.setKeyboardTracking(False)
        row.addWidget(self.search_contrast)
        layout.addLayout(row)
        self.search_button = QPushButton("Find image candidates")
        self.search_button.setToolTip("Compute focal low-intensity image candidates in the selected coverage. This is not a diagnostic lesion model.")
        self.search_button.clicked.connect(self.search_candidates)
        layout.addWidget(self.search_button)
        self.coverage = QLabel("Limited search: normal marrow, anatomy and artifacts can be returned. No candidates does not mean normal.")
        self.coverage.setTextFormat(Qt.TextFormat.PlainText)
        self.coverage.setWordWrap(True)
        self.coverage.setObjectName("subtitle")
        layout.addWidget(self.coverage)
        label = QLabel("Clinical indication · enter verbatim")
        label.setObjectName("sectionTitle")
        layout.addWidget(label)
        self.indication = QPlainTextEdit()
        self.indication.setPlaceholderText("Clinical indication provided for this source…")
        self.indication.setMaximumHeight(72)
        layout.addWidget(self.indication)
        self.findings = QListWidget()
        self.findings.setMinimumHeight(92)
        self.findings.setWordWrap(True)
        self.findings.currentRowChanged.connect(self.select_finding)
        self.findings.itemDoubleClicked.connect(self.focus_selected)
        layout.addWidget(self.findings)
        self.observation = QPlainTextEdit()
        self.observation.setPlaceholderText("Observation you reviewed on the source images…")
        self.observation.setMaximumHeight(94)
        layout.addWidget(self.observation)
        self.differential = QPlainTextEdit()
        self.differential.setPlaceholderText("Differential / impression you enter (optional)…")
        self.differential.setMaximumHeight(72)
        layout.addWidget(self.differential)
        self.report_section = QComboBox()
        for key, label in SECTIONS.items():
            self.report_section.addItem(label, key)
        self.report_section.setCurrentIndex(self.report_section.findData("other"))
        self.report_section.setToolTip("University report section for this reviewed finding.")
        self.report_section.currentIndexChanged.connect(self._form_changed)
        layout.addWidget(self.report_section)
        row = QHBoxLayout()
        self.bookmark_button = QPushButton("Bookmark position")
        self.bookmark_button.setToolTip("Attach the current patient LPS position to this finding.")
        self.bookmark_button.clicked.connect(self.capture_location)
        row.addWidget(self.bookmark_button)
        clear_location = QPushButton("Clear")
        clear_location.setToolTip("Clear this finding's source bookmark.")
        clear_location.clicked.connect(self.clear_location)
        row.addWidget(clear_location)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.accept_button, self.reject_button = QPushButton("Accept reviewed"), QPushButton("Reject")
        self.accept_button.setToolTip("Accept this finding after reviewing the source images. Only accepted findings enter generated drafts.")
        self.accept_button.clicked.connect(lambda: self.review_finding("accepted"))
        self.reject_button.clicked.connect(lambda: self.review_finding("rejected"))
        row.addWidget(self.accept_button)
        row.addWidget(self.reject_button)
        layout.addLayout(row)
        self.morphology_toggle = QPushButton("Features & references")
        self.morphology_toggle.setToolTip("Reviewer-confirmed morphology and limited referenced differential support.")
        self.morphology_toggle.setCheckable(True)
        layout.addWidget(self.morphology_toggle)
        self.morphology_widget = QWidget()
        morphology_layout = QVBoxLayout(self.morphology_widget)
        morphology_layout.setContentsMargins(0, 0, 0, 0)
        self.features = {}
        short_labels = {"periapical": "Periapical relation", "nonvital_confirmed": "Nonvitality confirmed", "pericoronal": "Pericoronal relation",
                        "well_defined": "Well-defined margin", "multilocular": "Multilocular appearance", "expansion": "Expansion on sections",
                        "root_resorption": "Root resorption", "maxillary": "Maxillary location", "ill_defined": "Ill-defined/destructive"}
        confirmed_label = QLabel("Check only features you verified.")
        confirmed_label.setWordWrap(True)
        morphology_layout.addWidget(confirmed_label)
        for key, title in FEATURES.items():
            item = QCheckBox(short_labels[key])
            item.setToolTip(title)
            item.toggled.connect(self._features_changed)
            self.features[key] = item
            morphology_layout.addWidget(item)
        self.support_choice = QComboBox()
        self.support_choice.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.support_choice.setMinimumContentsLength(12)
        morphology_layout.addWidget(self.support_choice)
        self.evidence = QTextBrowser()
        self.evidence.setOpenExternalLinks(True)
        self.evidence.setMaximumHeight(160)
        morphology_layout.addWidget(self.evidence)
        self.support_button = QPushButton("Append consideration")
        self.support_button.setToolTip("Append the selected referenced consideration to your editable differential. It is not an automatic diagnosis.")
        self.support_button.clicked.connect(self.apply_support)
        morphology_layout.addWidget(self.support_button)
        self.support_choice.currentIndexChanged.connect(self.show_support)
        layout.addWidget(self.morphology_widget)
        self.morphology_widget.hide()
        self.morphology_toggle.toggled.connect(self.morphology_widget.setVisible)
        self.location = QLabel("No linked location")
        self.location.setTextFormat(Qt.TextFormat.PlainText)
        self.location.setObjectName("subtitle")
        self.location.setWordWrap(True)
        layout.addWidget(self.location)
        row = QHBoxLayout()
        self.add_button = QPushButton("Add finding")
        self.add_button.clicked.connect(self.add_finding)
        row.addWidget(self.add_button)
        self.update_button = QPushButton("Update selected")
        self.update_button.clicked.connect(self.update_finding)
        row.addWidget(self.update_button)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_finding)
        row.addWidget(self.remove_button)
        self.focus_button = QPushButton("Locate")
        self.focus_button.setToolTip("Go to this finding's exact source bookmark in the orthogonal views.")
        self.focus_button.clicked.connect(self.focus_selected)
        row.addWidget(self.focus_button)
        layout.addLayout(row)
        self.include_measurements = QCheckBox("Include reviewed values")
        self.include_measurements.setToolTip("Includes only values supplied by the workstation's reviewed-measurement selection, when available.")
        layout.addWidget(self.include_measurements)
        generate = QPushButton("Generate draft")
        generate.setToolTip("Replace the draft from accepted findings. Save findings first if you need to preserve the current manually edited draft.")
        generate.setObjectName("primary")
        generate.clicked.connect(self.generate_draft)
        layout.addWidget(generate)
        self.report_links = QListWidget()
        self.report_links.setWordWrap(True)
        self.report_links.setMaximumHeight(90)
        self.report_links.setToolTip("Accepted finding → exact source bookmark. Double-click to locate.")
        self.report_links.itemDoubleClicked.connect(self.focus_report_link)
        layout.addWidget(self.report_links)
        self.draft = QPlainTextEdit()
        self.draft.setMinimumHeight(160)
        self.draft.setPlaceholderText("The draft contains entered findings and selected reviewed values. Edit it before exporting.")
        layout.addWidget(self.draft, 1)
        export = QPushButton("Export TXT…")
        export.clicked.connect(self.export_draft)
        layout.addWidget(export)
        report = QPushButton("University report…")
        report.clicked.connect(self.open_report)
        layout.addWidget(report)
        row = QHBoxLayout()
        for text, callback in (("Save findings…", self.save_file), ("Open findings…", self.open_file)):
            action = QPushButton(text)
            action.clicked.connect(callback)
            row.addWidget(action)
        layout.addLayout(row)
        self.hint = QLabel("Only accepted findings enter generated drafts. Save source-bound findings to retain review, references, bookmarks and draft edits.")
        self.hint.setObjectName("subtitle")
        self.hint.setTextFormat(Qt.TextFormat.PlainText)
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        for editor in (self.observation, self.differential, self.indication, self.draft):
            editor.textChanged.connect(self._form_changed)
        self.refresh()

    def _ensure(self):
        w = self.workstation
        if getattr(w, "volume", None) is None:
            raise ValueError("Load the matching source image before recording findings.")
        if self.document is None:
            self.document = new_findings(w.volume, w.volume_fingerprint())
        else:
            self.document = validate_findings(self.document, w.volume, w.volume_fingerprint())
        return self.document

    def reset_source(self):
        self._epoch += 1
        self._dispose_report()
        self.analysis_status = {"message": "Analyze locally, then review each proposal on its source images.",
                                "progress": None, "busy": False}
        self._revision += 1
        self.document = None
        self._draft_context = None
        self._editor_id = None
        self.indication.clear()
        self.draft.clear()
        self.observation.clear()
        self.differential.clear()
        self._editing_location = None
        self._editing_support_ids = []
        self.coverage.setText("Limited search: normal anatomy and artifacts can be returned. No candidates does not mean normal.")
        self.coverage.setToolTip("")
        self.include_measurements.setChecked(False)
        self.refresh()

    clear = reset_source

    def _error(self, exc):
        QMessageBox.warning(self, "Report action could not be completed", str(exc))

    def refresh(self, selected=-1):
        self._refreshing = True
        self.findings.clear()
        for item in self.document["findings"] if self.document else []:
            label = item["observation"].splitlines()[0][:100]
            self.findings.addItem(f"{item['review'].capitalize()} · {label}")
        self._refresh_report_links()
        self.findings.setCurrentRow(selected)
        self._refreshing = False
        self.select_finding()
        self.findings_changed.emit()

    def _refresh_report_links(self):
        self.report_links.clear()
        for item in self.document["findings"] if self.document else []:
            if item["review"] == "accepted" and "location_lps_mm" in item:
                link = QListWidgetItem("Locate · " + item["observation"].splitlines()[0][:80])
                link.setData(Qt.ItemDataRole.UserRole, item["id"])
                self.report_links.addItem(link)

    def selected(self):
        row = self.findings.currentRow()
        items = self.document["findings"] if self.document else []
        return items[row] if 0 <= row < len(items) else None

    def _location_label(self):
        text = "No linked location" if self._editing_location is None else "LPS mm: " + ", ".join(f"{value:.2f}" for value in self._editing_location)
        self.location.setText(text)

    def select_finding(self, *_):
        if self._refreshing:
            return
        if self._editor_id and self.document is not None:
            old_index = next((index for index, old in enumerate(self.document["findings"]) if old["id"] == self._editor_id), -1)
            if old_index >= 0:
                try:
                    self._ensure()
                    previous = self.document["findings"][old_index]
                    candidate = self._editor_finding(self._editor_id)
                    if candidate != previous:
                        self.document["findings"][old_index] = candidate
                        self._revision += 1
                        self.findings.item(old_index).setText(candidate["review"].capitalize() + " · " + candidate["observation"].splitlines()[0][:100])
                        self._refresh_report_links()
                        self.hint.setText("Finding edits were saved locally and returned to pending review. Regenerate the draft after review.")
                        self.findings_changed.emit()
                except (ValueError, TypeError) as exc:
                    self.findings.blockSignals(True)
                    self.findings.setCurrentRow(old_index)
                    self.findings.blockSignals(False)
                    self.hint.setText("Complete the current finding before changing selection: " + str(exc))
                    return
        item = self.selected()
        self._refreshing = True
        self.observation.setPlainText(item["observation"] if item else "")
        self.differential.setPlainText(item["differential"] if item else "")
        self.report_section.setCurrentIndex(self.report_section.findData(item.get("report_section", "other") if item else "other"))
        self._editing_location = copy.deepcopy(item.get("location_lps_mm")) if item else None
        self._editor_id = item["id"] if item else None
        self._editing_support_ids = copy.deepcopy(item.get("support_ids", [])) if item else []
        for key, checkbox in self.features.items():
            checkbox.setChecked(item is not None and key in item.get("morphology", []))
        self._refreshing = False
        self._features_changed(programmatic=True)
        self._location_label()
        self.update_button.setEnabled(item is not None)
        self.remove_button.setEnabled(item is not None)
        self.focus_button.setEnabled(item is not None and "location_lps_mm" in item)
        self.accept_button.setEnabled(item is not None)
        self.reject_button.setEnabled(item is not None)

    def capture_location(self):
        try:
            self._ensure()
            w = self.workstation
            item = make_finding("Location validation", location_lps_mm=w.crosshair)
            document = new_findings(w.volume, w.volume_fingerprint())
            document["findings"] = [item]
            validate_findings(document, w.volume, w.volume_fingerprint())
            self._editing_location = item["location_lps_mm"]
            self._form_changed()
            self._location_label()
        except (ValueError, TypeError, AttributeError) as exc:
            self._error(exc)

    def clear_location(self):
        self._editing_location = None
        self._form_changed()
        self._location_label()

    def _editor_finding(self, finding_id=None):
        previous = next((item for item in self.document["findings"] if item["id"] == finding_id), None) if self.document else None
        item = make_finding(self.observation.toPlainText(), self.differential.toPlainText(),
                            self._editing_location, finding_id=finding_id,
                            morphology=[key for key, checkbox in self.features.items() if checkbox.isChecked()], support_ids=self._editing_support_ids,
                            origin=previous["origin"] if previous else "manual", method=previous["method"] if previous else "Clinician entered",
                            search_id=previous["search_id"] if previous else "",
                            report_section=self.report_section.currentData(),
                            bounds_index_zyx=previous.get("bounds_index_zyx") if previous and previous.get("location_lps_mm") == self._editing_location else None)
        if previous and all(item.get(key) == previous.get(key) for key in ("observation", "differential", "location_lps_mm", "morphology", "support_ids")):
            item["review"] = previous["review"]
        w = self.workstation
        probe = new_findings(w.volume, w.volume_fingerprint())
        probe["searches"] = copy.deepcopy(self.document["searches"])
        probe["findings"] = [item]
        validate_findings(probe, w.volume, w.volume_fingerprint())
        return item

    def add_finding(self):
        try:
            document = self._ensure()
            if len(document["findings"]) >= MAX_FINDINGS:
                raise ValueError("The source's findings limit has been reached.")
            document["findings"].append(self._editor_finding())
            self._revision += 1
            self._editor_id = None
            self.refresh(-1)
            self.hint.setText("Finding recorded locally. Select it to update, locate or remove it.")
        except (ValueError, TypeError) as exc:
            self._error(exc)

    def update_finding(self):
        row = self.findings.currentRow()
        try:
            document = self._ensure()
            item = self.selected()
            if item is None:
                return
            document["findings"][row] = self._editor_finding(item["id"])
            self._revision += 1
            self.refresh(row)
        except (ValueError, TypeError) as exc:
            self._error(exc)

    def remove_finding(self):
        row = self.findings.currentRow()
        try:
            document = self._ensure()
            if self.selected() is not None:
                document["findings"].pop(row)
                self._revision += 1
                self.refresh(-1)
        except ValueError as exc:
            self._error(exc)

    def focus_selected(self, *_):
        try:
            self._ensure()
            item = self.selected()
            if item is None or "location_lps_mm" not in item:
                return
            w = self.workstation
            w.crosshair = np.asarray(item["location_lps_mm"], dtype=np.float64)
            w.tabs.setCurrentIndex(0)
            w.refresh_references()
            w.schedule_render()
            w.statusBar().showMessage(f"{item['review'].capitalize()} · {'image candidate' if item['origin'] == 'candidate' else 'clinician finding'} · linked source location")
        except ValueError as exc:
            self._error(exc)

    def _sync(self, *, commit_editor=False):
        document = self._ensure()
        document["clinical_indication"] = self.indication.toPlainText()
        document["draft"] = self.draft.toPlainText()
        if commit_editor and (self.selected() is not None or self.observation.toPlainText().strip() or self.differential.toPlainText().strip()):
            item = self.selected()
            candidate = self._editor_finding(item["id"] if item else None)
            if item is None:
                if len(document["findings"]) >= MAX_FINDINGS:
                    raise ValueError("The source's findings limit has been reached.")
                document["findings"].append(candidate)
                self._revision += 1
                self.refresh(len(document["findings"]) - 1)
            else:
                if document["findings"][self.findings.currentRow()] != candidate:
                    document["findings"][self.findings.currentRow()] = candidate
                    self._revision += 1
        return validate_findings(document, self.workstation.volume, self.workstation.volume_fingerprint())

    def generate_draft(self):
        try:
            document = self._sync(commit_editor=True)
            measurements = self._current_measurements()
            text = build_report(document, self.workstation.volume, self.workstation.volume_fingerprint(), reviewed_measurements=measurements)
            self.draft.setPlainText(text)
            document["draft"] = text
            self.document = document
            self._draft_context = (self._epoch, self._revision, self._draft_token(document, measurements))
            self.hint.setText("Draft generated from explicitly accepted findings. Double-click a Locate bookmark to revisit its source. Edit before export." +
                              (" No reviewed measurement selection was supplied." if self.include_measurements.isChecked() and not measurements else ""))
        except (ValueError, TypeError, OSError) as exc:
            self._error(exc)

    def export_draft(self):
        try:
            self.document = self._sync(commit_editor=True)
            current = (self._epoch, self._revision, self._draft_token(self.document, self._current_measurements()))
            if self._draft_context is None or current != self._draft_context:
                raise ValueError("The draft is stale or has not been generated for these reviewed findings. Its text was retained. Save findings to preserve edits, then regenerate from the current accepted findings before TXT export.")
            path, _ = QFileDialog.getSaveFileName(self, "Export clinician-authored draft", "report-draft.txt", "Plain text (*.txt)")
            if path:
                export_text(path, self.draft.toPlainText())
                self.hint.setText("Editable clinician-authored TXT saved locally.")
        except (ValueError, OSError) as exc:
            self._error(exc)

    def save_file(self):
        try:
            document = self._sync(commit_editor=True)
            path, _ = QFileDialog.getSaveFileName(self, "Save findings for this source", "findings.aster.json", "AsterView findings (*.json)")
            if path:
                export_findings(path, document, self.workstation.volume, self.workstation.volume_fingerprint())
                self.document = document
                self.hint.setText("Findings, bookmarks and editable draft saved for the exact source series.")
        except (ValueError, OSError) as exc:
            self._error(exc)

    def open_file(self):
        try:
            self._ensure()
            path, _ = QFileDialog.getOpenFileName(self, "Open findings for the exact source", "", "AsterView findings (*.json)")
            if path:
                document = read_findings(path, self.workstation.volume, self.workstation.volume_fingerprint())
                self._epoch += 1
                self._dispose_report()
                self._editor_id = None
                self.document = document
                self._revision += 1
                self._draft_context = None
                self.indication.setPlainText(document["clinical_indication"])
                self.draft.setPlainText(document["draft"])
                self.refresh(-1)
                self.hint.setText("Source-matched clinician findings loaded. Review the draft before use.")
        except (ValueError, OSError) as exc:
            self._error(exc)

    def _form_changed(self, *_):
        if not self._refreshing:
            self._form_revision += 1

    def _current_measurements(self):
        if not self.include_measurements.isChecked():
            return []
        provider = getattr(self.workstation, "reviewed_measurements_for_report", None)
        return provider() if callable(provider) else []

    def _draft_token(self, document, measurements):
        # Draft wording may be edited freely. Source/finding/coverage or selected
        # reviewed-measurement changes require regeneration before export.
        from .reporting import validate_reviewed_measurements
        payload = {key: value for key, value in document.items() if key not in {"draft", "university_report", "analysis_summary"}}
        payload["selected_reviewed_measurements"] = validate_reviewed_measurements(measurements)
        payload["include_measurements"] = self.include_measurements.isChecked()
        return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()

    def _features_changed(self, *_, programmatic=False):
        if self._refreshing:
            return
        if not programmatic:
            self._form_changed()
        options = suggestions([key for key, item in self.features.items() if item.isChecked()])
        self.support_choice.blockSignals(True)
        self.support_choice.clear()
        for option in options:
            self.support_choice.addItem(option["title"], option)
        self.support_choice.blockSignals(False)
        self.show_support()

    def show_support(self, *_):
        option = self.support_choice.currentData()
        self.support_button.setEnabled(option is not None)
        if option is None:
            self.evidence.setPlainText("Confirm observed morphology to see the limited reference aid. No differential is inferred from the image candidate. This reference set does not cover all sites or aggressive/ill-defined lesions.")
        else:
            self.evidence.setHtml("<p>" + html.escape(option["consideration"]) + "</p><p>" + html.escape(option["limit"]) +
                                  '</p><p><a href="' + option["url"] + '">' + html.escape(option["title"]) + "</a></p>")

    def apply_support(self):
        option = self.support_choice.currentData()
        if option is None:
            return
        old = self.differential.toPlainText().strip()
        self.differential.setPlainText(old + ("\n" if old else "") + option["consideration"])
        if option["id"] not in self._editing_support_ids:
            self._editing_support_ids.append(option["id"])
        self._form_changed()
        self.hint.setText("Reference consideration appended by your action. Edit it, update the finding and review before accepting.")

    def review_finding(self, state):
        try:
            if state not in {"accepted", "rejected"}:
                raise ValueError("Choose accepted or rejected review.")
            row = self.findings.currentRow()
            if row < 0:
                return
            self.document = self._sync(commit_editor=True)
            self.document["findings"][row]["review"] = state
            self._revision += 1
            self.refresh(row)
            self.hint.setText("Finding " + state + ". Regenerate the draft after changing finding review or wording.")
        except (ValueError, TypeError) as exc:
            self._error(exc)

    def focus_report_link(self, link):
        identifier = link.data(Qt.ItemDataRole.UserRole)
        if self.document is None:
            return
        row = next((index for index, item in enumerate(self.document["findings"]) if item["id"] == identifier), -1)
        if row >= 0:
            self.findings.setCurrentRow(row)
            self.focus_selected()

    def open_report(self):
        """Public top-bar hook; a source-bound wizard inside this sidebar."""
        try:
            from .clinician_profile import load_profile
            from .report_wizard import ReportWizard
            self.document = self._sync(commit_editor=True)
            if "university_report" not in self.document:
                w = self.workstation
                self.document["university_report"] = create_report(
                    self.document, w.volume, w.volume_fingerprint(), clinician=load_profile())
            if self.report_wizard is None:
                self.report_wizard = ReportWizard(self)
                self._report_dialog = self.report_wizard  # Compatibility with saved UI integrations.
            if self._stack.indexOf(self.report_wizard) < 0:
                self._stack.addWidget(self.report_wizard)
            self._stack.removeWidget(self._legacy_page)
            self._legacy_page.hide()
            self._stack.setCurrentWidget(self.report_wizard)
            self.report_wizard.update_status()
            inspector = getattr(self.workstation, "inspector", None)
            if inspector is not None:
                inspector.show()
            tabs = getattr(self.workstation, "inspector_tabs", None)
            if tabs is not None:
                for index in range(tabs.count()):
                    if tabs.widget(index).isAncestorOf(self) or tabs.widget(index) is self:
                        tabs.setCurrentIndex(index)
                        break
            return self.report_wizard
        except (ValueError, TypeError, OSError) as exc:
            self._error(exc)
            return None

    def _dispose_report(self):
        if self.report_wizard is not None:
            if self._stack.indexOf(self._legacy_page) < 0:
                self._stack.addWidget(self._legacy_page)
            self._stack.setCurrentWidget(self._legacy_page)
            self._stack.removeWidget(self.report_wizard)
            self.report_wizard.deleteLater()
            self.report_wizard = self._report_dialog = None

    def show_findings_editor(self, select_pending=False):
        if self._stack.indexOf(self._legacy_page) < 0:
            self._stack.addWidget(self._legacy_page)
        if self.report_wizard is not None:
            self._stack.removeWidget(self.report_wizard)
            self.report_wizard.hide()
        self._stack.setCurrentWidget(self._legacy_page)
        if select_pending and self.document:
            row = next((i for i, item in enumerate(self.document["findings"])
                        if item["review"] == "pending"), -1)
            self.findings.setCurrentRow(row)

    def set_analysis_status(self, message, progress=None, busy=False):
        if not isinstance(message, str) or len(message) > 16000:
            raise ValueError("Analysis status must be bounded text.")
        if progress is not None and (not isinstance(progress, (int, float)) or
                                     not np.isfinite(progress) or not 0 <= progress <= 100):
            raise ValueError("Analysis progress must be between 0 and 100.")
        self.analysis_status = {"message": message, "progress": progress, "busy": bool(busy)}
        if self.report_wizard is not None:
            self.report_wizard.refresh_analysis_status()

    def install_analysis_summary(self, summary, *, source, coverage=""):
        from .session import _match_volume
        document = self._ensure()
        _match_volume(source, document["source"])
        proposed = copy.deepcopy(document)
        proposed["analysis_summary"] = {"summary": summary, "coverage": coverage}
        self.document = validate_findings(proposed, self.workstation.volume,
                                          self.workstation.volume_fingerprint())
        if self.report_wizard is not None:
            self.report_wizard.refresh_analysis_status()

    def add_report_images(self, captures):
        wizard = self.open_report()
        if wizard is not None:
            wizard.add_captures(captures)

    def set_busy(self, busy):
        self._busy = bool(busy)
        self.search_button.setEnabled(not self._busy)
        self.search_scope.setEnabled(not self._busy)
        self.search_size.setEnabled(not self._busy)
        self.search_contrast.setEnabled(not self._busy)

    def install_candidates(self, batch, context):
        """Apply a complete current-source result, or discard it unchanged."""
        w = self.workstation
        volume, epoch, revision, form_revision, snapshot, segmentation = context
        if (w.volume is not volume or self._epoch != epoch or self._revision != revision
                or self._form_revision != form_revision or self.document != snapshot
                or getattr(w, "segmentation", None) is not segmentation
                or (segmentation is not None and segmentation.revision != batch.mask_revision)):
            self.hint.setText("Source, anatomy or findings changed during search; candidates were discarded. Run the search again.")
            return False
        from .session import _match_volume
        from .reporting import _source
        _match_volume(batch.source, _source(volume, w.volume_fingerprint()))
        document = copy.deepcopy(self._ensure())
        candidates = []
        for item in batch.findings:
            duplicate = any(old["origin"] == "candidate" and "location_lps_mm" in old
                            and np.linalg.norm(np.asarray(old["location_lps_mm"]) - item["location_lps_mm"]) < 2.
                            for old in document["findings"])
            if not duplicate:
                candidates.append(copy.deepcopy(item))
        if len(document["findings"]) + len(candidates) > MAX_FINDINGS:
            raise ValueError("The source finding limit would be exceeded; no partial candidate result was installed.")
        document["findings"].extend(candidates)
        document["searches"].append(copy.deepcopy(batch.coverage))
        checked = validate_findings(document, volume, w.volume_fingerprint())
        self.document = checked
        self._revision += 1
        self.coverage.setText(f"{batch.coverage['scope']} · {len(candidates)} new pending candidates. "
                              "Locate and review: normal anatomy and artifacts can be returned. "
                              "The search has limited coverage; no candidates does not mean normal.")
        self.coverage.setToolTip(batch.coverage["description"])
        self.refresh(len(checked["findings"]) - len(candidates) if candidates else -1)
        self.hint.setText(f"{len(candidates)} new pending image candidates. Existing edited/accepted/rejected candidates were preserved. Locate and review serial sections; this is not a diagnostic lesion detector.")
        return True

    def search_candidates(self):
        try:
            w = self.workstation
            if getattr(w, "job", None) is not None:
                self.hint.setText("Wait for the current operation or cancel it first.")
                return
            if not callable(getattr(w, "start_job", None)):
                raise ValueError("Background image-candidate search is unavailable in this host.")
            from .abnormality_candidates import detect_radiolucency_candidates
            self.document = self._sync(commit_editor=True)
            volume, fingerprint = w.volume, w.volume_fingerprint()
            segmentation = getattr(w, "segmentation", None)
            center = None if self.search_scope.currentIndex() == 0 else np.asarray(w.crosshair).copy()
            size, contrast = self.search_size.value(), self.search_contrast.value()
            context = (volume, self._epoch, self._revision, self._form_revision, copy.deepcopy(self.document), segmentation)
            def compute(progress, cancel):
                return detect_radiolucency_candidates(volume, fingerprint, segmentation,
                    roi_center_lps=center, roi_size_mm=size, min_contrast=contrast, progress=progress, cancel=cancel)
            def completed(batch):
                try:
                    self.install_candidates(batch, context)
                except (ValueError, RuntimeError) as exc:
                    self._error(exc)
            if w.start_job(compute, completed):
                self.set_busy(True)
        except (ValueError, TypeError) as exc:
            self._error(exc)

