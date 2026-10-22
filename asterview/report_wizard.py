"""A compact sidebar report wizard with persistent steps and source-bound images."""
from __future__ import annotations

import base64
import copy
import qtawesome as qta

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QStackedWidget, QPlainTextEdit, QLineEdit, QListWidget,
    QListWidgetItem, QFileDialog, QProgressBar)

from .clinician_profile import ensure_clinician_profile, load_profile
from .university_report import (FIELDS, SECTIONS, create_report, report_basis,
                                check_current_report, export_docx, today_in_riyadh)
from .report_images import make_attachment, validate_attachments
from .session import _match_volume


class ReportWizard(QWidget):
    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self._epoch = panel._epoch
        self._volume = panel.workstation.volume
        self._loading = False
        self._clinician = load_profile()
        self._step_keys = []
        self._pages = {}
        self.fields, self.sections = {}, {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 8, 2, 2)
        title = QLabel("Report")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        row = QHBoxLayout()
        self.profile = QComboBox()
        self.profile.addItem("University · full", "full")
        self.profile.addItem("Quick · Findings / Impression / Recommendations", "concise")
        self.profile.setMinimumContentsLength(16)
        self.profile.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.profile.currentIndexChanged.connect(self.profile_changed)
        row.addWidget(self.profile, 1)
        settings = QPushButton("Profile…")
        settings.setToolTip("Change your saved role and sign-off name.")
        settings.clicked.connect(self.edit_profile)
        row.addWidget(settings)
        layout.addLayout(row)
        self.profile_name = QLabel()
        self.profile_name.setWordWrap(True)
        layout.addWidget(self.profile_name)
        self.status = QLabel()
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.step_picker = QComboBox()
        self.step_picker.setMinimumContentsLength(16)
        self.step_picker.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.step_picker.currentIndexChanged.connect(self.go_to)
        layout.addWidget(self.step_picker)
        navigation = QHBoxLayout()
        self.back_button, self.next_button = QPushButton("Back"), QPushButton("Next")
        self.back_button.setIcon(qta.icon("fa6s.chevron-left", color="#c8cbd3"))
        self.next_button.setIcon(qta.icon("fa6s.chevron-right", color="#c8cbd3"))
        self.back_button.setToolTip("Previous report step. Your text is retained.")
        self.next_button.setToolTip("Next report step. Empty fields may be skipped.")
        self.back_button.clicked.connect(lambda: self.step_picker.setCurrentIndex(self.step_picker.currentIndex() - 1))
        self.next_button.clicked.connect(lambda: self.step_picker.setCurrentIndex(self.step_picker.currentIndex() + 1))
        navigation.addWidget(self.back_button)
        navigation.addWidget(self.next_button)
        layout.addLayout(navigation)
        self.stack = QStackedWidget()
        layout.addWidget(self.stack, 1)
        # Create every editor once; navigating never reconstructs or drops text.
        full_fields = ["patient_name", "patient_id", "age", "referrer", "examination_date",
                       "clinical_indication", "region", "voxel_size", "comparison", "image_quality"]
        for key in full_fields:
            label = "Clinical Indication" if key == "clinical_indication" else FIELDS[key]
            page, body = self._page(key, label)
            editor = QPlainTextEdit() if key == "clinical_indication" else QLineEdit()
            if isinstance(editor, QPlainTextEdit):
                editor.setMinimumHeight(145)
                editor.setPlaceholderText("Enter the indication verbatim.")
                self.indication = editor
                editor.textChanged.connect(self.indication_changed)
            else:
                editor.setPlaceholderText("Enter if available")
                editor.textChanged.connect(self.store_edits)
                self.fields[key] = editor
            body.addWidget(editor)
            hint = QLabel("Prefilled values remain editable. Empty fields are omitted from the exported report.")
            hint.setWordWrap(True)
            body.addWidget(hint)
            body.addStretch(1)
        self._build_analysis()
        for key, label in SECTIONS.items():
            page, body = self._page(key, label)
            editor = QPlainTextEdit()
            editor.setMinimumHeight(190)
            editor.setPlaceholderText("Enter reviewed observations. Leave empty if no report text is supplied.")
            editor.textChanged.connect(self.store_edits)
            self.sections[key] = editor
            body.addWidget(editor)
            reminder = QLabel("This section remains in your checklist. Empty sections are omitted from export; they do not mean normal.")
            reminder.setWordWrap(True)
            body.addWidget(reminder)
        page, body = self._page("concise_findings", "Findings")
        self.concise_findings = QPlainTextEdit()
        self.concise_findings.setMinimumHeight(230)
        self.concise_findings.textChanged.connect(self.store_edits)
        body.addWidget(self.concise_findings)
        for key in ("impression", "recommendations"):
            page, body = self._page(key, FIELDS[key])
            editor = QPlainTextEdit()
            editor.setMinimumHeight(210)
            editor.setPlaceholderText("Enter your reviewed " + key + ".")
            editor.textChanged.connect(self.store_edits)
            self.fields[key] = editor
            body.addWidget(editor)
        page, body = self._page("signoff", "Sign-off")
        for key in ("resident", "faculty", "report_date"):
            body.addWidget(QLabel(FIELDS[key]))
            editor = QLineEdit()
            if key == "report_date":
                editor.setReadOnly(True)
                editor.setToolTip("Always today's date in Asia/Riyadh at export.")
            else:
                editor.textChanged.connect(self.store_edits)
            self.fields[key] = editor
            body.addWidget(editor)
        body.addStretch(1)
        self._build_images()
        self._build_export()
        panel.findings_changed.connect(self.update_status)
        panel.indication.textChanged.connect(self.sync_indication)
        self._date_timer = QTimer(self)
        self._date_timer.timeout.connect(self.refresh_date)
        self._date_timer.start(60000)
        self.load_report()

    def _page(self, key, title):
        page = QWidget()
        body = QVBoxLayout(page)
        body.setContentsMargins(0, 8, 0, 4)
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        heading.setWordWrap(True)
        body.addWidget(heading)
        self._pages[key] = (page, title)
        self.stack.addWidget(page)
        return page, body

    def _build_analysis(self):
        page, body = self._page("analysis", "Analyze and review")
        self.analysis_text = QLabel()
        self.analysis_text.setTextFormat(Qt.TextFormat.PlainText)
        self.analysis_text.setWordWrap(True)
        body.addWidget(self.analysis_text)
        self.analysis_progress = QProgressBar()
        body.addWidget(self.analysis_progress)
        self.analyze_button = QPushButton("Prepare local review")
        self.analyze_button.setToolTip("Inventory the source volume and prepare limited image candidates for your review. This does not perform a diagnostic interpretation.")
        self.analyze_button.setIcon(qta.icon("fa6s.magnifying-glass", color="#c8cbd3"))
        self.analyze_button.clicked.connect(self.analyze)
        body.addWidget(self.analyze_button)
        review = QPushButton("Review image candidates / findings")
        review.setIcon(qta.icon("fa6s.list-check", color="#c8cbd3"))
        review.setToolTip("Open source-linked proposals to correct and accept or reject them.")
        review.clicked.connect(lambda: self.panel.show_findings_editor(select_pending=True))
        body.addWidget(review)
        self.refresh_button = QPushButton("Build from reviewed findings")
        self.refresh_button.setToolTip("Replace Findings and Impression from accepted observations. Other fields and images are retained.")
        self.refresh_button.clicked.connect(self.refresh_draft)
        body.addWidget(self.refresh_button)
        self.links = QListWidget()
        self.links.setMinimumHeight(100)
        self.links.setWordWrap(True)
        self.links.itemDoubleClicked.connect(self.locate)
        body.addWidget(self.links)
        reminder = QLabel("Image-linked proposals require your review. Analysis coverage and a lack of proposals do not establish a normal examination.")
        reminder.setWordWrap(True)
        body.addWidget(reminder)

    def _build_images(self):
        page, body = self._page("images", "Images and captions")
        self.capture_choice = QComboBox()
        for title, keys in (("All four views", ["Coronal", "Sagittal", "Axial", "3D"]),
                            ("Coronal", ["Coronal"]), ("Sagittal", ["Sagittal"]),
                            ("Axial", ["Axial"]), ("3D", ["3D"]),
                            ("Panoramic", ["Panoramic"])):
            self.capture_choice.addItem(title, keys)
        body.addWidget(self.capture_choice)
        capture = QPushButton("Capture clean view")
        capture.setIcon(qta.icon("fa6s.camera", color="#c8cbd3"))
        capture.setToolTip("Capture the selected scan views with current zoom/correction, without application controls.")
        capture.clicked.connect(self.capture_images)
        body.addWidget(capture)
        self.images = QListWidget()
        self.images.setMaximumHeight(120)
        self.images.currentRowChanged.connect(self.select_image)
        body.addWidget(self.images)
        self.thumbnail = QLabel("No image selected")
        self.thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail.setMinimumHeight(110)
        self.thumbnail.setMaximumHeight(190)
        body.addWidget(self.thumbnail)
        body.addWidget(QLabel("Caption · editable in Word"))
        self.caption = QPlainTextEdit()
        self.caption.setMaximumHeight(95)
        self.caption.textChanged.connect(self.caption_changed)
        body.addWidget(self.caption)
        row = QHBoxLayout()
        for text, callback in (("↑", lambda: self.move_image(-1)), ("↓", lambda: self.move_image(1)),
                               ("Remove image", self.remove_image)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            row.addWidget(button)
        body.addLayout(row)

    def _build_export(self):
        page, body = self._page("export", "Review and export")
        self.count = QLabel()
        self.count.setWordWrap(True)
        body.addWidget(self.count)
        review = QPushButton("Review source findings")
        review.clicked.connect(self.panel.show_findings_editor)
        body.addWidget(review)
        build = QPushButton("Build from reviewed findings")
        build.clicked.connect(self.refresh_draft)
        body.addWidget(build)
        save = QPushButton("Save draft and findings…")
        save.clicked.connect(self.panel.save_file)
        body.addWidget(save)
        self.export_button = QPushButton("Export Word draft…")
        self.export_button.setObjectName("primary")
        self.export_button.clicked.connect(self.export_word)
        body.addWidget(self.export_button)
        info = QLabel("Review the report and scan before signing. Empty fields are omitted. Images and editable captions follow sign-off and precede End of Report.")
        info.setWordWrap(True)
        body.addWidget(info)
        body.addStretch(1)

    def is_current(self):
        return self.panel._epoch == self._epoch and self.panel.workstation.volume is self._volume and self.panel.document is not None

    def rebuild_steps(self, selected=None):
        full = ["patient_name", "patient_id", "age", "referrer", "examination_date",
                "clinical_indication", "region", "voxel_size", "comparison", "image_quality",
                "analysis", *SECTIONS, "impression", "recommendations", "signoff", "images", "export"]
        concise = ["analysis", "concise_findings", "impression", "recommendations", "signoff", "images", "export"]
        self._step_keys = full if self.profile.currentData() == "full" else concise
        self.step_picker.blockSignals(True)
        self.step_picker.clear()
        for index, key in enumerate(self._step_keys):
            self.step_picker.addItem(f"{index + 1}/{len(self._step_keys)} · {self._pages[key][1]}", key)
        index = self._step_keys.index(selected) if selected in self._step_keys else 0
        self.step_picker.setCurrentIndex(index)
        self.step_picker.blockSignals(False)
        self.go_to(index)

    def go_to(self, index):
        if not 0 <= index < len(self._step_keys):
            return
        self.stack.setCurrentWidget(self._pages[self._step_keys[index]][0])
        self.back_button.setEnabled(index > 0)
        self.next_button.setEnabled(index + 1 < len(self._step_keys))
        self.refresh_date()

    def profile_changed(self):
        if self._loading or not self.is_current():
            return
        self.panel.document["university_report"]["profile"] = self.profile.currentData()
        self.rebuild_steps()
        self.update_status()

    def load_report(self):
        if not self.is_current():
            return
        self._loading = True
        report = self.panel.document["university_report"]
        for key, editor in self.fields.items():
            value = today_in_riyadh() if key == "report_date" else report["fields"][key]
            editor.setPlainText(value) if isinstance(editor, QPlainTextEdit) else editor.setText(value)
        for key, editor in self.sections.items():
            editor.setPlainText(report["sections"][key])
        self.concise_findings.setPlainText(report["concise_findings"])
        self.indication.setPlainText(self.panel.indication.toPlainText())
        self.profile.setCurrentIndex(self.profile.findData(report["profile"]))
        self._loading = False
        self.rebuild_steps(self.step_picker.currentData())
        self.refresh_images()
        self.update_status()

    def store_edits(self, *_):
        if self._loading or not self.is_current():
            return
        report = self.panel.document["university_report"]
        report["fields"] = {key: (editor.toPlainText() if isinstance(editor, QPlainTextEdit) else editor.text())
                            for key, editor in self.fields.items()}
        report["fields"]["report_date"] = today_in_riyadh()
        report["sections"] = {key: editor.toPlainText() for key, editor in self.sections.items()}
        report["concise_findings"] = self.concise_findings.toPlainText()
        self.update_status()

    def refresh_date(self):
        if self.is_current() and "report_date" in self.fields:
            value = today_in_riyadh()
            self.fields["report_date"].setText(value)
            self.panel.document["university_report"]["fields"]["report_date"] = value

    def indication_changed(self):
        if self._loading or not self.is_current():
            return
        self.panel.indication.setPlainText(self.indication.toPlainText())
        self.panel.document["clinical_indication"] = self.indication.toPlainText()
        self.update_status()

    def sync_indication(self):
        if self._loading or not self.is_current():
            return
        document = self.panel.document
        report = document["university_report"]
        current = report["basis_token"] == report_basis(document)
        self._loading = True
        self.indication.setPlainText(self.panel.indication.toPlainText())
        self._loading = False
        document["clinical_indication"] = self.panel.indication.toPlainText()
        # Direct, visible indication entry is already the user's reviewed
        # wording. Keep a current narrative editable without rebuilding it.
        # This never acknowledges an outstanding accepted-finding change.
        if current:
            report["basis_token"] = report_basis(document)
        self.update_status()

    def update_status(self):
        if not self.is_current():
            self.export_button.setEnabled(False)
            return
        document = self.panel.document
        report = document["university_report"]
        current = report["basis_token"] == report_basis(document)
        self.export_button.setEnabled(current)
        accepted = sum(item["review"] == "accepted" for item in document["findings"])
        pending = sum(item["review"] == "pending" for item in document["findings"])
        self.status.setText(f"{accepted} accepted findings · {pending} pending review." if current else
                            "Reviewed findings or indication changed. Build from reviewed findings before export; your draft is retained.")
        texts = [self.concise_findings.toPlainText()] if report["profile"] == "concise" else [editor.toPlainText() for editor in self.sections.values()]
        self.count.setText(f"Findings: {sum(len(text.split()) for text in texts)} words · {len(report['attachments'])} selected images.\n"
                           "Export preserves all entered text. Check pagination in Word.")
        self.links.clear()
        for item in document["findings"]:
            if item["review"] == "accepted":
                link = QListWidgetItem(("Locate · " if "location_lps_mm" in item else "No bookmark · ") + item["observation"])
                link.setData(Qt.ItemDataRole.UserRole, item["id"])
                self.links.addItem(link)
        clinician = self._clinician
        self.profile_name.setText(("Faculty / Consultant" if clinician["role"] == "faculty" else "Student / Resident")
                                  + " · " + clinician["name"] if clinician else "Set your profile to reuse your sign-off.")
        self.refresh_analysis_status()

    def locate(self, link):
        if not self.is_current():
            return
        identifier = link.data(Qt.ItemDataRole.UserRole)
        item = next((item for item in self.panel.document["findings"] if item["id"] == identifier), None)
        if item is None or "location_lps_mm" not in item:
            self.status.setText("This finding has no source bookmark. Add one in the findings inspector.")
            return
        self.panel.focus_report_link(link)

    def edit_profile(self):
        clinician = ensure_clinician_profile(self.window(), force=True)
        if clinician is None or not self.is_current():
            return
        self._clinician = clinician
        self._loading = True
        self.fields["resident"].setText(clinician["name"] if clinician["role"] == "resident" else "")
        self.fields["faculty"].setText(clinician["name"] if clinician["role"] == "faculty" else "")
        self._loading = False
        self.store_edits()

    def refresh_draft(self):
        if not self.is_current():
            return
        try:
            self.store_edits()
            document = self.panel._sync(commit_editor=True)
            w = self.panel.workstation
            document["university_report"] = create_report(document, w.volume, w.volume_fingerprint(),
                                                         previous=document["university_report"])
            self.panel.document = document
            self.load_report()
        except (ValueError, TypeError, OSError) as exc:
            self.panel._error(exc)

    def analyze(self):
        action = getattr(self.panel.workstation, "analyze_for_report", None)
        if not callable(action):
            self.panel.set_analysis_status("Local report analysis is not configured in this host.")
            return
        try:
            self.panel.document = self.panel._sync(commit_editor=True)
            action()
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            self.panel._error(exc)

    def refresh_analysis_status(self):
        state = self.panel.analysis_status
        message = state["message"]
        summary = self.panel.document.get("analysis_summary") if self.panel.document else None
        if summary:
            message += "\n\n" + summary["summary"] + "\n\nCoverage: " + summary["coverage"]
        self.analysis_text.setText(message)
        progress = state["progress"]
        self.analysis_progress.setRange(0, 0 if state["busy"] and progress is None else 100)
        if progress is not None:
            self.analysis_progress.setValue(int(progress))
        self.analysis_progress.setVisible(state["busy"] or progress is not None)
        self.analyze_button.setEnabled(not state["busy"])

    def capture_images(self):
        action = getattr(self.panel.workstation, "capture_report_views", None)
        if not callable(action):
            self.panel._error(ValueError("Clean view capture is unavailable in this host."))
            return
        try:
            if not self.is_current():
                return
            captures = action(list(self.capture_choice.currentData()))
            self.add_captures(captures)
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            self.panel._error(exc)

    def add_captures(self, captures):
        if not self.is_current():
            raise ValueError("The report source changed during image capture.")
        if not isinstance(captures, list):
            raise ValueError("View capture must return a list of source-bound PNG images.")
        document = self.panel._ensure()
        proposed = copy.deepcopy(document["university_report"]["attachments"])
        for capture in captures:
            if not isinstance(capture, dict) or not {"source", "png", "title"}.issubset(capture):
                raise ValueError("Each captured image requires its source, title and PNG pixels.")
            _match_volume(capture["source"], document["source"])
            proposed.append(make_attachment(capture["png"], capture["title"], capture["source"],
                                            caption=capture["title"], view_state=capture.get("view_state", {})))
        document["university_report"]["attachments"] = validate_attachments(proposed, document["source"])
        self.panel.document = document
        self.refresh_images(len(proposed) - 1)
        self.update_status()

    def refresh_images(self, selected=0):
        if not self.is_current():
            return
        self.images.blockSignals(True)
        self.images.clear()
        for index, item in enumerate(self.panel.document["university_report"]["attachments"]):
            self.images.addItem(f"{index + 1}. {item['title']}")
        self.images.setCurrentRow(selected if 0 <= selected < self.images.count() else -1)
        self.images.blockSignals(False)
        self.select_image(self.images.currentRow())

    def select_image(self, row):
        if not self.is_current():
            return
        items = self.panel.document["university_report"]["attachments"]
        self.caption.blockSignals(True)
        if 0 <= row < len(items):
            item = items[row]
            self.caption.setPlainText(item["caption"])
            pixels = QImage.fromData(base64.b64decode(item["png_base64"]), "PNG")
            if pixels.isNull():
                self.thumbnail.setText("Image preview unavailable")
            else:
                self.thumbnail.setPixmap(QPixmap.fromImage(pixels).scaled(280, 180, Qt.AspectRatioMode.KeepAspectRatio,
                                                                          Qt.TransformationMode.SmoothTransformation))
        else:
            self.caption.clear()
            self.thumbnail.clear()
            self.thumbnail.setText("No image selected")
        self.caption.setEnabled(0 <= row < len(items))
        self.caption.blockSignals(False)

    def caption_changed(self):
        if not self.is_current():
            return
        row = self.images.currentRow()
        items = self.panel.document["university_report"]["attachments"]
        if 0 <= row < len(items):
            items[row]["caption"] = self.caption.toPlainText()

    def remove_image(self):
        if self.is_current():
            row = self.images.currentRow()
            items = self.panel.document["university_report"]["attachments"]
            if 0 <= row < len(items):
                items.pop(row)
                self.refresh_images(min(row, len(items) - 1))
                self.update_status()

    def move_image(self, delta):
        if self.is_current():
            row = self.images.currentRow()
            items = self.panel.document["university_report"]["attachments"]
            if 0 <= row < len(items) and 0 <= row + delta < len(items):
                items[row], items[row + delta] = items[row + delta], items[row]
                self.refresh_images(row + delta)

    def export_word(self):
        if not self.is_current():
            return
        try:
            self.store_edits()
            document = self.panel._sync(commit_editor=True)
            w = self.panel.workstation
            check_current_report(document, w.volume, w.volume_fingerprint())
            path, _ = QFileDialog.getSaveFileName(self, "Export report draft",
                                                  "CBCT-report-draft.docx", "Word document (*.docx)")
            if path:
                if not self.is_current():
                    raise ValueError("The source changed while choosing the export destination.")
                document = self.panel._sync(commit_editor=True)
                export_docx(path, document, w.volume, w.volume_fingerprint())
                self.panel.document = document
                self.status.setText("Word draft saved with editable text and captions. Review before signing.")
        except (ValueError, TypeError, OSError) as exc:
            self.panel._error(exc)
