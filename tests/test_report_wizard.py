"""Report workflow and export guards; all source/image data are synthetic."""
import copy
from datetime import datetime, timezone
import io
import struct
import zlib
from xml.dom import minidom
from zipfile import ZipFile

import pytest

from asterview.clinician_profile import load_profile, save_profile, ensure_clinician_profile
from asterview.report_images import make_attachment, validate_attachments, inspect_png
from asterview.reporting import export_findings, read_findings
from asterview.university_report import (create_report, today_in_riyadh, dicom_age,
                                        render_docx_bytes, source_slice_thickness)
from test_university_report import case, panel_case, qapp


def png(width=4, height=3):
    def chunk(kind, content):
        return struct.pack(">I", len(content)) + kind + content + struct.pack(">I", zlib.crc32(kind + content) & 0xffffffff)
    pixels = b"".join(b"\0" + bytes([80, 120, 160]) * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


def body_xml(document, volume, source):
    with ZipFile(io.BytesIO(render_docx_bytes(document, volume, source))) as package:
        return minidom.parseString(package.read("word/document.xml"))


def visible(dom):
    return "".join(node.firstChild.data if node.firstChild else "" for node in dom.getElementsByTagName("w:t"))


def test_profile_is_local_reusable_and_role_drives_new_report_signoff(tmp_path):
    target = tmp_path / "profile.json"
    assert load_profile(target) is None
    saved = save_profile({"role": "faculty", "name": "  Example Author  "}, target)
    assert ensure_clinician_profile(path=target) == saved  # Must not create a dialog.
    assert set(saved) == {"role", "name"}
    volume, source, document = case()
    report = create_report(document, volume, source, clinician=saved)
    assert report["profile"] == "concise"
    assert report["fields"]["faculty"] == "Example Author" and not report["fields"]["resident"]
    report = create_report(document, volume, source, clinician={"role": "resident", "name": "Example Student"})
    assert report["profile"] == "full"
    assert report["fields"]["resident"] == "Example Student" and not report["fields"]["faculty"]
    target.write_text('{"role": "invalid", "name": "Example"}')
    assert load_profile(target) is None


def test_report_day_uses_riyadh_and_recorded_age_has_exam_date_fallback():
    assert today_in_riyadh(datetime(2026, 10, 20, 22, tzinfo=timezone.utc)) == "21-Oct-2026"
    assert today_in_riyadh(datetime(2026, 10, 20, 20, tzinfo=timezone.utc)) == "20-Oct-2026"
    assert dicom_age({"patient_age": "042Y"}) == "42"
    assert dicom_age({"patient_age": "006M"}) == "6 months"
    assert dicom_age({"patient_birth_date": "20001101", "study_date": "20261020"}) == "25"
    assert dicom_age({"patient_birth_date": "20271101", "study_date": "20261020"}) == ""
    assert dicom_age({"patient_birth_date": "bad", "study_date": "20261020"}) == ""


def test_anisotropic_report_uses_recorded_slice_thickness_and_single_export_label():
    volume, source, document = case()
    volume.metadata["source_slice_thickness_mm"] = .9
    report = create_report(document, volume, source)
    assert report["fields"]["voxel_size"] == "0.9 mm"
    document["university_report"] = report
    text = visible(body_xml(document, volume, source))
    assert "Slice thickness:" in text and "0.9 mm" in text
    assert "Voxel size:" not in text and "0.7 × 1.1 × 1.4" not in text


@pytest.mark.parametrize("value", [None, "", "invalid", float("nan"), float("inf"), -1, 0, True, []])
def test_invalid_recorded_thickness_falls_back_to_source_slice_spacing(value):
    volume, source, document = case()
    volume.metadata["source_slice_thickness_mm"] = value
    assert create_report(document, volume, source)["fields"]["voxel_size"] == "1.4 mm"
    volume.metadata["geometry_calibrated"] = False
    assert source_slice_thickness(volume) == ""


def test_empty_export_sections_are_omitted_but_reviewed_text_and_signer_remain():
    volume, source, document = case()
    report = document["university_report"]
    report["fields"].update(resident="Example Resident", report_date="01-Jan-1999")
    text = visible(body_xml(document, volume, source))
    assert "Example Resident" in text and "Resident / Graduate Student:" in text
    assert "Faculty / Consultant:" not in text
    assert "Other Findings:" not in text and "Dentoalveolar Findings:" not in text
    assert "Referring Clinician / Department:" not in text
    assert "Clinician observation." in text
    assert "01-Jan-1999" not in text and today_in_riyadh() in text


def test_concise_export_has_only_narrative_sections_without_patient_demographics():
    volume, source, document = case()
    report = document["university_report"]
    report.update(profile="concise", concise_findings="Entered first line.\nEntered second line.")
    report["fields"].update(faculty="Example Faculty", recommendations="Entered action.")
    dom = body_xml(document, volume, source)
    text = visible(dom)
    assert "Entered first line." in text and "Entered second line." in text
    assert dom.getElementsByTagName("w:br")
    assert "EXAMPLE" not in text and "Person Sample" not in text
    assert "Dentoalveolar Findings:" not in text and "Technical" not in text
    assert "Example Faculty" in text and "Entered action." in text
    assert len(dom.getElementsByTagName("w:tbl")) == 3


def test_images_roundtrip_exact_source_and_follow_signoff_before_ending(tmp_path):
    volume, source, document = case()
    item = make_attachment(png(), "Axial", source, caption="Editable caption <literal>.",
                           view_state={"slice": 3, "zoom": 1.25})
    document["university_report"]["attachments"] = [item]
    document["university_report"]["fields"]["resident"] = "Example Signer"
    path = tmp_path / "with-images.json"
    export_findings(path, document, volume, source)
    assert read_findings(path, volume, source) == document
    with ZipFile(io.BytesIO(render_docx_bytes(document, volume, source))) as package:
        dom = minidom.parseString(package.read("word/document.xml"))
        body = dom.getElementsByTagName("w:body")[0]
        nodes = [node for node in body.childNodes if node.nodeType == node.ELEMENT_NODE]
        image_i = next(i for i, node in enumerate(nodes) if node.getElementsByTagName("w:drawing"))
        signer_i = next(i for i, node in enumerate(nodes) if "Example Signer" in visible(node))
        end_i = next(i for i, node in enumerate(nodes) if "end of report" in visible(node).lower())
        assert signer_i < image_i < end_i
        assert visible(nodes[image_i + 1]) == "Editable caption <literal>."
        media = [name for name in package.namelist() if "aster-report-" in name]
        assert len(media) == 1 and package.read(media[0]) == png()
        assert b"rIdAsterReportImage1" in package.read("word/_rels/document.xml.rels")


def test_report_images_reject_changed_source_corruption_and_unbounded_dimensions():
    _, source, _ = case()
    item = make_attachment(png(), "Axial", source)
    with pytest.raises(ValueError, match="different source"):
        validate_attachments([item], dict(source, origin=[1., 2., 3.]))
    broken = bytearray(png())
    broken[-5] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        inspect_png(bytes(broken))
    with pytest.raises(ValueError, match="dimensions"):
        inspect_png(png(8193, 1))
    changed = dict(item, width=10)
    with pytest.raises(ValueError, match="dimensions or hash"):
        validate_attachments([changed], source)
    with pytest.raises(ValueError, match="unique"):
        validate_attachments([item, item], source)


def test_sidebar_back_next_profiles_preserve_fields_without_new_window(qapp):
    panel, _ = panel_case()
    wizard = panel.open_report()
    assert not wizard.isWindow()
    assert panel._stack.currentWidget() is wizard
    wizard.fields["patient_name"].setText("Given Family")
    wizard.next_button.click()
    wizard.fields["patient_id"].setText("EXAMPLE-2")
    wizard.back_button.click()
    assert wizard.fields["patient_name"].text() == "Given Family"
    wizard.sections["jaws"].setPlainText("Entered jaw narrative.")
    wizard.profile.setCurrentIndex(wizard.profile.findData("concise"))
    wizard.concise_findings.setPlainText("Entered concise narrative.")
    wizard.profile.setCurrentIndex(wizard.profile.findData("full"))
    assert wizard.sections["jaws"].toPlainText() == "Entered jaw narrative."
    assert wizard.concise_findings.toPlainText() == "Entered concise narrative."
    panel.show_findings_editor(select_pending=True)
    assert panel.selected()["review"] == "pending"
    assert panel.open_report() is wizard
    panel.reset_source()
    assert not wizard.is_current() and panel.report_wizard is None
    panel.deleteLater()
    qapp.processEvents()


def test_capture_batch_is_atomic_source_bound_and_caption_edits_survive_refresh(qapp):
    panel, _ = panel_case()
    wizard = panel.open_report()
    source = panel.document["source"]
    good = {"title": "Axial", "source": source, "png": png(), "view_state": {"zoom": 2}}
    bad = dict(good, source=dict(source, origin=[0., 0., 0.]))
    with pytest.raises(ValueError):
        wizard.add_captures([good, bad])
    assert panel.document["university_report"]["attachments"] == []
    wizard.add_captures([good])
    wizard.caption.setPlainText("Edited image caption.")
    wizard.refresh_draft()
    assert panel.document["university_report"]["attachments"][0]["caption"] == "Edited image caption."
    assert wizard.images.count() == 1
    wizard.remove_image()
    assert panel.document["university_report"]["attachments"] == []
    panel.reset_source()
    with pytest.raises(ValueError, match="source changed"):
        wizard.add_captures([good])
    panel.deleteLater()
    qapp.processEvents()


def test_analysis_summary_is_source_bound_separate_from_accepted_findings(qapp):
    panel, _ = panel_case()
    wizard = panel.open_report()
    original = copy.deepcopy(panel.document["findings"])
    source = panel.document["source"]
    panel.set_analysis_status("Reviewing source stack", progress=25, busy=True)
    assert wizard.analysis_progress.value() == 25 and not wizard.analyze_button.isEnabled()
    panel.install_analysis_summary("One proposal for review.", source=source, coverage="12 sampled planes")
    assert panel.document["findings"] == original
    assert "12 sampled planes" in wizard.analysis_text.text()
    assert "One proposal" not in wizard.sections["jaws"].toPlainText()
    with pytest.raises(ValueError):
        panel.install_analysis_summary("Wrong source", source=dict(source, origin=[0., 0., 0.]))
    panel.set_analysis_status("Finished", progress=100, busy=False)
    assert wizard.analyze_button.isEnabled()
    panel.reset_source()
    panel.deleteLater()
    qapp.processEvents()


def test_narrow_sidebar_keeps_navigation_visible_without_hidden_editor_expansion(qapp):
    from PySide6.QtWidgets import QScrollArea
    from PySide6.QtGui import QFontDatabase
    from asterview.ui_shell import STYLE
    old_style = qapp.styleSheet()
    for name in ("segoeui.ttf", "segoeuib.ttf", "SegUIVar.ttf"):
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/" + name)
    qapp.setStyleSheet(STYLE)
    panel, _ = panel_case()
    area = QScrollArea()
    try:
        area.setWidgetResizable(True)
        area.setWidget(panel)
        area.resize(350, 900)
        wizard = panel.open_report()
        area.show()
        qapp.processEvents()
        assert area.horizontalScrollBar().maximum() == 0
        assert area.verticalScrollBar().maximum() == 0
        assert wizard.next_button.y() < 250
        assert wizard.next_button.isVisible()
        panel.show_findings_editor()
        qapp.processEvents()
        assert panel.open_report() is wizard
        qapp.processEvents()
        assert area.verticalScrollBar().maximum() == 0
    finally:
        panel.reset_source()
        area.close()
        area.deleteLater()
        qapp.processEvents()
        qapp.setStyleSheet(old_style)
