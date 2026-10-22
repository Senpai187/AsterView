import copy
import io
from types import SimpleNamespace
from xml.dom import minidom
from zipfile import ZipFile

import numpy as np
import pytest

from asterview.model import Volume
from asterview.reporting import (new_findings, make_finding, validate_findings,
                                 export_findings, read_findings)
from asterview.university_report import (create_report, check_current_report,
    render_docx_bytes, export_docx, TEMPLATE_PATH, SLOTS, validate_report)


def case():
    v = Volume(np.zeros((8, 10, 12)), (.7, 1.1, 1.4), [12, -8, 30], np.eye(3),
               metadata={"synthetic": True, "patient_name": "Sample^Person", "patient_id": "EXAMPLE",
                         "study_date": "20261020"})
    fingerprint = dict(shape=list(v.data.shape), spacing=list(v.spacing), origin=v.origin.tolist(),
                       direction=v.direction.tolist(), synthetic=True)
    d = new_findings(v, fingerprint)
    d["clinical_indication"] = "  Exact indication & <literal text>.\nSecond line.  "
    d["findings"] = [
        make_finding("Clinician observation.", "Clinician differential.", v.center,
                     finding_id="reviewed-1", review="accepted", report_section="jaws"),
        make_finding("Pending candidate.", review="pending"),
        make_finding("Rejected candidate.", review="rejected"),
    ]
    d["university_report"] = create_report(d, v, fingerprint)
    return v, fingerprint, d


def control_text(dom, field):
    for control in dom.getElementsByTagName("w:sdt"):
        ids = control.getElementsByTagName("w:id")
        if ids and ids[0].getAttribute("w:val") == SLOTS[field]:
            content = control.getElementsByTagName("w:sdtContent")[0]
            def text(node):
                if node.nodeName == "w:br":
                    return "\n"
                if node.nodeName == "#text":
                    return node.data
                return "".join(text(child) for child in node.childNodes)
            return text(content)
    raise AssertionError(field)


def test_only_accepted_findings_seed_the_correct_sections_and_no_normals_are_invented():
    v, fp, d = case()
    report = d["university_report"]
    assert report["sections"]["jaws"] == "Clinician observation."
    assert report["sections"]["other"] == ""
    assert report["fields"]["impression"] == "Clinician differential."
    assert report["fields"]["recommendations"] == ""
    assert report["fields"]["resident"] == report["fields"]["faculty"] == ""
    assert report["fields"]["patient_name"] == "Person Sample"
    assert report["fields"]["patient_id"] == "EXAMPLE"
    assert report["finding_ids"] == ["reviewed-1"]
    assert d["clinical_indication"] == "  Exact indication & <literal text>.\nSecond line.  "
    assert v.metadata["patient_name"] == "Sample^Person"
    d["findings"] = []
    empty = create_report(d, v, fp)
    assert not any(empty["sections"].values())
    assert not empty["fields"]["impression"]


def test_refresh_preserves_entered_details_recommendations_and_never_rewrites_indication():
    v, fp, d = case()
    old = d["university_report"]
    old["fields"]["recommendations"] = "Reviewer instructions."
    old["fields"]["resident"] = "Entered signer"
    old["sections"]["jaws"] = "Previously edited narrative."
    d["findings"][0]["observation"] = "Corrected accepted finding."
    refreshed = create_report(d, v, fp, previous=old)
    assert refreshed["sections"]["jaws"] == "Corrected accepted finding."
    assert refreshed["fields"]["recommendations"] == "Reviewer instructions."
    assert refreshed["fields"]["resident"] == "Entered signer"
    assert old["sections"]["jaws"] == "Previously edited narrative."


@pytest.mark.parametrize("change", ["observation", "review", "indication", "location", "section", "removed"])
def test_changed_reviewed_evidence_blocks_export_and_retains_editable_report(change):
    v, fp, d = case()
    d["university_report"]["fields"]["recommendations"] = "Preserve my wording."
    original = copy.deepcopy(d["university_report"])
    if change == "observation":
        d["findings"][0]["observation"] = "Changed"
    elif change == "review":
        d["findings"][0]["review"] = "rejected"
    elif change == "indication":
        d["clinical_indication"] += " Changed"
    elif change == "location":
        d["findings"][0]["location_lps_mm"] = v.index_to_world([1, 2, 3]).tolist()
    elif change == "section":
        d["findings"][0]["report_section"] = "dentoalveolar"
    else:
        d["findings"].pop(0)
    with pytest.raises(ValueError, match="Refresh"):
        render_docx_bytes(d, v, fp)
    assert d["university_report"] == original


def test_report_source_guard_and_source_bound_persistence(tmp_path):
    v, fp, d = case()
    d["university_report"]["sections"]["jaws"] = "Editable interpretation."
    path = tmp_path / "findings.json"
    export_findings(path, d, v, fp)
    assert read_findings(path, v, fp) == d
    wrong = dict(fp, origin=[0., 0., 0.])
    with pytest.raises(ValueError, match="geometry"):
        export_docx(tmp_path / "wrong.docx", d, v, wrong)
    assert not (tmp_path / "wrong.docx").exists()


def test_docx_preserves_template_chrome_and_exact_text_without_executing_markup(tmp_path):
    v, fp, d = case()
    d["university_report"]["sections"]["jaws"] = "<script>literal & editable</script>\nSecond line."
    d["university_report"]["fields"]["age"] = "42"
    path = tmp_path / "report.docx"
    before = TEMPLATE_PATH.read_bytes()
    export_docx(path, d, v, fp)
    assert TEMPLATE_PATH.read_bytes() == before
    with ZipFile(path) as generated, ZipFile(TEMPLATE_PATH) as template:
        assert generated.namelist() == template.namelist()
        for part in template.namelist():
            if part not in {"word/document.xml", "word/settings.xml"}:
                assert generated.read(part) == template.read(part), part
        dom = minidom.parseString(generated.read("word/document.xml"))
        assert control_text(dom, "clinical_indication") == d["clinical_indication"]
        assert control_text(dom, "jaws") == d["university_report"]["sections"]["jaws"]
        assert control_text(dom, "patient_name") == "Person Sample"
        text = generated.read("word/document.xml").decode()
        assert "Pending candidate." not in text and "Rejected candidate." not in text
        assert "Click or tap here" not in text
        assert "DRAFT" in text and "42" in text
        assert 0 < len(dom.getElementsByTagName("w:sdt")) < len(SLOTS)
        assert len(dom.getElementsByTagName("w:keepNext")) >= 1
        assert "Cervical Spine:" not in text
        assert "Faculty / Consultant:" not in text
        assert "updateFields" in generated.read("word/settings.xml").decode()


def test_blank_template_contains_no_case_or_author_data():
    with ZipFile(TEMPLATE_PATH) as source:
        core = source.read("docProps/core.xml")
        assert b"lastModifiedBy" not in core and b"creator" not in core
        document = source.read("word/document.xml").decode()
        assert "Sample" not in document and "EXAMPLE" not in document
        assert "Click or tap here" in document


@pytest.mark.parametrize("field,value", [
    ("basis_token", "bad"), ("finding_ids", [["nested"]]), ("finding_ids", ["duplicate", "duplicate"])
])
def test_malformed_report_is_rejected(field, value):
    _, _, d = case()
    d["university_report"][field] = value
    with pytest.raises(ValueError):
        validate_report(d["university_report"])


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def panel_case():
    from asterview.reporting_panel import ReportingPanel
    v, fp, d = case()
    events = []
    w = SimpleNamespace(volume=v, volume_fingerprint=lambda: fp, crosshair=v.center.copy(),
        tabs=SimpleNamespace(setCurrentIndex=lambda index: events.append(index)),
        refresh_references=lambda: events.append("references"), schedule_render=lambda: events.append("render"),
        statusBar=lambda: SimpleNamespace(showMessage=lambda message: None))
    panel = ReportingPanel(w)
    panel.document = d
    panel.indication.setPlainText(d["clinical_indication"])
    panel.refresh(-1)
    return panel, events


def test_dialog_edit_save_link_refresh_and_source_reset(qapp, tmp_path):
    panel, events = panel_case()
    dialog = panel.open_report()
    assert dialog is not None and dialog.export_button.isEnabled()
    dialog.fields["recommendations"].setPlainText("Entered recommendation.")
    assert panel.document["university_report"]["fields"]["recommendations"] == "Entered recommendation."
    dialog.sections["jaws"].setPlainText("Manually corrected narrative.")
    dialog.locate(dialog.links.item(0))
    assert "render" in events
    assert np.allclose(panel.workstation.crosshair, panel.document["findings"][0]["location_lps_mm"])
    panel.document["findings"][0]["observation"] = "New reviewed text."
    panel.findings_changed.emit()
    assert not dialog.export_button.isEnabled()
    # Clear the finding editor's old loaded observation before refreshing this
    # externally changed example; real user edits are committed by _sync.
    panel._editor_id = None
    panel.refresh(-1)
    dialog.refresh_draft()
    assert dialog.export_button.isEnabled()
    assert dialog.sections["jaws"].toPlainText() == "New reviewed text."
    assert dialog.fields["recommendations"].toPlainText() == "Entered recommendation."
    panel.reset_source()
    assert panel._report_dialog is None and panel.document is None
    panel.close()
    panel.deleteLater()
    qapp.processEvents()


def test_export_rechecks_source_after_save_picker(qapp, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog
    panel, _ = panel_case()
    dialog = panel.open_report()
    errors = []
    monkeypatch.setattr(panel, "_error", lambda exc: errors.append(str(exc)))
    def pick(*args):
        panel.reset_source()
        return str(tmp_path / "stale.docx"), "Word document (*.docx)"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", pick)
    dialog.export_word()
    assert errors and "source changed" in errors[0].lower()
    assert not (tmp_path / "stale.docx").exists()
    panel.close()
    panel.deleteLater()
    qapp.processEvents()


def test_wizard_indication_edits_are_verbatim_without_overwriting_entered_narrative(qapp):
    panel, _ = panel_case()
    dialog = panel.open_report()
    dialog.sections["jaws"].setPlainText("Manual reviewed narrative.")
    exact = "  Do not rewrite this indication.\nSecond line.  "
    dialog.indication.setPlainText(exact)
    assert panel.indication.toPlainText() == exact
    assert panel.document["clinical_indication"] == exact
    assert dialog.export_button.isEnabled()
    assert dialog.sections["jaws"].toPlainText() == "Manual reviewed narrative."
    assert panel.document["clinical_indication"] == exact
    panel.reset_source()
    panel.close()
    panel.deleteLater()
    qapp.processEvents()


def test_old_findings_without_section_remain_importable():
    v, fp, d = case()
    del d["university_report"]
    for item in d["findings"]:
        item.pop("report_section")
    checked = validate_findings(d, v, fp)
    assert all(item["report_section"] == "other" for item in checked["findings"])


def test_unbookmarked_finding_keeps_report_open(qapp):
    panel, _ = panel_case()
    panel.document["findings"][0].pop("location_lps_mm")
    panel.document["university_report"] = create_report(
        panel.document, panel.workstation.volume, panel.workstation.volume_fingerprint())
    dialog = panel.open_report()
    dialog.locate(dialog.links.item(0))
    assert not dialog.isMinimized()
    assert "no source bookmark" in dialog.status.text()
    panel.reset_source()
    panel.close()
    panel.deleteLater()
    qapp.processEvents()
