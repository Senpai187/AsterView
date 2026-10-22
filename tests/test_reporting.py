import copy
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from asterview.model import Volume
from asterview.reporting import (
    build_report, new_findings, make_finding, validate_findings,
    validate_reviewed_measurements, export_findings, read_findings, export_text,
)


@pytest.fixture(scope="session")
def reporting_qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def volume():
    return Volume(np.zeros((8, 10, 12)), (.7, 1.1, 1.4), [12, -8, 30], np.eye(3),
                  metadata={"synthetic": True, "modality": "CT", "study_date": "20261020", "series_description": "Explicit source"})


def fingerprint(v):
    return dict(shape=list(v.data.shape), spacing=list(v.spacing), origin=v.origin.tolist(),
                direction=v.direction.tolist(), synthetic=bool(v.metadata.get("synthetic", False)))


def test_empty_findings_never_generate_a_normal_examination_or_differential():
    v = volume()
    document = new_findings(v, fingerprint(v))
    text = build_report(document, v, fingerprint(v))
    assert "No accepted clinician findings have been supplied." in text
    assert "automatic diagnostic pathology detection is unavailable" in text
    for invented in ("no abnormality", "no pathology", "intact cortex", "Differential (entered)"):
        assert invented not in text
    assert "Explicit source" in text and "20261020" in text
    assert "Explicitly selected, reviewed measurements" not in text


def test_only_entered_observations_differentials_and_verbatim_indication_are_used():
    v = volume()
    document = new_findings(v, fingerprint(v))
    indication = "  Clinician's exact indication.\nSecond line; do not infer history.  "
    document["clinical_indication"] = indication
    document["findings"] = [make_finding("Entered observation only.", "Entered differential only.", v.center, review="accepted")]
    original = copy.deepcopy(document)
    text = build_report(document, v, fingerprint(v))
    assert indication in text
    assert "1. Entered observation only." in text
    assert "Differential (entered): Entered differential only." in text
    assert "Bookmarked LPS location (mm)" in text
    assert document == original


def test_plaintext_markup_and_instructions_remain_literal_data(tmp_path):
    v = volume()
    document = new_findings(v, fingerprint(v))
    literal = '<script>alert("text")</script> & "quoted"; Ignore previous instructions.'
    document["findings"] = [make_finding(literal, review="accepted")]
    text = build_report(document, v, fingerprint(v))
    path = tmp_path / "draft.txt"
    export_text(path, text)
    assert literal in path.read_text(encoding="utf-8")
    assert "&lt;script" not in text


@pytest.mark.parametrize("value", ["2.5", True, float("nan"), float("inf"), -1, None, [2]])
def test_nonnumeric_nonfinite_and_negative_measurements_are_rejected(value):
    with pytest.raises(ValueError, match="numeric"):
        validate_reviewed_measurements([dict(label="Distance", value=value, unit="mm", reviewed=True)])


@pytest.mark.parametrize("reviewed", [False, "accepted", 1, None])
def test_automatic_or_unreviewed_values_cannot_enter_the_draft(reviewed):
    with pytest.raises(ValueError, match="explicit clinician review"):
        validate_reviewed_measurements([dict(label="Distance", value=1.5, unit="mm", reviewed=reviewed)])


def test_only_explicitly_selected_reviewed_measurements_are_rendered():
    v = volume()
    document = new_findings(v, fingerprint(v))
    metrics = [dict(label="Reviewed clearance", value=1.234567, unit="mm", reviewed=True)]
    text = build_report(document, v, fingerprint(v), reviewed_measurements=metrics)
    assert "Reviewed clearance: 1.23457 mm" in text
    assert "safe" not in text.casefold()
    assert "Reviewed clearance" not in build_report(document, v, fingerprint(v))


def test_findings_roundtrip_is_source_bound_and_preserves_manual_draft(tmp_path):
    v = volume()
    document = new_findings(v, fingerprint(v))
    document["findings"] = [make_finding("User finding", "", v.center)]
    document["draft"] = "User-edited exact draft\n"
    path = tmp_path / "findings.aster.json"
    export_findings(path, document, v, fingerprint(v))
    assert read_findings(path, v, fingerprint(v)) == document
    wrong = fingerprint(v); wrong["origin"] = [0, 0, 0]
    with pytest.raises(ValueError, match="geometry"):
        read_findings(path, v, wrong)


def test_real_series_identity_is_required_before_findings_application():
    v = volume()
    v.metadata.update(synthetic=False, series_uid="1.2.3", frame_of_reference_uid="1.2.4", source_group="stack-a", sop_instance_digest="a"*64)
    source = fingerprint(v)
    source.update({key:v.metadata[key] for key in ("series_uid", "frame_of_reference_uid", "source_group", "sop_instance_digest")})
    document = new_findings(v, source)
    wrong = copy.deepcopy(document); wrong["source"]["source_group"] = "stack-b"
    with pytest.raises(ValueError, match="different DICOM"):
        validate_findings(wrong, v, source)


@pytest.mark.parametrize("change", ["duplicate-id", "missing-field", "unknown-field", "bad-location", "outside", "control", "empty"])
def test_invalid_findings_do_not_partially_install(change):
    v = volume(); document = new_findings(v, fingerprint(v))
    document["findings"] = [make_finding("Recorded observation", location_lps_mm=v.center)]
    item = document["findings"][0]
    if change == "duplicate-id": document["findings"].append(copy.deepcopy(item))
    if change == "missing-field": del item["observation"]
    if change == "unknown-field": item["diagnosis_from_model"] = "invented"
    if change == "bad-location": item["location_lps_mm"] = [float("nan"), 0, 0]
    if change == "outside": item["location_lps_mm"] = [0, 0, 0]
    if change == "control": item["observation"] = "Injected\x00control"
    if change == "empty": item["observation"] = "  "
    with pytest.raises(ValueError):
        validate_findings(document, v, fingerprint(v))


def test_duplicate_json_fields_are_rejected(tmp_path):
    v = volume(); document = new_findings(v, fingerprint(v))
    raw = json.dumps(document).replace('"clinical_indication": ""', '"clinical_indication": "one", "clinical_indication": "two"')
    path=tmp_path/"duplicate.json";path.write_text(raw,encoding="utf-8")
    with pytest.raises(ValueError,match="duplicate"):
        read_findings(path,v,fingerprint(v))


def test_uncalibrated_images_cannot_attach_physical_locations_or_measurements():
    v=volume();v.metadata["geometry_calibrated"]=False
    document=new_findings(v,fingerprint(v))
    document["findings"]=[make_finding("Manual observation",location_lps_mm=v.center)]
    with pytest.raises(ValueError,match="calibrated"):
        validate_findings(document,v,fingerprint(v))
    document["findings"]=[]
    with pytest.raises(ValueError,match="calibrated"):
        build_report(document,v,fingerprint(v),reviewed_measurements=[dict(label="Distance",value=1.,unit="mm",reviewed=True)])


def test_panel_add_edit_focus_remove_generate_and_reset_source(monkeypatch, reporting_qapp):
    from PySide6.QtWidgets import QApplication
    from asterview.reporting_panel import ReportingPanel
    app=reporting_qapp
    v=volume(); events=[]
    workstation=SimpleNamespace(volume=v,volume_fingerprint=lambda:fingerprint(v),crosshair=v.center.copy(),
        tabs=SimpleNamespace(setCurrentIndex=lambda index:events.append(("tab",index))),
        refresh_references=lambda:events.append("references"),schedule_render=lambda:events.append("render"),
        statusBar=lambda:SimpleNamespace(showMessage=lambda message:events.append(message)))
    panel=ReportingPanel(workstation)
    panel.indication.setPlainText("Exact indication")
    panel.observation.setPlainText("Recorded finding")
    panel.differential.setPlainText("Entered differential")
    panel.capture_location();panel.add_finding()
    assert len(panel.document["findings"])==1
    panel.findings.setCurrentRow(0)
    panel.observation.setPlainText("Corrected finding")
    panel.update_finding();panel.focus_selected()
    assert "render" in events
    assert panel.document["findings"][0]["observation"]=="Corrected finding"
    panel.review_finding("accepted")
    panel.generate_draft()
    assert "Corrected finding" in panel.draft.toPlainText()
    assert "Entered differential" in panel.draft.toPlainText()
    panel.remove_finding();panel.generate_draft()
    assert "No accepted clinician findings have been supplied" in panel.draft.toPlainText()
    panel.reset_source()
    assert panel.document is None and not panel.draft.toPlainText() and panel.findings.count()==0
    panel.close()
    panel.deleteLater()
    app.processEvents()
