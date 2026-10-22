import copy
from types import SimpleNamespace

import numpy as np
import pytest

from asterview.abnormality_candidates import detect_radiolucency_candidates, CandidateSearchCancelled
from asterview.model import Volume
from asterview.segmentation import create_result
from asterview.reporting import new_findings, validate_findings, make_finding, build_report, export_findings, read_findings
from asterview.differential_support import suggestions


@pytest.fixture(scope="session")
def candidate_qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def source(volume):
    return {"shape": list(volume.data.shape), "spacing": list(volume.spacing), "origin": volume.origin.tolist(),
            "direction": volume.direction.tolist(), "synthetic": True}


def phantom(*, reflected=False, exclude=False):
    shape, spacing = (56, 64, 72), (.8, 1., 1.2)
    z, y, x = np.indices(shape)
    center_index = np.array([38., 31., 25.])
    radius = ((x - 38) * spacing[0]) ** 2 + ((y - 31) * spacing[1]) ** 2 + ((z - 25) * spacing[2]) ** 2
    domain = (z >= 7) & (z < 49) & (y >= 8) & (y < 56) & (x >= 9) & (x < 63)
    data = np.where(domain, 600 + 12 * np.sin(x / 4) + 10 * np.cos(y / 3), -500).astype(np.float32)
    cavity = radius < 3.5 ** 2
    data[cavity] = 50
    labels = np.where(domain, 2, 0).astype(np.uint16)
    labels[cavity] = 3 if exclude else 0
    angle = np.deg2rad(23)
    direction = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    if reflected:
        direction[:, 0] *= -1
    volume = Volume(data, spacing, [17, -29, 8], direction, {"synthetic": True})
    segmentation = create_result(volume, source(volume), labels,
                                 {2: {"name": "Mandible", "kind": "jaw"}, 3: {"name": "Tooth group", "kind": "tooth"}},
                                 {"name": "Synthetic fixture", "version": "1"})
    return volume, segmentation, volume.index_to_world(center_index)


@pytest.mark.parametrize("reflected", [False, True])
def test_actual_candidates_localize_focal_intensity_depression_in_source_lps_without_mutation(reflected):
    volume, segmentation, expected = phantom(reflected=reflected)
    original, mask = volume.data.copy(), segmentation.data.copy()
    batch = detect_radiolucency_candidates(volume, source(volume), segmentation)
    assert batch.findings
    nearest = min(batch.findings, key=lambda item: np.linalg.norm(np.asarray(item["location_lps_mm"]) - expected))
    assert np.linalg.norm(np.asarray(nearest["location_lps_mm"]) - expected) < 1.
    assert nearest["origin"] == "candidate" and nearest["review"] == "pending" and not nearest["differential"]
    assert "normal" in batch.coverage["description"] and batch.mask_revision == segmentation.revision
    document = new_findings(volume, source(volume))
    document["findings"], document["searches"] = batch.findings, [batch.coverage]
    validate_findings(document, volume, source(volume))
    report = build_report(document, volume, source(volume))
    assert "1. Focal" not in report and "No accepted clinician findings" in report
    np.testing.assert_array_equal(volume.data, original)
    np.testing.assert_array_equal(segmentation.data, mask)


def test_known_tooth_mask_is_excluded_and_local_roi_is_explicit():
    volume, segmentation, expected = phantom(exclude=True)
    batch = detect_radiolucency_candidates(volume, source(volume), segmentation)
    assert not any(np.linalg.norm(np.asarray(item["location_lps_mm"]) - expected) < 2 for item in batch.findings)
    local = detect_radiolucency_candidates(volume, source(volume), roi_center_lps=expected, roi_size_mm=30)
    assert local.findings and "Reviewer-selected local" in local.coverage["scope"]
    with pytest.raises(ValueError, match="mandibular segmentation"):
        detect_radiolucency_candidates(volume, source(volume))


def test_cancellation_and_stale_anatomy_never_return_partial_candidates():
    volume, segmentation, _ = phantom()
    with pytest.raises(CandidateSearchCancelled):
        detect_radiolucency_candidates(volume, source(volume), segmentation, cancel=lambda: True)
    def progress(percent, message):
        if percent == 25:
            segmentation.set_review(2, "accepted")
    with pytest.raises(RuntimeError, match="segmentation changed"):
        detect_radiolucency_candidates(volume, source(volume), segmentation, progress=progress)


def test_candidate_padding_and_nonuniform_geometry_are_not_anatomy():
    volume, segmentation, point = phantom()
    volume.metadata["nonuniform_spacing"] = True
    with pytest.raises(ValueError, match="Nonuniform"):
        detect_radiolucency_candidates(volume, source(volume), segmentation)
    volume.metadata.clear()
    volume.metadata["synthetic"] = True
    mask = np.ones(volume.data.shape, bool)
    index = np.rint(volume.world_to_index(point)).astype(int)
    mask[index[2], index[1], index[0]] = False
    padded = Volume(volume.data.copy(), volume.spacing, volume.origin, volume.direction, {"synthetic": True}, valid_mask=mask)
    document = new_findings(padded, source(padded))
    document["findings"] = [make_finding("Bookmark", location_lps_mm=point)]
    with pytest.raises(ValueError, match="padding"):
        validate_findings(document, padded, source(padded))


def test_review_lifecycle_and_saved_bookmarks_keep_only_accepted_findings_in_draft(tmp_path):
    volume, _, point = phantom()
    document = new_findings(volume, source(volume))
    document["findings"] = [make_finding("Accepted observation", "My differential", point, review="accepted", morphology=["periapical"], support_ids=["periapical_histology"]),
                             make_finding("Unreviewed candidate", location_lps_mm=point),
                             make_finding("Rejected normal space", location_lps_mm=point, review="rejected")]
    document["draft"] = "Exact clinician-edited draft"
    path = tmp_path / "findings.json"
    export_findings(path, document, volume, source(volume))
    restored = read_findings(path, volume, source(volume))
    assert restored == document
    text = build_report(restored, volume, source(volume))
    assert "Accepted observation" in text and "My differential" in text
    assert "Unreviewed candidate" not in text and "Rejected normal space" not in text
    assert document["findings"][0]["id"] in text and "https://pubmed.ncbi.nlm.nih.gov/20171356/" in text


def test_reference_aid_requires_confirmed_features_and_never_ranks_disease():
    assert suggestions([]) == []
    assert suggestions(["ill_defined", "multilocular"]) == []
    assert [item["id"] for item in suggestions(["periapical"])] == ["periapical_histology"]
    assert not any(item["id"] == "maxillary_overlap" for item in suggestions(["pericoronal"]))
    assert any(item["id"] == "maxillary_overlap" for item in suggestions(["pericoronal", "maxillary"]))


def test_panel_stale_async_results_discard_without_overwriting_review_and_edits(candidate_qapp):
    from asterview.reporting_panel import ReportingPanel
    app = candidate_qapp
    volume, segmentation, _ = phantom()
    workstation = SimpleNamespace(volume=volume, segmentation=segmentation, volume_fingerprint=lambda: source(volume))
    panel = ReportingPanel(workstation)
    panel.document = new_findings(volume, source(volume))
    context = (volume, panel._epoch, panel._revision, panel._form_revision, copy.deepcopy(panel.document), segmentation)
    batch = detect_radiolucency_candidates(volume, source(volume), segmentation)
    panel.observation.setPlainText("An edit entered while computation was running")
    assert not panel.install_candidates(batch, context)
    assert panel.document["findings"] == [] and "edit entered" in panel.observation.toPlainText()
    panel.observation.clear()
    context = (volume, panel._epoch, panel._revision, panel._form_revision, copy.deepcopy(panel.document), segmentation)
    assert panel.install_candidates(batch, context)
    original_id = panel.document["findings"][0]["id"]
    panel.review_finding("accepted")
    assert panel.document["findings"][0]["review"] == "accepted"
    panel.observation.setPlainText("Corrected source-reviewed observation")
    panel.update_finding()
    assert panel.document["findings"][0]["review"] == "pending"
    assert panel.document["findings"][0]["id"] == original_id
    panel.reset_source()
    assert not panel.install_candidates(batch, context) and panel.document is None
    panel.close()
    panel.deleteLater()
    app.processEvents()


def test_accepted_finding_edits_survive_selection_and_return_to_pending(candidate_qapp):
    from asterview.reporting_panel import ReportingPanel
    app = candidate_qapp
    volume, _, point = phantom()
    workstation = SimpleNamespace(volume=volume, volume_fingerprint=lambda: source(volume))
    panel = ReportingPanel(workstation)
    panel.document = new_findings(volume, source(volume))
    panel.document["findings"] = [make_finding("First accepted observation", "First differential", point, review="accepted"),
                                 make_finding("Second observation", location_lps_mm=point)]
    panel.refresh(0)
    first_id = panel.document["findings"][0]["id"]
    panel.observation.setPlainText("Edited first observation")
    panel.differential.setPlainText("Edited first differential")
    panel._editing_location = (point + [1, 0, 0]).tolist()
    panel.findings.setCurrentRow(1)
    first = panel.document["findings"][0]
    assert first["observation"] == "Edited first observation" and first["differential"] == "Edited first differential"
    assert first["review"] == "pending" and first["id"] == first_id
    np.testing.assert_allclose(first["location_lps_mm"], point + [1, 0, 0])
    panel.findings.setCurrentRow(0)
    assert panel.observation.toPlainText() == "Edited first observation"
    panel.observation.clear()
    panel.findings.setCurrentRow(1)
    assert panel.findings.currentRow() == 0 and panel.observation.toPlainText() == ""
    assert "Complete the current finding" in panel.hint.text()
    panel.close()
    panel.deleteLater()
    app.processEvents()


def test_local_candidate_result_invalidates_if_anatomy_is_added_mid_search(candidate_qapp):
    from asterview.reporting_panel import ReportingPanel
    volume, segmentation, point = phantom()
    workstation = SimpleNamespace(volume=volume, segmentation=None, volume_fingerprint=lambda: source(volume))
    panel = ReportingPanel(workstation)
    panel.document = new_findings(volume, source(volume))
    context = (volume, panel._epoch, panel._revision, panel._form_revision, copy.deepcopy(panel.document), None)
    batch = detect_radiolucency_candidates(volume, source(volume), roi_center_lps=point, roi_size_mm=30)
    workstation.segmentation = segmentation
    assert not panel.install_candidates(batch, context) and panel.document["findings"] == []
    panel.close()
    panel.deleteLater()
    candidate_qapp.processEvents()


def test_rejecting_reported_finding_blocks_stale_export_and_preserves_manual_draft(candidate_qapp, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    from asterview.reporting_panel import ReportingPanel
    volume, _, point = phantom()
    workstation = SimpleNamespace(volume=volume, volume_fingerprint=lambda: source(volume))
    panel = ReportingPanel(workstation)
    panel.document = new_findings(volume, source(volume))
    panel.document["findings"] = [make_finding("Source-reviewed observation", location_lps_mm=point, review="accepted")]
    panel.refresh(0)
    panel.generate_draft()
    panel.draft.appendPlainText("Clinician's retained manual wording")
    before = panel.draft.toPlainText()
    panel.review_finding("rejected")
    path = tmp_path / "stale.txt"
    errors = []
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(args[-1]))
    panel.export_draft()
    assert not path.exists() and errors and "stale" in errors[-1]
    assert panel.draft.toPlainText() == before
    panel.generate_draft()
    assert "Source-reviewed observation" not in panel.draft.toPlainText()
    panel.draft.appendPlainText("Reviewed manual wording after regeneration")
    panel.export_draft()
    assert path.exists() and "Reviewed manual wording" in path.read_text(encoding="utf-8")
    exported = path.read_text(encoding="utf-8")
    panel.observation.clear()
    panel.export_draft()
    assert "Observation cannot be empty" in errors[-1]
    assert path.read_text(encoding="utf-8") == exported
    panel.close()
    panel.deleteLater()
    candidate_qapp.processEvents()


def test_legacy_findings_are_pending_even_if_old_file_contains_review_metadata():
    volume, _, point = phantom()
    document = new_findings(volume, source(volume))
    document["format"] = "asterview.findings.v1"
    document["findings"] = [make_finding("Imported observation", location_lps_mm=point, review="accepted")]
    imported = validate_findings(document, volume, source(volume))
    assert imported["findings"][0]["review"] == "pending"
