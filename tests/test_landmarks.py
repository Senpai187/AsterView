"""Source identity, coordinate correctness and atomic human-review integration."""
from copy import deepcopy
import json
import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QInputDialog

from asterview.app import MainWindow
from asterview.landmarks import (
    FORMAT, MAX_BYTES, MAX_LANDMARKS, export_landmarks, read_landmarks,
    validate_landmarks,
)
from asterview.model import Volume
from asterview.reformat import rotation_matrix


def fingerprint(volume):
    return {
        "series_uid": "1.2.840.100.test", "frame_of_reference_uid": "1.2.840.101.test",
        "source_group": "series:phase-1", "sop_instance_digest": "a" * 64,
        "shape": list(volume.data.shape), "spacing": list(volume.spacing),
        "origin": volume.origin.tolist(), "direction": volume.direction.tolist(),
        "synthetic": False,
    }


def result_for(volume, source=None):
    return {
        "format": FORMAT, "volume": deepcopy(source or fingerprint(volume)),
        "producer": "Synthetic geometry test; no inference",
        "landmarks": [
            {"id": "point-1", "name": "Test point", "position_lps_mm": volume.center.tolist(),
             "review": "accepted", "confidence": 0.75},
        ],
    }


@pytest.fixture
def oblique_volume():
    # Translation, anisotropy and reflection make axis-aligned bounds insufficient.
    direction = rotation_matrix(24, -17, 33) @ np.diag([-1, 1, 1])
    return Volume(np.zeros((7, 9, 11), dtype=np.float32), (0.3, 0.8, 1.6),
                  np.array([140., -78., 27.]), direction)


def test_lps_points_resolve_in_oblique_reflected_source_and_are_independent(oblique_volume):
    volume = oblique_volume
    payload = result_for(volume)
    original = deepcopy(payload)
    payload["landmarks"][0]["position_lps_mm"] = volume.index_to_world([2.25, 5.5, 1.75]).tolist()
    validated = validate_landmarks(payload, volume, fingerprint(volume))
    np.testing.assert_allclose(volume.world_to_index(validated["landmarks"][0]["position_lps_mm"]), [2.25, 5.5, 1.75])
    assert validated["landmarks"][0]["review"] == "pending"
    assert payload["landmarks"][0]["review"] == "accepted"
    validated["landmarks"][0]["position_lps_mm"][0] = -123
    assert payload["landmarks"][0]["position_lps_mm"][0] != -123
    assert payload["volume"] == original["volume"]


@pytest.mark.parametrize("field,value", [
    ("series_uid", "1.2.840.other"), ("frame_of_reference_uid", "1.2.840.other"),
    ("source_group", "series:phase-2"), ("sop_instance_digest", "b" * 64),
    ("shape", [8, 9, 11]), ("spacing", [0.4, 0.8, 1.6]),
    ("origin", [140., -78., 28.]), ("direction", np.eye(3).tolist()),
    ("synthetic", True),
])
def test_geometrically_similar_or_same_uid_result_cannot_cross_source(oblique_volume, field, value):
    payload = result_for(oblique_volume)
    payload["volume"][field] = value
    original = deepcopy(payload)
    with pytest.raises(ValueError):
        validate_landmarks(payload, oblique_volume, fingerprint(oblique_volume))
    assert payload == original


def test_world_axis_box_cannot_authorize_point_outside_oblique_voxels(oblique_volume):
    volume = oblique_volume
    # A corner of an enclosing LPS AABB can be outside the oriented source box.
    bounds = np.asarray(volume.bounds).reshape(3, 2)
    candidate = bounds[:, 0]
    assert np.any(volume.world_to_index(candidate) < 0) or np.any(volume.world_to_index(candidate) > np.array(volume.dimensions) - 1)
    payload = result_for(volume)
    payload["landmarks"][0]["position_lps_mm"] = candidate.tolist()
    with pytest.raises(ValueError, match="outside"):
        validate_landmarks(payload, volume, fingerprint(volume))


@pytest.mark.parametrize("index", [
    [0., 0., 0.], [10., 8., 6.], [3.25, 5.5, 1.75],
])
def test_voxel_center_extent_includes_boundary_and_fractional_positions(oblique_volume, index):
    payload = result_for(oblique_volume)
    payload["landmarks"][0]["position_lps_mm"] = oblique_volume.index_to_world(index).tolist()
    validated = validate_landmarks(payload, oblique_volume, fingerprint(oblique_volume))
    np.testing.assert_allclose(oblique_volume.world_to_index(validated["landmarks"][0]["position_lps_mm"]), index, atol=1e-10)


@pytest.mark.parametrize("coordinate", [np.nan, np.inf, -np.inf, True, "140", 10 ** 1000])
def test_nonfinite_boolean_and_ambiguous_numeric_coordinates_are_rejected(oblique_volume, coordinate):
    payload = result_for(oblique_volume)
    payload["landmarks"][0]["position_lps_mm"][0] = coordinate
    with pytest.raises(ValueError):
        validate_landmarks(payload, oblique_volume, fingerprint(oblique_volume))


@pytest.mark.parametrize("confidence", [-0.001, 1.001, np.nan, np.inf, True, "0.9"])
def test_confidence_is_finite_bounded_numeric_producer_metadata(oblique_volume, confidence):
    payload = result_for(oblique_volume)
    payload["landmarks"][0]["confidence"] = confidence
    with pytest.raises(ValueError):
        validate_landmarks(payload, oblique_volume, fingerprint(oblique_volume))


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(format="romexis.world.points"),
    lambda p: p["landmarks"][0].update(position=[1, 2, 3]),
    lambda p: p["landmarks"][0].pop("position_lps_mm"),
    lambda p: p["landmarks"][0].update(position_lps_mm=[1, 2]),
    lambda p: p["landmarks"][0].update(review="diagnosed"),
    lambda p: p["landmarks"][0].update(id=""),
    lambda p: p["landmarks"][0].update(name="line\nbreak"),
    lambda p: p.update(producer="x" * 161),
    lambda p: p["landmarks"].append(deepcopy(p["landmarks"][0])),
    lambda p: p.update(landmarks=[p["landmarks"][0]] * (MAX_LANDMARKS + 1)),
])
def test_ambiguous_coordinate_contract_and_malformed_candidates_fail_as_a_whole(oblique_volume, mutate):
    payload = result_for(oblique_volume)
    mutate(payload)
    original = repr(payload)
    with pytest.raises(ValueError):
        validate_landmarks(payload, oblique_volume, fingerprint(oblique_volume))
    assert repr(payload) == original


def test_uncalibrated_and_single_image_sources_cannot_accept_physical_landmarks(oblique_volume):
    payload = result_for(oblique_volume)
    oblique_volume.metadata["geometry_calibrated"] = False
    with pytest.raises(ValueError, match="calibrated"):
        validate_landmarks(payload, oblique_volume, fingerprint(oblique_volume))
    single = Volume(np.zeros((1, 9, 11), dtype=np.float32), (1, 1, 1), np.zeros(3), np.eye(3))
    with pytest.raises(ValueError, match="three-dimensional"):
        validate_landmarks(result_for(single), single, fingerprint(single))


def test_export_preserves_review_but_reimport_starts_pending(oblique_volume, tmp_path):
    volume = oblique_volume
    payload = result_for(volume)
    payload["landmarks"].append({"id": "point-2", "name": "Rejected candidate",
        "position_lps_mm": volume.index_to_world([1, 2, 3]).tolist(),
        "producer": "Separate test producer", "review": "rejected"})
    path = tmp_path / "review.json"
    export_landmarks(path, payload, volume, fingerprint(volume))
    exported = json.loads(path.read_text(encoding="utf-8"))
    assert [p["review"] for p in exported["landmarks"]] == ["accepted", "rejected"]
    assert exported["landmarks"][1]["producer"] == "Separate test producer"
    assert exported["landmarks"][0]["confidence"] == 0.75
    imported = read_landmarks(path, volume, fingerprint(volume))
    assert [p["review"] for p in imported["landmarks"]] == ["pending", "pending"]
    assert payload["landmarks"][0]["review"] == "accepted"


def test_oversized_or_malformed_file_is_not_partially_loaded(oblique_volume, tmp_path):
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (MAX_BYTES + 1))
    with pytest.raises(ValueError, match="2 MiB"):
        read_landmarks(path, oblique_volume, fingerprint(oblique_volume))
    path.write_text('{"format":', encoding="utf-8")
    with pytest.raises(ValueError):
        read_landmarks(path, oblique_volume, fingerprint(oblique_volume))


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def settle(app, window):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        app.processEvents()
        if not window.render_timer.isActive() and window.slice_pending is None and not (window.slice_job and window.slice_job.isRunning()):
            app.processEvents()
            return
        time.sleep(.002)
    raise AssertionError("Background sections did not settle")


@pytest.fixture
def window(app):
    result = MainWindow()
    result.load_demo()
    settle(app, result)
    yield result
    settle(app, result)
    result.close()
    app.processEvents()


def test_invalid_late_candidate_cannot_replace_current_review_or_move_view(app, window):
    panel = window.landmark_panel
    payload = result_for(window.volume, window.volume_fingerprint())
    panel.install(payload)
    panel.points.setCurrentRow(0)
    panel.review_selected("accepted")
    before = deepcopy(panel.result)
    center, rotation = window.crosshair.copy(), window.rotation()
    invalid = deepcopy(payload)
    invalid["landmarks"].append({"id": "bad-point", "name": "Outside", "position_lps_mm": [99999, 99999, 99999]})
    with pytest.raises(ValueError, match="outside"):
        panel.install(invalid)
    assert panel.result == before
    np.testing.assert_array_equal(window.crosshair, center)
    np.testing.assert_array_equal(window.rotation(), rotation)


def test_rejected_candidates_disappear_from_sections_but_remain_reviewable(app, window):
    panel = window.landmark_panel
    panel.install(result_for(window.volume, window.volume_fingerprint()))
    assert all(len(v.landmarks) == 1 for v in window.mpr_views.values())
    panel.points.setCurrentRow(0)
    panel.review_selected("rejected")
    assert all(not v.landmarks for v in window.mpr_views.values())
    assert panel.points.count() == 1
    assert panel.selected()["review"] == "rejected"
    panel.review_selected("accepted")
    assert all(len(v.landmarks) == 1 for v in window.mpr_views.values())


def test_landmark_focus_moves_linked_location_and_preserves_oblique_rotation(app, window):
    window.rotate_planes("Axial", 31.25)
    settle(app, window)
    rotation = window.rotation()
    point = window.volume.index_to_world(np.array(window.volume.dimensions) * [0.3, 0.6, 0.4])
    payload = result_for(window.volume, window.volume_fingerprint())
    payload["landmarks"][0]["position_lps_mm"] = point.tolist()
    panel = window.landmark_panel
    panel.install(payload)
    panel.points.setCurrentRow(0)
    window.tabs.setCurrentIndex(2)
    panel.focus_selected()
    settle(app, window)
    assert window.tabs.currentIndex() == 0
    np.testing.assert_allclose(window.crosshair, point)
    np.testing.assert_array_equal(window.rotation(), rotation)
    np.testing.assert_allclose(window.volume_view._reference_center, point)
    np.testing.assert_array_equal(window.volume_view._reference_rotation, rotation)
    for view in window.mpr_views.values():
        np.testing.assert_allclose(view.crosshair, point)


def test_series_change_clears_review_and_old_point_overlays(app, window):
    panel = window.landmark_panel
    panel.install(result_for(window.volume, window.volume_fingerprint()))
    assert panel.points.count() == 1
    window.load_demo()
    settle(app, window)
    assert panel.result is None
    assert panel.points.count() == 0
    assert all(not view.landmarks for view in window.mpr_views.values())


def test_manual_bookmark_records_explicit_patient_space_and_is_reviewed(app, window, monkeypatch):
    point = window.volume.index_to_world([42.5, 51.25, 63.75])
    window.crosshair = point.copy()
    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: ("Tooth 36 reference", True))
    window.landmark_panel.bookmark()
    item = window.landmark_panel.selected()
    assert item["name"] == "Tooth 36 reference"
    assert item["producer"] == "AsterView manual landmark"
    assert item["review"] == "accepted"
    np.testing.assert_allclose(item["position_lps_mm"], point)
    assert "confidence" not in item
