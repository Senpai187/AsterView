"""Analytic finite-cylinder clearance and source-bound planning settings."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import numpy as np
import pytest

from asterview.implant_planning import (
    ClearanceCancelled, closest_clearance, evaluate_clearances,
    export_planning, read_planning,
)
from asterview.model import Volume
from asterview.reformat import rotation_matrix


def implant(**changes):
    return {"id": "implant-1", "name": "Test implant", "center": [0., 0, 0],
            "axis": [0., 0, 1.], "length": 10., "diameter": 4., **changes}


def path(points=None, *, diameter=2., review="accepted", identifier="canal-1"):
    return {"id": identifier, "name": "Canal candidate", "diameter_mm": diameter,
            "review": review, "points_lps_mm": points or [[5., 0, -2], [5., 0, 2]]}


def measure(points, *, diameter=2., tolerance=.001, model=None):
    return closest_clearance(model or implant(), path(diameter=diameter), np.asarray(points, dtype=float), tolerance_mm=tolerance)


def test_side_gap_uses_both_surfaces_not_centerline_distance():
    result = measure([[5, 0, -2], [5, 0, 2]])
    assert result["distance_mm"] == pytest.approx(2.)
    assert result["centerline_signed_distance_mm"] == pytest.approx(3.)
    np.testing.assert_allclose(result["implant_point_lps"][:2], [2, 0])
    np.testing.assert_allclose(result["canal_point_lps"][:2], [4, 0])
    assert np.linalg.norm(result["implant_point_lps"] - result["canal_point_lps"]) == pytest.approx(2.)
    assert result["measurement_kind"] == "surface_gap"


def test_flat_end_cap_is_not_replaced_with_a_capsule_hemisphere():
    result = measure([[0, 0, 8]])
    assert result["distance_mm"] == pytest.approx(2.)
    np.testing.assert_allclose(result["implant_point_lps"], [0, 0, 5])
    np.testing.assert_allclose(result["canal_point_lps"], [0, 0, 7])


@pytest.mark.parametrize("sign", [-1, 1])
def test_circular_end_edge_has_exact_corner_distance(sign):
    result = measure([[5, 0, sign * 9]])
    assert result["distance_mm"] == pytest.approx(4.)
    np.testing.assert_allclose(result["implant_point_lps"], [2, 0, sign * 5])
    np.testing.assert_allclose(result["canal_point_lps"], [4.4, 0, sign * 8.2])
    assert result["approx_error_mm"] == 0


def test_segment_interior_minimum_and_closest_segment_index_are_found():
    points = [[10, -5, 0], [6, -5, 0], [6, 3, 0], [9, 5, 0]]
    result = measure(points, diameter=1.)
    assert 0 <= result["distance_mm"] - 3.5 <= result["approx_error_mm"] + 1e-10
    assert result["centerline_segment_index"] == 1
    assert result["segment_fraction"] == pytest.approx(5 / 8, abs=.001)
    np.testing.assert_allclose(result["centerline_point_lps"], [6, 0, 0], atol=.002)


def test_segment_minimum_near_cap_edge_is_not_side_or_endpoint_only():
    result = measure([[4, -2, 8], [4, 3, 8]], diameter=1.)
    expected = np.sqrt(13.) - .5
    assert 0 <= result["distance_mm"] - expected <= result["approx_error_mm"] + 1e-10
    assert result["approx_error_mm"] <= .001 + 1e-10


@pytest.mark.parametrize("point,expected", [([2, 0, 0], -.5), ([0, 0, 0], -2.5), ([0, 0, 5], -.5)])
def test_overlap_is_negative_signed_indicator_with_no_clinically_safe_claim(point, expected):
    result = measure([point], diameter=1.)
    assert result["distance_mm"] == pytest.approx(expected)
    assert result["measurement_kind"] == "signed_overlap_indicator"
    assert "safe" not in result
    assert np.isfinite(result["implant_point_lps"]).all()
    assert np.isfinite(result["canal_point_lps"]).all()


def test_centerline_crossing_implant_has_interior_minimum_and_rounded_tube_radius():
    result = measure([[-6, 0, 0], [6, 0, 0]], diameter=1.)
    assert result["distance_mm"] == pytest.approx(-2.5)
    np.testing.assert_allclose(result["centerline_point_lps"], [0, 0, 0])


def test_controlled_error_matches_independent_dense_reference_over_random_segments():
    rng = np.random.default_rng(283)
    for _ in range(18):
        points = rng.uniform(-9, 9, size=(2, 3))
        result = measure(points, diameter=1., tolerance=.002)
        t = np.linspace(0, 1, 100_001)[:, None]
        samples = points[0] + t * (points[1] - points[0])
        radial = np.linalg.norm(samples[:, :2], axis=1)
        axial = np.abs(samples[:, 2])
        outside_side, outside_cap = np.maximum(radial - 2, 0), np.maximum(axial - 5, 0)
        reference = np.hypot(outside_side, outside_cap)
        inside = (radial <= 2) & (axial <= 5)
        reference[inside] = -np.minimum(2 - radial[inside], 5 - axial[inside])
        dense_minimum = float(reference.min() - .5)
        assert result["distance_mm"] >= dense_minimum - .0003
        assert result["distance_mm"] - dense_minimum <= result["approx_error_mm"] + .0003
        assert result["approx_error_mm"] <= .002 + 1e-10


@pytest.mark.parametrize("reflected", [False, True])
def test_clearance_and_contact_points_are_rigid_transform_and_reflection_equivalent(reflected):
    points = np.array([[5., 0, 9]])
    original = measure(points)
    matrix = rotation_matrix(23, -31, 19)
    if reflected:
        matrix[:, 0] *= -1
    translation = np.array([13., -22., 31.])
    moved = implant(center=translation.tolist(), axis=(matrix @ np.array([0., 0, 1.])).tolist())
    transformed = measure(points @ matrix.T + translation, model=moved)
    assert transformed["distance_mm"] == pytest.approx(original["distance_mm"])
    for field in ("implant_point_lps", "canal_point_lps", "centerline_point_lps"):
        np.testing.assert_allclose(transformed[field], original[field] @ matrix.T + translation, atol=1e-10)


def test_duplicate_or_single_point_curve_is_finite_without_mutating_inputs():
    points = np.array([[5., 0, 0], [5., 0, 0], [5., 0, 0]])
    saved = points.copy()
    result = measure(points)
    assert result["distance_mm"] == 2
    assert result["approx_error_mm"] == 0
    np.testing.assert_array_equal(points, saved)


@pytest.mark.parametrize("change", [{"axis": [0, 0, 0]}, {"length": 0}, {"diameter": -1}, {"center": [np.nan, 0, 0]}, {"axis": [np.inf, 0, 1]}])
def test_invalid_implant_geometry_is_rejected(change):
    with pytest.raises(ValueError):
        measure([[5., 0, 0]], model=implant(**change))


def test_margin_status_is_explicit_adjustable_and_defaults_to_zero():
    model = implant()
    canal = path()
    default = evaluate_clearances([model], [canal])[0]
    assert default["margin_mm"] == 0 and default["status"] == "beyond_margin"
    within = evaluate_clearances([model], [canal], margins_mm={"implant-1": 2.1})[0]
    assert within["status"] == "within_margin"
    beyond = evaluate_clearances([model], [canal], margins_mm={"implant-1": 1.})[0]
    assert beyond["status"] == "beyond_margin"
    assert within["distance_mm"] == beyond["distance_mm"]


def test_pending_and_rejected_paths_never_become_accepted_geometric_evidence():
    pending = evaluate_clearances([implant()], [path(review="pending")])[0]
    assert pending["status"] == "unreviewed"
    assert pending["geometry_status"] == "beyond_margin"
    rejected = evaluate_clearances([implant()], [path(review="rejected")])[0]
    assert rejected["status"] == "no_path" and rejected["distance_mm"] is None
    assert evaluate_clearances([implant()], [])[0]["status"] == "no_path"
    hidden = {**path(), "visible": False}
    assert evaluate_clearances([implant()], [hidden])[0]["distance_mm"] is not None


def test_nearest_path_and_live_implant_or_path_changes_recompute_distance():
    paths = [path(identifier="far"), path(identifier="near", diameter=1.)]
    curves = {"far": np.array([[8., 0, 0]]), "near": np.array([[4., 0, 0]])}
    result = evaluate_clearances([implant()], paths, curve_samples=curves)[0]
    assert result["path_id"] == "near" and result["distance_mm"] == 1.5
    moved = evaluate_clearances([implant(center=[2., 0, 0])], paths, curve_samples=curves)[0]
    assert moved["distance_mm"] == -.5 and moved["geometry_status"] == "overlap"
    curves["near"] = np.array([[10., 0, 0]])
    corrected = evaluate_clearances([implant()], paths, curve_samples=curves)[0]
    assert corrected["path_id"] == "far" and corrected["distance_mm"] == 5.


@pytest.mark.parametrize("margin", [-1, np.nan, np.inf, True, 101])
def test_invalid_margins_are_rejected(margin):
    with pytest.raises(ValueError):
        evaluate_clearances([implant()], [path()], margins_mm={"implant-1": margin})


def test_geometry_is_reentrant_and_keeps_result_arrays_independent():
    points = np.array([[6., -5, 0], [6., 3, 0]])
    before = points.copy()
    def run(_):
        return measure(points, diameter=1.)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run, range(16)))
    assert all(item["distance_mm"] == results[0]["distance_mm"] for item in results)
    results[0]["implant_point_lps"][:] = 999
    assert not np.any(results[1]["implant_point_lps"] == 999)
    np.testing.assert_array_equal(points, before)


def test_cancellation_does_not_return_partial_measurements():
    canceled = threading.Event()
    canceled.set()
    with pytest.raises(ClearanceCancelled):
        evaluate_clearances([implant()], [path()], cancel=canceled)


def planning_source():
    v = Volume(np.zeros((21, 21, 21), dtype=np.float32), (1, 1, 1), np.array([-10., -10., -10.]), np.eye(3), metadata={"synthetic": True})
    source = {"shape": list(v.data.shape), "spacing": list(v.spacing), "origin": v.origin.tolist(), "direction": v.direction.tolist(), "synthetic": True}
    return v, source


def test_planning_settings_roundtrip_keeps_exact_source_and_explicit_margins(tmp_path):
    v, source = planning_source()
    output = tmp_path / "planning.json"
    export_planning(output, [implant()], {"implant-1": 2.5}, v, source)
    loaded = read_planning(output, v, source)
    assert loaded["source"] == source
    assert loaded["margins_mm"] == {"implant-1": 2.5}
    assert loaded["implants"][0]["id"] == "implant-1"
    assert "distance_mm" not in output.read_text()
    np.testing.assert_array_equal(v.data, np.zeros(v.data.shape))


def test_wrong_source_or_invalid_export_is_atomic_and_duplicate_json_is_rejected(tmp_path):
    v, source = planning_source()
    output = tmp_path / "planning.json"
    export_planning(output, [implant()], {}, v, source)
    before = output.read_bytes()
    wrong = copy.deepcopy(source)
    wrong["origin"][0] += 1
    with pytest.raises(ValueError, match="geometry"):
        read_planning(output, v, wrong)
    with pytest.raises(ValueError):
        export_planning(output, [implant(length=-1)], {}, v, source)
    assert output.read_bytes() == before
    output.write_text('{"format":1,"format":2}')
    with pytest.raises(ValueError, match="unique"):
        read_planning(output, v, source)


@pytest.mark.parametrize("geometry", [{"length": 40}, {"diameter": 20}])
def test_plan_cannot_import_geometry_that_the_controls_would_silently_clamp(tmp_path, geometry):
    v, source = planning_source()
    output = tmp_path / "planning.json"
    export_planning(output, [implant()], {}, v, source)
    original = output.read_bytes()
    with pytest.raises(ValueError, match="matching the planning controls"):
        export_planning(output, [implant(**geometry)], {}, v, source)
    assert output.read_bytes() == original
    payload = json.loads(original)
    payload["implants"][0].update(geometry)
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="matching the planning controls"):
        read_planning(output, v, source)
