"""Source-bound canal editing and supported-mask trajectory extraction."""
import copy
import json
import threading

import numpy as np
import pytest
from scipy.ndimage import binary_dilation

from asterview.model import Volume
from asterview.nerve_paths import (
    MAX_PATHS, NervePathCollection, PathExtractionCancelled, create_paths,
    export_nerve_paths, extract_canal_paths, read_nerve_paths, sample_path,
)
from asterview.reformat import rotation_matrix


def volume(*, reflected=False, real=False, oblique=True):
    direction = rotation_matrix(14, -19, 27) if oblique else np.eye(3)
    if reflected:
        direction[:, 0] *= -1
    metadata = {"synthetic": not real, "modality": "CT"}
    if real:
        metadata.update(series_uid="1.2.3", frame_of_reference_uid="1.2.4", source_group="test-stack", sop_instance_digest="a" * 64)
    return Volume(np.zeros((41, 41, 41), dtype=np.float32), (.5, .75, 1.2), np.array([-10., -15., -24.]), direction, metadata=metadata)


def fingerprint(v):
    result = {"shape": list(v.data.shape), "spacing": list(v.spacing), "origin": v.origin.tolist(),
              "direction": v.direction.tolist(), "synthetic": bool(v.metadata["synthetic"])}
    result.update({key: v.metadata[key] for key in ("series_uid", "frame_of_reference_uid", "source_group", "sop_instance_digest") if key in v.metadata})
    return result


def path(v, *, identifier="canal-test-1", side="unknown"):
    return {"id": identifier, "name": "Canal candidate", "side": side,
            "points_lps_mm": v.index_to_world([[7, 10, 8], [8, 13, 14], [10, 17, 20], [14, 20, 27]]).tolist(),
            "diameter_mm": 2., "color": "#EE8487", "review": "pending",
            "provenance": {"name": "Test candidate generator", "version": "1"}}


def collection(v=None):
    v = volume() if v is None else v
    return create_paths(v, fingerprint(v), [path(v)])


def canal_mask(v, *, split=False):
    result = np.zeros(v.data.shape, dtype=np.uint16)
    if split:
        result[4:15, 16, 8] = 5
        result[25:37, 16, 8] = 5
    else:
        result[5:36, 12, 8] = 5
        result[5:36, 28, 32] = 5
    return result


def test_collection_owns_original_candidates_and_moves_in_patient_space():
    v = volume(reflected=True)
    original = path(v)
    result = create_paths(v, fingerprint(v), [original])
    original["points_lps_mm"][0][0] += 100
    before = result.original_paths
    result.set_review("canal-test-1", "accepted")
    new_position = v.index_to_world([9.5, 14.2, 15.7])
    assert result.move_node(v, "canal-test-1", 1, new_position)
    np.testing.assert_allclose(result.paths[0]["points_lps_mm"][1], new_position)
    assert result.paths[0]["review"] == "pending"
    assert result.original_paths == before
    before[0]["points_lps_mm"][0][0] += 999
    assert result.original_paths != before
    assert result.undo()
    assert result.paths[0]["review"] == "accepted"


def test_many_node_moves_form_one_gesture_and_undo_restores_nodes_and_review():
    v = volume()
    result = collection(v)
    result.set_review("canal-test-1", "accepted")
    before = copy.deepcopy(result.paths)
    result.begin_gesture()
    for x in (8., 8.3, 8.7, 9.):
        result.move_node(v, "canal-test-1", 1, v.index_to_world([x, 13, 14]))
    assert result.end_gesture()
    assert len(result._undo) == 1 and result.can_undo
    assert result.undo()
    assert result.paths == before
    assert not result.can_undo


def test_insertion_deletion_properties_and_path_removal_are_undoable():
    v = volume()
    result = collection(v)
    position = v.index_to_world([9, 16, 17])
    result.insert_node(v, "canal-test-1", 2, position)
    np.testing.assert_allclose(result.paths[0]["points_lps_mm"][2], position)
    assert len(result.paths[0]["points_lps_mm"]) == 5
    result.delete_node("canal-test-1", 2)
    assert len(result.paths[0]["points_lps_mm"]) == 4
    result.set_diameter("canal-test-1", 3.2)
    result.set_color("canal-test-1", "#55AA88")
    result.set_side("canal-test-1", "right")
    result.set_name("canal-test-1", "Reviewed right canal")
    assert result.paths[0]["side"] == "right"
    result.remove_path("canal-test-1")
    assert not result.paths
    result.undo()
    assert result.paths[0]["name"] == "Reviewed right canal"
    result.undo()
    assert result.paths[0]["name"] == "Canal candidate"


def test_manual_path_addition_uses_stable_id_pending_review_and_preserves_originals():
    v = volume()
    result = collection(v)
    original = result.original_paths
    draft = path(v, identifier="manual-1")
    draft["review"] = "accepted"
    identifier = result.add_path(v, draft)
    assert identifier == "manual-1"
    assert result.get_path(identifier)["review"] == "pending"
    assert result.original_paths == original
    result.undo()
    assert len(result.paths) == 1
    generated = result.add_path(v, v.index_to_world([[20, 20, 20], [20, 21, 22]]))
    assert len(generated) == 36


def test_invalid_move_and_duplicate_path_are_atomic_and_do_not_add_undo_records():
    v = volume()
    result = collection(v)
    before = copy.deepcopy(result.paths)
    with pytest.raises(ValueError, match="inside"):
        result.move_node(v, "canal-test-1", 0, v.index_to_world([-1, 10, 8]))
    with pytest.raises(ValueError, match="unique"):
        result.add_path(v, path(v))
    assert result.paths == before and not result.can_undo


@pytest.mark.parametrize("change", ["nonfinite", "one-node", "side", "diameter", "color", "review", "id", "extra"])
def test_path_validation_rejects_malformed_candidates(change):
    v = volume()
    item = path(v)
    if change == "nonfinite": item["points_lps_mm"][0][0] = np.nan
    if change == "one-node": item["points_lps_mm"] = item["points_lps_mm"][:1]
    if change == "side": item["side"] = "probably-right"
    if change == "diameter": item["diameter_mm"] = -1
    if change == "color": item["color"] = "pink"
    if change == "review": item["review"] = "AI-approved"
    if change == "id": item["id"] = "../../path"
    if change == "extra": item["confidence"] = .99
    with pytest.raises(ValueError):
        create_paths(v, fingerprint(v), [item])


def test_minimum_two_nodes_and_calibrated_matching_source_are_required():
    v = volume(real=True)
    result = collection(v)
    result.delete_node("canal-test-1", 0)
    result.delete_node("canal-test-1", 0)
    before = copy.deepcopy(result.paths)
    with pytest.raises(ValueError, match="2 to"):
        result.delete_node("canal-test-1", 0)
    assert result.paths == before
    wrong = volume(real=True)
    wrong.metadata["source_group"] = "different-stack"
    with pytest.raises(ValueError, match="different DICOM"):
        result.move_node(wrong, "canal-test-1", 0, wrong.center)
    v.metadata["geometry_calibrated"] = False
    with pytest.raises(ValueError, match="calibrated"):
        result.validate_source(v, fingerprint(v))


def test_rejected_paths_retain_editable_nodes_and_noop_gestures_do_not_pollute_undo():
    v = volume()
    result = collection(v)
    before = copy.deepcopy(result.paths[0]["points_lps_mm"])
    result.set_review("canal-test-1", "rejected")
    assert result.paths[0]["points_lps_mm"] == before
    result.begin_gesture()
    assert not result.move_node(v, "canal-test-1", 0, before[0])
    assert not result.end_gesture()
    assert not result.can_undo


@pytest.mark.parametrize("smooth", [False, True])
def test_curve_is_endpoint_anchored_bounded_and_rigid_transform_invariant(smooth):
    points = np.array([[0., 0, 0], [3., 0, 0], [3., 4, 0], [5., 4, 2]])
    samples = sample_path(points, step_mm=.15, smooth=smooth)
    np.testing.assert_allclose(samples[0], points[0])
    np.testing.assert_allclose(samples[-1], points[-1])
    assert np.all(samples >= points.min(axis=0) - 1e-10)
    assert np.all(samples <= points.max(axis=0) + 1e-10)
    assert np.max(np.linalg.norm(np.diff(samples, axis=0), axis=1)) <= .15 + 1e-10
    rotation = rotation_matrix(23, -17, 34)
    translation = [31, -20, 16]
    moved = sample_path(points @ rotation.T + translation, step_mm=.15, smooth=smooth)
    np.testing.assert_allclose(moved, samples @ rotation.T + translation, atol=1e-10)


def test_linear_sampling_passes_all_control_nodes_and_duplicate_nodes_are_safe():
    points = np.array([[0., 0, 0], [1, 0, 0], [1, 0, 0], [1, 2, 0]])
    samples = sample_path(points, step_mm=.4, smooth=False)
    assert any(np.allclose(point, [1, 0, 0]) for point in samples)
    assert np.isfinite(sample_path(points)).all()


@pytest.mark.parametrize("reflected", [False, True])
def test_extraction_preserves_disconnected_components_and_source_geometry(reflected):
    v = volume(reflected=reflected)
    data = canal_mask(v)
    before = data.copy()
    result = extract_canal_paths(v, fingerprint(v), data, node_spacing_mm=2.)
    assert len(result.paths) == 2
    assert all(item["side"] == "unknown" and item["review"] == "pending" for item in result.paths)
    assert len({item["id"] for item in result.paths}) == 2
    assert result.original_paths == result.paths
    for item in result.paths:
        index = v.world_to_index(result.samples(item["id"], step_mm=.2, smooth=False))
        nearest = np.rint(index).astype(int)
        assert np.all(data[nearest[:, 2], nearest[:, 1], nearest[:, 0]] == 5)
        assert np.ptp(index[:, 0]) < 1e-6 and np.ptp(index[:, 1]) < 1e-6
    assert any("Laterality" in warning for warning in result.warnings)
    np.testing.assert_array_equal(data, before)


def test_fragmented_canal_evidence_is_never_bridged_across_missing_mask():
    v = volume(oblique=False)
    data = canal_mask(v, split=True)
    result = extract_canal_paths(v, fingerprint(v), data)
    assert len(result.paths) == 2
    ranges = []
    for item in result.paths:
        z = v.world_to_index(item["points_lps_mm"])[:, 2]
        ranges.append((z.min(), z.max()))
    ranges.sort()
    assert ranges[0][1] <= 14 + 1e-8 and ranges[1][0] >= 25 - 1e-8


def test_laterality_requires_explicit_anatomical_midline_and_crossing_remains_unknown():
    v = volume(oblique=False)
    data = canal_mask(v)
    known = extract_canal_paths(v, fingerprint(v), data, midline_lps_x=0)
    assert {item["side"] for item in known.paths} == {"right", "left"}
    unilateral = data.copy()
    unilateral[:, :, 20:] = 0
    unknown = extract_canal_paths(v, fingerprint(v), unilateral)
    assert len(unknown.paths) == 1 and unknown.paths[0]["side"] == "unknown"
    crossing = np.zeros(data.shape, dtype=np.uint8)
    crossing[20, 20, 4:37] = 5
    cross = extract_canal_paths(v, fingerprint(v), crossing, midline_lps_x=0)
    assert len(cross.paths) == 1 and cross.paths[0]["side"] == "unknown"


def test_bent_thick_mask_yields_supported_smooth_nodes_without_changing_mask():
    v = volume(reflected=True)
    centerline = np.zeros(v.data.shape, dtype=bool)
    centerline[5:21, 10, 8] = True
    centerline[20, 10:26, 8] = True
    mask = binary_dilation(centerline, iterations=2)
    data = np.where(mask, 5, 0).astype(np.uint16)
    result = extract_canal_paths(v, fingerprint(v), data)
    assert len(result.paths) == 1
    item = result.paths[0]
    index = v.world_to_index(result.samples(item["id"], step_mm=.15))
    nearest = np.rint(index).astype(int)
    assert np.all(data[nearest[:, 2], nearest[:, 1], nearest[:, 0]] == 5)
    assert 0 < item["diameter_mm"] <= 20
    assert "diameter estimated from mask" in item["provenance"]["method"]


def test_branched_graph_uses_physical_geodesic_length_and_keeps_ambiguity_explicit():
    v = volume(oblique=False)
    data = np.zeros(v.data.shape, dtype=np.uint8)
    data[20:36, 20, 20] = 5  # 18 mm arm, fewer voxels than either other arm.
    data[20, 20:40, 20] = 5  # 14.25 mm arm.
    data[20, 20, 20:40] = 5  # 9.5 mm arm.
    result = extract_canal_paths(v, fingerprint(v), data)
    assert len(result.paths) == 1
    nodes = v.world_to_index(result.paths[0]["points_lps_mm"])
    assert nodes[:, 2].max() >= 34 and nodes[:, 1].max() >= 38
    assert np.ptp(nodes[:, 0]) < 1e-6
    assert any("branched" in note for note in result.warnings)


def test_loop_evidence_yields_provisional_supported_path_without_fabricated_endpoints():
    v = volume(oblique=False)
    data = np.zeros(v.data.shape, dtype=np.uint8)
    data[20, 10, 10:31] = data[20, 30, 10:31] = 5
    data[20, 10:31, 10] = data[20, 10:31, 30] = 5
    result = extract_canal_paths(v, fingerprint(v), data)
    assert len(result.paths) == 1
    assert result.paths[0]["review"] == "pending"
    assert any("ambiguous endpoints or a loop" in note for note in result.warnings)
    nodes = v.world_to_index(result.samples(result.paths[0]["id"], smooth=False))
    nearest = np.rint(nodes).astype(int)
    assert np.all(data[nearest[:, 2], nearest[:, 1], nearest[:, 0]] == 5)


def test_empty_and_isolated_fragment_masks_return_no_fabricated_paths():
    v = volume()
    data = np.zeros(v.data.shape, dtype=np.uint8)
    empty = extract_canal_paths(v, fingerprint(v), data)
    assert not empty.paths and empty.warnings
    data[10, 10, 10] = 5
    isolated = extract_canal_paths(v, fingerprint(v), data)
    assert not isolated.paths
    assert any("could not form a path" in note for note in isolated.warnings)


def test_extraction_cancellation_never_returns_partial_paths_or_changes_source():
    v = volume()
    data = canal_mask(v)
    before = data.copy()
    event = threading.Event()
    event.set()
    with pytest.raises(PathExtractionCancelled):
        extract_canal_paths(v, fingerprint(v), data, cancel=event)
    event.clear()
    def progress(_message):
        event.set()
    with pytest.raises(PathExtractionCancelled):
        extract_canal_paths(v, fingerprint(v), data, cancel=event, progress=progress)
    np.testing.assert_array_equal(data, before)


def test_source_padding_is_not_anatomy_for_path_extraction():
    v = volume()
    data = canal_mask(v)
    v.valid_mask = np.ones(v.data.shape, dtype=bool)
    v.valid_mask[:, :, :20] = False
    result = extract_canal_paths(v, fingerprint(v), data)
    assert len(result.paths) == 1
    assert np.all(v.world_to_index(result.paths[0]["points_lps_mm"])[:, 0] > 20)


def test_json_roundtrip_preserves_original_candidate_and_explicit_review_rules(tmp_path):
    v = volume(real=True)
    result = collection(v)
    original = result.original_paths
    result.move_node(v, "canal-test-1", 0, v.index_to_world([8, 11, 9]))
    result.set_review("canal-test-1", "accepted")
    output = tmp_path / "canal.paths.json"
    export_nerve_paths(output, result, v, fingerprint(v))
    loaded = read_nerve_paths(output, v, fingerprint(v))
    assert loaded.paths[0]["review"] == "pending"
    assert loaded.original_paths == original
    reopened = read_nerve_paths(output, v, fingerprint(v), imported=False)
    assert reopened.paths == result.paths
    assert reopened.original_paths == original
    assert not reopened.can_undo
    payload = json.loads(output.read_text())
    assert not {"patient_name", "filename", "path"} & payload["source"].keys()


def test_wrong_source_import_and_malformed_json_are_atomic(tmp_path):
    v = volume(real=True)
    result = collection(v)
    output = tmp_path / "paths.json"
    export_nerve_paths(output, result, v, fingerprint(v))
    original_bytes = output.read_bytes()
    wrong = fingerprint(v)
    wrong["source_group"] = "other-stack"
    with pytest.raises(ValueError, match="different DICOM"):
        read_nerve_paths(output, v, wrong)
    result.paths[0]["points_lps_mm"][0][0] = np.nan
    with pytest.raises(ValueError):
        export_nerve_paths(output, result, v, fingerprint(v))
    assert output.read_bytes() == original_bytes
    output.write_text('{"format":1,"format":2}')
    with pytest.raises(ValueError, match="duplicate JSON"):
        read_nerve_paths(output, v, fingerprint(v))


def test_path_count_and_stable_identity_are_bounded():
    v = volume()
    items = [path(v, identifier=f"candidate-{index}") for index in range(MAX_PATHS + 1)]
    with pytest.raises(ValueError, match="at most"):
        create_paths(v, fingerprint(v), items)
