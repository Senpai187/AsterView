"""Independent tooth identity, FDI correction, and source-safe label merging."""
import copy
import json

import numpy as np
import pytest

from asterview.dental_controls import combine_teeth
from asterview.model import Volume
from asterview.segmentation import create_result, export_segmentation, read_segmentation


def volume():
    return Volume(np.arange(7 * 9 * 11, dtype=np.float32).reshape(7, 9, 11),
                  (.5, .8, 1.3), np.array([3., -8., 12.]), np.eye(3),
                  metadata={"synthetic": True, "modality": "CT"})


def fingerprint(v):
    return {"shape": list(v.data.shape), "spacing": list(v.spacing), "origin": v.origin.tolist(),
            "direction": v.direction.tolist(), "synthetic": True}


def teeth(v, *, unassigned=False, collision=False):
    first, second = (2, 5) if collision else (101, 103)
    data = np.zeros(v.data.shape, dtype=np.uint16)
    data[2:4, 2:4, 2:4] = first
    data[2:4, 2:4, 6:8] = second
    labels = {first: {"name": "Tooth 11", "kind": "tooth", "instance": True,
                      "fdi": 11, "predicted_fdi": 11, "review": "accepted"},
              second: {"name": "Tooth 12", "kind": "tooth", "instance": True,
                       "fdi": 12, "predicted_fdi": 12, "review": "accepted"}}
    if unassigned:
        labels[first].pop("fdi")
        labels[first]["name"] = "Tooth object (unassigned)"
    return create_result(v, fingerprint(v), data, labels,
                         {"name": "Independent tooth fixture", "version": "1", "method": "Fixture only"})


def grouped(v):
    data = np.zeros(v.data.shape, dtype=np.uint16)
    data[0, :, :] = 2
    data[1, 1, 1:9] = 5
    data[4, 2:5, 2:9] = 3
    data[5, 2:5, 2:9] = 4
    labels = {2: {"name": "Mandible", "kind": "jaw", "review": "accepted"},
              5: {"name": "Canal proxy", "kind": "canal", "review": "rejected"},
              3: {"name": "Upper teeth group", "kind": "tooth", "instance": False},
              4: {"name": "Lower teeth group", "kind": "tooth", "instance": False}}
    return create_result(v, fingerprint(v), data, labels,
                         {"name": "Base anatomy fixture", "version": "1", "license": "Fixture"})


def test_fdi_change_preserves_physical_instance_voxel_ids_and_original_model_proposal():
    v = volume()
    result = teeth(v)
    source, mask = v.data.copy(), result.data.copy()
    result.assign_fdi(101, 21)
    assert result.labels[101]["fdi"] == 21
    assert result.labels[101]["predicted_fdi"] == 11
    assert result.labels[101]["name"] == "Tooth 21"
    assert result.labels[101]["review"] == "pending"
    assert result.labels[103]["fdi"] == 12
    np.testing.assert_array_equal(result.data, mask)
    np.testing.assert_array_equal(v.data, source)


def test_fdi_swap_exchanges_display_numbers_without_merging_or_renumbering_voxels():
    v = volume()
    result = teeth(v)
    mask = result.data.copy()
    with pytest.raises(ValueError, match="already assigned"):
        result.assign_fdi(101, 12)
    assert result.labels[101]["fdi"] == 11 and not result._number_undo
    result.assign_fdi(101, 12, swap=True)
    assert result.labels[101]["fdi"] == 12 and result.labels[103]["fdi"] == 11
    assert result.labels[101]["predicted_fdi"] == 11 and result.labels[103]["predicted_fdi"] == 12
    assert result.labels[101]["review"] == result.labels[103]["review"] == "pending"
    np.testing.assert_array_equal(result.data, mask)
    assert result.undo_fdi()
    assert result.labels[101]["fdi"] == 11 and result.labels[103]["fdi"] == 12
    assert result.labels[101]["review"] == result.labels[103]["review"] == "pending"


def test_unassigned_instance_can_take_number_or_leave_previous_owner_unassigned_on_swap():
    v = volume()
    result = teeth(v, unassigned=True)
    result.assign_fdi(101, 12, swap=True)
    assert result.labels[101]["fdi"] == 12
    assert "fdi" not in result.labels[103]
    assert result.labels[103]["predicted_fdi"] == 12
    assert result.labels[103]["instance"]
    result.assign_fdi(101, None)
    assert "fdi" not in result.labels[101]
    assert result.labels[101]["predicted_fdi"] == 11
    assert result.labels[101]["instance"]


@pytest.mark.parametrize("number", [11, 18, 21, 28, 31, 38, 41, 48, 51, 55, 61, 65, 71, 75, 81, 85])
def test_permanent_and_primary_fdi_formats_are_supported(number):
    v = volume()
    result = teeth(v)
    result.assign_fdi(101, number)
    assert result.labels[101]["fdi"] == number


@pytest.mark.parametrize("number", [0, 10, 19, 29, 49, 50, 56, 66, 76, 86, 99, True, 11.5, np.nan, "11"])
def test_invalid_fdi_is_rejected_atomically(number):
    v = volume()
    result = teeth(v)
    before = copy.deepcopy(result.labels)
    with pytest.raises(ValueError):
        result.assign_fdi(101, number)
    assert result.labels == before
    assert not result._number_undo


def test_grouped_dentition_and_non_tooth_objects_cannot_receive_one_fdi_number():
    v = volume()
    result = grouped(v)
    for identifier in (2, 3, 4, 5):
        with pytest.raises(ValueError):
            result.assign_fdi(identifier, 11)
    assert not result._number_undo


def test_fdi_undo_does_not_resurrect_acceptance_after_later_voxel_correction():
    v = volume()
    result = teeth(v)
    result.assign_fdi(101, 21)
    result.set_review(101, "accepted")
    result.set_review(103, "rejected")
    point = v.index_to_world([5, 5, 3])
    assert result.paint(v, point, [1, 0, 0], [0, 1, 0], .2, 101) == 1
    assert result.labels[101]["review"] == "pending"
    changed_mask = result.data.copy()
    assert result.undo_fdi()
    assert result.labels[101]["fdi"] == 11
    assert result.labels[101]["predicted_fdi"] == 11
    assert result.labels[101]["review"] == "pending"
    assert result.labels[103]["review"] == "rejected"
    np.testing.assert_array_equal(result.data, changed_mask)


def test_predicted_fdi_and_instance_identity_survive_export_import(tmp_path):
    v = volume()
    result = teeth(v)
    result.assign_fdi(101, 51)
    output = tmp_path / "teeth.npz"
    export_segmentation(output, result, v, fingerprint(v))
    loaded = read_segmentation(output, v, fingerprint(v))
    assert loaded.labels[101]["instance"] is True
    assert loaded.labels[101]["fdi"] == 51
    assert loaded.labels[101]["predicted_fdi"] == 11
    assert loaded.labels[101]["review"] == "pending"
    np.testing.assert_array_equal(loaded.data, result.data)


def test_duplicate_imported_fdi_is_rejected_while_duplicate_predictions_are_retained(tmp_path):
    v = volume()
    result = teeth(v)
    result.labels[103]["predicted_fdi"] = 11
    output = tmp_path / "teeth.npz"
    export_segmentation(output, result, v, fingerprint(v))
    assert read_segmentation(output, v, fingerprint(v)).labels[103]["predicted_fdi"] == 11
    with np.load(output, allow_pickle=False) as archive:
        data, metadata = archive["data"].copy(), json.loads(archive["metadata"].tobytes())
    metadata["labels"][1]["fdi"] = 11
    np.savez_compressed(output, data=data, metadata=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8))
    with pytest.raises(ValueError, match="only one tooth"):
        read_segmentation(output, v, fingerprint(v))


@pytest.mark.parametrize("label", [
    {"name": "Grouped", "kind": "tooth", "instance": False, "fdi": 11},
    {"name": "Grouped", "kind": "tooth", "instance": False, "predicted_fdi": 11},
    {"name": "Jaw", "kind": "jaw", "instance": True},
    {"name": "Tooth", "kind": "tooth", "instance": 1},
])
def test_group_or_non_tooth_cannot_masquerade_as_predicted_individual_instance(label):
    v = volume()
    with pytest.raises(ValueError):
        create_result(v, fingerprint(v), np.zeros(v.data.shape, dtype=np.uint16), {1: label}, {"name": "Fixture", "version": "1"})


def test_combine_replaces_all_old_tooth_groups_and_retains_nonoverlap_anatomy_reviews():
    v = volume()
    base, individual = grouped(v), teeth(v)
    original_source, original_base, original_teeth = v.data.copy(), base.data.copy(), individual.data.copy()
    result = combine_teeth(v, fingerprint(v), individual, base)
    assert {key for key, label in result.labels.items() if label["kind"] == "tooth"} == {101, 103}
    assert result.labels[2]["review"] == "accepted"
    assert result.labels[5]["review"] == "rejected"
    assert not np.any(result.data == 3) and not np.any(result.data == 4)
    np.testing.assert_array_equal(result.data[base.data == 2], np.full(np.count_nonzero(base.data == 2), 2))
    np.testing.assert_array_equal(result.data[base.data == 5], np.full(np.count_nonzero(base.data == 5), 5))
    np.testing.assert_array_equal(base.data, original_base)
    np.testing.assert_array_equal(individual.data, original_teeth)
    np.testing.assert_array_equal(v.data, original_source)
    assert not np.shares_memory(result.data, individual.data)
    assert not np.shares_memory(result.data, base.data)
    assert result.source == individual.source == base.source


def test_combine_resolves_voxel_id_collision_without_aliasing_tooth_and_canal_objects():
    v = volume()
    base, individual = grouped(v), teeth(v, collision=True)
    result = combine_teeth(v, fingerprint(v), individual, base)
    tooth_ids = {key for key, label in result.labels.items() if label["kind"] == "tooth"}
    canal_id = next(key for key, label in result.labels.items() if label["kind"] == "canal")
    jaw_id = next(key for key, label in result.labels.items() if label["kind"] == "jaw")
    assert tooth_ids == {2, 5}
    assert canal_id not in tooth_ids and jaw_id not in tooth_ids and canal_id != jaw_id
    assert np.all(result.data[base.data == 5] == canal_id)
    assert np.all(result.data[base.data == 2] == jaw_id)
    assert result.labels[canal_id]["review"] == "rejected"
    assert result.labels[jaw_id]["review"] == "accepted"


def test_tooth_priority_overlap_returns_changed_anatomy_to_pending_review():
    v = volume()
    base, individual = grouped(v), teeth(v)
    base.set_review(5, "accepted")
    base.data[2, 2, 2] = 5
    result = combine_teeth(v, fingerprint(v), individual, base)
    assert result.data[2, 2, 2] == 101
    assert result.labels[5]["review"] == "pending"
    assert result.labels[2]["review"] == "accepted"
    assert base.labels[5]["review"] == "accepted" and base.data[2, 2, 2] == 5


def test_wrong_source_combination_is_atomic_before_mutating_any_input():
    v = volume()
    base, individual = grouped(v), teeth(v)
    old_base, old_teeth = base.data.copy(), individual.data.copy()
    wrong = copy.deepcopy(fingerprint(v))
    wrong["origin"][0] += 1
    with pytest.raises(ValueError, match="geometry"):
        combine_teeth(v, wrong, individual, base)
    np.testing.assert_array_equal(base.data, old_base)
    np.testing.assert_array_equal(individual.data, old_teeth)
