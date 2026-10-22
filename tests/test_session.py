from copy import deepcopy

import numpy as np
import pytest

from asterview.session import SessionValidationError, validate_session


@pytest.fixture
def state():
    fingerprint = {"series_uid": "1.2.3", "frame_of_reference_uid": "1.2.5", "source_group": "1.2.3:phase-a",
                   "sop_instance_digest": "a" * 64, "shape": [10, 20, 30], "spacing": [0.4, 0.5, 1],
                   "origin": [12, 24, -5], "direction": np.eye(3).tolist(), "synthetic": False}
    return {"schema": 1, "volume": fingerprint, "crosshair": [13, 25, 0], "rotation": [0, 30, 5],
            "window": [500, 1500], "arch": [[12, 24, 0], [13, 25, 0]],
            "implants": [{"name": "Implant 1", "center": [15, 25, 1], "axis": [0, 0, 1], "diameter": 4, "length": 10}],
            "tmj_centers": {"Right": None, "Left": [12, 25, 1]}, "tmj_angles": {"Right": 0, "Left": 24}}


def test_validated_state_is_independent_and_does_not_mutate_inputs(state):
    original = deepcopy(state)
    current = deepcopy(state["volume"])
    validated = validate_session(state, current)
    validated["implants"][0]["center"][0] = -99
    validated["volume"]["direction"][0][0] = 0
    assert state == original
    assert current == original["volume"]


@pytest.mark.parametrize("field,value", [
    ("series_uid", "9.9.9"), ("source_group", "1.2.3:phase-b"),
    ("sop_instance_digest", "b" * 64), ("frame_of_reference_uid", "9.8.7"),
    ("shape", [11, 20, 30]), ("spacing", [0.6, 0.5, 1]), ("origin", [12, 24, -4]),
])
def test_identical_shape_or_series_uid_cannot_authorize_different_scan(state, field, value):
    current = deepcopy(state["volume"])
    current[field] = value
    with pytest.raises(SessionValidationError, match="different|does not match"):
        validate_session(state, current)


@pytest.mark.parametrize("mutate", [
    lambda s: s.update(schema=2),
    lambda s: s.update(schema=True),
    lambda s: s.update(crosshair=[np.nan, 2, 3]),
    lambda s: s.update(crosshair=["1", 2, 3]),
    lambda s: s.update(crosshair=[10 ** 1000, 2, 3]),
    lambda s: s.update(rotation=[0, 0, 181]),
    lambda s: s.update(window=[0, 0]),
    lambda s: s.update(window=[0, 2_000_001]),
    lambda s: s["implants"][0].update(axis=[0, 0, 0]),
    lambda s: s["implants"][0].update(axis=[0, 0, 2]),
    lambda s: s["implants"][0].update(diameter=-1),
    lambda s: s["implants"][0].update(length=31),
    lambda s: s["implants"][0].update(name="x" * 129),
    lambda s: s["tmj_centers"].update(Left=[0, np.inf, 0]),
    lambda s: s["tmj_angles"].update(Right=181),
    lambda s: s["tmj_angles"].update(Other=0),
    lambda s: s.update(arch=[[1, 2, 3]] * 2049),
    lambda s: s.update(implants=s["implants"] * 201),
    lambda s: s.pop("arch"),
])
def test_malformed_state_fails_before_any_input_is_mutated(state, mutate):
    mutate(state)
    before = deepcopy(state)
    with pytest.raises(SessionValidationError):
        validate_session(state, before["volume"])
    # NaN is not equal to itself, so compare stable representations here.
    assert repr(state) == repr(before)


def test_empty_real_series_uid_cannot_match_by_geometry(state):
    state["volume"]["series_uid"] = ""
    with pytest.raises(SessionValidationError):
        validate_session(state, deepcopy(state["volume"]))


def test_synthetic_state_can_omit_real_scan_identifiers(state):
    for key in ("series_uid", "frame_of_reference_uid", "source_group", "sop_instance_digest"):
        state["volume"].pop(key)
    state["volume"]["synthetic"] = True
    validated = validate_session(state, deepcopy(state["volume"]))
    assert validated["volume"]["synthetic"]


def test_synthetic_session_cannot_apply_to_real_scan(state):
    current = deepcopy(state["volume"])
    state["volume"]["synthetic"] = True
    with pytest.raises(SessionValidationError, match="different volume"):
        validate_session(state, current)


def test_optional_viewing_state_preserves_offset_width_and_independent_tmj_orientation(state):
    from asterview.reformat import rotation_matrix
    state["viewing"] = {
        "pano_offset": -2.8, "pano_bounds": [-22, 31], "implant_section_width": 33,
        "tmj_corrections": {"Right": rotation_matrix(13, -21, 8).tolist(),
                            "Left": np.eye(3).tolist()},
    }
    validated = validate_session(state, deepcopy(state["volume"]))
    assert validated["viewing"] == state["viewing"]
    validated["viewing"]["tmj_corrections"]["Right"][0][0] = 42
    assert state["viewing"]["tmj_corrections"]["Right"][0][0] != 42


def test_old_session_receives_defined_viewing_defaults(state):
    viewing = validate_session(state, deepcopy(state["volume"]))["viewing"]
    assert viewing["pano_offset"] == 0 and viewing["implant_section_width"] == 24
    assert viewing["pano_bounds"] == [-45, 45]
    assert viewing["tmj_corrections"] == {side: np.eye(3).tolist() for side in ("Right", "Left")}


@pytest.mark.parametrize("viewing", [
    {"pano_offset": 20.1}, {"pano_offset": True},
    {"implant_section_width": 4.9}, {"implant_section_width": 150.1},
    {"implant_section_width": float("nan")}, {"unexpected": 1},
    {"pano_bounds": [2, 2]}, {"pano_bounds": [3, -4]},
    {"pano_bounds": [-1001, 20]}, {"pano_bounds": [0, float("nan")]},
    {"tmj_corrections": {"Right": np.eye(3).tolist()}},
    {"tmj_corrections": {"Right": np.diag([1, 1, -1]).tolist(), "Left": np.eye(3).tolist()}},
    {"tmj_corrections": {"Right": (np.eye(3) * 2).tolist(), "Left": np.eye(3).tolist()}},
])
def test_invalid_viewing_settings_fail_before_mutation(state, viewing):
    state["viewing"] = viewing
    before = deepcopy(state)
    with pytest.raises(SessionValidationError):
        validate_session(state, deepcopy(state["volume"]))
    assert repr(state) == repr(before)
