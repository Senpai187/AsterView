"""Validate saved workspace state completely before mutating the live viewer."""
from __future__ import annotations

from numbers import Real
import re

import numpy as np

STATE_KEYS = {"schema", "volume", "crosshair", "rotation", "window", "arch", "implants", "tmj_centers", "tmj_angles"}
GEOMETRY_KEYS = {"shape", "spacing", "origin", "direction", "synthetic"}
IDENTITY_KEYS = {"series_uid", "frame_of_reference_uid", "source_group", "sop_instance_digest"}
MAX_ARCH_POINTS = 2048
MAX_IMPLANTS = 200


class SessionValidationError(ValueError):
    """A saved state is malformed or does not belong to the current volume."""


def _keys(value, required: set[str], label: str, optional: set[str] | None = None) -> None:
    if not isinstance(value, dict) or not required.issubset(value) or set(value) - required - (optional or set()):
        raise SessionValidationError(f"{label} contains missing or unsupported fields.")


def _number(value, label: str, minimum: float = -1_000_000, maximum: float = 1_000_000) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SessionValidationError(f"{label} must be numeric.")
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise SessionValidationError(f"{label} is outside the supported finite range.") from None
    if not np.isfinite(result) or not minimum <= result <= maximum:
        raise SessionValidationError(f"{label} is outside the supported finite range.")
    return result


def _vector(value, length: int, label: str, minimum=-1_000_000, maximum=1_000_000) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise SessionValidationError(f"{label} must contain {length} coordinates.")
    return [_number(v, label, minimum, maximum) for v in value]


def _string(value, label: str, maximum=256, *, allow_empty=False) -> str:
    if not isinstance(value, str) or len(value) > maximum or (not allow_empty and not value.strip()) or any(ord(c) < 32 for c in value):
        raise SessionValidationError(f"{label} must be a bounded text value.")
    return value


def _fingerprint(value, label: str) -> dict:
    if not isinstance(value, dict) or type(value.get("synthetic")) is not bool:
        raise SessionValidationError(f"{label} must explicitly identify whether the volume is synthetic.")
    synthetic = value["synthetic"]
    _keys(value, GEOMETRY_KEYS, label, IDENTITY_KEYS) if synthetic else _keys(value, GEOMETRY_KEYS | IDENTITY_KEYS, label)
    shape = value["shape"]
    if not isinstance(shape, (list, tuple)) or len(shape) != 3 or any(type(n) is not int or n < 1 or n > 1_000_000 for n in shape):
        raise SessionValidationError(f"{label} has invalid volume dimensions.")
    spacing = _vector(value["spacing"], 3, f"{label} spacing", 0.000001, 1_000_000)
    origin = _vector(value["origin"], 3, f"{label} origin")
    direction = value["direction"]
    if not isinstance(direction, (list, tuple)) or len(direction) != 3:
        raise SessionValidationError(f"{label} direction must be a 3 by 3 matrix.")
    direction = [_vector(row, 3, f"{label} direction", -1.0001, 1.0001) for row in direction]
    matrix = np.asarray(direction)
    if not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0, atol=1e-4):
        raise SessionValidationError(f"{label} direction axes must be orthonormal.")
    result = {"shape": list(shape), "spacing": spacing, "origin": origin, "direction": direction, "synthetic": synthetic}
    for key in IDENTITY_KEYS:
        if key in value:
            result[key] = _string(value[key], f"{label} identity", allow_empty=synthetic or key == "frame_of_reference_uid")
    if not synthetic and re.fullmatch(r"[0-9a-f]{64}", result["sop_instance_digest"]) is None:
        raise SessionValidationError(f"{label} must contain a valid source-instance digest.")
    return result


def _match_volume(saved: dict, current: dict) -> None:
    if saved["synthetic"] != current["synthetic"] or saved["shape"] != current["shape"]:
        raise SessionValidationError("This session belongs to a different volume. Load the matching DICOM series first.")
    if not saved["synthetic"] and any(saved[key] != current[key] for key in IDENTITY_KEYS):
        raise SessionValidationError("This session belongs to a different DICOM series or source stack. Load the exact matching series first.")
    for key, tolerance in (("spacing", 1e-6), ("origin", 1e-4), ("direction", 1e-6)):
        if not np.allclose(saved[key], current[key], rtol=0, atol=tolerance):
            raise SessionValidationError("Session geometry does not match the current volume.")


def validate_session(payload, current_fingerprint) -> dict:
    """Return an independent validated state, or raise without changing input.

    Real scans are matched by SeriesUID, split-group identity, source instance
    digest, frame of reference, and calibrated geometry. A shared SeriesUID alone
    cannot authorize applying implant or TMJ coordinates to another phase.
    """
    _keys(payload, STATE_KEYS, "Session", {"implant_margins", "viewing"})
    if type(payload["schema"]) is not int or payload["schema"] != 1:
        raise SessionValidationError("This session schema is unsupported; expected schema 1.")
    saved = _fingerprint(payload["volume"], "Saved volume")
    current = _fingerprint(current_fingerprint, "Current volume")
    _match_volume(saved, current)
    state = {"schema": 1, "volume": saved,
             "crosshair": _vector(payload["crosshair"], 3, "Crosshair"),
             "rotation": _vector(payload["rotation"], 3, "Rotation", -180, 180)}
    window = _vector(payload["window"], 2, "Window", -1_000_000, 2_000_000)
    window[0] = _number(window[0], "Window level", -1_000_000, 1_000_000)
    window[1] = _number(window[1], "Window width", 1, 2_000_000)
    state["window"] = window
    arch = payload["arch"]
    if not isinstance(arch, list) or len(arch) > MAX_ARCH_POINTS:
        raise SessionValidationError(f"Arch must contain no more than {MAX_ARCH_POINTS} points.")
    state["arch"] = [_vector(point, 3, "Arch point") for point in arch]
    implants = payload["implants"]
    if not isinstance(implants, list) or len(implants) > MAX_IMPLANTS:
        raise SessionValidationError(f"A session may contain at most {MAX_IMPLANTS} implants.")
    state["implants"] = []
    for implant in implants:
        _keys(implant, {"name", "center", "axis", "diameter", "length"}, "Implant", {"id"})
        axis = np.asarray(_vector(implant["axis"], 3, "Implant axis", -1.0001, 1.0001))
        norm = float(np.linalg.norm(axis))
        if not np.isclose(norm, 1, rtol=0, atol=1e-4):
            raise SessionValidationError("Implant axis must be a nonzero normalized direction.")
        state["implants"].append({
            "name": _string(implant["name"], "Implant name", 128),
            "center": _vector(implant["center"], 3, "Implant center"),
            "axis": (axis / norm).tolist(),
            "diameter": _number(implant["diameter"], "Implant diameter", 1, 10),
            "length": _number(implant["length"], "Implant length", 3, 30),
        })
        if "id" in implant:
            state["implants"][-1]["id"] = _string(implant["id"], "Implant ID", 96)
    ids = [item["id"] for item in state["implants"] if "id" in item]
    if len(ids) != len(set(ids)):
        raise SessionValidationError("Implant IDs must be unique.")
    margins = payload.get("implant_margins", {})
    if not isinstance(margins, dict) or set(margins) - set(ids):
        raise SessionValidationError("Implant margins must refer to implants in this session.")
    state["implant_margins"] = {key: _number(value, "Implant margin", 0, 100) for key, value in margins.items()}
    _keys(payload["tmj_centers"], {"Right", "Left"}, "TMJ centers")
    _keys(payload["tmj_angles"], {"Right", "Left"}, "TMJ angles")
    state["tmj_centers"] = {side: None if payload["tmj_centers"][side] is None else _vector(payload["tmj_centers"][side], 3, "TMJ center") for side in ("Right", "Left")}
    state["tmj_angles"] = {side: _number(payload["tmj_angles"][side], "TMJ angle", -180, 180) for side in ("Right", "Left")}
    viewing = payload.get("viewing", {})
    _keys(viewing, set(), "Viewing settings",
          {"pano_offset", "pano_bounds", "implant_section_width", "tmj_corrections"})
    state["viewing"] = {
        "pano_offset": _number(viewing.get("pano_offset", 0), "Panoramic offset", -20, 20),
        "pano_bounds": _vector(viewing.get("pano_bounds", [-45, 45]), 2,
                               "Panoramic bounds", -1000, 1000),
        "implant_section_width": _number(viewing.get("implant_section_width", 24),
                                         "Implant section width", 5, 150),
    }
    if state["viewing"]["pano_bounds"][1] <= state["viewing"]["pano_bounds"][0]:
        raise SessionValidationError("Panoramic upper bound must exceed the lower bound.")
    corrections = viewing.get("tmj_corrections",
                              {side: np.eye(3).tolist() for side in ("Right", "Left")})
    _keys(corrections, {"Right", "Left"}, "TMJ corrections")
    state["viewing"]["tmj_corrections"] = {}
    for side, value in corrections.items():
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise SessionValidationError("TMJ correction must be a 3 by 3 matrix.")
        matrix = [_vector(row, 3, "TMJ correction", -1.0001, 1.0001) for row in value]
        if (not np.allclose(np.asarray(matrix).T @ matrix, np.eye(3), rtol=0, atol=1e-6)
                or not np.isclose(np.linalg.det(matrix), 1., rtol=0, atol=1e-6)):
            raise SessionValidationError("TMJ correction must be a proper orthonormal rotation.")
        state["viewing"]["tmj_corrections"][side] = matrix
    return state
