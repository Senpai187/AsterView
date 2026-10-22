"""Finite implant/canal geometry and source-bound planning settings.

An implant is a flat-capped finite circular cylinder. A canal display tube is
the union of spheres along its displayed polyline, with radius diameter/2.
Positive distance is their surface gap. A negative value is a signed-distance
overlap indicator, not a validated penetration depth or a surgical decision.
The numeric error bound covers minimizing along that polyline only; it does
not cover imaging, segmentation, tracing, interpolation, or implant-model error.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .segmentation import REVIEW_STATES, _positive
from .session import _fingerprint, _keys, _match_volume, _string


FORMAT = "asterview.implant-planning.v1"
MAX_IMPLANTS = 200
MAX_PATHS = 64
MAX_CURVE_POINTS = 32768
MAX_FILE_BYTES = 2 * 1024 * 1024


class ClearanceCancelled(RuntimeError):
    pass


def _check_cancel(cancel):
    if cancel is not None and bool(cancel() if callable(cancel) else cancel.is_set()):
        raise ClearanceCancelled("Implant clearance calculation canceled.")


def _vector(value, name):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError):
        raise ValueError(f"{name} requires three finite LPS coordinates.") from None
    if array.shape != (3,) or not np.isfinite(array).all() or np.any(np.abs(array) > 1_000_000):
        raise ValueError(f"{name} requires three bounded finite LPS coordinates.")
    return array


def _implant(implant):
    if not isinstance(implant, dict):
        raise ValueError("An implant requires finite cylinder geometry.")
    required = {"center", "axis", "length", "diameter"}
    if not required.issubset(implant):
        raise ValueError("An implant requires center, axis, length, and diameter.")
    center = _vector(implant["center"], "Implant center")
    axis = _vector(implant["axis"], "Implant axis")
    norm = float(np.linalg.norm(axis))
    if norm < 1e-10:
        raise ValueError("The implant axis must be nonzero.")
    return center, axis / norm, _positive(implant["length"], "Implant length", 1000) / 2, _positive(implant["diameter"], "Implant diameter", 1000) / 2


def _curve(value):
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("Canal samples must contain finite LPS coordinate triples.") from None
    if (points.ndim != 2 or points.shape[1] != 3 or not 1 <= len(points) <= MAX_CURVE_POINTS
            or not np.isfinite(points).all() or np.any(np.abs(points) > 1_000_000)):
        raise ValueError("Canal samples must be a bounded nonempty N by 3 LPS array.")
    if len(points) > 1:
        points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12]]
    return points


def _path_geometry(path, curve_samples):
    if not isinstance(path, dict):
        raise ValueError("Choose a canal path with a physical display diameter.")
    identifier = _string(path.get("id"), "Canal path ID", 96)
    review = path.get("review", "pending")
    if not isinstance(review, str) or review not in REVIEW_STATES:
        raise ValueError("Canal review must be pending, accepted, or rejected.")
    radius = _positive(path.get("diameter_mm"), "Canal display diameter", 20) / 2
    if curve_samples is None:
        curve_samples = path.get("sampled_points_lps_mm")
    if curve_samples is None:
        from .nerve_paths import sample_path
        curve_samples = sample_path(path.get("points_lps_mm"))
    return identifier, review, radius, _curve(curve_samples)


def _signed_distance(points, center, axis, half_length, radius):
    delta = points - center
    z = delta @ axis
    radial = delta - z[..., None] * axis
    radial_distance = np.linalg.norm(radial, axis=-1) - radius
    cap_distance = np.abs(z) - half_length
    outside = np.hypot(np.maximum(radial_distance, 0), np.maximum(cap_distance, 0))
    return outside + np.minimum(np.maximum(radial_distance, cap_distance), 0)


def _surface_pair(point, center, axis, half_length, radius):
    """Signed point distance, nearest cylinder boundary, and outward normal."""
    delta = point - center
    z = float(delta @ axis)
    radial = delta - z * axis
    rho = float(np.linalg.norm(radial))
    if rho > 1e-12:
        radial_unit = radial / rho
    else:
        reference = np.eye(3)[int(np.argmin(np.abs(axis)))]
        radial_unit = np.cross(axis, reference)
        radial_unit /= np.linalg.norm(radial_unit)
    side_distance, cap_distance = rho - radius, abs(z) - half_length
    if side_distance > 0 or cap_distance > 0:
        surface = center + radial_unit * min(rho, radius) + np.clip(z, -half_length, half_length) * axis
        distance = float(np.linalg.norm(point - surface))
        normal = (point - surface) / distance
        return distance, surface, normal
    if side_distance >= cap_distance:
        normal = radial_unit
        distance = side_distance
    else:
        normal = axis if z >= 0 else -axis
        distance = cap_distance
    return float(distance), point - distance * normal, normal


def closest_clearance(implant, path, curve_samples=None, *, tolerance_mm=.01, cancel=None):
    """Minimum signed surface clearance to the displayed canal polyline.

    Cylinder signed Euclidean distance is convex and 1-Lipschitz. Vectorized
    golden-section brackets minimize it along candidate segments, with a final
    physical bracket-width bound. Lipschitz lower bounds discard only segments
    that cannot beat an already measured candidate. Radius subtraction yields
    the rounded tube gap. Flat end caps and their circular edges are retained;
    no capsule approximation is used for the implant.
    """
    _check_cancel(cancel)
    center, axis, half_length, radius = _implant(implant)
    identifier, review, canal_radius, points = _path_geometry(path, curve_samples)
    tolerance = _positive(tolerance_mm, "Clearance numerical tolerance", .5)
    if tolerance < 1e-6:
        raise ValueError("Clearance tolerance must be at least one millionth of a millimeter.")
    distance = lambda value: _signed_distance(value, center, axis, half_length, radius)
    if len(points) == 1:
        best_point, best_segment, best_fraction, error = points[0].copy(), 0, 0., 0.
    else:
        start, delta = points[:-1], np.diff(points, axis=0)
        length = np.linalg.norm(delta, axis=1)
        f0, f1 = distance(start), distance(points[1:])
        fm = distance(start + .5 * delta)
        tested = np.stack([f0, fm, f1], axis=1)
        segment, candidate = np.unravel_index(int(np.argmin(tested)), tested.shape)
        best_fraction = float((0., .5, 1.)[candidate])
        best_point = start[segment] + best_fraction * delta[segment]
        best_segment, best_value = int(segment), float(tested[segment, candidate])
        # Endpoint cones and midpoint cones are valid global lower bounds.
        lower = np.maximum((f0 + f1 - length) / 2, fm - length / 2)
        active = np.flatnonzero(lower < best_value - tolerance)
        error = min(tolerance, max(0., best_value - float(np.min(lower))))
        if len(active):
            a, b = np.zeros(len(active)), np.ones(len(active))
            ratio = (np.sqrt(5.) - 1) / 2
            x1, x2 = b - ratio * (b - a), a + ratio * (b - a)
            local_start, local_delta, local_length = start[active], delta[active], length[active]
            f_left = distance(local_start + x1[:, None] * local_delta)
            f_right = distance(local_start + x2[:, None] * local_delta)
            for _ in range(80):
                _check_cancel(cancel)
                if float(np.max(local_length * (b - a) / 2)) <= tolerance:
                    break
                left = f_left <= f_right
                b = np.where(left, x2, b)
                a = np.where(left, a, x1)
                # Re-evaluation is vectorized and avoids Python per segment.
                x1, x2 = b - ratio * (b - a), a + ratio * (b - a)
                f_left = distance(local_start + x1[:, None] * local_delta)
                f_right = distance(local_start + x2[:, None] * local_delta)
            middle = (a + b) / 2
            values = distance(local_start + middle[:, None] * local_delta)
            at = int(np.argmin(values))
            if float(values[at]) < best_value:
                best_value = float(values[at])
                best_segment, best_fraction = int(active[at]), float(middle[at])
                best_point = local_start[at] + best_fraction * local_delta[at]
            error = max(float(np.max(local_length * (b - a) / 2)),
                        max(0., best_value - float(np.min(lower[np.setdiff1d(np.arange(len(lower)), active)])))
                        if len(active) < len(lower) else 0.)
    signed_centerline, implant_point, normal = _surface_pair(best_point, center, axis, half_length, radius)
    gap = float(signed_centerline - canal_radius)
    canal_point = best_point - canal_radius * normal
    return {"implant_id": str(implant.get("id", "")), "implant_name": str(implant.get("name", "Implant")),
            "path_id": identifier, "path_name": str(path.get("name", "Canal path")),
            "review": review, "path_review": review, "distance_mm": gap,
            "centerline_signed_distance_mm": float(signed_centerline),
            "centerline_distance_mm": max(0., float(signed_centerline)),
            "implant_point_lps": implant_point.copy(), "canal_point_lps": canal_point.copy(),
            "centerline_point_lps": best_point.copy(), "centerline_segment_index": best_segment,
            "segment_fraction": best_fraction, "approx_error_mm": float(error),
            "measurement_kind": "surface_gap" if gap >= 0 else "signed_overlap_indicator",
            "canal_radius_mm": canal_radius}


def _margin(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Planning margin must be a finite nonnegative millimeter value.")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("Planning margin must be a finite nonnegative millimeter value.") from None
    if not np.isfinite(result) or not 0 <= result <= 100:
        raise ValueError("Planning margin must be between zero and 100 millimeters.")
    return result


def evaluate_clearances(implants, paths, *, margins_mm=None, curve_samples=None, tolerance_mm=.01, cancel=None):
    """One nearest-path result per implant, with review-aware threshold status.

    Use actual collection paths, not gray original-comparison display copies.
    Rejected paths are excluded; ordinary visibility does not exclude evidence.
    ``margins_mm`` is keyed by stable implant ID. An absent margin defaults to
    zero and expresses no clinical recommendation. Pending nearest paths give
    status ``unreviewed`` while retaining a separate geometric status.
    """
    if not isinstance(implants, (list, tuple)) or len(implants) > MAX_IMPLANTS:
        raise ValueError("Implants must be a bounded list.")
    if hasattr(paths, "paths"):
        paths = paths.paths
    if not isinstance(paths, (list, tuple)) or len(paths) > MAX_PATHS:
        raise ValueError("Canal paths must be a bounded list.")
    if margins_mm is not None and not isinstance(margins_mm, dict):
        raise ValueError("Margins must be keyed by implant ID.")
    if curve_samples is not None and not isinstance(curve_samples, dict):
        raise ValueError("Curve samples must be keyed by path ID.")
    result, identifiers = [], set()
    for index, implant in enumerate(implants):
        _check_cancel(cancel)
        _implant(implant)
        identifier = _string(implant.get("id", f"implant-{index}"), "Implant ID", 96)
        if identifier in identifiers:
            raise ValueError("Implant IDs must be unique.")
        identifiers.add(identifier)
        margin = _margin(margins_mm.get(identifier, 0) if margins_mm is not None else implant.get("safety_margin_mm", implant.get("margin_mm", 0)))
        measured = []
        path_ids = set()
        for path in paths:
            if not isinstance(path, dict):
                raise ValueError("Each canal path requires a semantic object.")
            if path.get("review", "pending") == "rejected":
                continue
            path_id = _string(path.get("id"), "Path ID", 96)
            if path_id in path_ids:
                raise ValueError("Canal path IDs must be unique.")
            path_ids.add(path_id)
            samples = None if curve_samples is None else curve_samples.get(path_id)
            measured.append(closest_clearance({**implant, "id": identifier}, path, samples,
                                             tolerance_mm=tolerance_mm, cancel=cancel))
        if not measured:
            result.append({"implant_id": identifier, "implant_name": str(implant.get("name", "Implant")),
                           "status": "no_path", "geometry_status": "no_path", "distance_mm": None,
                           "margin_mm": margin, "path_id": None, "review": None, "path_review": None,
                           "approx_error_mm": None, "threshold_uncertain": False, "path_count_considered": 0})
            continue
        best = min(measured, key=lambda item: item["distance_mm"])
        global_lower = min(item["distance_mm"] - item["approx_error_mm"] for item in measured)
        error = max(0., best["distance_mm"] - global_lower)
        distance = best["distance_mm"]
        status = "overlap" if distance < 0 else ("within_margin" if global_lower <= margin else "beyond_margin")
        uncertain = (global_lower <= 0 <= distance) or (global_lower <= margin <= distance)
        result.append({**best, "approx_error_mm": error, "margin_mm": margin,
                       "geometry_status": status, "status": status if best["review"] == "accepted" else "unreviewed",
                       "threshold_uncertain": bool(uncertain), "path_count_considered": len(measured)})
    return result


def _validate_planning(implants, margins_mm, volume, fingerprint):
    source = _fingerprint(fingerprint, "Planning source")
    from .nerve_paths import _geometry
    _geometry(volume, source)
    if not isinstance(implants, list) or len(implants) > MAX_IMPLANTS or not isinstance(margins_mm, dict):
        raise ValueError("Planning requires bounded implants and margins keyed by their IDs.")
    normalized, identifiers = [], set()
    for implant in implants:
        _keys(implant, {"id", "name", "center", "axis", "length", "diameter"}, "Planning implant", {"safety_margin_mm", "margin_mm"})
        center, axis, half_length, radius = _implant(implant)
        if not 3 <= 2 * half_length <= 30 or not 1 <= 2 * radius <= 10:
            raise ValueError("Saved plans require implant length 3–30 mm and diameter 1–10 mm, matching the planning controls.")
        identifier = _string(implant["id"], "Implant ID", 96)
        if identifier in identifiers:
            raise ValueError("Planning implant IDs must be unique.")
        identifiers.add(identifier)
        index = volume.world_to_index(center)
        if np.any(index < -1e-5) or np.any(index > np.asarray(volume.dimensions) - 1 + 1e-5):
            raise ValueError("Planning implant centers must lie inside their source volume.")
        normalized.append({"id": identifier, "name": _string(implant["name"], "Implant name", 128),
                           "center": center.tolist(), "axis": axis.tolist(),
                           "length": 2 * half_length, "diameter": 2 * radius})
    if set(margins_mm) - identifiers:
        raise ValueError("A planning margin references a different implant ID.")
    margins = {identifier: _margin(margins_mm.get(identifier, 0)) for identifier in identifiers}
    return source, normalized, margins


def export_planning(path, implants, margins_mm, volume, fingerprint):
    source, implants, margins = _validate_planning(implants, margins_mm, volume, fingerprint)
    payload = {"format": FORMAT, "source": source, "implants": implants, "margins_mm": margins}
    content = json.dumps(payload, indent=2, allow_nan=False).encode("utf-8")
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("The planning file exceeds its supported size.")
    destination = Path(path)
    handle, temporary = tempfile.mkstemp(prefix=".aster-plan-", suffix=".json", dir=destination.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Planning JSON fields must be unique.")
        result[key] = value
    return result


def read_planning(path, volume, fingerprint):
    with Path(path).open("rb") as stream:
        content = stream.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("The planning file exceeds its supported size.")
    try:
        payload = json.loads(content.decode("utf-8-sig"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("The planning file is malformed.") from exc
    _keys(payload, {"format", "source", "implants", "margins_mm"}, "Planning file")
    if payload["format"] != FORMAT:
        raise ValueError("Unsupported implant planning format.")
    _match_volume(_fingerprint(payload["source"], "Saved planning source"), _fingerprint(fingerprint, "Current planning source"))
    source, implants, margins = _validate_planning(payload["implants"], payload["margins_mm"], volume, fingerprint)
    return {"format": FORMAT, "source": source, "implants": implants, "margins_mm": margins}
