"""Source-bound reviewed findings and deterministic editable report drafts.

Only explicitly accepted observations enter generated drafts. Image candidates
and anatomy segmentation never become automatic diagnoses or normal findings.
"""
from __future__ import annotations

import copy
import json
from numbers import Real
import os
from pathlib import Path
import tempfile
import uuid

import numpy as np

from .session import _fingerprint, _match_volume, _keys
from .differential_support import validate_features, validate_support, REFERENCES


FORMAT = "asterview.findings.v2"
LEGACY_FORMAT = "asterview.findings.v1"
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_FINDINGS = 128
MAX_DRAFT_CHARS = 524_288


def _text(value, name, maximum, *, required=False):
    if not isinstance(value, str) or len(value) > maximum or any(ord(c) < 32 and c not in "\n\r\t" for c in value):
        raise ValueError(f"{name} must be bounded plain text without control characters.")
    if required and not value.strip():
        raise ValueError(f"{name} cannot be empty.")
    return value


def _source(volume, fingerprint):
    source = _fingerprint(fingerprint, "Current reporting source")
    actual = dict(source, shape=list(volume.data.shape), spacing=list(volume.spacing),
                  origin=np.asarray(volume.origin).tolist(), direction=np.asarray(volume.direction).tolist(),
                  synthetic=bool(volume.metadata.get("synthetic", False)))
    for key in ("series_uid", "frame_of_reference_uid", "source_group", "sop_instance_digest"):
        if not source["synthetic"]:
            actual[key] = str(volume.metadata.get(key, ""))
    _match_volume(source, _fingerprint(actual, "Actual reporting geometry"))
    return source


def make_finding(observation, differential="", location_lps_mm=None, *, finding_id=None, review="pending", origin="manual", morphology=(), support_ids=(), method="Clinician entered", search_id="", bounds_index_zyx=None, report_section="other"):
    from .university_report import SECTIONS
    if not isinstance(report_section, str) or report_section not in SECTIONS:
        raise ValueError("Choose a supported university report section.")
    if not isinstance(review, str) or review not in {"pending", "accepted", "rejected"}:
        raise ValueError("A finding requires pending, accepted or rejected review.")
    if not isinstance(origin, str) or origin not in {"manual", "candidate"}:
        raise ValueError("A finding origin must be manual or candidate.")
    item = {"id": str(uuid.uuid4()) if finding_id is None else _text(finding_id, "Finding ID", 128, required=True),
            "observation": _text(observation, "Observation", 8000, required=True),
            "differential": _text(differential, "Differential", 4000), "review": review, "origin": origin,
            "morphology": validate_features(morphology), "support_ids": validate_support(support_ids),
            "method": _text(method, "Finding method", 2048, required=True), "search_id": _text(search_id, "Search ID", 128),
            "report_section": report_section}
    if location_lps_mm is not None:
        point = np.asarray(location_lps_mm, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Finding location must contain three finite patient LPS coordinates.")
        item["location_lps_mm"] = point.tolist()
    if bounds_index_zyx is not None:
        bounds = np.asarray(bounds_index_zyx)
        if bounds.shape != (3, 2) or bounds.dtype.kind not in "iu" or np.any(bounds < 0) or np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("Finding bounds require increasing source z/y/x start-stop indices.")
        item["bounds_index_zyx"] = bounds.astype(int).tolist()
    return item


def new_findings(volume, fingerprint):
    return {"format": FORMAT, "source": _source(volume, fingerprint), "clinical_indication": "", "findings": [], "draft": "", "searches": []}


def validate_findings(document, volume, fingerprint):
    _keys(document, {"format", "source", "clinical_indication", "findings", "draft"}, "Findings document", {"searches", "university_report", "analysis_summary"})
    if document["format"] not in {FORMAT, LEGACY_FORMAT}:
        raise ValueError("Unsupported findings format.")
    current = _source(volume, fingerprint)
    saved = _fingerprint(document["source"], "Saved findings source")
    _match_volume(saved, current)
    items = document["findings"]
    if not isinstance(items, list) or len(items) > MAX_FINDINGS:
        raise ValueError("A findings document may contain at most 128 findings.")
    result, seen = [], set()
    for item in items:
        extras = {"location_lps_mm", "review", "origin", "morphology", "support_ids", "method", "search_id", "bounds_index_zyx", "report_section"}
        _keys(item, {"id", "observation", "differential"}, "Finding", extras)
        candidate = make_finding(item["observation"], item["differential"], item.get("location_lps_mm"), finding_id=item["id"],
                                 review="pending" if document["format"] == LEGACY_FORMAT else item.get("review", "pending"), origin=item.get("origin", "manual"),
                                 morphology=item.get("morphology", []), support_ids=item.get("support_ids", []),
                                 method=item.get("method", "Imported clinician finding; review required"), search_id=item.get("search_id", ""),
                                 bounds_index_zyx=item.get("bounds_index_zyx"), report_section=item.get("report_section", "other"))
        if candidate["id"] in seen:
            raise ValueError("Finding IDs must be unique.")
        seen.add(candidate["id"])
        if "location_lps_mm" in candidate:
            if not volume.metadata.get("geometry_calibrated", True) or min(volume.dimensions) < 2:
                raise ValueError("A physical finding location requires calibrated three-dimensional geometry.")
            index = volume.world_to_index(candidate["location_lps_mm"])
            if np.any(index < -1e-6) or np.any(index > np.asarray(volume.dimensions) - 1 + 1e-6):
                raise ValueError("Finding location is outside the exact source volume.")
            if volume.valid_mask is not None:
                nearest = np.rint(index).astype(int)
                if not volume.valid_mask[nearest[2], nearest[1], nearest[0]]:
                    raise ValueError("Finding bookmarks cannot occupy source padding.")
        if "bounds_index_zyx" in candidate and np.any(np.asarray(candidate["bounds_index_zyx"])[:, 1] > volume.data.shape):
            raise ValueError("Finding bounds exceed the exact source volume.")
        result.append(candidate)
    searches = document.get("searches", [])
    if not isinstance(searches, list) or len(searches) > 64:
        raise ValueError("A findings document may contain at most 64 candidate-search records.")
    validated_searches, search_ids = [], set()
    for search in searches:
        _keys(search, {"id", "method", "scope", "description", "candidate_count"}, "Candidate search")
        item = {key: _text(search[key], "Candidate search " + key, 4096 if key == "description" else 256, required=True) for key in ("id", "method", "scope", "description")}
        count = search["candidate_count"]
        if type(count) is not int or not 0 <= count <= MAX_FINDINGS or item["id"] in search_ids:
            raise ValueError("Candidate-search IDs must be unique and counts bounded.")
        search_ids.add(item["id"])
        item["candidate_count"] = count
        validated_searches.append(item)
    if any(item["search_id"] and item["search_id"] not in search_ids for item in result):
        raise ValueError("A candidate finding refers to an absent search record.")
    checked = {"format": FORMAT, "source": copy.deepcopy(saved),
            "clinical_indication": _text(document["clinical_indication"], "Clinical indication", 16000),
            "findings": result, "draft": _text(document["draft"], "Editable draft", MAX_DRAFT_CHARS), "searches": validated_searches}
    if "university_report" in document:
        from .university_report import validate_report
        checked["university_report"] = validate_report(document["university_report"], saved)
    if "analysis_summary" in document:
        summary = document["analysis_summary"]
        _keys(summary, {"summary", "coverage"}, "Report analysis summary")
        checked["analysis_summary"] = {key: _text(summary[key], "Analysis " + key, 16000)
                                       for key in ("summary", "coverage")}
    return checked


def validate_reviewed_measurements(measurements):
    """Only explicit finite reviewed values may enter a generated report.

    Each record has label/value/unit/reviewed; reviewed must be literally True.
    These are supplied measurements, never calculated or inferred by reporting.
    """
    if not isinstance(measurements, (list, tuple)) or len(measurements) > 128:
        raise ValueError("Reviewed measurements must be a bounded explicit list.")
    validated = []
    for item in measurements:
        _keys(item, {"label", "value", "unit", "reviewed"}, "Reviewed measurement")
        number = item["value"]
        if item["reviewed"] is not True:
            raise ValueError("A report measurement requires explicit clinician review.")
        if isinstance(number, (bool, np.bool_)) or not isinstance(number, Real) or not np.isfinite(number) or not 0 <= number <= 1_000_000:
            raise ValueError("A report measurement must be finite, numeric and nonnegative.")
        if not isinstance(item["unit"], str) or item["unit"] not in {"mm", "mm²", "mm³", "degrees"}:
            raise ValueError("Unsupported report measurement unit.")
        validated.append({"label": _text(item["label"], "Measurement label", 160, required=True),
                          "value": float(number), "unit": item["unit"], "reviewed": True})
    return validated


def build_report(document, volume, fingerprint, *, reviewed_measurements=()):
    """Build a draft solely from entered findings and selected reviewed facts."""
    document = validate_findings(document, volume, fingerprint)
    measurements = validate_reviewed_measurements(reviewed_measurements)
    if measurements and not volume.metadata.get("geometry_calibrated", True):
        raise ValueError("Physical measurements require calibrated source geometry.")
    lines = ["AsterView — reviewed editable draft", "Research prototype; automatic diagnostic pathology detection is unavailable.", "Only explicitly accepted findings are included; an empty list does not imply a normal examination.", "", "Source"]
    metadata = volume.metadata
    for label, key in (("Patient name", "patient_name"), ("Patient ID", "patient_id"),
                       ("Study date", "study_date"), ("Study", "study_description"),
                       ("Series", "series_description"), ("Modality", "modality")):
        value = metadata.get(key, "")
        if value is not None and str(value).strip():
            value = _text(str(value), f"Source {label}", 1024)
            lines.append(f"{label}: {value}")
    if document["source"]["synthetic"]:
        lines.append("Source type: synthetic research volume")
    elif document["source"].get("series_uid"):
        lines.append("Series UID: " + document["source"]["series_uid"])
    if document["clinical_indication"].strip():
        lines.extend(["", "Clinical indication (entered)", document["clinical_indication"]])
    accepted = [item for item in document["findings"] if item["review"] == "accepted"]
    lines.extend(["", "Findings (explicitly reviewed and accepted)"])
    if not accepted:
        lines.append("No accepted clinician findings have been supplied. This does not exclude disease.")
    for index, item in enumerate(accepted, 1):
        lines.append(f"{index}. {item['observation']}")
        lines.append("   Finding bookmark ID: " + item["id"])
        if item["differential"].strip():
            lines.append("   Differential (entered): " + item["differential"])
        if "location_lps_mm" in item:
            lines.append("   Bookmarked LPS location (mm): " + ", ".join(f"{value:.3f}" for value in item["location_lps_mm"]))
        for key in item["support_ids"]:
            lines.append("   Clinician-selected reference: " + REFERENCES[key]["title"] + " — " + REFERENCES[key]["url"])
    if document["searches"]:
        lines.extend(["", "Limited candidate-search coverage (not a negative examination)"])
        for search in document["searches"]:
            lines.append(search["scope"] + ": " + search["description"])
    if measurements:
        lines.extend(["", "Explicitly selected, reviewed measurements"])
        lines.extend(f"- {item['label']}: {item['value']:.6g} {item['unit']}" for item in measurements)
    text = "\n".join(lines) + "\n"
    return _text(text, "Generated draft", MAX_DRAFT_CHARS)


def _atomic_text(path, text):
    destination = Path(path)
    content = text.encode("utf-8")
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("The reporting file exceeds its supported size.")
    handle, temporary = tempfile.mkstemp(prefix=".aster-report-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def export_text(path, text):
    _atomic_text(path, _text(text, "Report draft", MAX_DRAFT_CHARS, required=True))


def export_findings(path, document, volume, fingerprint):
    document = validate_findings(document, volume, fingerprint)
    _atomic_text(path, json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Findings JSON contains duplicate fields.")
        result[key] = value
    return result


def read_findings(path, volume, fingerprint):
    with Path(path).open("rb") as stream:
        content = stream.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("The findings file exceeds its supported size.")
    try:
        document = json.loads(content.decode("utf-8"), object_pairs_hook=_unique_object)
        return validate_findings(document, volume, fingerprint)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("The findings file is malformed.") from exc


generate_report = build_report
