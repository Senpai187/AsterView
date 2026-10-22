"""Source-bound navigation landmarks and human review, independent of any model.

This format is AsterView's own interchange contract. It does not load models,
execute code, or imply that a producer's confidence has been calibrated.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .session import _fingerprint, _match_volume, _keys, _number, _string, _vector


FORMAT = "asterview.landmarks.v1"
MAX_BYTES = 2 * 1024 * 1024
MAX_LANDMARKS = 512
REVIEW_STATES = {"pending", "accepted", "rejected"}


def validate_landmarks(payload, volume, fingerprint, *, imported=True):
    """Validate the entire file before applying anything to the current study.

    Imported review states are reset to pending; reopening someone else's
    result never silently accepts their candidates. This is a navigation and
    review format, not a segmentation or diagnostic result format.
    """
    _keys(payload, {"format", "volume", "producer", "landmarks"}, "Landmark file")
    if payload["format"] != FORMAT:
        raise ValueError("Unsupported landmark format; expected asterview.landmarks.v1.")
    if not volume.metadata.get("geometry_calibrated", True) or min(volume.dimensions) < 2:
        raise ValueError("Landmarks require calibrated three-dimensional volume geometry.")
    saved = _fingerprint(payload["volume"], "Landmark source volume")
    _match_volume(saved, _fingerprint(fingerprint, "Current volume"))
    producer = _string(payload["producer"], "Landmark producer", 160)
    candidates = payload["landmarks"]
    if not isinstance(candidates, list) or len(candidates) > MAX_LANDMARKS:
        raise ValueError(f"A landmark file may contain at most {MAX_LANDMARKS} points.")
    items, identifiers = [], set()
    for candidate in candidates:
        _keys(candidate, {"id", "name", "position_lps_mm"}, "Landmark",
              {"confidence", "review", "producer"})
        identifier = _string(candidate["id"], "Landmark ID", 96)
        if identifier in identifiers:
            raise ValueError("Landmark IDs must be unique within a result.")
        identifiers.add(identifier)
        position = _vector(candidate["position_lps_mm"], 3, "Landmark position")
        index = volume.world_to_index(position)
        if np.any(index < -1e-5) or np.any(index > np.asarray(volume.dimensions) - 1 + 1e-5):
            raise ValueError("A landmark lies outside the current source volume.")
        review = candidate.get("review", "pending")
        if not isinstance(review, str) or review not in REVIEW_STATES:
            raise ValueError("Landmark review must be pending, accepted, or rejected.")
        item = {"id": identifier, "name": _string(candidate["name"], "Landmark name", 96),
                "position_lps_mm": position, "review": "pending" if imported else review,
                "producer": _string(candidate.get("producer", producer), "Point producer", 160)}
        if "confidence" in candidate:
            item["confidence"] = _number(candidate["confidence"], "Producer confidence", 0, 1)
        items.append(item)
    return {"format": FORMAT, "volume": saved, "producer": producer, "landmarks": items}


def read_landmarks(path, volume, fingerprint):
    with Path(path).open("rb") as stream:
        content = stream.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise ValueError("Landmark files must be no larger than 2 MiB.")
    return validate_landmarks(json.loads(content.decode("utf-8-sig")), volume, fingerprint)


def export_landmarks(path, result, volume, fingerprint):
    validated = validate_landmarks(result, volume, fingerprint, imported=False)
    content = json.dumps(validated, indent=2, allow_nan=False)
    if len(content.encode("utf-8")) > MAX_BYTES:
        raise ValueError("The landmark export exceeds the supported file size.")
    Path(path).write_text(content, encoding="utf-8")


def empty_result(fingerprint):
    return {"format": FORMAT, "volume": copy.deepcopy(fingerprint),
            "producer": "AsterView manual landmarks", "landmarks": []}
