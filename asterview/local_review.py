"""Local source inventory and bounded image candidates, without VLM diagnosis.

Every source-grid slice is visited for data integrity. This is not an anatomical
examination. The optional candidate search retains its existing limited
mandibular coverage and explicit clinician review requirement.
"""
from __future__ import annotations

import copy
import numpy as np

from .abnormality_candidates import detect_radiolucency_candidates, CandidateSearchCancelled
from .reporting import _source

CHUNK_VOXELS = 1_048_576
LIMITATIONS = (
    "The full-volume pass checks source data and padding; it does not interpret anatomy.",
    "Image candidates are limited to supported mandibular-mask coverage and focal relative low intensity.",
    "Other anatomy, lesion types and incidental findings are not comprehensively screened.",
    "No candidates does not establish a normal scan. Each candidate requires serial source review.",
    "No automatic diagnosis or differential diagnosis is generated.",
)


class LocalReviewCancelled(RuntimeError):
    pass


def _check(cancelled):
    if cancelled is not None and bool(cancelled() if callable(cancelled) else cancelled.is_set()):
        raise LocalReviewCancelled("Local review canceled.")


def analyze_local(volume, fingerprint, segmentation=None, *, progress=None, cancel=None, canalpaths=None):
    """Return inventory, optional ``batch``, explicit status and coverage text.

    Source arrays are read in at most CHUNK_VOXELS-element blocks. No source
    volume copy, external network call, model invocation or report acceptance
    occurs. Cancellation and changed anatomy discard the operation.
    """
    report = progress or (lambda percent, message: None)
    _check(cancel)
    data = volume.data
    if (not isinstance(data, np.ndarray) or data.ndim != 3 or min(data.shape) < 1
            or data.dtype.kind not in "fiu"):
        raise ValueError("Local review requires a nonempty real numeric three-dimensional source array.")
    source = _source(volume, fingerprint)
    mask = volume.valid_mask
    if mask is not None and (mask.shape != data.shape or mask.dtype != np.bool_):
        raise ValueError("Source padding mask must match the source grid.")
    anatomy = {"segmentation_status": "absent", "mask_revision": None,
               "label_count": 0, "canal_status": "absent", "canal_revision": None,
               "canal_count": 0}
    eligible = False
    segmentation_revision = None
    if segmentation is not None:
        try:
            segmentation.validate_source(volume, source)
        except ValueError as exc:
            anatomy["segmentation_status"] = "unavailable: " + str(exc)
        else:
            with segmentation._lock:
                segmentation_revision = segmentation.revision
                labels = copy.deepcopy(segmentation.labels)
            anatomy.update(segmentation_status="source_bound", mask_revision=segmentation_revision,
                           label_count=len(labels))
            eligible = any(item["kind"] == "jaw" and "mandib" in item["name"].lower()
                           and item.get("review") != "rejected" for item in labels.values())
    canal_revision = None
    if canalpaths is not None:
        try:
            canalpaths.validate_source(volume, source)
        except ValueError as exc:
            anatomy["canal_status"] = "unavailable: " + str(exc)
        else:
            canal_revision = canalpaths.revision
            anatomy.update(canal_status="source_bound", canal_revision=canal_revision,
                           canal_count=len(canalpaths.paths))
    records = []
    total_valid = total_padding = total_nonfinite = 0
    minimum, maximum = np.inf, -np.inf
    for z, plane in enumerate(data):
        _check(cancel)
        valid_count = padding_count = nonfinite_count = 0
        low, high = np.inf, -np.inf
        for start in range(0, plane.size, CHUNK_VOXELS):
            _check(cancel)
            # flat slicing has bounded allocation even for a noncontiguous view.
            block = plane.flat[start:start + CHUNK_VOXELS]
            if mask is None:
                valid = block
            else:
                valid_mask = mask[z].flat[start:start + CHUNK_VOXELS]
                padding_count += int(valid_mask.size - np.count_nonzero(valid_mask))
                valid = block[valid_mask]
            valid_count += valid.size
            finite = np.isfinite(valid)
            nonfinite_count += int(valid.size - np.count_nonzero(finite))
            values = valid[finite]
            if values.size:
                low, high = min(low, float(values.min())), max(high, float(values.max()))
        total_valid += valid_count
        total_padding += padding_count
        total_nonfinite += nonfinite_count
        minimum, maximum = min(minimum, low), max(maximum, high)
        records.append({"source_slice_index": z, "valid_voxels": int(valid_count),
                        "padding_voxels": int(padding_count), "nonfinite_valid_voxels": int(nonfinite_count),
                        "minimum": float(low) if np.isfinite(low) else None,
                        "maximum": float(high) if np.isfinite(high) else None})
        report(round(30 * (z + 1) / len(data)),
               f"Checking source data: slice {z + 1} of {len(data)}")
    inventory = {
        "shape_zyx": list(data.shape), "dtype": str(data.dtype),
        "slices_visited": len(records), "source_voxels": int(data.size),
        "valid_voxels": int(total_valid), "padding_voxels": int(total_padding),
        "nonfinite_valid_voxels": int(total_nonfinite),
        "minimum": float(minimum) if np.isfinite(minimum) else None,
        "maximum": float(maximum) if np.isfinite(maximum) else None,
        "slices": records,
    }
    batch = None
    status = "inventory_only"
    candidate_status = "Candidate search not run: no eligible source-bound mandibular mask."
    if total_nonfinite or not total_valid:
        status = "invalid_source"
        candidate_status = "Candidate search not run: invalid or absent non-padding source values."
    elif eligible:
        try:
            batch = detect_radiolucency_candidates(
                volume, source, segmentation,
                progress=lambda percent, message: report(30 + round(.65 * percent), message),
                cancel=cancel)
        except CandidateSearchCancelled as exc:
            raise LocalReviewCancelled(str(exc)) from exc
        except ValueError as exc:
            candidate_status = "Candidate search unavailable: " + str(exc)
        else:
            status = "candidates_ready"
            candidate_status = (f"{len(batch.findings)} pending image candidates in "
                                f"{batch.coverage['scope'].lower()}.")
    _check(cancel)
    if segmentation_revision is not None and segmentation.revision != segmentation_revision:
        raise RuntimeError("Anatomy changed during local review; run the review again.")
    if canal_revision is not None and canalpaths.revision != canal_revision:
        raise RuntimeError("Canal paths changed during local review; run the review again.")
    _source(volume, source)
    summary = (
        f"Local data inventory visited all {len(records)} source-grid slices "
        f"({total_valid:,} valid and {total_padding:,} padding voxels). "
        f"{total_nonfinite:,} nonfinite valid-source values. {candidate_status} "
        "This is limited image analysis, not a complete anatomical examination. "
        "No diagnosis, differential diagnosis or normal-scan conclusion was generated."
    )
    report(100, "Local inventory and available limited candidates are ready for review.")
    coverage = (batch.coverage["description"] if batch is not None else candidate_status)
    coverage = "All source-grid slices received data-integrity checks only. " + coverage
    return {"source": source, "status": status, "inventory": inventory, "anatomy": anatomy,
            "batch": batch, "coverage": coverage, "candidate_status": candidate_status,
            "summary": summary, "limitations": list(LIMITATIONS),
            "vision_model_enabled": False}
