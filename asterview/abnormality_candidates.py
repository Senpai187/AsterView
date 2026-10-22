"""Limited local image-candidate search; no learned pathology diagnosis.

This computes focal relative low-intensity regions inside a mandibular-mask
envelope or an explicitly selected local box. Normal marrow, anatomy, noise and
artifacts can be returned. It does not detect all lesions or infer normality.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import uuid

import numpy as np
from scipy.ndimage import (gaussian_filter, binary_closing, binary_fill_holes,
                           distance_transform_edt, label, find_objects)

from .reporting import _source, make_finding

MAX_WORKING_VOXELS = 8_000_000
MAX_SOURCE_VOXELS = 400_000_000
MAX_CANDIDATES = 32
METHOD = "Focal relative low-intensity search v1; not a diagnostic AI model"


class CandidateSearchCancelled(RuntimeError):
    pass


def _check(cancel):
    if cancel is not None and bool(cancel() if callable(cancel) else cancel.is_set()):
        raise CandidateSearchCancelled("Image-candidate search cancelled.")


@dataclass
class CandidateBatch:
    source: dict
    findings: list[dict]
    coverage: dict
    mask_revision: int | None


def _bounds(mask_data, identifiers, cancel):
    low = np.asarray(mask_data.shape)
    high = np.zeros(3, int)
    found = False
    for start in range(0, mask_data.shape[0], 16):
        _check(cancel)
        block = np.isin(mask_data[start:start + 16], identifiers)
        if block.any():
            positions = [np.flatnonzero(block.any(axis=axes)) for axes in ((1, 2), (0, 2), (0, 1))]
            first = np.array([axis[0] for axis in positions])
            last = np.array([axis[-1] + 1 for axis in positions])
            first[0] += start
            last[0] += start
            low, high, found = np.minimum(low, first), np.maximum(high, last), True
    if not found:
        raise ValueError("No mandibular-mask voxels are available. Review the mask or choose a local box explicitly.")
    return low, high


def detect_radiolucency_candidates(volume, fingerprint, segmentation=None, *, roi_center_lps=None, roi_size_mm=50., min_contrast=.12, progress=None, cancel=None):
    """Return pending source bookmarks from actual bounded CPU computation.

    Default scope requires a non-rejected mandible semantic mask. An explicit
    local box is a reviewer-chosen region and is not anatomical segmentation.
    Relative image contrast is never interpreted as HU or disease probability.
    """
    report = progress or (lambda percent, message: None)
    _check(cancel)
    source = _source(volume, fingerprint)
    if volume.data.size > MAX_SOURCE_VOXELS:
        raise ValueError("This source exceeds the bounded candidate-search size (400 million voxels).")
    if not volume.metadata.get("geometry_calibrated", True) or min(volume.dimensions) < 8:
        raise ValueError("Candidate search requires calibrated three-dimensional geometry with at least eight voxels per axis.")
    if volume.metadata.get("nonuniform_spacing") or volume.metadata.get("nonuniform") or volume.metadata.get("regular_grid", True) is False:
        raise ValueError("Nonuniform source slices cannot be searched without explicit resampling.")
    if isinstance(min_contrast, bool) or not np.isfinite(min_contrast) or not .03 <= min_contrast <= .5:
        raise ValueError("Minimum relative contrast must be between 0.03 and 0.50.")
    spacing = np.asarray(volume.spacing)[::-1]
    revision, labels = None, {}
    if segmentation is not None:
        segmentation.validate_source(volume, fingerprint)
        with segmentation._lock:
            revision, labels = segmentation.revision, copy.deepcopy(segmentation.labels)
    mandible = [key for key, item in labels.items() if item["kind"] == "jaw" and "mandib" in item["name"].lower() and item.get("review") != "rejected"]
    report(5, "Bounding the explicitly selected search coverage…")
    if roi_center_lps is None:
        if segmentation is None or not mandible:
            raise ValueError("Create/open a mandibular segmentation first, or explicitly choose a local search box.")
        with segmentation._lock:
            low, high = _bounds(segmentation.data, mandible, cancel)
        scope = "Mandibular-mask envelope only"
    else:
        center = np.asarray(roi_center_lps, float)
        if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(roi_size_mm) or not 15 <= roi_size_mm <= 100:
            raise ValueError("A local box needs a finite source point and a 15–100 mm side length.")
        index = volume.world_to_index(center)[::-1]
        if np.any(index < 0) or np.any(index > np.asarray(volume.data.shape) - 1):
            raise ValueError("The selected local-box center is outside the source volume.")
        radius = float(roi_size_mm) / spacing / 2
        low = np.maximum(0, np.floor(index - radius)).astype(int)
        high = np.minimum(volume.data.shape, np.ceil(index + radius) + 1).astype(int)
        scope = "Reviewer-selected local source-axis box only"
    pitch = .75
    while True:
        stride = np.maximum(1, np.ceil(pitch / spacing)).astype(int)
        size = (high - low + stride - 1) // stride
        if int(np.prod(size)) <= MAX_WORKING_VOXELS:
            break
        pitch *= 1.1
    if min(size) < 5:
        raise ValueError("The bounded search region is too thin for three-dimensional candidates.")
    slices = tuple(slice(int(a), int(b), int(s)) for a, b, s in zip(low, high, stride))
    data = np.array(volume.data[slices], dtype=np.float32, copy=True)
    valid = np.ones(data.shape, bool) if volume.valid_mask is None else volume.valid_mask[slices].copy()
    work_spacing = spacing * stride
    _check(cancel)
    excluded = np.zeros(data.shape, bool)
    if segmentation is not None:
        with segmentation._lock:
            if segmentation.revision != revision:
                raise RuntimeError("The segmentation changed during candidate search; run it again.")
            mask = segmentation.data[slices].copy()
        if roi_center_lps is None:
            domain = np.isin(mask, mandible)
            # Fill only bounded mask gaps. This is a search envelope, not a new
            # lesion segmentation or proof that an internal cavity is disease.
            domain = binary_fill_holes(binary_closing(domain, iterations=2))
        else:
            domain = valid.copy()
        exclusions = [key for key, item in labels.items() if item["kind"] in {"tooth", "canal"} and item.get("review") != "rejected"]
        if exclusions:
            excluded = np.isin(mask, exclusions)
            if excluded.any():
                excluded = distance_transform_edt(~excluded, sampling=work_spacing) < 1.
        del mask
    else:
        domain = valid.copy()
    domain &= valid
    if np.count_nonzero(domain) < 256:
        raise ValueError("Too few supported voxels exist in this search envelope.")
    values = data[domain]
    lower, upper = np.percentile(values, [2, 98])
    # Avoid stretching a nearly uniform bony background's small texture over
    # the entire display range when a sparse dark focus occupies <2% of it.
    # This is an image-scale guard, not intensity calibration or disease score.
    full_range = float(values.max() - values.min())
    lower = min(float(lower), float(upper) - .25 * full_range)
    if not np.isfinite([lower, upper]).all() or upper - lower < 1e-5:
        raise ValueError("The selected region has insufficient image contrast for candidate search.")
    normalized = np.clip((data - lower) / (upper - lower), 0, 1)
    normalized[~valid] = 0
    del values, data
    report(25, "Comparing local image intensity at bounded physical scales…")
    _check(cancel)
    support = gaussian_filter(domain.astype(np.float32), 3. / work_spacing, truncate=3)
    baseline = gaussian_filter(normalized * domain, 3. / work_spacing, truncate=3) / np.maximum(support, 1e-6)
    _check(cancel)
    fine_support = gaussian_filter(domain.astype(np.float32), .65 / work_spacing, truncate=3)
    smooth = gaussian_filter(normalized * domain, .65 / work_spacing, truncate=3) / np.maximum(fine_support, 1e-6)
    residual = baseline - smooth
    edge_distance = distance_transform_edt(np.pad(domain, 1), sampling=work_spacing)[1:-1, 1:-1, 1:-1]
    interior = edge_distance >= 1.
    search = domain & interior & ~excluded & (support > .5) & (residual >= min_contrast)
    _check(cancel)
    report(55, "Grouping focal image candidates and excluding edge/elongated regions…")
    components, count = label(search, structure=np.ones((3, 3, 3), bool))
    if count > 10000:
        raise ValueError("The search is too fragmented. Narrow the explicit region or increase minimum contrast.")
    counts = np.bincount(components.reshape(-1), minlength=count + 1)
    objects = find_objects(components)
    voxel_volume = float(np.prod(work_spacing))
    candidates = []
    for identifier, bounds in enumerate(objects, 1):
        if identifier % 32 == 0:
            _check(cancel)
        physical_volume = counts[identifier] * voxel_volume
        if bounds is None or not 8 <= physical_volume <= 1500:
            continue
        extent = np.array([part.stop - part.start for part in bounds]) * work_spacing
        if extent.max() > 25 or extent.max() / max(extent.min(), 1e-6) > 4:
            continue
        local = components[bounds] == identifier
        if np.any(edge_distance[bounds][local] < 1.5):
            continue
        positions = np.argwhere(local) + [part.start for part in bounds]
        weights = np.maximum(residual[bounds][local], 1e-6)
        centroid = np.average(positions, axis=0, weights=weights)
        point = volume.index_to_world((low + centroid * stride)[::-1])
        source_bounds = [[int(low[i] + part.start * stride[i]), int(min(high[i], low[i] + part.stop * stride[i]))] for i, part in enumerate(bounds)]
        candidates.append((float(weights.mean()), point, source_bounds))
    candidates.sort(key=lambda item: item[0], reverse=True)
    found_count = len(candidates)
    candidates = candidates[:MAX_CANDIDATES]
    search_id = str(uuid.uuid4())
    findings = []
    for _, point, source_bounds in candidates:
        findings.append(make_finding("Focal low-intensity image candidate. Review serial sections to confirm anatomy and exclude normal spaces, marrow variation and artifacts.",
                                     location_lps_mm=point, origin="candidate", method=METHOD, search_id=search_id, bounds_index_zyx=source_bounds))
    _check(cancel)
    if segmentation is not None and segmentation.revision != revision:
        raise RuntimeError("The segmentation changed during candidate search; run it again.")
    _source(volume, fingerprint)
    description = (f"Source z/y/x start-stop indices {[[int(a), int(b)] for a, b in zip(low, high)]}; working pitch xyz "
                   f"{work_spacing[::-1].round(3).tolist()} mm; minimum relative local contrast {min_contrast:.3f}; "
                   f"8–1500 mm³ image components, maximum extent 25 mm, 1 mm envelope-edge exclusion; "
                   f"known tooth/canal masks excluded when available. {len(findings)} pending bookmarks retained"
                   + (f" from {found_count} eligible components (32-result cap). " if found_count > MAX_CANDIDATES else ". ")
                   + "Normal anatomy/artifacts may be returned. This search misses other regions and lesion types; no candidates does not mean normal.")
    if segmentation is not None:
        description += f" Anatomy mask revision {revision}; mask accuracy was not established by this search."
    coverage = {"id": search_id, "method": METHOD, "scope": scope, "description": description, "candidate_count": len(findings)}
    report(100, "Limited image candidates are ready for explicit source review.")
    return CandidateBatch(source, findings, coverage, revision)
