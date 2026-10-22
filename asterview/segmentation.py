"""Source-bound voxel labels, human review, and physical-plane correction.

This is AsterView's own non-executable interchange format, not DICOM SEG.
The source fingerprint uses the same exact-series/stack and calibrated LPS
geometry contract as workspace sessions. It deliberately contains no patient
name, filename, or filesystem path. Model provenance is a declaration, not a
guarantee of model validity. Imported review decisions default to pending.

The labelmap is owned by the result. Read ``data`` for rendering; modify it only
through ``paint``/``undo``. These methods and ``sample_plane`` share a lock, so a
worker can sample a plane without copying the entire source-grid labelmap.
An explicit ``snapshot`` is the independent, read-only copy for long operations.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import zipfile

import numpy as np
from scipy.ndimage import affine_transform

from .session import _fingerprint, _keys, _match_volume, _string


FORMAT = "asterview.segmentation.v1"
MAX_VOXELS = 536_870_912  # 1 GiB of uint16 labels; source image is not copied.
MAX_METADATA_BYTES = 262_144
MAX_ARCHIVE_BYTES = 1_100_000_000
MAX_LABELS = 512
MAX_STROKE_VOXELS = 2_000_000
MAX_BRUSH_ROI_VOXELS = 8_000_000
MAX_UNDO_BYTES = 32 * 1024 * 1024
REVIEW_STATES = {"pending", "accepted", "rejected"}
LABEL_KINDS = {"tooth", "canal", "jaw", "skull", "other"}


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}.")
    return int(value)


def _vector(value, name):
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite LPS coordinates.")
    return vector


def _axes(right, up):
    right, up = _vector(right, "Plane right"), _vector(up, "Plane up")
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-10:
        raise ValueError("Plane right must be nonzero.")
    right = right / right_norm
    up = up - np.dot(up, right) * right
    up_norm = float(np.linalg.norm(up))
    if up_norm < 1e-10:
        raise ValueError("Plane axes must be independent.")
    up /= up_norm
    return right, up, np.cross(right, up)


def _positive(value, name, maximum):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be positive and finite.") from None
    if not np.isfinite(number) or not 0 < number <= maximum:
        raise ValueError(f"{name} must be positive and no greater than {maximum:g}.")
    return number


def _labels(value):
    if not isinstance(value, dict) or not 1 <= len(value) <= MAX_LABELS:
        raise ValueError(f"A segmentation requires 1 to {MAX_LABELS} semantic labels.")
    result = {}
    for label_id, label in value.items():
        identifier = _integer(label_id, "Label ID", 1, 65535)
        _keys(label, {"name", "kind"}, "Segmentation label", {"review", "color", "fdi", "instance", "predicted_fdi"})
        name = _string(label["name"], "Label name", 128)
        kind = label["kind"]
        if not isinstance(kind, str) or kind not in LABEL_KINDS:
            raise ValueError("Unsupported segmentation label kind.")
        review = label.get("review", "pending")
        if not isinstance(review, str) or review not in REVIEW_STATES:
            raise ValueError("Review must be pending, accepted, or rejected.")
        item = {"name": name, "kind": kind, "review": review}
        if "color" in label:
            color = label["color"]
            if not isinstance(color, str) or re.fullmatch(r"#[0-9a-fA-F]{6}", color) is None:
                raise ValueError("Label color must use #RRGGBB.")
            item["color"] = color
        if "instance" in label:
            if type(label["instance"]) is not bool or kind != "tooth":
                raise ValueError("Instance must be a boolean on a tooth label.")
            item["instance"] = label["instance"]
        for field_name in ("fdi", "predicted_fdi"):
            if field_name not in label:
                continue
            fdi = _integer(label[field_name], "FDI tooth number", 11, 85)
            if kind != "tooth" or not (1 <= fdi // 10 <= 4 and 1 <= fdi % 10 <= 8 or 5 <= fdi // 10 <= 8 and 1 <= fdi % 10 <= 5):
                raise ValueError("FDI must be a valid permanent or primary tooth number.")
            if label.get("instance") is False:
                raise ValueError("Grouped teeth cannot carry an individual FDI number.")
            item[field_name] = fdi
        result[identifier] = item
    assigned = [item["fdi"] for item in result.values() if "fdi" in item]
    if len(assigned) != len(set(assigned)):
        raise ValueError("Each FDI number can identify only one tooth instance.")
    return result


def _provenance(value):
    _keys(value, {"name", "version"}, "Segmentation provenance", {"model_sha256", "method", "license"})
    result = {"name": _string(value["name"], "Producer name", 160),
              "version": _string(value["version"], "Producer version", 160)}
    for key in ("method", "license"):
        if key in value:
            result[key] = _string(value[key], f"Producer {key}", 1024)
    if "model_sha256" in value:
        digest = value["model_sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("The model digest must be lowercase SHA-256 hexadecimal.")
        result["model_sha256"] = digest
    return result


def _validate_data(data, source, labels):
    if not isinstance(data, np.ndarray) or data.ndim != 3 or data.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError("Labels must be a z, y, x uint8 or uint16 array.")
    if list(data.shape) != source["shape"] or data.size > MAX_VOXELS:
        raise ValueError("Labelmap dimensions must exactly match the bounded source grid.")
    allowed = {0, *labels}
    flat = data.reshape(-1)
    for start in range(0, flat.size, 1_048_576):
        if not set(int(v) for v in np.unique(flat[start:start + 1_048_576])).issubset(allowed):
            raise ValueError("Every nonzero voxel value must identify a declared label.")


@dataclass
class SegmentationResult:
    data: np.ndarray
    labels: dict[int, dict]
    source: dict
    provenance: dict
    revision: int = 0
    max_undo_bytes: int = MAX_UNDO_BYTES
    _lock: threading.RLock = field(init=False, repr=False)
    _undo: list = field(init=False, repr=False, default_factory=list)
    _undo_bytes: int = field(init=False, repr=False, default=0)
    _stroke: list | None = field(init=False, repr=False, default=None)
    _stroke_reviews: dict = field(init=False, repr=False, default_factory=dict)
    _stroke_bytes: int = field(init=False, repr=False, default=0)
    _number_undo: list = field(init=False, repr=False, default_factory=list)

    def __post_init__(self):
        source = _fingerprint(self.source, "Segmentation source")
        labels = _labels(self.labels)
        provenance = _provenance(self.provenance)
        revision = _integer(self.revision, "Segmentation revision", 0, 2**63 - 1)
        undo_bytes = _integer(self.max_undo_bytes, "Undo limit", 0, MAX_UNDO_BYTES)
        _validate_data(self.data, source, labels)
        self.data = np.array(self.data, copy=True, order="C")
        self.labels, self.source, self.provenance = labels, source, provenance
        self.revision, self.max_undo_bytes = revision, undo_bytes
        self._lock = threading.RLock()

    @property
    def labelmap(self):
        """Alias for integrations; mutation must use the correction methods."""
        return self.data

    @property
    def producer(self):
        return self.provenance

    @property
    def can_undo(self):
        return bool(self._undo or self._stroke)

    def validate_source(self, volume, fingerprint):
        if not volume.metadata.get("geometry_calibrated", True) or min(volume.dimensions) < 2:
            raise ValueError("Segmentation requires calibrated three-dimensional geometry.")
        current = _fingerprint(fingerprint, "Current segmentation source")
        _match_volume(_fingerprint(self.source, "Segmentation source"), current)
        actual = dict(current, shape=list(volume.data.shape), spacing=list(volume.spacing),
                      origin=np.asarray(volume.origin).tolist(), direction=np.asarray(volume.direction).tolist())
        _match_volume(current, _fingerprint(actual, "Actual volume geometry"))
        self._check_geometry(volume)
        if volume.valid_mask is not None:
            labels, valid = self.data.reshape(-1), volume.valid_mask.reshape(-1)
            for start in range(0, labels.size, 1_048_576):
                block = labels[start:start + 1_048_576]
                if np.any(block[~valid[start:start + block.size]]):
                    raise ValueError("Segmentation labels cannot occupy source padding voxels.")
        return self

    def _check_geometry(self, volume):
        """Cheap geometry gate during a brush or plane read; no voxel scan."""
        if bool(volume.metadata.get("synthetic", False)) != self.source["synthetic"]:
            raise ValueError("Segmentation source type does not match the current volume.")
        if list(volume.data.shape) != self.source["shape"]:
            raise ValueError("Segmentation shape does not match the current volume.")
        for name, atol in (("spacing", 1e-6), ("origin", 1e-4), ("direction", 1e-6)):
            if not np.allclose(getattr(volume, name), self.source[name], rtol=0, atol=atol):
                raise ValueError("Segmentation geometry does not match the current volume.")
        if not self.source["synthetic"]:
            for name in ("series_uid", "frame_of_reference_uid", "source_group", "sop_instance_digest"):
                if str(volume.metadata.get(name, "")) != self.source[name]:
                    raise ValueError("Segmentation belongs to a different source series or stack.")

    def set_review(self, label_id, status):
        label_id = _integer(label_id, "Label ID", 1, 65535)
        if label_id not in self.labels or not isinstance(status, str) or status not in REVIEW_STATES:
            raise ValueError("Choose a declared label and a supported review state.")
        with self._lock:
            if self.labels[label_id]["review"] != status:
                self.labels[label_id]["review"] = status
                self.revision += 1

    def assign_fdi(self, label_id, fdi, *, swap=False):
        """Rename an existing instance without relabeling voxels or losing prediction."""
        label_id = _integer(label_id, "Label ID", 1, 65535)
        if label_id not in self.labels or self.labels[label_id]["kind"] != "tooth":
            raise ValueError("Choose an individual tooth object.")
        with self._lock:
            current = self.labels[label_id]
            if not current.get("instance", "fdi" in current):
                raise ValueError("Grouped dentitions cannot be assigned one tooth number.")
            if type(swap) is not bool:
                raise ValueError("Swap must be boolean.")
            candidate = copy.deepcopy(self.labels)
            previous = current.get("fdi")
            if fdi is None:
                candidate[label_id].pop("fdi", None)
            else:
                fdi = _integer(fdi, "FDI tooth number", 11, 85)
                occupied = next((key for key, item in candidate.items() if key != label_id and item.get("fdi") == fdi), None)
                if occupied is not None:
                    if not swap:
                        raise ValueError(f"FDI {fdi} is already assigned. Enable swapping to exchange the numbers.")
                    if previous is None:
                        candidate[occupied].pop("fdi", None)
                    else:
                        candidate[occupied]["fdi"] = previous
                    candidate[occupied]["review"] = "pending"
                    candidate[occupied]["name"] = f"Tooth {previous}" if previous is not None else f"Tooth object {occupied} (unassigned)"
                candidate[label_id]["fdi"] = fdi
            candidate[label_id]["instance"] = True
            candidate[label_id]["name"] = f"Tooth {fdi}" if fdi is not None else f"Tooth object {label_id} (unassigned)"
            candidate[label_id]["review"] = "pending"
            checked = _labels(candidate)
            if checked == self.labels:
                return
            self._number_undo.append(copy.deepcopy(self.labels))
            self._number_undo = self._number_undo[-64:]
            self.labels = checked
            self.revision += 1

    def undo_fdi(self):
        with self._lock:
            if self._number_undo:
                before = self._number_undo.pop()
                # A numbering undo must not restore acceptance after a later
                # voxel correction or discard unrelated review decisions.
                for key, original in before.items():
                    current = self.labels[key]
                    fields = ("fdi", "instance", "name")
                    if any(current.get(field_name) != original.get(field_name) for field_name in fields):
                        for field_name in fields:
                            if field_name in original:
                                current[field_name] = original[field_name]
                            else:
                                current.pop(field_name, None)
                        current["review"] = "pending"
                self.revision += 1
                return True
            return False

    def snapshot(self):
        with self._lock:
            array = self.data.copy()
        array.flags.writeable = False
        return array

    def begin_stroke(self):
        """Group many pointer dabs into one sparse undo record."""
        with self._lock:
            if self._stroke is not None:
                raise ValueError("Finish the current brush stroke before starting another.")
            self._stroke, self._stroke_reviews, self._stroke_bytes = [], {}, 0

    def end_stroke(self):
        """Finish a gesture, retaining each voxel's value before its first dab."""
        with self._lock:
            if self._stroke is None:
                return 0
            stroke, reviews = self._stroke, self._stroke_reviews
            self._stroke, self._stroke_reviews, self._stroke_bytes = None, {}, 0
            if not stroke:
                return 0
            indices = np.concatenate([item[0] for item in stroke])
            before = np.concatenate([item[1] for item in stroke])
            indices, first = np.unique(indices, return_index=True)
            before = before[first]
            self._undo.append((indices, before, reviews))
            self._undo_bytes += indices.nbytes + before.nbytes
            return len(indices)

    def sample_plane(self, volume, frame, shape):
        """Nearest-neighbor labels with the same pixel centers as sample_plane.

        ``frame`` is ``(center_lps, right, up, (width_mm,height_mm))``. Out of
        bounds returns background. Review rejection never destroys voxel data.
        """
        self._check_geometry(volume)
        if not isinstance(frame, (tuple, list)) or len(frame) != 4:
            raise ValueError("A label plane requires center, right, up, and size.")
        center = _vector(frame[0], "Plane center")
        right, up, _ = _axes(frame[1], frame[2])
        if not isinstance(shape, (tuple, list)) or len(shape) != 2:
            raise ValueError("Plane shape must be rows and columns.")
        rows, columns = (_integer(v, "Plane size", 1, 8192) for v in shape)
        if rows * columns > 16_777_216:
            raise ValueError("The requested label plane is too large.")
        if len(frame[3]) != 2:
            raise ValueError("Plane field of view requires width and height.")
        width, height = (_positive(v, "Plane field of view", 1_000_000) for v in frame[3])
        spacing = np.asarray(volume.spacing)
        right_index = (np.asarray(volume.direction).T @ right) / spacing
        up_index = (np.asarray(volume.direction).T @ up) / spacing
        first = volume.world_to_index(center)
        if columns > 1:
            first = first - width / 2 * right_index
        if rows > 1:
            first = first + height / 2 * up_index
        down = -height / (rows - 1) * up_index if rows > 1 else np.zeros(3)
        across = width / (columns - 1) * right_index if columns > 1 else np.zeros(3)
        last = np.asarray(volume.dimensions) - 1
        corners = np.array([first + row * down + column * across
                            for row in (0, rows - 1) for column in (0, columns - 1)])
        valid = None
        if np.any(corners < -1e-7) or np.any(corners > last + 1e-7):
            yy, xx = np.ogrid[:rows, :columns]
            valid = np.ones((rows, columns), dtype=bool)
            for axis in range(3):
                index = first[axis] + yy * down[axis] + xx * across[axis]
                valid &= (index >= -1e-7) & (index <= last[axis] + 1e-7)
        # grid-constant avoids losing an edge pixel to -1e-15 roundoff; the
        # explicit center-domain mask above still returns zero beyond source.
        with self._lock:
            sampled = affine_transform(self.data, np.stack([down[::-1], across[::-1]], axis=1),
                                       offset=first[::-1], output_shape=(rows, columns),
                                       order=0, mode="grid-constant", cval=0, prefilter=False)
        if valid is not None:
            sampled[~valid] = 0
        return sampled

    def label_centroid(self, volume, label_id):
        self._check_geometry(volume)
        label_id = _integer(label_id, "Label ID", 1, 65535)
        if label_id not in self.labels:
            raise ValueError("Unknown segmentation label.")
        # Bounded per-slice scratch; no full-volume coordinate array.
        total, summed = 0, np.zeros(3, dtype=np.float64)
        with self._lock:
            for z, plane in enumerate(self.data):
                yy, xx = np.nonzero(plane == label_id)
                total += len(xx)
                summed += (float(xx.sum()), float(yy.sum()), z * len(xx))
        return None if not total else volume.index_to_world(summed / total)

    def paint(self, volume, center_lps, right, up, radius_mm, label_id, *, erase=False,
              thickness_mm=None, overwrite=False):
        """Apply a physical disk to source voxel centers and record sparse undo.

        The default slab depth is one source voxel's projected physical width
        along the plane normal. Explicit ``thickness_mm`` gives a cylindrical
        physical brush. Only selected-label voxels are erased. Other labels are
        protected during painting unless ``overwrite=True`` is explicit.
        """
        self._check_geometry(volume)
        center = _vector(center_lps, "Brush center")
        right, up, normal = _axes(right, up)
        radius = _positive(radius_mm, "Brush radius in millimeters", 100)
        label_id = _integer(label_id, "Label ID", 1, 65535)
        if label_id not in self.labels or label_id > np.iinfo(self.data.dtype).max:
            raise ValueError("Choose a declared label that fits the labelmap type.")
        if type(erase) is not bool or type(overwrite) is not bool:
            raise ValueError("Brush erase/overwrite flags must be boolean.")
        spacing = np.asarray(volume.spacing)
        projected_normal = np.asarray(volume.direction).T @ normal
        depth = float(np.sum(np.abs(projected_normal) * spacing)) if thickness_mm is None else _positive(thickness_mm, "Brush thickness", 100)
        center_index = volume.world_to_index(center)
        projected_right = np.asarray(volume.direction).T @ right
        projected_up = np.asarray(volume.direction).T @ up
        extent = (radius * (np.abs(projected_right) + np.abs(projected_up)) + depth / 2 * np.abs(projected_normal)) / spacing
        last = np.asarray(volume.dimensions) - 1
        if np.any(center_index + extent < 0) or np.any(center_index - extent > last):
            return 0
        lower = np.maximum(0, np.floor(np.clip(center_index - extent, -1, last + 1))).astype(np.int64)
        upper = np.minimum(last, np.ceil(np.clip(center_index + extent, -1, last + 1))).astype(np.int64)
        if np.any(lower > upper):
            return 0
        roi_shape = (upper - lower + 1)[::-1]
        if int(np.prod(roi_shape)) > MAX_BRUSH_ROI_VOXELS:
            raise ValueError("Brush region exceeds the bounded edit size; use a smaller brush.")
        z, y, x = np.ogrid[lower[2]:upper[2] + 1, lower[1]:upper[1] + 1, lower[0]:upper[0] + 1]
        delta = ((x - center_index[0]) * spacing[0], (y - center_index[1]) * spacing[1], (z - center_index[2]) * spacing[2])
        dr = sum(delta[i] * projected_right[i] for i in range(3))
        du = sum(delta[i] * projected_up[i] for i in range(3))
        dn = sum(delta[i] * projected_normal[i] for i in range(3))
        inside = (dr * dr + du * du <= radius * radius + 1e-10) & (np.abs(dn) <= depth / 2 + 1e-10)
        region = (slice(lower[2], upper[2] + 1), slice(lower[1], upper[1] + 1), slice(lower[0], upper[0] + 1))
        if volume.valid_mask is not None:
            inside &= volume.valid_mask[region]
        with self._lock:
            old_roi = self.data[region]
            inside &= old_roi == label_id if erase else ((old_roi != label_id) if overwrite else (old_roi == 0))
            count = int(np.count_nonzero(inside))
            if count > MAX_STROKE_VOXELS:
                raise ValueError("Brush changes too many voxels; use a smaller brush.")
            if not count:
                return 0
            zz, yy, xx = np.nonzero(inside)
            flat_indices = np.ravel_multi_index((zz + lower[2], yy + lower[1], xx + lower[0]), self.data.shape)
            flat = self.data.reshape(-1)
            before = flat[flat_indices].copy()
            needed = flat_indices.nbytes + before.nbytes
            if self._stroke_bytes + needed > self.max_undo_bytes:
                # Refuse an edit that cannot be undone under the chosen limit.
                raise ValueError("Brush stroke exceeds the sparse undo memory limit.")
            while self._undo and self._undo_bytes + self._stroke_bytes + needed > self.max_undo_bytes:
                removed_indices, removed_before, _ = self._undo.pop(0)
                self._undo_bytes -= removed_indices.nbytes + removed_before.nbytes
            affected = {label_id, *(int(v) for v in np.unique(before) if v)}
            reviews = {key: self.labels[key]["review"] for key in affected}
            if self._stroke is None:
                self._undo.append((flat_indices, before, reviews))
                self._undo_bytes += needed
            else:
                self._stroke.append((flat_indices, before))
                self._stroke_bytes += needed
                for key, review in reviews.items():
                    self._stroke_reviews.setdefault(key, review)
            flat[flat_indices] = 0 if erase else label_id
            for key in affected:
                self.labels[key]["review"] = "pending"
            self.revision += 1
            return count

    def undo(self):
        with self._lock:
            if self._stroke is not None:
                self.end_stroke()
            if not self._undo:
                return 0
            indices, before, reviews = self._undo.pop()
            self._undo_bytes -= indices.nbytes + before.nbytes
            flat = self.data.reshape(-1)
            flat[indices] = before
            for label_id, review in reviews.items():
                self.labels[label_id]["review"] = review
            self.revision += 1
            return len(indices)


def create_result(volume, fingerprint, data, labels, provenance):
    return SegmentationResult(data, labels, fingerprint, provenance).validate_source(volume, fingerprint)


def _metadata(result):
    return {"format": FORMAT, "source": copy.deepcopy(result.source),
            "provenance": copy.deepcopy(result.provenance),
            "labels": [{"id": key, **copy.deepcopy(value)} for key, value in result.labels.items()],
            "revision": result.revision}


def export_segmentation(path, result, volume, fingerprint):
    """Atomically save owned labels and JSON metadata, with no executable objects."""
    result.validate_source(volume, fingerprint)
    with result._lock:
        labels = _labels(result.labels)
        _provenance(result.provenance)
        _integer(result.revision, "Segmentation revision", 0, 2**63 - 1)
        _validate_data(result.data, result.source, labels)
        metadata = json.dumps(_metadata(result), ensure_ascii=True, allow_nan=False).encode("utf-8")
        if len(metadata) > MAX_METADATA_BYTES:
            raise ValueError("Segmentation metadata exceeds the supported size.")
        snapshot = result.data.copy()
    destination = Path(path)
    handle, temporary = tempfile.mkstemp(prefix=".asterview-seg-", suffix=".npz", dir=destination.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, data=snapshot, metadata=np.frombuffer(metadata, dtype=np.uint8))
            stream.flush()
            os.fsync(stream.fileno())
        if Path(temporary).stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Segmentation archive exceeds the supported size.")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _npy_header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(stream, max_header_size=16_384)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(stream, max_header_size=16_384)
    raise ValueError("Unsupported segmentation array header.")


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Segmentation metadata contains duplicate JSON fields.")
        result[key] = value
    return result


def read_segmentation(path, volume, fingerprint, *, imported=True):
    """Validate headers, source identity, labels and arrays before returning.

    ZIP sizes and NPY dimensions are checked before NumPy allocates the mask;
    pickle loading is always disabled. A result is returned only after every
    field validates. Nothing is applied to the current viewer by this function.
    """
    if type(imported) is not bool:
        raise ValueError("The imported flag must be boolean.")
    path = Path(path)
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Segmentation archive exceeds the supported size.")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) != 2 or {item.filename for item in entries} != {"data.npy", "metadata.npy"}:
                raise ValueError("A segmentation archive must contain only data and metadata arrays.")
            specs = {}
            for name in ("metadata.npy", "data.npy"):
                info = archive.getinfo(name)
                maximum = MAX_METADATA_BYTES if name == "metadata.npy" else int(volume.data.size) * 2
                if info.file_size > maximum + 16_384 or info.flag_bits & 1:
                    raise ValueError("Segmentation array exceeds its supported size or is encrypted.")
                with archive.open(name) as stream:
                    shape, fortran, dtype = _npy_header(stream)
                if fortran or dtype.hasobject:
                    raise ValueError("Segmentation arrays must be plain contiguous numeric arrays.")
                specs[name] = (shape, dtype)
            meta_shape, meta_dtype = specs["metadata.npy"]
            if len(meta_shape) != 1 or meta_dtype != np.dtype("uint8") or meta_shape[0] > MAX_METADATA_BYTES:
                raise ValueError("Segmentation metadata must be bounded UTF-8 JSON bytes.")
            data_shape, data_dtype = specs["data.npy"]
            if tuple(data_shape) != tuple(volume.data.shape) or data_dtype not in (np.dtype("uint8"), np.dtype("uint16")) or volume.data.size > MAX_VOXELS:
                raise ValueError("Labelmap dimensions and integer type must match the bounded source grid.")
        with np.load(path, allow_pickle=False, max_header_size=16_384) as payload:
            metadata = json.loads(payload["metadata"].tobytes().decode("utf-8"), object_pairs_hook=_unique_json_object)
            _keys(metadata, {"format", "source", "provenance", "labels", "revision"}, "Segmentation metadata")
            if metadata["format"] != FORMAT:
                raise ValueError("Unsupported segmentation format.")
            source = _fingerprint(metadata["source"], "Saved segmentation source")
            _match_volume(source, _fingerprint(fingerprint, "Current segmentation source"))
            candidates = metadata["labels"]
            if not isinstance(candidates, list) or not 1 <= len(candidates) <= MAX_LABELS:
                raise ValueError("Segmentation labels must be a bounded list.")
            labels = {}
            for candidate in candidates:
                _keys(candidate, {"id", "name", "kind"}, "Segmentation label", {"review", "color", "fdi", "instance", "predicted_fdi"})
                identifier = _integer(candidate["id"], "Label ID", 1, 65535)
                if identifier in labels:
                    raise ValueError("Segmentation label IDs must be unique.")
                labels[identifier] = {key: value for key, value in candidate.items() if key != "id"}
            labels = _labels(labels)
            if imported:
                for label in labels.values():
                    label["review"] = "pending"
            result = SegmentationResult(payload["data"], labels, source, metadata["provenance"], metadata["revision"])
            return result.validate_source(volume, fingerprint)
    except (zipfile.BadZipFile, EOFError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("The segmentation archive is malformed or incomplete.") from exc
