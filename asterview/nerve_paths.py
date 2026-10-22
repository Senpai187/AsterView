"""Editable, source-bound mandibular canal paths in patient LPS millimeters.

Paths describe candidate bony canal trajectories, not nerve fibers or clinical
clearance margins. Lee skeletons and weighted graph geodesics preserve separate
mask components; extraction never joins gaps. An anatomical midsagittal X
position must be supplied to assign laterality: scanner origin and field-of-
view center are not anatomical midlines. Original candidates remain available
after edits and export. Curve rounding is display geometry, not new evidence.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from itertools import product
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects, label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from .segmentation import REVIEW_STATES, _integer, _positive, _provenance
from .session import _fingerprint, _keys, _match_volume, _string


FORMAT = "asterview.nerve-paths.v1"
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_PATHS = 64
MAX_NODES = 4096
MAX_TOTAL_NODES = 16384
MAX_CURVE_SAMPLES = 32768
MAX_UNDO_BYTES = 32 * 1024 * 1024
MAX_UNDO_STEPS = 64
MAX_ROI_VOXELS = 64_000_000
MAX_SKELETON_NODES = 32_000
MAX_COMPONENTS = 4096


class PathExtractionCancelled(RuntimeError):
    """No partial path collection is returned after cancellation."""


def _check_cancel(cancel):
    if cancel is not None and bool(cancel() if callable(cancel) else cancel.is_set()):
        raise PathExtractionCancelled("Canal path extraction canceled.")


def _points(value):
    try:
        points = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError):
        raise ValueError("Path nodes must be finite LPS coordinate triples.") from None
    if points.ndim != 2 or points.shape[1] != 3 or not 2 <= len(points) <= MAX_NODES:
        raise ValueError(f"A path requires 2 to {MAX_NODES} three-dimensional control nodes.")
    if not np.isfinite(points).all() or np.any(np.abs(points) > 1_000_000):
        raise ValueError("Path nodes must contain bounded finite LPS millimeters.")
    if not np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8):
        raise ValueError("A path must have nonzero physical length.")
    return points


def _path(value):
    _keys(value, {"id", "name", "side", "points_lps_mm", "diameter_mm", "color", "review", "provenance"}, "Canal path")
    identifier = _string(value["id"], "Path ID", 96)
    if re.fullmatch(r"[A-Za-z0-9_-]+", identifier) is None:
        raise ValueError("Path IDs must contain only letters, digits, underscores, or hyphens.")
    side, review = value["side"], value["review"]
    if not isinstance(side, str) or side not in {"right", "left", "unknown"}:
        raise ValueError("Path side must be right, left, or unknown.")
    if not isinstance(review, str) or review not in REVIEW_STATES:
        raise ValueError("Path review must be pending, accepted, or rejected.")
    color = value["color"]
    if not isinstance(color, str) or re.fullmatch(r"#[0-9a-fA-F]{6}", color) is None:
        raise ValueError("Path color must use #RRGGBB.")
    return {"id": identifier, "name": _string(value["name"], "Path name", 128),
            "side": side, "points_lps_mm": _points(value["points_lps_mm"]).tolist(),
            "diameter_mm": _positive(value["diameter_mm"], "Path visualization diameter", 20),
            "color": color, "review": review, "provenance": _provenance(value["provenance"])}


def _paths(value):
    if not isinstance(value, list) or len(value) > MAX_PATHS:
        raise ValueError(f"A collection may contain at most {MAX_PATHS} canal paths.")
    result, identifiers = [], set()
    for item in value:
        path = _path(item)
        if path["id"] in identifiers:
            raise ValueError("Canal path IDs must be unique.")
        identifiers.add(path["id"])
        result.append(path)
    if sum(len(path["points_lps_mm"]) for path in result) > MAX_TOTAL_NODES:
        raise ValueError("The collection exceeds the bounded control-node count.")
    return result


def _geometry(volume, source):
    if not volume.metadata.get("geometry_calibrated", True) or min(volume.dimensions) < 2:
        raise ValueError("Canal paths require calibrated three-dimensional source geometry.")
    actual = {"shape": list(volume.data.shape), "spacing": list(volume.spacing),
              "origin": volume.origin.tolist(), "direction": volume.direction.tolist(),
              "synthetic": bool(volume.metadata.get("synthetic", False))}
    for name in ("series_uid", "source_group", "sop_instance_digest", "frame_of_reference_uid"):
        if name in volume.metadata:
            actual[name] = str(volume.metadata[name])
    _match_volume(source, _fingerprint(actual, "Actual canal source"))


def _bounds(volume, points):
    index = volume.world_to_index(points)
    if np.any(index < -1e-6) or np.any(index > np.asarray(volume.dimensions) - 1 + 1e-6):
        raise ValueError("Canal control nodes must remain inside the current source volume.")
    if volume.valid_mask is not None:
        nearest = np.clip(np.rint(index).astype(int), 0, np.asarray(volume.dimensions) - 1)
        if not np.all(volume.valid_mask[nearest[..., 2], nearest[..., 1], nearest[..., 0]]):
            raise ValueError("Canal control nodes cannot occupy source padding voxels.")


def _linear_samples(points, step):
    result = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(end - start) / step - 1e-10)))
        if len(result) + count > MAX_CURVE_SAMPLES:
            raise ValueError("The requested curve exceeds the bounded display sample count.")
        result.extend(start + (end - start) * fraction for fraction in np.linspace(0, 1, count + 1)[1:])
    return np.asarray(result, dtype=np.float64)


def sample_path(points, step_mm=.5, smooth=True):
    """Sample an endpoint-anchored polyline or bounded rounded display curve.

    Rounded corners are quadratic Bézier pieces inside neighboring control
    triangles. Internal nodes are control handles, rather than interpolated
    anatomy points. No piece extrapolates or joins a different path. This is
    rigid-transform invariant and stays in the convex source grid when nodes
    are in bounds. ``smooth=False`` passes through every original node.
    """
    points = _points(points)
    step = _positive(step_mm, "Curve sample pitch", 20)
    if type(smooth) is not bool:
        raise ValueError("Curve smoothing must be boolean.")
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]]
    if not smooth or len(points) < 3:
        return _linear_samples(points, step)
    pieces, current = [], points[0]
    total = 0
    for before, node, after in zip(points[:-2], points[1:-1], points[2:]):
        # Trim no more than 20% of either edge; adjacent corners cannot overlap.
        radius = .2 * min(np.linalg.norm(node - before), np.linalg.norm(after - node))
        entry = node + (before - node) * radius / np.linalg.norm(before - node)
        exit = node + (after - node) * radius / np.linalg.norm(after - node)
        straight = _linear_samples(np.array([current, entry]), step)
        count = max(2, int(np.ceil((np.linalg.norm(entry - node) + np.linalg.norm(exit - node)) / step - 1e-10)))
        t = np.linspace(0, 1, count + 1)[1:, None]
        rounded = (1 - t)**2 * entry + 2 * (1 - t) * t * node + t**2 * exit
        pieces.extend([straight[:-1], np.array([entry]), rounded])
        total += len(straight) + len(rounded)
        if total > MAX_CURVE_SAMPLES:
            raise ValueError("The requested curve exceeds the bounded display sample count.")
        current = exit
    pieces.append(_linear_samples(np.array([current, points[-1]]), step)[1:])
    sampled = np.concatenate(pieces)
    if len(sampled) > MAX_CURVE_SAMPLES:
        raise ValueError("The requested curve exceeds the bounded display sample count.")
    return sampled


@dataclass
class NervePathCollection:
    source: dict
    paths: list[dict] = field(default_factory=list)
    revision: int = 0
    warnings: list[str] = field(default_factory=list)
    _original_paths: list[dict] | None = field(default=None, repr=False)
    _undo: list = field(init=False, default_factory=list, repr=False)
    _undo_bytes: int = field(init=False, default=0, repr=False)
    _gesture: list | None = field(init=False, default=None, repr=False)
    _gesture_changed: bool = field(init=False, default=False, repr=False)

    def __post_init__(self):
        self.source = _fingerprint(self.source, "Canal path source")
        self.paths = _paths(self.paths)
        self.revision = _integer(self.revision, "Path revision", 0, 2**63 - 1)
        if not isinstance(self.warnings, list) or len(self.warnings) > 256:
            raise ValueError("Path notes must be a bounded list.")
        self.warnings = [_string(item, "Path note", 1024) for item in self.warnings]
        self._original_paths = copy.deepcopy(self.paths) if self._original_paths is None else _paths(self._original_paths)

    @property
    def original_paths(self):
        return copy.deepcopy(self._original_paths)

    @property
    def can_undo(self):
        return bool(self._undo or self._gesture_changed)

    def validate_source(self, volume, fingerprint):
        _match_volume(self.source, _fingerprint(fingerprint, "Current canal source"))
        _geometry(volume, self.source)
        for item in self.paths + self._original_paths:
            _bounds(volume, item["points_lps_mm"])
        return self

    def get_path(self, path_id):
        for path in self.paths:
            if path["id"] == path_id:
                return path
        raise ValueError("Choose an existing canal path.")

    def samples(self, path_id, step_mm=.5, smooth=True):
        return sample_path(self.get_path(path_id)["points_lps_mm"], step_mm=step_mm, smooth=smooth)

    def _snapshot(self):
        snapshot = copy.deepcopy(self.paths)
        if len(json.dumps(snapshot, allow_nan=False).encode("utf-8")) > MAX_UNDO_BYTES:
            raise ValueError("This edit exceeds the bounded undo memory limit.")
        return snapshot

    def _push(self, snapshot):
        size = len(json.dumps(snapshot, allow_nan=False).encode("utf-8"))
        while self._undo and (len(self._undo) >= MAX_UNDO_STEPS or self._undo_bytes + size > MAX_UNDO_BYTES):
            _, removed_size = self._undo.pop(0)
            self._undo_bytes -= removed_size
        self._undo.append((snapshot, size))
        self._undo_bytes += size

    def _before_change(self):
        if self._gesture is None:
            self._push(self._snapshot())
        else:
            self._gesture_changed = True

    def begin_gesture(self):
        if self._gesture is not None:
            raise ValueError("Finish the current path edit before starting another.")
        self._gesture = self._snapshot()
        self._gesture_changed = False

    def end_gesture(self):
        changed = self._gesture_changed
        if self._gesture is not None and changed:
            self._push(self._gesture)
        self._gesture, self._gesture_changed = None, False
        return changed

    begin_stroke = begin_gesture
    end_stroke = end_gesture

    def undo(self):
        self.end_gesture()
        if not self._undo:
            return False
        self.paths, removed_size = self._undo.pop()
        self._undo_bytes -= removed_size
        self.revision += 1
        return True

    def _edit_nodes(self, volume, path_id, points):
        _geometry(volume, self.source)
        points = _points(points)
        _bounds(volume, points)
        path = self.get_path(path_id)
        if np.array_equal(points, np.asarray(path["points_lps_mm"])):
            return False
        if sum(len(item["points_lps_mm"]) for item in self.paths) - len(path["points_lps_mm"]) + len(points) > MAX_TOTAL_NODES:
            raise ValueError("The collection exceeds the bounded control-node count.")
        self._before_change()
        path["points_lps_mm"] = points.tolist()
        path["review"] = "pending"
        self.revision += 1
        return True

    def move_node(self, volume, path_id, node_index, position_lps_mm):
        path = self.get_path(path_id)
        node_index = _integer(node_index, "Control node", 0, len(path["points_lps_mm"]) - 1)
        position = np.asarray(position_lps_mm, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError("A control node requires an LPS coordinate triple.")
        points = np.asarray(path["points_lps_mm"]).copy()
        points[node_index] = position
        return self._edit_nodes(volume, path_id, points)

    def insert_node(self, volume, path_id, node_index, position_lps_mm):
        path = self.get_path(path_id)
        node_index = _integer(node_index, "Insertion index", 0, len(path["points_lps_mm"]))
        position = np.asarray(position_lps_mm, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError("A control node requires an LPS coordinate triple.")
        return self._edit_nodes(volume, path_id, np.insert(np.asarray(path["points_lps_mm"]), node_index, position, axis=0))

    def delete_node(self, path_id, node_index):
        path = self.get_path(path_id)
        node_index = _integer(node_index, "Control node", 0, len(path["points_lps_mm"]) - 1)
        points = _points(np.delete(np.asarray(path["points_lps_mm"]), node_index, axis=0))
        self._before_change()
        path["points_lps_mm"], path["review"] = points.tolist(), "pending"
        self.revision += 1
        return True

    def set_review(self, path_id, status):
        if not isinstance(status, str) or status not in REVIEW_STATES:
            raise ValueError("Path review must be pending, accepted, or rejected.")
        path = self.get_path(path_id)
        if path["review"] != status:
            path["review"] = status
            self.revision += 1

    def _set_property(self, path_id, key, value):
        path = self.get_path(path_id)
        updated = _path({**path, key: value})
        if path[key] != updated[key]:
            self._before_change()
            path[key], path["review"] = updated[key], "pending"
            self.revision += 1

    def set_diameter(self, path_id, diameter_mm):
        self._set_property(path_id, "diameter_mm", diameter_mm)

    def set_color(self, path_id, color):
        self._set_property(path_id, "color", color)

    def set_side(self, path_id, side):
        self._set_property(path_id, "side", side)

    def set_name(self, path_id, name):
        self._set_property(path_id, "name", name)

    def add_path(self, volume, points_lps_mm, *, name="Canal path", side="unknown", diameter_mm=2., color="#EE8487", provenance=None):
        _geometry(volume, self.source)
        if len(self.paths) >= MAX_PATHS:
            raise ValueError("The canal path count exceeds the supported limit.")
        candidate = {"id": str(uuid.uuid4()), "name": name, "side": side,
                     "points_lps_mm": points_lps_mm, "diameter_mm": diameter_mm,
                     "color": color, "review": "pending",
                     "provenance": provenance or {"name": "AsterView manual path", "version": "1"}}
        if isinstance(points_lps_mm, dict):
            candidate.update(copy.deepcopy(points_lps_mm))
            candidate["review"] = "pending"
        item = _path(candidate)
        _bounds(volume, item["points_lps_mm"])
        _paths(self.paths + [item])
        self._before_change()
        self.paths.append(item)
        self.revision += 1
        return item["id"]

    def delete_path(self, path_id):
        path = self.get_path(path_id)
        self._before_change()
        self.paths.remove(path)
        self.revision += 1

    remove_path = delete_path


def create_paths(volume, fingerprint, paths):
    return NervePathCollection(fingerprint, paths).validate_source(volume, fingerprint)


def _principal_path(skeleton, spacing_zyx, cancel):
    """A physical longest-endpoint geodesic, or a two-sweep loop candidate."""
    nodes = np.argwhere(skeleton)
    if not 2 <= len(nodes) <= MAX_SKELETON_NODES:
        raise ValueError("A canal skeleton must have 2 to 32000 supported centerline voxels.")
    lookup = {tuple(point): index for index, point in enumerate(nodes)}
    offsets = [(a, b, c) for a, b, c in product((-1, 0, 1), repeat=3) if (a, b, c) != (0, 0, 0)]
    rows, columns, weights = [], [], []
    for index, node in enumerate(nodes):
        if index % 256 == 0:
            _check_cancel(cancel)
        for offset in offsets:
            neighbor = lookup.get(tuple(node + offset))
            if neighbor is not None:
                rows.append(index)
                columns.append(neighbor)
                weights.append(float(np.linalg.norm(np.asarray(offset) * spacing_zyx)))
    graph = coo_matrix((weights, (rows, columns)), shape=(len(nodes), len(nodes))).tocsr()
    if connected_components(graph, directed=False, return_labels=False) != 1:
        raise ValueError("A canal skeleton split during thinning; review the retained source mask before tracing.")
    degree = np.diff(graph.indptr)
    endpoints = np.flatnonzero(degree == 1)
    branched = bool(np.any(degree > 2))
    ambiguous = not len(endpoints) or len(endpoints) > 64
    if len(endpoints) and len(endpoints) <= 64:
        sources = endpoints
    else:
        distances = dijkstra(graph, directed=False, indices=0)
        finite = np.isfinite(distances)
        sources = [int(np.argmax(np.where(finite, distances, -1)))]
    longest, best = -1., None
    for source in sources:
        _check_cancel(cancel)
        distances, previous = dijkstra(graph, directed=False, indices=int(source), return_predecessors=True)
        finite = np.isfinite(distances)
        target = int(np.argmax(np.where(finite, distances, -1)))
        if float(distances[target]) <= longest:
            continue
        chain, current = [], target
        while current != int(source):
            chain.append(current)
            current = int(previous[current])
            if current < 0 or len(chain) > len(nodes):
                raise ValueError("The extracted canal graph has disconnected centerline evidence.")
        chain.append(int(source))
        longest, best = float(distances[target]), nodes[chain[::-1]]
    return best, longest, branched, ambiguous


def _simplified_nodes(world, pitch, tolerance):
    """Retain physical bends, then fill long segments at the chosen node pitch."""
    selected = {0, len(world) - 1}
    stack = [(0, len(world) - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        direction = world[last] - world[first]
        denominator = float(direction @ direction)
        delta = world[first + 1:last] - world[first]
        projection = np.clip(delta @ direction / denominator, 0, 1) if denominator > 1e-15 else np.zeros(len(delta))
        distance = np.linalg.norm(delta - projection[:, None] * direction, axis=1)
        at = int(np.argmax(distance))
        if distance[at] > tolerance:
            index = first + 1 + at
            selected.add(index)
            stack.extend([(first, index), (index, last)])
    # Sampling each retained straight segment stays within the convex node hull.
    return _linear_samples(world[sorted(selected)], pitch)


def extract_canal_paths(volume, fingerprint, data, label_id=5, *, cancel=None, progress=None,
                        node_spacing_mm=2., midline_lps_x=None, provenance=None):
    """Extract separate editable candidates from supported source-mask components.

    Call on an immutable mask snapshot away from the GUI thread. The method is
    cancellable between bounded native skeleton/graph calls. It does not alter
    the segmentation mask, bridge gaps, infer an anatomical midline, or accept
    a candidate automatically. Branched/looped evidence is explicitly noted.
    """
    source = _fingerprint(fingerprint, "Canal extraction source")
    _geometry(volume, source)
    if not isinstance(data, np.ndarray) or data.shape != volume.data.shape or data.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError("Canal extraction requires unsigned labels on the exact source grid.")
    label_id = _integer(label_id, "Canal mask label", 1, 65535)
    pitch = _positive(node_spacing_mm, "Control-node pitch", 10)
    if midline_lps_x is not None:
        if isinstance(midline_lps_x, (bool, np.bool_)):
            raise ValueError("An anatomical midline must be a finite LPS X position.")
        midline_lps_x = float(midline_lps_x)
        if not np.isfinite(midline_lps_x) or abs(midline_lps_x) > 1_000_000:
            raise ValueError("An anatomical midline must be a finite LPS X position.")
    report = progress or (lambda _message: None)
    _check_cancel(cancel)
    low, high = np.asarray(data.shape), np.zeros(3, dtype=int)
    found = False
    for first in range(0, len(data), 8):
        _check_cancel(cancel)
        occupied = data[first:first + 8] == label_id
        if volume.valid_mask is not None:
            occupied &= volume.valid_mask[first:first + 8]
        if not occupied.any():
            continue
        # Axis projections bound scratch space even for a large source volume.
        axes = [np.flatnonzero(occupied.any(axis=tuple(other for other in range(3) if other != axis))) for axis in range(3)]
        offset = np.array([first, 0, 0])
        low = np.minimum(low, np.array([axis[0] for axis in axes]) + offset)
        high = np.maximum(high, np.array([axis[-1] for axis in axes]) + offset)
        found = True
    if not found:
        return NervePathCollection(source, [], warnings=["The source mask contains no canal voxels."])
    extent = high - low + 1
    if int(np.prod(extent)) > MAX_ROI_VOXELS:
        raise ValueError("The occupied canal region exceeds the bounded extraction size; correct or restrict the mask first.")
    region = tuple(slice(int(a), int(b + 1)) for a, b in zip(low, high))
    mask = data[region] == label_id
    if volume.valid_mask is not None:
        mask &= volume.valid_mask[region]
    report("Separating supported canal components…")
    connected, count = label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count > MAX_COMPONENTS:
        raise ValueError("The canal mask contains too many disconnected fragments; correct the mask before tracing.")
    objects = find_objects(connected)
    paths, notes, ignored = [], [], 0
    from skimage.morphology import skeletonize
    for number, slices in enumerate(objects, start=1):
        _check_cancel(cancel)
        if slices is None:
            continue
        component = connected[slices] == number
        if component.sum() < 2:
            ignored += 1
            continue
        report(f"Tracing canal component {number} of {count}…")
        skeleton = skeletonize(np.pad(component, 1), method="lee")[1:-1, 1:-1, 1:-1]
        _check_cancel(cancel)
        if np.count_nonzero(skeleton) < 2:
            ignored += 1
            continue
        chain, length, branched, ambiguous = _principal_path(skeleton, np.asarray(volume.spacing)[::-1], cancel)
        if length < min(float(np.min(volume.spacing)), .5):
            ignored += 1
            continue
        offset = low + np.array([item.start for item in slices])
        source_indices = chain + offset
        world = volume.index_to_world(source_indices[:, ::-1])
        nodes = _simplified_nodes(world, pitch, min(.35, min(volume.spacing) * .5))
        side = "unknown"
        if midline_lps_x is not None:
            if np.all(world[:, 0] < midline_lps_x - .5):
                side = "right"
            elif np.all(world[:, 0] > midline_lps_x + .5):
                side = "left"
        distance = distance_transform_edt(np.pad(component, 1), sampling=np.asarray(volume.spacing)[::-1])[1:-1, 1:-1, 1:-1]
        radius = distance[tuple(chain.T)]
        diameter = float(np.clip(2 * np.median(radius), .3, 20))
        producer = copy.deepcopy(provenance) if provenance is not None else {"name": "AsterView mask-derived canal path", "version": "1"}
        method = "Lee 3D skeleton; physically weighted connected geodesic; no gap bridging; diameter estimated from mask"
        producer["method"] = (producer.get("method", "") + "; " + method).strip("; ")[:1024]
        item = {"id": str(uuid.uuid4()), "name": f"Canal candidate {len(paths) + 1}", "side": side,
                "points_lps_mm": nodes.tolist(), "diameter_mm": diameter,
                "color": "#EE8487", "review": "pending", "provenance": producer}
        paths.append(item)
        if branched:
            notes.append(f"Candidate {len(paths)} has branched skeleton evidence; only its dominant geodesic is shown. Review the retained source mask.")
        if ambiguous:
            notes.append(f"Candidate {len(paths)} has ambiguous endpoints or a loop; its displayed trajectory is provisional.")
        if len(paths) > MAX_PATHS:
            raise ValueError("The source mask contains too many supported canal fragments; review the mask first.")
    _check_cancel(cancel)
    if ignored:
        notes.append(f"{ignored} isolated or untraceable fragment(s) were retained in the source mask but could not form a path.")
    if midline_lps_x is None and paths:
        notes.append("Laterality remains unknown until an anatomical midline or manual side is supplied.")
    return NervePathCollection(source, paths, warnings=notes).validate_source(volume, fingerprint)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Path files may not contain duplicate JSON fields.")
        result[key] = value
    return result


def export_nerve_paths(path, result, volume, fingerprint):
    result.validate_source(volume, fingerprint)
    payload = {"format": FORMAT, "source": _fingerprint(result.source, "Path source"),
               "paths": _paths(result.paths), "original_paths": _paths(result.original_paths),
               "revision": result.revision, "warnings": list(result.warnings)}
    # Validate even when a caller has bypassed the public mutation methods.
    NervePathCollection(payload["source"], payload["paths"], payload["revision"], payload["warnings"], payload["original_paths"])
    content = json.dumps(payload, ensure_ascii=True, allow_nan=False, indent=2).encode("utf-8")
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("The path export exceeds the supported 8 MiB size.")
    destination = Path(path)
    handle, temporary = tempfile.mkstemp(prefix=".aster-path-", suffix=".json", dir=destination.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_nerve_paths(path, volume, fingerprint, *, imported=True):
    if type(imported) is not bool:
        raise ValueError("The imported flag must be boolean.")
    with Path(path).open("rb") as stream:
        content = stream.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("Path files must be no larger than 8 MiB.")
    try:
        payload = json.loads(content.decode("utf-8-sig"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("The canal path file is malformed.") from exc
    _keys(payload, {"format", "source", "paths", "original_paths", "revision", "warnings"}, "Path file")
    if payload["format"] != FORMAT:
        raise ValueError("Unsupported canal path format.")
    result = NervePathCollection(payload["source"], payload["paths"], payload["revision"], payload["warnings"], payload["original_paths"])
    result.validate_source(volume, fingerprint)
    if imported:
        for path in result.paths:
            path["review"] = "pending"
    return result
