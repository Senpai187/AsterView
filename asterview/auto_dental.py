"""Source-bound, editable dental setup proposals derived from reviewed masks.

These geometric helpers perform no new learned inference or diagnosis. They
never modify source images, masks, or review decisions. Every returned setup
is pending and retains the exact source fingerprint and mask revision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
from itertools import product
from numbers import Real

import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter, label as label_components
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize


class AutoDentalCancelled(RuntimeError):
    """A cancelled setup never supplies a partial proposal."""


@dataclass
class DentalArchProposal:
    points_lps_mm: np.ndarray
    source: dict
    provenance: dict
    warnings: list[str] = field(default_factory=list)
    review: str = "pending"
    source_revision: int = 0


@dataclass
class TMJProposal:
    centers_lps_mm: dict[str, np.ndarray]
    axes_lps: dict[str, np.ndarray]
    source: dict
    provenance: dict
    warnings: list[str] = field(default_factory=list)
    review: str = "pending"
    source_revision: int = 0
    midline_lps_x: float | None = None

    @property
    def centers(self):
        return self.centers_lps_mm

    @property
    def axes(self):
        return self.axes_lps


def _check_cancel(cancel):
    if cancel is not None and (cancel() if callable(cancel) else cancel.is_set()):
        raise AutoDentalCancelled("Automatic dental setup cancelled.")


def _vector(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain three finite patient LPS coordinates.")
    return value


def _frame(center, right, up):
    center = _vector(center, "Plane center")
    right, up = _vector(right, "Plane right"), _vector(up, "Plane up")
    if np.linalg.norm(right) < 1e-10 or np.linalg.norm(up) < 1e-10:
        raise ValueError("Plane axes must be nonzero.")
    right, up = right / np.linalg.norm(right), up / np.linalg.norm(up)
    if abs(float(right @ up)) > 1e-5:
        raise ValueError("Plane axes must be orthogonal.")
    return center, right, up


def _input(volume, segmentation, fingerprint, cancel):
    _check_cancel(cancel)
    segmentation.validate_source(volume, fingerprint)
    with segmentation._lock:
        revision = segmentation.revision
        data = segmentation.data.copy()
        labels = copy.deepcopy(segmentation.labels)
        source = copy.deepcopy(segmentation.source)
        producer = copy.deepcopy(segmentation.provenance)
    _check_cancel(cancel)
    return data, labels, source, producer, revision


def _check_revision(segmentation, revision, cancel):
    _check_cancel(cancel)
    if segmentation.revision != revision:
        raise ValueError("The segmentation changed during automatic setup; run the proposal again.")


def _tooth_side(label):
    number = label.get("fdi", label.get("predicted_fdi"))
    if number is not None:
        return "lower" if number // 10 in (3, 4, 7, 8) else "upper"
    name = label["name"].casefold()
    if "lower" in name or "mandibular" in name:
        return "lower"
    if "upper" in name or "maxillary" in name:
        return "upper"
    return None


def _selected_cloud(volume, data, ids, cancel, limit=350_000):
    """Physical point cloud with bounded per-slice scratch and sample count."""
    counts = 0
    for plane in data:
        _check_cancel(cancel)
        counts += int(np.count_nonzero(np.isin(plane, ids)))
    if counts == 0:
        return np.empty((0, 3))
    step = max(1, int(np.ceil(counts / limit)))
    chunks, seen = [], 0
    for z, plane in enumerate(data):
        _check_cancel(cancel)
        yy, xx = np.nonzero(np.isin(plane, ids))
        start = (-seen) % step
        seen += len(xx)
        if start < len(xx):
            points = np.column_stack((xx[start::step], yy[start::step], np.full(len(xx[start::step]), z)))
            chunks.append(volume.index_to_world(points))
    return np.concatenate(chunks) if chunks else np.empty((0, 3))


def _centroids(volume, data, ids, cancel):
    sums = {key: np.zeros(3) for key in ids}
    counts = {key: 0 for key in ids}
    for z, plane in enumerate(data):
        _check_cancel(cancel)
        for key in ids:
            yy, xx = np.nonzero(plane == key)
            counts[key] += len(xx)
            sums[key] += (float(xx.sum()), float(yy.sum()), z * len(xx))
    return np.asarray([volume.index_to_world(sums[key] / counts[key]) for key in ids if counts[key]], dtype=np.float64).reshape(-1, 3)


def _graph_path(graph, cancel):
    """A bounded longest geodesic without fabricating disconnected links."""
    size = graph.shape[0]
    if size < 3:
        raise ValueError("At least three supported arch locations are needed.")
    degree = np.diff(graph.indptr)
    endpoints = np.flatnonzero(degree == 1)
    warnings = []
    if np.any(degree > 2):
        warnings.append("The mask-derived arch has branches; only its dominant course is proposed.")
    if not len(endpoints) or len(endpoints) > 48:
        distances = dijkstra(graph, directed=False, indices=0)
        endpoints = [int(np.argmax(np.where(np.isfinite(distances), distances, -1)))]
        warnings.append("The projected mask has a loop or complex branch pattern; verify the proposed course.")
    best_length, best = -1, None
    for first in endpoints:
        _check_cancel(cancel)
        distances, previous = dijkstra(graph, directed=False, indices=int(first), return_predecessors=True)
        last = int(np.argmax(np.where(np.isfinite(distances), distances, -1)))
        if distances[last] <= best_length:
            continue
        chain = [last]
        while chain[-1] != int(first):
            parent = int(previous[chain[-1]])
            if parent < 0 or len(chain) > size:
                raise ValueError("The arch support is disconnected.")
            chain.append(parent)
        best_length, best = float(distances[last]), np.asarray(chain[::-1])
    return best, warnings


def _centroid_arch(points, max_gap_mm, cancel):
    distance = cdist(points, points)
    adjacency = (distance > 1e-7) & (distance <= max_gap_mm)
    count, membership = connected_components(csr_matrix(adjacency), directed=False)
    groups = [np.flatnonzero(membership == key) for key in range(count)]
    selected = max(groups, key=len)
    warnings = []
    if count > 1:
        warnings.append("Large gaps separate dental support; the proposal uses one contiguous region and does not imply full-jaw coverage.")
    if len(selected) < 3:
        raise ValueError("No connected region contains three teeth; correct the mask or draw the arch manually.")
    nodes = points[selected]
    distance = cdist(nodes, nodes)
    distance[distance > max_gap_mm] = 0
    tree = minimum_spanning_tree(csr_matrix(distance)).tocsr()
    graph = tree + tree.T
    order, extra = _graph_path(graph, cancel)
    if len(order) < len(nodes):
        warnings.append("Some tooth candidates lie away from the dominant arch and were not used; inspect every tooth.")
    if graph.data.size and graph.data.max() > 12:
        warnings.append("The proposed arch spans a dental-support gap above 12 mm; inspect the interpolation across that gap.")
    return nodes[order], warnings + extra


def _projection_arch(points, cancel, pitch=.8):
    """Centerline of the actual projected occupancy, bounded to a small raster."""
    low = points.min(axis=0) - 4
    shape = np.ceil((points.max(axis=0) - low + 4) / pitch).astype(int) + 1
    if np.prod(shape, dtype=np.int64) > 1_000_000 or max(shape) > 1600:
        raise ValueError("Projected dental region exceeds the bounded setup size.")
    grid = np.zeros(tuple(shape), dtype=bool)
    ij = np.rint((points - low) / pitch).astype(int)
    grid[ij[:, 0], ij[:, 1]] = True
    radius = int(np.ceil(2.2 / pitch))
    xy = np.arange(-radius, radius + 1) * pitch
    disk = xy[:, None] ** 2 + xy[None, :] ** 2 <= 2.2 ** 2
    grid = binary_closing(gaussian_filter(grid.astype(float), .7) > .18, structure=disk)
    components, count = label_components(grid, np.ones((3, 3), dtype=bool))
    sizes = np.bincount(components.ravel())
    if count == 0:
        raise ValueError("The projected mask does not contain a usable dental ridge.")
    chosen = int(np.argmax(sizes[1:]) + 1)
    warnings = []
    if count > 1:
        warnings.append("Projected mask regions are disconnected; only the largest supported ridge is proposed, without bridging the missing anatomy.")
    nodes = np.argwhere(skeletonize(components == chosen))
    if not 3 <= len(nodes) <= 20_000:
        raise ValueError("The projected dental ridge is empty or too complex; draw or correct the arch manually.")
    lookup = {tuple(node): i for i, node in enumerate(nodes)}
    rows, columns, weights = [], [], []
    for i, node in enumerate(nodes):
        if i % 256 == 0:
            _check_cancel(cancel)
        for offset in product((-1, 0, 1), repeat=2):
            if offset == (0, 0):
                continue
            j = lookup.get(tuple(node + offset))
            if j is not None:
                rows.append(i); columns.append(j); weights.append(pitch * np.linalg.norm(offset))
    graph = coo_matrix((weights, (rows, columns)), shape=(len(nodes), len(nodes))).tocsr()
    if connected_components(graph, directed=False, return_labels=False) != 1:
        raise ValueError("Projected ridge thinning is disconnected; review the mask before automatic setup.")
    order, extra = _graph_path(graph, cancel)
    return low + nodes[order] * pitch, warnings + extra


def _uniform_controls(points, count):
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-7]
    points = points[keep]
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if len(points) < 3 or arc[-1] < 4:
        raise ValueError("The supported arch is too short for automatic reformatting.")
    steps = np.linspace(0, arc[-1], min(count, max(3, len(points))))
    return np.column_stack([np.interp(steps, arc, points[:, axis]) for axis in range(points.shape[1])])


def propose_arch(volume, segmentation, fingerprint, *, plane_center_lps, right_lps, up_lps,
                 preferred_jaw="lower", max_gap_mm=18., control_points=15, cancel=None,
                 depth_mode="current"):
    """Propose an editable mask-derived arch with the corrected orientation.

    Tooth-instance centroids are preferred. Grouped teeth or mandibular masks
    use a projected occupancy ridge. Missing regions are not replaced with a
    phantom/full-jaw template, and the result is always pending review.
    ``depth_mode="anatomy"`` places the plane at the median supporting anatomy
    along its normal, independently of the initial viewing/crosshair height.
    ``"current"`` retains the supplied plane depth for intentional placement.
    """
    center, right, up = _frame(plane_center_lps, right_lps, up_lps)
    requested_center = center.copy()
    if depth_mode not in {"current", "anatomy"}:
        raise ValueError("Arch depth mode must be current or anatomy.")
    normal = np.cross(right, up)
    if preferred_jaw not in {"lower", "upper", "auto"}:
        raise ValueError("Preferred jaw must be lower, upper or auto.")
    if isinstance(max_gap_mm, bool) or not isinstance(max_gap_mm, Real) or not np.isfinite(max_gap_mm) or not 2 <= max_gap_mm <= 50:
        raise ValueError("Maximum dental gap must be from 2 to 50 mm.")
    if type(control_points) is not int or not 3 <= control_points <= 64:
        raise ValueError("Choose 3 to 64 arch control points.")
    data, labels, source, producer, revision = _input(volume, segmentation, fingerprint, cancel)
    eligible = {key: item for key, item in labels.items() if item["review"] != "rejected"}
    jaw = preferred_jaw
    if jaw == "auto":
        jaw = "lower" if any(item["kind"] == "tooth" and _tooth_side(item) == "lower" for item in eligible.values()) else "upper"
    ids = [key for key, item in eligible.items() if item["kind"] == "tooth" and _tooth_side(item) == jaw]
    instances = [key for key in ids if eligible[key].get("instance")]
    warnings = ["Automatic geometry proposal only: review and edit the arch against the original scan before use."]
    if len(instances) >= 3:
        world = _centroids(volume, data, instances, cancel)
        if len(world) < 3:
            raise ValueError("Three nonempty tooth candidates are needed; review the mask or draw the arch manually.")
        if depth_mode == "anatomy":
            center = center + normal * np.median((world - center) @ normal)
        delta = world - center
        points = np.column_stack((delta @ right, delta @ up))
        points, extra = _centroid_arch(points, max_gap_mm, cancel)
        method = "tooth-instance-centroid-geodesic"
        warnings += extra
    else:
        if not ids:
            if jaw == "upper":
                raise ValueError("No upper dental mask is available; segment/correct teeth or draw an upper arch manually.")
            ids = [key for key, item in eligible.items() if item["kind"] == "jaw" and (key == 2 or "mandib" in item["name"].casefold())]
            warnings.append("No usable tooth mask was available; this arch follows the mandibular projection and may include ramus or cortical contours.")
            method = "mandibular-projection-ridge"
        else:
            method = "grouped-tooth-projection-ridge"
        world = _selected_cloud(volume, data, ids, cancel)
        if len(world) < 20:
            raise ValueError("No usable dental or mandibular mask is available for automatic arch setup.")
        if depth_mode == "anatomy":
            center = center + normal * np.median((world - center) @ normal)
        delta = world - center
        points, extra = _projection_arch(np.column_stack((delta @ right, delta @ up)), cancel)
        warnings += extra
    points = _uniform_controls(points, control_points)
    world = center + points[:, :1] * right + points[:, 1:] * up
    index = volume.world_to_index(world)
    inside = np.all((index >= -1e-6) & (index <= np.asarray(volume.dimensions) - 1 + 1e-6), axis=1)
    if not inside.all():
        runs = np.split(np.arange(len(world)), np.flatnonzero(np.diff(inside)) + 1)
        runs = [run for run in runs if inside[run[0]]]
        if not runs or len(max(runs, key=len)) < 3:
            raise ValueError("The current corrected plane does not contain enough source coverage; move the locator or draw the arch manually.")
        world = world[max(runs, key=len)]
        warnings.append("The corrected plane clips the proposed arch; only its contiguous in-volume segment is retained.")
    if (world[-1] - world[0]) @ right < 0:
        world = world[::-1]
    _check_revision(segmentation, revision, cancel)
    provenance = {"name": "AsterView automatic dental setup", "version": "1", "method": method,
                  "source_producer": producer, "source_label_ids": ids, "jaw": jaw,
                  "plane_center_lps_mm": center.tolist(), "right_lps": right.tolist(), "up_lps": up.tolist(),
                  "depth_mode": depth_mode, "requested_plane_center_lps_mm": requested_center.tolist()}
    return DentalArchProposal(world.copy(), source, provenance, warnings, source_revision=revision)


propose_panoramic_arch = propose_arch


def _bilateral_symmetry(points, midline):
    """Mask-shape evidence only; never infer a midline from image origin."""
    xy = np.rint(np.column_stack(((points[:, 0] - midline) / 2, points[:, 1] / 2))).astype(np.int64)
    occupied = set(map(tuple, xy))
    reflected = {(-x, y) for x, y in occupied}
    return len(occupied & reflected) / max(1, len(occupied | reflected))


def _condyle_candidate(volume, cloud, side, midline, cancel):
    _check_cancel(cancel)
    span = np.ptp(cloud[:, 0])
    lateral = max(3., .08 * span)
    part = cloud[cloud[:, 0] > midline + lateral] if side == "left" else cloud[cloud[:, 0] < midline - lateral]
    if len(part) < 120 or np.ptp(part[:, 1]) < 18 or np.ptp(part[:, 2]) < 22:
        return None, f"No sufficient {side} mandibular ramus evidence was present; mark that condyle manually."
    # Condylar heads lie at the superior posterior mandibular branches. This
    # rejects the more anterior coronoid peaks instead of choosing any top voxel.
    posterior = part[part[:, 1] >= np.quantile(part[:, 1], .72)]
    if len(posterior) < 80 or np.ptp(posterior[:, 2]) < 20:
        return None, f"The {side} posterior ramus is incomplete or ambiguous; mark it manually."
    top = float(np.quantile(posterior[:, 2], .995))
    head = posterior[posterior[:, 2] >= top - 7.]
    if len(head) < 40 or top - np.median(cloud[:, 2]) < 12:
        return None, f"No distinct superior posterior {side} condylar candidate was supported."
    # A head cut by the acquisition boundary cannot supply a trustworthy peak.
    index = volume.world_to_index(head)
    last = np.asarray(volume.dimensions) - 1
    if np.any(index.min(axis=0) < .75) or np.any(last - index.max(axis=0) < .75):
        return None, f"The {side} condylar candidate touches the source boundary; truncated anatomy requires manual setup."
    center = np.mean(head, axis=0)
    # The in-plane long-axis estimate uses actual head occupancy, not a fixed
    # bilateral angle. An almost round head receives an explicit ambiguity note.
    local = head[np.linalg.norm(head - center, axis=1) <= 12]
    covariance = np.cov((local - np.mean(local, axis=0))[:, :2], rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    axis = np.r_[vectors[:, -1], 0.]
    if axis[0] < 0:
        axis = -axis
    axis /= np.linalg.norm(axis)
    warning = None
    if values[-1] < 1e-6 or values[-1] / max(values[0], 1e-6) < 1.25:
        warning = f"The {side} head has an ambiguous projected long axis; correct its orientation manually."
    return (center, axis), warning


def propose_tmj(volume, segmentation, fingerprint, *, midline_lps_x=None, cancel=None):
    """Propose editable condylar centers/long axes from mandibular occupancy.

    A clinician-confirmed LPS x midline may be supplied. Otherwise an inferred
    mandibular-shape midline is admitted only with bilateral U-shape symmetry
    and sufficient physical extent. Cropped/unilateral evidence is refused,
    rather than creating a contralateral condyle at a mirrored/template point.
    LPS posterior/superior are anatomical priors; severe head tilt or deformity
    can invalidate them and always requires user review.
    """
    if midline_lps_x is not None and (isinstance(midline_lps_x, (bool, np.bool_))
                                     or not isinstance(midline_lps_x, Real)
                                     or not np.isfinite(midline_lps_x)
                                     or abs(midline_lps_x) > 1_000_000):
        raise ValueError("A confirmed anatomical midline must be a finite LPS x coordinate.")
    data, labels, source, producer, revision = _input(volume, segmentation, fingerprint, cancel)
    ids = [key for key, item in labels.items() if item["review"] != "rejected" and item["kind"] == "jaw"
           and (key == 2 or "mandib" in item["name"].casefold())]
    cloud = _selected_cloud(volume, data, ids, cancel)
    if len(cloud) < 300:
        raise ValueError("A usable mandible mask is required before automatic TMJ setup.")
    warnings = ["Provisional mask-derived TMJ setup: verify both condyles and long axes in serial source sections.",
                "Posterior/superior LPS priors may be unreliable with severe head tilt, deformity or segmentation errors."]
    inferred = midline_lps_x is None
    midline = float((np.quantile(cloud[:, 0], .05) + np.quantile(cloud[:, 0], .95)) / 2) if inferred else float(midline_lps_x)
    symmetry = _bilateral_symmetry(cloud, midline)
    eligible = True
    if inferred:
        warnings.append("Anatomical midline is inferred provisionally from bilateral mandibular shape, not scanner zero or field-of-view center; confirm laterality.")
        if np.any(np.ptp(cloud, axis=0) < [40., 35., 30.]) or symmetry < .40:
            eligible = False
            warnings.append("Bilateral full-jaw support is insufficient or asymmetric; no automatic condyles were invented. Supply a confirmed midline or mark centers manually.")
    centers, axes = {}, {}
    if eligible:
        for side in ("right", "left"):
            candidate, warning = _condyle_candidate(volume, cloud, side, midline, cancel)
            if warning:
                warnings.append(warning)
            if candidate is not None:
                centers[side], axes[side] = candidate
        if inferred and len(centers) != 2:
            centers, axes = {}, {}
            warnings.append("The unconfirmed bilateral setup was withheld because both supported condyles were not found; use manual setup or confirm the midline.")
    _check_revision(segmentation, revision, cancel)
    provenance = {"name": "AsterView automatic dental setup", "version": "1",
                  "method": "mandible-superoposterior-head-PCA", "source_producer": producer,
                  "source_label_ids": ids, "midline_method": "bilateral-mask-shape-provisional" if inferred else "clinician-supplied-LPS-x",
                  "midline_lps_x": midline, "projected_symmetry_iou": symmetry}
    return TMJProposal(centers, axes, source, provenance, warnings, source_revision=revision, midline_lps_x=midline)


propose_tmj_setup = propose_tmj
