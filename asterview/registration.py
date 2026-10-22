"""Source-bound rigid registration in DICOM patient LPS millimeters.

Public transforms always map moving points to fixed points:
``p_fixed = R @ p_moving + t``. Resampling uses the inverse, because each fixed
display location must look up its corresponding moving image intensity.
No registration result is automatically accepted or a change diagnosis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
from numbers import Real

import numpy as np

from .session import _fingerprint, _match_volume, _keys, _string, IDENTITY_KEYS

FORMAT = "asterview.registration.v1"
MAX_SOURCE_VOXELS = 400_000_000
MAX_JSON_BYTES = 64 * 1024
REVIEW_STATES = {"pending", "accepted", "rejected"}


class RegistrationCancelled(RuntimeError):
    pass


def validate_transform(matrix):
    try:
        value = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("A rigid transform must contain numeric matrix values.") from None
    if value.shape != (4, 4) or not np.isfinite(value).all() or np.max(np.abs(value)) > 1_000_000:
        raise ValueError("A rigid transform must be a bounded finite 4 by 4 matrix.")
    rotation = value[:3, :3]
    if not np.allclose(value[3], [0, 0, 0, 1], rtol=0, atol=1e-9) or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-6):
        raise ValueError("Registration requires a proper rigid transform, without reflection, scale or shear.")
    return value.copy()


def transform_points(points, matrix):
    transform = validate_transform(matrix)
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("Points must have a final dimension of three finite LPS coordinates.")
    return points @ transform[:3, :3].T + transform[:3, 3]


def rigid_matrix(translation_mm=(0, 0, 0), rotation_deg=(0, 0, 0), center_lps=(0, 0, 0)):
    """Manual active LPS transform, Euler Rz @ Ry @ Rx about a physical pivot.

    Positive translations point left/posterior/superior. Angles are degrees
    about those patient axes. Translation follows rotation about the pivot.
    """
    values = [np.asarray(x, dtype=np.float64) for x in (translation_mm, rotation_deg, center_lps)]
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in values):
        raise ValueError("Rigid parameters must be three finite translations, angles and pivot coordinates.")
    translation, angles, center = values
    if np.any(np.abs(angles) > 360):
        raise ValueError("Manual angles must be between -360 and 360 degrees.")
    x, y, z = np.deg2rad(angles)
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    result = np.eye(4)
    result[:3, :3] = rz @ ry @ rx
    result[:3, 3] = center + translation - result[:3, :3] @ center
    return validate_transform(result)


def refine_transform(base, translation_mm=(0, 0, 0), rotation_deg=(0, 0, 0), center_lps=(0, 0, 0)):
    """Apply a manual correction in the fixed coordinate frame: delta @ base."""
    return validate_transform(rigid_matrix(translation_mm, rotation_deg, center_lps) @ validate_transform(base))


def _source(volume, fingerprint):
    if not volume.metadata.get("geometry_calibrated", True) or min(volume.dimensions) < 2:
        raise ValueError("Registration requires calibrated three-dimensional source geometry.")
    if volume.metadata.get("nonuniform", False) or volume.metadata.get("nonuniform_spacing", False) or volume.metadata.get("regular_grid", True) is False:
        raise ValueError("Nonuniform slices require explicit source resampling before registration.")
    if volume.data.size > MAX_SOURCE_VOXELS:
        raise ValueError("This source exceeds the bounded registration size (400 million voxels).")
    source = _fingerprint(fingerprint, "Registration source")
    actual = {"shape": list(volume.data.shape), "spacing": list(volume.spacing), "origin": volume.origin.tolist(),
              "direction": volume.direction.tolist(), "synthetic": bool(volume.metadata.get("synthetic", False))}
    for key in IDENTITY_KEYS:
        if not actual["synthetic"] or key in volume.metadata:
            actual[key] = str(volume.metadata.get(key, ""))
    _match_volume(source, _fingerprint(actual, "Loaded registration volume"))
    return source


@dataclass
class RegistrationResult:
    transform: np.ndarray
    fixed_source: dict
    moving_source: dict
    metric: float | None = None
    method: str = "Manual rigid alignment"
    warnings: list[str] = field(default_factory=list)
    review: str = "pending"

    def __post_init__(self):
        self.transform = validate_transform(self.transform)
        self.fixed_source = _fingerprint(self.fixed_source, "Fixed registration source")
        self.moving_source = _fingerprint(self.moving_source, "Moving registration source")
        if self.metric is not None and (isinstance(self.metric, bool) or not isinstance(self.metric, Real) or not np.isfinite(self.metric)):
            raise ValueError("The registration metric must be finite or absent.")
        self.metric = None if self.metric is None else float(self.metric)
        self.method = _string(self.method, "Registration method", 1024)
        if not isinstance(self.warnings, list) or len(self.warnings) > 32:
            raise ValueError("Registration warnings must be a bounded list.")
        self.warnings = [_string(warning, "Registration warning", 512) for warning in self.warnings]
        if not isinstance(self.review, str) or self.review not in REVIEW_STATES:
            raise ValueError("Registration review must be pending, accepted or rejected.")

    @property
    def moving_to_fixed(self):
        return self.transform.copy()

    def validate_source(self, fixed, moving, fixed_fingerprint, moving_fingerprint):
        _match_volume(self.fixed_source, _source(fixed, fixed_fingerprint))
        _match_volume(self.moving_source, _source(moving, moving_fingerprint))
        validate_transform(self.transform)

    def refined(self, translation_mm=(0, 0, 0), rotation_deg=(0, 0, 0), center_lps=(0, 0, 0)):
        return RegistrationResult(refine_transform(self.transform, translation_mm, rotation_deg, center_lps),
                                  self.fixed_source, self.moving_source, self.metric,
                                  self.method + "; manual correction in fixed LPS", self.warnings.copy(), "pending")


def save_registration(path, result, fixed, moving, fixed_fp, moving_fp):
    result.validate_source(fixed, moving, fixed_fp, moving_fp)
    # Revalidate mutable result metadata completely before serialization.
    checked = RegistrationResult(result.transform, result.fixed_source, result.moving_source, result.metric, result.method, result.warnings, result.review)
    payload = {"format": FORMAT, "moving_to_fixed": checked.transform.tolist(), "fixed_source": checked.fixed_source,
               "moving_source": checked.moving_source, "metric": checked.metric, "method": checked.method, "warnings": checked.warnings, "review": checked.review}
    text = json.dumps(payload, indent=2, allow_nan=False)
    if len(text.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError("The registration file exceeds its bounded size.")
    Path(path).write_text(text, encoding="utf-8")


def load_registration(path, fixed, moving, fixed_fp, moving_fp):
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_JSON_BYTES + 1)
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("Registration files must not exceed 64 KiB.")
    payload = json.loads(data.decode("utf-8-sig"))
    _keys(payload, {"format", "moving_to_fixed", "fixed_source", "moving_source", "metric", "method", "warnings", "review"}, "Registration file")
    if payload["format"] != FORMAT or not isinstance(payload["review"], str) or payload["review"] not in REVIEW_STATES:
        raise ValueError("Unsupported registration file or review state.")
    result = RegistrationResult(payload["moving_to_fixed"], payload["fixed_source"], payload["moving_source"], payload["metric"], payload["method"], payload["warnings"], "pending")
    result.validate_source(fixed, moving, fixed_fp, moving_fp)
    return result


def sample_registered_plane(moving, transform, frame, shape, *, slab_mm=0, mode="mean", cancel=None, return_valid=False, sampler=None):
    """Sample moving scalars on a fixed-LPS plane using the inverse transform.

    ``frame`` is (center, right, up, (width_mm,height_mm)). ``return_valid=True``
    also returns full-support overlap, excluding outside FOV and padded samples.
    Compare/display differences only where both source planes are valid.
    """
    from .reformat import sample_plane, _frame, _slab_offsets
    from scipy.ndimage import map_coordinates
    inverse = np.linalg.inv(validate_transform(transform))
    if len(frame) != 4 or len(shape) != 2 or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or x < 1 for x in shape) or np.prod(shape) > 4_194_304:
        raise ValueError("A bounded fixed plane frame and positive output shape are required.")
    center, right, up, size = frame
    right, up, normal = _frame(right, up)
    mapped_center = transform_points(center, inverse)
    mapped_right, mapped_up = inverse[:3, :3] @ right, inverse[:3, :3] @ up
    if sampler is not None:
        if sampler.volume is not moving:
            raise ValueError("The plane sampler must belong to the registered source volume.")
        image = sampler.sample_plane(mapped_center, mapped_right, mapped_up, size, shape, slab_mm, mode, cancel)
    else:
        image = sample_plane(moving, mapped_center, mapped_right, mapped_up, size, shape, slab_mm, mode, cancel)
    if not return_valid:
        return image
    rows, cols = [int(v) for v in shape]
    x = np.zeros(1) if cols == 1 else np.linspace(-size[0] / 2, size[0] / 2, cols)
    y = np.zeros(1) if rows == 1 else -np.linspace(-size[1] / 2, size[1] / 2, rows)
    mapped_normal = inverse[:3, :3] @ normal
    # Evaluate the affine pixel grid one coordinate at a time. Constructing
    # full LPS and index vector grids for each of six panes wastes both memory
    # bandwidth and several plane-sized 3-vector allocations on every wheel step.
    center_index = moving.world_to_index(mapped_center)
    right_index = (moving.direction.T @ mapped_right) / moving.spacing
    up_index = (moving.direction.T @ mapped_up) / moving.spacing
    normal_index = (moving.direction.T @ mapped_normal) / moving.spacing
    valid = np.ones((rows, cols), dtype=bool)
    for offset in _slab_offsets(slab_mm, np.asarray(moving.spacing)):
        if cancel is not None and cancel():
            raise RegistrationCancelled("Registered plane sampling cancelled.")
        coordinates = [] if moving.valid_mask is not None else None
        for axis, count in enumerate(moving.dimensions):
            values = center_index[axis] + offset * normal_index[axis]
            if abs(right_index[axis]) > 1e-14:
                values = values + x[None, :] * right_index[axis]
            if abs(up_index[axis]) > 1e-14:
                values = values + y[:, None] * up_index[axis]
            valid &= (values >= -1e-7) & (values <= count - 1 + 1e-7)
            if coordinates is not None:
                coordinates.append(np.broadcast_to(values, (rows, cols)))
        if moving.valid_mask is not None:
            support = map_coordinates(moving.valid_mask, np.asarray(coordinates[::-1]), order=1, mode="constant", cval=0, prefilter=False, output=np.float32)
            valid &= support > .999
    return image, valid


def detect_registration_status():
    project = Path(__file__).resolve().parents[1]
    for work in [project / "work"] + [p / "work" for p in project.parents]:
        for name in ("registration-venv", "ai-venv"):
            runtime = work / name
            python = runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            sites = [runtime / "Lib/site-packages"] + list((runtime / "lib").glob("python*/site-packages"))
            if python.is_file() and any((site / "SimpleITK/__init__.py").is_file() for site in sites):
                return {"ready": True, "python": str(python), "work": str(work), "reason": "Local CPU registration is ready"}
    if importlib.util.find_spec("SimpleITK") is not None:
        return {"ready": True, "python": sys.executable, "work": str(project / "work"), "reason": "Local CPU registration is ready"}
    return {"ready": False, "reason": "Run Setup Registration.ps1 to install the lightweight CPU registration runtime."}


def _cancelled(cancel):
    return False if cancel is None else bool(cancel() if callable(cancel) else cancel.is_set())


def _run_worker(status, job, report, cancel):
    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "1", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1"})
    command = [status["python"], "-u", str(Path(__file__).with_name("_registration_worker.py")), "--job", str(job)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    messages = queue.Queue()
    def read_output():
        for line in process.stdout:
            messages.put(line.rstrip())
    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    tail = []
    try:
        while True:
            if _cancelled(cancel):
                raise RegistrationCancelled("Volume registration cancelled.")
            try:
                line = messages.get(timeout=.1)
                tail = (tail + [line])[-16:]
                if line.startswith("ASTER_PROGRESS:"):
                    report(line.partition(":")[2].strip())
            except queue.Empty:
                pass
            if process.poll() is not None and not reader.is_alive() and messages.empty():
                break
        if process.returncode:
            raise RuntimeError("Rigid registration failed. " + "\n".join(tail)[-1600:])
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=15)
        reader.join(timeout=3)
        process.stdout.close()


def register_volumes(fixed, moving, fixed_fp, moving_fp, progress=None, cancel=None, *, initialization="centers", initial_transform=None, bone_focus=False):
    """Estimate local six-DOF rigid alignment; never accepts the result.

    Source arrays stay immutable. Initial/manual transforms map moving→fixed.
    Use initialization='geometry' to preserve physical geometry initially, or
    'centers' to align acquired image centers when scan frames differ.
    """
    fixed_source, moving_source = _source(fixed, fixed_fp), _source(moving, moving_fp)
    if min(*fixed.dimensions, *moving.dimensions) < 8:
        raise ValueError("Automatic 3D registration needs at least eight voxels along every source axis; use manual alignment for thinner sources.")
    if any(volume.data_max - volume.data_min < 1e-5 for volume in (fixed, moving)):
        raise ValueError("Both sources need intensity variation for automatic registration.")
    if initialization not in {"centers", "geometry"} or type(bone_focus) is not bool:
        raise ValueError("Choose centers/geometry initialization and a boolean bone-focus setting.")
    initial = None if initial_transform is None else validate_transform(initial_transform).tolist()
    if _cancelled(cancel):
        raise RegistrationCancelled("Volume registration cancelled.")
    status = detect_registration_status()
    if not status["ready"]:
        raise RuntimeError(status["reason"])
    report = progress or (lambda _: None)
    report("Preparing two local registration sources…")
    jobs = Path(status["work"]) / "registration/jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aster-register-", dir=jobs) as folder:
        job = Path(folder)
        config = {"fixed": fixed_source, "moving": moving_source, "initialization": initialization,
                  "initial_moving_to_fixed": initial, "bone_focus": bone_focus, "seed": 20261020}
        for name, volume in (("fixed", fixed), ("moving", moving)):
            if _cancelled(cancel):
                raise RegistrationCancelled("Volume registration cancelled.")
            np.save(job / (name + ".npy"), volume.data, allow_pickle=False)
            if volume.valid_mask is not None:
                np.save(job / (name + "-valid.npy"), volume.valid_mask, allow_pickle=False)
            config[name + "_is_hu"] = bool(volume.is_hu)
            config[name + "_data_min"] = volume.data_min
        (job / "config.json").write_text(json.dumps(config, allow_nan=False), encoding="utf-8")
        _run_worker(status, job, report, cancel)
        if _cancelled(cancel):
            raise RegistrationCancelled("Volume registration cancelled.")
        _match_volume(fixed_source, _source(fixed, fixed_fp))
        _match_volume(moving_source, _source(moving, moving_fp))
        output = json.loads((job / "result.json").read_text(encoding="utf-8"))
        return RegistrationResult(output["moving_to_fixed"], fixed_source, moving_source, output["metric"], output["method"], output["warnings"], "pending")
