"""Conservative, read-only DICOM discovery and regular grayscale volume loading.

Geometry follows DICOM PS3.3 C.7.6.2 / C.7.6.16. Array x is the first
ImageOrientationPatient triplet; PixelSpacing is row, column, in that order.
No instance-number ordering, invented 3D spacing, or cross-series merging is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from pydicom.pixels import apply_modality_lut

from .model import Volume

Progress = Callable[[int, str], None]
Cancel = Callable[[], bool]
MAX_WORKING_BYTES = 8 * 1024**3
MAX_GRID_RESIDUAL_MM = 0.01
_FRAME_TYPE_GROUPS = ("CTImageFrameTypeSequence", "MRImageFrameTypeSequence", "PETFrameTypeSequence")
_PADDING_PAIRS = (("PixelPaddingValue", "PixelPaddingRangeLimit"),
                  ("FloatPixelPaddingValue", "FloatPixelPaddingRangeLimit"),
                  ("DoubleFloatPixelPaddingValue", "DoubleFloatPixelPaddingRangeLimit"))


class DicomLoadError(ValueError):
    """Unsupported or unsafe-to-assemble input, with a non-identifying message."""


class LoadingCancelled(DicomLoadError):
    pass


@dataclass
class SeriesInfo:
    uid: str
    label: str
    paths: list[Path]
    modality: str
    count: int


@dataclass
class _Frame:
    path: Path
    index: int
    dataset: Dataset
    position: np.ndarray | None
    orientation: np.ndarray | None
    pixel_spacing: np.ndarray | None
    transform: Dataset
    thickness: float | None
    phase_key: tuple


def _check_cancel(cancel: Cancel | None) -> None:
    if cancel is not None and cancel():
        raise LoadingCancelled("Loading canceled.")


def _progress(callback: Progress | None, percent: int, message: str) -> None:
    if callback is not None:
        callback(int(percent), message)


def _numbers(value, length: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=np.float64).reshape(length)
    except (TypeError, ValueError):
        raise DicomLoadError("Invalid numeric DICOM geometry.") from None
    if not np.isfinite(result).all():
        raise DicomLoadError("DICOM geometry contains nonfinite values.")
    return result


def _scalar(value, default=None):
    if value is None:
        return default
    try:
        if not isinstance(value, (str, bytes)) and hasattr(value, "__len__"):
            value = value[0]
        result = float(value)
    except (ValueError, TypeError, IndexError):
        raise DicomLoadError("Invalid numeric DICOM value.") from None
    if not np.isfinite(result):
        raise DicomLoadError("DICOM contains a nonfinite numeric value.")
    return result


def _seq_item(dataset, name: str) -> Dataset | None:
    sequence = getattr(dataset, name, None) if dataset is not None else None
    if not sequence:
        return None
    if len(sequence) != 1:
        raise DicomLoadError("A functional-group sequence must have exactly one item.")
    return sequence[0]


def _functional(frame, shared, name: str) -> Dataset | None:
    return _seq_item(frame, name) or _seq_item(shared, name)


def _header(path: Path) -> Dataset | None:
    try:
        dataset = pydicom.dcmread(str(path), stop_before_pixels=True)
    except InvalidDicomError:
        # Valid older non-Part-10 DICOM may lack a preamble. Require essential
        # image identity fields so arbitrary extensionless files are not accepted.
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            return None
    except (OSError, ValueError, EOFError):
        return None
    if not all(hasattr(dataset, key) for key in ("SOPClassUID", "SOPInstanceUID", "Rows", "Columns")):
        return None
    if not str(getattr(dataset, "SOPClassUID", "")).startswith("1.2.840."):
        return None
    return dataset


def _group_key(dataset: Dataset) -> tuple:
    # Orientation/shape grouping keeps scouts and separate echoes/phases visible
    # as distinct choices even when a vendor reuses its SeriesInstanceUID.
    orient = _numbers(getattr(dataset, "ImageOrientationPatient", None), 6)
    spacing = _numbers(getattr(dataset, "PixelSpacing", None), 2)
    frame_of_reference = str(getattr(dataset, "FrameOfReferenceUID", ""))
    return (
        str(getattr(dataset, "StudyInstanceUID", "")),
        str(getattr(dataset, "SeriesInstanceUID", getattr(dataset, "SOPInstanceUID", ""))),
        str(dataset.SOPClassUID), int(dataset.Rows), int(dataset.Columns),
        tuple(np.round(orient, 4)) if orient is not None else None,
        tuple(np.round(spacing, 4)) if spacing is not None else None,
        frame_of_reference, tuple(str(v) for v in getattr(dataset, "ImageType", [])),
        str(getattr(dataset, "TemporalPositionIdentifier", "")),
        str(getattr(dataset, "EchoNumbers", "")),
        str(getattr(dataset, "EchoTime", "")),
        str(getattr(dataset, "AcquisitionNumber", "")),
        # Multiframe objects remain independent; never concatenate temporal runs.
        str(dataset.SOPInstanceUID) if int(getattr(dataset, "NumberOfFrames", 1)) > 1 else "",
    )


def _instance_phase_key(dataset: Dataset) -> tuple:
    """Nonspatial acquisition dimensions that must not be silently stacked."""
    return (
        str(getattr(dataset, "StudyInstanceUID", "")),
        str(getattr(dataset, "PatientID", "")),
        tuple(str(v) for v in getattr(dataset, "ImageType", [])),
        str(getattr(dataset, "TemporalPositionIdentifier", "")),
        str(getattr(dataset, "EchoNumbers", "")),
        str(getattr(dataset, "EchoTime", "")),
        str(getattr(dataset, "AcquisitionNumber", "")),
    )


def discover_series(folder: str | Path, progress: Progress | None = None,
                    cancel: Cancel | None = None) -> list[SeriesInfo]:
    """Recursively discover image objects, including extensionless DICOM.

    Progress callbacks receive (percent, generic_message). No patient names,
    paths, or header values are written to logs or progress callbacks.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise DicomLoadError("Choose an existing folder containing DICOM images.")
    _progress(progress, 0, "Finding DICOM files…")
    try:
        paths = sorted((p for p in folder.rglob("*") if p.is_file()), key=lambda p: str(p).casefold())
    except OSError:
        raise DicomLoadError("The selected folder could not be read.") from None
    groups: dict[tuple, tuple[Dataset, list[Path], int]] = {}
    for i, path in enumerate(paths):
        _check_cancel(cancel)
        dataset = _header(path)
        if dataset is not None:
            try:
                key = _group_key(dataset)
                count = int(getattr(dataset, "NumberOfFrames", 1))
                if count < 1:
                    continue
                if key not in groups:
                    groups[key] = (dataset, [], 0)
                first, members, previous = groups[key]
                members.append(path)
                groups[key] = (first, members, previous + count)
            except (DicomLoadError, ValueError, TypeError):
                # Malformed geometry remains a visible choice; load gives the
                # actionable error rather than silently fabricating geometry.
                key = ("malformed", str(path))
                groups[key] = (dataset, [path], 1)
        if i % 20 == 0:
            _progress(progress, round(100 * (i + 1) / max(1, len(paths))), "Reading DICOM headers…")
    result = []
    for key, (dataset, members, count) in groups.items():
        modality = str(getattr(dataset, "Modality", "DICOM"))
        description = str(getattr(dataset, "SeriesDescription", "Grayscale image series")).strip()[:100]
        number = str(getattr(dataset, "SeriesNumber", ""))
        suffix = sha256(repr(key).encode()).hexdigest()[:12]
        uid = f"{getattr(dataset, 'SeriesInstanceUID', 'unidentified')}:{suffix}"
        label = f"{modality} · {description} · {count} image{'s' if count != 1 else ''}"
        if number:
            label = f"{number} · {label}"
        local_identity = [str(getattr(dataset, "PatientID", "")).strip(),
                          str(getattr(dataset, "StudyDate", "")).strip()]
        local_identity = [value for value in local_identity if value]
        if local_identity:
            label = " · ".join(local_identity) + "  |  " + label
        result.append(SeriesInfo(uid, label, members, modality, count))
    result.sort(key=lambda info: (info.modality, info.label, info.uid))
    _progress(progress, 100, f"Found {len(result)} image series.")
    return result


def _validate_image(dataset: Dataset) -> None:
    modality = str(getattr(dataset, "Modality", ""))
    if modality in {"SEG", "RTDOSE", "RTSTRUCT", "RTPLAN", "SR", "PR", "KO"}:
        raise DicomLoadError("This DICOM object type is not a grayscale image volume supported by this build.")
    photometric = str(getattr(dataset, "PhotometricInterpretation", ""))
    if int(getattr(dataset, "SamplesPerPixel", 1)) != 1 or photometric not in {"MONOCHROME1", "MONOCHROME2"}:
        raise DicomLoadError("Only single-channel MONOCHROME1/MONOCHROME2 DICOM images are supported in this build.")
    if int(getattr(dataset, "Rows", 0)) <= 0 or int(getattr(dataset, "Columns", 0)) <= 0:
        raise DicomLoadError("The DICOM image dimensions are invalid.")
    if getattr(dataset, "AnatomicalOrientationType", "BIPED") == "QUADRUPED":
        raise DicomLoadError("Quadruped patient-coordinate geometry is not supported in this build.")
    if "LOCALIZER" in [str(v).upper() for v in getattr(dataset, "ImageType", [])] and int(getattr(dataset, "NumberOfFrames", 1)) > 1:
        raise DicomLoadError("Multiframe localizer images cannot be assembled as a 3D volume.")


def _phase(frame: Dataset | None, shared: Dataset | None = None) -> tuple:
    content = _seq_item(frame, "FrameContentSequence")
    mr_echo = _seq_item(frame, "MREchoSequence")
    cardiac = _seq_item(frame, "CardiacSynchronizationSequence")
    diffusion = _functional(frame, shared, "MRDiffusionSequence")
    gradient = _seq_item(diffusion, "DiffusionGradientDirectionSequence")
    frame_type = _functional(frame, shared, "CTImageFrameTypeSequence") or _functional(frame, shared, "MRImageFrameTypeSequence")
    return (
        str(getattr(content, "StackID", "")),
        str(getattr(content, "TemporalPositionIndex", "")),
        str(getattr(mr_echo, "EffectiveEchoTime", "")),
        str(getattr(cardiac, "NominalPercentageOfCardiacPhase", "")),
        str(getattr(diffusion, "DiffusionBValue", "")),
        str(getattr(gradient, "DiffusionGradientOrientation", "")),
        tuple(str(v) for v in getattr(frame_type, "FrameType", [])),
    )


def _frames(path: Path, dataset: Dataset) -> list[_Frame]:
    _validate_image(dataset)
    count = int(getattr(dataset, "NumberOfFrames", 1))
    if count < 1:
        raise DicomLoadError("The DICOM frame count is invalid.")
    shared = _seq_item(dataset, "SharedFunctionalGroupsSequence")
    per_frames = getattr(dataset, "PerFrameFunctionalGroupsSequence", None)
    if per_frames is not None and len(per_frames) != count:
        raise DicomLoadError("Per-frame geometry count does not match NumberOfFrames.")
    if count > 1 and per_frames is None:
        raise DicomLoadError("This multiframe object has no explicit per-frame patient positions. Cine and legacy multiframe objects cannot be treated as 3D volumes.")
    dimensions = getattr(dataset, "DimensionIndexSequence", [])
    # Enhanced objects may contain time, diffusion, energy, or echo dimensions
    # even when their frame positions happen not to overlap. Only a spatial
    # index is permitted to vary in a single scalar volume.
    for axis, dimension in enumerate(dimensions):
        pointer = int(getattr(dimension, "DimensionIndexPointer", 0))
        if pointer in {0x00209057, 0x00200032}:  # InStackPositionNumber / ImagePositionPatient
            continue
        values = set()
        for frame in per_frames or []:
            content = _seq_item(frame, "FrameContentSequence")
            indices = getattr(content, "DimensionIndexValues", [])
            if isinstance(indices, (int, np.integer)):
                indices = [indices]
            if len(indices) != len(dimensions):
                raise DicomLoadError("Enhanced dimension index values are missing or inconsistent.")
            values.add(int(indices[axis]))
        if len(values) > 1:
            raise DicomLoadError("The enhanced object varies along a nonspatial dimension. Select a single phase, echo, energy, or diffusion acquisition.")
    result = []
    for i in range(count):
        frame = per_frames[i] if per_frames is not None else None
        orientation_group = _functional(frame, shared, "PlaneOrientationSequence")
        position_group = _functional(frame, shared, "PlanePositionSequence")
        measure_group = _functional(frame, shared, "PixelMeasuresSequence")
        transform = _functional(frame, shared, "PixelValueTransformationSequence")
        position = getattr(position_group, "ImagePositionPatient", None) if position_group is not None else getattr(dataset, "ImagePositionPatient", None)
        orientation = getattr(orientation_group, "ImageOrientationPatient", None) if orientation_group is not None else getattr(dataset, "ImageOrientationPatient", None)
        pixel_spacing = getattr(measure_group, "PixelSpacing", None) if measure_group is not None else getattr(dataset, "PixelSpacing", None)
        thickness = getattr(measure_group, "SliceThickness", None) if measure_group is not None else getattr(dataset, "SliceThickness", None)
        # Repeating a root-level position for every multiframe image is not
        # sufficient geometry, even if a spacing/thickness happens to be present.
        if count > 1 and (position_group is None or position is None):
            raise DicomLoadError("Every multiframe slice requires an explicit patient position.")
        result.append(_Frame(path, i, dataset, _numbers(position, 3), _numbers(orientation, 6),
                             _numbers(pixel_spacing, 2), transform if transform is not None else dataset,
                             _scalar(thickness), _phase(frame, shared)))
    return result


def _frame_type_group(group: Dataset | None) -> Dataset | None:
    for name in _FRAME_TYPE_GROUPS:
        item = _seq_item(group, name)
        if item is not None:
            return item
    return None


def _validate_volumetric_frames(frames: list[_Frame]) -> None:
    """Patient positions do not make projections or distorted images a volume."""
    if len(frames) <= 1:
        return
    allowed_techniques = {"", "NONE", "MPR"}
    projection_tokens = {"MAX_IP", "MIN_IP", "CURVED_MPR", "VOLUME_RENDER", "SURFACE_RENDER", "MIP", "MINIP"}
    for frame in frames:
        dataset = frame.dataset
        shared = _seq_item(dataset, "SharedFunctionalGroupsSequence")
        per_frames = getattr(dataset, "PerFrameFunctionalGroupsSequence", None)
        per_frame = per_frames[frame.index] if per_frames is not None else None
        local_type, shared_type = _frame_type_group(per_frame), _frame_type_group(shared)
        root_property = str(getattr(dataset, "VolumetricProperties", "")).upper().strip()
        local_property = str(getattr(local_type, "VolumetricProperties", "")).upper().strip()
        shared_property = str(getattr(shared_type, "VolumetricProperties", "")).upper().strip()
        if root_property not in {"", "VOLUME", "MIXED"} or any(p not in {"", "VOLUME"} for p in (local_property, shared_property)):
            raise DicomLoadError("SAMPLED, DISTORTED, or unsupported volumetric properties cannot be reconstructed as a 3D anatomical volume.")
        if root_property == "MIXED" and local_property != "VOLUME":
            raise DicomLoadError("MIXED volumetric properties require every frame to explicitly declare VOLUME before 3D reconstruction.")
        root_technique = str(getattr(dataset, "VolumeBasedCalculationTechnique", "")).upper().strip()
        local_technique = str(getattr(local_type, "VolumeBasedCalculationTechnique", "")).upper().strip()
        shared_technique = str(getattr(shared_type, "VolumeBasedCalculationTechnique", "")).upper().strip()
        if root_technique not in allowed_techniques | {"MIXED"} or any(t not in allowed_techniques for t in (local_technique, shared_technique)):
            raise DicomLoadError("Projection, rendered, or curved-MPR calculation techniques cannot be reconstructed as a 3D anatomical volume.")
        if root_technique == "MIXED" and local_technique not in {"NONE", "MPR"}:
            raise DicomLoadError("MIXED calculation techniques require an explicit volume-preserving technique for every frame.")
        image_tokens = {str(v).upper().strip() for v in getattr(dataset, "ImageType", [])}
        image_tokens.update(str(v).upper().strip() for v in getattr(local_type, "FrameType", []))
        image_tokens.update(str(v).upper().strip() for v in getattr(shared_type, "FrameType", []))
        if image_tokens & projection_tokens:
            raise DicomLoadError("Projection or curved-MPR image/frame types cannot be reconstructed as a 3D anatomical volume.")


def _geometry(frames: list[_Frame], warnings: list[str]):
    first = frames[0]
    if len(frames) > 1 and any(f.position is None or f.orientation is None or f.pixel_spacing is None for f in frames):
        raise DicomLoadError("3D reconstruction requires ImagePositionPatient, ImageOrientationPatient, and PixelSpacing for every slice. Missing geometry cannot be inferred safely.")
    if first.orientation is None:
        first.orientation = np.array([1., 0., 0., 0., 1., 0.])
        warnings.append("Single 2D image has no patient orientation; an uncalibrated display orientation is used.")
    if first.position is None:
        first.position = np.zeros(3)
        warnings.append("Single 2D image has no patient position; its display origin is arbitrary.")
    if first.pixel_spacing is None:
        first.pixel_spacing = np.ones(2)
        warnings.append("Single 2D image has no PixelSpacing; distances are uncalibrated and 1 display unit per pixel is used.")
    x_axis, y_axis = first.orientation[:3], first.orientation[3:]
    if not np.isclose(np.linalg.norm(x_axis), 1, atol=1e-4) or not np.isclose(np.linalg.norm(y_axis), 1, atol=1e-4) or not np.isclose(x_axis @ y_axis, 0, atol=1e-4):
        raise DicomLoadError("DICOM orientation axes are not valid orthogonal unit vectors.")
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    for frame in frames:
        if frame.pixel_spacing is None or min(frame.pixel_spacing) <= 0:
            raise DicomLoadError("PixelSpacing must contain positive row and column spacing.")
        if not np.allclose(frame.pixel_spacing, first.pixel_spacing, rtol=1e-4, atol=1e-4):
            raise DicomLoadError("In-plane pixel spacing varies between slices; resampling is required.")
        if frame.orientation is None or not np.allclose(frame.orientation, first.orientation, rtol=0, atol=1e-4):
            raise DicomLoadError("Slice orientations vary; this series cannot be represented by a regular volume.")
    projections = np.array([f.position @ z_axis for f in frames])
    order = np.argsort(projections, kind="stable")
    frames = [frames[int(i)] for i in order]
    projections = projections[order]
    if len(frames) > 1:
        differences = np.diff(projections)
        if (differences < 0.001).any():
            raise DicomLoadError("Duplicate or overlapping slice positions detected. The selection may contain multiple phases, echoes, or repeated images; select one spatial stack.")
        dz = float(np.median(differences))
        tolerance = max(0.002, 0.005 * dz)
        if np.max(np.abs(differences - dz)) > tolerance:
            raise DicomLoadError("Slice positions are nonuniform or contain gaps. A regular volume requires explicit resampling; slices were not silently joined.")
        expected_grid = projections[0] + np.arange(len(frames), dtype=np.float64) * dz
        if np.max(np.abs(projections - expected_grid)) > MAX_GRID_RESIDUAL_MM:
            raise DicomLoadError("Cumulative slice positions deviate from a regular physical grid. Geometry-preserving resampling is required; the stack was not silently regularized.")
        if np.max(np.abs(differences - dz)) > 0.0001:
            warnings.append("Small slice-coordinate rounding differences were represented by the median slice spacing.")
        base = frames[0].position
        for frame, projection in zip(frames, projections):
            residual = frame.position - base - z_axis * (projection - projections[0])
            if np.linalg.norm(residual) > max(0.05, 0.05 * min(first.pixel_spacing)):
                raise DicomLoadError("Slice positions contain gantry tilt or in-plane shifts. Geometry-preserving resampling is required before this build can display this stack.")
    else:
        dz = first.thickness if first.thickness is not None and first.thickness > 0 else 1.0
        warnings.append("Single 2D slice: through-plane reconstruction contains no additional anatomy.")
    direction = np.column_stack((x_axis, y_axis, z_axis))
    return frames, (float(first.pixel_spacing[1]), float(first.pixel_spacing[0]), dz), frames[0].position, direction


def _transform_pixels(pixels: np.ndarray, frame: _Frame) -> np.ndarray:
    transform = frame.transform
    # Explicit per-frame transformations take priority over root-level values.
    if getattr(transform, "ModalityLUTSequence", None):
        try:
            return np.asarray(apply_modality_lut(pixels, transform), dtype=np.float32)
        except Exception:
            raise DicomLoadError("The modality lookup table could not be applied.") from None
    slope = _scalar(getattr(transform, "RescaleSlope", None), 1.0)
    intercept = _scalar(getattr(transform, "RescaleIntercept", None), 0.0)
    if slope == 0:
        raise DicomLoadError("A zero RescaleSlope is invalid.")
    result = pixels.astype(np.float32)
    np.multiply(result, slope, out=result)
    np.add(result, intercept, out=result)
    return result


def _voxel_units(frame: _Frame) -> str:
    lut = _seq_item(frame.transform, "ModalityLUTSequence")
    if lut is not None:
        if hasattr(frame.transform, "RescaleSlope") or hasattr(frame.transform, "RescaleIntercept"):
            raise DicomLoadError("A modality LUT and a linear rescale transform cannot both be applied to the same image.")
        descriptor = getattr(lut, "LUTDescriptor", None)
        if descriptor is None or len(descriptor) != 3 or not hasattr(lut, "LUTData"):
            raise DicomLoadError("The modality lookup table is incomplete.")
        return str(getattr(lut, "ModalityLUTType", "")).strip().upper()
    has_slope = hasattr(frame.transform, "RescaleSlope")
    has_intercept = hasattr(frame.transform, "RescaleIntercept")
    if has_slope != has_intercept:
        raise DicomLoadError("The modality rescale transform is incomplete: both slope and intercept are required.")
    if not has_slope:
        return ""
    if _scalar(frame.transform.RescaleSlope) == 0:
        raise DicomLoadError("A zero RescaleSlope is invalid.")
    _scalar(frame.transform.RescaleIntercept)
    return str(getattr(frame.transform, "RescaleType", "")).strip().upper()


def _padding_spec(dataset: Dataset) -> tuple[float, float] | None:
    present = [(value_name, limit_name) for value_name, limit_name in _PADDING_PAIRS
               if hasattr(dataset, value_name) or hasattr(dataset, limit_name)]
    if not present:
        return None
    if len(present) != 1:
        raise DicomLoadError("Conflicting integer and floating-point pixel padding attributes are unsupported.")
    value_name, limit_name = present[0]
    if not hasattr(dataset, value_name):
        raise DicomLoadError("PixelPaddingRangeLimit requires a PixelPaddingValue.")
    try:
        low = float(getattr(dataset, value_name))
        high = float(getattr(dataset, limit_name, low))
    except (ValueError, TypeError, OverflowError):
        raise DicomLoadError("Pixel padding values are invalid.") from None
    if value_name == "PixelPaddingValue" and (not np.isfinite(low) or not np.isfinite(high)):
        raise DicomLoadError("Integer pixel padding values must be finite.")
    if np.isnan(low) or np.isnan(high):
        if not (np.isnan(low) and np.isnan(high)):
            raise DicomLoadError("A NaN padding range must specify NaN at both endpoints.")
        return np.nan, np.nan
    return min(low, high), max(low, high)


def _valid_pixels(pixels: np.ndarray, specification: tuple[float, float]) -> np.ndarray:
    low, high = specification
    if np.isnan(low):
        return ~np.isnan(pixels)
    if low == high:
        return pixels != low
    return (pixels < low) | (pixels > high)


def load_series(info: SeriesInfo, progress: Progress | None = None,
                cancel: Cancel | None = None, *, memory_budget_bytes: int = MAX_WORKING_BYTES) -> Volume:
    """Decode a single regular grayscale stack without modifying source files.

    Explicitly unsupported geometry is rejected before allocation. An estimated
    volume plus decoder working set is capped at 8 GiB by default.
    """
    if not info.paths:
        raise DicomLoadError("This selection contains no image files.")
    _progress(progress, 0, "Validating image geometry…")
    frames = []
    seen_instances: set[str] = set()
    datasets: dict[Path, Dataset] = {}
    for path_value in info.paths:
        _check_cancel(cancel)
        path = Path(path_value)
        dataset = _header(path)
        if dataset is None:
            raise DicomLoadError("An image in the selection is unreadable or is no longer available.")
        uid = str(getattr(dataset, "SOPInstanceUID", ""))
        if uid in seen_instances:
            raise DicomLoadError("Duplicate DICOM instances were selected; remove duplicate copies before loading.")
        seen_instances.add(uid)
        # Check a claimed large object before constructing its frame list.
        claimed_voxels = int(dataset.Rows) * int(dataset.Columns) * int(getattr(dataset, "NumberOfFrames", 1))
        if claimed_voxels > memory_budget_bytes // 4:
            raise DicomLoadError("The claimed image dimensions exceed the configured decode memory safety budget.")
        datasets[path] = dataset
        frames.extend(_frames(path, dataset))
    first = frames[0].dataset
    if len(datasets) > 1:
        original_series = {str(getattr(d, "SeriesInstanceUID", "")) for d in datasets.values()}
        if len(original_series) != 1 or "" in original_series:
            raise DicomLoadError("Images from different or unidentified DICOM series cannot be combined.")
        if len({_instance_phase_key(d) for d in datasets.values()}) != 1:
            raise DicomLoadError("The selection contains different acquisitions, phases, echoes, or image roles. Select one spatial stack.")
    identities = {(str(getattr(d, "FrameOfReferenceUID", "")), str(getattr(d, "Modality", "")),
                   str(d.SOPClassUID), int(d.Rows), int(d.Columns),
                   str(getattr(d, "PhotometricInterpretation", ""))) for d in datasets.values()}
    if len(identities) != 1:
        raise DicomLoadError("The selection contains incompatible image types, dimensions, or coordinate frames.")
    phases = {frame.phase_key for frame in frames}
    if len(phases) != 1:
        raise DicomLoadError("The enhanced object contains multiple stacks, phases, or echoes. A single spatial stack is required.")
    if len(frames) > 1 and any("LOCALIZER" in [str(v).upper() for v in getattr(d, "ImageType", [])]
                               for d in datasets.values()):
        raise DicomLoadError("Localizer projections cannot be combined into a 3D volume.")
    _validate_volumetric_frames(frames)
    warnings: list[str] = []
    frames, spacing, origin, direction = _geometry(frames, warnings)
    rescale_types = {_voxel_units(f) for f in frames}
    is_hu = rescale_types == {"HU"}
    if len(rescale_types) > 1:
        raise DicomLoadError("Voxel units vary between slices; they cannot be combined as one scalar volume.")
    rows, columns = int(first.Rows), int(first.Columns)
    voxel_count = len(frames) * rows * columns
    padding_specs = {path: _padding_spec(dataset) for path, dataset in datasets.items()}
    has_padding = any(specification is not None for specification in padding_specs.values())
    try:
        # One encoded object, its decoded pixels, decoder scratch, and one
        # float32 rescaled frame coexist with the output. Header datasets are
        # metadata-only. File size also catches unusually padded PixelData.
        transient = max(
            path.stat().st_size + int(getattr(d, "NumberOfFrames", 1)) * rows * columns
            * max(1, int(getattr(d, "BitsAllocated", 16)) // 8) * 2
            + rows * columns * (7 if has_padding else 4)
            for path, d in datasets.items()
        )
    except OSError:
        raise DicomLoadError("An image is no longer available for decoding.") from None
    estimated = voxel_count * (5 if has_padding else 4) + transient
    if estimated > memory_budget_bytes:
        raise DicomLoadError(f"Estimated decode working set is {estimated / 1024**3:.1f} GiB, exceeding the configured {memory_budget_bytes / 1024**3:.1f} GiB safety budget. Load a smaller series or change the explicit budget.")
    _check_cancel(cancel)
    try:
        data = np.empty((len(frames), rows, columns), dtype=np.float32)
        valid_mask = np.ones(data.shape, dtype=bool) if has_padding else None
    except (MemoryError, ValueError):
        raise DicomLoadError("There is not enough memory to allocate this volume.") from None
    # Decode each object once; frame order is independent of file order.
    by_path: dict[Path, list[tuple[int, _Frame]]] = {}
    for output_index, frame in enumerate(frames):
        by_path.setdefault(frame.path, []).append((output_index, frame))
    completed = 0
    padding_voxel_count = 0
    for path, members in by_path.items():
        _check_cancel(cancel)
        try:
            dataset = pydicom.dcmread(str(path), force=not pydicom.misc.is_dicom(str(path)))
            if not any(tag in dataset for tag in ("PixelData", "FloatPixelData", "DoubleFloatPixelData")):
                raise DicomLoadError("A selected DICOM object contains no image pixel data.")
            pixels = dataset.pixel_array
        except DicomLoadError:
            raise
        except MemoryError:
            raise DicomLoadError("The pixel decoder ran out of memory.") from None
        except Exception:
            raise DicomLoadError("Pixel data could not be decoded. The transfer syntax may require an additional codec, or the image may be damaged.") from None
        count = int(getattr(dataset, "NumberOfFrames", 1))
        expected = (rows, columns) if count == 1 else (count, rows, columns)
        if pixels.shape != expected:
            # pydicom may preserve a singleton frame axis on some decoders.
            if count == 1 and pixels.shape == (1, rows, columns):
                pixels = pixels[0]
            else:
                raise DicomLoadError("Decoded pixel dimensions do not match the DICOM header.")
        for output_index, frame in members:
            _check_cancel(cancel)
            frame_pixels = pixels if count == 1 else pixels[frame.index]
            specification = padding_specs[path]
            if specification is not None:
                valid_mask[output_index] = _valid_pixels(frame_pixels, specification)
                padding_voxel_count += rows * columns - int(np.count_nonzero(valid_mask[output_index]))
            data[output_index] = _transform_pixels(frame_pixels, frame)
            completed += 1
            _progress(progress, 10 + round(85 * completed / len(frames)), "Decoding image pixels…")
        # The last frame is a view into the complete decoded multiframe array;
        # drop it too before decoding the next object.
        del frame_pixels, pixels, dataset
    _check_cancel(cancel)
    modality = str(getattr(first, "Modality", "DICOM"))
    units = "HU" if is_hu else (next(iter(rescale_types)) or "Unspecified intensity")
    if modality == "CT" and not is_hu:
        warnings.append("DICOM does not explicitly declare HU units. CT/CBCT voxel values are displayed as intensity and must not be assumed to be calibrated HU.")
    if any(getattr(d, "RealWorldValueMappingSequence", None) for d in datasets.values()):
        warnings.append("A real-world value mapping is present. This build displays modality-rescaled values; separate real-world mappings are not applied.")
    for frame in frames:
        if frame.transform is not first and not any(hasattr(frame.transform, k) for k in ("RescaleSlope", "RescaleIntercept", "ModalityLUTSequence")):
            warnings.append("An image has no modality value transformation; stored voxel values are displayed.")
            break
    center = _scalar(getattr(first, "WindowCenter", None))
    width = _scalar(getattr(first, "WindowWidth", None))
    auto_window = center is None or width is None or width <= 0
    if auto_window:
        center, width = 0., 1.
    if has_padding:
        warnings.append("DICOM pixel padding is excluded from intensity ranges, automatic windowing, and validity-aware resampling; padding is displayed as background.")
    metadata = {
        "modality": modality,
        "series_description": str(getattr(first, "SeriesDescription", "")),
        "study_description": str(getattr(first, "StudyDescription", "")),
        "patient_id": str(getattr(first, "PatientID", "")),
        "patient_name": str(getattr(first, "PatientName", "")),
        "patient_age": str(getattr(first, "PatientAge", "")),
        "patient_birth_date": str(getattr(first, "PatientBirthDate", "")),
        "study_date": str(getattr(first, "StudyDate", "")),
        "study_time": str(getattr(first, "StudyTime", "")),
        "acquisition_date": str(getattr(first, "AcquisitionDate", "")),
        "acquisition_time": str(getattr(first, "AcquisitionTime", "")),
        "acquisition_datetime": str(getattr(first, "AcquisitionDateTime", "")),
        "source_slice_thickness_mm": str(getattr(first, "SliceThickness", "")),
        "kvp": str(getattr(first, "KVP", "")),
        "tube_current_ma": str(getattr(first, "XRayTubeCurrent", "")),
        "exposure_mas": str(getattr(first, "Exposure", "")),
        "exposure_time_ms": str(getattr(first, "ExposureTime", "")),
        "series_uid": str(getattr(first, "SeriesInstanceUID", "")),
        "frame_of_reference_uid": str(getattr(first, "FrameOfReferenceUID", "")),
        "source_group": str(info.uid),
        "sop_instance_digest": sha256("\n".join(sorted(seen_instances)).encode("ascii")).hexdigest(),
        "manufacturer": str(getattr(first, "Manufacturer", "")),
        "model": str(getattr(first, "ManufacturerModelName", "")),
        "units": units,
        "photometric_interpretation": str(first.PhotometricInterpretation),
        "monochrome1": str(first.PhotometricInterpretation) == "MONOCHROME1",
        "source_file_count": len(datasets),
        "frame_count": len(frames),
        "geometry_calibrated": all(f.position is not None and f.orientation is not None and f.pixel_spacing is not None for f in frames) and not any("uncalibrated" in warning or "arbitrary" in warning for warning in warnings),
        "synthetic": False,
        "padding_excluded": has_padding,
        "padding_voxel_count": padding_voxel_count,
    }
    try:
        volume = Volume(data, spacing, origin, direction, metadata, (center, width), is_hu, warnings,
                        valid_mask=valid_mask)
    except ValueError as error:
        if "no valid non-padding voxels" in str(error):
            raise DicomLoadError("The selected images contain no valid non-padding voxels.") from None
        raise DicomLoadError("The decoded volume contains invalid voxel values or invalid geometry.") from None
    if auto_window:
        step = max(1, data.size // 200_000)
        sample = data.reshape(-1)[::step]
        if valid_mask is not None:
            sample = sample[valid_mask.reshape(-1)[::step]]
        if not sample.size:
            sample = np.array([volume.data_min, volume.data_max], dtype=np.float32)
        low, high = np.percentile(sample, (1, 99.8))
        volume.window = (float((low + high) / 2), float(max(1, high - low)))
    _progress(progress, 100, "Volume loaded.")
    return volume
