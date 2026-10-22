"""Native Qt/VTK GPU volume rendering in DICOM patient coordinates (LPS).

The full source volume remains owned by the data model. Only this 3-D view may
use a smaller, explicitly reported proxy; slice and measurement tools must keep
using the original image. CBCT presets operate on relative scalar values and
never relabel those values as calibrated Hounsfield units.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QThread, QTimer, Signal, Slot
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QVBoxLayout, QWidget
from scipy.ndimage import affine_transform

import vtkmodules.qt

# Select the binding before importing QVTK. This avoids a PyQt installation
# elsewhere on the machine changing which Qt runtime VTK attempts to use.
vtkmodules.qt.PyQtImpl = "PySide6"
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkCommand, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkImageData, vtkPiecewiseFunction, vtkPolyData
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkFiltersCore import vtkTubeFilter
from vtkmodules.vtkFiltersSources import vtkCylinderSource, vtkLineSource, vtkSphereSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor, vtkCornerAnnotation
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkBillboardTextActor3D,
    vtkCellPicker,
    vtkColorTransferFunction,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty,
    vtkWindowToImageFilter,
)
from vtkmodules.vtkRenderingVolume import vtkGPUVolumeRayCastMapper

# Register OpenGL rendering implementations and font support with VTK's factory.
import vtkmodules.vtkRenderingFreeType  # noqa: F401
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
import vtkmodules.vtkRenderingVolumeOpenGL2  # noqa: F401


PRESETS: dict[str, str] = {
    "Bone": "Cortical and trabecular bone",
    "Teeth": "High-density dental structures",
    "Soft tissue": "Low-opacity soft-tissue overview",
    "Transparent": "Low-opacity anatomy overview",
    "X-ray": "Grayscale translucent visualization",
    "MIP": "Maximum-intensity projection",
}

_SPATIAL_VOLUME_REQUIRED = (
    "3D rendering requires a spatial volume; single 2D image available in section view"
)

# Proxy budgets include scalar voxels only. VTK needs additional texture and
# working memory, so the mapper also receives a separate conservative VRAM cap.
_QUALITY = {
    "Interactive": {"voxels": 64_000_000, "edge": 384, "ray": 1.6, "image": 1.6},
    "Balanced": {"voxels": 128_000_000, "edge": 640, "ray": 1.0, "image": 1.0},
    "High": {"voxels": 256_000_000, "edge": 1024, "ray": 0.6, "image": 1.0},
}

# These match the section-view accents. Geometry is always expressed in LPS/mm,
# independent of the texture proxy or the volume's acquired orientation.
_REFERENCE_PLANES = {
    "Axial": (2, (0.443, 0.663, 0.969)),
    "Coronal": (1, (0.420, 0.780, 0.639)),
    "Sagittal": (0, (0.898, 0.522, 0.545)),
}
_BOX_CORNER_BITS = np.indices((2, 2, 2)).reshape(3, -1).T
_BOX_EDGES = tuple(
    (first, second)
    for first in range(8) for second in range(first + 1, 8)
    if np.count_nonzero(_BOX_CORNER_BITS[first] != _BOX_CORNER_BITS[second]) == 1
)


def _volume_corners(volume: Any) -> np.ndarray:
    """Eight acquired voxel-center corners, in patient LPS coordinates."""
    extent = (np.asarray(volume.data.shape[::-1], dtype=float) - 1) * np.asarray(volume.spacing)
    local = _BOX_CORNER_BITS * extent
    return local @ np.asarray(volume.direction, dtype=float).T + np.asarray(volume.origin, dtype=float)


def _plane_box_polygon(corners: np.ndarray, center: np.ndarray,
                       normal: np.ndarray) -> np.ndarray:
    """Intersect a physical plane with an oriented box; return ordered vertices.

    This deliberately clips against the acquired box, rather than its larger
    world-axis bounding box. At oblique orientations, the polygon can have
    three to six vertices. Tangent points/edges have no visible surface.
    """
    corners = np.asarray(corners, dtype=float)
    center = np.asarray(center, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    tolerance = max(float(np.linalg.norm(np.ptp(corners, axis=0))), 1.0) * 1e-8
    distances = (corners - center) @ normal
    points: list[np.ndarray] = []

    def add(point: np.ndarray) -> None:
        if not any(np.linalg.norm(point - previous) <= tolerance for previous in points):
            points.append(point)

    for first, second in _BOX_EDGES:
        start, end = corners[first], corners[second]
        distance_start, distance_end = distances[first], distances[second]
        if abs(distance_start) <= tolerance:
            add(start)
        if abs(distance_end) <= tolerance:
            add(end)
        if ((distance_start < -tolerance and distance_end > tolerance)
                or (distance_start > tolerance and distance_end < -tolerance)):
            add(start + (end - start) * (distance_start / (distance_start - distance_end)))
    if len(points) < 3:
        return np.empty((0, 3), dtype=float)
    polygon = np.asarray(points)
    # Choose a well-conditioned in-plane basis for stable angular ordering.
    reference = np.eye(3)[int(np.argmin(np.abs(normal)))]
    right = np.cross(normal, reference)
    right /= np.linalg.norm(right)
    up = np.cross(normal, right)
    offsets = polygon - polygon.mean(axis=0)
    order = np.argsort(np.arctan2(offsets @ up, offsets @ right))
    return polygon[order]


def _proxy_shape(shape: tuple[int, ...], quality: str) -> tuple[int, int, int]:
    """Fit z/y/x dimensions into a quality budget without changing proportions."""
    spec = _QUALITY[quality]
    shape_array = np.asarray(shape, dtype=np.int64)
    scale = min(
        1.0,
        float(spec["edge"]) / int(shape_array.max()),
        (float(spec["voxels"]) / int(np.prod(shape_array))) ** (1.0 / 3.0),
    )
    target = np.maximum(2, np.floor(shape_array * scale)).astype(int)
    target = np.minimum(target, shape_array)
    return tuple(int(v) for v in target)


class _ProxyCancelled(Exception):
    pass


def _prepare_data(volume: Any, quality: str, cancelled=None) -> tuple[np.ndarray, np.ndarray, bool]:
    """Pure NumPy/SciPy work, safe to run away from the Qt/VTK GUI thread."""
    source = np.asarray(volume.data)
    if source.ndim != 3 or min(source.shape) < 1:
        raise ValueError("3-D rendering requires a nonempty z/y/x volume.")
    if source.dtype != np.float32 or not source.flags.c_contiguous:
        source = np.ascontiguousarray(source, dtype=np.float32)
    target = _proxy_shape(source.shape, quality)
    proxy = target != source.shape
    spacing = np.asarray(volume.spacing, dtype=float)
    valid_mask = getattr(volume, "valid_mask", None)
    padding_value = getattr(volume, "padding_value", None)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != source.shape:
            raise ValueError("The validity mask must match the source volume shape.")
        if padding_value is None or not np.isfinite(padding_value):
            raise ValueError("A masked volume requires a finite padding sentinel.")
    if proxy:
        # Preserve the first/last voxel centers, including oblique geometry.
        # Small output z-tiles allow cancellation between bounded C calls.
        data = np.empty(target, dtype=np.float32)
        step = np.divide(np.asarray(source.shape) - 1, np.asarray(target) - 1,
                         out=np.zeros(3, dtype=float), where=np.asarray(target) > 1)
        for first in range(0, target[0], 4):
            if cancelled is not None and cancelled():
                raise _ProxyCancelled()
            slab = data[first:min(first + 4, target[0])]
            affine_transform(source, np.diag(step), offset=step * (first, 0, 0),
                             output=slab, order=1, prefilter=False, mode="nearest")
            if valid_mask is not None:
                weights = affine_transform(valid_mask, np.diag(step),
                                           offset=step * (first, 0, 0), output_shape=slab.shape,
                                           output=np.float32, order=1, prefilter=False, mode="nearest")
                # A proxy sample that touches padding remains transparent.
                # This avoids scalar blends creating opaque artificial borders.
                slab[weights < 1.0 - 1e-6] = float(padding_value)
        spacing = spacing * np.divide(
            np.asarray(source.shape[::-1], dtype=float) - 1,
            np.asarray(data.shape[::-1], dtype=float) - 1,
            out=np.ones(3, dtype=float),
            where=np.asarray(data.shape[::-1]) > 1,
        )
    else:
        data = source
    return data, spacing, proxy


def _image_from_data(volume: Any, data: np.ndarray, spacing: np.ndarray) -> vtkImageData:
    """VTK object creation stays on the GUI thread in the actual widget."""
    image = vtkImageData()
    image.SetDimensions(*data.shape[::-1])
    image.SetSpacing(*spacing)
    image.SetOrigin(*np.asarray(volume.origin, dtype=float))
    image.SetDirectionMatrix(*np.asarray(volume.direction, dtype=float).ravel())
    scalars = numpy_to_vtk(data.ravel(), deep=False)
    scalars.SetName("Scan intensity")
    image.GetPointData().SetScalars(scalars)
    # The caller keeps data alive for at least as long as image/scalars are used.
    return image


def _make_image(volume: Any, quality: str) -> tuple[vtkImageData, np.ndarray, bool]:
    """Synchronous helper for geometry tests and noninteractive callers."""
    data, spacing, proxy = _prepare_data(volume, quality)
    return _image_from_data(volume, data, spacing), data, proxy


class _ProxyWorker(QThread):
    ready = Signal(object, object, bool, int)
    failed = Signal(str, int)

    def __init__(self, volume: Any, quality: str, generation: int, parent: QWidget):
        super().__init__(parent)
        # Own a strong reference until run() finishes, even after a new series
        # replaces the widget's current volume.
        self.volume = volume
        self.quality = quality
        self.generation = generation

    def run(self) -> None:
        try:
            data, spacing, proxy = _prepare_data(self.volume, self.quality,
                                                self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.ready.emit(data, spacing, proxy, self.generation)
        except _ProxyCancelled:
            pass
        except Exception as error:
            self.failed.emit(str(error), self.generation)


class VolumeView(QWidget):
    """GPU ray-cast volume with orbit, zoom, pan, presets, and implant overlays."""

    status = Signal(str)
    path_node_dragged = Signal(object)
    implant_dragged = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(200, 180)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.vtk_widget)
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0.026, 0.028, 0.034)
        self.renderer.SetBackground2(0.075, 0.080, 0.092)
        self.renderer.GradientBackgroundOn()
        self._window = self.vtk_widget.GetRenderWindow()
        self._window.SetMultiSamples(0)
        self._window.SetNumberOfLayers(3)
        self._window.AddRenderer(self.renderer)
        # Editable control nodes are interaction overlays: their physical
        # positions stay in LPS, but anatomy and the enclosing path tube must
        # not hide them. Share the camera without accepting camera interaction
        # in this layer; clear only depth so the anatomy color is retained.
        self._node_renderer = vtkRenderer()
        self._node_renderer.SetLayer(1)
        self._node_renderer.InteractiveOff()
        self._node_renderer.SetPreserveColorBuffer(True)
        self._node_renderer.SetPreserveDepthBuffer(False)
        self._node_renderer.SetActiveCamera(self.renderer.GetActiveCamera())
        self._window.AddRenderer(self._node_renderer)
        self._interactor = self._window.GetInteractor()
        self._interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
        self._interactor.SetDesiredUpdateRate(45)
        self._interactor.SetStillUpdateRate(0.25)
        self._interactor.AddObserver(vtkCommand.StartInteractionEvent, self._interaction_start)
        self._interactor.AddObserver(vtkCommand.EndInteractionEvent, self._interaction_end)

        self.mapper = vtkGPUVolumeRayCastMapper()
        self.mapper.SetMaxMemoryInBytes(2 * 1024**3)
        self.mapper.SetMaxMemoryFraction(0.35)
        self.mapper.SetAutoAdjustSampleDistances(True)
        self.mapper.SetMinimumImageSampleDistance(1.0)
        self.mapper.SetMaximumImageSampleDistance(3.0)
        self.property = vtkVolumeProperty()
        self.property.SetInterpolationTypeToLinear()
        self.property.SetIndependentComponents(True)
        self.property.SetAmbient(0.24)
        self.property.SetDiffuse(0.78)
        self.property.SetSpecular(0.24)
        self.property.SetSpecularPower(18)
        self.property.SetScalarOpacityUnitDistance(1.0)
        self.actor = vtkVolume()
        self.actor.SetMapper(self.mapper)
        self.actor.SetProperty(self.property)
        self.actor.SetVisibility(False)
        self.renderer.AddVolume(self.actor)

        self.segment_mapper = vtkGPUVolumeRayCastMapper()
        self.segment_mapper.SetMaxMemoryInBytes(1024**3)
        self.segment_mapper.SetMaxMemoryFraction(.25)
        self.segment_mapper.SetAutoAdjustSampleDistances(True)
        # Keep shader scaling tied to actual image scalars. VTK 9.7.1's NATIVE
        # range path misregisters category colors on the tested OpenGL backend.
        # Bounded palette IDs/plateaus below keep SCALAR tables within 8192
        # texels, without the unsafe 65536-wide lookup fallback.
        self.segment_mapper.SetColorRangeType(vtkGPUVolumeRayCastMapper.SCALAR)
        self.segment_mapper.SetScalarOpacityRangeType(vtkGPUVolumeRayCastMapper.SCALAR)
        self.segment_property = vtkVolumeProperty()
        self.segment_property.SetInterpolationTypeToNearest()
        self.segment_property.SetIndependentComponents(True)
        self.segment_property.ShadeOff()
        self.segment_actor = vtkVolume()
        self.segment_actor.SetMapper(self.segment_mapper)
        self.segment_actor.SetProperty(self.segment_property)
        self.segment_actor.SetVisibility(False)
        self.renderer.AddVolume(self.segment_actor)
        self._segment_data = self._segment_image = None
        self._segment_display_data = None
        self._segment_display_ids: dict[int, int] = {}
        self._segment_display_lookup: np.ndarray | None = None
        self._segment_display_remapped = False

        self.annotation = vtkCornerAnnotation()
        self.annotation.SetLinearFontScaleFactor(1.0)
        self.annotation.SetNonlinearFontScaleFactor(1.0)
        self.annotation.SetMaximumFontSize(12)
        self.annotation.GetTextProperty().SetFontFamilyToArial()
        self.annotation.GetTextProperty().BoldOff()
        self.annotation.GetTextProperty().ShadowOff()
        self.annotation.GetTextProperty().SetColor(0.70, 0.77, 0.85)
        self.annotation.SetText(2, "3D anatomy")
        self.renderer.AddViewProp(self.annotation)
        self.axes = vtkAxesActor()
        self.axes.SetXAxisLabelText("L")
        self.axes.SetYAxisLabelText("P")
        self.axes.SetZAxisLabelText("S")
        self.axes.SetTotalLength(1, 1, 1)
        for caption in (self.axes.GetXAxisCaptionActor2D(), self.axes.GetYAxisCaptionActor2D(),
                        self.axes.GetZAxisCaptionActor2D()):
            caption.GetTextActor().SetTextScaleModeToNone()
            text = caption.GetCaptionTextProperty()
            text.SetFontFamilyToArial()
            text.SetFontSize(11)
            text.BoldOff()
            text.ShadowOff()
            text.SetColor(0.80, 0.85, 0.91)
        self.marker = vtkOrientationMarkerWidget()
        self.marker.SetOrientationMarker(self.axes)
        self.marker.SetInteractor(self._interactor)
        self.marker.GetRenderer().SetLayer(2)
        self.marker.SetCurrentRenderer(self.renderer)
        self.marker.SetViewport(0.02, 0.02, 0.16, 0.19)

        # The objects stay alive for the lifetime of the widget. Moving a
        # crosshair updates at most eighteen points, with no texture uploads.
        self._reference_geometry: dict[str, tuple[vtkPolyData, vtkPolyData]] = {}
        self._reference_actors: dict[str, tuple[vtkActor, vtkActor]] = {}
        for name, (_, color) in _REFERENCE_PLANES.items():
            surface, outline = vtkPolyData(), vtkPolyData()
            points = vtkPoints()
            points.SetDataTypeToDouble()
            surface.SetPoints(points)
            surface.SetPolys(vtkCellArray())
            outline.SetPoints(surface.GetPoints())
            outline.SetLines(vtkCellArray())
            actors = []
            for geometry in (surface, outline):
                mapper = vtkPolyDataMapper()
                mapper.SetInputData(geometry)
                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(*color)
                actor.GetProperty().LightingOff()
                actor.SetVisibility(False)
                actor.PickableOff()
                self.renderer.AddActor(actor)
                actors.append(actor)
            actors[0].GetProperty().SetOpacity(0.16)
            actors[1].GetProperty().SetOpacity(0.80)
            actors[1].GetProperty().SetLineWidth(1.25)
            self._reference_geometry[name] = (surface, outline)
            self._reference_actors[name] = (actors[0], actors[1])
        self._cursor_source = vtkSphereSource()
        self._cursor_source.SetRadius(0.6)
        self._cursor_source.SetThetaResolution(16)
        self._cursor_source.SetPhiResolution(12)
        cursor_mapper = vtkPolyDataMapper()
        cursor_mapper.SetInputConnection(self._cursor_source.GetOutputPort())
        self._cursor_actor = vtkActor()
        self._cursor_actor.SetMapper(cursor_mapper)
        self._cursor_actor.GetProperty().SetColor(0.91, 0.95, 1.0)
        self._cursor_actor.GetProperty().LightingOff()
        self._cursor_actor.SetVisibility(False)
        self._cursor_actor.PickableOff()
        self.renderer.AddActor(self._cursor_actor)

        self._volume: Any | None = None
        self._render_volume: Any | None = None
        self._image: vtkImageData | None = None
        self._shared_array: np.ndarray | None = None
        self._implant_actors: list[vtkActor] = []
        self._implants: dict[str, dict[str, Any]] = {}
        self._selected_implant_id: str | None = None
        self._implant_edit_enabled = False
        self._implant_drag: dict[str, Any] | None = None
        self._implant_picker = vtkCellPicker()
        self._implant_picker.PickFromListOn()
        self._implant_picker.SetTolerance(.003)
        self._implant_handle_lookup: dict[vtkActor, tuple[str, str]] = {}
        self._camera_drag: dict[str, Any] | None = None
        self._free_rotation_enabled = True
        self._quality = "Balanced"
        self._preset = "Bone"
        self._threshold = 0.0
        self._range = (0.0, 1.0)
        self._robust_range = (0.0, 1.0)
        self._proxy = False
        self._preparing = False
        self._single_image = False
        self._generation = 0
        self._worker: _ProxyWorker | None = None
        self._pending_proxy: tuple[int, Any, str] | None = None
        self._initialized = False
        self._gpu_supported: bool | None = None
        self.gpu_name = ""
        self._gpu_reduction = (1.0, 1.0, 1.0)
        self._render_pending = False
        self._rendering = False
        self._cleaned = False
        self._volume_visible = True
        self._annotation_detail = False
        self._reference_center: np.ndarray | None = None
        self._reference_rotation = np.eye(3)
        self._reference_visible = False
        self._reference_enabled = {name: True for name in _REFERENCE_PLANES}
        self._reference_opacity = 0.16
        self._reference_corners: np.ndarray | None = None
        self._reference_revision = 0
        self._reference_applied_revision = -1
        self._nerve_paths: dict[str, dict[str, Any]] = {}
        self._nerve_selected_path: str | None = None
        self._nerve_selected_node: int | None = None
        self._nerve_edit_enabled = False
        self._nerve_drag: dict[str, Any] | None = None
        self._nerve_picker = vtkCellPicker()
        self._nerve_picker.PickFromListOn()
        self._nerve_picker.SetTolerance(0.003)
        self._nerve_node_lookup: dict[vtkActor, tuple[str, int]] = {}
        # Priority observers consume only an editable node hit. Misses retain
        # VTK's existing camera interaction, including its sampling controls.
        self._nerve_observers = {
            vtkCommand.LeftButtonPressEvent: self._interactor.AddObserver(
                vtkCommand.LeftButtonPressEvent, self._nerve_pointer_press, 1.0),
            vtkCommand.MouseMoveEvent: self._interactor.AddObserver(
                vtkCommand.MouseMoveEvent, self._nerve_pointer_move, 1.0),
            vtkCommand.LeftButtonReleaseEvent: self._interactor.AddObserver(
                vtkCommand.LeftButtonReleaseEvent, self._nerve_pointer_release, 1.0),
        }
        self._implant_observers = {
            vtkCommand.LeftButtonPressEvent: self._interactor.AddObserver(
                vtkCommand.LeftButtonPressEvent, self._implant_pointer_press, .9),
            vtkCommand.MouseMoveEvent: self._interactor.AddObserver(
                vtkCommand.MouseMoveEvent, self._implant_pointer_move, .9),
            vtkCommand.LeftButtonReleaseEvent: self._interactor.AddObserver(
                vtkCommand.LeftButtonReleaseEvent, self._implant_pointer_release, .9),
        }
        self._camera_observers = {
            vtkCommand.LeftButtonPressEvent: self._interactor.AddObserver(
                vtkCommand.LeftButtonPressEvent, self._camera_pointer_press, .4),
            vtkCommand.MouseMoveEvent: self._interactor.AddObserver(
                vtkCommand.MouseMoveEvent, self._camera_pointer_move, .4),
            vtkCommand.LeftButtonReleaseEvent: self._interactor.AddObserver(
                vtkCommand.LeftButtonReleaseEvent, self._camera_pointer_release, .4),
        }
        self.vtk_widget.setToolTip("Drag left/right to turn · Drag up/down to tilt · Middle-drag to pan · Wheel to zoom · Home restores the view")
        self._clearance_signature = None
        self._clearance_measurement = None
        self._clearance_line = vtkLineSource()
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(self._clearance_line.GetOutputPort())
        self._clearance_actor = vtkActor()
        self._clearance_actor.SetMapper(mapper)
        self._clearance_actor.GetProperty().SetLineWidth(2.5)
        self._clearance_actor.GetProperty().LightingOff()
        self._clearance_actor.PickableOff()
        self._clearance_actor.SetVisibility(False)
        self._node_renderer.AddActor(self._clearance_actor)
        self._clearance_handles = []
        for _ in range(2):
            sphere, actor = self._new_overlay_handle(.2)
            actor.SetVisibility(False)
            actor.PickableOff()
            self._clearance_handles.append((sphere, actor))
        self._clearance_label = vtkBillboardTextActor3D()
        self._clearance_label.GetTextProperty().SetFontFamilyToArial()
        self._clearance_label.GetTextProperty().SetFontSize(14)
        self._clearance_label.GetTextProperty().BoldOff()
        self._clearance_label.GetTextProperty().SetBackgroundColor(.045, .05, .065)
        self._clearance_label.GetTextProperty().SetBackgroundOpacity(.8)
        self._clearance_label.SetDisplayOffset(8, 8)
        self._clearance_label.SetVisibility(False)
        self._clearance_label.PickableOff()
        self._node_renderer.AddActor(self._clearance_label)
        self.info = "GPU support is checked when the 3D view is displayed."

    def set_volume(self, volume: Any) -> None:
        self._camera_drag = None
        self.set_implants([])
        self.set_clearance_measurement(None)
        self.set_nerve_paths([])
        self.set_segmentation(None)
        self._volume = volume
        self._reference_corners = _volume_corners(volume)
        self._reference_center = self._reference_corners.mean(axis=0)
        self._reference_revision += 1
        self._threshold = 0.0
        self._single_image = min(volume.data.shape) < 2
        self._update_reference_geometry()
        if self._single_image:
            self._show_section_only()
            return
        flat = np.asarray(volume.data).reshape(-1)
        stride = max(1, len(flat) // 1_000_000)
        sample = flat[::stride]
        valid_mask = getattr(volume, "valid_mask", None)
        if valid_mask is not None:
            valid_flat = np.asarray(valid_mask).reshape(-1)
            sample = sample[valid_flat[::stride]]
            # A tiny field of valid anatomy can fall between sampled indices.
            # Search bounded blocks instead of copying the full masked volume.
            if not len(sample):
                for first in range(0, len(flat), 1_000_000):
                    block_mask = valid_flat[first:first + 1_000_000]
                    if block_mask.any():
                        sample = flat[first:first + 1_000_000][block_mask]
                        break
        sample = sample[np.isfinite(sample)]
        if not len(sample):
            raise ValueError("The volume contains no finite intensity values.")
        self._range = (float(getattr(volume, "data_min", sample.min())),
                       float(getattr(volume, "data_max", sample.max())))
        low, high = np.percentile(sample, [0.5, 99.8])
        if high <= low:
            high = low + 1.0
        self._robust_range = (float(low), float(high))
        self._build_proxy()
        self._apply_preset()
        self.reset_camera()
        self._update_info()
        self._queue_render()

    def set_preset(self, name: str) -> None:
        normalized = str(name).strip().lower().replace("-", " ").replace("_", " ")
        aliases = {key.lower().replace("-", " "): key for key in PRESETS}
        aliases["xray"] = "X-ray"
        aliases["soft"] = "Soft tissue"
        aliases["maximum intensity"] = "MIP"
        if normalized not in aliases:
            raise ValueError(f"Unknown volume preset: {name}")
        self._preset = aliases[normalized]
        if self._volume is not None and not self._single_image:
            self._apply_preset()
        self._update_info()
        self._queue_render()

    def set_quality(self, name: str) -> None:
        names = {key.lower(): key for key in _QUALITY}
        if str(name).lower() not in names:
            raise ValueError(f"Unknown rendering quality: {name}")
        quality = names[str(name).lower()]
        if quality == self._quality:
            return
        self._quality = quality
        if self._volume is not None:
            self._build_proxy()
        self._update_info()
        self._queue_render()

    def set_threshold(self, offset: float) -> None:
        """Shift transfer-function intensity points by scan scalar units."""
        if not np.isfinite(offset):
            raise ValueError("The intensity offset must be finite.")
        self._threshold = float(offset)
        if self._volume is not None and not self._single_image:
            self._apply_preset()
        self._queue_render()

    def set_volume_visibility(self, visible: bool) -> None:
        """Show/hide ray casting while keeping section locators available."""
        self._volume_visible = bool(visible)
        self.actor.SetVisibility(bool(self._gpu_supported) and self._image is not None
                                 and not self._single_image and self._volume_visible)
        self._queue_render()

    def set_annotation_detail(self, enabled: bool) -> None:
        """Keep the viewport quiet when controls already show render details."""
        self._annotation_detail = bool(enabled)
        self._update_annotation()
        self._queue_render()

    def set_reference_planes(self, center: Any, rotation: Any, visible: bool = True,
                             enabled_planes: dict[str, bool] | None = None,
                             opacity: float = 0.16) -> None:
        """Synchronize three section locators with an LPS/mm crosshair.

        ``rotation`` is the same active, proper LPS rotation used by MPR. Each
        surface is clipped to acquired voxel-center bounds, including oblique
        scans. Plane toggles affect only the locators, never the source volume.
        Missing names in ``enabled_planes`` retain their current setting.
        """
        center = np.asarray(center, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("The reference center must be a finite LPS point.")
        if (rotation.shape != (3, 3) or not np.all(np.isfinite(rotation))
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)):
            raise ValueError("Reference planes require a finite orthonormal 3 x 3 rotation.")
        if not np.isfinite(opacity) or not 0 <= opacity <= 1:
            raise ValueError("Reference plane opacity must be between zero and one.")
        enabled = dict(self._reference_enabled)
        if enabled_planes is not None:
            unknown = set(enabled_planes) - set(_REFERENCE_PLANES)
            if unknown:
                raise ValueError(f"Unknown reference plane: {', '.join(sorted(unknown))}")
            enabled.update({name: bool(value) for name, value in enabled_planes.items()})
        changed = (self._reference_center is None
                   or not np.array_equal(center, self._reference_center)
                   or not np.array_equal(rotation, self._reference_rotation)
                   or self._reference_visible != bool(visible)
                   or self._reference_enabled != enabled
                   or self._reference_opacity != float(opacity))
        if not changed:
            return
        self._reference_center = center.copy()
        self._reference_rotation = rotation.copy()
        self._reference_visible = bool(visible)
        self._reference_enabled = enabled
        self._reference_opacity = float(opacity)
        self._reference_revision += 1
        self._update_reference_geometry()
        self._queue_render()

    def _update_reference_geometry(self) -> None:
        if self._reference_applied_revision == self._reference_revision:
            return
        self._reference_applied_revision = self._reference_revision
        visible = (self._reference_visible and self._volume is not None
                   and not self._single_image and self._reference_center is not None)
        for name, (axis, _) in _REFERENCE_PLANES.items():
            surface, outline = self._reference_geometry[name]
            actors = self._reference_actors[name]
            polygon = (_plane_box_polygon(self._reference_corners, self._reference_center,
                                          self._reference_rotation[:, axis])
                       if visible and self._reference_enabled[name] else np.empty((0, 3)))
            point_count = len(polygon)
            points = surface.GetPoints()
            points.SetNumberOfPoints(point_count)
            for index, point in enumerate(polygon):
                points.SetPoint(index, *point)
            points.Modified()
            cells = surface.GetPolys()
            cells.Reset()
            lines = outline.GetLines()
            lines.Reset()
            if point_count >= 3:
                cells.InsertNextCell(point_count)
                lines.InsertNextCell(point_count + 1)
                for index in range(point_count):
                    cells.InsertCellPoint(index)
                    lines.InsertCellPoint(index)
                lines.InsertCellPoint(0)
            cells.Modified()
            lines.Modified()
            surface.Modified()
            outline.Modified()
            actors[0].GetProperty().SetOpacity(self._reference_opacity)
            for actor in actors:
                actor.SetVisibility(point_count >= 3)
        # A small cursor stays meaningful at every study scale. It is hidden
        # outside the acquired support rather than marking unacquired space.
        inside = False
        if visible and any(self._reference_enabled.values()):
            local = np.linalg.solve(np.asarray(self._volume.direction),
                                    self._reference_center - np.asarray(self._volume.origin))
            extent = ((np.asarray(self._volume.data.shape[::-1]) - 1)
                      * np.asarray(self._volume.spacing))
            inside = bool(np.all(local >= -1e-6) and np.all(local <= extent + 1e-6))
            self._cursor_source.SetCenter(*self._reference_center)
            self._cursor_source.SetRadius(float(np.clip(np.min(extent) * .008, .15, 1.1)))
        self._cursor_actor.SetVisibility(inside)

    def set_view_orientation(self, name: str) -> None:
        """Frame the volume from a named anatomical direction in patient LPS."""
        self._camera_drag = None
        views = {
            "anterior": ((0, -1, 0), (0, 0, 1)),
            "posterior": ((0, 1, 0), (0, 0, 1)),
            "left": ((1, 0, 0), (0, 0, 1)),
            "right": ((-1, 0, 0), (0, 0, 1)),
            "superior": ((0, 0, 1), (0, 1, 0)),
            "inferior": ((0, 0, -1), (0, -1, 0)),
            "isometric": ((1, -1, .65), (0, 0, 1)),
        }
        try:
            direction, up = views[str(name).strip().lower()]
        except KeyError as error:
            raise ValueError(f"Unknown 3D view orientation: {name}") from error
        if self._volume is None or self._single_image:
            return
        corners = self._reference_corners
        center = corners.mean(axis=0)
        distance = max(float(np.linalg.norm(np.ptp(corners, axis=0))) * 1.7, 10.0)
        direction = np.asarray(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(*center)
        camera.SetPosition(*(center + direction * distance))
        camera.SetViewUp(*up)
        camera.ParallelProjectionOn()
        bounds = tuple(float(value) for pair in zip(corners.min(axis=0), corners.max(axis=0))
                       for value in pair)
        self.renderer.ResetCamera(*bounds)
        self.renderer.ResetCameraClippingRange(*bounds)
        self._queue_render()

    def reset_camera(self) -> None:
        self.set_view_orientation("Isometric")

    def set_segmentation(self, data, labels=None, visible_ids=None, *, visible=True):
        """Display exact nearest-neighbor categories separately from CT.

        Source IDs through 2047 share mask storage directly. Higher IDs use a bounded display palette
        (at most 512 categories). Only that fallback owns a compact display
        copy; source IDs, editing masks, and exported data remain unchanged.
        """
        if data is None:
            self.segment_actor.SetVisibility(False)
            self.segment_mapper.RemoveAllInputs()
            self._segment_data = self._segment_image = None
            self._segment_display_data = self._segment_display_lookup = None
            self._segment_display_ids = {}
            self._segment_display_remapped = False
            self._queue_render()
            return
        if self._volume is None or data.shape != self._volume.data.shape or data.dtype not in (np.uint8, np.uint16) or not data.flags.c_contiguous:
            raise ValueError("3D masks must match the source grid as contiguous unsigned labels")
        labels = labels or {}
        identities = sorted(labels)
        if (len(identities) > 512 or any(not isinstance(key, (int, np.integer)) or isinstance(key, bool)
                                        or not 1 <= key <= np.iinfo(data.dtype).max for key in identities)):
            raise ValueError("3D masks require at most 512 declared nonzero unsigned label IDs.")
        remapped = bool(identities and identities[-1] > 2047)
        display_ids = {int(key): index + 1 if remapped else int(key)
                       for index, key in enumerate(identities)}
        if self._segment_data is not data or self._segment_display_ids != display_ids:
            self._segment_data = data
            self._segment_display_ids = display_ids
            self._segment_display_remapped = remapped
            if remapped:
                dtype = np.uint8 if len(identities) <= 255 else np.uint16
                lookup = np.zeros(np.iinfo(data.dtype).max + 1, dtype=dtype)
                for key, display_id in display_ids.items():
                    lookup[key] = display_id
                self._segment_display_lookup = lookup
                self._segment_display_data = np.empty(data.shape, dtype=dtype)
                np.take(lookup, data, out=self._segment_display_data, mode="clip")
            else:
                self._segment_display_lookup = None
                self._segment_display_data = data
            self._segment_image = _image_from_data(self._volume, self._segment_display_data,
                                                   np.asarray(self._volume.spacing))
            self._segment_image.GetPointData().GetScalars().SetName("Reviewed segmentation labels")
            self.segment_mapper.SetInputData(self._segment_image)
        color, opacity = vtkColorTransferFunction(), vtkPiecewiseFunction()
        if not identities:
            for value in (0., 1.):
                opacity.AddPoint(value, 0)
                color.AddRGBPoint(value, 0, 0, 0)
        for key in identities:
            label = labels[key]
            value = label.get("color", "#d8b78a").lstrip("#")
            rgb = [int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)]
            shown = key in (visible_ids or set()) and label["review"] != "rejected"
            display_id = display_ids[key]
            # Flat centers prevent RGB/opacity bleeding from LUT texel rounding.
            # Transparent half-integer guards also keep undeclared gaps clear.
            # A 0.25-unit minimum spacing caps ideal table width below 8192
            # (2048 for a compact palette), avoiding 65536-wide LUTs.
            for offset, alpha, point_color in ((-.5, 0., (0., 0., 0.)),
                                                (-.25, .8 if shown else 0., rgb),
                                                (.25, .8 if shown else 0., rgb),
                                                (.5, 0., (0., 0., 0.))):
                color.AddRGBPoint(display_id + offset, *point_color)
                opacity.AddPoint(display_id + offset, alpha)
        self.segment_property.SetColor(color)
        self.segment_property.SetScalarOpacity(opacity)
        self.segment_actor.SetVisibility(bool(visible and visible_ids))
        self._queue_render()

    def segmentation_modified(self):
        if self._segment_image is not None:
            if self._segment_display_remapped:
                np.take(self._segment_display_lookup, self._segment_data,
                        out=self._segment_display_data, mode="clip")
            self._segment_image.GetPointData().GetScalars().Modified()
            self._segment_image.Modified()
            self._queue_render()

    def set_nerve_paths(self, paths: list[dict[str, Any]], selected_path_id: str | None = None,
                       selected_node: int | None = None, edit_enabled: bool = False) -> None:
        """Display reviewed canal paths and editable control nodes in LPS/mm.

        ``points_lps_mm`` contains original control nodes. The optional
        ``sampled_points_lps_mm`` contains the same bounded smooth curve used by
        the 2D views. Paths use a physical ``diameter_mm`` and ``#RRGGBB`` color;
        rejected or explicitly hidden paths cannot be displayed or picked.
        Updating a path reuses its tube, nodes, and all volume textures.
        """
        if QThread.currentThread() != self.thread():
            raise RuntimeError("3D path geometry must be updated on the GUI thread.")
        prepared = {}
        for path in paths:
            identity = str(path["id"])
            if not identity or identity in prepared:
                raise ValueError("Nerve paths require unique nonempty IDs.")
            nodes = np.asarray(path["points_lps_mm"], dtype=float)
            if nodes.size == 0:
                nodes = np.empty((0, 3), dtype=float)
            curve = np.asarray(path.get("sampled_points_lps_mm", nodes), dtype=float)
            if curve.size == 0:
                curve = np.empty((0, 3), dtype=float)
            for points in (nodes, curve):
                if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
                    raise ValueError("Nerve path points must be finite N x 3 patient LPS coordinates.")
            diameter = float(path.get("diameter_mm", 2.0))
            if not np.isfinite(diameter) or diameter <= 0:
                raise ValueError("Nerve path diameter must be finite and positive.")
            value = str(path.get("color", "#E5858B"))
            try:
                if len(value) != 7 or not value.startswith("#"):
                    raise ValueError()
                color = tuple(int(value[index:index + 2], 16) / 255 for index in (1, 3, 5))
            except ValueError as error:
                raise ValueError("Nerve path colors must use #RRGGBB.") from error
            prepared[identity] = {
                "nodes": nodes.copy(), "curve": curve.copy(), "diameter": diameter, "color": color,
                "shown": bool(path.get("visible", True)) and path.get("review", "pending") != "rejected",
            }
        if selected_node is not None and (isinstance(selected_node, bool)
                                         or not isinstance(selected_node, (int, np.integer))):
            raise ValueError("The selected path node must be an integer index.")
        self._nerve_selected_path = None if selected_path_id is None else str(selected_path_id)
        self._nerve_selected_node = None if selected_node is None else int(selected_node)
        self._nerve_edit_enabled = bool(edit_enabled)
        for identity in set(self._nerve_paths) - set(prepared):
            record = self._nerve_paths.pop(identity)
            self.renderer.RemoveActor(record["actor"])
            for source, actor in record["handles"]:
                self._node_renderer.RemoveActor(actor)
        for identity, path in prepared.items():
            record = self._nerve_paths.get(identity)
            if record is None:
                points = vtkPoints()
                points.SetDataTypeToDouble()
                polydata = vtkPolyData()
                polydata.SetPoints(points)
                polydata.SetLines(vtkCellArray())
                tube = vtkTubeFilter()
                tube.SetInputData(polydata)
                tube.SetNumberOfSides(16)
                tube.CappingOn()
                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(tube.GetOutputPort())
                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetAmbient(.45)
                actor.GetProperty().SetDiffuse(.55)
                actor.GetProperty().SetSpecular(.25)
                actor.PickableOff()
                self.renderer.AddActor(actor)
                record = {"polydata": polydata, "tube": tube, "actor": actor, "handles": [],
                          "nodes": np.empty((0, 3)), "curve": np.empty((0, 3))}
                self._nerve_paths[identity] = record
            if not np.array_equal(record["curve"], path["curve"]):
                polydata = record["polydata"]
                points = polydata.GetPoints()
                points.SetNumberOfPoints(len(path["curve"]))
                for index, point in enumerate(path["curve"]):
                    points.SetPoint(index, *point)
                points.Modified()
                lines = polydata.GetLines()
                lines.Reset()
                if len(path["curve"]) >= 2:
                    lines.InsertNextCell(len(path["curve"]))
                    for index in range(len(path["curve"])):
                        lines.InsertCellPoint(index)
                lines.Modified()
                polydata.Modified()
            record["tube"].SetRadius(path["diameter"] / 2)
            record["actor"].GetProperty().SetColor(*path["color"])
            record["actor"].GetProperty().SetOpacity(.95 if identity == self._nerve_selected_path else .80)
            shown = bool(path["shown"] and self._volume is not None and not self._single_image)
            record["actor"].SetVisibility(shown and len(path["curve"]) >= 2)
            while len(record["handles"]) > len(path["nodes"]):
                source, actor = record["handles"].pop()
                self._node_renderer.RemoveActor(actor)
            while len(record["handles"]) < len(path["nodes"]):
                source = vtkSphereSource()
                source.SetThetaResolution(14)
                source.SetPhiResolution(10)
                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(source.GetOutputPort())
                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().LightingOff()
                self._node_renderer.AddActor(actor)
                record["handles"].append((source, actor))
            for index, (source, actor) in enumerate(record["handles"]):
                selected = identity == self._nerve_selected_path and index == self._nerve_selected_node
                source.SetCenter(*path["nodes"][index])
                source.SetRadius(float(np.clip(path["diameter"] * .35, .45, 1.1)) * (1.3 if selected else 1))
                actor.GetProperty().SetColor(*((1., .88, .55) if selected else (.95, .97, 1.)))
                actor.SetVisibility(shown and identity == self._nerve_selected_path)
                actor.SetPickable(self._nerve_edit_enabled)
            record.update(path)
        self._nerve_picker.InitializePickList()
        self._nerve_node_lookup.clear()
        for identity, record in self._nerve_paths.items():
            for index, (source, actor) in enumerate(record["handles"]):
                if actor.GetVisibility() and self._nerve_edit_enabled:
                    self._nerve_picker.AddPickList(actor)
                    self._nerve_node_lookup[actor] = (identity, index)
        if self._nerve_drag is not None:
            record = self._nerve_paths.get(self._nerve_drag["path_id"])
            index = self._nerve_drag["node_index"]
            if (not self._nerve_edit_enabled or record is None or not record["shown"]
                    or self._nerve_drag["path_id"] != self._nerve_selected_path or index >= len(record["nodes"])):
                self._nerve_drag = None
            else:
                self._nerve_drag["position"] = record["nodes"][index].copy()
        self._queue_render()

    def _pick_nerve_node(self, x: float, y: float) -> tuple[str, int] | None:
        if not self._nerve_edit_enabled or not self._nerve_node_lookup:
            return None
        if not self._nerve_picker.Pick(float(x), float(y), 0.0, self._node_renderer):
            return None
        return self._nerve_node_lookup.get(self._nerve_picker.GetActor())

    def focus_nerve_path(self, path_id: str) -> bool:
        """Explicitly frame one visible path while preserving camera direction."""
        record = self._nerve_paths.get(str(path_id))
        if record is None or not record["shown"] or not len(record["nodes"]):
            return False
        points = record["nodes"]
        margin = max(float(record["diameter"]) * 1.5, 1.5)
        bounds = tuple(float(value) for pair in zip(points.min(axis=0) - margin,
                                                    points.max(axis=0) + margin) for value in pair)
        self.renderer.ResetCamera(*bounds)
        self.renderer.ResetCameraClippingRange(*bounds)
        self._queue_render()
        return True

    def _display_on_plane(self, x: float, y: float, origin: np.ndarray,
                          normal: np.ndarray) -> np.ndarray | None:
        """Unproject a display ray onto a fixed patient-space edit plane."""
        ray = []
        for depth in (0., 1.):
            self.renderer.SetDisplayPoint(float(x), float(y), depth)
            self.renderer.DisplayToWorld()
            homogeneous = np.asarray(self.renderer.GetWorldPoint(), dtype=float)
            if not np.all(np.isfinite(homogeneous)) or abs(homogeneous[3]) < 1e-12:
                return None
            ray.append(homogeneous[:3] / homogeneous[3])
        direction = ray[1] - ray[0]
        denominator = float(direction @ normal)
        if abs(denominator) < 1e-12:
            return None
        return ray[0] + direction * float((origin - ray[0]) @ normal) / denominator

    def _nerve_abort_event(self, event: str, handled: bool) -> None:
        event_id = vtkCommand.GetEventIdFromString(event)
        self._interactor.GetCommand(self._nerve_observers[event_id]).SetAbortFlag(bool(handled))

    def _emit_nerve_drag(self, phase: str, position: np.ndarray) -> None:
        drag = self._nerve_drag
        drag["position"] = np.asarray(position, dtype=float).copy()
        self.path_node_dragged.emit({"phase": phase, "path_id": drag["path_id"],
                                     "node_index": drag["node_index"],
                                     "position_lps_mm": drag["position"].copy()})

    def _nerve_pointer_press(self, _caller: Any, event: str) -> None:
        self._nerve_abort_event(event, False)
        if self._cleaned:
            return
        x, y = self._interactor.GetEventPosition()
        hit = self._pick_nerve_node(x, y)
        if hit is None:
            return
        identity, index = hit
        position = self._nerve_paths[identity]["nodes"][index].copy()
        normal = np.asarray(self.renderer.GetActiveCamera().GetDirectionOfProjection(), dtype=float)
        anchor = self._display_on_plane(x, y, position, normal)
        if anchor is None:
            return
        self._nerve_drag = {"path_id": identity, "node_index": index,
                            "origin": position.copy(), "normal": normal,
                            "offset": position - anchor, "position": position.copy()}
        self._nerve_abort_event(event, True)
        self._emit_nerve_drag("start", position)

    def _nerve_pointer_move(self, _caller: Any, event: str) -> None:
        self._nerve_abort_event(event, self._nerve_drag is not None)
        if self._nerve_drag is None or self._cleaned:
            return
        drag = self._nerve_drag
        position = self._display_on_plane(*self._interactor.GetEventPosition(), drag["origin"], drag["normal"])
        if position is not None:
            self._emit_nerve_drag("move", position + drag["offset"])

    def _nerve_pointer_release(self, _caller: Any, event: str) -> None:
        self._nerve_abort_event(event, self._nerve_drag is not None)
        if self._nerve_drag is None or self._cleaned:
            return
        drag = self._nerve_drag
        position = self._display_on_plane(*self._interactor.GetEventPosition(), drag["origin"], drag["normal"])
        self._emit_nerve_drag("end", drag["position"] if position is None else position + drag["offset"])
        self._nerve_drag = None

    def _new_overlay_handle(self, radius):
        sphere = vtkSphereSource()
        sphere.SetRadius(float(radius))
        sphere.SetThetaResolution(18)
        sphere.SetPhiResolution(14)
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().LightingOff()
        self._node_renderer.AddActor(actor)
        return sphere, actor

    @property
    def selected_implant_id(self):
        return self._selected_implant_id

    @selected_implant_id.setter
    def selected_implant_id(self, value):
        self._selected_implant_id = None if value is None else str(value)
        self._update_implant_handles()
        self._queue_render()

    def set_implant_editing(self, enabled, selected_implant_id=None):
        if selected_implant_id is not None:
            self._selected_implant_id = str(selected_implant_id)
        self._implant_edit_enabled = bool(enabled)
        if not self._implant_edit_enabled and self._implant_drag is not None:
            drag = self._implant_drag
            self._emit_implant_drag("end", drag["center"], drag["axis"])
            self._implant_drag = None
            self._interactor.InvokeEvent(vtkCommand.EndInteractionEvent)
        self._update_implant_handles()
        self._queue_render()

    def _set_implant_transform(self, record, center, axis):
        center, axis = np.asarray(center, dtype=float), np.asarray(axis, dtype=float)
        reference = np.eye(3)[int(np.argmin(np.abs(axis)))]
        x_axis = np.cross(axis, reference)
        x_axis /= np.linalg.norm(x_axis)
        basis = np.column_stack((x_axis, axis, np.cross(x_axis, axis)))
        matrix = record["matrix"]
        for row in range(3):
            for col in range(3):
                matrix.SetElement(row, col, float(basis[row, col]))
            matrix.SetElement(row, 3, float(center[row]))
        record["center"], record["axis"] = center.copy(), axis.copy()
        record["handles"]["center"][0].SetCenter(*center)
        record["handles"]["axis"][0].SetCenter(*(center + axis * record["length"] / 2))
        record["axis_line"].SetPoint1(*center)
        record["axis_line"].SetPoint2(*(center + axis * record["length"] / 2))

    def _update_implant_handles(self):
        self._implant_picker.InitializePickList()
        self._implant_handle_lookup.clear()
        for identity, record in self._implants.items():
            selected = identity == self._selected_implant_id
            record["actor"].GetProperty().SetOpacity(.95 if selected else .80)
            shown = self._implant_edit_enabled and selected and not self._single_image
            record["axis_actor"].SetVisibility(shown)
            for kind, (_, actor) in record["handles"].items():
                actor.SetVisibility(shown)
                actor.SetPickable(shown)
                if shown:
                    self._implant_picker.AddPickList(actor)
                    self._implant_handle_lookup[actor] = (identity, kind)

    def set_implants(self, implants: list[dict[str, Any]], selected_implant_id=None,
                     edit_enabled=None) -> None:
        """Update cached finite cylinders and foreground handles in LPS/mm.

        Existing positional calls retain selection/edit mode. IDs are stable
        application IDs; legacy callers receive an index-based display ID.
        No image or texture is rebuilt when a cylinder moves.
        """
        if QThread.currentThread() != self.thread():
            raise RuntimeError("Implant geometry must be updated on the GUI thread.")
        if self._single_image:
            implants = []
        if not isinstance(implants, (list, tuple)) or len(implants) > 200:
            raise ValueError("A bounded list of implant cylinders is required.")
        prepared = {}
        for index, implant in enumerate(implants):
            center = np.asarray(implant["center"], dtype=float)
            axis = np.asarray(implant["axis"], dtype=float)
            diameter, length = float(implant["diameter"]), float(implant["length"])
            norm = float(np.linalg.norm(axis))
            identity = str(implant.get("id", f"legacy-{index}"))
            if (center.shape != (3,) or axis.shape != (3,) or not np.isfinite(center).all()
                    or not np.isfinite(axis).all() or not np.isfinite(norm) or norm < 1e-9
                    or not np.isfinite(diameter + length) or diameter <= 0 or length <= 0
                    or not identity or identity in prepared):
                raise ValueError("Implants need unique IDs and a finite LPS center, axis, diameter, and length.")
            prepared[identity] = (center.copy(), axis / norm, diameter, length)
        if selected_implant_id is not None:
            self._selected_implant_id = str(selected_implant_id)
        if edit_enabled is not None:
            self._implant_edit_enabled = bool(edit_enabled)
        if self._selected_implant_id not in prepared:
            self._selected_implant_id = next(iter(prepared), None)
        if self._implant_drag is not None and (self._implant_drag["implant_id"] not in prepared or not self._implant_edit_enabled):
            drag = self._implant_drag
            self._emit_implant_drag("end", drag["center"], drag["axis"])
            self._implant_drag = None
            self._interactor.InvokeEvent(vtkCommand.EndInteractionEvent)
        for identity in set(self._implants) - set(prepared):
            record = self._implants.pop(identity)
            self.renderer.RemoveActor(record["actor"])
            self._node_renderer.RemoveActor(record["axis_actor"])
            for _, actor in record["handles"].values():
                self._node_renderer.RemoveActor(actor)
        for identity, (center, axis, diameter, length) in prepared.items():
            record = self._implants.get(identity)
            if record is None:
                source = vtkCylinderSource()
                source.SetResolution(48)
                source.CappingOn()
                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(source.GetOutputPort())
                actor, matrix = vtkActor(), vtkMatrix4x4()
                actor.SetMapper(mapper)
                actor.SetUserMatrix(matrix)
                actor.GetProperty().SetColor(.96, .65, .21)
                actor.GetProperty().SetSpecular(.4)
                self.renderer.AddActor(actor)
                handles = {kind: self._new_overlay_handle(.5) for kind in ("center", "axis")}
                handles["center"][1].GetProperty().SetColor(.98, .82, .44)
                handles["axis"][1].GetProperty().SetColor(.55, .78, .98)
                line = vtkLineSource()
                line_mapper = vtkPolyDataMapper()
                line_mapper.SetInputConnection(line.GetOutputPort())
                line_actor = vtkActor()
                line_actor.SetMapper(line_mapper)
                line_actor.GetProperty().SetColor(.55, .78, .98)
                line_actor.GetProperty().SetLineWidth(2.)
                line_actor.GetProperty().LightingOff()
                line_actor.PickableOff()
                self._node_renderer.AddActor(line_actor)
                record = {"source": source, "actor": actor, "matrix": matrix, "handles": handles,
                          "axis_line": line, "axis_actor": line_actor}
                self._implants[identity] = record
            record["diameter"], record["length"] = diameter, length
            record["source"].SetRadius(diameter / 2)
            record["source"].SetHeight(length)
            for sphere, _ in record["handles"].values():
                sphere.SetRadius(float(np.clip(diameter * .16, .35, .8)))
            self._set_implant_transform(record, center, axis)
            if self._implant_drag is not None and self._implant_drag["implant_id"] == identity:
                self._implant_drag["center"], self._implant_drag["axis"] = center.copy(), axis.copy()
        self._implant_actors = [self._implants[identity]["actor"] for identity in prepared]
        self._update_implant_handles()
        self._queue_render()

    def _pick_implant_handle(self, x, y):
        if not self._implant_edit_enabled or not self._implant_handle_lookup:
            return None
        if not self._implant_picker.Pick(float(x), float(y), 0., self._node_renderer):
            return None
        return self._implant_handle_lookup.get(self._implant_picker.GetActor())

    def _event_abort(self, observers, event, handled):
        event_id = vtkCommand.GetEventIdFromString(event)
        self._interactor.GetCommand(observers[event_id]).SetAbortFlag(bool(handled))

    def _emit_implant_drag(self, phase, center, axis):
        drag = self._implant_drag
        drag["center"], drag["axis"] = np.asarray(center).copy(), np.asarray(axis).copy()
        record = self._implants.get(drag["implant_id"])
        if record is not None:
            self._set_implant_transform(record, center, axis)
        if phase == "end":
            self._implant_drag = None
        self.implant_dragged.emit({"phase": phase, "implant_id": drag["implant_id"],
                                   "center_lps": drag["center"].copy(), "axis_lps": drag["axis"].copy()})
        self._queue_render()

    def _implant_pointer_press(self, _caller, event):
        self._event_abort(self._implant_observers, event, False)
        if self._cleaned:
            return
        x, y = self._interactor.GetEventPosition()
        hit = self._pick_implant_handle(x, y)
        if hit is None:
            return
        identity, kind = hit
        record = self._implants[identity]
        center, axis = record["center"].copy(), record["axis"].copy()
        point = center if kind == "center" else center + axis * record["length"] / 2
        normal = np.asarray(self.renderer.GetActiveCamera().GetDirectionOfProjection())
        anchor = self._display_on_plane(x, y, point, normal)
        if anchor is None:
            return
        self._implant_drag = {"implant_id": identity, "kind": kind, "origin": point.copy(),
                              "normal": normal.copy(), "offset": point - anchor,
                              "initial_center": center.copy(), "initial_axis": axis.copy(),
                              "center": center, "axis": axis}
        self._event_abort(self._implant_observers, event, True)
        self._interactor.InvokeEvent(vtkCommand.StartInteractionEvent)
        self._emit_implant_drag("start", center, axis)

    def _implant_pointer_position(self):
        drag = self._implant_drag
        point = self._display_on_plane(*self._interactor.GetEventPosition(), drag["origin"], drag["normal"])
        if point is None:
            return drag["center"], drag["axis"]
        point += drag["offset"]
        if drag["kind"] == "center":
            return point, drag["initial_axis"]
        axis = point - drag["initial_center"]
        norm = float(np.linalg.norm(axis))
        return drag["initial_center"], axis / norm if norm > 1e-8 else drag["axis"]

    def _implant_pointer_move(self, _caller, event):
        self._event_abort(self._implant_observers, event, self._implant_drag is not None)
        if self._implant_drag is not None and not self._cleaned:
            self._emit_implant_drag("move", *self._implant_pointer_position())

    def _implant_pointer_release(self, _caller, event):
        self._event_abort(self._implant_observers, event, self._implant_drag is not None)
        if self._implant_drag is None or self._cleaned:
            return
        self._emit_implant_drag("end", *self._implant_pointer_position())
        self._implant_drag = None
        self._interactor.InvokeEvent(vtkCommand.EndInteractionEvent)

    def set_clearance_measurement(self, result):
        """Update one persistent foreground ruler, without touching textures."""
        if QThread.currentThread() != self.thread():
            raise RuntimeError("Clearance geometry must be updated on the GUI thread.")
        if result is None or result.get("distance_mm") is None:
            signature = None
        else:
            first = np.asarray(result["implant_point_lps"], dtype=float)
            second = np.asarray(result["canal_point_lps"], dtype=float)
            distance = float(result["distance_mm"])
            if first.shape != (3,) or second.shape != (3,) or not np.isfinite(np.r_[first, second, distance]).all():
                raise ValueError("A clearance ruler needs finite LPS endpoints and distance.")
            status = str(result.get("status", "unreviewed"))
            geometry = str(result.get("geometry_status", status))
            signature = (*first, *second, distance, status, geometry)
        if signature == self._clearance_signature:
            return
        self._clearance_signature = signature
        shown = signature is not None and not self._single_image
        self._clearance_actor.SetVisibility(shown)
        self._clearance_label.SetVisibility(shown)
        for _, actor in self._clearance_handles:
            actor.SetVisibility(shown)
        if not shown:
            self._clearance_measurement = None
            self._queue_render()
            return
        color = ((.95, .40, .43) if distance < 0 or geometry == "overlap" else
                 ((.91, .68, .35) if geometry == "within_margin" else (.55, .72, .84)))
        if status == "unreviewed" and distance >= 0:
            color = (.85, .75, .52)
        self._clearance_line.SetPoint1(*first)
        self._clearance_line.SetPoint2(*second)
        self._clearance_actor.GetProperty().SetColor(*color)
        for point, (sphere, actor) in zip((first, second), self._clearance_handles):
            sphere.SetCenter(*point)
            actor.GetProperty().SetColor(*color)
        self._clearance_label.SetPosition(*((first + second) / 2))
        self._clearance_label.SetInput(f"{distance:.3f} mm · {status.replace('_', ' ')}")
        self._clearance_label.GetTextProperty().SetColor(*color)
        self._clearance_measurement = {"implant_point_lps": first.copy(), "canal_point_lps": second.copy(), "distance_mm": distance, "status": status}
        self._queue_render()

    def set_free_rotation(self, enabled=True):
        """Enable the upright fixed-anchor orbit; native pan/zoom remain."""
        self._free_rotation_enabled = bool(enabled)
        if self._camera_drag is not None:
            self._camera_drag = None
            self._interactor.InvokeEvent(vtkCommand.EndInteractionEvent)

    def _camera_pointer_press(self, _caller, event):
        self._event_abort(self._camera_observers, event, False)
        if self._nerve_drag is not None or self._implant_drag is not None:
            self._event_abort(self._camera_observers, event, True)
            return
        if self._cleaned or not self._free_rotation_enabled:
            return
        camera = self.renderer.GetActiveCamera()
        position, focus = np.asarray(camera.GetPosition()), np.asarray(camera.GetFocalPoint())
        direction = focus - position
        if np.linalg.norm(direction) < 1e-8:
            return
        direction /= np.linalg.norm(direction)
        viewport = tuple(max(1, value) for value in self._window.GetSize())
        x, y = self._interactor.GetEventPosition()
        self._camera_drag = {"position": position.copy(), "focus": focus.copy(),
                             "up": np.asarray(camera.GetViewUp()).copy(),
                             "viewport": viewport, "start": (x, y)}
        self._event_abort(self._camera_observers, event, True)
        self._interactor.InvokeEvent(vtkCommand.StartInteractionEvent)

    def _apply_camera_drag(self):
        from .camera_navigation import upright_orbit
        drag = self._camera_drag
        x, y = self._interactor.GetEventPosition()
        position = upright_orbit(drag["position"], drag["focus"], drag["start"], (x, y), drag["viewport"])
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(*position)
        camera.SetFocalPoint(*drag["focus"])
        # A consistent patient-superior horizon removes accumulated camera roll.
        camera.SetViewUp(*(drag["up"] if (x, y) == drag["start"] else (0., 0., 1.)))
        self.renderer.ResetCameraClippingRange()
        self._queue_render()

    def _camera_pointer_move(self, _caller, event):
        self._event_abort(self._camera_observers, event, self._camera_drag is not None)
        if self._nerve_drag is not None or self._implant_drag is not None:
            self._event_abort(self._camera_observers, event, True)
            return
        if self._camera_drag is not None and not self._cleaned:
            self._apply_camera_drag()

    def _camera_pointer_release(self, _caller, event):
        self._event_abort(self._camera_observers, event, self._camera_drag is not None)
        if self._nerve_drag is not None or self._implant_drag is not None:
            self._event_abort(self._camera_observers, event, True)
            return
        if self._camera_drag is None or self._cleaned:
            return
        self._apply_camera_drag()
        self._camera_drag = None
        self._interactor.InvokeEvent(vtkCommand.EndInteractionEvent)

    def capture_image(self, *, clean=False) -> QImage:
        """Return native VTK pixels for screenshots containing the Qt workspace."""
        if self._volume is None:
            raise RuntimeError("Load a volume before exporting a 3D screenshot.")
        if self._single_image:
            raise RuntimeError(_SPATIAL_VOLUME_REQUIRED)
        if not self.isVisible():
            raise RuntimeError("Display the 3D view before exporting a screenshot.")
        if self._preparing:
            raise RuntimeError("The 3D proxy is still being prepared.")
        self._render()
        if not self._gpu_supported:
            raise RuntimeError("GPU volume rendering is unavailable on the active display.")
        marker_visible = bool(self.marker.GetEnabled())
        try:
            if clean and marker_visible:
                self.marker.SetEnabled(False)
                self._window.Render()
            capture = vtkWindowToImageFilter()
            capture.SetInput(self._window)
            capture.SetInputBufferTypeToRGB()
            capture.ReadFrontBufferOff()
            capture.Update()
            vtk_image = capture.GetOutput()
            width, height, _ = vtk_image.GetDimensions()
            rgb = vtk_to_numpy(vtk_image.GetPointData().GetScalars()).reshape(height, width, 3)
            rgb = np.ascontiguousarray(rgb[::-1])
            # QImage must own its storage after the VTK capture/local NumPy expires.
            return QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        finally:
            if clean and marker_visible:
                self.marker.SetEnabled(True)
                self.marker.InteractiveOff()
                self._window.Render()

    def screenshot(self, path: str | Path) -> None:
        """Save the visible render as a PNG; fail explicitly if no GPU render exists."""
        target = Path(path)
        if target.suffix.lower() != ".png":
            raise ValueError("3D screenshots must use the .png extension.")
        if not self.capture_image().save(str(target), "PNG"):
            raise OSError(f"Could not write screenshot: {target}")

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        self._nerve_drag = None
        self._implant_drag = self._camera_drag = None
        self.set_implants([])
        self.set_clearance_measurement(None)
        self.set_nerve_paths([])
        self._generation += 1
        self._pending_proxy = None
        if self._worker is not None:
            self._worker.requestInterruption()
            # Resampling checks interruption between four output slices. Never
            # destroy a running QThread or its source arrays during a close.
            self._worker.wait()
            self._worker.volume = None
            self._worker = None
        if self._initialized:
            self.marker.SetEnabled(False)
        self._interactor.RemoveObservers(vtkCommand.StartInteractionEvent)
        self._interactor.RemoveObservers(vtkCommand.EndInteractionEvent)
        for tag in self._nerve_observers.values():
            self._interactor.RemoveObserver(tag)
        for observers in (self._implant_observers, self._camera_observers):
            for tag in observers.values():
                self._interactor.RemoveObserver(tag)
        # QVTK owns a separate native-event timer and its paintEvent calls
        # Iren.Render directly, outside our coalesced/visibility gate. Stop
        # both routes before destroying the OpenGL context.
        self.vtk_widget._Timer.stop()
        self.vtk_widget.setUpdatesEnabled(False)
        self._interactor.Disable()
        self.mapper.RemoveAllInputs()
        self.segment_mapper.RemoveAllInputs()
        self._segment_data = self._segment_image = None
        self._segment_display_data = self._segment_display_lookup = None
        self._segment_display_ids = {}
        self._segment_display_remapped = False
        # QVTK normally finalizes from the parent's destroyed signal. We have
        # already finalized explicitly; disconnect that late Python callback
        # before Qt/interpreter teardown clears QVTK's Python attributes.
        self.destroyed.disconnect(self.vtk_widget.close)
        self.vtk_widget.Finalize()
        self._image = None
        self._shared_array = None
        self._volume = None
        self._render_volume = None
        self._reference_corners = None
        self._reference_center = None

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self._queue_render()

    def closeEvent(self, event: Any) -> None:
        self.cleanup()
        super().closeEvent(event)

    def _build_proxy(self) -> None:
        if self._single_image:
            self._update_info()
            return
        target = _proxy_shape(self._volume.data.shape, self._quality)
        if (self._image is not None and self._render_volume is self._volume
                and self._shared_array.shape == target):
            # A ray-quality change on a small study does not need a duplicate
            # texture upload or a fresh copy of the same scalar array.
            self._apply_sampling()
            self._update_info()
            self._queue_render()
            return
        self._generation += 1
        generation = self._generation
        if self._worker is not None:
            self._worker.requestInterruption()
        self.mapper.RemoveAllInputs()
        self.actor.SetVisibility(False)
        self._image = None
        self._render_volume = None
        self._shared_array = None
        self._gpu_supported = None
        self._gpu_reduction = (1.0, 1.0, 1.0)
        self._preparing = target != self._volume.data.shape
        if self._preparing:
            # Keep one active allocation, with only the newest request pending.
            self._pending_proxy = (generation, self._volume, self._quality)
            self._update_info()
            if self._worker is None:
                self._start_pending_proxy()
        else:
            self._pending_proxy = None
            data, spacing, proxy = _prepare_data(self._volume, self._quality)
            self._accept_prepared(data, spacing, proxy, generation)

    def _start_pending_proxy(self) -> None:
        if self._pending_proxy is None or self._cleaned:
            return
        generation, volume, quality = self._pending_proxy
        self._pending_proxy = None
        worker = _ProxyWorker(volume, quality, generation, self)
        self._worker = worker
        worker.ready.connect(self._accept_prepared)
        worker.failed.connect(self._proxy_failed)
        worker.finished.connect(self._proxy_finished)
        worker.start()

    @Slot(object, object, bool, int)
    def _accept_prepared(self, shared: np.ndarray, spacing: np.ndarray,
                         proxy: bool, generation: int) -> None:
        if self._cleaned or generation != self._generation:
            return
        image = _image_from_data(self._volume, shared, spacing)
        self.mapper.SetInputData(image)
        self._image = image
        self._render_volume = self._volume
        self._shared_array = shared
        self._proxy = proxy
        self._preparing = False
        self._apply_sampling()
        self.actor.SetVisibility(False)
        self._apply_preset()
        self._update_info()
        self._queue_render()

    def _apply_sampling(self) -> None:
        spec = _QUALITY[self._quality]
        self.mapper.SetSampleDistance(float(min(self._image.GetSpacing())) * float(spec["ray"]))
        self.mapper.SetImageSampleDistance(float(spec["image"]))
        self.mapper.SetAutoAdjustSampleDistances(self._quality != "High")

    @Slot(str, int)
    def _proxy_failed(self, message: str, generation: int) -> None:
        if self._cleaned or generation != self._generation:
            return
        self._preparing = False
        self.annotation.SetText(2, f"3D proxy preparation failed\n{message}")
        self.info = f"3D proxy preparation failed: {message}"
        self.status.emit(self.info)
        self._queue_render()

    @Slot()
    def _proxy_finished(self) -> None:
        worker = self.sender()
        if worker is self._worker:
            self._worker = None
        if worker is not None:
            worker.volume = None
            worker.deleteLater()
        if not self._cleaned:
            self._start_pending_proxy()

    def _show_section_only(self) -> None:
        self._generation += 1
        self._pending_proxy = None
        if self._worker is not None:
            self._worker.requestInterruption()
        self._preparing = False
        self._proxy = False
        self.actor.SetVisibility(False)
        self.mapper.RemoveAllInputs()
        self._image = None
        self._render_volume = None
        self._shared_array = None
        self._gpu_supported = None
        self.set_implants([])
        self.annotation.SetText(0, "")
        self._update_info()
        self._queue_render()

    def _apply_preset(self) -> None:
        hu = bool(getattr(self._volume, "is_hu", False))
        low, high = self._robust_range
        span = high - low
        if self._preset in ("Bone", "Transparent"):
            points = [-1024, 80, 180, 400, 1000, 2500, 4000] if hu else [low + span * q for q in [0, .38, .46, .57, .72, .9, 1]]
            opacity = [0, 0, .015, .12, .42, .75, .88]
            colors = [(0, 0, 0), (.22, .16, .13), (.55, .37, .26), (.83, .68, .48), (.96, .89, .72), (1, .99, .92), (1, 1, 1)]
            if self._preset == "Transparent":
                opacity = [value * .16 for value in opacity]
        elif self._preset == "Teeth":
            points = [-1024, 250, 500, 900, 1500, 2700, 4000] if hu else [low + span * q for q in [0, .57, .66, .73, .82, .94, 1]]
            opacity = [0, 0, .015, .16, .50, .85, .95]
            colors = [(0, 0, 0), (.28, .24, .2), (.62, .52, .37), (.86, .77, .58), (.98, .94, .8), (1, 1, .96), (1, 1, 1)]
        elif self._preset == "Soft tissue":
            points = [-1024, -500, -120, 40, 140, 500, 3000] if hu else [low + span * q for q in [0, .12, .24, .39, .54, .73, 1]]
            opacity = [0, 0, .015, .055, .10, .12, .18]
            colors = [(0, 0, 0), (.15, .10, .13), (.48, .24, .24), (.82, .48, .4), (.96, .75, .6), (.98, .88, .73), (1, 1, .95)]
        elif self._preset == "X-ray":
            # A grayscale composite visualization, not simulated attenuation or
            # a quantitative radiograph. Source intensities remain untouched.
            points = [-1024, -250, 80, 300, 1000, 2500, 4000] if hu else [low + span * q for q in [0, .2, .35, .5, .7, .9, 1]]
            opacity = [0, 0, .004, .015, .04, .10, .16]
            colors = [(q, q, q) for q in (0, .1, .25, .4, .65, .9, 1.)]
        else:
            points = [low, low + span * .1, high]
            opacity = [0, .01, 1]
            colors = [(0, 0, 0), (.10, .10, .10), (1, 1, 1)]
        color = vtkColorTransferFunction()
        alpha = vtkPiecewiseFunction()
        for scalar, value, rgb in zip(points, opacity, colors):
            intensity = float(scalar + self._threshold)
            color.AddRGBPoint(intensity, *rgb)
            alpha.AddPoint(intensity, value)
        padding_value = getattr(self._volume, "padding_value", None)
        if padding_value is not None:
            # Padding remains transparent under every preset/offset. Preserve
            # the intended opacity at the first valid value, and discard any
            # transfer points in the forbidden padding-to-anatomy interval.
            minimum = self._range[0]
            minimum_opacity = alpha.GetValue(minimum)
            alpha.RemoveAllPoints()
            alpha.AddPoint(float(padding_value), 0.0)
            alpha.AddPoint(float(np.nextafter(minimum, -np.inf)), 0.0)
            alpha.AddPoint(minimum, minimum_opacity)
            for scalar, value in zip(points, opacity):
                intensity = float(scalar + self._threshold)
                if intensity > minimum:
                    alpha.AddPoint(intensity, value)
        self.property.SetColor(color)
        self.property.SetScalarOpacity(alpha)
        self.property.SetShade(self._preset not in ("MIP", "X-ray"))
        if self._preset == "MIP":
            self.mapper.SetBlendModeToMaximumIntensity()
        else:
            self.mapper.SetBlendModeToComposite()
        self._update_annotation()

    def _update_annotation(self) -> None:
        if self._volume is None:
            return
        self.annotation.SetText(0, "")
        if self._single_image:
            self.annotation.SetText(2, "3D rendering requires a spatial volume\nSingle 2D image available in section view")
            return
        if self._preparing:
            self.annotation.SetText(2, f"Preparing {self._quality.lower()} 3D proxy…\nFull-resolution MPR remains available")
            return
        if self._shared_array is None:
            return
        if self._gpu_supported is False:
            self.annotation.SetText(2, "GPU volume rendering is unavailable\nCheck the graphics driver / active display.")
            return
        units = "HU" if getattr(self._volume, "is_hu", False) else "Relative scalar values (not HU)"
        proxy = "3D proxy" if self._proxy else "Full volume"
        shape = " × ".join(str(v) for v in self._shared_array.shape[::-1])
        if self._annotation_detail:
            self.annotation.SetText(2, f"{self._preset} · {self._quality}\n{proxy}: {shape}\n{units}")
            self.annotation.SetText(0, "LPS patient coordinates\nDrag to turn and tilt · Middle drag to pan · Wheel to zoom")
        else:
            self.annotation.SetText(2, f"3D anatomy · {self._preset}")

    def _update_info(self) -> None:
        if self._volume is None:
            return
        if self._single_image:
            self.info = _SPATIAL_VOLUME_REQUIRED
            self._update_annotation()
            self.status.emit(self.info)
            return
        if self._preparing:
            self.info = f"Preparing {self._quality.lower()} 3D proxy in background · full-resolution MPR remains available"
            self._update_annotation()
            self.status.emit(self.info)
            return
        if self._shared_array is None:
            return
        gpu = "GPU pending" if self._gpu_supported is None else ("GPU ray casting" if self._gpu_supported else "GPU unavailable")
        if self._gpu_supported and self.gpu_name:
            gpu += f" ({self.gpu_name})"
        units = "HU" if getattr(self._volume, "is_hu", False) else "relative values, not HU"
        shape = " × ".join(str(v) for v in self._shared_array.shape[::-1])
        detail = f"3D proxy {shape}; original slices remain full resolution" if self._proxy else f"Full-resolution 3D {shape}"
        if min(self._gpu_reduction) < 0.999:
            detail += "; GPU texture reduced further to fit available memory"
        self.info = f"{gpu} · {detail} · {self._preset} · {units} · mapper cap 2 GiB / 35%"
        self._update_annotation()
        self.status.emit(self.info)

    def _queue_render(self) -> None:
        if self._cleaned or self._render_pending:
            return
        self._render_pending = True
        # Collapse bursts of crosshair/slider updates into at most one render
        # per display frame. VTK camera interaction keeps its native event path.
        QTimer.singleShot(16, self._render)

    def _render(self) -> None:
        self._render_pending = False
        if self._cleaned or self._rendering or not self.isVisible() or not self.vtk_widget.isVisible():
            return
        top = self.window()
        native = top.windowHandle()
        if top.isMinimized() or native is None or not native.isExposed():
            return
        self._rendering = True
        try:
            if not self._initialized:
                self._interactor.Initialize()
                self.marker.SetEnabled(True)
                self.marker.InteractiveOff()
                self._initialized = True
            if self._image is not None and self._gpu_supported is None:
                # Establish an OpenGL context with the volume hidden before the
                # capability probe, so unsupported rendering is never reported as GPU.
                self.actor.SetVisibility(False)
                self._window.Render()
                self._gpu_supported = bool(self.mapper.IsRenderSupported(self._window, self.property))
                self.actor.SetVisibility(self._gpu_supported and self._volume_visible)
                if self._gpu_supported:
                    for line in self._window.ReportCapabilities().splitlines():
                        if line.startswith("OpenGL renderer string:"):
                            self.gpu_name = line.split(":", 1)[1].strip()
                            break
                    ratio = [1.0, 1.0, 1.0]
                    self.mapper.GetReductionRatio(ratio)
                    self._gpu_reduction = tuple(float(v) for v in ratio)
                self._update_info()
            self._window.Render()
        except Exception as error:
            self._gpu_supported = False
            self.actor.SetVisibility(False)
            self.info = f"3D rendering failed: {error}"
            self.status.emit(self.info)
        finally:
            self._rendering = False

    def _interaction_start(self, *_args: Any) -> None:
        if self._volume is not None:
            self.mapper.SetAutoAdjustSampleDistances(True)
            self.mapper.SetImageSampleDistance(2.0)

    def _interaction_end(self, *_args: Any) -> None:
        spec = _QUALITY[self._quality]
        self.mapper.SetAutoAdjustSampleDistances(self._quality != "High")
        self.mapper.SetImageSampleDistance(float(spec["image"]))
        self._queue_render()
