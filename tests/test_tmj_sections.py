"""Production TMJ signals and patient-space corrections without a GPU."""
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, QTimer
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QCheckBox, QDoubleSpinBox, QGridLayout, QWidget

from asterview.model import Volume
from asterview.reformat import rotation_matrix
from asterview.slice_view import SliceView
from asterview.workflow_controls import WorkflowControls


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class TMJHarness(QWidget, WorkflowControls):
    def __init__(self):
        super().__init__()
        self.volume = Volume(np.zeros((51, 51, 51), np.float32), (.4, .4, .6),
                             np.zeros(3), np.eye(3))
        self.tmj_centers = {"Right": self.volume.center.copy(),
                            "Left": self.volume.center + [2., 0, 0]}
        self.tmj_angles = {side: QDoubleSpinBox() for side in ("Right", "Left")}
        for item in self.tmj_angles.values():
            item.setRange(-180, 180)
            item.setValue(23)
        self.tmj_offset = QDoubleSpinBox()
        self.tmj_offset.setRange(-30, 30)
        self.tmj_offset.setDecimals(3)
        self.lines_visible = QCheckBox()
        self.lines_visible.setChecked(True)
        self._wheel_settle = QTimer(self)
        self._wheel_settle.setSingleShot(True)
        self._wheel_settle.setInterval(120)
        self.tmj_views = {}
        self.render_count = 0
        self.active = False
        layout = QGridLayout(self)
        for side_index, side in enumerate(("Right", "Left")):
            for index, plane in enumerate(("Sagittal", "Coronal")):
                view = SliceView(side + plane)
                self.tmj_views[(side, plane)] = view
                layout.addWidget(view, index, side_index)
                self.wire_tmj_section(view, side, plane)
        self.resize(1200, 1000)
        self.render_tmj_section_planes()

    def rotation(self):
        return rotation_matrix(12, -5, 8)

    def schedule_render(self, *_):
        self.render_count += 1

    def begin_interaction(self):
        self.active = True

    def end_interaction(self):
        self.active = False

    def move_tmj_center(self, side, center):
        index = self.volume.world_to_index(center)
        if np.all(index >= 0) and np.all(index <= np.asarray(self.volume.dimensions) - 1):
            self.tmj_centers[side] = np.asarray(center).copy()
            self.schedule_render()

    def draw_plane(self, view, center, right, up, size, note, *, crosshair):
        view.set_image(np.zeros((201, 201), np.float32),
                       (center, right, up, size), note=note, crosshair=crosshair)


@pytest.fixture
def viewer(app):
    window = TMJHarness()
    window.show()
    app.processEvents()
    yield window
    window._wheel_settle.stop()
    window.close()
    app.processEvents()


def send_mouse(view, kind, point, button):
    buttons = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseButtonRelease else button
    event_button = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseMove else button
    QApplication.sendEvent(view, QMouseEvent(kind, point, point, event_button, buttons,
                                            Qt.KeyboardModifier.NoModifier))


def test_tmj_right_drag_corrects_other_plane_without_rotating_active_image(viewer):
    view = viewer.tmj_views[("Right", "Sagittal")]
    previous = [item.copy() for item in view.frame[:3]]
    other_before = np.cross(*viewer.tmj_section_frame("Right", "Coronal")[1:3])
    left_before = viewer.tmj_section_rotation("Left").copy()
    source = viewer.volume.data.copy()
    center = view.world_to_screen(view.crosshair)
    start, end = center + QPointF(60, 0), center + QPointF(50, -30)
    send_mouse(view, QEvent.Type.MouseButtonPress, start, Qt.MouseButton.RightButton)
    assert viewer.active
    send_mouse(view, QEvent.Type.MouseMove, end, Qt.MouseButton.RightButton)
    send_mouse(view, QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.RightButton)
    assert not viewer.active
    assert "Right" in viewer._tmj_section_corrections
    viewer.render_tmj_section_planes()
    for current, expected in zip(view.frame[:3], previous):
        np.testing.assert_allclose(current, expected, atol=1e-10)
    assert not np.allclose(np.cross(*viewer.tmj_section_frame("Right", "Coronal")[1:3]), other_before)
    np.testing.assert_allclose(viewer.tmj_section_rotation("Left"), left_before)
    np.testing.assert_array_equal(viewer.volume.data, source)


def test_tmj_center_drag_and_wheel_use_linked_physical_coordinates(viewer):
    view = viewer.tmj_views[("Right", "Coronal")]
    old = viewer.tmj_centers["Right"].copy()
    left = viewer.tmj_centers["Left"].copy()
    start = view.world_to_screen(view.crosshair)
    end = start + QPointF(7, -3)
    delta = view.screen_to_world(end) - view.screen_to_world(start)
    send_mouse(view, QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
    send_mouse(view, QEvent.Type.MouseMove, end, Qt.MouseButton.LeftButton)
    send_mouse(view, QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.LeftButton)
    np.testing.assert_allclose(viewer.tmj_centers["Right"], old + delta)
    np.testing.assert_allclose(viewer.tmj_centers["Left"], left)
    event = QWheelEvent(start, start, QPoint(), QPoint(0, 240),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(view, event)
    assert viewer.tmj_offset.value() == .8
    assert viewer._wheel_settle.isActive()
    viewer.render_tmj_section_planes()
    for side in ("Right", "Left"):
        cursor = viewer.tmj_section_cursor(side)
        for plane in ("Sagittal", "Coronal"):
            frame = viewer.tmj_views[(side, plane)].frame
            assert abs((cursor - frame[0]) @ np.cross(frame[1], frame[2])) < 1e-10


def test_tmj_pick_with_nonzero_offset_preserves_clicked_intersection(viewer):
    viewer.tmj_offset.setValue(1.2)
    viewer.render_tmj_section_planes()
    view = viewer.tmj_views[("Right", "Coronal")]
    clicked = view.crosshair + view.frame[1] * .7 + view.frame[2] * .3
    view.picked.emit({"world": clicked})
    np.testing.assert_allclose(viewer.tmj_section_cursor("Right"), clicked)
    assert viewer.tmj_offset.value() == 1.2


def test_tmj_rotation_at_nonzero_offset_pivots_at_cursor_and_preserves_active_plane(viewer):
    viewer.tmj_offset.setValue(2.4)
    frame = viewer.tmj_section_frame("Right", "Sagittal")
    cursor = viewer.tmj_section_cursor("Right").copy()
    normal = np.cross(frame[1], frame[2])
    viewer.rotate_tmj_section("Right", "Sagittal", 31)
    updated = viewer.tmj_section_frame("Right", "Sagittal")
    np.testing.assert_allclose(viewer.tmj_section_cursor("Right"), cursor, atol=1e-10)
    np.testing.assert_allclose(updated[1:3], frame[1:3], atol=1e-10)
    assert abs(np.dot(updated[0] - frame[0], normal)) < 1e-10
