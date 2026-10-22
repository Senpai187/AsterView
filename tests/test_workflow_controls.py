"""Exercise actual pointer editing across the dental workspaces."""
import time

import numpy as np
import pytest
from PySide6.QtCore import Qt, QEvent, QPointF
from PySide6.QtGui import QMouseEvent, QKeyEvent
from PySide6.QtWidgets import QApplication

from asterview.app import MainWindow
from asterview.mpr_geometry import plane_normal


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def settle(app, window):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        app.processEvents()
        if (not window.render_timer.isActive() and not window._wheel_settle.isActive()
                and window.slice_pending is None and window.slice_job is None):
            return
        time.sleep(.002)
    raise AssertionError("Workflow rendering did not settle")


@pytest.fixture
def viewer(app):
    window = MainWindow()
    window.load_demo()
    settle(app, window)
    yield window
    settle(app, window)
    window.close()
    app.processEvents()


def mouse(view, event_type, pos, button=Qt.MouseButton.LeftButton):
    buttons = Qt.MouseButton.NoButton if event_type == QEvent.Type.MouseButtonRelease else button
    event_button = Qt.MouseButton.NoButton if event_type == QEvent.Type.MouseMove else button
    QApplication.sendEvent(view, QMouseEvent(event_type, pos, pos, event_button, buttons,
                                            Qt.KeyboardModifier.NoModifier))


def prepare_arch(app, viewer):
    viewer.tabs.setCurrentIndex(1)
    viewer.arch_points = [viewer.volume.center + d for d in ([-25, 5, 0], [0, -25, 0], [25, 5, 0])]
    viewer.schedule_render()
    settle(app, viewer)
    viewer.arch_view.resize(640, 560)
    return viewer.arch_view


def test_arch_drag_moves_one_control_point_without_adding_or_changing_depth(app, viewer):
    view = prepare_arch(app, viewer)
    old = np.array(viewer.arch_points)
    start = view.world_to_screen(old[1])
    end = start + QPointF(12, -7)
    expected_delta = view.screen_to_world(end) - view.screen_to_world(start)
    mouse(view, QEvent.Type.MouseButtonPress, start)
    assert viewer._interaction_active
    mouse(view, QEvent.Type.MouseMove, end)
    mouse(view, QEvent.Type.MouseButtonRelease, end)
    assert not viewer._interaction_active
    assert len(viewer.arch_points) == 3
    np.testing.assert_allclose(viewer.arch_points[1], old[1] + expected_delta)
    np.testing.assert_allclose(np.array(viewer.arch_points)[[0, 2]], old[[0, 2]])
    assert viewer.pano_result is None


def test_arch_right_click_finishes_then_drag_delete_remain_available(app, viewer):
    view = prepare_arch(app, viewer)
    pos = view.world_to_screen(viewer.arch_points[1])
    mouse(view, QEvent.Type.MouseButtonPress, pos, Qt.MouseButton.RightButton)
    mouse(view, QEvent.Type.MouseButtonRelease, pos, Qt.MouseButton.RightButton)
    assert not viewer.arch_toggle.isChecked()
    assert view.mode == "Edit arch"
    mouse(view, QEvent.Type.MouseButtonPress, pos)
    mouse(view, QEvent.Type.MouseButtonRelease, pos)
    QApplication.sendEvent(view, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete,
                                         Qt.KeyboardModifier.NoModifier))
    assert len(viewer.arch_points) == 2
    assert view.selected_point is None


def test_arch_mode_switch_does_not_append_duplicate_when_grabbing_existing_point(app, viewer):
    view = prepare_arch(app, viewer)
    viewer.tool_combo.setCurrentText("Pan")
    assert view.mode == "Pan"
    viewer.tool_combo.setCurrentText("Navigate")
    assert view.mode == "Add arch point"
    position = view.world_to_screen(viewer.arch_points[1])
    mouse(view, QEvent.Type.MouseButtonPress, position)
    mouse(view, QEvent.Type.MouseMove, position + QPointF(5, 3))
    mouse(view, QEvent.Type.MouseButtonRelease, position + QPointF(5, 3))
    assert len(viewer.arch_points) == 3
    viewer.arch_toggle.setChecked(False)
    assert view.mode == "Edit arch"


def test_axial_reference_rotation_keeps_existing_arch_when_its_plane_is_unchanged(app, viewer):
    prepare_arch(app, viewer)
    original = np.array(viewer.arch_points)
    viewer.rotate_planes("Axial", 27)
    np.testing.assert_allclose(viewer.arch_points, original)
    viewer.rotate_planes("Coronal", 8)
    assert not viewer.arch_points


def test_orientation_reset_preserves_location_zoom_contrast_and_camera(app, viewer):
    viewer.crosshair += [4, -3, 2]
    viewer.rotate_planes("Coronal", 23)
    viewer.window_level.setValue(333)
    viewer.window_width.setValue(1777)
    view = viewer.mpr_views["Axial"]
    view.zoom, view.pan = 2.1, QPointF(12, -8)
    position = viewer.crosshair.copy()
    camera = viewer.volume_view.renderer.GetActiveCamera()
    camera_state = camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp()
    viewer.reset_orientation()
    np.testing.assert_allclose(viewer.rotation(), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(viewer.crosshair, position)
    assert (view.zoom, view.pan) == (2.1, QPointF(12, -8))
    assert (viewer.window_level.value(), viewer.window_width.value()) == (333, 1777)
    assert (camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp()) == camera_state


def test_tmj_locator_drag_updates_source_center_and_corrected_sections(app, viewer):
    viewer.tmj_centers = {"Right": viewer.volume.center + [-25, 10, 0],
                          "Left": viewer.volume.center + [25, 10, 0]}
    viewer.tabs.setCurrentIndex(3)
    settle(app, viewer)
    locator = viewer.tmj_locator
    locator.resize(640, 560)
    start = locator.world_to_screen(viewer.tmj_centers["Right"])
    old_left = viewer.tmj_centers["Left"].copy()
    expected = viewer.tmj_centers["Right"] + locator.screen_to_world(start + QPointF(8, 4)) - locator.screen_to_world(start)
    mouse(locator, QEvent.Type.MouseButtonPress, start)
    mouse(locator, QEvent.Type.MouseMove, start + QPointF(8, 4))
    mouse(locator, QEvent.Type.MouseButtonRelease, start + QPointF(8, 4))
    settle(app, viewer)
    np.testing.assert_allclose(viewer.tmj_centers["Right"], expected)
    np.testing.assert_allclose(viewer.tmj_centers["Left"], old_left)
    for plane in ("Sagittal", "Coronal"):
        np.testing.assert_allclose(viewer.tmj_views[("Right", plane)].frame[0], expected)


def test_oblique_wheel_motion_stops_at_boundary_without_lateral_drift(app, viewer):
    viewer.rotate_planes("Coronal", 33)
    normal = plane_normal("Axial", viewer.rotation())
    start = viewer.crosshair.copy()
    viewer.scroll_plane("Axial", 10000)
    delta = viewer.crosshair - start
    np.testing.assert_allclose(delta - normal * (delta @ normal), np.zeros(3), atol=1e-8)
    index = viewer.volume.world_to_index(viewer.crosshair)
    assert np.all(index >= -1e-6)
    assert np.all(index <= np.asarray(viewer.volume.dimensions) - 1 + 1e-6)
