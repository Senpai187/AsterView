"""Physical-coordinate tests for independent manual condylar navigation."""
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from asterview.mpr_geometry import plane_frame
from asterview.reformat import rotation_matrix
from asterview.tmj_locator import TMJLocatorView


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def locator(qt_app):
    view = TMJLocatorView()
    view.resize(720, 580)
    rotation = rotation_matrix(17.0, -23.0, 38.0)
    center = np.array([10.0, -4.0, 30.0])
    right, up = plane_frame("Axial", rotation)
    view.set_image(np.zeros((101, 141), np.float32),
                   (center, right, up, (140.0, 100.0)))
    centers = {"Right": center + rotation @ np.array([-22.0, 3.0, 4.0]),
               "Left": center + rotation @ np.array([22.0, 3.0, -8.0])}
    angles = {"Right": 35.0, "Left": -15.0}
    view.set_tmj_geometry(centers, angles, rotation)
    yield view
    view.close()
    qt_app.processEvents()


def send(view, kind, point, button, buttons):
    event = QMouseEvent(kind, QPointF(point), QPointF(point), button, buttons,
                        Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(view, event)


def drag(view, start, finish, button=Qt.MouseButton.LeftButton):
    send(view, QEvent.Type.MouseButtonPress, start, button, button)
    send(view, QEvent.Type.MouseMove, finish, Qt.MouseButton.NoButton, button)
    send(view, QEvent.Type.MouseButtonRelease, finish, button, Qt.MouseButton.NoButton)


def item(view, side):
    return next(part for part in view.locator_geometry() if part["side"] == side)


def test_move_center_preserves_depth_and_other_side_under_obliquity(locator):
    view = locator
    original = {side: center.copy() for side, center in view.tmj_centers.items()}
    rotation = view.tmj_rotation.copy()
    view.zoom, view.pan = 1.4, QPointF(13.0, -19.0)
    start = item(view, "Right")["center"]
    finish = start + QPointF(27.0, -18.0)
    expected_delta = view.screen_to_world(finish) - view.screen_to_world(start)
    changes, interactions = [], []
    view.center_changed.connect(lambda side, world: changes.append((side, world)))
    view.interaction_started.connect(lambda: interactions.append("start"))
    view.interaction_finished.connect(lambda: interactions.append("finish"))
    drag(view, start, finish)
    assert len(changes) == 1 and changes[0][0] == "Right"
    np.testing.assert_allclose(changes[0][1], original["Right"] + expected_delta)
    np.testing.assert_allclose(view.tmj_centers["Left"], original["Left"])
    assert abs(np.dot(view.tmj_centers["Right"] - original["Right"], rotation[:, 2])) < 1e-10
    np.testing.assert_allclose(view.tmj_rotation, rotation)
    assert interactions == ["start", "finish"]


@pytest.mark.parametrize("side", ["Right", "Left"])
def test_round_handle_sets_long_axis_in_corrected_patient_frame(locator, side):
    view = locator
    center = view.tmj_centers[side].copy()
    other = "Left" if side == "Right" else "Right"
    other_angle = view.tmj_angles[other]
    target_angle = 62.0 if side == "Right" else -42.0
    theta = np.deg2rad(target_angle)
    target = center + view.tmj_rotation @ np.array([12.0 * np.cos(theta), 12.0 * np.sin(theta), 0.0])
    changes = []
    view.angle_changed.connect(lambda name, angle: changes.append((name, angle)))
    drag(view, item(view, side)["handle"], view.world_to_screen(target))
    assert changes[-1][0] == side
    assert changes[-1][1] == pytest.approx(target_angle, abs=1e-9)
    assert view.tmj_angles[other] == other_angle
    np.testing.assert_array_equal(view.tmj_centers[side], center)


def test_right_drag_axis_rotates_without_translation_or_windowing(locator):
    view = locator
    geometry = item(view, "Left")
    start = geometry["center"] + (geometry["handle"] - geometry["center"]) * 0.6
    center = view.tmj_centers["Left"].copy()
    target = center + view.tmj_rotation @ np.array([8.0, -8.0, 0.0])
    windows, centers = [], []
    view.window_dragged.connect(lambda x, y: windows.append((x, y)))
    view.center_changed.connect(lambda *args: centers.append(args))
    drag(view, start, view.world_to_screen(target), Qt.MouseButton.RightButton)
    assert view.tmj_angles["Left"] == pytest.approx(-45.0)
    assert windows == centers == []
    np.testing.assert_array_equal(view.tmj_centers["Left"], center)


def test_right_drag_in_margin_rotates_nearest_axis_without_snap(locator):
    view = locator
    view.resize(1200, 400)
    start, finish = QPointF(70, 160), QPointF(70, 210)
    assert not view._rect().contains(start)
    centers = {side: center.copy() for side, center in view.tmj_centers.items()}
    old_angle, other_angle = view.tmj_angles["Right"], view.tmj_angles["Left"]
    a = view.screen_to_world(start, allow_outside=True) - centers["Right"]
    b = view.screen_to_world(finish, allow_outside=True) - centers["Right"]
    r = view.tmj_rotation
    delta = np.rad2deg(np.arctan2(b @ r[:, 1], b @ r[:, 0]) - np.arctan2(a @ r[:, 1], a @ r[:, 0]))
    expected = view._normalize_angle(old_angle + view._normalize_angle(delta))
    windows = []
    view.window_dragged.connect(lambda x, y: windows.append((x, y)))
    drag(view, start, finish, Qt.MouseButton.RightButton)
    assert view.tmj_angles["Right"] == pytest.approx(expected)
    assert view.tmj_angles["Left"] == other_angle
    assert windows == []
    for side in centers:
        np.testing.assert_array_equal(view.tmj_centers[side], centers[side])


def test_background_pick_and_measurement_tools_pass_through(locator):
    view = locator
    picks = []
    view.picked.connect(picks.append)
    point = view.world_to_screen(view.frame[0] + view.frame[1] * 45.0)
    drag(view, point, point)
    assert len(picks) == 1
    np.testing.assert_allclose(picks[0]["world"], view.screen_to_world(point))
    view.mode = "Measure"
    first = item(view, "Right")["center"]
    second = first + QPointF(25, 0)
    original = view.tmj_centers["Right"].copy()
    drag(view, first, first)
    drag(view, second, second)
    assert len(view.measurements) == 1
    np.testing.assert_array_equal(view.tmj_centers["Right"], original)


@pytest.mark.parametrize("mode", ["Pan", "Window / level"])
def test_center_does_not_steal_other_active_tools(locator, mode):
    view = locator
    view.mode = mode
    center = view.tmj_centers["Right"].copy()
    start = item(view, "Right")["center"]
    original_pan = QPointF(view.pan)
    windows = []
    view.window_dragged.connect(lambda x, y: windows.append((x, y)))
    drag(view, start, start + QPointF(15, -8))
    np.testing.assert_array_equal(view.tmj_centers["Right"], center)
    if mode == "Pan":
        assert view.pan == original_pan + QPointF(15, -8)
    else:
        # The retired drag tool must not bypass the brightness sliders.
        assert windows == []


def test_uncalibrated_images_have_no_physical_locator_controls(locator):
    view = locator
    start = item(view, "Right")["center"]
    centers = {side: point.copy() for side, point in view.tmj_centers.items()}
    view.calibrated = False
    changes = []
    view.center_changed.connect(lambda *args: changes.append(args))
    assert view.locator_geometry() == []
    drag(view, start, start + QPointF(12, 0))
    assert changes == []
    for side in centers:
        np.testing.assert_array_equal(view.tmj_centers[side], centers[side])


def test_leaving_image_during_translation_does_not_jump_when_reentering(locator):
    view = locator
    start = item(view, "Right")["center"]
    original = view.tmj_centers["Right"].copy()
    send(view, QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
    send(view, QEvent.Type.MouseMove, QPointF(-20, -20), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    send(view, QEvent.Type.MouseMove, start + QPointF(25, 5), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    np.testing.assert_array_equal(view.tmj_centers["Right"], original)
    send(view, QEvent.Type.MouseMove, start + QPointF(28, 7), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    assert np.linalg.norm(view.tmj_centers["Right"] - original) > 0
    send(view, QEvent.Type.MouseButtonRelease, start, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)


def test_removing_selected_center_finishes_active_interaction(locator):
    view = locator
    finished = []
    view.interaction_finished.connect(lambda: finished.append(True))
    start = item(view, "Right")["center"]
    send(view, QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
    view.set_tmj_geometry({"Right": None, "Left": None}, {}, np.eye(3))
    assert view._tmj_drag is None and finished == [True]
    assert view.locator_geometry() == []


def test_config_is_copied_and_invalid_geometry_is_rejected_atomically(locator):
    view = locator
    point = np.array([1.0, 2.0, 3.0])
    rotation = np.eye(3)
    view.set_tmj_geometry({"Right": point}, {"Right": 200}, rotation)
    point[:] = 99
    rotation[:] = 0
    np.testing.assert_array_equal(view.tmj_centers["Right"], [1.0, 2.0, 3.0])
    assert view.tmj_angles["Right"] == -160.0
    with pytest.raises(ValueError):
        view.set_tmj_geometry({"Right": [1, np.nan, 3]}, {}, np.eye(3))
    with pytest.raises(ValueError):
        view.set_tmj_geometry({}, {}, np.diag([1, 1, -1]))
    np.testing.assert_array_equal(view.tmj_centers["Right"], [1.0, 2.0, 3.0])


def test_locator_paints_on_native_qt_surface(locator, qt_app):
    locator.show()
    qt_app.processEvents()
    assert not locator.grab().isNull()


def test_corrected_long_axis_projects_and_azimuth_handle_preserves_tilt_without_jump(locator):
    view = locator
    side = "Right"
    rotation = view.tmj_rotation.copy()
    old_angle = view.tmj_angles[side]
    local_axis = rotation_matrix(0, 0, old_angle) @ rotation_matrix(25, -15, 20)[:, 0]
    axis = rotation @ local_axis
    view.set_tmj_geometry(view.tmj_centers, view.tmj_angles, rotation, axes={side: axis})
    geometry = item(view, side)
    expected = view.world_to_screen(view.tmj_centers[side] + axis * view.AXIS_HALF_LENGTH_MM)
    assert (geometry["handle"] - expected).manhattanLength() < 1e-8
    target_axis = rotation @ rotation_matrix(0, 0, 17) @ local_axis
    target = view.tmj_centers[side] + target_axis * view.AXIS_HALF_LENGTH_MM
    drag(view, geometry["handle"], view.world_to_screen(target))
    assert view.tmj_angles[side] == pytest.approx(old_angle + 17)
    np.testing.assert_allclose(view.tmj_axes[side], target_axis, atol=1e-10)
    assert view.tmj_axes[side] @ rotation[:, 2] == pytest.approx(axis @ rotation[:, 2])


@pytest.mark.parametrize("axes", [{"Right": [0, 0, 0]}, {"Right": [1, 2]},
                                 {"Other": [1, 0, 0]}, {"Right": [np.nan, 0, 1]}])
def test_invalid_corrected_axis_is_rejected_atomically(locator, axes):
    centers = {side: point.copy() for side, point in locator.tmj_centers.items()}
    with pytest.raises(ValueError):
        locator.set_tmj_geometry({}, {}, np.eye(3), axes=axes)
    for side, point in centers.items():
        np.testing.assert_array_equal(locator.tmj_centers[side], point)
