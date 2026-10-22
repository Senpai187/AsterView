"""Mouse-to-volume integration checks for our linked MPR workstation."""

import time

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from asterview.app import MainWindow
from asterview.fast_reslice import FastReslicer
from asterview.mpr_geometry import plane_frame, plane_normal, rotate_about_view


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def wait_until(app, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(.001)
    raise AssertionError("Qt navigation work did not reach its expected state")


def settle(app, window):
    wait_until(app, lambda: not window.render_timer.isActive()
               and window.slice_pending is None and window.slice_job is None)
    app.processEvents()


@pytest.fixture
def viewer(qt_app):
    window = MainWindow()
    window.load_demo()
    settle(qt_app, window)
    for view in window.mpr_views.values():
        view.resize(640, 560)
        # Hidden fixtures do not receive the visible widget's native resize
        # delivery; explicitly exercise the same sampling notification.
        view.sampling_changed.emit()
    settle(qt_app, window)
    yield window
    window._wheel_settle.stop()
    if window._interaction_active:
        window.end_interaction()
    settle(qt_app, window)
    window.close()
    qt_app.processEvents()


def mouse(view, event_type, position, button, buttons):
    event = QMouseEvent(event_type, QPointF(position), QPointF(position), button,
                        buttons, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(view, event)


def press(view, position, button=Qt.MouseButton.LeftButton):
    mouse(view, QEvent.Type.MouseButtonPress, position, button, button)


def move(view, position, button=Qt.MouseButton.LeftButton):
    mouse(view, QEvent.Type.MouseMove, position, Qt.MouseButton.NoButton, button)


def release(view, position, button=Qt.MouseButton.LeftButton):
    mouse(view, QEvent.Type.MouseButtonRelease, position, button, Qt.MouseButton.NoButton)


def record_sampling(monkeypatch):
    samples = []
    original = FastReslicer.sample_plane

    def recorded(self, *args, **kwargs):
        samples.append({"shape": kwargs["shape"], "slab": kwargs["slab_mm"]})
        return original(self, *args, **kwargs)

    monkeypatch.setattr(FastReslicer, "sample_plane", recorded)
    return samples


def test_center_drag_accumulates_all_mouse_moves_and_updates_3d_crosshair(qt_app, viewer):
    view = viewer.mpr_views["Axial"]
    original = viewer.crosshair.copy()
    center = view.world_to_screen(original)
    final = center + QPointF(24, -12)
    expected = view.screen_to_world(final) - view.screen_to_world(center)
    press(view, center)
    assert viewer._interaction_active
    for offset in (QPointF(8, -4), QPointF(16, -8), QPointF(24, -12)):
        move(view, center + offset)
    np.testing.assert_allclose(viewer.crosshair - original, expected, atol=1e-10)
    np.testing.assert_allclose(viewer.volume_view._reference_center, viewer.crosshair)
    for linked_view in viewer.mpr_views.values():
        np.testing.assert_allclose(linked_view.crosshair, viewer.crosshair)
    release(view, final)
    assert not viewer._interaction_active
    settle(qt_app, viewer)
    np.testing.assert_allclose(viewer.volume_view._cursor_source.GetCenter(), viewer.crosshair)


def test_colored_line_drag_moves_only_its_target_plane(qt_app, viewer):
    viewer.preview_enabled.setChecked(False)
    view = viewer.mpr_views["Axial"]
    original = viewer.crosshair.copy()
    rotation = viewer.rotation()
    reference, direction, handle = next(item for item in view._reference_geometry()
                                        if item[0]["name"] == "Sagittal")
    center = view.world_to_screen(original)
    start = center + (handle - center) * .55
    assert view.reference_hit(start) == ("line", "Sagittal")
    right, up = plane_frame("Axial", rotation)
    target_normal = plane_normal("Sagittal", rotation)
    screen_normal = QPointF(float(target_normal @ right), -float(target_normal @ up))
    end = start + screen_normal * 15
    delta = view.screen_to_world(end) - view.screen_to_world(start)
    expected = target_normal * float(delta @ target_normal)
    press(view, start)
    move(view, start + screen_normal * 5)
    move(view, start + screen_normal * 10)
    move(view, end)
    np.testing.assert_allclose(viewer.crosshair - original, expected, atol=1e-10)
    for name in ("Axial", "Coronal"):
        assert float((viewer.crosshair - original) @ plane_normal(name, rotation)) == pytest.approx(0, abs=1e-10)
    release(view, end)
    settle(qt_app, viewer)


def test_hidden_reference_lines_do_not_capture_right_drag_for_rotation(qt_app, viewer):
    viewer.lines_visible.setChecked(False)
    view = viewer.mpr_views["Axial"]
    center = view.world_to_screen(viewer.crosshair)
    original_rotation = viewer.rotation().copy()
    original_window = view.center, view.window_width
    press(view, center, Qt.MouseButton.RightButton)
    assert view._drag_kind is None
    assert not viewer._interaction_active
    move(view, center + QPointF(12, -6), Qt.MouseButton.RightButton)
    np.testing.assert_allclose(viewer.rotation(), original_rotation, atol=1e-12)
    assert (view.center, view.window_width) == original_window
    release(view, center + QPointF(12, -6), Qt.MouseButton.RightButton)


@pytest.mark.parametrize("view_name,button,handle,angles", [
    ("Axial", Qt.MouseButton.LeftButton, True, (0, 0, 0)),
    ("Axial", Qt.MouseButton.RightButton, False, (0, 0, 0)),
    ("Coronal", Qt.MouseButton.LeftButton, True, (17.3, -28.2, 43.7)),
    ("Sagittal", Qt.MouseButton.RightButton, False, (-23.1, 36.4, 18.8)),
])
def test_pointer_rotation_corrects_other_planes_while_view_stays_fixed(
        qt_app, viewer, view_name, button, handle, angles):
    for spin, angle in zip(viewer.rotation_spins, angles):
        spin.setValue(angle)
    settle(qt_app, viewer)
    view = viewer.mpr_views[view_name]
    original_rotation = viewer.rotation().copy()
    original_frame = plane_frame(view_name, original_rotation)
    original_normal = plane_normal(view_name, original_rotation)
    original_crosshair = viewer.crosshair.copy()
    reference, direction, outer = view._reference_geometry()[0]
    center = view.world_to_screen(viewer.crosshair)
    start = outer if handle else center + (outer - center) * .55
    assert view.reference_hit(start)[0] == ("rotate" if handle else "line")
    start_world = view.screen_to_world(start) - viewer.crosshair
    expected_degrees = 0.0
    previous_world = start_world
    press(view, start, button)
    assert view._drag_kind == "rotate"
    # Negative screen rotation is positive in the anatomical right/up frame.
    delta = start - center
    for radians in (.07, .14, .21):
        point = center + QPointF(delta.x() * np.cos(radians) + delta.y() * np.sin(radians),
                                 -delta.x() * np.sin(radians) + delta.y() * np.cos(radians))
        current_world = view.screen_to_world(point) - viewer.crosshair
        right, up = original_frame
        before = np.arctan2(previous_world @ up, previous_world @ right)
        after = np.arctan2(current_world @ up, current_world @ right)
        expected_degrees += np.rad2deg((after - before + np.pi) % (2 * np.pi) - np.pi)
        move(view, point, button)
        previous_world = current_world
    expected_rotation = rotate_about_view(original_rotation, view_name, expected_degrees)
    np.testing.assert_allclose(viewer.rotation(), expected_rotation, atol=1e-10)
    np.testing.assert_allclose(plane_normal(view_name, viewer.rotation()), original_normal, atol=1e-10)
    np.testing.assert_allclose(plane_frame(view_name, viewer.rotation()), original_frame, atol=1e-10)
    np.testing.assert_allclose(viewer.crosshair, original_crosshair, atol=1e-10)
    release(view, point, button)
    settle(qt_app, viewer)
    np.testing.assert_allclose(view.frame[1:3], original_frame, atol=1e-10)
    np.testing.assert_allclose(viewer.volume_view._reference_rotation, viewer.rotation(), atol=1e-10)


def test_drag_preserves_resolution_and_selected_slab_until_release(qt_app, viewer, monkeypatch):
    view = viewer.mpr_views["Axial"]
    full_shape = view.data.shape
    assert max(full_shape) > 224
    viewer.slab.setValue(2)
    settle(qt_app, viewer)
    samples = record_sampling(monkeypatch)
    center = view.world_to_screen(viewer.crosshair)
    press(view, center)
    move(view, center + QPointF(10, -4))
    settle(qt_app, viewer)
    assert viewer._interaction_active
    assert samples
    assert all(max(item["shape"]) > 224 and item["slab"] == 2 for item in samples)
    assert all(max(item.data.shape) > 224 for item in viewer.mpr_views.values())
    assert view.data.shape == full_shape
    assert "Interactive preview" not in view.note
    release(view, center + QPointF(10, -4))
    settle(qt_app, viewer)
    assert view.data.shape == full_shape
    assert "Interactive preview" not in view.note


def test_render_deadline_is_not_deferred_by_a_stream_of_new_events(qt_app, viewer):
    viewer.render_timer.setInterval(40)
    viewer.schedule_render()
    deadline = time.monotonic() + .025
    while time.monotonic() < deadline:
        qt_app.processEvents()
        time.sleep(.001)
    before = viewer.render_timer.remainingTime()
    assert 0 < before < 35
    viewer.schedule_render()
    after = viewer.render_timer.remainingTime()
    assert 0 <= after <= before + 1
    revision = viewer.slice_revision
    # Rescheduling faster than the timeout must still allow an actual frame.
    deadline = time.monotonic() + .10
    while time.monotonic() < deadline and viewer.slice_revision == revision:
        viewer.schedule_render()
        qt_app.processEvents()
        time.sleep(.002)
    assert viewer.slice_revision > revision
    settle(qt_app, viewer)


def test_geometry_cache_retains_unmoved_images_and_reslices_only_changed_plane(qt_app, viewer, monkeypatch):
    viewer.preview_enabled.setChecked(False)
    originals = {name: view.data for name, view in viewer.mpr_views.items()}
    samples = record_sampling(monkeypatch)
    viewer.translate_planes("Axial", {"delta": np.array((3., 0., 0.)), "target": "Sagittal"})
    settle(qt_app, viewer)
    assert viewer.mpr_views["Axial"].data is originals["Axial"]
    assert viewer.mpr_views["Coronal"].data is originals["Coronal"]
    assert viewer.mpr_views["Sagittal"].data is not originals["Sagittal"]
    assert len(samples) == 1
    np.testing.assert_allclose(viewer.volume_view._reference_center, viewer.crosshair)


def test_wheel_navigation_keeps_full_detail_during_input_and_idle(qt_app, viewer, monkeypatch):
    samples = record_sampling(monkeypatch)
    for _ in range(8):
        viewer.scroll_plane("Axial", .25)
        # Let the fixed render deadline fire while the separate input-idle
        # timer continues to be extended by the wheel stream.
        deadline = time.monotonic() + .012
        while time.monotonic() < deadline:
            qt_app.processEvents()
            time.sleep(.001)
    assert viewer._interaction_active
    assert samples
    assert all(max(item["shape"]) > 224 for item in samples)
    wait_until(qt_app, lambda: not viewer._interaction_active)
    settle(qt_app, viewer)
    assert all(max(view.data.shape) > 224 for view in viewer.mpr_views.values())


def test_slow_full_detail_sampler_presents_frames_during_continuous_wheel(qt_app, viewer, monkeypatch):
    """Input faster than reslicing must not cancel every frame until release."""
    view = viewer.mpr_views["Axial"]
    original_sample = FastReslicer.sample_plane
    original_present = view.set_image
    presented = []

    def slow_sample(self, *args, **kwargs):
        time.sleep(.06)
        return original_sample(self, *args, **kwargs)

    def present(image, frame=None, *args, **kwargs):
        presented.append((frame[0].copy(), image.shape))
        return original_present(image, frame, *args, **kwargs)

    monkeypatch.setattr(FastReslicer, "sample_plane", slow_sample)
    monkeypatch.setattr(view, "set_image", present)
    viewer.render_timer.setInterval(8)
    until = time.monotonic() + .45
    while time.monotonic() < until:
        viewer.scroll_plane("Axial", .02)
        next_input = time.monotonic() + .005
        while time.monotonic() < next_input:
            qt_app.processEvents()
            time.sleep(.001)
    assert viewer._interaction_active
    assert len(presented) >= 2, "Continuous input starved the displayed section"
    expected_shape = view.preferred_sample_shape(viewer._mpr_sizes["Axial"])
    assert all(shape == expected_shape for _, shape in presented)
    wait_until(qt_app, lambda: not viewer._interaction_active)
    settle(qt_app, viewer)
    normal = plane_normal("Axial", viewer.rotation())
    expected = viewer.volume.center + normal * np.dot(viewer.crosshair - viewer.volume.center, normal)
    np.testing.assert_allclose(view.frame[0], expected, atol=1e-9)
    assert viewer.slice_queue.active is None and viewer.slice_queue.pending is None
