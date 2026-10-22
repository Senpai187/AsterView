"""Production MainWindow sampling/CPR wiring without showing a GPU window."""
import time

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from asterview.app import MainWindow
from asterview.reformat import curved_planar


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def settle(app, window):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        app.processEvents()
        if not window.render_timer.isActive() and window.slice_job is None and window.slice_pending is None:
            return
        time.sleep(.002)
    raise AssertionError("CPU sampling did not settle")


@pytest.fixture
def viewer(app):
    window = MainWindow()
    window.load_demo()
    settle(app, window)
    yield window
    window._wheel_settle.stop()
    window.end_interaction()
    window.pano_refresh_timer.stop()
    settle(app, window)
    window.close()
    app.processEvents()


def test_real_workstation_requests_display_sized_pixels_on_resize_and_zoom(app, viewer):
    view = viewer.mpr_views["Axial"]
    source = viewer.volume.data.copy()
    view.resize(1400, 1000)
    view.sampling_changed.emit()
    settle(app, viewer)
    expected = view.preferred_sample_shape(viewer._mpr_sizes["Axial"])
    assert view.data.shape == expected
    assert max(expected) > 768
    first = view.data.shape
    view.zoom = 1.6
    view.sampling_changed.emit()
    settle(app, viewer)
    assert view.data.shape == view.preferred_sample_shape(viewer._mpr_sizes["Axial"])
    assert max(view.data.shape) > max(first)
    assert max(view.data.shape) <= 2048
    np.testing.assert_array_equal(viewer.volume.data, source)


def prepare_panorama(viewer):
    center = viewer.volume.center
    points = center + np.array([[-12., 9., 0.], [0., -8., 0.], [12., 9., 0.]])
    viewer.arch_points = list(points)
    viewer.arch_toggle.setChecked(False)
    result = curved_planar(viewer.volume, points, (0, 0, 1), height_mm=40,
                           thickness_mm=0, rows=129, samples=257)
    viewer.panorama_ready((viewer.pano_revision, id(viewer.volume), np.array([0., 0., 1.]), 0., "mean", result))
    viewer.pano_refresh_timer.stop()
    viewer.implant_pano_view.set_image(result.image, physical_size_mm=(
        (result.image.shape[1] - 1) * result.column_pitch_mm, 40.))
    return result


def test_cpr_drag_updates_scan_location_and_both_panorama_cursors(app, viewer):
    result = prepare_panorama(viewer)
    view = viewer.implant_pano_view
    view.resize(1000, 500)
    viewer.panorama_picked({"fraction": (.5, .5)})
    old = viewer.crosshair.copy()
    start = view._rect().center()
    end = start + QPointF(30, -12)
    for kind, point, button, buttons in [
        (QEvent.Type.MouseButtonPress, start, Qt.MouseButton.RightButton, Qt.MouseButton.RightButton),
        (QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton, Qt.MouseButton.RightButton),
        (QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.RightButton, Qt.MouseButton.NoButton),
    ]:
        QApplication.sendEvent(view, QMouseEvent(kind, point, point, button, buttons, Qt.KeyboardModifier.NoModifier))
    assert np.linalg.norm(viewer.crosshair - old) > 1
    assert viewer.pano_view.cursor_fraction == viewer.implant_pano_view.cursor_fraction
    index = viewer.pano_index.value()
    np.testing.assert_allclose(viewer.crosshair[:2], result.path[index, :2])
    assert not viewer._interaction_active
    settle(app, viewer)


def test_cpr_detail_plan_uses_large_visible_pane_and_flags_old_low_detail(app, viewer, monkeypatch):
    prepare_panorama(viewer)
    viewer.pano_view.resize(1600, 900)
    monkeypatch.setattr(viewer.pano_view, "isVisible", lambda: True)
    rows, columns = viewer.panorama_sample_shape()
    assert rows > 360 and columns > viewer.pano_result.image.shape[1]
    assert viewer.panorama_needs_detail()
    viewer.pano_refresh_timer.stop()
    viewer.slice_sampling_changed(viewer.pano_view)
    assert viewer.pano_refresh_timer.isActive()
    viewer.pano_refresh_timer.stop()
    settle(app, viewer)
