"""Visible slider and linked grid navigation with real Qt widgets, CPU only."""
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from asterview.model import Volume
from asterview.physical_slider import PhysicalSlider, line_scroll_limits
from asterview.reformat import PanoramicResult, rotation_matrix
from asterview.section_navigation import SectionNavigationControls
from asterview.slice_view import SliceView


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_physical_slider_updates_without_reemitting_and_uses_finite_mm(app):
    slider = PhysicalSlider()
    received = []
    slider.position_changed.connect(received.append)
    slider.set_position(-11.5, 24.5, 4.25, .4)
    assert received == []
    assert slider.physical_value == 4.25
    slider.setValue(50_000)
    assert received == [6.5]
    assert "6.50 mm" in slider.toolTip()
    slider.set_position(-11.5, 24.5, -11.5, .4)
    assert slider.value() == 0 and received == [6.5]
    with pytest.raises(ValueError):
        slider.set_position(1, 0, 0)
    slider.close()


def test_oblique_slider_limits_follow_cursor_line_inside_source_box():
    rotation = rotation_matrix(17, -24, 38)
    volume = Volume(np.zeros((21, 41, 31), np.float32), (.4, .5, .7),
                    np.array([7., -3, 20]), rotation)
    position = volume.index_to_world([7., 10., 16.])
    normal = np.array([.3, -.4, .8])
    normal /= np.linalg.norm(normal)
    lower, upper = line_scroll_limits(volume, position, normal)
    for value in (lower, upper):
        point = position + normal * (value - position @ normal)
        index = volume.world_to_index(point)
        assert np.all(index >= -1e-8) and np.all(index <= np.asarray(volume.dimensions) - 1 + 1e-8)
        assert np.any(np.isclose(index, 0, atol=1e-8)) or np.any(np.isclose(index, np.asarray(volume.dimensions) - 1, atol=1e-8))


class NavigationHarness(SectionNavigationControls):
    def __init__(self):
        self.volume = Volume(np.zeros((41, 41, 41), np.float32), (.4, .4, .4), np.zeros(3), np.eye(3))
        self.crosshair = self.volume.center.copy()
        self.implant_serial_views = []
        self.renders = 0

    def schedule_render(self):
        self.renders += 1


def test_mpr_slider_moves_only_plane_normal_and_syncs_without_feedback(app):
    window = NavigationHarness()
    view = SliceView("Coronal")
    right, up = np.array([1., 0, 0]), np.array([0., 0, 1.])
    frame = (window.crosshair.copy(), right, up, (16, 16))
    window.sync_section_position(view, frame, window.crosshair)
    view.show()
    app.processEvents()
    assert view.position_slider.isVisible()
    before = window.crosshair.copy()
    view.position_slider.setValue(75_000)
    np.testing.assert_allclose(window.crosshair[[0, 2]], before[[0, 2]])
    assert window.crosshair[1] == pytest.approx(4.)
    assert window.renders == 1
    window.sync_section_position(view, (window.crosshair.copy(), right, up, (16, 16)), window.crosshair)
    assert view.position_slider.value() == 75_000 and window.renders == 1
    assert not view.image_canvas_rect().intersects(view.position_slider.geometry())
    view.close()


def test_ctrl_wheel_in_any_implant_cell_updates_entire_grid_once(app):
    window = NavigationHarness()
    for index in range(8):
        view = SliceView(str(index))
        view.zoom_changed.connect(window.link_implant_section_zoom)
        window.implant_serial_views.append(view)
    for index, steps in ((3, 2), (7, -1)):
        view = window.implant_serial_views[index]
        point = QPointF(100, 100)
        event = QWheelEvent(point, point, QPoint(), QPoint(0, steps * 120),
                            Qt.MouseButton.NoButton, Qt.KeyboardModifier.ControlModifier,
                            Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(view, event)
        assert all(pane.zoom == view.zoom for pane in window.implant_serial_views)
    assert window._implant_grid_zoom == pytest.approx(1.15)
    assert window.renders == 2
    for view in window.implant_serial_views:
        view.close()


def test_report_capture_excludes_ui_and_scrollbar_but_preserves_zoom(app):
    view = SliceView("UI header must not appear")
    view.resize(440, 360)
    view.set_window(.5, 1.)
    data = np.zeros((101, 101), np.float32)
    data[40:61, 40:61] = 1
    view.set_image(data, (np.zeros(3), [1., 0, 0], [0, 1., 0], (20., 20.)))
    view.set_scroll_position(-10, 10, 0, .4)
    view.show()
    app.processEvents()
    canvas = view.image_canvas_rect().toAlignedRect()
    captured = view.capture_content_image()
    assert captured.width() == round(canvas.width() * view.devicePixelRatioF())
    assert captured.height() == round(canvas.height() * view.devicePixelRatioF())
    assert captured.pixelColor(captured.width() // 2, captured.height() // 2).red() > 240
    assert captured.pixelColor(0, 0).red() < 20
    view.zoom = 4.
    zoomed = view.capture_content_image()
    offset = int(zoomed.width() * .15)
    assert captured.pixelColor(captured.width() // 2 + offset, captured.height() // 2).red() < 20
    assert zoomed.pixelColor(zoomed.width() // 2 + offset, zoomed.height() // 2).red() > 240
    view.close()


def test_dragging_data_range_bound_changes_physical_crop_without_moving_crosshair(app):
    view = SliceView("Sagittal range")
    view.resize(600, 500)
    view.set_image(np.zeros((201, 201), np.float32),
                   (np.zeros(3), np.array([1., 0, 0]), np.array([0., 0, 1.]), (40, 40)),
                   crosshair=np.zeros(3))
    view.set_data_range(np.zeros(3), [0, 0, 1], (-8, 8), (-20, 20), .4)
    changed = []
    view.range_bounds_changed.connect(lambda lower, upper: changed.append((lower, upper)))
    q = view.world_to_screen([0, 0, 8])
    finish = view.world_to_screen([0, 0, 12])
    for kind, point, button, buttons in (
        (QEvent.Type.MouseButtonPress, q, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton),
        (QEvent.Type.MouseMove, finish, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton),
        (QEvent.Type.MouseButtonRelease, finish, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton),
    ):
        QApplication.sendEvent(view, QMouseEvent(kind, point, point, button, buttons, Qt.KeyboardModifier.NoModifier))
    assert changed[-1] == pytest.approx((-8, 12))
    np.testing.assert_array_equal(view.crosshair, np.zeros(3))
    view.close()


def test_section_markers_match_same_continuous_centers_in_arch_and_panorama():
    from asterview.section_geometry import ArchSectionGeometry
    window = NavigationHarness()
    path = np.array([[0., 0, 0], [1., -1, 0], [3., -2, 0], [8., 0, 0]])
    geometry = ArchSectionGeometry(path, [0, 0, 1])
    window.pano_result = PanoramicResult(np.zeros((21, 4)), path, geometry.length / 3, 1.)
    window.arch_view = SimpleNamespace(update=lambda: None)
    window.implant_pano_view = SimpleNamespace(update=lambda: None)
    positions = [(1.3, 1), (4.2, 2), (7.1, 3)]
    window.update_implant_section_markers(geometry, positions, 33.)
    for (distance, label), (start, end, actual_label), (fraction, column_label) in zip(
            positions, window.arch_view.section_markers, window.implant_pano_view.section_columns):
        center = (start + end) / 2
        np.testing.assert_allclose(center, geometry.frame(distance)[0])
        assert np.linalg.norm(end - start) == pytest.approx(33)
        column = fraction * len(path) - .5
        segment = min(len(path) - 2, int(column))
        point = path[segment] + (column - segment) * (path[segment + 1] - path[segment])
        np.testing.assert_allclose(point, center)
        assert label == actual_label == column_label
