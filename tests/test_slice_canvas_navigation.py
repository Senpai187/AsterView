"""Canvas gesture regressions without a GPU or a MainWindow."""
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from asterview.slice_view import SliceView


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def view(qt_app):
    widget = SliceView("Axial")
    widget.resize(1000, 400)
    widget.set_image(np.arange(512 * 512, dtype=np.float32).reshape((512, 512)),
                     (np.zeros(3), np.array([1., 0., 0.]), np.array([0., 1., 0.]), (100., 100.)),
                     np.zeros(3))
    widget.reference_lines = [{"name": "Sagittal", "direction": np.array([1., 0., 0.]), "color": "#ff0000"}]
    yield widget
    widget.close()


def event(view, kind, point, button, buttons):
    QApplication.sendEvent(view, QMouseEvent(kind, point, point, button, buttons, Qt.KeyboardModifier.NoModifier))


@pytest.mark.parametrize("mode", ["Navigate", "Correct planes", "Window / level", "Measure", "Edit arch"])
def test_right_drag_in_letterbox_rotates_lines_without_changing_intensity(view, mode):
    view.mode = mode
    start, end = QPointF(80, 180), QPointF(80, 230)
    assert not view._rect().contains(start)
    assert view.screen_to_world(start) is None
    rotations, windows, phases = [], [], []
    view.reference_rotated.connect(rotations.append)
    view.window_dragged.connect(lambda x, y: windows.append((x, y)))
    view.interaction_started.connect(lambda: phases.append("start"))
    view.interaction_finished.connect(lambda: phases.append("end"))
    before = view.gray.copy()
    event(view, QEvent.Type.MouseButtonPress, start, Qt.MouseButton.RightButton, Qt.MouseButton.RightButton)
    assert view._drag_kind == "rotate"
    event(view, QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton, Qt.MouseButton.RightButton)
    event(view, QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.RightButton, Qt.MouseButton.NoButton)
    assert len(rotations) == 1 and abs(rotations[0]) > 1
    assert windows == []
    assert phases == ["start", "end"]
    np.testing.assert_array_equal(view.gray, before)


@pytest.mark.parametrize("in_margin", [False, True])
def test_add_arch_right_click_finishes_trace_with_reference_lines_present(view, in_margin):
    view.mode = "Add arch point"
    view.points = [np.array([-15., 0., 0.]), np.zeros(3), np.array([15., 0., 0.])]
    point = QPointF(80, 180) if in_margin else view._rect().center()
    finished, rotations, interactions = [], [], []
    view.path_finished.connect(lambda: finished.append(True))
    view.reference_rotated.connect(rotations.append)
    view.interaction_started.connect(lambda: interactions.append(True))
    event(view, QEvent.Type.MouseButtonPress, point, Qt.MouseButton.RightButton, Qt.MouseButton.RightButton)
    event(view, QEvent.Type.MouseButtonRelease, point, Qt.MouseButton.RightButton, Qt.MouseButton.NoButton)
    assert finished == [True]
    assert rotations == [] and interactions == []
    assert view._drag_kind is None


def test_left_reference_drag_continues_across_fitted_image_edge(view):
    start = view.world_to_screen(np.zeros(3))
    end = QPointF(80, start.y() + 20)
    moved = []
    view.reference_translated.connect(moved.append)
    event(view, QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
    event(view, QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    event(view, QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
    expected = view.screen_to_world(end, allow_outside=True) - view.screen_to_world(start)
    np.testing.assert_allclose(moved[-1]["delta"], expected)


def test_right_drag_with_hidden_lines_never_changes_window_level(view):
    view.show_references = False
    windows = []
    view.window_dragged.connect(lambda x, y: windows.append((x, y)))
    point = QPointF(80, 180)
    event(view, QEvent.Type.MouseButtonPress, point, Qt.MouseButton.RightButton, Qt.MouseButton.RightButton)
    event(view, QEvent.Type.MouseMove, point + QPointF(30, 30), Qt.MouseButton.NoButton, Qt.MouseButton.RightButton)
    assert view._drag_kind is None
    assert windows == []


def test_recoloring_reuses_plane_buffers_and_preserves_source(view):
    before = view.data.copy()
    scratch, gray = view._window_buffer, view.gray
    view.center, view.window_width = 100000., 210000.
    view._window_image()
    expected = np.clip((before - (100000. - 210000. / 2)) * (255 / 210000.), 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(view.gray, expected)
    assert view._window_buffer is scratch and view.gray is gray
    view.invert = True
    view._window_image()
    np.testing.assert_array_equal(view.gray, 255 - expected)
    np.testing.assert_array_equal(view.data, before)
    assert view.gray is gray


def test_cpr_right_drag_in_margin_moves_linked_fraction_without_windowing(view):
    view.set_image(np.ones((300, 600), np.float32), physical_size_mm=(180., 90.))
    view.fraction_navigation_enabled = True
    view.cursor_fraction = (.3, .6)
    start = QPointF(50, 160)
    assert not view._rect().contains(start)
    picks, windows, interactions = [], [], []
    view.picked.connect(picks.append)
    view.window_dragged.connect(lambda *values: windows.append(values))
    view.interaction_started.connect(lambda: interactions.append("start"))
    view.interaction_finished.connect(lambda: interactions.append("end"))
    end = start + QPointF(30, -15)
    event(view, QEvent.Type.MouseButtonPress, start, Qt.MouseButton.RightButton, Qt.MouseButton.RightButton)
    event(view, QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton, Qt.MouseButton.RightButton)
    event(view, QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.RightButton, Qt.MouseButton.NoButton)
    rect = view._rect()
    np.testing.assert_allclose(picks[-1]["fraction"], (.3 + 30 / rect.width(), .6 - 15 / rect.height()))
    assert picks[-1]["world"] is None
    assert windows == [] and interactions == ["start", "end"]


def test_cpr_left_drag_is_continuous_and_clamped_to_image(view):
    view.set_image(np.ones((300, 600), np.float32), physical_size_mm=(180., 90.))
    view.fraction_navigation_enabled = True
    picks = []
    view.picked.connect(picks.append)
    start = view._rect().center()
    event(view, QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
    event(view, QEvent.Type.MouseMove, start + QPointF(5000, -5000), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    event(view, QEvent.Type.MouseButtonRelease, start, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
    np.testing.assert_allclose(picks[0]["fraction"], (.5, .5))
    np.testing.assert_allclose(picks[-1]["fraction"], (1., 0.))


def test_preferred_sample_shape_grows_with_zoom_without_changing_world_geometry(view):
    before = tuple(np.asarray(value).copy() for value in view.frame)
    ordinary = view.preferred_sample_shape(view.frame[3])
    view.zoom = 2
    enlarged = view.preferred_sample_shape(view.frame[3])
    assert enlarged[0] > ordinary[0] and enlarged[1] > ordinary[1]
    for actual, original in zip(view.frame, before):
        np.testing.assert_array_equal(actual, original)
