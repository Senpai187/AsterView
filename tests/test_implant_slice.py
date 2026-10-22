import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt, QEvent
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from asterview.slice_view import SliceView
from asterview.reformat import rotation_matrix


@pytest.fixture
def view():
    app = QApplication.instance() or QApplication([])
    widget = SliceView("Oblique")
    widget.resize(800, 600)
    matrix = rotation_matrix(17, 22, -31)
    center = np.array([12., -9., 83.])
    widget.set_image(np.zeros((201, 301), np.float32), (center, matrix[:, 0], matrix[:, 1], (90, 60)), center)
    widget.mode = "Move implant"
    widget.implants = [{"id": "one", "name": "One", "center": center.tolist(), "axis": matrix[:, 1].tolist(), "length": 12., "diameter": 4.}]
    yield widget
    widget.close()


def event(kind, point, button=Qt.MouseButton.LeftButton):
    return QMouseEvent(kind, point, point, button, button, Qt.KeyboardModifier.NoModifier)


def test_oblique_implant_drag_keeps_original_pixel_mapping_during_resample(view):
    events = []
    view.implant_dragged.connect(events.append)
    original = view.frame
    center = np.asarray(view.implants[0]["center"])
    start = view.world_to_screen(center) + QPointF(2, -1)
    edge = view.edge_size_mm()
    rect = view._rect()
    expected = center + 31 * edge[0] / rect.width() * original[1] - 17 * edge[1] / rect.height() * original[2]
    view.mousePressEvent(event(QEvent.Type.MouseButtonPress, start))
    # Interactive preview and linked slice motion may replace both shape and frame.
    view.set_image(np.zeros((55, 73), np.float32), (center + original[2] * 4, original[1], original[2], (40, 20)), center)
    view.mouseReleaseEvent(event(QEvent.Type.MouseButtonRelease, start + QPointF(31, 17)))
    assert [item["phase"] for item in events] == ["start", "end"]
    np.testing.assert_allclose(events[-1]["center_lps"], expected)
    np.testing.assert_allclose(events[-1]["axis_lps"], view.implants[0]["axis"])
    assert view._implant_drag is None


def test_axis_drag_changes_normalized_direction_without_translating_center(view):
    events = []
    view.implant_dragged.connect(events.append)
    implant, kind, tip = view.implant_handles()[1]
    assert kind == "axis"
    point = view.world_to_screen(tip)
    view.mousePressEvent(event(QEvent.Type.MouseButtonPress, point))
    view.mouseReleaseEvent(event(QEvent.Type.MouseButtonRelease, point + QPointF(30, 8)))
    np.testing.assert_allclose(events[-1]["center_lps"], implant["center"])
    assert np.linalg.norm(events[-1]["axis_lps"]) == pytest.approx(1)
    assert not np.allclose(events[-1]["axis_lps"], implant["axis"])


def test_off_plane_implant_handle_is_not_pickable(view):
    center, right, up, size = view.frame
    view.implants[0]["center"] = (center + np.cross(right, up) * 20).tolist()
    assert view.implant_handles() == []
    assert view.implant_hit(view.world_to_screen(center)) is None
