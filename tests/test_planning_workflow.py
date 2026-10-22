"""Integration checks for source-isolated planning across actual Qt workspaces."""
from types import SimpleNamespace
import sys
import time
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from asterview.app import MainWindow
from asterview.nerve_paths import create_paths
from asterview.segmentation import create_result


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait_for(app, predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            app.processEvents()
            return
        time.sleep(.002)
    raise AssertionError("Planning workflow did not settle")


def settle(app, window):
    wait_for(app, lambda: window.job is None and window.slice_job is None and window.slice_pending is None
             and not window.render_timer.isActive() and not window.clearance_timer.isActive()
             and not window.path_timer.isActive() and not window.mask_timer.isActive()
             and not window._wheel_settle.isActive())


@pytest.fixture
def viewer(app, monkeypatch):
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *error: errors.append(error))
    window = MainWindow()
    window.load_demo()
    settle(app, window)
    yield window
    if getattr(window, "_section_dialog", None):
        window._section_dialog.close()
    window.comparison.reset_comparison()
    window.comparison._executor.shutdown(wait=True, cancel_futures=True)
    settle(app, window)
    window.close()
    app.processEvents()
    assert not errors, errors


def install_plan(app, viewer):
    viewer.crosshair = viewer.volume.center.copy()
    viewer.add_implant()
    center = viewer.crosshair.copy()
    path = {"id": "planning-canal", "name": "Controlled test canal", "side": "unknown",
            "points_lps_mm": (center + np.array([[8, -15, 0], [8, 15, 0]])).tolist(),
            "diameter_mm": 2., "color": "#ee8487", "review": "accepted",
            "provenance": {"name": "Analytic planning fixture", "version": "1"}}
    viewer.install_nerve_paths(create_paths(viewer.volume, viewer.volume_fingerprint(), [path]))
    viewer.implant_list.setCurrentRow(0)
    viewer.select_implant(0)
    settle(app, viewer)
    return center


def mouse(view, kind, position):
    buttons = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseButtonRelease else Qt.MouseButton.LeftButton
    event_button = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseMove else Qt.MouseButton.LeftButton
    QApplication.sendEvent(view, QMouseEvent(kind, position, position, event_button, buttons, Qt.KeyboardModifier.NoModifier))


def test_timer_updates_margin_path_and_selected_measurement(app, viewer):
    install_plan(app, viewer)
    assert viewer.clearance_results[0]["distance_mm"] == pytest.approx(5, abs=.02)
    assert viewer.clearance_results[0]["status"] == "beyond_margin"
    viewer.planning_panel.margin.setValue(6)
    settle(app, viewer)
    assert viewer.clearance_results[0]["status"] == "within_margin"
    assert viewer.implant_margins[viewer.implants[0]["id"]] == 6
    collection = viewer.nerve_paths
    points = np.asarray(collection.paths[0]["points_lps_mm"]).copy()
    for index, point in enumerate(points):
        collection.move_node(viewer.volume, "planning-canal", index, point + [3, 0, 0])
    viewer.refresh_nerve_paths()
    settle(app, viewer)
    assert viewer.clearance_results[0]["distance_mm"] == pytest.approx(8, abs=.02)
    assert viewer.clearance_results[0]["status"] == "unreviewed"
    assert viewer._clearance_display is viewer.clearance_results[0]
    assert all(view.clearance_measurement is viewer._clearance_display for view in viewer.all_slices)


def test_actual_slice_pointer_and_3d_signal_update_one_shared_implant(app, viewer):
    install_plan(app, viewer)
    viewer.tool_combo.setCurrentText("Move implant")
    view = viewer.mpr_views["Axial"]
    view.resize(640, 560)
    center = np.asarray(viewer.implants[0]["center"]).copy()
    start = view.world_to_screen(center)
    end = start + QPointF(10, -6)
    expected = center + view.screen_to_world(end) - view.screen_to_world(start)
    mouse(view, QEvent.Type.MouseButtonPress, start)
    assert viewer._implant_drag_active
    mouse(view, QEvent.Type.MouseMove, end)
    mouse(view, QEvent.Type.MouseButtonRelease, end)
    settle(app, viewer)
    assert not viewer._implant_drag_active
    np.testing.assert_allclose(viewer.implants[0]["center"], expected, atol=1e-8)
    identifier = viewer.implants[0]["id"]
    payload = {"implant_id": identifier, "center_lps": expected + [1, 2, 3], "axis_lps": [0, 1, 1]}
    viewer.volume_view.implant_dragged.emit(dict(payload, phase="start"))
    viewer.volume_view.implant_dragged.emit(dict(payload, phase="move"))
    viewer.volume_view.implant_dragged.emit(dict(payload, phase="end"))
    settle(app, viewer)
    assert len(viewer.implants) == 1 and viewer.implants[0]["id"] == identifier
    np.testing.assert_allclose(viewer.implants[0]["center"], expected + [1, 2, 3])
    np.testing.assert_allclose(viewer.implants[0]["axis"], [0, 2 ** -.5, 2 ** -.5])
    np.testing.assert_allclose(viewer.crosshair, viewer.implants[0]["center"])


def test_shared_3d_serial_sections_and_maximize_restore_across_workflows(app, viewer):
    install_plan(app, viewer)
    original_volume_view = viewer.volume_view
    native_handle = int(viewer.volume_view.vtk_widget.winId())
    for index in (1, 2, 3, 0):
        viewer.tabs.setCurrentIndex(index)
        settle(app, viewer)
        grid, slot = viewer._volume_slots[index]
        assert grid.itemAtPosition(slot[0], slot[1]).widget() is viewer._volume_placeholders[index]
        assert viewer.volume_card.parentWidget() is viewer.tabs
        assert int(viewer.volume_view.vtk_widget.winId()) == native_handle
        assert viewer.volume_view is original_volume_view and viewer._volume_workflow == index
        viewer.maximize_volume()
        app.processEvents()
        assert viewer.volume_card.parentWidget() is viewer.tabs
        assert int(viewer.volume_view.vtk_widget.winId()) == native_handle
        viewer.maximize_volume()
        app.processEvents()
        assert viewer._maximized_view is None and not viewer._volume_maximized
    viewer.tabs.setCurrentIndex(2)
    viewer.implant_section_count.setValue(9)
    settle(app, viewer)
    assert len(viewer.implant_serial_views) >= 9
    assert all(view.frame is not None for view in viewer.implant_serial_views[:9])
    viewer.implant_section_count.setValue(3)
    settle(app, viewer)
    assert all(view.isHidden() for view in viewer.implant_serial_views[3:])
    section = viewer.implant_views[0]
    original_parent, original_layout = section.parentWidget(), section.parentWidget().layout()
    viewer.maximize_section(section)
    assert viewer._section_dialog is not None and section.parentWidget() is viewer._section_dialog
    viewer._section_dialog.close()
    app.processEvents()
    assert viewer._section_dialog is None and section.parentWidget() is original_parent
    assert original_layout.indexOf(section) >= 0


def test_replacing_reference_clears_plan_comparison_and_proposals(app, viewer):
    install_plan(app, viewer)
    old_volume = viewer.volume
    viewer.comparison.set_moving(old_volume)
    viewer.auto_setup_provenance["test"] = {"pending": True}
    viewer.planning_panel.margin.setValue(3)
    viewer.load_demo()
    settle(app, viewer)
    assert viewer.volume is not old_volume
    assert not viewer.implants and not viewer.implant_margins and not viewer.clearance_results
    assert viewer._clearance_display is None and viewer.nerve_paths is None
    assert viewer.comparison.moving is None and viewer.comparison.result is None
    assert not viewer.auto_setup_provenance


@pytest.mark.parametrize("change", ["source", "revision", "arch"])
def test_auto_arch_result_rejected_when_source_or_user_setup_changes(app, viewer, monkeypatch, change):
    mask = np.zeros(viewer.volume.data.shape, np.uint8)
    mask[30:35, 70:75, 80:85] = 3
    viewer.segmentation = create_result(viewer.volume, viewer.volume_fingerprint(), mask,
                                       {3: {"name": "Tooth group", "kind": "tooth"}}, {"name": "Fixture", "version": "1"})
    captured = {}
    def capture(function, callback):
        captured["callback"] = callback
        return True
    monkeypatch.setattr(viewer, "start_job", capture)
    viewer.auto_arch()
    assert "callback" in captured
    points = np.array([[-10, 10, 0], [0, -10, 0], [10, 10, 0]]) + viewer.volume.center
    proposed = SimpleNamespace(points_lps_mm=points, warnings=[])
    if change == "source":
        viewer.load_demo()
    elif change == "revision":
        viewer.segmentation.set_review(3, "accepted")
    else:
        viewer.arch_points = [viewer.volume.center + [2, 0, 0]]
    before = np.array(viewer.arch_points).copy()
    captured["callback"](proposed)
    np.testing.assert_array_equal(np.array(viewer.arch_points), before)
    assert "arch" not in viewer.auto_setup_provenance
