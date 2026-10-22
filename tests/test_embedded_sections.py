"""Real workspace checks for embedded grids and native-renderer stability."""
import sys
import time

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from asterview.app import MainWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def settle(app, window, timeout=15):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if (window.slice_job is None and window.slice_pending is None
                and window.job is None and not window.render_timer.isActive()
                and not window._wheel_settle.isActive()):
            app.processEvents()
            return
        time.sleep(.002)
    raise AssertionError("Embedded section workspace did not settle")


@pytest.fixture
def viewer(app, monkeypatch):
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *error: errors.append(error))
    window = MainWindow()
    window.resize(1450, 1000)
    window.show()
    window.load_demo()
    settle(app, window)
    yield window
    window.close_section_grids()
    if getattr(window, "_section_dialog", None):
        window._section_dialog.close()
    settle(app, window)
    window.close()
    app.processEvents()
    assert not errors, errors


def wheel(view, clicks):
    point = QPointF(view.rect().center())
    event = QWheelEvent(point, point, QPoint(), QPoint(0, clicks * 120),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(view, event)
    assert event.isAccepted()


def test_embedded_mpr_grid_real_wheel_changes_every_cell_and_preserves_source(app, viewer):
    source = viewer.mpr_views["Axial"]
    source_data = viewer.volume.data.copy()
    native = int(viewer.volume_view.vtk_widget.winId())
    record = viewer.open_section_grid(source, 4, 6, .75)
    settle(app, viewer)
    assert record["dialog"].parentWidget() is source.parentWidget()
    assert source.isHidden()
    original = [view.frame[0].copy() for view in record["views"]]
    normal = np.cross(*record["views"][0].frame[1:3])
    for cell in (record["views"][0], record["views"][11], record["views"][-1]):
        wheel(cell, 1)
    settle(app, viewer)
    for cell, center in zip(record["views"], original):
        np.testing.assert_allclose(cell.frame[0], center + normal * 2.25)
        assert cell.data is not None
    first = record["views"][0]
    viewer.open_section_grid(source, 6, 7, 1.)
    settle(app, viewer)
    assert record["views"][0] is first
    assert len(record["views"]) == 42
    assert record["offset"].value() == 2.25
    viewer.maximize_view(source)
    settle(app, viewer)
    assert viewer._maximized_view is source
    panes = viewer._workspace_splitters[0]["wrappers"]
    assert [pane.isVisible() for pane in panes] == [False, False, True, False]
    assert panes[2].layout().indexOf(record["dialog"]) >= 0
    assert record["dialog"].isVisible()
    assert source.isHidden()
    viewer.maximize_view(source)
    settle(app, viewer)
    assert viewer._maximized_view is None
    viewer.maximize_section(first)
    settle(app, viewer)
    assert viewer._section_dialog_view is first
    wheel(first, -1)
    settle(app, viewer)
    assert record["offset"].value() == 1.25
    viewer._section_dialog.close()
    settle(app, viewer)
    viewer.open_section_grid(source, 1, 1, 1.)
    settle(app, viewer)
    assert not source.isHidden()
    assert int(viewer.volume_view.vtk_widget.winId()) == native
    np.testing.assert_array_equal(viewer.volume.data, source_data)


def test_tmj_embedded_group_follows_corrected_axis_and_entire_range_wheels(app, viewer):
    viewer.tmj_centers = {"Right": viewer.volume.center.copy(),
                          "Left": viewer.volume.center + [10., 0, 0]}
    viewer.tabs.setCurrentIndex(3)
    settle(app, viewer)
    source = viewer.tmj_views[("Right", "Sagittal")]
    record = viewer.open_section_grid(source, 2, 3, .5)
    settle(app, viewer)
    viewer.tmj_angles["Right"].setValue(37)
    viewer.tmj_offset.setValue(2)
    settle(app, viewer)
    expected = viewer.section_source_frame(source)
    normal = np.cross(expected[1], expected[2])
    for index, cell in enumerate(record["views"]):
        np.testing.assert_allclose(cell.frame[0], expected[0] + normal * ((index - 2.5) * .5))
        np.testing.assert_allclose(cell.frame[1], expected[1])
    wheel(record["views"][3], 2)
    settle(app, viewer)
    for index, cell in enumerate(record["views"]):
        np.testing.assert_allclose(cell.frame[0], expected[0] + normal * (1 + (index - 2.5) * .5))
    # Other workflows retain their original pane and can keep operating.
    viewer.tabs.setCurrentIndex(0)
    viewer.rotation_spins[0].setValue(11)
    settle(app, viewer)
    viewer.tabs.setCurrentIndex(3)
    settle(app, viewer)
    frame = viewer.section_source_frame(source)
    np.testing.assert_allclose(record["views"][0].frame[1], frame[1])
    record["dialog"].close()
    settle(app, viewer)
    assert not source.isHidden()
    assert viewer._section_parent_layout(source).indexOf(source) >= 0
