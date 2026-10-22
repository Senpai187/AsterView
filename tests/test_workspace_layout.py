"""Exercise draggable workspace proportions without starting the native GPU."""
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QGridLayout, QTabWidget, QVBoxLayout, QWidget

from asterview.workspace_layout import (apply_explorer_maximize,
                                        install_workspace_splitters,
                                        reset_workspace_splitters)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def workspace(app):
    host = QWidget()
    tabs = QTabWidget(host)
    QVBoxLayout(host).addWidget(tabs)
    window = SimpleNamespace(_volume_slots={}, _volume_placeholders={},
                             _volume_geometry_timer=QTimer(), render_count=0)
    def schedule_render():
        window.render_count += 1
    window.schedule_render = schedule_render
    window.reset_workspace_splitters = lambda: reset_workspace_splitters(window)
    all_panes = []
    for index in range(3):
        page = QWidget()
        layout = QGridLayout(page)
        panes = [QWidget() for _ in range(4)]
        for position, pane in enumerate(panes):
            layout.addWidget(pane, position // 2, position % 2)
        tabs.addTab(page, str(index))
        window._volume_placeholders[index] = panes[-1]
        window._volume_slots[index] = (layout, (1, 1, 1, 1))
        all_panes.append(panes)
    window.mpr_views = dict(zip(("Coronal", "Sagittal", "Axial"), all_panes[0][:3]))
    window.pano_sagittal_view, window.arch_view, window.pano_view = all_panes[1][:3]
    window.implant_arch_view, window.implant_pano_view, window.implant_display_tabs = all_panes[2][:3]
    # This represents the fixed native renderer parent; it must never move.
    window.volume_card = QWidget(tabs)
    window.volume_card.hide()
    native_parent = window.volume_card.parentWidget()
    install_workspace_splitters(window)
    host.resize(1200, 900)
    host.show()
    app.processEvents()
    yield window, tabs, native_parent
    window._volume_geometry_timer.stop()
    host.close()
    app.processEvents()


def test_panorama_bottom_splitter_resizes_cpr_independently_of_top(app, workspace):
    window, tabs, native_parent = workspace
    tabs.setCurrentIndex(1)
    app.processEvents()
    reset_workspace_splitters(window)
    record = window._workspace_splitters[1]
    bottom, top = record["bottom"], record["top"]
    before, top_before = bottom.sizes(), top.sizes()
    handle = bottom.handle(1)
    point = handle.rect().center()
    QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=point)
    QTest.mouseMove(handle, point + QPoint(-150, 0), delay=1)
    QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=point + QPoint(-150, 0))
    app.processEvents()
    assert bottom.sizes()[0] < before[0] - 100
    assert top.sizes() == top_before
    assert window.render_count >= 1
    assert window.volume_card.parentWidget() is native_parent
    layout, position = window._volume_slots[1]
    assert layout.itemAtPosition(*position[:2]).widget() is window._volume_placeholders[1]
    bottom.setSizes([0, 9999])
    assert min(bottom.sizes()) >= 120


def test_explorer_maximize_restores_custom_sizes_and_keeps_grid_container(app, workspace):
    window, _, native_parent = workspace
    record = window._workspace_splitters[0]
    record["vertical"].setSizes([330, 520])
    record["top"].setSizes([420, 720])
    record["bottom"].setSizes([750, 390])
    original = {key: record[key].sizes() for key in ("vertical", "top", "bottom")}
    source = window.mpr_views["Axial"]
    wrapper = record["wrappers"][2]
    container = QWidget(wrapper)
    wrapper.layout().replaceWidget(source, container)
    source.hide()
    assert apply_explorer_maximize(window, source)
    app.processEvents()
    assert [pane.isVisible() for pane in record["wrappers"]] == [False, False, True, False]
    assert container.isVisible() and source.isHidden()
    assert apply_explorer_maximize(window, None)
    app.processEvents()
    assert all(pane.isVisible() for pane in record["wrappers"])
    assert {key: record[key].sizes() for key in original} == original
    assert window.volume_card.parentWidget() is native_parent
    reset_workspace_splitters(window)
    app.processEvents()
    for name in ("vertical", "top", "bottom"):
        left, right = record[name].sizes()
        assert abs(left - right) <= 1


def test_explorer_rejects_unrelated_target_without_hiding_panes(app, workspace):
    window, _, _ = workspace
    assert not apply_explorer_maximize(window, window.pano_view)
    assert all(pane.isVisible() for pane in window._workspace_splitters[0]["wrappers"])


def test_workspace_without_3d_placeholder_hides_native_card_without_reparenting(app, workspace):
    from asterview.workflow_controls import WorkflowControls
    window, tabs, native_parent = workspace
    index = tabs.addTab(QWidget(), "Comparison")
    window.tabs = tabs
    window._closing = False
    window.volume_card.show()
    tabs.setCurrentIndex(index)
    WorkflowControls.sync_volume_geometry(window)
    assert window.volume_card.isHidden()
    assert window.volume_card.parentWidget() is native_parent
