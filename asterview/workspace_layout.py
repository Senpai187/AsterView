"""Draggable Qt pane proportions around one stationary native 3D renderer."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QGridLayout, QSplitter, QWidget


def _splitter(orientation, name, changed):
    splitter = QSplitter(orientation)
    splitter.setObjectName(name)
    splitter.setChildrenCollapsible(False)
    splitter.setHandleWidth(7)
    splitter.setOpaqueResize(True)
    splitter.splitterMoved.connect(changed)
    splitter.setStyleSheet(
        "QSplitter::handle { background:#333843; }"
        "QSplitter::handle:hover { background:#b99b75; }"
    )
    return splitter


def install_workspace_splitters(window):
    """Replace only planar Qt panes and 3D placeholders, never the VTK widget."""
    window._workspace_splitters = {}

    def changed(*_):
        window._volume_geometry_timer.start(0)
        window.schedule_render()

    panes = {
        0: (window.mpr_views["Coronal"], window.mpr_views["Sagittal"],
            window.mpr_views["Axial"], window._volume_placeholders[0]),
        1: (window.pano_sagittal_view, window.arch_view,
            window.pano_view, window._volume_placeholders[1]),
        2: (window.implant_arch_view, window.implant_pano_view,
            window.implant_display_tabs, window._volume_placeholders[2]),
    }
    for index, widgets in panes.items():
        host, _ = window._volume_slots[index]
        vertical = _splitter(Qt.Orientation.Vertical, f"workspaceRows{index}", changed)
        top = _splitter(Qt.Orientation.Horizontal, f"workspaceTop{index}", changed)
        bottom = _splitter(Qt.Orientation.Horizontal, f"workspaceBottom{index}", changed)
        wrappers, wrapper_layouts = [], []
        for position, widget in enumerate(widgets):
            host.removeWidget(widget)
            wrapper = QWidget()
            wrapper.setObjectName(f"workspacePane{index}_{position}")
            wrapper.setMinimumSize(120, 100)
            layout = QGridLayout(wrapper)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(widget, 0, 0)
            (top if position < 2 else bottom).addWidget(wrapper)
            wrappers.append(wrapper)
            wrapper_layouts.append(layout)
        vertical.addWidget(top)
        vertical.addWidget(bottom)
        host.addWidget(vertical, 0, 0, 2, 2)
        for splitter in (top, bottom, vertical):
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 1)
            splitter.handle(1).setToolTip("Drag to resize these image panes")
        window._volume_slots[index] = (wrapper_layouts[3], (0, 0, 1, 1))
        window._workspace_splitters[index] = {
            "vertical": vertical, "top": top, "bottom": bottom, "wrappers": wrappers,
            "logical": (*widgets[:3], window.volume_card),
            "ratios": ((1, 1), (1, 1), (2, 1) if index == 1 else (1, 1)),
            "maximized": None, "saved_sizes": None,
        }
    QTimer.singleShot(0, window.reset_workspace_splitters)


def reset_workspace_splitters(window):
    for record in getattr(window, "_workspace_splitters", {}).values():
        for name, weights in zip(("vertical", "top", "bottom"), record["ratios"]):
            splitter = record[name]
            span = splitter.height() if splitter.orientation() == Qt.Orientation.Vertical else splitter.width()
            span = max(span - splitter.handleWidth(), sum(splitter.sizes()), 2)
            splitter.setSizes([round(span * weight / sum(weights)) for weight in weights])
        record["saved_sizes"] = None
    timer = getattr(window, "_volume_geometry_timer", None)
    if timer is not None:
        timer.start(0)


def apply_explorer_maximize(window, target):
    record = getattr(window, "_workspace_splitters", {}).get(0)
    if record is None:
        return False
    if target is not None and target not in record["logical"]:
        return False
    if target is not None and record["maximized"] is None:
        record["saved_sizes"] = {name: record[name].sizes() for name in ("vertical", "top", "bottom")}
    record["maximized"] = target
    selected = None if target is None else record["logical"].index(target)
    for index, wrapper in enumerate(record["wrappers"]):
        wrapper.setVisible(selected is None or index == selected)
    record["top"].setVisible(selected is None or selected < 2)
    record["bottom"].setVisible(selected is None or selected >= 2)
    if selected is None and record["saved_sizes"] is not None:
        for name, sizes in record["saved_sizes"].items():
            record[name].setSizes(sizes)
        record["saved_sizes"] = None
    return True
