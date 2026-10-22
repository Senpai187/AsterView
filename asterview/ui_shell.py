"""Aster's restrained native workstation shell; diagnostic content stays central."""
from PySide6.QtCore import Qt, QSize, Signal, QRectF, QSignalBlocker, QSettings
from PySide6.QtGui import QAction, QKeySequence, QPainter, QColor, QPen
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QToolButton, QButtonGroup, QMenu, QSplitter, QTabWidget, QScrollArea,
    QGroupBox, QFormLayout, QComboBox, QListWidget, QProgressBar, QCheckBox,
    QSpinBox, QDoubleSpinBox,
)
import qtawesome as qta
from pathlib import Path

from .render3d import VolumeView, PRESETS
from .display_slider import DisplaySlider
from .presentation import startup_joke
from .thickness_control import ThicknessControl


STYLE = """
QWidget { background: #17191d; color: #e8e9ed; font-family: 'Segoe UI Variable', 'Segoe UI'; font-size: 13px; }
QLabel { background: transparent; }
QMainWindow { background: #17191d; }
QFrame#topbar { background: #1b1d22; border-bottom: 1px solid #30333b; }
QLabel#brand { color: #f4f2ee; font-size: 23px; font-weight: 600; letter-spacing: 1px; }
QLabel#brandSub { color: #9499a3; font-size: 10px; }
QLabel#badge { color: #d6b686; background: #363027; border-radius: 10px; padding: 3px 9px; font-size: 10px; }
QLabel#study { color: #c8cbd3; font-size: 11px; padding: 4px 1px; }
QLabel#subtitle { color: #959ba8; font-size: 11px; }
QFrame#rail { background: #1b1d22; border-right: 1px solid #30333b; }
QFrame#inspector { background: #1e2026; border-left: 1px solid #30333b; }
QLabel#sectionTitle { color: #f2f1ed; font-size: 13px; font-weight: 600; padding: 3px 0; }
QPushButton, QToolButton { background: #2b2e36; border: 1px solid transparent; border-radius: 6px; padding: 6px 10px; color: #dfe1e7; }
QPushButton:hover, QToolButton:hover { background: #363a44; }
QPushButton:pressed, QToolButton:pressed { background: #45413b; }
QPushButton:checked, QToolButton:checked { color: #f0cf9e; background: #3a332b; border-color: #63523b; }
QPushButton:disabled, QToolButton:disabled { color: #626670; background: #25272e; }
QPushButton#primary { color: #211d18; background: #d8b78a; font-weight: 600; padding: 7px 15px; }
QPushButton#primary:hover { background: #e6c89e; }
QPushButton#segment { background: transparent; color: #999eaa; border-radius: 5px; padding: 6px 15px; border: 0; }
QPushButton#segment:checked { color: #f0f0f2; background: #42454d; }
QFrame#segments { background: #272a30; border-radius: 8px; }
QToolButton#railTool { background: transparent; color: #939aa7; padding: 8px 3px; border: 0; border-radius: 8px; font-size: 10px; }
QToolButton#railTool:hover { background: #2d3038; }
QToolButton#railTool:checked { background: #3c3329; color: #e6c28f; }
QToolButton#quiet { background: transparent; padding: 6px; }
QGroupBox { border: 0; border-bottom: 1px solid #353841; border-radius: 0; margin-top: 16px; padding: 9px 0 14px; font-weight: 500; }
QGroupBox::title { subcontrol-origin: margin; left: 0; color: #b9bec9; font-weight: 600; }
QDoubleSpinBox, QSpinBox, QComboBox { background: #2d3038; border: 1px solid #3b3e47; border-radius: 5px; padding: 4px 5px; min-height: 20px; selection-background-color: #68533b; }
QComboBox::drop-down { width: 18px; border: none; }
QListWidget { background: #191b20; border: 1px solid #383b45; border-radius: 6px; }
QListWidget::item { padding: 9px; }
QListWidget::item:selected { background: #40372c; color: #f3d6b0; }
QTabWidget::pane { border: none; }
QTabBar::tab { background: transparent; color: #9da3af; padding: 7px 10px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #e3c29a; border-bottom: 2px solid #b99b75; }
QScrollArea { border: 0; }
QScrollBar:vertical { background: transparent; width: 7px; margin: 0; }
QScrollBar::handle:vertical { background: #4e525d; border-radius: 3px; min-height: 40px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QStatusBar { background: #1b1d22; color: #9097a5; font-size: 10px; border-top: 1px solid #30333b; }
QProgressBar { background: #191b20; border: 0; border-radius: 3px; text-align: center; }
QProgressBar::chunk { background: #bda077; }
QSlider::groove:horizontal { height: 3px; background: #424650; border-radius: 1px; }
QSlider::handle:horizontal { background: #dcc09b; width: 12px; margin: -5px 0; border-radius: 6px; }
QSplitter::handle { background: #2b2e35; width: 3px; height: 3px; }
QMenu { background: #282b32; border: 1px solid #484b54; padding: 5px; border-radius: 7px; }
QMenu::item { padding: 7px 22px; }
QMenu::item:selected { background: #464037; border-radius: 4px; }
QToolTip { color: #f3f2ef; background: #343741; border: 1px solid #555b68; padding: 6px; }
"""

STYLE += '\nQComboBox::down-arrow { image: url("' + (
    Path(__file__).with_name("assets") / "chevron-down.svg"
).as_posix() + '"); width: 12px; height: 8px; }\n'


def icon(name, color="#bdc1cb"):
    return qta.icon("fa6s." + name, color=color)


def quiet_icon(name, tooltip, callback, parent=None):
    result = QToolButton(parent)
    result.setObjectName("quiet")
    result.setIcon(icon(name))
    result.setIconSize(QSize(16, 16))
    result.setToolTip(tooltip)
    result.clicked.connect(callback)
    return result


def group(title, parent_layout):
    widget = QGroupBox(title)
    form = QFormLayout(widget)
    form.setContentsMargins(0, 7, 0, 12)
    form.setHorizontalSpacing(10)
    form.setVerticalSpacing(9)
    parent_layout.addWidget(widget)
    return form


class SectionGridPicker(QWidget):
    """Compact row/column matrix with keyboard and exact numeric alternatives."""
    grid_changed = Signal(int, int)

    def __init__(self, rows=1, columns=5, parent=None):
        super().__init__(parent)
        self.rows, self.columns = rows, columns
        self.preview = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFixedSize(12 * 23 + 2, 8 * 23 + 2)
        self.setAccessibleName("Section grid matrix")
        self.setToolTip("Choose a rectangle of sections · up to 8 rows × 12 columns")

    def set_grid(self, rows, columns):
        self.rows, self.columns = int(rows), int(columns)
        self.setAccessibleDescription(f"{self.rows} rows by {self.columns} columns")
        self.update()

    def _cell(self, point):
        return min(8, max(1, int(point.y() // 23) + 1)), min(12, max(1, int(point.x() // 23) + 1))

    def mouseMoveEvent(self, event):
        self.preview = self._cell(event.position())
        self.update()

    def leaveEvent(self, event):
        self.preview = None
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.set_grid(*self._cell(event.position()))
            self.grid_changed.emit(self.rows, self.columns)
            event.accept()

    def keyPressEvent(self, event):
        changes = {Qt.Key.Key_Left: (0, -1), Qt.Key.Key_Right: (0, 1),
                   Qt.Key.Key_Up: (-1, 0), Qt.Key.Key_Down: (1, 0)}
        if event.key() in changes:
            dr, dc = changes[event.key()]
            self.set_grid(min(8, max(1, self.rows + dr)), min(12, max(1, self.columns + dc)))
            self.grid_changed.emit(self.rows, self.columns)
            event.accept()
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        rows, columns = self.preview or (self.rows, self.columns)
        for row in range(8):
            for column in range(12):
                selected = row < rows and column < columns
                painter.setBrush(QColor("#b99b75" if selected else "#292c33"))
                painter.setPen(QPen(QColor("#e5c595" if selected else "#555b68"), 1))
                painter.drawRoundedRect(QRectF(column * 23 + 2, row * 23 + 2, 18, 18), 2, 2)


def wire_section_view(w, view, *, grid=False):
    def panoramic():
        return view is getattr(w, "pano_view", None) or view is getattr(w, "implant_pano_view", None)
    view.section_follow_label = "Follow panoramic settings" if panoramic() else "Follow Image inspector settings"
    if getattr(w, "volume", None) is not None:
        view.thickness_presets_spacing = min(w.volume.spacing)
    if not getattr(view, "_section_controls_bound", False):
        view._section_controls_bound = True
        view.section_defaults = lambda: ((w.pano_thickness.value(), w.pano_projection.currentText()) if panoramic()
                                         else (w.slab.value(), w.slab_mode.currentText()))
        view.section_settings_changed.connect(lambda: w.section_settings_changed(view)
                                              if hasattr(w, "section_settings_changed") else w.schedule_render())
        view.grid_settings_requested.connect(lambda v=view: w.configure_section_grid(v))
    if grid:
        view.enable_section_grid()


def build_section_grid_controls(w, layout):
    """Build serial-grid controls; retain the old count field for saved callers."""
    w.implant_section_count = QSpinBox(w)
    w.implant_section_count.setRange(1, 96)
    w.implant_section_count.setValue(5)
    w.implant_section_count.hide()
    w.implant_section_rows, w.implant_section_columns = QSpinBox(), QSpinBox()
    w.implant_section_rows.setRange(1, 8)
    w.implant_section_columns.setRange(1, 12)
    w.implant_section_rows.setValue(1)
    w.implant_section_columns.setValue(5)
    w.implant_section_rows.setAccessibleName("Section grid rows")
    w.implant_section_columns.setAccessibleName("Section grid columns")
    for item in (w.implant_section_rows, w.implant_section_columns):
        item.setMaximumWidth(65)
    def grid_changed(*_):
        with QSignalBlocker(w.implant_section_count):
            w.implant_section_count.setValue(w.implant_section_rows.value() * w.implant_section_columns.value())
        w.rebuild_implant_sections()
    w.implant_section_rows.valueChanged.connect(grid_changed)
    w.implant_section_columns.valueChanged.connect(grid_changed)
    def count_changed(count):
        columns = min(12, count)
        with QSignalBlocker(w.implant_section_rows), QSignalBlocker(w.implant_section_columns):
            w.implant_section_rows.setValue((count + columns - 1) // columns)
            w.implant_section_columns.setValue(columns)
        w.rebuild_implant_sections()
    w.implant_section_count.valueChanged.connect(count_changed)
    w.implant_section_spacing = QDoubleSpinBox()
    w.implant_section_spacing.setRange(.2, 20)
    w.implant_section_spacing.setDecimals(2)
    w.implant_section_spacing.setSingleStep(.2)
    w.implant_section_spacing.setValue(2)
    w.implant_section_spacing.setSuffix(" mm")
    w.implant_section_spacing.valueChanged.connect(w.schedule_render)
    w.implant_section_width = QDoubleSpinBox()
    w.implant_section_width.setRange(5, 150)
    w.implant_section_width.setDecimals(1)
    w.implant_section_width.setSingleStep(1)
    w.implant_section_width.setValue(24)
    w.implant_section_width.setSuffix(" mm")
    w.implant_section_width.setAccessibleName("Section image width")
    w.implant_section_width.setToolTip("Physical horizontal field of view of each implant section")
    w.implant_section_width.valueChanged.connect(w.schedule_render)
    w.implant_section_offset = QDoubleSpinBox()
    w.implant_section_offset.setRange(-1000, 1000)
    w.implant_section_offset.setDecimals(2)
    w.implant_section_offset.setSingleStep(2)
    w.implant_section_offset.setSuffix(" mm")
    w.implant_section_offset.setToolTip("Shift the linked section range along the arch; scroll any section to navigate")
    w.implant_section_offset.valueChanged.connect(w.schedule_render)
    layout.addWidget(QLabel("Grid"))
    layout.addWidget(w.implant_section_rows)
    layout.addWidget(QLabel("×"))
    layout.addWidget(w.implant_section_columns)
    layout.addWidget(QLabel("Spacing"))
    layout.addWidget(w.implant_section_spacing)
    layout.addWidget(QLabel("Width"))
    layout.addWidget(w.implant_section_width)
    layout.addWidget(QLabel("Offset"))
    layout.addWidget(w.implant_section_offset)
    layout.addWidget(quiet_icon("wrench", "Choose section grid", lambda: w.configure_section_grid()))
    layout.addWidget(quiet_icon("expand", "Maximize section grid", lambda: w.maximize_section(w.implant_display_tabs)))
    layout.addStretch()


def build(w):
    from .app import spin, button
    from . import __version__
    w.setWindowTitle(f"AsterView {__version__}")
    w.setMinimumSize(1060, 720)
    root = QWidget()
    w.setCentralWidget(root)
    layout = QVBoxLayout(root)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    top = QFrame()
    top.setObjectName("topbar")
    header = QHBoxLayout(top)
    header.setContentsMargins(18, 11, 16, 11)
    header.setSpacing(10)
    mark = QLabel()
    mark.setPixmap(icon("asterisk", "#d8b78a").pixmap(24, 24))
    header.addWidget(mark)
    brand = QLabel("aster")
    brand.setObjectName("brand")
    header.addWidget(brand)
    caption = QLabel("IMAGING\nWORKSPACE")
    caption.setObjectName("brandSub")
    header.addWidget(caption)
    header.addStretch()
    segments = QFrame()
    segments.setObjectName("segments")
    seg = QHBoxLayout(segments)
    seg.setContentsMargins(4, 3, 4, 3)
    seg.setSpacing(2)
    w.workflow_buttons = QButtonGroup(w)
    workflow_names = ("Explore", "Panoramic", "Implants", "TMJ", "Superimposition")
    workflow_icons = ("table-cells-large", "archway", "screwdriver", "arrows-left-right", "layer-group")
    for i, (name, glyph) in enumerate(zip(workflow_names, workflow_icons)):
        item = QPushButton(name)
        item.setIcon(icon(glyph))
        item.setToolTip(name)
        item.setObjectName("segment")
        item.setCheckable(True)
        item.setChecked(i == 0)
        w.workflow_buttons.addButton(item, i)
        seg.addWidget(item)
    open_button = button("Open scan", w.open_folder, True)
    open_button.setIcon(icon("folder-open", "#211d18"))
    header.addWidget(open_button)
    segment_button = button("Segment", w.open_segmentation)
    segment_button.setIcon(icon("brain"))
    segment_button.setToolTip("Automatic tooth and mandibular canal segmentation and review")
    header.addWidget(segment_button)
    plan_button = button("Plan", w.open_planning)
    plan_button.setIcon(icon("ruler-combined"))
    header.addWidget(plan_button)
    w.report_button = button("Report", lambda: w.reporting_panel.open_report())
    w.report_button.setIcon(icon("file-lines"))
    w.report_button.setToolTip("Generate and edit a report in your university template")
    header.addWidget(w.report_button)
    restore = button("Restore", w.restore_loaded_scan)
    restore.setIcon(icon("arrow-rotate-left"))
    restore.setToolTip("Start this loaded DICOM again · clears this workspace's unsaved report, masks and planning edits")
    header.addWidget(restore)
    header.addWidget(quiet_icon("camera", "Save workspace snapshot", w.snapshot))
    more = QToolButton()
    more.setObjectName("quiet")
    more.setIcon(icon("ellipsis"))
    more.setToolTip("Session and demo")
    more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    menu = QMenu(more)
    for text, glyph, callback in (
            ("Clinician profile…", "user-doctor", w.edit_clinician_profile),
            ("Load synthetic demo", "cube", w.load_demo),
            ("Save session…", "floppy-disk", w.save_session),
            ("Open session…", "folder-open", w.open_session)):
        action = menu.addAction(icon(glyph), text)
        action.triggered.connect(callback)
    more.setMenu(menu)
    header.addWidget(more)
    layout.addWidget(top)
    context = QHBoxLayout()
    context.setContentsMargins(18, 5, 18, 5)
    w.study_label = QLabel("Open a DICOM series to begin")
    w.study_label.setTextFormat(Qt.TextFormat.PlainText)
    w.study_label.setObjectName("study")
    w.study_label.setSizePolicy(w.study_label.sizePolicy().Policy.Ignored, w.study_label.sizePolicy().Policy.Preferred)
    context.addWidget(w.study_label, 1)
    layout.addLayout(context)
    acquisition_row = QHBoxLayout()
    acquisition_row.setContentsMargins(18, 0, 18, 5)
    w.acquisition_label = QLabel("")
    w.acquisition_label.setTextFormat(Qt.TextFormat.PlainText)
    w.acquisition_label.setObjectName("study")
    w.acquisition_label.setSizePolicy(w.acquisition_label.sizePolicy().Policy.Ignored, w.acquisition_label.sizePolicy().Policy.Preferred)
    acquisition_row.addWidget(w.acquisition_label, 1)
    acquisition_row.addWidget(QLabel("Slice thickness"))
    w.header_thickness = ThicknessControl()
    w.header_thickness.setSuffix(" mm")
    w.header_thickness.setAccessibleName("Slice thickness")
    w.header_thickness.setToolTip("Section thickness starts at the smallest source voxel spacing. A pane's wrench can set its own thickness.")
    w.header_thickness.setFixedWidth(122)
    acquisition_row.addWidget(w.header_thickness)
    layout.addLayout(acquisition_row)

    body = QHBoxLayout()
    body.setContentsMargins(0, 0, 0, 0)
    body.setSpacing(0)
    layout.addLayout(body, 1)
    rail = QFrame()
    rail.setObjectName("rail")
    rail.setFixedWidth(70)
    tools = QVBoxLayout(rail)
    tools.setContentsMargins(6, 10, 6, 9)
    tools.setSpacing(5)
    w.rail_group = QButtonGroup(w)
    w.tool_combo = QComboBox()
    modes = ["Navigate", "Correct planes", "Measure", "Pan", "Window / level", "Place implant", "Move implant", "Mark right TMJ", "Mark left TMJ", "Paint mask", "Erase mask", "Edit canal path"]
    w.tool_combo.addItems(modes)
    w.tool_combo.currentTextChanged.connect(w.set_tool)
    for i, (name, glyph, mode, tip) in enumerate((
        ("Move", "arrows-up-down-left-right", "Navigate", "Drag the center or a colored line to move sections · 1"),
        ("Rotate", "rotate", "Correct planes", "Drag to align the reference lines with a tooth or structure · 2"),
        ("Ruler", "ruler", "Measure", "Measure between two points · 3"),
        ("Pan", "hand", "Pan", "Drag the image to pan · 4"),
    )):
        tool = QToolButton()
        tool.setObjectName("railTool")
        tool.setText(name)
        tool.setIcon(icon(glyph, "#c8c8c7"))
        tool.setIconSize(QSize(19, 19))
        tool.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        tool.setCheckable(True)
        tool.setFixedSize(58, 58)
        tool.setToolTip(tip)
        tool.setProperty("mode", mode)
        tool.clicked.connect(lambda checked=False, m=mode: w.tool_combo.setCurrentText(m))
        w.rail_group.addButton(tool, i)
        tools.addWidget(tool)
        action = QAction(w)
        action.setShortcut(QKeySequence(str(i + 1)))
        action.triggered.connect(lambda checked=False, m=mode: w.tool_combo.setCurrentText(m))
        w.addAction(action)
    w.rail_group.button(0).setChecked(True)
    display = QToolButton()
    display.setObjectName("railTool")
    display.setText("Display")
    display.setIcon(icon("sliders"))
    display.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
    display.setFixedSize(58, 58)
    display.setToolTip("Brightness and contrast sliders · 5")
    def show_display():
        w.inspector.show()
        w.inspector_tabs.setCurrentIndex(0)
        w.brightness.slider.setFocus()
    display.clicked.connect(show_display)
    tools.addWidget(display)
    restore_sections = QToolButton()
    restore_sections.setObjectName("railTool")
    restore_sections.setText("Restore\nsections")
    restore_sections.setIcon(icon("rotate-left"))
    restore_sections.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
    restore_sections.setFixedSize(58, 66)
    restore_sections.setToolTip("Restore section orientation, zoom and pan · R")
    restore_sections.clicked.connect(w.restore_section_views)
    tools.addWidget(restore_sections)
    display_action = QAction(w)
    display_action.setShortcut(QKeySequence("5"))
    display_action.triggered.connect(show_display)
    w.addAction(display_action)
    tools.addStretch()
    tools.addWidget(quiet_icon("sliders", "Show / hide inspector", lambda: w.inspector.setVisible(not w.inspector.isVisible())))
    body.addWidget(rail)
    splitter = QSplitter(Qt.Orientation.Horizontal)
    w.main_splitter = splitter
    body.addWidget(splitter, 1)
    viewing = QWidget()
    viewing_layout = QVBoxLayout(viewing)
    viewing_layout.setContentsMargins(0, 0, 0, 0)
    viewing_layout.setSpacing(4)
    viewing_layout.addWidget(segments)
    w.tabs = QTabWidget()
    w.tabs.tabBar().hide()
    viewing_layout.addWidget(w.tabs, 1)
    splitter.addWidget(viewing)
    inspector = QFrame()
    inspector.setObjectName("inspector")
    inspector.setMinimumWidth(300)
    inspector.setMaximumWidth(620)
    w.inspector = inspector
    inspect_layout = QVBoxLayout(inspector)
    inspect_layout.setContentsMargins(12, 10, 12, 8)
    inspect_layout.setSpacing(7)
    inspect_header = QHBoxLayout()
    title = QLabel("Inspector")
    title.setObjectName("sectionTitle")
    inspect_header.addWidget(title)
    inspect_header.addStretch()
    inspect_header.addWidget(quiet_icon("xmark", "Hide inspector", inspector.hide))
    inspect_layout.addLayout(inspect_header)
    w.inspector_tabs = QTabWidget()
    w.inspector_selector = QComboBox()
    w.inspector_selector.setAccessibleName("Inspector section")
    inspect_layout.addWidget(w.inspector_selector)
    w.inspector_tabs.tabBar().hide()
    inspect_layout.addWidget(w.inspector_tabs, 1)
    pages = []
    for name in ("Image", "3D", "Studies"):
        page = QWidget()
        content = QVBoxLayout(page)
        content.setContentsMargins(2, 7, 2, 2)
        content.setSpacing(2)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        w.inspector_tabs.addTab(scroll, name)
        pages.append(content)

    form = group("Display", pages[0])
    w.window_preset = QComboBox()
    w.window_preset.addItems(["Scan default", "Bone (HU)", "Soft tissue (HU)", "Lung (HU)", "Full intensity range"])
    form.addRow("Preset", w.window_preset)
    w.window_preset.currentTextChanged.connect(w.apply_window_preset)
    w.window_level, w.window_width = spin(500, -1e6, 1e6, 50), spin(2000, 1, 2e6, 100)
    w.brightness = DisplaySlider(w.window_level, "Brightness", inverted=True)
    w.contrast = DisplaySlider(w.window_width, "Contrast", logarithmic=True, inverted=True)
    form.addRow("Brightness", w.brightness)
    form.addRow("Contrast", w.contrast)
    w.window_level.setToolTip("Exact window level in source intensity units")
    w.window_width.setToolTip("Exact window width in source intensity units")
    w.window_level.valueChanged.connect(w.update_window)
    w.window_width.valueChanged.connect(w.update_window)
    w.slab, w.slab_mode = ThicknessControl(), QComboBox()
    w.slab.setSuffix(" mm")
    w.slab.setDecimals(3)
    w.slab_mode.addItems(["mean", "max", "min"])
    form.addRow("Thickness", w.slab)
    form.addRow("Projection", w.slab_mode)
    w.slab.valueChanged.connect(w.schedule_render)
    w.slab.valueChanged.connect(w.header_thickness.setValue)
    w.header_thickness.valueChanged.connect(w.slab.setValue)
    w.slab_mode.currentTextChanged.connect(w.schedule_render)
    w.resolution = QComboBox()
    w.resolution.addItems(["Efficient · display", "Native display", "Fine · supersample"])
    w.resolution.setCurrentIndex(1)
    w.resolution.currentIndexChanged.connect(w.display_quality_changed)
    form.addRow("Detail", w.resolution)
    w.preview_enabled = QCheckBox("Adaptive preview while moving")
    w.preview_enabled.setChecked(False)
    w.preview_enabled.hide()
    w.lines_visible = QCheckBox("MPR reference lines")
    w.lines_visible.setChecked(True)
    w.lines_visible.toggled.connect(w.refresh_references)
    form.addRow(w.lines_visible)
    form.addRow(button("Clear rulers", w.clear_measurements))
    form = group("Fine orientation", pages[0])
    w.rotation_spins = []
    for label in ("Pitch", "Roll", "Yaw"):
        item = spin(0, -180, 180, 1)
        item.setSuffix("°")
        item.valueChanged.connect(w.orientation_changed)
        w.rotation_spins.append(item)
        form.addRow(label, item)
    hint = QLabel("Drag colored lines to move sections. Right-drag anywhere in a pane, including its black margins, to correct the planes. Use the sliders above for brightness and contrast.")
    hint.setWordWrap(True)
    hint.setObjectName("subtitle")
    pages[0].addWidget(hint)
    pages[0].addStretch()

    form = group("Volume appearance", pages[1])
    w.render_preset, w.render_quality = QComboBox(), QComboBox()
    w.render_preset.addItems(list(PRESETS))
    w.render_quality.addItems(["Interactive", "Balanced", "High"])
    w.render_quality.setCurrentText("Balanced")
    w.threshold = spin(0, -1e6, 1e6, 50)
    form.addRow("Preset", w.render_preset)
    form.addRow("Quality", w.render_quality)
    form.addRow("Offset", w.threshold)
    form = group("Slice location in 3D", pages[1])
    w.plane_toggles = {}
    for name in ("Axial", "Coronal", "Sagittal"):
        item = QCheckBox(name + " plane")
        item.setChecked(True)
        item.toggled.connect(w.refresh_references)
        w.plane_toggles[name] = item
        form.addRow(item)
    w.volume_visible = QCheckBox("Show volume rendering")
    w.volume_visible.setChecked(True)
    form.addRow(w.volume_visible)
    form.addRow(button("Reset camera", lambda: w.volume_view.reset_camera()))
    w.gpu_label = QLabel("GPU information appears after a volume is opened")
    w.gpu_label.setObjectName("subtitle")
    w.gpu_label.setWordWrap(True)
    pages[1].addWidget(w.gpu_label)
    pages[1].addStretch()

    form = group("Study library", pages[2])
    w.series_list = QListWidget()
    w.series_list.setWordWrap(True)
    w.series_list.setMinimumHeight(160)
    w.series_list.itemDoubleClicked.connect(w.load_selected)
    form.addRow(w.series_list)
    w.load_button = button("Load series", w.load_selected)
    w.load_button.setEnabled(False)
    form.addRow(w.load_button)
    w.progress_bar = QProgressBar()
    w.progress_bar.setFixedWidth(150)
    w.progress_bar.setMaximumHeight(18)
    w.progress_bar.hide()
    w.cancel_button = button("Cancel", w.cancel_job)
    w.cancel_button.setMaximumWidth(72)
    w.cancel_button.hide()
    pages[2].addStretch()
    from .landmark_panel import LandmarkPanel
    w.landmark_panel = LandmarkPanel(w)
    w.review_scroll = QScrollArea()
    w.review_scroll.setWidgetResizable(True)
    w.review_scroll.setWidget(w.landmark_panel)
    from .segmentation_panel import SegmentationPanel
    w.segmentation_panel = SegmentationPanel(w)
    object_scroll = QScrollArea()
    object_scroll.setWidgetResizable(True)
    object_scroll.setWidget(w.segmentation_panel)
    from .nerve_panel import NervePathPanel
    w.nerve_panel = NervePathPanel(w)
    w.nerve_scroll = QScrollArea()
    w.nerve_scroll.setWidgetResizable(True)
    w.nerve_scroll.setWidget(w.nerve_panel)
    w.review_page = QWidget()
    review_layout = QVBoxLayout(w.review_page)
    review_layout.setContentsMargins(0, 0, 0, 0)
    w.review_tabs = QTabWidget()
    w.review_tabs.addTab(object_scroll, "Objects")
    w.review_tabs.addTab(w.review_scroll, "Landmarks")
    w.review_tabs.addTab(w.nerve_scroll, "Canals")
    review_layout.addWidget(w.review_tabs)
    w.inspector_tabs.addTab(w.review_page, "Review")
    from .planning_panel import PlanningPanel
    w.planning_panel = PlanningPanel(w)
    w.planning_scroll = QScrollArea()
    w.planning_scroll.setWidgetResizable(True)
    w.planning_scroll.setWidget(w.planning_panel)
    w.inspector_tabs.addTab(w.planning_scroll, "Plan")
    from .comparison_controls import ComparisonControls
    w.comparison = ComparisonControls(w)
    comparison_scroll = QScrollArea()
    comparison_scroll.setWidgetResizable(True)
    comparison_scroll.setWidget(w.comparison.panel)
    w.inspector_tabs.addTab(comparison_scroll, "Superimposition")
    from .reporting_panel import ReportingPanel
    w.reporting_panel = ReportingPanel(w)
    report_scroll = QScrollArea()
    report_scroll.setWidgetResizable(True)
    report_scroll.setWidget(w.reporting_panel)
    w.inspector_tabs.addTab(report_scroll, "Report")
    splitter.addWidget(inspector)
    splitter.setSizes([1100, 350])

    w.all_slices = []
    mpr = QWidget()
    grid = QGridLayout(mpr)
    w.mpr_grid = grid
    w._volume_slots = {0: (grid, (1, 1, 1, 1))}
    w._volume_workflow = 0
    w.mpr_layout = {}
    w._maximized_view = None
    grid.setContentsMargins(5, 0, 5, 5)
    grid.setSpacing(5)
    w.mpr_views = {}
    for name, color, row, col in (("Coronal", "#6bc7a3", 0, 0), ("Sagittal", "#e5858b", 0, 1), ("Axial", "#71a9f7", 1, 0)):
        view = w.new_slice(name, color)
        view.picked.connect(lambda info, n=name: w.slice_picked(info, n))
        view.scrolled.connect(lambda steps, n=name: w.scroll_plane(n, steps))
        view.reference_translated.connect(lambda payload, n=name: w.translate_planes(n, payload))
        view.reference_rotated.connect(lambda degrees, n=name: w.rotate_planes(n, degrees))
        view.interaction_started.connect(w.begin_interaction)
        view.interaction_finished.connect(w.end_interaction)
        view.max_button.show()
        view.maximize_requested.connect(lambda v=view: w.maximize_view(v))
        w.mpr_views[name] = view
        w.mpr_layout[view] = (row, col)
        grid.addWidget(view, row, col)
    # QVTK binds its native HWND during construction. Keep this parent stable
    # across workflows; layout placeholders position one overlaid GPU widget.
    volume_card = QWidget(w.tabs)
    w.volume_card = volume_card
    w._volume_placeholders = {}
    w._volume_maximized = False
    w.mpr_layout[volume_card] = (1, 1)
    volume_layout = QVBoxLayout(volume_card)
    volume_layout.setContentsMargins(0, 0, 0, 0)
    volume_layout.setSpacing(0)
    volume_bar = QHBoxLayout()
    volume_bar.setContentsMargins(12, 5, 7, 5)
    volume_bar.addWidget(QLabel("Volume"))
    volume_bar.addStretch()
    for name, color in (("Axial", "#71a9f7"), ("Coronal", "#6bc7a3"), ("Sagittal", "#e5858b")):
        toggle = QToolButton()
        toggle.setText(name[0])
        toggle.setStyleSheet(f"QToolButton {{ color:{color}; padding:3px 8px; }}")
        toggle.setCheckable(True)
        toggle.setChecked(True)
        toggle.setToolTip("Show / hide " + name.lower() + " plane")
        toggle.toggled.connect(w.plane_toggles[name].setChecked)
        w.plane_toggles[name].toggled.connect(toggle.setChecked)
        volume_bar.addWidget(toggle)
    camera = QToolButton()
    camera.setIcon(icon("cube"))
    camera.setToolTip("Anatomical camera view")
    camera.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    camera_menu = QMenu(camera)
    for name in ("Isometric", "Anterior", "Posterior", "Left", "Right", "Superior", "Inferior"):
        camera_menu.addAction(name, lambda checked=False, n=name: w.volume_view.set_view_orientation(n))
    camera.setMenu(camera_menu)
    volume_bar.addWidget(camera)
    volume_bar.addWidget(quiet_icon("house", "Return to isometric view", lambda: w.volume_view.set_view_orientation("Isometric")))
    volume_bar.addWidget(quiet_icon("camera", "Save 3D quick shot", w.snapshot_volume))
    volume_bar.addWidget(quiet_icon("expand", "Maximize / restore volume", lambda: w.maximize_section(volume_card)))
    volume_layout.addLayout(volume_bar)
    w.volume_view = VolumeView()
    w.setWindowIcon(icon("asterisk", "#d8b78a"))
    w.volume_view.status.connect(w.renderer_status)
    w.volume_view.path_node_dragged.connect(w.path_node_dragged)
    w.volume_view.implant_dragged.connect(w.implant_dragged)
    volume_layout.addWidget(w.volume_view, 1)
    volume_placeholder = QWidget()
    w._volume_placeholders[0] = volume_placeholder
    grid.addWidget(volume_placeholder, 1, 1)
    for i in range(2):
        grid.setRowStretch(i, 1)
        grid.setColumnStretch(i, 1)
    w.tabs.addTab(mpr, "Explore")
    w.render_preset.currentTextChanged.connect(w.volume_view.set_preset)
    w.render_quality.currentTextChanged.connect(w.volume_view.set_quality)
    w.threshold.valueChanged.connect(w.volume_view.set_threshold)
    w.volume_visible.toggled.connect(w.volume_view.set_volume_visibility)
    w._build_panorama()
    w._build_implants()
    w._build_tmj()
    w.superimposition_index = w.tabs.addTab(w.comparison.workspace, "Superimposition")
    for index, (slot_grid, slot) in w._volume_slots.items():
        if index not in w._volume_placeholders:
            placeholder = QWidget()
            slot_grid.addWidget(placeholder, *slot)
            w._volume_placeholders[index] = placeholder
        w._volume_placeholders[index].installEventFilter(w)
    from PySide6.QtCore import QTimer
    w._volume_geometry_timer = QTimer(w)
    w._volume_geometry_timer.setSingleShot(True)
    w._volume_geometry_timer.timeout.connect(w.sync_volume_geometry)
    w.tabs.installEventFilter(w)
    from .workspace_layout import install_workspace_splitters
    install_workspace_splitters(w)
    w._volume_geometry_timer.start(0)
    w.planning_panel.results.currentRowChanged.connect(w.implant_list.setCurrentRow)
    for view in w.all_slices:
        wire_section_view(w, view, grid=True)
        if view not in w.mpr_layout and view not in w.implant_serial_views:
            view.max_button.show()
            view.maximize_requested.connect(lambda v=view: w.maximize_section(v))
    # Keep inspector navigation compact while retaining stable page widgets.
    inspector_icons = {"Image": "sliders", "3D": "cube", "Studies": "folder-open",
                       "Review": "list-check", "Plan": "ruler-combined",
                       "Superimposition": "layer-group", "Report": "file-lines"}
    for index in range(w.inspector_tabs.count()):
        name = w.inspector_tabs.tabText(index)
        glyph = icon(inspector_icons.get(name, "list"))
        w.inspector_tabs.setTabIcon(index, glyph)
        w.inspector_selector.addItem(glyph, name)
    w.inspector_selector.currentIndexChanged.connect(w.inspector_tabs.setCurrentIndex)
    w.inspector_tabs.currentChanged.connect(w.inspector_selector.setCurrentIndex)
    w.inspector_selector.activated.connect(lambda index: w.show_superimposition() if w.inspector_selector.itemText(index) == "Superimposition" else None)
    w.workflow_buttons.idClicked.connect(w.tabs.setCurrentIndex)
    w.tabs.currentChanged.connect(lambda index: w.workflow_buttons.button(index).setChecked(True))
    w.tabs.currentChanged.connect(w.schedule_render)
    w.tabs.currentChanged.connect(w.workflow_changed)
    w.tabs.currentChanged.connect(lambda _: w.update_window())
    w.statusBar().showMessage("Ready")
    w.statusBar().addPermanentWidget(w.progress_bar)
    w.statusBar().addPermanentWidget(w.cancel_button)
    settings = QSettings("AsterView", "AsterView")
    joke = startup_joke(settings.value("lastStartupJoke", ""))
    settings.setValue("lastStartupJoke", joke)
    w.startup_note = QLabel(joke + "   ")
    w.startup_note.setObjectName("subtitle")
    w.statusBar().addPermanentWidget(w.startup_note)
    for shortcut, callback in (("Ctrl+O", w.open_folder), ("R", w.restore_section_views), ("Ctrl+Z", w.undo_edit)):
        action = QAction(w)
        action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(callback)
        w.addAction(action)
