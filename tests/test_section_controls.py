"""Physical section-grid behavior using real Qt panes and CPU reslicing only."""
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                               QGridLayout, QDoubleSpinBox, QComboBox, QCheckBox,
                               QDialogButtonBox, QSpinBox, QTabWidget)

from asterview.app import MainWindow
from asterview.fast_reslice import FastReslicer
from asterview.model import Volume
from asterview.reformat import rotation_matrix
from asterview.slice_view import SliceView
from asterview.ui_shell import SectionGridPicker, build_section_grid_controls, wire_section_view
from asterview.workflow_controls import WorkflowControls
from asterview.thickness_control import ThicknessControl


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class SectionHarness(QWidget, WorkflowControls):
    """Use production request and section logic without creating a native renderer."""
    draw_plane = MainWindow.draw_plane

    def __init__(self):
        super().__init__()
        zz, yy, xx = np.mgrid[:41, :41, :41]
        self.volume = Volume(zz.astype(np.float32) ** 2, (1., 1., 1.), np.zeros(3), np.eye(3))
        self.crosshair = self.volume.center.copy()
        self.resolution = QComboBox()
        self.resolution.addItems(["320", "512", "768"])
        self.slab, self.slab_mode = QDoubleSpinBox(), QComboBox()
        self.slab.setRange(0, 30)
        self.slab_mode.addItems(["mean", "max", "min"])
        self.preview_enabled = QCheckBox()
        self.preview_enabled.setChecked(True)
        self._interaction_active = self._closing = False
        self.all_slices = []
        self.implant_serial_views = []
        self.implant_serial_layout = QGridLayout()
        self.serial_widget = QWidget(self)
        self.serial_widget.setLayout(self.implant_serial_layout)
        self.implant_display_tabs = QTabWidget(self)
        self.implant_display_tabs.addTab(QWidget(), "Axes")
        self.implant_display_tabs.addTab(self.serial_widget, "Serial")
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        build_section_grid_controls(self, controls)
        layout.addLayout(controls)
        layout.addWidget(self.implant_display_tabs)
        self.mode = "Navigate"
        self.pano_result = None
        self.pano_up = np.array([0., 0., 1.])
        self.volume_card = QWidget(self)
        self.mpr_layout = {}
        self.tabs = SimpleNamespace(currentIndex=lambda: 2)
        self._section_dialog = self._section_dialog_view = None
        self._plane_requests = []
        self.slice_revision = 0
        self.slice_job = None
        self.render_count = 0
        self.message = ""
        self.resize(1200, 850)
        self.rebuild_implant_sections()

    def statusBar(self):
        return self

    def showMessage(self, message):
        self.message = message

    def schedule_render(self, *_):
        self.render_count += 1

    def new_slice(self, title, color="#65c6db"):
        view = SliceView(title, color)
        wire_section_view(self, view, grid=True)
        self.all_slices.append(view)
        self.layout().addWidget(view)
        return view

    def begin_interaction(self):
        self._interaction_active = True

    def end_interaction(self):
        self._interaction_active = False

    def slice_picked(self, info, plane):
        self.crosshair = np.asarray(info["world"]).copy()
        self.schedule_render()


@pytest.fixture
def harness(app):
    widget = SectionHarness()
    yield widget
    widget._closing = True
    if widget._section_dialog is not None:
        widget._section_dialog.close()
    widget.close_section_grids()
    for view in widget.all_slices:
        if view._settings_dialog:
            view._settings_dialog.close()
    widget.close()
    app.processEvents()


def test_visual_matrix_selects_four_by_six_and_six_by_seven(app):
    picker = SectionGridPicker()
    selected = []
    picker.grid_changed.connect(lambda rows, columns: selected.append((rows, columns)))
    QTest.mouseClick(picker, Qt.MouseButton.LeftButton, pos=QPoint(5 * 23 + 10, 3 * 23 + 10))
    assert selected[-1] == (4, 6)
    QTest.mouseClick(picker, Qt.MouseButton.LeftButton, pos=QPoint(6 * 23 + 10, 5 * 23 + 10))
    assert selected[-1] == (6, 7)
    QTest.keyClick(picker, Qt.Key.Key_Right)
    assert selected[-1] == (6, 8)
    picker.set_grid(8, 12)
    QTest.keyClick(picker, Qt.Key.Key_Right)
    QTest.keyClick(picker, Qt.Key.Key_Down)
    assert selected[-1] == (8, 12)
    picker.close()


def test_dense_grid_header_keeps_wrench_accessible_without_colliding_controls(app):
    view = SliceView("Dense section")
    view.setMinimumSize(96, 100)
    view.max_button.show()
    view.resize(96, 120)
    view.show()
    app.processEvents()
    assert view.settings_button.geometry().right() < view.width()
    assert view.settings_button.isVisible()
    assert view.max_button.isHidden() and view.snapshot_button.isHidden()
    assert view.settings_menu.actions()[-2].text().startswith("Maximize")
    view.resize(500, 400)
    app.processEvents()
    assert not view.max_button.isHidden() and not view.snapshot_button.isHidden()
    assert not view.settings_button.geometry().intersects(view.snapshot_button.geometry())
    view.close()


def test_dense_serial_cell_gives_image_space_back_and_restores_detail_when_large(app):
    view = SliceView("Cross-section 7")
    view.set_grid_cell(6)
    view.grid_caption = "7 · 42.0"
    view.max_button.show()
    view.resize(120, 110)
    view.set_window(0, 2)
    view.set_image(np.ones((101, 49), np.float32),
                   (np.zeros(3), np.array([1., 0, 0]), np.array([0., 0, 1.]), (24, 50)),
                   note="Arch 42.0 mm · Slab 1.2 mm · mean")
    view.show()
    app.processEvents()
    assert view.compact_grid()
    assert view.image_area_rect().height() == 86
    assert view._rect().height() > view.height() * .75
    assert view.settings_button.geometry().bottom() <= 21
    assert view.max_button.isHidden() and view.snapshot_button.isHidden()
    # The former footer is now part of the anatomical image at its center.
    image = view.grab().toImage()
    assert image.pixelColor(60, 100).red() > 200
    assert "Slab 1.2 mm" in view.toolTip()
    assert any(action.text().startswith("Maximize") for action in view.settings_menu.actions())
    assert any(action.text().startswith("Save section") for action in view.settings_menu.actions())
    view.resize(600, 550)
    app.processEvents()
    assert not view.compact_grid()
    assert view.image_area_rect().top() == 35
    assert not view.max_button.isHidden() and not view.snapshot_button.isHidden()
    view.close()


def test_pane_settings_apply_only_to_the_selected_pane_and_cancel_is_inert(app, harness):
    view, other = harness.implant_serial_views[:2]
    harness.slab.setValue(1.5)
    dialog = view.open_section_settings()
    assert view.open_section_settings() is dialog
    checkbox = dialog.findChild(QCheckBox)
    thickness = dialog.findChild(ThicknessControl, "sectionThickness")
    projection = dialog.findChild(QComboBox, "sectionProjection")
    assert checkbox.isChecked() and thickness.value() == 1.5
    checkbox.setChecked(False)
    thickness.setValue(3.2)
    projection.setCurrentText("max")
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Apply).click()
    assert view.slab_mm == 3.2 and view.slab_mode == "max"
    assert other.slab_mm is None and other.slab_mode is None
    thickness.setValue(9)
    dialog.close()
    assert view.slab_mm == 3.2
    view.set_section_settings()
    assert view.slab_mm is None and view.slab_mode is None
    assert view.grid_action.isVisible()


def test_panorama_alias_gets_its_own_defaults_and_one_specialized_callback(app, harness):
    # new_slice is called before its pane is assigned to the workflow alias.
    view = harness.new_slice("Panoramic reformat")
    harness.pano_view = view
    harness.pano_thickness = QDoubleSpinBox()
    harness.pano_thickness.setValue(5)
    harness.pano_projection = QComboBox()
    harness.pano_projection.addItems(["mean", "max", "min"])
    harness.pano_projection.setCurrentText("max")
    callbacks = []
    harness.section_settings_changed = callbacks.append
    wire_section_view(harness, view, grid=True)
    assert view.section_defaults() == (5., "max")
    dialog = view.open_section_settings()
    assert dialog.findChild(QCheckBox).text() == "Follow panoramic settings"
    dialog.close()
    view.set_section_settings(3, "min")
    assert callbacks == [view]


@pytest.mark.parametrize("thickness,mode", [(-1, "mean"), (np.nan, "max"), (31, "min"), (2, "bogus")])
def test_invalid_pane_settings_are_atomic(app, harness, thickness, mode):
    view = harness.implant_serial_views[0]
    view.set_section_settings(2, "max")
    with pytest.raises(ValueError):
        view.set_section_settings(thickness, mode)
    assert (view.slab_mm, view.slab_mode) == (2., "max")


def test_per_pane_thickness_reaches_real_cpu_sampler_and_cache_key(app, harness):
    engine = FastReslicer(harness.volume)
    view, other = harness.implant_serial_views[:2]
    harness.slab.setValue(0)
    view.set_section_settings(4, "max")
    harness._plane_requests = []
    for pane in (view, other):
        harness.draw_plane(pane, harness.crosshair, [1, 0, 0], [0, 1, 0], (8, 8), shape=(9, 9))
    maxima = []
    for _, frame, _, _, shape, key, thickness, mode in harness._plane_requests:
        center, right, up, size = frame
        maxima.append(float(engine.sample_plane(center, right, up, size, shape, thickness, mode)[4, 4]))
        assert key[-2:] == (thickness, mode)
    assert maxima == pytest.approx([22 ** 2, 20 ** 2])
    harness._interaction_active = True
    harness._plane_requests = []
    harness.draw_plane(view, harness.crosshair, [1, 0, 0], [0, 1, 0], (8, 8))
    assert harness._plane_requests[0][-2:] == (4, "max")
    assert view.slab_mm == 4  # interaction keeps the requested physical slab


@pytest.mark.parametrize("rows,columns", [(4, 6), (6, 7)])
def test_implant_grid_layout_retains_overrides_and_linked_physical_offsets(app, harness, rows, columns):
    source_before = harness.volume.data.copy()
    first = harness.implant_serial_views[0]
    first.set_section_settings(1.2, "min")
    harness.implant_section_rows.setValue(rows)
    harness.implant_section_columns.setValue(columns)
    count = rows * columns
    assert harness.implant_section_count.value() == count
    assert len(harness.implant_serial_views) >= count
    for index, view in enumerate(harness.implant_serial_views[:count]):
        assert harness.implant_serial_layout.itemAtPosition(index // columns, index % columns).widget() is view
        assert not view.isHidden()
    harness.implant_section_spacing.setValue(.8)
    harness.implant_section_offset.setValue(2.3)
    harness._plane_requests = []
    center = harness.crosshair.copy()
    harness.render_serial_sections(center, np.array([0., 0., 1.]), np.array([1., 0., 0.]), np.array([0., 1., 0.]))
    assert len(harness._plane_requests) == count
    for index, request in enumerate(harness._plane_requests):
        np.testing.assert_allclose(request[1][0], center + [0, 2.3 + (index - (count - 1) / 2) * .8, 0])
        np.testing.assert_array_equal(request[2], harness.crosshair)
    harness.implant_serial_views[5].scrolled.emit(2)
    assert harness.implant_section_offset.value() == pytest.approx(3.9)
    harness.implant_section_count.setValue(3)
    assert all(view.isHidden() for view in harness.implant_serial_views[3:])
    assert all(view.data is None and view.gray is None and view.image is None
               for view in harness.implant_serial_views[3:])
    harness.implant_section_rows.setValue(rows)
    harness.implant_section_columns.setValue(columns)
    assert harness.implant_serial_views[0] is first
    assert (first.slab_mm, first.slab_mode) == (1.2, "min")
    np.testing.assert_array_equal(harness.volume.data, source_before)


def test_grid_dialog_matrix_drives_exact_numeric_dimensions(app, harness):
    dialog = harness.configure_section_grid()
    picker = dialog.findChild(SectionGridPicker)
    QTest.mouseClick(picker, Qt.MouseButton.LeftButton, pos=QPoint(6 * 23 + 10, 5 * 23 + 10))
    assert dialog.findChild(QSpinBox, "sectionGridRows").value() == 6
    assert dialog.findChild(QSpinBox, "sectionGridColumns").value() == 7
    dialog.findChild(QDoubleSpinBox, "sectionGridSpacing").setValue(1.3)
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()
    assert harness.implant_section_count.value() == 42
    assert harness.implant_section_spacing.value() == 1.3
    assert harness.implant_display_tabs.currentIndex() == 1


def test_implant_grid_popup_width_and_thickness_reach_section_sampling(app, harness):
    dialog = harness.configure_section_grid()
    dialog.findChild(QDoubleSpinBox, "sectionGridWidth").setValue(33)
    dialog.findChild(QCheckBox).setChecked(False)
    dialog.findChild(ThicknessControl, "sectionGridThickness").setValue(1.2)
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()
    harness._plane_requests = []
    harness.render_serial_sections(harness.crosshair, np.array([0., 0, 1.]),
                                   np.array([1., 0, 0]), np.array([0., 1, 0]))
    assert len(harness._plane_requests) == len(harness.implant_serial_views)
    assert all(request[1][3] == (33., 50) for request in harness._plane_requests)
    assert all(request[-2] == 1.2 for request in harness._plane_requests)


def test_grid_physical_width_and_group_slab_survive_reconfiguration(app, harness):
    source = harness.new_slice("Axial")
    source.set_image(np.zeros((11, 11)),
                     (harness.crosshair, [1., 0, 0], [0, 1., 0], (12, 18)))
    record = harness.open_section_grid(source, 2, 3, .75, width_mm=33)
    record["views"][5].set_section_settings(2.4, "min")
    harness.open_section_grid(source, 2, 2, .75)
    dialog = harness.configure_section_grid(source)
    assert dialog.findChild(QDoubleSpinBox, "sectionGridWidth").value() == 33
    dialog.findChild(QCheckBox).setChecked(False)
    dialog.findChild(ThicknessControl, "sectionGridThickness").setValue(1.2)
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()
    harness.open_section_grid(source, 2, 3, .75)
    assert record["views"][5].slab_mm == 1.2
    assert record["views"][5].slab_mode == "min"
    harness._plane_requests = []
    harness.render_section_grids()
    for request in harness._plane_requests:
        assert request[1][3] == (33., 18)
        assert request[-2] == 1.2
    dialog = harness.configure_section_grid(source)
    assert not dialog.findChild(QCheckBox).isChecked()
    assert dialog.findChild(ThicknessControl, "sectionGridThickness").value() == 1.2
    dialog.reject()


def test_oblique_grid_uses_current_source_request_and_independent_slabs(app, harness):
    source = harness.new_slice("Right TMJ · Sagittal")
    rotation = rotation_matrix(13, -21, 33)
    right, up = -rotation[:, 0], rotation[:, 1]  # reflected display axes
    normal = np.cross(right, up)
    source.set_image(np.zeros((11, 11)), (harness.crosshair - 5, [1., 0, 0], [0, 1., 0], (12, 12)))
    source.set_section_settings(2, "max")
    record = harness.open_section_grid(source, 2, 3, 1.25)
    assert record["volume"] is harness.volume
    assert record["dialog"].parentWidget() is harness
    assert harness.layout().indexOf(record["dialog"]) >= 0
    assert source.isHidden()
    child = record["views"][1]
    child.set_section_settings(3.4, "min")
    harness._plane_requests = []
    harness.draw_plane(source, harness.crosshair, right, up, (12, 12))
    harness.render_section_grids()
    requests = harness._plane_requests
    assert len(requests) == 6
    for index, request in enumerate(requests):
        np.testing.assert_allclose(request[1][0], harness.crosshair + normal * ((index - 2.5) * 1.25))
        np.testing.assert_allclose(request[1][1], right)
        np.testing.assert_allclose(request[1][2], up)
    assert requests[1][-2:] == (3.4, "min")
    assert requests[0][-2:] == (2., "max")
    child.scrolled.emit(2)
    assert record["offset"].value() == 2.5
    child.picked.emit({"world": harness.crosshair + [1., 2., 0]})
    np.testing.assert_allclose(harness.crosshair, harness.volume.center + [1., 2., 0])


def test_embedded_grid_resize_preserves_range_cells_and_pane_settings(app, harness):
    source = harness.new_slice("Coronal")
    source.set_image(np.zeros((11, 11)), (harness.crosshair, [1., 0, 0], [0, 0, 1.], (12, 12)))
    record = harness.open_section_grid(source, 4, 6, .5)
    first, last = record["views"][0], record["views"][23]
    first.set_section_settings(1.8, "max")
    last.set_section_settings(2.4, "min")
    record["offset"].setValue(3.5)
    assert harness.open_section_grid(source, 6, 7, .75) is record
    assert record["views"][0] is first
    assert record["views"][23] is last
    assert record["offset"].value() == 3.5
    assert (first.slab_mm, first.slab_mode) == (1.8, "max")
    record["views"][-1].scrolled.emit(-2)
    assert record["offset"].value() == 2.
    harness.open_section_grid(source, 2, 3, .75)
    assert len(record["views"]) == 6 and last not in harness.all_slices
    harness.open_section_grid(source, 4, 6, .75)
    assert (record["views"][23].slab_mm, record["views"][23].slab_mode) == (2.4, "min")
    harness.open_section_grid(source, 1, 1, .75)
    assert not harness._section_grids
    assert not source.isHidden()
    assert harness.layout().indexOf(source) >= 0
    assert harness.section_layout_widget(source) is source
    app.processEvents()


def test_grid_cell_wrench_reconfigures_its_group(app, harness):
    source = harness.new_slice("Axial")
    source.set_image(np.zeros((11, 11)), (harness.crosshair, [1., 0, 0], [0, 1., 0], (12, 12)))
    record = harness.open_section_grid(source, 2, 3, .75)
    dialog = harness.configure_section_grid(record["views"][3])
    assert dialog.findChild(QSpinBox, "sectionGridRows").value() == 2
    assert dialog.findChild(QSpinBox, "sectionGridColumns").value() == 3
    dialog.findChild(QSpinBox, "sectionGridColumns").setValue(4)
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()
    assert len(harness._section_grids) == 1
    assert record["columns"] == 4 and len(record["views"]) == 8


def test_hidden_mpr_source_tracks_live_correction_and_cell_pick_does_not_shift_range(app, harness):
    source = harness.new_slice("Coronal")
    source.set_image(np.zeros((11, 11)), (harness.crosshair - 4, [1., 0, 0], [0, 0, 1.], (12, 12)))
    harness.mpr_views = {"Coronal": source}
    harness._mpr_sizes = {"Coronal": (12, 18)}
    rotation = rotation_matrix(10, -20, 30)
    harness.rotation = lambda: rotation
    record = harness.open_section_grid(source, 2, 2, .5)
    record["offset"].setValue(3)
    harness._plane_requests = []
    harness.render_section_grids()
    before = [item[1][0].copy() for item in harness._plane_requests]
    from asterview.mpr_geometry import plane_frame
    right, up = plane_frame("Coronal", rotation)
    for item in harness._plane_requests:
        np.testing.assert_allclose(item[1][1], right)
        np.testing.assert_allclose(item[1][2], up)
    record["views"][0].picked.emit({"world": before[0]})
    harness._plane_requests = []
    harness.render_section_grids()
    for item, expected in zip(harness._plane_requests, before):
        np.testing.assert_allclose(item[1][0], expected, atol=.005)
    record["views"][2].scrolled.emit(2)
    harness._plane_requests = []
    harness.render_section_grids()
    normal = np.cross(right, up)
    for item, expected in zip(harness._plane_requests, before):
        np.testing.assert_allclose(item[1][0], expected + normal, atol=.005)


def test_grid_samples_fit_visible_pane_pixels_instead_of_fixed_large_tiles(app, harness):
    view = harness.implant_serial_views[0]
    view.resize(120, 160)
    small = harness._section_sample_shape(view, (24, 50))
    expected_small_height = round(view.image_area_rect().height() * view.devicePixelRatioF())
    view.resize(900, 900)
    large = harness._section_sample_shape(view, (24, 50))
    assert expected_small_height <= max(small) <= expected_small_height + 2
    assert max(large) <= 2048
    assert np.prod(large) > np.prod(small)


def test_arch_sections_match_full_panorama_height_and_center_until_implant_selected(app, harness):
    from asterview.reformat import PanoramicResult
    center = harness.crosshair.copy()
    path = center + np.c_[np.linspace(-8, 8, 33), np.zeros((33, 2))]
    harness.pano_result = PanoramicResult(np.zeros((301, 33), np.float32), path, .5, .4)
    displaced = center + [0., 0., 7.]
    harness._plane_requests = []
    harness.render_serial_sections(displaced, np.array([0., 0, 1.]),
                                   np.array([1., 0, 0]), np.array([0., 1, 0]))
    for request in harness._plane_requests:
        assert request[1][3] == (24., 120.)
        assert request[1][0][2] == center[2]
    harness.implant_list = SimpleNamespace(currentRow=lambda: 0)
    harness.implants = [{}]
    harness._plane_requests = []
    harness.render_serial_sections(displaced, np.array([0., 0, 1.]),
                                   np.array([1., 0, 0]), np.array([0., 1, 0]))
    for request in harness._plane_requests:
        assert request[1][3] == (24., 120.)
        assert request[1][0][2] == displaced[2]


def test_implant_arch_grid_has_subcolumn_spacing_without_terminal_duplicates(app, harness):
    center = harness.crosshair.copy()
    path = center + np.c_[np.linspace(-5, 5, 6), np.zeros((6, 2))]
    harness.pano_result = SimpleNamespace(path=path, column_pitch_mm=2.)
    harness.implant_section_spacing.setValue(.35)
    harness._plane_requests = []
    harness.render_serial_sections(center, np.array([0., 0, 1.]),
                                   np.array([1., 0, 0]), np.array([0., 1, 0]))
    positions = np.asarray([item[1][0] for item in harness._plane_requests])
    np.testing.assert_allclose(np.diff(positions, axis=0), np.tile([.35, 0, 0], (4, 1)))
    assert len(np.unique(positions, axis=0)) == 5
    # A range wider than the traced arch has empty outside cells, not copies
    # of the terminal anatomy that misleadingly appear to be more sections.
    harness.implant_section_count.setValue(12)
    harness.implant_section_spacing.setValue(2.)
    harness._plane_requests = []
    harness.render_serial_sections(center, np.array([0., 0, 1.]),
                                   np.array([1., 0, 0]), np.array([0., 1, 0]))
    positions = np.asarray([item[1][0] for item in harness._plane_requests])
    assert len(positions) == 6
    assert len(np.unique(positions, axis=0)) == 6
    assert sum(view.note == "Outside traced arch" for view in harness.implant_serial_views) == 6
    np.testing.assert_array_equal(path, harness.pano_result.path)


def test_grid_close_restores_maximized_cell_and_invalidates_queued_frames(app, harness):
    source = harness.new_slice("Coronal")
    source.set_image(np.zeros((11, 11)), (harness.crosshair, [1., 0, 0], [0, 0, 1.], (12, 12)))
    record = harness.open_section_grid(source, 2, 2, 1.)
    cell = record["views"][0]
    harness.maximize_section(cell)
    assert harness._section_dialog_view is cell
    before = harness.slice_revision
    record["dialog"].close()
    assert harness._section_dialog is None
    assert harness.slice_revision > before
    assert not harness._section_grids
    assert all(view not in harness.all_slices for view in record["views"])
    app.processEvents()


@pytest.mark.parametrize("operation", ["close", "resize", "implant-resize"])
def test_grid_lifecycle_rejects_emitted_queue_frames_before_retiring_panes(app, harness, monkeypatch, operation):
    from asterview.slice_scheduler import LatestSliceQueue
    source = harness.new_slice("Axial")
    source.set_image(np.zeros((11, 11)), (harness.crosshair, [1., 0, 0], [0, 1., 0], (12, 12)))
    record = harness.open_section_grid(source, 2, 3, 1.)
    queue = harness.slice_queue = LatestSliceQueue()
    queue.request("already emitted pixels", context="source and layout")
    active = queue.take()
    queue.request("queued next pixels", context="source and layout")
    harness.slice_pending = "queued next pixels"
    worker = harness.slice_job = SimpleNamespace(cancelled=False)
    retiring = record["views"][-1]
    before_delete = []
    original_delete = retiring.deleteLater
    monkeypatch.setattr(retiring, "deleteLater",
                        lambda: (before_delete.append(queue.accepts(active.ticket, interacting=True)),
                                 original_delete()))
    if operation == "close":
        record["dialog"].close()
    elif operation == "resize":
        harness.open_section_grid(source, 2, 2, 1.)
        assert before_delete == [False]
    else:
        harness.implant_section_count.setValue(3)
    assert not queue.accepts(active.ticket, interacting=True)
    assert not queue.accepts(active.ticket, interacting=False)
    assert queue.pending is None and harness.slice_pending is None
    assert worker.cancelled
    # Ownership persists until the actual old worker ends; no second worker
    # can launch merely because the receiving widget was removed.
    assert queue.active is active
    assert queue.complete(active.ticket)
    harness.slice_job = None


def test_replacing_source_closes_bound_grid_and_cannot_render_old_volume(app, harness):
    source = harness.new_slice("Axial")
    source.set_image(np.zeros((11, 11)), (harness.crosshair, [1., 0, 0], [0, 1., 0], (12, 12)))
    harness.open_section_grid(source, 2, 2, 1.)
    harness.volume = Volume(np.ones((21, 21, 21), np.float32), (2, 2, 2), np.zeros(3), np.eye(3))
    harness._plane_requests = []
    harness.render_section_grids()
    assert not harness._section_grids
    assert not harness._plane_requests


@pytest.mark.parametrize("rows,columns,spacing", [(0, 3, 1), (9, 3, 1), (2, 13, 1), (2, 3, np.nan), (2, 3, .1)])
def test_invalid_grid_dimensions_are_rejected_before_creating_panes(app, harness, rows, columns, spacing):
    source = harness.new_slice("Axial")
    source.set_image(np.zeros((11, 11)), (harness.crosshair, [1., 0, 0], [0, 1., 0], (12, 12)))
    before = len(harness.all_slices)
    with pytest.raises(ValueError):
        harness.open_section_grid(source, rows, columns, spacing)
    assert len(harness.all_slices) == before
    assert not getattr(harness, "_section_grids", [])
