"""Actual range dialog and production reconstruction dispatch without VTK."""
import json

import numpy as np
import pytest
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QApplication, QWidget, QDoubleSpinBox, QComboBox,
                               QSlider, QDialogButtonBox, QPushButton, QFileDialog)

from asterview.app import MainWindow
from asterview.model import Volume
from asterview.reformat import sample_plane
from asterview.slice_view import SliceView
from asterview.thickness_control import ThicknessControl
from asterview.workflow_controls import WorkflowControls


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class RangeHarness(QWidget, WorkflowControls):
    panorama_range_geometry = MainWindow.panorama_range_geometry
    current_panorama_bounds = MainWindow.current_panorama_bounds
    sync_panorama_range_overlay = MainWindow.sync_panorama_range_overlay
    define_data_range = MainWindow.define_data_range
    apply_panorama_range = MainWindow.apply_panorama_range
    generate_panorama = MainWindow.generate_panorama
    panorama_sample_shape = MainWindow.panorama_sample_shape
    panorama_ready = MainWindow.panorama_ready
    panorama_needs_detail = MainWindow.panorama_needs_detail
    update_panorama_cursor = MainWindow.update_panorama_cursor
    volume_fingerprint = MainWindow.volume_fingerprint
    save_session = MainWindow.save_session
    restore_session = MainWindow.restore_session

    def __init__(self):
        super().__init__()
        z, _, _ = np.mgrid[:51, :51, :51]
        self.volume = Volume(z.astype(np.float32), (1., 1., 1.), np.zeros(3),
                             np.eye(3), metadata={"synthetic": True})
        self.arch_points = [np.array(point, float) for point in ([15, 20, 25], [25, 15, 25], [35, 20, 25])]
        self.crosshair = self.volume.center.copy()
        self.pano_up = np.array([0., 0, 1.])
        self._pano_bounds = (-45., 45.)
        self.pano_result = None
        self.pano_revision = 0
        self.pano_view, self.implant_pano_view, self.pano_sagittal_view = [SliceView(name) for name in ("Panorama", "Implant panorama", "Range")]
        frame = (self.volume.center, np.array([0., 1, 0]), np.array([0., 0, 1.]), (50, 50))
        self.pano_sagittal_view.set_image(sample_plane(self.volume, *frame, shape=(101, 101)), frame)
        self.pano_sagittal_view.range_bounds_changed.connect(self.apply_panorama_range)
        self.pano_thickness = ThicknessControl()
        self.pano_thickness.configure_for_spacing(self.volume.spacing)
        self.pano_projection = QComboBox()
        self.pano_projection.addItems(["mean", "max", "min"])
        self.pano_offset = QDoubleSpinBox()
        self.pano_offset.setRange(-20, 20)
        self.pano_index = QSlider(Qt.Orientation.Horizontal)
        self.pano_index.setRange(0, 20)
        self.pano_index.setValue(10)
        self.pano_refresh_timer = QTimer(self)
        self.pano_refresh_timer.setSingleShot(True)
        self.pano_refresh_timer.setInterval(10000)
        self.resolution = QComboBox()
        self.resolution.addItems(["Efficient", "Native", "Fine"])
        self.resolution.setCurrentIndex(1)
        self.window_level, self.window_width = QDoubleSpinBox(), QDoubleSpinBox()
        self.window_width.setRange(1, 10000)
        self.window_width.setValue(1000)
        self.rotation_spins = [QDoubleSpinBox() for _ in range(3)]
        self.implants, self.implant_margins = [], {}
        self.tmj_centers = {"Right": None, "Left": None}
        self.tmj_angles = {side: QDoubleSpinBox() for side in ("Right", "Left")}
        self.implant_section_width = QDoubleSpinBox()
        self.implant_section_width.setRange(5, 150)
        self.implant_section_width.setValue(24)
        self.job = None
        self._closing = False
        self.renders = 0
        self.messages = []

    def rotation(self):
        return np.eye(3)

    def schedule_render(self, *_):
        self.renders += 1

    def start_job(self, compute, install):
        self.compute, self.install = compute, install

    def statusBar(self):
        return self

    def showMessage(self, message):
        self.messages.append(message)

    def refresh_implants(self, *_):
        pass

    def refresh_references(self):
        pass

    def clear_arch(self):
        self.arch_points = []
        self._pano_bounds = (-45, 45)
        self.pano_result = None


@pytest.fixture
def viewer(app):
    window = RangeHarness()
    yield window
    window.pano_refresh_timer.stop()
    dialog = getattr(window, "_pano_range_dialog", None)
    if dialog is not None:
        dialog.reject()
    for pane in (window.pano_sagittal_view, window.pano_view, window.implant_pano_view):
        pane.close()
    window.close()
    app.processEvents()


def test_range_dialog_presets_and_numeric_apply_control_real_reconstruction(viewer, app):
    dialog = viewer.define_data_range()
    assert viewer.define_data_range() is dialog
    assert (dialog.lower.value(), dialog.upper.value()) == (-25., 25.)
    button = next(item for item in dialog.findChildren(QPushButton) if item.text() == "Above arch")
    button.click()
    assert (dialog.lower.value(), dialog.upper.value()) == (0., 25.)
    dialog.lower.setValue(-3.)
    dialog.upper.setValue(9.)
    dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).click()
    assert viewer._pano_bounds == (-3., 9.)
    assert viewer._pano_offset_dirty and viewer.pano_revision == 1
    assert viewer.pano_sagittal_view.data_range["bounds"] == (-3., 9.)
    viewer.generate_panorama()
    payload = viewer.compute(lambda *_: None, lambda: False)
    result = payload[-1]
    assert result.vertical_limits_mm == (-3., 9.)
    np.testing.assert_allclose(result.image[0], 34., atol=1e-5)
    np.testing.assert_allclose(result.image[-1], 22., atol=1e-5)
    viewer.install(payload)
    assert viewer.pano_result is result
    assert viewer.pano_view.data is result.image


def test_range_edit_rejects_old_completed_result_without_blank_during_drag(viewer):
    viewer.generate_panorama()
    payload = viewer.compute(lambda *_: None, lambda: False)
    old = payload[-1]
    viewer.pano_result = old
    viewer.apply_panorama_range(-8, 12)
    assert viewer.pano_result is old
    viewer.install(payload)
    assert viewer.pano_result is old
    assert "changed" in viewer.messages[-1]


def test_range_cancel_and_inverted_inputs_leave_live_state_unchanged(viewer):
    dialog = viewer.define_data_range()
    dialog.lower.setValue(5)
    dialog.upper.setValue(2)
    assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    dialog.reject()
    assert viewer._pano_bounds == (-45., 45.)
    with pytest.raises(ValueError):
        viewer.apply_panorama_range(3, 2)
    assert viewer.pano_revision == 0


def test_oblique_source_limits_round_outward_without_exceeding_acquired_range(viewer):
    angle = np.deg2rad(17.)
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    viewer.rotation = lambda: rotation
    viewer.arch_points = [point + np.array([.0004, 0, .0004]) for point in viewer.arch_points]
    dialog = viewer.define_data_range()
    dialog.set_bounds(dialog.limits)
    bounds = dialog.preview.data_range["bounds"]
    assert bounds[0] >= dialog.limits[0]
    assert bounds[1] <= dialog.limits[1]
    dialog.apply_range()
    np.testing.assert_allclose(viewer._pano_bounds, bounds)


def test_session_roundtrip_restores_physical_crop_and_overlay(viewer, tmp_path, monkeypatch):
    viewer.apply_panorama_range(-7.5, 11.25)
    target = tmp_path / "range-session.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_: (str(target), ""))
    viewer.save_session()
    payload = json.loads(target.read_text())
    assert payload["viewing"]["pano_bounds"] == [-7.5, 11.25]
    viewer._pano_bounds = (-2, 2)
    viewer.restore_session(payload)
    assert viewer._pano_bounds == (-7.5, 11.25)
    assert viewer.pano_sagittal_view.data_range["bounds"] == (-7.5, 11.25)
