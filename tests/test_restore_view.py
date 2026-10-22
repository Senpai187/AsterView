"""Restore resets display/navigation while retaining authored review objects."""
import time
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QLabel

from asterview.app import MainWindow


def settle(app, window):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        app.processEvents()
        if window.slice_job is None and window.slice_pending is None and not window.render_timer.isActive():
            return
        time.sleep(.002)
    raise AssertionError("Slices failed to settle")


@pytest.fixture
def viewer():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.load_demo()
    settle(app, window)
    yield app, window
    settle(app, window)
    window.close()
    app.processEvents()


def test_restore_resets_initial_view_without_copying_source_or_deleting_plans(viewer):
    app, window = viewer
    source = window.volume.data
    initial = window.capture_view_defaults()
    window.add_implant()
    plans = [dict(item) for item in window.implants]
    window.window_level.setValue(1800)
    window.window_width.setValue(500)
    window.header_thickness.setValue(3.2)
    window.rotation_spins[0].setValue(27)
    window.crosshair += [3, 5, 8]
    window.mpr_views["Axial"].slab_mm = 5
    window.mpr_views["Axial"].zoom = 2
    window.threshold.setValue(200)
    window.lines_visible.setChecked(False)
    window.volume_visible.setChecked(False)
    window.tmj_offset.setValue(10)
    window.maximize_view(window.mpr_views["Axial"])
    window.reset_workspace()
    settle(app, window)
    assert window.volume.data is source
    assert window.implants == plans
    assert window.capture_view_defaults() == initial
    np.testing.assert_allclose(window.crosshair, window.volume.center)
    np.testing.assert_allclose(window.rotation(), np.eye(3), atol=1e-10)
    assert window._maximized_view is None
    assert window.tabs.currentIndex() == 0
    assert all(v.slab_mm is None and v.zoom == 1 for v in window.mpr_views.values())
    native_pitch = min(window.volume.spacing)
    assert window.header_thickness.value() == window.slab.value() == pytest.approx(native_pitch)
    assert window.header_thickness.minimum() == pytest.approx(native_pitch)


def test_display_labels_and_patient_name_are_presentable(viewer):
    app, window = viewer
    window.volume.metadata["patient_name"] = "Example^Synthetic"
    window.volume.metadata["synthetic"] = False
    window.volume.metadata["series_description"] = "Display formatting fixture"
    window.set_volume(window.volume)
    settle(app, window)
    assert "Synthetic Example" in window.study_label.text()
    assert window.volume.metadata["patient_name"] == "Example^Synthetic"
    assert "Research" not in window.windowTitle()
    assert not any("NOT FOR CLINICAL USE" in label.text().upper()
                   for label in window.findChildren(QLabel) if label.isVisibleTo(window))
    assert window.header_thickness.accessibleName() == "Slice thickness"
    window.slab.setValue(1.2)
    assert window.header_thickness.value() == 1.2
    window.header_thickness.setValue(2.4)
    assert window.slab.value() == 2.4
