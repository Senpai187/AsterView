"""Integrated patient-coordinate workflows and asynchronous UI result safety."""
import copy
import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from asterview.app import MainWindow
from asterview.reformat import curved_planar
from asterview.slice_view import SliceView


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def settle(app, window):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        app.processEvents()
        if not window.render_timer.isActive() and window.slice_pending is None and not (window.slice_job and window.slice_job.isRunning()):
            app.processEvents()
            return
        time.sleep(.002)
    raise AssertionError("Background slices did not settle")


@pytest.fixture
def window(app):
    result = MainWindow()
    result.load_demo()
    settle(app, result)
    yield result
    settle(app, result)
    result.close()
    app.processEvents()


def test_all_dental_workflows_use_full_source_geometry(app, window):
    assert all(view.image is not None for view in window.mpr_views.values())
    assert "SYNTHETIC" in window.study_label.text()
    window.add_implant()
    window.tabs.setCurrentIndex(2)
    settle(app, window)
    assert all(view.image is not None for view in window.implant_views)
    np.testing.assert_allclose(window.implant_views[0].frame[0], window.implants[0]["center"])
    center = window.volume.center
    window.tmj_centers = {"Right": center + [-30, 20, 0], "Left": center + [30, 20, 0]}
    window.tabs.setCurrentIndex(3)
    settle(app, window)
    for side in ("Right", "Left"):
        for plane in ("Sagittal", "Coronal"):
            view = window.tmj_views[(side, plane)]
            assert view.image is not None
            np.testing.assert_allclose(view.crosshair, window.tmj_centers[side])
    window.tabs.setCurrentIndex(1)
    window.arch_points = [center + p for p in ([-30, 20, 0], [0, -30, 0], [30, 20, 0])]
    result = curved_planar(window.volume, window.arch_points, [0, 0, 1], rows=90, samples=100)
    window.panorama_ready((window.pano_revision, id(window.volume), np.array([0,0,1]), 5, "mean", result))
    settle(app, window)
    assert window.pano_sagittal_view.image is not None
    from asterview.mpr_geometry import plane_frame
    right, up = plane_frame("Sagittal", window.rotation())
    np.testing.assert_allclose(window.pano_sagittal_view.frame[1], right)
    np.testing.assert_allclose(window.pano_sagittal_view.frame[2], up)
    np.testing.assert_allclose(window.pano_sagittal_view.crosshair, window.crosshair)
    assert window.pano_view.physical_size_mm == pytest.approx((result.column_pitch_mm * 99, result.row_pitch_mm * 89))


def test_stale_panorama_cannot_restore_cleared_arch(app, window):
    points = [[-20, 10, 0], [0, -20, 0], [20, 10, 0]]
    result = curved_planar(window.volume, points, [0,0,1], rows=25, samples=25)
    revision = window.pano_revision
    window.clear_arch()
    window.panorama_ready((revision, id(window.volume), np.array([0,0,1]), 5, "mean", result))
    assert window.pano_result is None


def test_pixel_center_world_mapping_and_measurement_plane_lock(app):
    view = SliceView("Test")
    view.resize(600, 480)
    frame = (np.zeros(3), np.array([1,0,0]), np.array([0,1,0]), (80,40))
    view.set_image(np.zeros((41,81), dtype=np.float32), frame)
    rect = view._rect()
    first = view.world_to_screen([-40, 20, 0])
    assert first.x() == pytest.approx(rect.left() + rect.width() / 162)
    assert first.y() == pytest.approx(rect.top() + rect.height() / 82)
    np.testing.assert_allclose(view.screen_to_world(first), [-40,20,0])
    view.pending_measure = np.zeros(3)
    view.set_image(view.data, (np.array([0,0,1]), *frame[1:]))
    assert view.pending_measure is None
    view.close()


def test_uncalibrated_single_image_disables_physical_tools(app, window):
    window.volume.metadata["geometry_calibrated"] = False
    window.tool_combo.setCurrentText("Measure")
    assert window.mode == "Navigate"
    window.add_implant()
    assert window.implants == []


def test_session_validation_is_atomic_before_live_mutation(app, window):
    state = {"schema":1,"volume":window.volume_fingerprint(),"crosshair":[1,2,3],"rotation":[5,10,15],"window":[500,2500],"arch":[],"implants":[],"tmj_centers":{"Right":None,"Left":None},"tmj_angles":{"Right":0,"Left":0}}
    window.restore_session(state)
    np.testing.assert_allclose(window.crosshair,[1,2,3])
    invalid = copy.deepcopy(state)
    invalid["volume"]["origin"][0] += 1
    invalid["crosshair"] = [10,20,30]
    with pytest.raises(ValueError):
        window.restore_session(invalid)
    np.testing.assert_allclose(window.crosshair,[1,2,3])


def test_pending_slices_install_only_latest_revision(app, window, monkeypatch):
    import asterview.app as app_module
    original = app_module.sample_plane
    def slow(*args, **kwargs):
        time.sleep(.015)
        return original(*args, **kwargs)
    monkeypatch.setattr(app_module, "sample_plane", slow)
    window.rotation_spins[0].setValue(15)
    window.render_timer.stop()
    window.render_views()
    window.rotation_spins[0].setValue(35)
    window.render_timer.stop()
    window.render_views()
    settle(app, window)
    expected = app_module.base_frame("Coronal", window.rotation())
    np.testing.assert_allclose(window.mpr_views["Coronal"].frame[2], expected[1])


def test_plane_correction_preserves_magnification_and_field_of_view(app, window):
    window.preview_enabled.setChecked(False)
    originals = {name: (view.frame[3], view.data.shape) for name, view in window.mpr_views.items()}
    window.rotate_planes("Axial", 37)
    settle(app, window)
    for name, view in window.mpr_views.items():
        np.testing.assert_allclose(view.frame[3], originals[name][0])
        assert view.data.shape == originals[name][1]


def test_maximized_section_snapshot_skips_hidden_native_renderer(app, window, monkeypatch):
    window.show()
    settle(app, window)
    view = window.mpr_views["Axial"]
    window.maximize_view(view)
    settle(app, window)
    assert view.isVisible() and not window.volume_view.isVisible()

    def hidden_capture():
        raise AssertionError("A hidden native renderer cannot supply snapshot pixels")

    monkeypatch.setattr(window.volume_view, "capture_image", hidden_capture)
    snapshot = window.workspace_snapshot()
    assert not snapshot.isNull()
    window.maximize_view(view)
    settle(app, window)
    assert all(widget.isVisible() for widget in window.mpr_layout)
