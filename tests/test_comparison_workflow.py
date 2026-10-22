"""Actual comparison navigation and asynchronous source selection regressions."""
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QTimer, QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QMainWindow

from asterview.comparison_controls import ComparisonControls, volume_fingerprint
from asterview.model import Volume
from asterview.mpr_geometry import plane_frame, rotate_about_view


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def source():
    z, y, x = np.indices((30, 32, 34), dtype=np.float32)
    return Volume(x * 10 + y * 5 + z * 2, (1, 1.1, 1.2),
                  np.array([-17, -17, -18]), np.eye(3), {"synthetic": True})


class Workstation(QMainWindow):
    def __init__(self):
        super().__init__()
        self.volume = source()
        self.crosshair = self.volume.center.copy()
        self.correction = rotate_about_view(np.eye(3), "Axial", 23)
        self.job = None
        self.operations = []
        self.renders = 0

    def volume_fingerprint(self):
        return volume_fingerprint(self.volume)

    def rotation(self):
        return self.correction

    def rotate_planes(self, plane, angle):
        self.correction = rotate_about_view(self.correction, plane, angle)

    def refresh_references(self):
        pass

    def schedule_render(self):
        self.renders += 1

    def start_job(self, function, callback):
        if self.job is not None:
            return False
        self.job = SimpleNamespace(cancelled=False, callback=callback, function=function)
        self.operations.append(self.job)
        return True

    def complete(self, controls, result):
        self.job.callback(result)
        self.job = None
        controls._job_finished()


@pytest.fixture
def controls(qapp):
    workstation = Workstation()
    controller = ComparisonControls(workstation)
    yield controller
    controller.job_poll.stop()
    controller.reset_comparison()
    controller._executor.shutdown(wait=True, cancel_futures=True)
    workstation.close()
    qapp.processEvents()


def wait_planes(qapp, controls):
    deadline = time.monotonic() + 6
    while (controls._future is not None or controls._pending_planes is not None
           or controls.dialog.resize_timer.isActive()) and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(.005)
    assert controls._future is None, "Comparison sampling did not finish"
    assert controls.dialog.cached_planes is not None, controls.dialog.notice.text()


def test_manual_start_auto_loads_one_series_and_opens_review(controls, monkeypatch):
    opened = []
    monkeypatch.setattr(controls, "open_comparison", lambda: opened.append("review"))
    controls.begin_workflow("C:/public-reference", "Manual")
    assert controls._busy
    assert len(controls.workstation.operations) == 1
    controls.workstation.complete(controls, [SimpleNamespace(label="Research CT")])
    assert len(controls.workstation.operations) == 2
    controls.workstation.complete(controls, source())
    assert controls.result.review == "pending"
    assert opened == ["review"]
    assert not controls._busy
    assert controls.panel.fit_mode.currentText() == "Manual"


def test_automatic_start_aligns_only_after_loading_finished(controls, monkeypatch):
    actions = []
    monkeypatch.setattr(controls, "open_comparison", lambda: actions.append("open"))
    monkeypatch.setattr(controls, "align", lambda: actions.append("align"))
    controls.begin_workflow("C:/public-reference", "Automatic")
    controls.workstation.complete(controls, [SimpleNamespace(label="Research CT")])
    assert actions == []
    controls.workstation.complete(controls, source())
    assert actions == ["open", "align"]


def test_multiple_series_remains_explicit_and_missing_folder_starts_no_job(controls, monkeypatch):
    controls.begin_workflow("C:/public-reference", "Manual")
    controls.workstation.complete(controls, [SimpleNamespace(label="Scout"), SimpleNamespace(label="3D CT")])
    assert len(controls.workstation.operations) == 1
    assert controls.panel.series.count() == 2
    assert controls.panel.load_button.isEnabled()
    monkeypatch.setattr("asterview.comparison_controls.QFileDialog.getExistingDirectory", lambda *args: "")
    controls.begin_workflow()
    assert len(controls.workstation.operations) == 1


def test_reset_discards_discovery_and_cancellation_discards_chained_load(controls):
    controls.begin_workflow("C:/public-reference", "Manual")
    controls.reset_comparison()
    controls.workstation.complete(controls, [SimpleNamespace(label="Stale CT")])
    assert controls.series == []
    controls.begin_workflow("C:/public-reference", "Manual")
    controls.workstation.job.callback([SimpleNamespace(label="CT")])
    controls.workstation.job.cancelled = True
    controls.workstation.job = None
    controls._job_finished()
    assert len(controls.workstation.operations) == 2
    assert controls.moving is None


def test_fixed_source_replacement_discards_load(controls):
    controls.begin_workflow("C:/public-reference", "Manual")
    controls.workstation.complete(controls, [SimpleNamespace(label="Research CT")])
    controls.workstation.volume = source()
    controls.workstation.complete(controls, source())
    assert controls.moving is None


def test_late_automatic_fit_cannot_overwrite_manual_correction(controls, monkeypatch):
    monkeypatch.setattr("asterview.comparison_controls.detect_registration_status", lambda: {"ready": True})
    monkeypatch.setattr(controls, "open_comparison", lambda: None)
    controls.set_moving(source())
    controls.align()
    automatic_result = controls.result.refined([8, 0, 0])
    controls.panel.manual[0].setValue(1.5)
    controls.apply_manual()
    manual_result = controls.result
    controls.workstation.complete(controls, automatic_result)
    assert controls.result is manual_result
    np.testing.assert_allclose(controls.result.transform[:3, 3], [1.5, 0, 0])


def test_corrected_comparison_scroll_and_drag_update_both_sources_and_main(qapp, controls):
    fixed = controls.workstation.volume
    moving = source()
    before = fixed.data.copy(), moving.data.copy()
    controls.set_moving(moving)
    controls.open_comparison()
    wait_planes(qapp, controls)
    dialog = controls.dialog
    assert dialog.link_main.isChecked()
    for (name, first, second), (frame, fixed_plane, moving_plane, valid) in zip(dialog.views, dialog.cached_planes):
        right, up = plane_frame(name, controls.workstation.rotation())
        np.testing.assert_allclose(frame[1], right)
        np.testing.assert_allclose(frame[2], up)
        np.testing.assert_allclose(fixed_plane, moving_plane)
        assert first.reference_lines and second.reference_lines
    previous = dialog.center.copy()
    right, up = plane_frame("Coronal", dialog.rotation)
    dialog.views[1][2].scrolled.emit(1)
    np.testing.assert_allclose(dialog.center, previous + np.cross(right, up))
    np.testing.assert_allclose(controls.workstation.crosshair, dialog.center)
    assert dialog.link_main.isChecked()
    corrected = rotate_about_view(dialog.rotation, "Sagittal", 12)
    dialog.views[2][1].reference_rotated.emit(12)
    np.testing.assert_allclose(dialog.rotation, corrected)
    np.testing.assert_allclose(controls.workstation.rotation(), corrected)
    previous = dialog.center.copy()
    right, up = plane_frame("Axial", dialog.rotation)
    dialog.views[0][2].reference_translated.emit({"delta": .3 * right + .2 * up, "target": None})
    np.testing.assert_allclose(dialog.center, previous + .3 * right + .2 * up)
    wait_planes(qapp, controls)
    assert controls.workstation.renders > 0
    np.testing.assert_array_equal(fixed.data, before[0])
    np.testing.assert_array_equal(moving.data, before[1])


def test_main_correction_sync_and_unlinked_navigation_are_explicit(qapp, controls):
    controls.set_moving(source())
    controls.open_comparison()
    dialog = controls.dialog
    controls.workstation.correction = rotate_about_view(np.eye(3), "Coronal", 17)
    controls.workstation.crosshair += [1, 0, 0]
    dialog.sync_position()
    np.testing.assert_allclose(dialog.rotation, controls.workstation.rotation())
    np.testing.assert_allclose(dialog.center, controls.workstation.crosshair)
    dialog.link_main.setChecked(False)
    old = controls.workstation.crosshair.copy()
    old_rotation = controls.workstation.rotation().copy()
    dialog.scroll("Axial", 2)
    dialog.rotate("Coronal", 4)
    np.testing.assert_allclose(controls.workstation.crosshair, old)
    np.testing.assert_allclose(controls.workstation.rotation(), old_rotation)
    dialog.link_main.setChecked(True)
    np.testing.assert_allclose(dialog.center, old)
    np.testing.assert_allclose(dialog.rotation, old_rotation)


def test_accept_moves_review_into_workspace_and_keeps_manual_refinement_available(qapp, controls):
    controls.set_moving(source())
    controls.open_comparison()
    fitting = controls.dialog
    assert fitting.fitting and fitting.isWindow()
    fitting.fit_values[0].setValue(1.2)
    fitting.accept.click()
    review = controls.dialog
    assert controls.result.review == "accepted"
    assert controls.result.transform[0, 3] == pytest.approx(1.2)
    assert not review.fitting and not review.isWindow()
    assert review.parentWidget() is controls.workspace
    assert review.mode.currentText() == "Registered moving"
    assert review.fitting_controls.isHidden()
    wait_planes(qapp, controls)
    for _, first, second in review.views:
        assert first.frame is not None and second.frame is not None
        for one, two in zip(first.frame, second.frame):
            np.testing.assert_allclose(one, two)
        assert first.geometry().top() < second.geometry().top()
    review.refine_button.click()
    assert controls.dialog.fitting
    controls.dialog.fit_values[2].setValue(-.8)
    controls.dialog.apply.click()
    assert controls.result.review == "pending"
    controls.dialog.accept.click()
    assert controls.result.review == "accepted" and not controls.dialog.fitting
    assert controls.result.transform[2, 3] == pytest.approx(-.8)


def test_scroll_reuses_unchanged_planes_and_recent_positions(qapp, controls, monkeypatch):
    import asterview.comparison_controls as module
    controls.set_moving(source())
    controls.open_comparison()
    wait_planes(qapp, controls)
    dialog = controls.dialog
    calls = []
    original = module.sample_registered_plane
    def counted(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "sample_registered_plane", counted)
    dialog.scroll("Axial", 1)
    wait_planes(qapp, controls)
    assert len(calls) == 2, "Only the changed axial plane needs two source samples"
    dialog.scroll("Axial", -1)
    wait_planes(qapp, controls)
    assert len(calls) == 2, "Returning to the prior physical position should reuse cached pixels"


def test_paired_thickness_and_sliders_keep_geometry_and_source_unchanged(qapp, controls):
    controls.set_moving(source())
    controls.open_comparison()
    wait_planes(qapp, controls)
    dialog = controls.dialog
    frame = tuple(x.copy() if isinstance(x, np.ndarray) else x for x in dialog.cached_planes[0][0])
    _, first, second = dialog.views[0]
    second.set_section_settings(2.4, "max")
    wait_planes(qapp, controls)
    assert first.slab_mm == second.slab_mm == 2.4
    assert first.slab_mode == second.slab_mode == "max"
    for left, right in zip(dialog.cached_planes[0][0], frame):
        np.testing.assert_allclose(left, right)
    fixed_pixels = first.gray.copy()
    dialog.windows[0].setValue(dialog.windows[0].minimum())
    assert not np.array_equal(first.gray, fixed_pixels)
    dialog.maximize(second)
    assert first.isHidden() and not second.isHidden()
    dialog.maximize(second)
    assert not first.isHidden()


def test_navigation_stops_at_fixed_fov_without_independent_axis_clipping(qapp, controls):
    controls.set_moving(source())
    controls.open_comparison()
    dialog = controls.dialog
    start = dialog.center.copy()
    right, up = plane_frame("Coronal", dialog.rotation)
    normal = np.cross(right, up)
    dialog.scroll("Coronal", 10000)
    index = controls.workstation.volume.world_to_index(dialog.center)
    assert np.all(index >= -1e-7)
    assert np.all(index <= np.asarray(controls.workstation.volume.dimensions) - 1 + 1e-7)
    movement = dialog.center - start
    np.testing.assert_allclose(movement / np.linalg.norm(movement), normal)


@pytest.mark.parametrize("plane_index,source_index", [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)])
def test_comparison_physical_sliders_move_exact_shared_oblique_plane(qapp, controls, plane_index, source_index):
    fixed = controls.workstation.volume
    moving = source()
    original = fixed.data.copy(), moving.data.copy()
    controls.workstation.correction = rotate_about_view(controls.workstation.correction, "Coronal", 19)
    controls.set_moving(moving)
    controls.open_comparison()
    wait_planes(qapp, controls)
    dialog = controls.dialog
    plane, first, second = dialog.views[plane_index]
    view = dialog.views[plane_index][source_index]
    right, up = plane_frame(plane, dialog.displayed_rotation)
    normal = np.cross(right, up)
    start = dialog.center.copy()
    low, high = view.position_slider.physical_limits
    # Acquired-box limits must terminate on a source face, including oblique
    # views; the source plane is not an independently clipped index triplet.
    for value in (low, high):
        endpoint = start + normal * (value - start @ normal)
        index = fixed.world_to_index(endpoint)
        assert np.all(index >= -1e-7)
        assert np.all(index <= np.asarray(fixed.dimensions) - 1 + 1e-7)
        assert np.any(np.isclose(index, 0) | np.isclose(index, np.asarray(fixed.dimensions) - 1))
    assert all(pane._position_visible and pane.position_slider.isVisible()
               for _, a, b in dialog.views for pane in (a, b))
    emissions = []
    second.position_changed.connect(emissions.append)
    dialog.refresh_references()
    assert emissions == [], "Position synchronization must not feed back into navigation"
    slider = view.position_slider
    slider.setValue(round(.64 * slider.maximum()))
    target = low + .64 * (high - low)
    expected = start + normal * (target - start @ normal)
    np.testing.assert_allclose(dialog.center, expected)
    np.testing.assert_allclose(controls.workstation.crosshair, expected)
    wait_planes(qapp, controls)
    frame, fixed_image, moving_image, valid = dialog.cached_planes[plane_index]
    assert abs((frame[0] - expected) @ normal) < 1e-8
    np.testing.assert_array_equal(fixed_image, moving_image)
    for a, b in zip(first.frame, second.frame):
        np.testing.assert_allclose(a, b)
    assert first.position_slider.physical_value == pytest.approx(target)
    assert second.position_slider.physical_value == pytest.approx(target)
    # Wheel navigation remains in step with the same slider after correction.
    second.scrolled.emit(1)
    wait_planes(qapp, controls)
    assert first.position_slider.physical_value == pytest.approx(target + min(fixed.spacing))
    np.testing.assert_array_equal(fixed.data, original[0])
    np.testing.assert_array_equal(moving.data, original[1])


def test_sampling_stale_generation_does_not_install_after_source_reset(qapp, controls):
    controls.set_moving(source())
    controls.open_comparison()
    controls.reset_comparison()
    deadline = time.monotonic() + 2
    while controls._future is not None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(.005)
    assert controls.dialog is None and controls._future is None


def test_scroll_retains_unchanged_images_but_updates_all_crosshairs_and_display_windows(qapp, controls):
    controls.set_moving(source())
    controls.open_comparison()
    wait_planes(qapp, controls)
    dialog = controls.dialog
    dialog.mode.setCurrentText("Registered moving")
    images = [(first.image, second.image) for _, first, second in dialog.views]
    dialog.scroll("Axial", .5)
    wait_planes(qapp, controls)
    for index, (_, first, second) in enumerate(dialog.views):
        if index == 0:
            assert first.image is not images[index][0]
            assert second.image is not images[index][1]
        else:
            assert first.image is images[index][0]
            assert second.image is images[index][1]
        np.testing.assert_allclose(first.crosshair, dialog.center)
        np.testing.assert_allclose(second.crosshair, dialog.center)

    moving_images = [second.image for _, _, second in dialog.views]
    reference_pixels = [first.gray.copy() for _, first, _ in dialog.views]
    dialog.windows[0].setValue(dialog.windows[0].minimum())
    assert any(not np.array_equal(first.gray, old)
               for (_, first, _), old in zip(dialog.views, reference_pixels))
    assert all(second.image is old for (_, _, second), old in zip(dialog.views, moving_images))

    dialog.mode.setCurrentText("Blend")
    blended = [second.gray.copy() for _, _, second in dialog.views]
    dialog.windows[0].setValue(dialog.windows[0].maximum())
    assert any(not np.array_equal(second.gray, old)
               for (_, _, second), old in zip(dialog.views, blended))


def test_continuous_scroll_displays_slow_frames_before_input_stops(qapp, controls, monkeypatch):
    import asterview.comparison_controls as module
    controls.set_moving(source())
    controls.open_comparison()
    wait_planes(qapp, controls)
    dialog = controls.dialog
    real_sample = module.sample_registered_plane
    def slow_sample(*args, **kwargs):
        time.sleep(.01)  # Six planes take at least 60 ms; input arrives at 16 ms.
        return real_sample(*args, **kwargs)
    monkeypatch.setattr(module, "sample_registered_plane", slow_sample)
    completed = []
    real_display = dialog.display_cached
    def observe(*args):
        completed.append((time.monotonic(), dialog.cached_planes[0][0][0].copy()))
        real_display(*args)
    monkeypatch.setattr(dialog, "display_cached", observe)
    timer = QTimer()
    timer.setInterval(16)
    timer.timeout.connect(lambda: dialog.scroll("Axial", .15))
    initial_generation = controls._generation
    timer.start()
    deadline = time.monotonic() + .7
    try:
        while time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(.002)
        # Continuous input is still running. A cancel-on-input loop would
        # produce zero completed frames until the timer was stopped.
        assert timer.isActive()
        assert len(completed) >= 3
        assert controls._generation == initial_generation
        assert not np.allclose(completed[0][1], completed[-1][1])
        assert controls._pending_planes is not None or controls._future is not None
    finally:
        timer.stop()
    wait_planes(qapp, controls)
    for frame, _, _, _ in dialog.cached_planes:
        normal = np.cross(frame[1], frame[2])
        assert np.dot(frame[0] - dialog.center, normal) == pytest.approx(0, abs=1e-9)
        np.testing.assert_allclose(np.cross(frame[0] - controls.workstation.volume.center, normal), 0, atol=1e-9)
    assert controls._pending_planes is None


def test_unused_comparison_reset_and_shutdown_never_start_workers(controls):
    for _ in range(4):
        controls.reset_comparison()
    assert not controls._executor._threads
    controls.shutdown()
    controls.shutdown()
    assert controls._closed
    assert not controls.poll.isActive() and not controls.job_poll.isActive()
    assert not controls._executor._threads


def test_shutdown_cancels_active_sampling_and_joins_only_owned_worker(qapp, controls, monkeypatch):
    import threading
    import asterview.comparison_controls as module
    controls.set_moving(source())
    controls.open_comparison()
    wait_planes(qapp, controls)
    real_sample = module.sample_registered_plane
    started = threading.Event()
    def slow_sample(*args, **kwargs):
        started.set()
        time.sleep(.03)
        return real_sample(*args, **kwargs)
    monkeypatch.setattr(module, "sample_registered_plane", slow_sample)
    controls.dialog.scroll("Axial", .5)
    assert started.wait(2)
    owned = list(controls._executor._threads)
    assert owned and any(thread.is_alive() for thread in owned)
    controls.shutdown()
    assert all(not thread.is_alive() for thread in owned)
    assert controls.dialog is None
    assert controls._future is None and controls._pending_planes is None
    assert not controls.poll.isActive() and not controls.job_poll.isActive()
    controls.reset_comparison()  # No task submission against a stopped executor.
    controls.shutdown()


def test_qobject_destruction_fallback_retires_executor_without_owner_capture(qapp):
    workstation = Workstation()
    controller = ComparisonControls(workstation)
    assert controller._executor.submit(lambda: 1).result(timeout=2) == 1
    owned = list(controller._executor._threads)
    workstation.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    deadline = time.monotonic() + 2
    while any(thread.is_alive() for thread in owned) and time.monotonic() < deadline:
        time.sleep(.005)
    assert all(not thread.is_alive() for thread in owned)
