"""Qt-to-source-grid segmentation review and cancellation integration.

Inference is replaced with a deterministic local test double. These tests do
not load a neural network, download a model, or measure diagnostic accuracy.
"""
import copy
from hashlib import sha256
import threading
import time

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QMessageBox
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
from vtkmodules.util.numpy_support import vtk_to_numpy

from asterview.app import MainWindow
from asterview.dicom_io import SeriesInfo, load_series
from asterview.model import Volume
from asterview.reformat import rotation_matrix
from asterview.segmentation import SegmentationResult, create_result, export_segmentation


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def wait_for(app, predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            app.processEvents()
            return
        time.sleep(.002)
    raise AssertionError("Segmentation UI did not reach the expected state")


def settle(app, window):
    wait_for(app, lambda: window.job is None and window.slice_job is None
             and window.slice_pending is None and not window.render_timer.isActive()
             and not window.mask_timer.isActive())


def small_volume(*, reflected=False, shift=0):
    direction = rotation_matrix(13, -21, 29)
    if reflected:
        direction[:, 0] *= -1
    return Volume(np.arange(17 * 19 * 21, dtype=np.float32).reshape(17, 19, 21),
                  (.5, .8, 1.3), np.array([12.1 + shift, -18.6, 9.3]), direction,
                  metadata={"synthetic": True, "modality": "CT"})


def candidates(window, *, empty=False):
    data = np.zeros(window.volume.data.shape, dtype=np.uint16)
    if not empty:
        z, y, x = np.indices(data.shape)
        data[(x + y + z) % 4 == 0] = 3
        data[(x + 2 * y + z) % 5 == 0] = 5
    return create_result(window.volume, window.volume_fingerprint(), data,
                         {2: {"name": "Mandible", "kind": "jaw", "color": "#A080D0"},
                          3: {"name": "Teeth group", "kind": "tooth", "color": "#60C0A0"},
                          5: {"name": "Canal proxy", "kind": "canal", "color": "#E08090"}},
                         {"name": "Deterministic integration fixture", "version": "1"})


@pytest.fixture
def viewer(qt_app, monkeypatch):
    import asterview.inference as inference
    monkeypatch.setattr(inference, "detect_model_status", lambda **_: {"ready": True, "reason": "Test runtime only"})
    errors = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, title, text: errors.append((title, text)))
    window = MainWindow()
    window.set_volume(small_volume())
    settle(qt_app, window)
    for view in window.mpr_views.values():
        view.resize(640, 560)
    window._test_errors = errors
    yield window
    window._wheel_settle.stop()
    window.finish_mask_stroke()
    if window._interaction_active:
        window.end_interaction()
    settle(qt_app, window)
    window.close()
    qt_app.processEvents()
    assert not errors, errors


def install(window, result=None):
    result = candidates(window) if result is None else result
    window.segmentation_ready((id(window.volume), result))
    return result


def test_hidden_objects_skip_mask_resampling(viewer, monkeypatch):
    result = install(viewer)
    viewer.segmentation_panel.visible_ids.clear()
    def unexpected_sample(*args, **kwargs):
        raise AssertionError("Hidden segmentation must not resample the volume")
    monkeypatch.setattr(type(result), "sample_plane", unexpected_sample)
    viewer.refresh_segmentation(update_3d=False)
    assert all(view.segmentation_image is None for view in viewer.mpr_views.values())


def select_label(window, key):
    panel = window.segmentation_panel
    for row in range(panel.objects.count()):
        item = panel.objects.item(row)
        if item.data(Qt.ItemDataRole.UserRole) == key:
            panel.objects.setCurrentItem(item)
            return item
    raise AssertionError(f"No semantic object {key} in the browser")


def mouse(view, event_type, position, button, buttons):
    event = QMouseEvent(event_type, QPointF(position), QPointF(position), button,
                        buttons, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(view, event)


def press(view, position):
    mouse(view, QEvent.Type.MouseButtonPress, position, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)


def move(view, position):
    mouse(view, QEvent.Type.MouseMove, position, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)


def release(view, position):
    mouse(view, QEvent.Type.MouseButtonRelease, position, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)


def test_result_install_is_atomic_for_wrong_source_and_stale_volume_identity(viewer):
    existing = install(viewer)
    image = viewer.volume_view._segment_image
    rejected_source = copy.deepcopy(existing.source)
    rejected_source["origin"][0] += 7
    wrong = SegmentationResult(existing.data, existing.labels, rejected_source, existing.provenance)
    with pytest.raises(ValueError, match="geometry"):
        viewer.segmentation_ready((id(viewer.volume), wrong))
    assert viewer.segmentation is existing
    assert viewer.volume_view._segment_image is image
    viewer.segmentation_ready((id(viewer.volume) + 1, wrong))
    assert viewer.segmentation is existing
    assert viewer.volume_view._segment_image is image


@pytest.mark.parametrize("reflected", [False, True])
def test_all_mpr_overlays_follow_independent_lps_nearest_voxel_registration(qt_app, viewer, reflected):
    viewer.set_volume(small_volume(reflected=reflected))
    for spin, angle in zip(viewer.rotation_spins, (17, -11, 23)):
        spin.setValue(angle)
    settle(qt_app, viewer)
    result = install(viewer)
    for view in viewer.mpr_views.values():
        rgba = view._segmentation_rgba
        assert rgba.shape[:2] == view.data.shape
        rows, columns = view.data.shape
        sample_y = np.arange(7, rows, 29)
        sample_x = np.arange(9, columns, 31)
        center, right, up, size = view.frame
        x = (sample_x / (columns - 1) - .5) * size[0]
        y = (.5 - sample_y / (rows - 1)) * size[1]
        world = center + x[None, :, None] * right + y[:, None, None] * up
        index = viewer.volume.world_to_index(world)
        valid = np.all((index >= -1e-7) & (index <= np.asarray(viewer.volume.dimensions) - 1 + 1e-7), axis=-1)
        nearest = np.clip(np.floor(index + .5).astype(int), 0, np.asarray(viewer.volume.dimensions) - 1)
        ids = result.data[nearest[..., 2], nearest[..., 1], nearest[..., 0]]
        ids[~valid] = 0
        visible = (ids == 3) | (ids == 5)
        actual = rgba[np.ix_(sample_y, sample_x)]
        assert np.array_equal(actual[..., 3] > 0, visible)
        assert np.array_equal(actual[ids == 3, :3], np.tile([96, 192, 160], (np.count_nonzero(ids == 3), 1)))
        assert np.array_equal(actual[ids == 5, :3], np.tile([224, 128, 144], (np.count_nonzero(ids == 5), 1)))


def test_object_review_and_visibility_consistently_hide_2d_and_3d_without_erasing(viewer):
    result = install(viewer)
    before = result.data.copy()
    item = select_label(viewer, 5)
    viewer.segmentation_panel.reject_button.click()
    assert result.labels[5]["review"] == "rejected"
    for view in viewer.mpr_views.values():
        assert not np.any(np.all(view._segmentation_rgba[..., :3] == [224, 128, 144], axis=-1))
    assert viewer.volume_view.segment_property.GetScalarOpacity().GetValue(5) == 0
    assert np.array_equal(result.data, before)
    viewer.segmentation_panel.accept_button.click()
    assert viewer.volume_view.segment_property.GetScalarOpacity().GetValue(5) > 0
    item = select_label(viewer, 5)
    item.setCheckState(Qt.CheckState.Unchecked)
    assert viewer.volume_view.segment_property.GetScalarOpacity().GetValue(5) == 0
    assert viewer.volume_view.segment_property.GetScalarOpacity().GetValue(3) > 0
    viewer.segmentation_panel.show_2d.setChecked(False)
    assert all(view.segmentation_image is None for view in viewer.mpr_views.values())
    viewer.segmentation_panel.show_3d.setChecked(False)
    assert not viewer.volume_view.segment_actor.GetVisibility()
    assert np.array_equal(result.data, before)


def test_native_qt_brush_drag_is_one_undo_and_returns_accepted_object_to_pending(qt_app, viewer):
    result = install(viewer, candidates(viewer, empty=True))
    select_label(viewer, 3)
    viewer.segmentation_panel.accept_button.click()
    viewer.segmentation_panel.radius.setValue(.7)
    viewer.segmentation_panel.paint_button.click()
    view = viewer.mpr_views["Axial"]
    center = view.world_to_screen(viewer.crosshair)
    before_source = viewer.volume.data.copy()
    press(view, center - QPointF(50, 0))
    for offset in (-35, -10, 20, 45):
        move(view, center + QPointF(offset, 0))
    assert result.labels[3]["review"] == "pending"
    assert result.data.any()
    release(view, center + QPointF(45, 0))
    settle(qt_app, viewer)
    assert len(result._undo) == 1
    assert viewer._mask_previous is None
    assert not viewer.mask_timer.isActive()
    assert all(view._segmentation_rgba.shape[:2] == view.data.shape for view in viewer.mpr_views.values())
    viewer.segmentation_panel.undo_button.click()
    assert not result.data.any()
    assert result.labels[3]["review"] == "accepted"
    np.testing.assert_array_equal(viewer.volume.data, before_source)


def test_noop_brush_release_restores_full_overlay_detail(qt_app, viewer):
    data = np.full(viewer.volume.data.shape, 3, dtype=np.uint16)
    empty = candidates(viewer, empty=True)
    result = create_result(viewer.volume, viewer.volume_fingerprint(), data, empty.labels, empty.provenance)
    install(viewer, result)
    select_label(viewer, 3)
    viewer.segmentation_panel.paint_button.click()
    view = viewer.mpr_views["Axial"]
    point = view.world_to_screen(viewer.crosshair)
    press(view, point)
    wait_for(qt_app, lambda: not viewer.mask_timer.isActive())
    assert max(view._segmentation_rgba.shape[:2]) <= 224
    assert not result.can_undo
    release(view, point)
    assert view._segmentation_rgba.shape[:2] == view.data.shape
    assert not result.can_undo


def test_preview_overlay_pixel_centers_remain_registered_to_displayed_anatomy(qt_app, viewer):
    result = install(viewer)
    select_label(viewer, 3)
    viewer.segmentation_panel.radius.setValue(.3)
    viewer.segmentation_panel.paint_button.click()
    axial = viewer.mpr_views["Axial"]
    point = axial.world_to_screen(viewer.crosshair)
    press(axial, point)
    wait_for(qt_app, lambda: not viewer.mask_timer.isActive())
    for view in viewer.mpr_views.values():
        assert view.segmentation_preview
        rgba = view._segmentation_rgba
        rows, columns = rgba.shape[:2]
        assert max(rows, columns) <= 224
        rect = view._rect()
        for row in range(9, rows, 31):
            for column in range(11, columns, 37):
                screen = QPointF(rect.left() + (column + .5) / columns * rect.width(),
                                 rect.top() + (row + .5) / rows * rect.height())
                world = view.screen_to_world(screen)
                index = viewer.volume.world_to_index(world)
                label = 0
                if np.all(index >= -1e-7) and np.all(index <= np.asarray(viewer.volume.dimensions) - 1 + 1e-7):
                    x, y, z = np.floor(index + .5).astype(int)
                    label = int(result.data[z, y, x])
                assert bool(rgba[row, column, 3]) == (label in (3, 5))
    release(axial, point)
    assert all(not view.segmentation_preview for view in viewer.mpr_views.values())
    assert all(view._segmentation_rgba.shape[:2] == view.data.shape for view in viewer.mpr_views.values())


def test_go_to_object_center_navigates_linked_sections_and_3d_cursor(qt_app, viewer):
    empty = candidates(viewer, empty=True)
    data = empty.data.copy()
    data[1:4, 2:6, 1:5] = 3
    result = create_result(viewer.volume, viewer.volume_fingerprint(), data, empty.labels, empty.provenance)
    install(viewer, result)
    select_label(viewer, 3)
    viewer.segmentation_panel.focus_button.click()
    settle(qt_app, viewer)
    expected = viewer.volume.index_to_world([2.5, 3.5, 2.])
    np.testing.assert_allclose(viewer.crosshair, expected)
    np.testing.assert_allclose(viewer.volume_view._cursor_source.GetCenter(), expected)
    for view in viewer.mpr_views.values():
        np.testing.assert_allclose(view.crosshair, expected)


def test_brush_protects_other_objects_and_requires_explicit_overwrite(viewer):
    data = np.full(viewer.volume.data.shape, 5, dtype=np.uint16)
    empty = candidates(viewer, empty=True)
    result = create_result(viewer.volume, viewer.volume_fingerprint(), data, empty.labels, empty.provenance)
    install(viewer, result)
    select_label(viewer, 3)
    viewer.segmentation_panel.paint_button.click()
    view = viewer.mpr_views["Axial"]
    point = view.world_to_screen(viewer.crosshair)
    press(view, point)
    release(view, point)
    assert np.all(result.data == 5)
    viewer.segmentation_panel.overwrite.setChecked(True)
    press(view, point)
    release(view, point)
    assert np.any(result.data == 3)
    viewer.undo_mask()
    assert np.all(result.data == 5)


@pytest.mark.parametrize("switch", ["tool", "workflow"])
def test_changing_tool_or_workflow_ends_current_stroke_and_blocks_further_paint(viewer, switch):
    result = install(viewer, candidates(viewer, empty=True))
    select_label(viewer, 3)
    viewer.segmentation_panel.paint_button.click()
    view = viewer.mpr_views["Axial"]
    point = view.world_to_screen(viewer.crosshair)
    press(view, point)
    if switch == "tool":
        viewer.tool_combo.setCurrentText("Pan")
    else:
        viewer.tabs.setCurrentIndex(1)
    ended = result.data.copy()
    move(view, point + QPointF(120, 0))
    release(view, point + QPointF(120, 0))
    assert viewer._mask_previous is None
    np.testing.assert_array_equal(result.data, ended)
    assert len(result._undo) == 1
    viewer.undo_mask()
    assert not result.data.any()


@pytest.mark.parametrize("reflected", [False, True])
def test_gpu_label_image_uses_source_physical_grid_and_nearest_interpolation(qt_app, viewer, reflected):
    viewer.set_volume(small_volume(reflected=reflected))
    settle(qt_app, viewer)
    result = install(viewer)
    renderer = viewer.volume_view
    image = renderer._segment_image
    assert image.GetDimensions() == viewer.volume.dimensions
    for index in ((0., 0., 0.), (3.5, 7.1, 4.6), (20., 18., 16.)):
        actual = [0., 0., 0.]
        image.TransformContinuousIndexToPhysicalPoint(index, actual)
        np.testing.assert_allclose(actual, viewer.volume.index_to_world(index), atol=1e-10)
    np.testing.assert_array_equal(vtk_to_numpy(image.GetPointData().GetScalars()), result.data.ravel())
    assert renderer.segment_property.GetInterpolationType() == 0  # VTK_NEAREST_INTERPOLATION
    assert not np.shares_memory(result.data, viewer.volume.data)


def test_new_volume_releases_mask_image_and_mapper_inputs_and_clears_all_overlays(qt_app, viewer):
    install(viewer)
    assert viewer.volume_view._segment_data is not None
    viewer.set_volume(small_volume(shift=5))
    settle(qt_app, viewer)
    assert viewer.segmentation is None
    assert viewer.segmentation_panel.objects.count() == 0
    assert viewer.volume_view._segment_image is None
    assert viewer.volume_view._segment_data is None
    assert viewer.volume_view.segment_mapper.GetInput() is None
    assert not viewer.volume_view.segment_actor.GetVisibility()
    assert all(view.segmentation_image is None for view in viewer.all_slices)


def test_segment_button_runs_async_and_installs_only_complete_current_candidates(qt_app, viewer, monkeypatch):
    import asterview.inference as inference
    calls = []
    expected = candidates(viewer)
    source = viewer.volume.data.copy()
    def run(volume, fingerprint, progress, cancel):
        calls.append((volume, fingerprint))
        progress("Test inference completed")
        return expected
    monkeypatch.setattr(inference, "run_segmentation", run)
    viewer.segmentation_panel.run_button.click()
    assert viewer.job is not None
    settle(qt_app, viewer)
    assert viewer.segmentation is expected
    assert calls[0][0] is viewer.volume
    assert calls[0][1] == viewer.volume_fingerprint()
    assert not viewer.segmentation_panel._busy
    assert all(label["review"] == "pending" for label in expected.labels.values())
    np.testing.assert_array_equal(viewer.volume.data, source)


def test_cancelled_async_inference_never_installs_partial_candidate(qt_app, viewer, monkeypatch):
    import asterview.inference as inference
    existing = install(viewer)
    started = threading.Event()
    def run(volume, fingerprint, progress, cancel):
        started.set()
        while not cancel():
            time.sleep(.001)
        raise inference.InferenceCancelled("Test cancellation")
    monkeypatch.setattr(inference, "run_segmentation", run)
    viewer.segmentation_panel.run_button.click()
    wait_for(qt_app, started.is_set)
    viewer.segmentation_panel.cancel_button.click()
    settle(qt_app, viewer)
    assert viewer.segmentation is existing
    assert not viewer.segmentation_panel._busy
    assert "cancel" in viewer.statusBar().currentMessage().lower()


def test_cancel_after_worker_emit_before_gui_delivery_does_not_install(qt_app, viewer, monkeypatch):
    import asterview.inference as inference
    existing = install(viewer)
    replacement = candidates(viewer, empty=True)
    monkeypatch.setattr(inference, "run_segmentation", lambda *args, **kwargs: replacement)
    viewer.segmentation_panel.run_button.click()
    job = viewer.job
    deadline = time.monotonic() + 5
    # Hold the GUI event queue deliberately: worker can emit, but the result
    # has not reached segmentation_ready when the user cancellation occurs.
    while not job.isFinished() and time.monotonic() < deadline:
        time.sleep(.001)
    assert job.isFinished()
    viewer.cancel_job()
    settle(qt_app, viewer)
    assert viewer.segmentation is existing


def test_async_result_from_previous_volume_is_discarded(qt_app, viewer, monkeypatch):
    import asterview.inference as inference
    completed = threading.Event()
    started = threading.Event()
    expected = candidates(viewer)
    def run(volume, fingerprint, progress, cancel):
        started.set()
        if not completed.wait(5):
            raise AssertionError("Test did not release the inference fixture")
        return expected
    monkeypatch.setattr(inference, "run_segmentation", run)
    viewer.segmentation_panel.run_button.click()
    try:
        wait_for(qt_app, started.is_set)
        viewer.set_volume(small_volume(shift=5))
    finally:
        completed.set()
    settle(qt_app, viewer)
    assert viewer.segmentation is None
    assert viewer.volume_view._segment_image is None


def test_brush_review_and_export_never_modify_source_dicom_files(qt_app, viewer, tmp_path):
    series_uid, study_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
    paths = []
    for z in range(5):
        path = tmp_path / f"source-{z}.dcm"
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        dataset.SOPClassUID, dataset.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        dataset.SeriesInstanceUID, dataset.StudyInstanceUID, dataset.FrameOfReferenceUID = series_uid, study_uid, frame_uid
        dataset.Modality = "CT"
        dataset.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        dataset.ImagePositionPatient = [0, 0, z * 1.2]
        dataset.PixelSpacing, dataset.SliceThickness = [.7, .5], 1.2
        dataset.SamplesPerPixel, dataset.PhotometricInterpretation = 1, "MONOCHROME2"
        dataset.Rows, dataset.Columns = 9, 11
        dataset.BitsAllocated, dataset.BitsStored, dataset.HighBit, dataset.PixelRepresentation = 16, 16, 15, 1
        dataset.PixelData = (np.arange(99, dtype=np.int16).reshape(9, 11) + z).tobytes()
        dataset.save_as(path, enforce_file_format=True)
        paths.append(path)
    hashes = [sha256(path.read_bytes()).hexdigest() for path in paths]
    viewer.set_volume(load_series(SeriesInfo(series_uid, "Source preservation phantom", paths, "CT", len(paths))))
    settle(qt_app, viewer)
    source = viewer.volume.data.copy()
    result = install(viewer, candidates(viewer, empty=True))
    select_label(viewer, 3)
    viewer.segmentation_panel.paint_button.click()
    view = viewer.mpr_views["Axial"]
    point = view.world_to_screen(viewer.crosshair)
    press(view, point)
    release(view, point)
    assert result.data.any()
    viewer.segmentation_panel.accept_button.click()
    export_segmentation(tmp_path / "review.aster.npz", result, viewer.volume, viewer.volume_fingerprint())
    assert [sha256(path.read_bytes()).hexdigest() for path in paths] == hashes
    np.testing.assert_array_equal(viewer.volume.data, source)
