"""Patient-space dental editing through the workstation's actual Qt controls.

Inference/extraction jobs use deterministic test doubles; these checks cover
editing, registration, review, source isolation, and asynchronous result safety.
They do not assess anatomical or diagnostic accuracy.
"""
import copy
import json
import sys
import threading
import time

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from asterview.app import MainWindow
from asterview.model import Volume
from asterview.mpr_geometry import plane_frame
from asterview.nerve_paths import create_paths, export_nerve_paths
from asterview.reformat import rotation_matrix
from asterview.segmentation import create_result


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_replacing_group_objects_with_instances_never_uses_stale_selection(viewer, monkeypatch):
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *error: errors.append(error))
    data = np.zeros(viewer.volume.data.shape, dtype=np.uint16)
    data[5:10, 6:11, 7:12] = 4
    base = create_result(viewer.volume, viewer.volume_fingerprint(), data,
        {4: {"name": "Lower group", "kind": "tooth"}}, {"name": "Group fixture", "version": "1"})
    viewer.segmentation_ready((id(viewer.volume), base))
    assert viewer.segmentation_panel.selected_label == 4
    data[data == 4] = 1001
    instance = create_result(viewer.volume, viewer.volume_fingerprint(), data,
        {1001: {"name": "Tooth 31", "kind": "tooth", "instance": True, "fdi": 31}},
        {"name": "Instance fixture", "version": "1"})
    viewer.segmentation_ready((id(viewer.volume), instance))
    assert viewer.segmentation_panel.selected_label == 1001
    assert viewer.segmentation_panel.assign_number.isEnabled()
    assert not errors


def test_path_drag_stops_along_its_oblique_plane_at_source_boundary(qt_app, viewer):
    install_paths(qt_app, viewer)
    panel = viewer.nerve_panel
    path = panel.current_path()
    node = panel.selected_node
    point = np.asarray(path["points_lps_mm"][node]).copy()
    view = viewer.mpr_views["Axial"]
    right, up = view.frame[1:3]
    normal = np.cross(right, up)
    request = right * 1000 + up * 700
    viewer.tool_combo.setCurrentText("Edit canal path")
    payload = {"path_id": path["id"], "node_index": node,
               "plane_center": view.frame[0], "plane_normal": normal}
    viewer.path_node_dragged(dict(payload, phase="start", position_lps_mm=point))
    viewer.path_node_dragged(dict(payload, phase="move", position_lps_mm=point + request))
    viewer.path_node_dragged({"phase": "end"})
    actual = np.asarray(viewer.nerve_paths.get_path(path["id"])["points_lps_mm"][node])
    assert abs(np.dot(actual - point, normal)) < 1e-8
    assert np.linalg.norm(np.cross(actual - point, request)) < 1e-7
    index = viewer.volume.world_to_index(actual)
    assert np.all(index >= -1e-8) and np.all(index <= np.asarray(viewer.volume.dimensions) - 1 + 1e-8)


def wait_for(app, predicate, timeout=6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            app.processEvents()
            return
        time.sleep(.001)
    raise AssertionError("Dental controls did not reach the expected state")


def settle(app, window):
    wait_for(app, lambda: window.job is None and window.slice_job is None
             and window.slice_pending is None and not window.render_timer.isActive()
             and not window.mask_timer.isActive() and not window.path_timer.isActive()
             and not window._wheel_settle.isActive())


def source_volume(*, reflected=False, shift=0):
    direction = rotation_matrix(13, -21, 29)
    if reflected:
        direction[:, 0] *= -1
    data = np.arange(25 * 27 * 29, dtype=np.float32).reshape(25, 27, 29)
    return Volume(data, (.7, .9, 1.2), np.array((12.1 + shift, -18.6, 9.3)), direction,
                  metadata={"synthetic": True, "modality": "CT"})


@pytest.fixture
def viewer(qt_app, monkeypatch):
    import asterview.inference as inference
    import asterview.tooth_inference as tooth_inference
    monkeypatch.setattr(inference, "detect_model_status", lambda **_: {"ready": True, "reason": "Test only"})
    monkeypatch.setattr(tooth_inference, "detect_tooth_model_status", lambda **_: {"ready": True, "reason": "Test only"})
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, title, text: warnings.append((title, text)))
    window = MainWindow()
    window.set_volume(source_volume())
    settle(qt_app, window)
    for view in window.mpr_views.values():
        view.resize(640, 560)
    window._test_warnings = warnings
    yield window
    window._wheel_settle.stop()
    window.finish_mask_stroke(reset_pointer=True)
    window.finish_path_gesture(reset_pointer=True)
    if window._interaction_active:
        window.end_interaction()
    settle(qt_app, window)
    window.close()
    qt_app.processEvents()
    assert not warnings, warnings


def mouse(view, kind, point):
    button = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseMove else Qt.MouseButton.LeftButton
    buttons = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseButtonRelease else Qt.MouseButton.LeftButton
    QApplication.sendEvent(view, QMouseEvent(kind, QPointF(point), QPointF(point), button,
                                            buttons, Qt.KeyboardModifier.NoModifier))


def press(view, point):
    mouse(view, QEvent.Type.MouseButtonPress, point)


def move(view, point):
    mouse(view, QEvent.Type.MouseMove, point)


def release(view, point):
    mouse(view, QEvent.Type.MouseButtonRelease, point)


def path_collection(window, *, identity="right-canal", review="accepted", off_plane=0):
    right, up = plane_frame("Axial", window.rotation())
    normal = np.cross(right, up)
    center = window.volume.center
    points = np.array((center - right * 3 + up * .5,
                       center + normal * off_plane,
                       center + right * 3 + up * .7))
    path = {"id": identity, "name": "Right canal fixture", "side": "right",
            "points_lps_mm": points.tolist(), "diameter_mm": 1., "color": "#EE8487",
            "review": review, "provenance": {"name": "Deterministic path fixture", "version": "1"}}
    return create_paths(window.volume, window.volume_fingerprint(), [path])


def install_paths(app, window, collection=None):
    collection = path_collection(window) if collection is None else collection
    window.install_nerve_paths(collection)
    window.nerve_panel.selected_node = 1
    window.focus_path_node()
    window.tool_combo.setCurrentText("Edit canal path")
    settle(app, window)
    return collection, collection.paths[0]["id"]


def install_canal_mask(window):
    mask = np.zeros(window.volume.data.shape, dtype=np.uint16)
    mask[11:14, 12:15, 8:22] = 5
    result = create_result(window.volume, window.volume_fingerprint(), mask,
                           {5: {"name": "Canal fixture", "kind": "canal", "color": "#EE8487"}},
                           {"name": "Deterministic mask fixture", "version": "1"})
    window.segmentation_ready((id(window.volume), result))
    return result


def install_tooth_instances(window):
    mask = np.zeros(window.volume.data.shape, dtype=np.uint16)
    mask[5:8, 9:12, 5:8] = 2
    mask[7:10, 14:17, 18:21] = 3
    mask[12:14, 12:16, 12:16] = 4
    result = create_result(window.volume, window.volume_fingerprint(), mask,
        {2: {"name": "Tooth 11", "kind": "tooth", "instance": True, "fdi": 11,
             "predicted_fdi": 11, "review": "accepted", "color": "#C0A060"},
         3: {"name": "Tooth 12", "kind": "tooth", "instance": True, "fdi": 12,
             "predicted_fdi": 12, "review": "accepted", "color": "#60C0A0"},
         4: {"name": "Grouped dentition", "kind": "tooth", "instance": False, "color": "#A060C0"}},
        {"name": "Deterministic tooth fixture", "version": "1"})
    window.segmentation_ready((id(window.volume), result))
    return result


def select_object(window, identity):
    for row in range(window.segmentation_panel.objects.count()):
        item = window.segmentation_panel.objects.item(row)
        if item.data(Qt.ItemDataRole.UserRole) == identity:
            window.segmentation_panel.objects.setCurrentItem(item)
            return
    raise AssertionError(f"Missing object {identity}")


def test_new_manual_path_is_source_bound_editable_and_pending(qt_app, viewer):
    source = viewer.volume.data.copy()
    viewer.new_nerve_path()
    settle(qt_app, viewer)
    collection = viewer.nerve_paths
    assert collection is not None and len(collection.paths) == 1
    collection.validate_source(viewer.volume, viewer.volume_fingerprint())
    path = collection.paths[0]
    assert path["review"] == "pending" and path["side"] == "unknown"
    assert len(path["points_lps_mm"]) == 2
    assert np.linalg.norm(np.diff(path["points_lps_mm"], axis=0)) == pytest.approx(6)
    assert viewer.mode == "Edit canal path"
    assert viewer.nerve_panel.selected_path_id == path["id"]
    assert viewer.nerve_panel.edit_button.isChecked()
    assert path["id"] in viewer.volume_view._nerve_paths
    assert all(view.path_edit_enabled for view in viewer.mpr_views.values())
    np.testing.assert_array_equal(viewer.volume.data, source)


@pytest.mark.parametrize("reflected,angles", [(False, (0, 0, 0)), (True, (17, -11, 23))])
def test_2d_node_drag_freezes_section_updates_linked_geometry_and_undoes_once(
        qt_app, viewer, reflected, angles):
    viewer.set_volume(source_volume(reflected=reflected))
    for spin, angle in zip(viewer.rotation_spins, angles):
        spin.setValue(angle)
    settle(qt_app, viewer)
    collection, key = install_paths(qt_app, viewer)
    source = viewer.volume.data.copy()
    original = copy.deepcopy(collection.paths)
    assert not collection.can_undo
    view = viewer.mpr_views["Axial"]
    view.resize(640, 560)
    view.zoom, view.pan = 1.8, QPointF(17, -11)
    frame = tuple(np.array(value).copy() for value in view.frame[:3])
    old = np.asarray(collection.get_path(key)["points_lps_mm"])[1]
    start = view.world_to_screen(old)
    end = start + QPointF(18, -9)
    expected = old + view.screen_to_world(end) - view.screen_to_world(start)
    assert view.nerve_node_hit(start)[:2] == (key, 1)
    press(view, start)
    assert view._drag_kind == "nerve" and viewer._path_gesture_active
    for offset in (QPointF(6, -3), QPointF(12, -6), QPointF(18, -9)):
        move(view, start + offset)
        settle(qt_app, viewer)
        np.testing.assert_allclose(view.frame[:3], frame, atol=1e-10)
    release(view, end)
    settle(qt_app, viewer)
    np.testing.assert_allclose(collection.get_path(key)["points_lps_mm"][1], expected, atol=1e-9)
    np.testing.assert_allclose(np.asarray(collection.get_path(key)["points_lps_mm"])[[0, 2]],
                               np.asarray(original[0]["points_lps_mm"])[[0, 2]])
    np.testing.assert_allclose(viewer.crosshair, expected, atol=1e-9)
    np.testing.assert_allclose(viewer.volume_view._nerve_paths[key]["nodes"][1], expected)
    np.testing.assert_allclose(viewer.volume_view._reference_center, expected)
    for linked in viewer.mpr_views.values():
        np.testing.assert_allclose(linked.crosshair, expected)
        assert linked.selected_nerve_node == 1
    assert collection.get_path(key)["review"] == "pending"
    assert "pending" in viewer.nerve_panel.paths.currentItem().text()
    assert not viewer._path_gesture_active
    assert view.zoom == 1.8 and view.pan == QPointF(17, -11)
    np.testing.assert_array_equal(viewer.volume.data, source)
    viewer.undo_path()
    settle(qt_app, viewer)
    assert collection.paths == original
    assert not collection.can_undo
    assert "accepted" in viewer.nerve_panel.paths.currentItem().text()
    np.testing.assert_allclose(viewer.volume_view._nerve_paths[key]["nodes"], original[0]["points_lps_mm"])


def test_projected_off_plane_node_cannot_be_dragged_as_a_solid_node(qt_app, viewer):
    collection, key = install_paths(qt_app, viewer, path_collection(viewer, off_plane=4))
    # Focus a different node to retain the original axial level, then examine
    # the middle node's dashed projection four millimeters away from it.
    viewer.nerve_panel.selected_node = 0
    viewer.focus_path_node()
    settle(qt_app, viewer)
    view = viewer.mpr_views["Axial"]
    point = view.world_to_screen(collection.get_path(key)["points_lps_mm"][1])
    entries = [item for item in view._nerve_nodes() if item[1] == 1]
    assert len(entries) == 1 and not entries[0][-1]
    assert view.nerve_node_hit(point) is None
    original = copy.deepcopy(collection.paths)
    press(view, point)
    assert view._drag_kind != "nerve"
    move(view, point + QPointF(12, 8))
    release(view, point + QPointF(12, 8))
    settle(qt_app, viewer)
    assert collection.paths == original


@pytest.mark.parametrize("change", ["mode", "source"])
def test_mode_or_source_change_cancels_stale_pointer_gesture(qt_app, viewer, change):
    collection, key = install_paths(qt_app, viewer)
    view = viewer.mpr_views["Axial"]
    start = view.world_to_screen(collection.get_path(key)["points_lps_mm"][1])
    press(view, start)
    move(view, start + QPointF(6, -3))
    assert viewer._path_gesture_active
    if change == "mode":
        viewer.tool_combo.setCurrentText("Navigate")
    else:
        viewer.set_volume(source_volume(shift=5))
    assert not viewer._path_gesture_active
    assert view._drag_kind != "nerve"
    frozen = copy.deepcopy(collection.paths)
    move(view, start + QPointF(30, -15))
    release(view, start + QPointF(30, -15))
    settle(qt_app, viewer)
    assert collection.paths == frozen
    if change == "source":
        assert viewer.nerve_paths is None
        assert viewer.nerve_panel.paths.count() == 0
        assert viewer.volume_view._nerve_paths == {}
        assert all(not linked.nerve_paths for linked in viewer.mpr_views.values())


def test_3d_drag_signal_updates_model_mpr_locator_and_review_panel(qt_app, viewer):
    collection, key = install_paths(qt_app, viewer)
    source = viewer.volume.data.copy()
    old = np.asarray(collection.get_path(key)["points_lps_mm"])[1]
    target = viewer.volume.index_to_world(viewer.volume.world_to_index(old) + (.4, -.6, .2))
    signal = viewer.volume_view.path_node_dragged
    for phase, point in (("start", old), ("move", target), ("end", target)):
        signal.emit({"phase": phase, "path_id": key, "node_index": 1, "position_lps_mm": point})
    settle(qt_app, viewer)
    np.testing.assert_allclose(collection.get_path(key)["points_lps_mm"][1], target)
    np.testing.assert_allclose(viewer.crosshair, target)
    np.testing.assert_allclose(viewer.volume_view._reference_center, target)
    for linked in viewer.mpr_views.values():
        np.testing.assert_allclose(linked.crosshair, target)
        assert linked.selected_nerve_path == key and linked.selected_nerve_node == 1
    assert "pending" in viewer.nerve_panel.paths.currentItem().text()
    assert not viewer._path_gesture_active
    np.testing.assert_array_equal(viewer.volume.data, source)


@pytest.mark.parametrize("change", ["mask", "source"])
def test_stale_extraction_job_is_discarded_after_mask_or_source_change(qt_app, viewer, monkeypatch, change):
    import asterview.nerve_paths as paths_module
    mask = install_canal_mask(viewer)
    existing, _ = install_paths(qt_app, viewer)
    replacement = path_collection(viewer, identity="stale-new-canal", review="pending")
    started, complete = threading.Event(), threading.Event()

    def extract(*args, **kwargs):
        started.set()
        if not complete.wait(5):
            raise AssertionError("Test did not release its deterministic extraction worker")
        return replacement

    monkeypatch.setattr(paths_module, "extract_canal_paths", extract)
    viewer.extract_nerve_paths()
    try:
        wait_for(qt_app, started.is_set)
        if change == "mask":
            mask.set_review(5, "accepted")
        else:
            viewer.set_volume(source_volume(shift=5))
    finally:
        complete.set()
    settle(qt_app, viewer)
    assert viewer.nerve_paths is (existing if change == "mask" else None)
    assert "stale-new-canal" not in viewer.volume_view._nerve_paths
    assert not viewer.nerve_panel._busy


def test_opening_path_file_for_wrong_source_is_atomic(qt_app, viewer, monkeypatch, tmp_path):
    collection, key = install_paths(qt_app, viewer)
    original = copy.deepcopy(collection.paths)
    target = tmp_path / "other-source.aster-paths.json"
    export_nerve_paths(target, collection, viewer.volume, viewer.volume_fingerprint())
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["source"]["origin"][0] += 2
    target.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(target), ""))
    viewer.nerve_panel.open_file()
    settle(qt_app, viewer)
    assert viewer.nerve_paths is collection and collection.paths == original
    assert viewer.nerve_panel.selected_path_id == key
    np.testing.assert_allclose(viewer.volume_view._nerve_paths[key]["nodes"], original[0]["points_lps_mm"])
    assert len(viewer._test_warnings) == 1
    title, message = viewer._test_warnings.pop()
    assert title == "Unable to open paths"
    assert "geometry" in message.lower() or "source" in message.lower()


def test_path_extraction_blocks_edits_and_cancellation_keeps_reviewed_paths(qt_app, viewer, monkeypatch):
    import asterview.nerve_paths as paths_module
    install_canal_mask(viewer)
    collection, key = install_paths(qt_app, viewer)
    original = copy.deepcopy(collection.paths)
    started = threading.Event()
    def extract(*args, cancel, **kwargs):
        started.set()
        while not cancel():
            time.sleep(.001)
        raise RuntimeError("Cancelled extraction")
    monkeypatch.setattr(paths_module, "extract_canal_paths", extract)
    viewer.extract_nerve_paths()
    wait_for(qt_app, started.is_set)
    assert viewer.nerve_panel._busy and not viewer.nerve_panel.edit_button.isEnabled()
    viewer.tool_combo.setCurrentText("Edit canal path")
    viewer.path_node_dragged({"phase": "start", "path_id": key, "node_index": 1,
                             "position_lps_mm": viewer.volume.center})
    viewer.delete_path_node()
    viewer.new_nerve_path()
    assert not viewer._path_gesture_active and collection.paths == original
    viewer.nerve_panel.cancel_button.click()
    settle(qt_app, viewer)
    assert viewer.nerve_paths is collection and collection.paths == original
    assert not viewer.nerve_panel._busy and viewer.nerve_panel.edit_button.isEnabled()


def test_fdi_duplicate_requires_swap_and_number_undo_restores_identity_pending_review(qt_app, viewer):
    result = install_tooth_instances(viewer)
    panel = viewer.segmentation_panel
    select_object(viewer, 2)
    original_labels = copy.deepcopy(result.labels)
    original_pixels = result.data.copy()
    original_source = viewer.volume.data.copy()
    assert panel.assign_number.isEnabled()
    panel.fdi_combo.setCurrentIndex(panel.fdi_combo.findData(12))
    panel.assign_number.click()
    assert result.labels == original_labels
    assert len(viewer._test_warnings) == 1
    title, message = viewer._test_warnings.pop()
    assert title == "Unable to assign FDI number" and "already assigned" in message
    panel.swap_number.setChecked(True)
    panel.assign_number.click()
    assert panel.selected_label == 2
    assert result.labels[2]["fdi"] == 12 and result.labels[3]["fdi"] == 11
    assert result.labels[2]["predicted_fdi"] == 11 and result.labels[3]["predicted_fdi"] == 12
    assert result.labels[2]["review"] == result.labels[3]["review"] == "pending"
    np.testing.assert_array_equal(result.data, original_pixels)
    panel.undo_number.click()
    restored_labels = copy.deepcopy(original_labels)
    restored_labels[2]["review"] = restored_labels[3]["review"] = "pending"
    assert result.labels == restored_labels
    assert panel.selected_label == 2
    assert not panel.undo_number.isEnabled()
    np.testing.assert_array_equal(result.data, original_pixels)
    np.testing.assert_array_equal(viewer.volume.data, original_source)


def test_unassigned_fdi_preserves_instance_and_focus_uses_source_grid_centroid(qt_app, viewer):
    result = install_tooth_instances(viewer)
    panel = viewer.segmentation_panel
    select_object(viewer, 2)
    panel.fdi_combo.setCurrentIndex(panel.fdi_combo.findData(None))
    panel.assign_number.click()
    assert "fdi" not in result.labels[2]
    assert result.labels[2]["instance"] and result.labels[2]["predicted_fdi"] == 11
    assert result.labels[2]["review"] == "pending"
    assert "unassigned" in panel.objects.currentItem().text().lower()
    select_object(viewer, 3)
    coordinates = np.argwhere(result.data == 3)[:, ::-1].mean(axis=0)
    expected = viewer.volume.index_to_world(coordinates)
    panel.focus_button.click()
    settle(qt_app, viewer)
    np.testing.assert_allclose(viewer.crosshair, expected, atol=1e-10)
    np.testing.assert_allclose(viewer.volume_view._reference_center, expected)
    select_object(viewer, 4)
    assert not panel.assign_number.isEnabled()
    assert not panel.fdi_combo.isEnabled()
    assert not panel.swap_number.isEnabled()
