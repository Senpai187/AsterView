"""Native VTK implant manipulation, persistent rulers and free camera rotation."""
import copy
import threading

import numpy as np
import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication
from vtkmodules.vtkCommonCore import vtkCommand

from asterview.model import Volume
from asterview.render3d import VolumeView


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def view(app):
    widget = VolumeView()
    widget.set_volume(Volume(np.arange(21**3, dtype=np.float32).reshape((21, 21, 21)),
                             (2, 2, 2), np.array([-20., -20, -20]), np.eye(3), metadata={"synthetic": True}))
    widget._window.SetSize(800, 600)
    widget.set_view_orientation("Anterior")
    yield widget
    widget.cleanup()
    app.processEvents()


def implant(identity="implant-1", **changes):
    return {"id": identity, "name": "Test implant", "center": [0., 0, 0],
            "axis": [0., 0, 1.], "diameter": 4., "length": 10., **changes}


def display(widget, point):
    widget.renderer.SetWorldPoint(*point, 1.)
    widget.renderer.WorldToDisplay()
    return np.asarray(widget.renderer.GetDisplayPoint())


def event(widget, event_id, point):
    widget._interactor.SetEventPosition(*(int(value) for value in point))
    widget._interactor.InvokeEvent(event_id)


def camera_state(view):
    camera = view.renderer.GetActiveCamera()
    return np.asarray(camera.GetPosition()), np.asarray(camera.GetFocalPoint()), np.asarray(camera.GetViewUp())


def test_implant_geometry_cache_reuses_pipeline_and_does_not_touch_volume_texture(view):
    model = implant()
    view.set_implants([model], model["id"], edit_enabled=True)
    record = view._implants[model["id"]]
    actor, source, matrix = record["actor"], record["source"], record["matrix"]
    image, shared = view._image, view._shared_array
    before = camera_state(view)
    model["center"] = [3., -2, 1]
    model["axis"] = [0., 1, 0]
    view.set_implants([model])
    assert view._implants[model["id"]]["actor"] is actor
    assert view._implants[model["id"]]["source"] is source
    assert view._implants[model["id"]]["matrix"] is matrix
    assert view._image is image and view._shared_array is shared
    endpoint = np.asarray(actor.GetUserMatrix().MultiplyPoint((0., 5., 0., 1.)))[:3]
    np.testing.assert_allclose(endpoint, [3., 3., 1.])
    for first, second in zip(before, camera_state(view)):
        np.testing.assert_allclose(first, second)
    with pytest.raises(ValueError, match="unique"):
        view.set_implants([model, model])
    assert len(view._implants) == 1 and view._implants[model["id"]]["actor"] is actor


@pytest.mark.parametrize("perspective", [False, True])
def test_native_center_drag_uses_fixed_camera_plane_and_offset_through_model_refresh(app, view, perspective):
    model = implant()
    view.set_implants([model], model["id"], edit_enabled=True)
    camera = view.renderer.GetActiveCamera()
    camera.SetParallelProjection(not perspective)
    center = np.asarray(model["center"])
    point = np.round(display(view, center)[:2]).astype(int)
    assert view._pick_implant_handle(*point) == (model["id"], "center")
    before = camera_state(view)
    events, threads, downstream = [], [], []
    view.implant_dragged.connect(lambda payload: (events.append(payload), threads.append(QThread.currentThread())))
    def synchronize(payload):
        model["center"], model["axis"] = payload["center_lps"].tolist(), payload["axis_lps"].tolist()
        view.set_implants([model])
    view.implant_dragged.connect(synchronize)
    for identifier in (vtkCommand.LeftButtonPressEvent, vtkCommand.MouseMoveEvent, vtkCommand.LeftButtonReleaseEvent):
        view._interactor.AddObserver(identifier, lambda *_: downstream.append(identifier), .5)
    event(view, vtkCommand.LeftButtonPressEvent, point)
    assert view._implant_drag is not None and view._camera_drag is None
    normal = np.asarray(camera.GetDirectionOfProjection())
    right = np.cross(normal, camera.GetViewUp())
    right /= np.linalg.norm(right)
    up = np.cross(right, normal)
    if perspective:
        depth = float((center - np.asarray(camera.GetPosition())) @ normal)
        pitch = 2 * depth * np.tan(np.deg2rad(camera.GetViewAngle() / 2)) / 600
    else:
        pitch = 2 * camera.GetParallelScale() / 600
    for offset in ((7, -3), (23, 14)):
        event(view, vtkCommand.MouseMoveEvent, point + offset)
    # Release has a new location: it must carry the actual final geometry.
    event(view, vtkCommand.LeftButtonReleaseEvent, point + (31, 18))
    assert [item["phase"] for item in events] == ["start", "move", "move", "end"]
    expected = center + pitch * (right * 31 + up * 18)
    np.testing.assert_allclose(events[-1]["center_lps"], expected, atol=1e-9)
    np.testing.assert_allclose(view._implants[model["id"]]["center"], expected, atol=1e-9)
    np.testing.assert_allclose(events[-1]["axis_lps"], [0, 0, 1])
    assert threads == [app.thread()] * 4 and not downstream
    assert view._implant_drag is None
    for first, second in zip(before, camera_state(view)):
        np.testing.assert_allclose(first, second)


def test_native_axis_handle_changes_axis_preserving_center_dimensions_and_cache(view):
    model = implant()
    view.set_implants([model], model["id"], edit_enabled=True)
    record = view._implants[model["id"]]
    point = np.round(display(view, [0., 0, 5])[:2]).astype(int)
    assert view._pick_implant_handle(*point) == (model["id"], "axis")
    signals = []
    view.implant_dragged.connect(signals.append)
    event(view, vtkCommand.LeftButtonPressEvent, point)
    event(view, vtkCommand.MouseMoveEvent, point + (35, 8))
    event(view, vtkCommand.LeftButtonReleaseEvent, point + (35, 8))
    np.testing.assert_allclose(signals[-1]["center_lps"], [0, 0, 0])
    assert np.linalg.norm(signals[-1]["axis_lps"]) == pytest.approx(1.)
    assert not np.allclose(signals[-1]["axis_lps"], [0, 0, 1])
    assert record["source"].GetHeight() == 10 and record["source"].GetRadius() == 2
    endpoint = np.asarray(record["matrix"].MultiplyPoint((0., 5., 0., 1.)))[:3]
    np.testing.assert_allclose(endpoint, 5 * signals[-1]["axis_lps"])


def test_edit_handles_are_selected_foreground_only_and_mode_change_ends_drag(view):
    models = [implant(), implant("implant-2", center=[8., 0, 0])]
    view.set_implants(models, models[1]["id"], edit_enabled=True)
    assert not view._implants[models[0]["id"]]["handles"]["center"][1].GetVisibility()
    handle = view._implants[models[1]["id"]]["handles"]["center"][1]
    assert view._node_renderer.HasViewProp(handle)
    assert handle.GetVisibility()
    signals = []
    view.implant_dragged.connect(signals.append)
    point = np.round(display(view, [8., 0, 0])[:2]).astype(int)
    event(view, vtkCommand.LeftButtonPressEvent, point)
    view.set_implant_editing(False)
    assert [item["phase"] for item in signals] == ["start", "end"]
    assert view._implant_drag is None and not handle.GetVisibility()
    assert view._pick_implant_handle(*point) is None


def test_clearance_ruler_reuses_foreground_pipeline_and_updates_label_and_endpoints(view):
    image, shared = view._image, view._shared_array
    line, actor, label = view._clearance_line, view._clearance_actor, view._clearance_label
    record = {"implant_point_lps": np.array([2., 0, 0]), "canal_point_lps": np.array([4., 0, 0]),
              "distance_mm": 2., "status": "unreviewed", "geometry_status": "beyond_margin"}
    view.set_clearance_measurement(record)
    assert actor.GetVisibility() and label.GetVisibility()
    assert view._node_renderer.HasViewProp(actor)
    np.testing.assert_allclose(line.GetPoint1(), record["implant_point_lps"])
    np.testing.assert_allclose(line.GetPoint2(), record["canal_point_lps"])
    assert "2.000 mm" in label.GetInput() and "unreviewed" in label.GetInput()
    mtime = line.GetMTime()
    view.set_clearance_measurement(record)
    assert line.GetMTime() == mtime
    record.update(distance_mm=-.5, status="overlap", geometry_status="overlap", canal_point_lps=np.array([1.5, 0, 0]))
    view.set_clearance_measurement(record)
    assert view._clearance_actor is actor and view._clearance_label is label
    assert "-0.500 mm" in label.GetInput()
    assert actor.GetProperty().GetColor()[0] > actor.GetProperty().GetColor()[1]
    assert view._image is image and view._shared_array is shared
    view.set_clearance_measurement(None)
    assert not actor.GetVisibility() and not label.GetVisibility()


def test_ruler_validation_and_gui_thread_gate_leave_current_scene_intact(view):
    view.set_implants([implant()], "implant-1", edit_enabled=True)
    record = {"implant_point_lps": [0., 0, 0], "canal_point_lps": [1., 0, 0], "distance_mm": 1., "status": "within_margin"}
    view.set_clearance_measurement(record)
    with pytest.raises(ValueError):
        view.set_clearance_measurement({**record, "distance_mm": np.nan})
    assert view._clearance_actor.GetVisibility()
    errors = []
    def wrong_thread():
        for action in (lambda: view.set_implants([implant()]), lambda: view.set_clearance_measurement(record)):
            try:
                action()
            except Exception as error:
                errors.append(error)
    thread = threading.Thread(target=wrong_thread)
    thread.start()
    thread.join()
    assert len(errors) == 2 and all(isinstance(error, RuntimeError) for error in errors)


def test_upright_orbit_has_fixed_anchor_return_without_camera_horizon_snap(view):
    view.set_view_orientation("Isometric")
    before = camera_state(view)
    first = np.array([440, 340])
    event(view, vtkCommand.LeftButtonPressEvent, first)
    event(view, vtkCommand.MouseMoveEvent, [680, 430])
    assert not np.allclose(camera_state(view)[0], before[0])
    event(view, vtkCommand.MouseMoveEvent, [250, 500])
    event(view, vtkCommand.LeftButtonReleaseEvent, first)
    for original, returned in zip(before, camera_state(view)):
        np.testing.assert_allclose(original, returned, atol=1e-10)
    assert view._camera_drag is None


def test_shift_drag_retains_upright_orbit_without_camera_roll(view):
    before = camera_state(view)
    view._interactor.SetShiftKey(1)
    event(view, vtkCommand.LeftButtonPressEvent, [600, 300])
    event(view, vtkCommand.LeftButtonReleaseEvent, [400, 500])
    view._interactor.SetShiftKey(0)
    after = camera_state(view)
    assert not np.allclose(before[0], after[0])
    np.testing.assert_allclose(before[1], after[1])
    np.testing.assert_allclose(after[2], [0, 0, 1])
    assert np.linalg.norm(after[0] - after[1]) == pytest.approx(np.linalg.norm(before[0] - before[1]))


def test_background_rotation_retains_default_camera_pan_zoom_and_reset(view):
    style = view._interactor.GetInteractorStyle()
    view.set_free_rotation(False)
    event(view, vtkCommand.LeftButtonPressEvent, [600, 400])
    assert view._camera_drag is None
    event(view, vtkCommand.LeftButtonReleaseEvent, [600, 400])
    assert view._interactor.GetInteractorStyle() is style
    view.set_free_rotation(True)
    view._interactor.SetControlKey(1)
    event(view, vtkCommand.LeftButtonPressEvent, [600, 400])
    assert view._camera_drag is not None
    event(view, vtkCommand.LeftButtonReleaseEvent, [600, 400])
    view._interactor.SetControlKey(0)
    event(view, vtkCommand.LeftButtonPressEvent, [500, 350])
    view.reset_camera()
    assert view._camera_drag is None
    direction = np.asarray(view.renderer.GetActiveCamera().GetPosition()) - np.asarray(view.renderer.GetActiveCamera().GetFocalPoint())
    direction /= np.linalg.norm(direction)
    expected = np.array([1., -1., .65]); expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(direction, expected)


def test_transparent_and_xray_presets_change_only_transfer_functions(view):
    data = view._volume.data.copy()
    image, shared = view._image, view._shared_array
    view.set_preset("Bone")
    points = np.linspace(*view._robust_range, 16)
    bone = np.array([view.property.GetScalarOpacity().GetValue(x) for x in points])
    view.set_preset("Transparent")
    transparent = np.array([view.property.GetScalarOpacity().GetValue(x) for x in points])
    np.testing.assert_allclose(transparent, bone * .16)
    view.set_preset("X-ray")
    assert view._preset == "X-ray" and not view.property.GetShade()
    for x in points:
        rgb = view.property.GetRGBTransferFunction().GetColor(x)
        assert rgb[0] == pytest.approx(rgb[1]) and rgb[1] == pytest.approx(rgb[2])
    assert view._image is image and view._shared_array is shared
    np.testing.assert_array_equal(view._volume.data, data)


def test_new_source_clears_implant_handle_ruler_and_drag_references(view):
    view.set_implants([implant()], "implant-1", edit_enabled=True)
    actor = view._implant_actors[0]
    handle = view._implants["implant-1"]["handles"]["center"][1]
    view.set_clearance_measurement({"implant_point_lps": [0, 0, 0], "canal_point_lps": [1, 0, 0], "distance_mm": 1.})
    view.set_volume(Volume(np.ones((3, 4, 5), dtype=np.float32), (1, 1, 1), np.zeros(3), np.eye(3)))
    assert not view._implants and not view._implant_handle_lookup
    assert view._implant_drag is None and view._camera_drag is None
    assert not view.renderer.HasViewProp(actor) and not view._node_renderer.HasViewProp(handle)
    assert not view._clearance_actor.GetVisibility()
