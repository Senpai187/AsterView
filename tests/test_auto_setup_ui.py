"""Real workstation callback contracts for automatic editable dental setup."""
import copy
import threading
import time
from types import SimpleNamespace
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from asterview.app import MainWindow
from asterview.auto_dental import DentalArchProposal, TMJProposal
from asterview.model import Volume
from asterview.mpr_geometry import plane_frame
from asterview.reformat import rotation_matrix
from asterview.segmentation import create_result


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, monkeypatch):
    w = MainWindow()
    w.render_timer.stop()
    monkeypatch.setattr(w, "schedule_render", lambda *_: None)
    w.volume = Volume(np.zeros((20, 24, 28)), (1., 1., 1.), [10., -5., 30.], np.eye(3), metadata={"synthetic": True})
    w.crosshair = w.volume.center.copy()
    data = np.zeros(w.volume.data.shape, dtype=np.uint16)
    data[3:8, 4:10, 5:15] = 4
    w.segmentation = create_result(w.volume, w.volume_fingerprint(), data,
        {4:{"name":"Lower teeth", "kind":"tooth"}}, {"name":"Synthetic callback test","version":"1"})
    w.reset_workflow_state()
    w._captured_jobs = []
    def capture(function, callback):
        w._captured_jobs.append((function, callback))
        return True
    monkeypatch.setattr(w, "start_job", capture)
    yield w
    w.job = None
    w.render_timer.stop()
    w.mask_timer.stop()
    w.path_timer.stop()
    w.clearance_timer.stop()
    w.close()
    app.processEvents()


def prepare_arch(window, monkeypatch):
    import asterview.auto_dental as module
    captured = {}
    def proposal(volume, segmentation, source, **kwargs):
        captured.update(kwargs)
        center, right, up = (kwargs[key] for key in ("plane_center_lps","right_lps","up_lps"))
        points = np.array([center - 5*right, center + 3*up, center + 5*right])
        return DentalArchProposal(points, source, {"method":"callback-test"}, ["Pending source review"], source_revision=segmentation.revision)
    monkeypatch.setattr(module,"propose_arch",proposal)
    window.auto_arch()
    function, install = window._captured_jobs[-1]
    result = function(lambda *_:None, lambda:False)
    return result, install, captured


def prepare_tmj(window, monkeypatch, axis=None):
    import asterview.auto_dental as module
    axis = np.asarray([.8,.6,0.] if axis is None else axis)
    def proposal(volume, segmentation, source, **kwargs):
        return TMJProposal({"right":volume.center+[-5,2,1],"left":volume.center+[5,2,1]},
                           {"right":axis.copy(),"left":axis.copy()},source,
                           {"method":"callback-test"},["Pending source review"],source_revision=segmentation.revision)
    monkeypatch.setattr(module,"propose_tmj",proposal)
    window.auto_tmj()
    function, install = window._captured_jobs[-1]
    return function(lambda *_:None,lambda:False), install


def test_arch_callback_uses_current_plane_and_independent_editable_points(window,monkeypatch):
    window._rotation_override=rotation_matrix(11,-16,28)
    normal=window.rotation()[:,2]
    window.crosshair=window.volume.center+normal*3
    result, install, captured=prepare_arch(window,monkeypatch)
    install(result)
    assert len(window.arch_points)==3
    assert np.allclose((np.asarray(window.arch_points)-window.crosshair)@normal,0,atol=1e-9)
    assert np.allclose(window._arch_up,normal)
    assert window.arch_view.points is window.arch_points
    assert not window.arch_toggle.isChecked()
    expected=window.arch_points[0].copy()
    result.points_lps_mm[0]+=1
    assert np.array_equal(window.arch_points[0],expected)
    assert window.auto_setup_provenance["arch"] is result
    assert captured["depth_mode"] == "anatomy"


def test_accepted_auto_arch_relocates_corrected_locator_to_dental_depth(window, monkeypatch):
    window._rotation_override = rotation_matrix(11, -16, 28)
    normal = window.rotation()[:, 2]
    before = window.crosshair.copy()
    result, install, captured = prepare_arch(window, monkeypatch)
    result.points_lps_mm -= normal * 4
    install(result)
    np.testing.assert_allclose(window.crosshair, before - normal * 4, atol=1e-9)
    np.testing.assert_allclose((np.asarray(window.arch_points) - window.crosshair) @ normal, 0, atol=1e-9)
    np.testing.assert_allclose(window.arch_view.crosshair, window.crosshair)


def test_accepted_auto_tmj_relocates_locator_to_average_condylar_depth(window, monkeypatch):
    window._rotation_override = rotation_matrix(11, -16, 28)
    normal = window.rotation()[:, 2]
    before = window.crosshair.copy()
    result, install = prepare_tmj(window, monkeypatch)
    install(result)
    average = np.mean(list(result.centers_lps_mm.values()), axis=0)
    np.testing.assert_allclose(window.crosshair, before + normal * np.dot(average - before, normal))
    np.testing.assert_allclose(window.crosshair @ normal, average @ normal)
    np.testing.assert_allclose(window.arch_view.crosshair, window.crosshair)


@pytest.mark.parametrize("change",["source","segmentation","revision","existing-points","rotation","plane-depth"])
def test_arch_stale_results_cannot_replace_current_setup(window,monkeypatch,change):
    result, install, captured=prepare_arch(window,monkeypatch)
    if change=="source": window.volume=Volume(window.volume.data.copy(),window.volume.spacing,window.volume.origin,window.volume.direction,metadata={"synthetic":True})
    if change=="segmentation": window.segmentation=copy.copy(window.segmentation)
    if change=="revision": window.segmentation.revision+=1
    if change=="existing-points": window.arch_points=[window.volume.center.copy()]
    if change=="rotation": window._rotation_override=rotation_matrix(2,3,4)
    if change=="plane-depth": window.crosshair+=np.cross(captured["right_lps"],captured["up_lps"])*2
    before=[point.copy() for point in window.arch_points]
    install(result)
    assert len(window.arch_points)==len(before)
    assert "arch" not in window.auto_setup_provenance
    assert "changed" in window.statusBar().currentMessage().casefold()


def test_arch_inplane_navigation_does_not_invalidate_the_same_locator_plane(window,monkeypatch):
    result, install, captured=prepare_arch(window,monkeypatch)
    window.crosshair+=captured["right_lps"]*2
    install(result)
    assert len(window.arch_points)==3


def test_tmj_axis_conversion_matches_corrected_view_basis(window,monkeypatch):
    rotation=rotation_matrix(17,-23,41)
    window._rotation_override=rotation
    result,install=prepare_tmj(window,monkeypatch)
    install(result)
    for key, side in (("right","Right"),("left","Left")):
        local=rotation.T@result.axes_lps[key]
        expected=rotation@np.r_[local[:2]/np.linalg.norm(local[:2]),0]
        angle=np.deg2rad(window.tmj_angles[side].value())
        displayed=rotation@np.array([np.cos(angle),np.sin(angle),0])
        assert np.linalg.norm(expected-displayed)<.002
        assert np.allclose(window.tmj_centers[side],result.centers_lps_mm[key])


def test_tmj_degenerate_axis_projection_preserves_previous_angle(window,monkeypatch):
    window._rotation_override=rotation_matrix(0,90,0)
    for spin in window.tmj_angles.values():spin.setValue(23)
    result,install=prepare_tmj(window,monkeypatch,axis=[1,0,0])
    install(result)
    assert all(spin.value()==23 for spin in window.tmj_angles.values())
    assert all(center is not None for center in window.tmj_centers.values())


@pytest.mark.parametrize("change",["source","segmentation","revision","center","angle","rotation"])
def test_tmj_stale_results_preserve_manual_edits(window,monkeypatch,change):
    result,install=prepare_tmj(window,monkeypatch)
    if change=="source":window.volume=Volume(window.volume.data.copy(),window.volume.spacing,window.volume.origin,window.volume.direction,metadata={"synthetic":True})
    if change=="segmentation":window.segmentation=copy.copy(window.segmentation)
    if change=="revision":window.segmentation.revision+=1
    if change=="center":window.tmj_centers["Right"]=window.volume.center.copy()
    if change=="angle":window.tmj_angles["Right"].setValue(23)
    if change=="rotation":window._rotation_override=rotation_matrix(2,3,4)
    before={side:None if center is None else center.copy() for side,center in window.tmj_centers.items()}
    install(result)
    for side,center in before.items():
        if center is None:assert window.tmj_centers[side] is None
        else:assert np.array_equal(center,window.tmj_centers[side])
    assert "tmj" not in window.auto_setup_provenance


def test_workflow_click_initializes_once_per_source_revision_and_preserves_edits(window,app,monkeypatch):
    calls=[]
    monkeypatch.setattr(window,"auto_arch",lambda:calls.append("arch"))
    monkeypatch.setattr(window,"auto_tmj",lambda:calls.append("tmj"))
    monkeypatch.setattr(window,"place_shared_volume",lambda *_:None)
    window.workflow_changed(1);app.processEvents()
    window.workflow_changed(2);app.processEvents()
    assert calls==["arch"]
    window.segmentation.revision+=1
    window.workflow_changed(2);app.processEvents()
    assert calls==["arch","arch"]
    window.arch_points=[window.volume.center.copy()]
    window.segmentation.revision+=1
    window.workflow_changed(1);app.processEvents()
    assert calls==["arch","arch"]
    window.workflow_changed(3);app.processEvents()
    window.workflow_changed(3);app.processEvents()
    assert calls==["arch","arch","tmj"]
    window.tmj_centers["Right"]=window.volume.center.copy()
    window.segmentation.revision+=1
    window.workflow_changed(3);app.processEvents()
    assert calls==["arch","arch","tmj"]


def test_fast_arch_worker_generates_panorama_after_job_is_cleared(window,app,monkeypatch):
    """The worker can finish before its queued result callback installs the arch."""
    prepare_arch(window,monkeypatch)
    monkeypatch.setattr(window,"start_job",MainWindow.start_job.__get__(window,MainWindow))
    calls=[]
    def generate():
        assert window.job is None
        assert window._auto_pano_pending is None
        calls.append(np.asarray(window.arch_points).copy())
    monkeypatch.setattr(window,"generate_panorama",generate)
    window.auto_arch()
    worker=window.job
    assert worker is not None
    # Complete the actual QThread before delivering its queued GUI signals.
    assert worker.wait(5000)
    deadline=time.monotonic()+5
    while not calls and time.monotonic()<deadline:
        app.processEvents()
        time.sleep(.001)
    app.processEvents()
    assert len(calls)==1
    assert calls[0].shape==(3,3)
    assert window.job is None
    assert window._auto_pano_pending is None


@pytest.mark.parametrize("change",["source","revision","busy","closing"])
def test_pending_panorama_cannot_generate_after_source_or_workflow_changes(window,monkeypatch,change):
    calls=[]
    monkeypatch.setattr(window,"generate_panorama",lambda:calls.append(True))
    window._auto_pano_pending=(id(window.volume),window.pano_revision)
    if change=="source":
        window.volume=Volume(window.volume.data.copy(),window.volume.spacing,window.volume.origin,window.volume.direction,metadata={"synthetic":True})
    elif change=="revision":window.pano_revision+=1
    elif change=="busy":window.job=object()
    elif change=="closing":window._closing=True
    window.finish_pending_auto_panorama()
    assert calls==[]
    if change=="busy":
        # A queued continuation may be drained again when the current job ends.
        assert window._auto_pano_pending==(id(window.volume),window.pano_revision)
    else:
        assert window._auto_pano_pending is None
    window.job=None
    window._closing=False


def test_unsupported_anatomy_proposal_leaves_workspace_editable_without_modal(window,monkeypatch):
    import asterview.auto_dental as dental
    def unavailable(*args, **kwargs):
        raise ValueError("No supported anatomy in this field of view")
    monkeypatch.setattr(dental, "propose_arch", unavailable)
    captured = {}
    monkeypatch.setattr(window, "start_job", lambda function, callback: captured.update(function=function, callback=callback) or True)
    window.auto_arch()
    result = captured["function"](lambda *_: None, lambda: False)
    captured["callback"](result)
    assert not window.arch_points
    assert window.pano_result is None
    assert "Automatic arch unavailable" in window.statusBar().currentMessage()


@pytest.fixture
def cold_setup(window,monkeypatch):
    """Raw CT, no saved masks; replace only model/proposal computations."""
    import asterview.inference as inference
    import asterview.auto_dental as dental
    window.segmentation=None
    window.volume.metadata["modality"]="CT"
    state={"inference_calls":0,"arch_calls":0,"tmj_calls":0,"refreshes":0}
    monkeypatch.setattr(inference,"detect_model_status",lambda:{"ready":True})
    def infer(volume,source,progress=None,cancel=None):
        state["inference_calls"]+=1
        progress("Testing local anatomy worker progress")
        data=np.zeros(volume.data.shape,dtype=np.uint16)
        data[3:8,4:10,5:15]=4
        data[8:12,5:17,4:22]=2
        return create_result(volume,source,data,
            {2:{"name":"Mandible","kind":"jaw"},4:{"name":"Lower teeth","kind":"tooth"}},
            {"name":"Synthetic inference contract","version":"1"})
    def arch(volume,segmentation,source,**kwargs):
        state["arch_calls"]+=1
        center,right,up=(kwargs[key] for key in ("plane_center_lps","right_lps","up_lps"))
        return DentalArchProposal(np.array([center-5*right,center+3*up,center+5*right]),
            source,{"method":"cold-start-test"},[],source_revision=segmentation.revision)
    def tmj(volume,segmentation,source,**kwargs):
        state["tmj_calls"]+=1
        return TMJProposal({"right":volume.center+[-5,2,1],"left":volume.center+[5,2,1]},
            {"right":np.array([1.,0,0]),"left":np.array([1.,0,0])},
            source,{"method":"cold-start-test"},[],source_revision=segmentation.revision)
    def refresh(**kwargs):state["refreshes"]+=1
    monkeypatch.setattr(inference,"run_segmentation",infer)
    monkeypatch.setattr(dental,"propose_arch",arch)
    monkeypatch.setattr(dental,"propose_tmj",tmj)
    monkeypatch.setattr(window,"refresh_segmentation",refresh)
    return state


def deliver_captured_job(window,index=-1):
    function,install=window._captured_jobs[index]
    progress=[]
    result=function(lambda *args:progress.append(args),lambda:False)
    install(result)
    return result,progress


@pytest.mark.parametrize("index",[1,2,3])
def test_raw_scan_workflow_click_infers_anatomy_then_completes_without_second_click(window,app,monkeypatch,cold_setup,index):
    source_before=window.volume.data.copy()
    monkeypatch.setattr(window,"place_shared_volume",lambda *_:None)
    generated=[]
    monkeypatch.setattr(window,"generate_panorama",lambda:generated.append(True))
    # The actual workflow callback must also handle a source with no masks.
    window.workflow_changed(index);app.processEvents()
    assert len(window._captured_jobs)==1
    assert window.segmentation is None
    inferred,progress=deliver_captured_job(window)
    assert cold_setup["inference_calls"]==1
    assert any("local anatomy worker" in message for _,message in progress)
    assert window.segmentation is inferred
    assert all(label["review"]=="pending" for label in inferred.labels.values())
    window.finish_pending_auto_panorama()
    assert len(window._captured_jobs)==2
    deliver_captured_job(window)
    window.finish_pending_auto_panorama()
    if index in (1,2):
        assert len(window.arch_points)==3
        assert generated==[True]
        assert cold_setup["arch_calls"]==1
    else:
        assert all(center is not None for center in window.tmj_centers.values())
        assert cold_setup["tmj_calls"]==1
    assert np.array_equal(window.volume.data,source_before)


def test_cold_request_keeps_selected_workflow_and_source_display(window,monkeypatch,cold_setup):
    window.tabs.blockSignals(True)
    window.tabs.setCurrentIndex(2)
    window.tabs.blockSignals(False)
    visible=window.volume_visible.isChecked()
    window.auto_arch()
    deliver_captured_job(window)
    assert window.tabs.currentIndex()==2
    assert window.volume_visible.isChecked()==visible
    assert cold_setup["refreshes"]==1


def test_tab_change_during_anatomy_reuses_one_model_job_and_completes_latest_request(window,cold_setup):
    window.auto_arch()
    window.job=object()
    window.auto_tmj()
    assert len(window._captured_jobs)==1
    window.job=None
    deliver_captured_job(window,0)
    window.finish_pending_auto_panorama()
    deliver_captured_job(window,1)
    assert cold_setup["inference_calls"]==1
    assert cold_setup["arch_calls"]==0
    assert cold_setup["tmj_calls"]==1
    assert all(center is not None for center in window.tmj_centers.values())


@pytest.mark.parametrize("change",["source","replacement-masks","reset"])
def test_anatomy_result_is_discarded_after_source_masks_or_request_reset(window,cold_setup,change):
    window.auto_arch()
    function,install=window._captured_jobs[0]
    result=function(lambda *_:None,lambda:False)
    if change=="source":
        window.volume=Volume(window.volume.data.copy(),window.volume.spacing,window.volume.origin,window.volume.direction,metadata={"synthetic":True,"modality":"CT"})
    elif change=="replacement-masks":window.segmentation=result
    else:window.reset_workflow_state()
    expected=window.segmentation
    install(result)
    assert window.segmentation is expected
    window.finish_pending_auto_panorama()
    assert len(window._captured_jobs)==1
    assert not window.arch_points


@pytest.mark.parametrize("change",["arch-points","plane-depth","rotation"])
def test_valid_anatomy_remains_reviewable_but_does_not_overwrite_setup_edited_while_waiting(window,cold_setup,change):
    window.auto_arch()
    if change=="arch-points":window.arch_points=[window.volume.center.copy()]
    elif change=="plane-depth":window.crosshair+=np.array([0.,0.,2.])
    else:window._rotation_override=rotation_matrix(7,0,0)
    result,_=deliver_captured_job(window)
    assert window.segmentation is result
    window.finish_pending_auto_panorama()
    assert len(window._captured_jobs)==1
    assert "discarded" in window.statusBar().currentMessage()


def test_cancel_clears_anatomy_and_view_continuations(window,cold_setup):
    window.auto_arch()
    window.clear_pending_automatic_setup()
    window.finish_pending_auto_panorama()
    assert window._auto_setup_request is None
    assert window._auto_anatomy_request is None
    assert window.segmentation is None
    assert len(window._captured_jobs)==1


def test_explicit_workflow_selection_can_retry_after_cancel(window,app,cold_setup):
    window.request_automatic_workflow(1);app.processEvents()
    assert len(window._captured_jobs)==1
    window.clear_pending_automatic_setup()
    window.request_automatic_workflow(1);app.processEvents()
    assert len(window._captured_jobs)==2


def test_missing_local_model_explains_setup_instead_of_silent_noop(window,monkeypatch,cold_setup):
    import asterview.inference as inference
    monkeypatch.setattr(inference,"detect_model_status",lambda:{"ready":False,"reason":"Run Setup AI.ps1"})
    window.auto_tmj()
    assert not window._captured_jobs
    assert "Setup AI.ps1" in window.statusBar().currentMessage()
    assert "Setup AI.ps1" in window.segmentation_panel.activity.text()


@pytest.mark.parametrize("metadata",[{"modality":"MR"},{"geometry_calibrated":False},{"modality":""}])
def test_ineligible_raw_sources_explain_why_automatic_anatomy_cannot_run(window,cold_setup,metadata):
    window.volume.metadata.update(metadata)
    window.auto_arch()
    assert not window._captured_jobs
    assert "CT/CBCT geometry" in window.statusBar().currentMessage()


def test_model_failure_is_visible_and_does_not_loop_or_apply_partial_setup(window,monkeypatch,cold_setup):
    import asterview.inference as inference
    def fail(*args,**kwargs):raise RuntimeError("CUDA worker failed in a bounded test")
    monkeypatch.setattr(inference,"run_segmentation",fail)
    window.auto_arch()
    deliver_captured_job(window)
    window.finish_pending_auto_panorama()
    assert window.segmentation is None
    assert len(window._captured_jobs)==1
    assert "CUDA worker failed" in window.segmentation_panel.activity.text()


def test_existing_source_bound_masks_are_reused_without_another_inference(window,monkeypatch,cold_setup):
    window.auto_arch()
    existing,_=deliver_captured_job(window)
    window.clear_pending_automatic_setup()
    window.auto_tmj()
    deliver_captured_job(window)
    assert window.segmentation is existing
    assert cold_setup["inference_calls"]==1
    assert cold_setup["tmj_calls"]==1


def test_real_fast_workers_chain_raw_scan_to_anatomy_to_arch_and_panorama(window,app,monkeypatch,cold_setup):
    monkeypatch.setattr(window,"start_job",MainWindow.start_job.__get__(window,MainWindow))
    generated=[]
    monkeypatch.setattr(window,"generate_panorama",lambda:generated.append(window.job))
    window.auto_arch()
    worker=window.job
    assert worker.wait(5000)
    deadline=time.monotonic()+8
    while not generated and time.monotonic()<deadline:
        app.processEvents()
        time.sleep(.001)
    assert generated==[None]
    assert cold_setup["inference_calls"]==1
    assert cold_setup["arch_calls"]==1
    assert window.segmentation is not None
    assert len(window.arch_points)==3


def test_actual_cancelled_anatomy_worker_clears_queued_tmj(window,app,monkeypatch,cold_setup):
    import asterview.inference as inference
    entered=threading.Event()
    def infer(volume,source,progress=None,cancel=None):
        entered.set()
        while not cancel():time.sleep(.001)
        raise inference.InferenceCancelled("Cancelled model contract test")
    monkeypatch.setattr(inference,"run_segmentation",infer)
    monkeypatch.setattr(window,"start_job",MainWindow.start_job.__get__(window,MainWindow))
    window.auto_arch()
    worker=window.job
    try:
        assert entered.wait(5)
        window.auto_tmj()
        assert window._auto_setup_request.kind=="tmj"
        window.cancel_job()
        assert worker.wait(5000)
        deadline=time.monotonic()+5
        while window.job is not None and time.monotonic()<deadline:
            app.processEvents();time.sleep(.001)
        assert window.job is None
        assert window._auto_setup_request is None
        assert window._auto_anatomy_request is None
        assert window.segmentation is None
        assert not window.arch_points
        assert all(center is None for center in window.tmj_centers.values())
    finally:
        if window.job is worker:
            worker.cancelled=True
            worker.wait(5000)
            app.processEvents()


@pytest.mark.parametrize("with_arch",[False,True])
def test_implant_anatomy_sections_reslice_before_any_implant_is_placed(window,monkeypatch,with_arch):
    from asterview.fast_reslice import FastReslicer
    window.segmentation=None
    window.volume.data[:]=np.indices(window.volume.data.shape).sum(axis=0)
    window.implants=[]
    if with_arch:
        center=window.volume.center
        window.pano_result=SimpleNamespace(path=np.array([center+[-4,2,0],center+[0,-2,0],center+[4,2,0]]),column_pitch_mm=4.)
        window.pano_up=np.array([0.,0.,1.])
        window.pano_index.setRange(0,2)
        window.pano_index.setValue(1)
    window._plane_requests=[]
    window.render_implant_views()
    requests=window._plane_requests
    assert len(requests)==3+window.implant_section_count.value()
    engine=FastReslicer(window.volume)
    for view,frame,cursor,note,shape,key,slab,mode in requests:
        center,right,up,size=frame
        image=engine.sample_plane(center,right,up,size_mm=size,shape=shape,slab_mm=slab,mode=mode)
        view.set_image(image,frame,cursor,note)
        assert view.image is not None
        assert np.isfinite(view.data).any()
        assert "Add or select" not in view.note
    assert all(view.image is not None for view in window.implant_serial_views[:window.implant_section_count.value()])
    assert window.implant_display_tabs.currentIndex()==1
