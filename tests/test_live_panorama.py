"""CPU-only live CPR scheduling contracts; no QApplication or GPU context."""
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from asterview.app import MainWindow


class Timer:
    def __init__(self):
        self.active=False
        self.starts=0
        self.stops=0

    def start(self):
        self.active=True
        self.starts+=1

    def stop(self):
        self.active=False
        self.stops+=1


@pytest.fixture
def workspace():
    state=SimpleNamespace(
        _closing=False, volume=SimpleNamespace(metadata={"geometry_calibrated":True}),
        arch_points=[np.array([0.,0.,0.]),np.array([1.,1.,0.]),np.array([2.,0.,0.])],
        arch_toggle=SimpleNamespace(isChecked=lambda:False),
        tabs=SimpleNamespace(currentIndex=lambda:1),pano_result=None,job=None,
        _interaction_active=False,pano_revision=4,pano_refresh_timer=Timer(),
        pano_thickness=SimpleNamespace(value=lambda:5.),
        pano_projection=SimpleNamespace(currentText=lambda:"mean"),
        pano_offset=SimpleNamespace(value=lambda:0.),
        panorama_sample_shape=lambda:(360,512),
        current_panorama_bounds=lambda:(-45.,45.),
        sync_panorama_range_overlay=lambda:None,
        panorama_needs_detail=lambda:False,
        _pano_offset_dirty=False,
        _report_analysis_active=False,
        rotation=lambda:np.eye(3),messages=[],jobs=[],cleared=0,
    )
    state.statusBar=lambda:SimpleNamespace(showMessage=state.messages.append)
    def start(function,callback):
        if state.job is not None or state._closing:
            return False
        state.jobs.append((function,callback))
        state.job=SimpleNamespace(cancelled=False)
        return True
    def clear():state.cleared+=1
    state.start_job=start
    state.clear_pending_automatic_setup=clear
    state.panorama_ready=lambda result:None
    state.generate_panorama=MethodType(MainWindow.generate_panorama,state)
    return state


def test_idle_refresh_starts_one_worker_and_disarms_debounce(workspace):
    workspace.pano_refresh_timer.start()
    MainWindow.refresh_panorama_after_edit(workspace)
    assert len(workspace.jobs)==1
    assert workspace.job is not None
    assert not workspace.pano_refresh_timer.active


def test_focal_offset_refresh_retains_previous_image_until_new_worker_completes(workspace):
    previous = object()
    workspace.pano_result = previous
    workspace._pano_offset_dirty = True
    MainWindow.refresh_panorama_after_edit(workspace)
    assert len(workspace.jobs) == 1
    assert workspace.pano_result is previous
    assert not workspace.pano_refresh_timer.active


@pytest.mark.parametrize("blocked",["worker","gesture"])
def test_busy_refresh_retries_timer_without_starting_or_cancelling_a_worker(workspace,blocked):
    original=SimpleNamespace(cancelled=False)
    if blocked=="worker":workspace.job=original
    else:workspace._interaction_active=True
    for _ in range(5):MainWindow.refresh_panorama_after_edit(workspace)
    assert not workspace.jobs
    assert workspace.pano_refresh_timer.starts==5
    assert not original.cancelled
    workspace.job=None
    workspace._interaction_active=False
    MainWindow.refresh_panorama_after_edit(workspace)
    assert len(workspace.jobs)==1
    assert not workspace.pano_refresh_timer.active


@pytest.mark.parametrize("blocked",["closing","no-source","short-arch","finished","drawing","explore","tmj"])
def test_unrequested_or_unsupported_refresh_does_not_schedule_retry(workspace,blocked):
    if blocked=="closing":workspace._closing=True
    elif blocked=="no-source":workspace.volume=None
    elif blocked=="short-arch":workspace.arch_points=workspace.arch_points[:2]
    elif blocked=="finished":workspace.pano_result=object()
    elif blocked=="drawing":workspace.arch_toggle.isChecked=lambda:True
    elif blocked=="explore":workspace.tabs.currentIndex=lambda:0
    else:workspace.tabs.currentIndex=lambda:3
    MainWindow.refresh_panorama_after_edit(workspace)
    assert not workspace.jobs
    assert workspace.pano_refresh_timer.starts==0


@pytest.mark.parametrize("has_worker",[False,True])
def test_cancel_disarms_live_retry_and_clears_auto_continuation(workspace,has_worker):
    workspace.pano_refresh_timer.start()
    if has_worker:workspace.job=SimpleNamespace(cancelled=False)
    MainWindow.cancel_job(workspace)
    assert not workspace.pano_refresh_timer.active
    assert workspace.cleared==1
    if has_worker:
        assert workspace.job.cancelled
        assert "Cancel requested" in workspace.messages[-1]


@pytest.mark.parametrize("cancelled",[False,True])
def test_worker_error_disarms_debounce_before_dialog_or_cancel_message(workspace,monkeypatch,cancelled):
    import asterview.app as module
    warnings=[]
    def warn(*args):
        assert not workspace.pano_refresh_timer.active
        warnings.append(args)
    monkeypatch.setattr(module.QMessageBox,"warning",warn)
    workspace.pano_refresh_timer.start()
    workspace.job=SimpleNamespace(cancelled=cancelled)
    MainWindow.job_error(workspace,"Controlled reconstruction failure")
    assert not workspace.pano_refresh_timer.active
    assert len(warnings)==(0 if cancelled else 1)
    if cancelled:assert workspace.messages[-1]=="Operation canceled"


def test_generate_button_during_busy_job_preserves_existing_edit_retry(workspace):
    workspace.job=SimpleNamespace(cancelled=False)
    workspace.pano_refresh_timer.start()
    MainWindow.generate_panorama(workspace)
    assert not workspace.jobs
    assert workspace.pano_refresh_timer.active

