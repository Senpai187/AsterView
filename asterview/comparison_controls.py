"""Independent second-volume loading, rigid alignment and review orchestration."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import copy
import threading
from itertools import product
from functools import partial

import numpy as np
from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QFileDialog, QMessageBox

from .comparison_panel import ComparisonPanel, ComparisonDialog, ComparisonWorkspace
from .dicom_io import discover_series, load_series
from .mpr_geometry import plane_frame
from .registration import (RegistrationResult, register_volumes, rigid_matrix, sample_registered_plane,
                           detect_registration_status, save_registration, load_registration)


def _shutdown_executor(executor, *_):
    """QObject destruction fallback without retaining the QObject itself."""
    executor.shutdown(wait=False, cancel_futures=True)


def volume_fingerprint(volume):
    """Source identity without patient names or voxel values."""
    metadata = volume.metadata
    return {"series_uid": str(metadata.get("series_uid", "")), "source_group": str(metadata.get("source_group", "")),
            "sop_instance_digest": str(metadata.get("sop_instance_digest", "")), "frame_of_reference_uid": str(metadata.get("frame_of_reference_uid", "")),
            "shape": list(volume.data.shape), "spacing": list(volume.spacing), "origin": volume.origin.tolist(),
            "direction": volume.direction.tolist(), "synthetic": bool(metadata.get("synthetic", False))}


class ComparisonControls(QObject):
    """Wire `.panel` into the inspector; reset_comparison on reference changes.

    Expensive source loading/registration uses workstation.start_job. Plane
    sampling uses a single owned thread with generation/cancellation checks,
    keeping shared background operations and the main viewer independent.
    """
    def __init__(self, workstation):
        super().__init__(workstation)
        self.workstation = workstation
        self.moving = self.result = self.dialog = None
        self.fixed_source = self.moving_source = None
        self.series = []
        self._epoch = 0
        self._busy = False
        self._closed = False
        self._sampling_started = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AsterComparison")
        self._future = None
        self._pending_planes = None
        self._sampling_key = None
        self._plane_cancel = None
        self._generation = 0
        self._sampling_state = threading.local()
        self._continuation = None
        self._active_job = None
        self._alignment_mode = "Manual"
        self.panel = ComparisonPanel(self)
        self.workspace = ComparisonWorkspace(self)
        self.poll = QTimer(self)
        self.poll.setInterval(8)
        self.poll.timeout.connect(self._plane_finished)
        self.job_poll = QTimer(self)
        self.job_poll.setInterval(40)
        self.job_poll.timeout.connect(self._job_finished)
        self.panel.refresh()
        # Do not capture the QObject in its own destroyed callback. Normal
        # workstation close calls shutdown() before native widgets disappear.
        self.destroyed.connect(partial(_shutdown_executor, self._executor))

    def reset_comparison(self):
        self._epoch += 1
        self._continuation = None
        self.cancel_planes()
        # ThreadPoolExecutor is lazy. Never start a thread merely because an
        # unrelated study was opened or a never-used viewer was closed.
        if self._sampling_started and not self._closed:
            self._executor.submit(self._release_sampling)
        if self.dialog is not None:
            self.dialog.close()
            self.dialog.deleteLater()
            self.dialog = None
        self.moving = self.result = self.fixed_source = self.moving_source = None
        self.workspace.clear()
        self.series = []
        self.panel.series.clear()
        self.panel.info.setText("Keep the reference scan open, then load a second DICOM volume.")
        self.panel.status.setText("No comparison scan loaded.")
        self.panel.refresh(self._busy)

    def shutdown(self):
        """Close this owner and join its CPU worker before native UI teardown."""
        if self._closed:
            return
        self._closed = True
        self.job_poll.stop()
        self._active_job = None
        self.reset_comparison()
        # The sampler never calls Qt or waits on the UI thread, and checks
        # its cancellation event between planes/slab samples. Joining here
        # prevents source/pipeline cleanup from racing native widget teardown.
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._future = None
        self.poll.stop()
        if self.workspace.parentWidget() is None:
            self.workspace.close()
        self._sampling_started = False

    def set_busy(self, busy):
        self._busy = bool(busy)
        self.panel.refresh(self._busy)
        if self.dialog is not None:
            self.dialog.set_fitting_busy(self._busy)

    def _start_job(self, function, callback):
        if self._closed:
            return False
        if self.workstation.start_job(function, callback):
            self.set_busy(True)
            self._active_job = self.workstation.job
            # Poll the main owner rather than attaching after QThread.start:
            # very small discovery jobs can finish before that connection.
            self.job_poll.start()
            return True
        self.workstation.statusBar().showMessage("Wait for the current operation or cancel it first")
        return False

    def _job_finished(self):
        if self.workstation.job is not None:
            return
        self.job_poll.stop()
        self.set_busy(False)
        cancelled = bool(getattr(self._active_job, "cancelled", False))
        self._active_job = None
        continuation, self._continuation = self._continuation, None
        if continuation is not None and not cancelled:
            epoch, callback = continuation
            if epoch == self._epoch and not getattr(self.workstation, "_closing", False):
                callback()

    def begin_workflow(self, folder=None, alignment=None):
        """User entry point: source selection, fitting choice, then linked review."""
        if self._closed:
            return
        if self.workstation.volume is None:
            self.workstation.statusBar().showMessage("Open the reference scan first")
            return
        if self.workstation.job is not None:
            self.workstation.statusBar().showMessage("Wait for the current operation or cancel it first")
            return
        if self.moving is not None and folder is None:
            self.open_comparison()
            return
        folder = folder or QFileDialog.getExistingDirectory(self.workstation, "Choose the second DICOM study folder")
        if not folder:
            return
        if alignment is None:
            choice = QMessageBox(self.workstation)
            choice.setWindowTitle("Fit the comparison scan")
            choice.setText("How would you like to align the two volumes?")
            choice.setInformativeText("Automatic fitting estimates a rigid alignment. Both methods open linked views for review and manual correction.")
            automatic = choice.addButton("Fit automatically", QMessageBox.ButtonRole.AcceptRole)
            manual = choice.addButton("Fit manually", QMessageBox.ButtonRole.ActionRole)
            choice.addButton(QMessageBox.StandardButton.Cancel)
            choice.exec()
            alignment = "Automatic" if choice.clickedButton() is automatic else "Manual" if choice.clickedButton() is manual else None
            if alignment is None:
                return
        if alignment not in {"Automatic", "Manual"}:
            raise ValueError("Choose Automatic or Manual fitting.")
        self._alignment_mode = alignment
        self.panel.fit_mode.setCurrentText(alignment)
        self.load_folder(folder, auto_load=True)

    def _valid(self):
        if self.result is None or self.moving is None or self.workstation.volume is None:
            raise ValueError("Load both source volumes first.")
        self.result.validate_source(self.workstation.volume, self.moving,
                                    self.workstation.volume_fingerprint(), self.moving_source)

    def load_folder(self, folder=None, *, auto_load=False):
        if self.workstation.volume is None:
            self.workstation.statusBar().showMessage("Open the reference scan first")
            return
        if self.workstation.job is not None:
            self.workstation.statusBar().showMessage("Wait for the current operation or cancel it first")
            return
        folder = folder or QFileDialog.getExistingDirectory(self.workstation, "Choose the second DICOM study folder")
        if not folder:
            return
        self._epoch += 1
        self._continuation = None
        epoch, fixed = self._epoch, self.workstation.volume
        def discovered(items):
            if epoch != self._epoch or fixed is not self.workstation.volume:
                return
            self.series = items
            self.panel.series.clear()
            for item in items:
                self.panel.series.addItem(item.label)
            self.panel.info.setText(f"Found {len(items)} comparison series. Choose the intended before/after scan, then load it." if len(items) > 1 else
                                    "One comparison series found. Loading the second volume…" if items and auto_load else
                                    "Choose the comparison series, then load it." if items else "No supported DICOM series was found in this folder.")
            self.panel.refresh(self._busy)
            if auto_load and len(items) == 1:
                self._continuation = (epoch, self.load_selected)
        self._start_job(lambda progress, cancel: discover_series(folder, progress=progress, cancel=cancel), discovered)

    def load_selected(self):
        index = self.panel.series.currentIndex()
        if not 0 <= index < len(self.series):
            return
        epoch, fixed = self._epoch, self.workstation.volume
        item = self.series[index]
        def loaded(volume):
            if epoch == self._epoch and fixed is self.workstation.volume:
                try:
                    self.set_moving(volume)
                except ValueError as exc:
                    self.panel.status.setText("Comparison source unavailable: " + str(exc))
                    self.workstation.statusBar().showMessage(str(exc))
                else:
                    self._continuation = (self._epoch, self._finish_loading)
        self._start_job(lambda progress, cancel: load_series(item, progress=progress, cancel=cancel), loaded)

    def _finish_loading(self):
        self.open_comparison()
        if self.panel.fit_mode.currentText() == "Automatic":
            self.align()
        else:
            self.workstation.statusBar().showMessage("Second volume loaded · adjust translation and rotation, then review the alignment")

    def set_moving(self, volume):
        """Install a loaded source; also useful for controlled UI QA."""
        if self.workstation.volume is None:
            raise ValueError("Open the reference scan first.")
        self._epoch += 1
        self.cancel_planes()
        if self.dialog is not None:
            self.dialog.close()
            self.dialog.deleteLater()
            self.dialog = None
        fixed = self.workstation.volume
        fixed_source, moving_source = self.workstation.volume_fingerprint(), volume_fingerprint(volume)
        initial = RegistrationResult(rigid_matrix(fixed.center - volume.center), fixed_source, moving_source,
                                     method="Manual initial image-center alignment",
                                     warnings=["Image centers were aligned only; anatomical alignment has not been established."])
        initial.validate_source(fixed, volume, fixed_source, moving_source)
        self.moving, self.result = volume, initial
        self.fixed_source, self.moving_source = fixed_source, moving_source
        self.panel.info.setText(f"Reference {fixed.dimensions} · comparison {volume.dimensions}. Original voxels remain unchanged.")
        self.panel.refresh(self._busy)

    def align(self):
        try:
            self._valid()
            status = detect_registration_status()
            if not status["ready"]:
                raise ValueError(status["reason"])
            fixed, moving, epoch = self.workstation.volume, self.moving, self._epoch
            fixed_fp, moving_fp = copy.deepcopy(self.fixed_source), copy.deepcopy(self.moving_source)
            initial = self.result.transform.copy()
            original_result = self.result
            bone_focus = self.panel.bone_focus.isChecked()
            def completed(result):
                if epoch != self._epoch or fixed is not self.workstation.volume or moving is not self.moving or self.result is not original_result:
                    return
                result.validate_source(fixed, moving, self.workstation.volume_fingerprint(), self.moving_source)
                self.result = result
                self.panel.refresh(self._busy)
                self.open_comparison()
            self._start_job(lambda progress, cancel: register_volumes(fixed, moving, fixed_fp, moving_fp,
                                                                     progress=lambda text: progress(0, text), cancel=cancel,
                                                                     initial_transform=initial, bone_focus=bone_focus), completed)
        except (ValueError, RuntimeError) as exc:
            QMessageBox.warning(self.workstation, "Unable to align scans", str(exc))

    def review(self, state):
        try:
            self._valid()
            if state not in {"accepted", "rejected"}:
                raise ValueError("Choose accepted or rejected alignment review.")
            if self._busy:
                return
            if state == "accepted" and any(item.value() != 0 for item in self.panel.manual):
                self.apply_manual()
            self.result.review = state
            self.panel.refresh(self._busy)
            if state == "accepted":
                self.open_comparison()
            elif self.dialog:
                self.dialog.display_cached()
        except ValueError as exc:
            QMessageBox.warning(self.workstation, "Alignment review unavailable", str(exc))

    def apply_manual(self):
        try:
            self._valid()
            values = [item.value() for item in self.panel.manual]
            self.result = self.result.refined(values[:3], values[3:], self.workstation.volume.center)
            for item in self.panel.manual:
                item.setValue(0)
            self.panel.refresh(self._busy)
            self.request_planes()
        except ValueError as exc:
            QMessageBox.warning(self.workstation, "Manual correction unavailable", str(exc))

    def save(self):
        try:
            self._valid()
            path, _ = QFileDialog.getSaveFileName(self.workstation, "Save source-bound alignment", "alignment.aster.json", "AsterView alignment (*.json)")
            if path:
                save_registration(path, self.result, self.workstation.volume, self.moving, self.fixed_source, self.moving_source)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self.workstation, "Alignment could not be saved", str(exc))

    def restore(self):
        try:
            self._valid()
            path, _ = QFileDialog.getOpenFileName(self.workstation, "Open alignment for these exact two scans", "", "AsterView alignment (*.json)")
            if path:
                self.result = load_registration(path, self.workstation.volume, self.moving, self.fixed_source, self.moving_source)
                self.panel.refresh(self._busy)
                self.request_planes()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self.workstation, "Alignment could not be opened", str(exc))

    def open_comparison(self):
        try:
            self._valid()
            if self.result.review != "accepted":
                self.open_alignment()
                return
            if self.dialog is None or self.dialog.fitting:
                self._close_surface()
                self.dialog = ComparisonDialog(self, fitting=False, parent=self.workspace)
                self.workspace.install(self.dialog)
            show_workspace = getattr(self.workstation, "show_superimposition", None)
            if callable(show_workspace):
                show_workspace()
            elif self.workspace.parentWidget() is None:
                self.workspace.resize(1320, 900)
                self.workspace.show()
            self.dialog.show()
            self.dialog.sync_timer.start()
            self.request_planes()
        except ValueError as exc:
            QMessageBox.warning(self.workstation, "Comparison unavailable", str(exc))

    def _close_surface(self):
        self.cancel_planes()
        if self.dialog is not None:
            old, self.dialog = self.dialog, None
            old.close()
            old.deleteLater()

    def open_alignment(self):
        try:
            self._valid()
            if self.dialog is None or not self.dialog.fitting:
                self._close_surface()
                self.workspace.clear()
                self.dialog = ComparisonDialog(self, fitting=True)
            self.dialog.show()
            self.dialog.raise_()
            self.dialog.activateWindow()
            self.dialog.set_fitting_busy(self._busy)
            self.dialog.sync_timer.start()
            self.request_planes()
        except ValueError as exc:
            QMessageBox.warning(self.workstation, "Comparison unavailable", str(exc))

    def cancel_planes(self):
        self._generation += 1
        if self._plane_cancel:
            self._plane_cancel.set()
        if self._future:
            self._future.cancel()
            if self._future.done():
                self._future = None
        self._pending_planes = None
        self._sampling_key = None
        if self._future is None:
            self.poll.stop()
        elif not self._closed:
            self.poll.start()

    def _release_sampling(self):
        self._sampling_state.sources = None
        self._sampling_state.engines = None
        self._sampling_state.cache = OrderedDict()
        self._sampling_state.cache_bytes = 0

    def request_planes(self):
        if self._closed or self.dialog is None or not self.dialog.isVisible():
            return
        try:
            self._valid()
        except ValueError as exc:
            self.dialog.notice.setText(str(exc))
            self.cancel_planes()
            return
        fixed, moving = self.workstation.volume, self.moving
        transform = self.result.transform.copy()
        center = self.dialog.center.copy()
        rotation = self.dialog.rotation.copy()
        corners = fixed.index_to_world(np.array(list(product(*[(0, n - 1) for n in fixed.dimensions]))))
        frames = []
        for plane, first, second in self.dialog.views:
            right, up = plane_frame(plane, rotation)
            size = (max(10., float(np.ptp(corners @ right))), max(10., float(np.ptp(corners @ up))))
            panes = [pane for pane in (first, second) if pane.isVisible()]
            # Match physical display pixels instead of supersampling every
            # small pane. Resolution does not drop during dragging/scrolling.
            requested = [pane.preferred_sample_shape(size) for pane in panes]
            shape = tuple(max(values) for values in zip(*requested)) if requested else (320, 320)
            slab = self.dialog.thickness.value() if first.slab_mm is None else first.slab_mm
            mode = first.slab_mode or self.dialog.projection.currentText()
            normal = np.cross(right, up)
            # In-plane cursor movement does not change the acquired section.
            # A stable FOV permits reuse of the two unaffected orthogonal planes.
            plane_center = fixed.center + normal * np.dot(center - fixed.center, normal)
            frames.append(((plane_center, right, up, size), shape, slab, mode))
        # Navigation is a latest-pending queue. Canceling an active 60 ms
        # sample for every 16 ms input would never show a frame until release.
        # Source/alignment/slab changes invalidate immediately; same-source
        # scrolling and correction allow active work to display, then sample
        # only the newest queued position.
        key = (id(fixed), id(moving), transform.tobytes(), tuple((slab, mode) for _, _, slab, mode in frames))
        if key != self._sampling_key:
            self.cancel_planes()
            self._sampling_key = key
        generation = self._generation
        cancelled = threading.Event()
        def sample():
            from .fast_reslice import FastReslicer
            if getattr(self._sampling_state, "sources", None) != (id(fixed), id(moving)):
                self._sampling_state.sources = (id(fixed), id(moving))
                self._sampling_state.engines = (FastReslicer(fixed), FastReslicer(moving))
                self._sampling_state.cache = OrderedDict()
                self._sampling_state.cache_bytes = 0
            first_engine, second_engine = self._sampling_state.engines
            output = []
            for frame, shape, slab, mode in frames:
                if cancelled.is_set():
                    return generation, [], rotation
                geometry = tuple(np.round(np.r_[frame[0], frame[1], frame[2], frame[3]], 9))
                cache_key = (transform.tobytes(), geometry, shape, slab, mode)
                cache = self._sampling_state.cache
                value = cache.pop(cache_key, None)
                if value is None:
                    fixed_image, fixed_valid = sample_registered_plane(fixed, np.eye(4), frame, shape, slab_mm=slab, mode=mode, cancel=cancelled.is_set, return_valid=True, sampler=first_engine)
                    moving_image, moving_valid = sample_registered_plane(moving, transform, frame, shape, slab_mm=slab, mode=mode, cancel=cancelled.is_set, return_valid=True, sampler=second_engine)
                    value = (fixed_image, moving_image, fixed_valid & moving_valid)
                    self._sampling_state.cache_bytes += sum(item.nbytes for item in value)
                cache[cache_key] = value
                # Cache pixels only, bounded independently of source volume size.
                while self._sampling_state.cache_bytes > 64 * 1024**2 and cache:
                    _, evicted = cache.popitem(last=False)
                    self._sampling_state.cache_bytes -= sum(item.nbytes for item in evicted)
                output.append((frame, *value))
            return generation, output, rotation
        self._pending_planes = (sample, cancelled)
        self._start_pending_planes()

    def _start_pending_planes(self):
        if self._closed or self._future is not None or self._pending_planes is None:
            return
        sample, self._plane_cancel = self._pending_planes
        self._pending_planes = None
        self._sampling_started = True
        self._future = self._executor.submit(sample)
        self.poll.start()

    def _plane_finished(self):
        future = self._future
        if future is None or not future.done():
            return
        self.poll.stop()
        self._future = None
        try:
            generation, planes, rotation = future.result()
            self._valid()
            if generation != self._generation or self.dialog is None or not self.dialog.isVisible():
                return
            self.dialog.cached_planes = planes
            self.dialog.displayed_rotation = rotation
            self.dialog.display_cached()
        except Exception as exc:
            if self.dialog is not None and not (self._plane_cancel and self._plane_cancel.is_set()):
                self.dialog.notice.setText("Comparison plane sampling unavailable: " + str(exc))
        finally:
            if self.dialog is not None and self.dialog.isVisible():
                self._start_pending_planes()
