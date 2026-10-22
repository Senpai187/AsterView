"""Local, cancellable DentalSegmentator inference in an isolated Python runtime.

This module imports no ML framework into the viewer process. The model's teeth
are upper/lower semantic groups; its canal mask is a bony canal proxy, not a
direct segmentation of nerve fibers. Every new result remains pending review.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading

import numpy as np

MODEL_SUBDIR = Path("Dataset112_DentalSegmentator_v100/nnUNetTrainer__nnUNetPlans__3d_fullres")
MODEL_SHA256 = "a3a91ae0f8d7ab403d8662e47d2f0bdf7fc51e7783b46632841274dff8266c32"
ARTIFACT_HASHES = {
    "fold_0/checkpoint_final.pth": MODEL_SHA256,
    "plans.json": "74989aadef6110e9fda8bc4138fe4504eac038da9b2315cc25bfdf9e1f153767",
    "dataset.json": "d2fdf28daef6da66a1ed8af68d27cef7ece76410a958f3827a3097ab6b787aa0",
}
LABELS = {
    1: {"name": "Upper skull", "kind": "skull", "color": "#D1B16F"},
    2: {"name": "Mandible", "kind": "jaw", "color": "#9E8ED9"},
    3: {"name": "Upper teeth (group)", "kind": "tooth", "color": "#70CDB5"},
    4: {"name": "Lower teeth (group)", "kind": "tooth", "color": "#77B3E5"},
    5: {"name": "Mandibular canal (nerve proxy)", "kind": "canal", "color": "#EE8487"},
}


class InferenceCancelled(RuntimeError):
    """The user cancelled; no partial inference result may be applied."""


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _locations():
    """Find project-local runtime locations without changing global settings."""
    project = Path(__file__).resolve().parents[1]
    candidates = [project / "work"] + [p / "work" for p in project.parents]
    for work in candidates:
        python = work / "ai-venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        model = work / "segmentation/models" / MODEL_SUBDIR
        if python.is_file() and model.is_dir():
            return work, python, model
    # Portable copies can carry a dedicated .ai-venv and models directory.
    return project, project / ".ai-venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"), project / "models" / MODEL_SUBDIR


def detect_model_status(*, verify=False):
    """Return durable paths and a readiness reason; never download on UI access.

    ``verify=True`` checks all trusted model artifacts. Normal status checks are
    inexpensive; the worker always verifies before loading the checkpoint.
    """
    work, python, model = _locations()
    missing = [name for name in ARTIFACT_HASHES if not (model / name).is_file()]
    marker = work / "segmentation/runtime-ready.json"
    try:
        runtime = json.loads(marker.read_text(encoding="utf-8"))
        runtime_verified = (isinstance(runtime, dict) and runtime.get("torch") == "2.6.0+cu124"
                            and runtime.get("nnunetv2") == "2.2.1" and runtime.get("cuda") == "12.4"
                            and isinstance(runtime.get("device"), str) and bool(runtime["device"]))
    except (OSError, ValueError, TypeError):
        runtime_verified = False
    ready = python.is_file() and not missing and runtime_verified
    reason = "Ready: DentalSegmentator, local CUDA inference" if ready else "Run Setup AI.ps1 to install and verify the local model and AI runtime."
    if ready and verify:
        for name, expected in ARTIFACT_HASHES.items():
            if _sha256(model / name) != expected:
                ready, reason = False, "Model verification failed; reinstall the official DentalSegmentator model."
                break
    return {"ready": ready, "reason": reason, "python": str(python), "model": str(model), "work": str(work),
            "name": "DentalSegmentator", "version": "v1", "model_sha256": MODEL_SHA256}


def _cancelled(cancel):
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    return bool(cancel.is_set())


def _check_volume(volume):
    if not volume.metadata.get("geometry_calibrated", True) or min(volume.dimensions) < 2:
        raise ValueError("Automatic segmentation requires calibrated three-dimensional CT/CBCT geometry.")
    if volume.data.size > 400_000_000:
        raise ValueError("This volume exceeds the bounded AI working size (400 million source voxels).")
    modality = str(volume.metadata.get("modality", "CT")).upper()
    if modality not in {"CT", "CBCT", ""}:
        raise ValueError("This dental model supports CT/CBCT intensity images only.")


def run_segmentation(volume, fingerprint, progress=None, cancel=None):
    """Return a source-grid SegmentationResult, or raise without applying data.

    Call this in the application's job thread. ``progress`` receives brief text;
    ``cancel`` can be a threading.Event or a no-argument predicate. All patient
    data stays in a temporary local job directory, removed on finish or cancel.
    Original source voxels are never modified, cropped, or intensity-windowed.
    """
    from .segmentation import create_result
    from .session import _fingerprint, _match_volume

    _check_volume(volume)
    # Validate and independently copy the caller's source identity before work.
    source = _fingerprint(fingerprint, "AI source volume")
    expected_geometry = {"shape": list(volume.data.shape), "spacing": list(volume.spacing),
                         "origin": volume.origin.tolist(), "direction": volume.direction.tolist(),
                         "synthetic": source["synthetic"]}
    expected_geometry.update({k: v for k, v in source.items() if k not in expected_geometry})
    _match_volume(source, expected_geometry)
    if _cancelled(cancel):
        raise InferenceCancelled("Automatic segmentation cancelled.")
    status = detect_model_status()
    if not status["ready"]:
        raise RuntimeError(status["reason"])
    report = progress or (lambda _message: None)
    report("Preparing local source volume…")
    job_root = Path(status["work"]) / "segmentation/jobs"
    job_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aster-ai-", dir=job_root) as folder:
        job = Path(folder)
        # np.save is non-pickled numeric interchange; no Qt or VTK is imported
        # in the worker. A local copy avoids lifetime and mutation races.
        if volume.valid_mask is None:
            np.save(job / "input.npy", volume.data, allow_pickle=False)
        else:
            clean = volume.data.copy()
            clean[~volume.valid_mask] = -1000.0 if volume.is_hu else volume.data_min
            np.save(job / "input.npy", clean, allow_pickle=False)
            del clean
        geometry = {"shape": list(volume.data.shape), "spacing": list(volume.spacing),
                    "origin": volume.origin.tolist(), "direction": volume.direction.tolist()}
        (job / "geometry.json").write_text(json.dumps(geometry, allow_nan=False), encoding="utf-8")
        if _cancelled(cancel):
            raise InferenceCancelled("Automatic segmentation cancelled.")
        env = os.environ.copy()
        env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
                    "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
                    "nnUNet_compile": "false", "nnUNet_raw": str(job / "raw"),
                    "nnUNet_preprocessed": str(job / "preprocessed"), "nnUNet_results": str(job / "results")})
        command = [status["python"], "-u", str(Path(__file__).with_name("_inference_worker.py")),
                   "--job", str(job), "--model", status["model"]]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", env=env, creationflags=flags)
        messages = queue.Queue()

        def read_output():
            for line in process.stdout:
                messages.put(line.rstrip())
        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        tail = []
        try:
            while True:
                if _cancelled(cancel):
                    raise InferenceCancelled("Automatic segmentation cancelled.")
                try:
                    line = messages.get(timeout=0.1)
                    tail.append(line)
                    tail = tail[-18:]
                    if line.startswith("ASTER_PROGRESS:"):
                        report(line.partition(":")[2].strip())
                except queue.Empty:
                    pass
                if process.poll() is not None and not reader.is_alive() and messages.empty():
                    break
            if process.returncode:
                raise RuntimeError("Local AI worker failed. " + "\n".join(tail)[-1800:])
            if _cancelled(cancel):
                raise InferenceCancelled("Automatic segmentation cancelled.")
            data = np.load(job / "output.npy", allow_pickle=False)
            if data.shape != volume.data.shape or data.dtype != np.uint16 or np.any(data > 5):
                raise RuntimeError("AI worker returned an invalid source-grid label map.")
            if volume.valid_mask is not None:
                data[~volume.valid_mask] = 0
            report("Preparing editable result for review…")
            provenance = {"name": "DentalSegmentator", "version": "v1 (2024-03-18)",
                          "model_sha256": MODEL_SHA256,
                          "method": "nnU-Net v2 full resolution; fold 0; no test-time mirroring; LPS source-grid restoration",
                          "license": "CC-BY-4.0 (weights); Apache-2.0 (nnU-Net)"}
            return create_result(volume, source, data, LABELS, provenance)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=15)
            reader.join(timeout=3)
            if process.stdout is not None:
                process.stdout.close()
