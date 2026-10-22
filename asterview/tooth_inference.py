"""Independent learned ToothSeg instance segmentation and FDI suggestions.

FDI suggestions come from the semantic network and published spatial model,
never from counting a thresholded dentition. Instance identity is independent
of tooth number, so duplicate suggestions cannot merge physical objects.
"""
from __future__ import annotations

from collections import Counter
import colorsys
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading

import numpy as np

from .inference import InferenceCancelled, _cancelled, _check_volume, _sha256

CODE_COMMIT = "b29d1017fa124f89645fa56f98649e3f3f43bdb0"
ARCHIVE_SHA256 = "803072574d12eb34c9989622a8a1508dc1998be95e9732f9718a4d0d032befd3"
SEMANTIC_SUBDIR = "Dataset121_ToothFairy2_Teeth/nnUNetTrainer_onlyMirror01_DASegOrd0__nnUNetPlans__3d_fullres_resample_torch_256_bs8_ctnorm"
INSTANCE_SUBDIR = "Dataset123_ToothFairy2fixed_teeth_spacing02_brd3px/nnUNetTrainer__nnUNetPlans__3d_fullres_resample_torch_192_bs8_ctnorm"
ARTIFACT_HASHES = {
    SEMANTIC_SUBDIR + "/dataset.json": "6586224875c5eb58dcec3a4648607cd53df5bbb4d3dd2c1affc68404e2ea3cc3",
    SEMANTIC_SUBDIR + "/fold_5/checkpoint_final.pth": "a724c26416bdd5fcb333aa7212870331fc586cccd8330fc8afe090cfdd9129c8",
    SEMANTIC_SUBDIR + "/plans.json": "a2a22e8180a1af47ea86481bc81af8157859edb8ed2d2dead2841fa2152ab88e",
    INSTANCE_SUBDIR + "/dataset.json": "614338e07b57f3fa56a9b70226fcffd63a79391d99c4990c28c956005d4ddd2d",
    INSTANCE_SUBDIR + "/fold_5/checkpoint_final.pth": "77e30a84af91e051f21bb43cea0f9e18e28388035741c9f2f4f357d47591a02f",
    INSTANCE_SUBDIR + "/plans.json": "6d15dac7a2235bd4bf640f23e025f59ac9f517765afafa4cc3e51a2b0d9f406c",
}
CODE_HASHES = {
    "toothseg/toothseg/postprocess_predictions/assign_mincost_tooth_labels.py": "849170809e9ab2851f54b01ea8cb0151f6cde4ac30b96b45fe10612f389d1538",
    "toothseg/datasets/toothfairy2/fdi_pair_distrs.json": "82ab04892277d36013be5ba9763ac334ea073fca7ebe8679086f1e33ed64ff29",
}


def _locations():
    project = Path(__file__).resolve().parents[1]
    for work in [project / "work"] + [p / "work" for p in project.parents]:
        python = work / "tooth-ai-venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        models = work / "toothseg/models/ToothSeg"
        if python.is_file() and models.is_dir():
            return work, python, models, work / "toothseg/ToothSeg-code"
    work = project / "work"
    return work, work / "tooth-ai-venv/Scripts/python.exe", work / "toothseg/models/ToothSeg", work / "toothseg/ToothSeg-code"


def detect_tooth_model_status(*, verify=False):
    work, python, models, code = _locations()
    try:
        marker = json.loads((work / "toothseg/runtime-ready.json").read_text(encoding="utf-8"))
        ready = marker.get("torch") == "2.6.0+cu124" and marker.get("nnunetv2") == "2.5.1" and marker.get("code_commit") == CODE_COMMIT
    except (OSError, ValueError, AttributeError):
        ready = False
    ready = bool(ready and python.is_file() and all((models / key).is_file() for key in ARTIFACT_HASHES)
                 and all((code / key).is_file() for key in CODE_HASHES))
    reason = "Ready: ToothSeg individual teeth and FDI suggestions" if ready else "Run Setup Tooth AI.ps1 to install individual tooth segmentation."
    if ready and verify:
        for root, artifacts in ((models, ARTIFACT_HASHES), (code, CODE_HASHES)):
            if any(_sha256(root / name) != expected for name, expected in artifacts.items()):
                ready, reason = False, "ToothSeg verification failed; reinstall the official model and code."
                break
    return {"ready": ready, "reason": reason, "python": str(python), "models": str(models), "code": str(code), "work": str(work)}


def tooth_roi(volume, base_result, fingerprint, *, margin_mm=15.0):
    """Return z/y/x slices from a source-checked, independent label snapshot."""
    if isinstance(margin_mm, bool) or not np.isscalar(margin_mm) or not np.isfinite(margin_mm) or not 0 <= margin_mm <= 50:
        raise ValueError("The tooth-region margin must be finite and between 0 and 50 mm.")
    base_result.validate_source(volume, fingerprint)
    revision = base_result.revision
    tooth_ids = [key for key, label in base_result.labels.items() if label["kind"] == "tooth" and label.get("review") != "rejected"]
    if not tooth_ids:
        raise ValueError("No tooth candidates are available to define the individual-tooth region.")
    snapshot = base_result.snapshot()
    mask = np.isin(snapshot, tooth_ids)
    if not np.any(mask):
        raise ValueError("The tooth candidates are empty. Review or correct the initial tooth masks first.")
    bounds = []
    for axis, spacing in enumerate(volume.spacing[::-1]):
        occupied = np.flatnonzero(np.any(mask, axis=tuple(i for i in range(3) if i != axis)))
        margin = int(np.ceil(margin_mm / spacing))
        bounds.append(slice(max(0, int(occupied[0]) - margin), min(mask.shape[axis], int(occupied[-1]) + margin + 1)))
    if base_result.revision != revision:
        raise RuntimeError("The source segmentation changed while preparing the tooth region. Run again.")
    return tuple(bounds), revision


def labels_from_instances(records):
    """Convert learned class suggestions; conflicting suggestions stay unassigned."""
    if not isinstance(records, list) or not 1 <= len(records) <= 512:
        raise ValueError("The model returned no valid tooth instances.")
    if any(not isinstance(item, dict) or set(item) != {"id", "predicted_fdi"}
           or type(item["predicted_fdi"]) is not int for item in records):
        raise ValueError("The model returned malformed tooth instance metadata.")
    counts = Counter(item.get("predicted_fdi") for item in records)
    labels = {}
    for item in records:
        identifier, fdi = item.get("id"), item.get("predicted_fdi")
        if type(identifier) is not int or not 1001 <= identifier <= 1512 or identifier in labels:
            raise ValueError("The model returned invalid instance identities.")
        if type(fdi) is not int or not 1 <= fdi // 10 <= 4 or not 1 <= fdi % 10 <= 8:
            raise ValueError("The model returned an unsupported permanent FDI suggestion.")
        rgb = colorsys.hsv_to_rgb((identifier * .61803398875) % 1, .5, .95)
        label = {"name": f"Tooth {fdi}" if counts[fdi] == 1 else f"Tooth object {identifier} (suggested {fdi})",
                 "kind": "tooth", "instance": True, "predicted_fdi": fdi, "review": "pending",
                 "color": "#" + "".join(f"{round(c * 255):02X}" for c in rgb)}
        if counts[fdi] == 1:
            label["fdi"] = fdi
        labels[identifier] = label
    return labels


def _execute(status, job, report, cancel):
    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
                "nnUNet_compile": "false", "nnUNet_raw": str(job / "raw"), "nnUNet_preprocessed": str(job / "preprocessed"), "nnUNet_results": str(job / "results")})
    command = [status["python"], "-u", str(Path(__file__).with_name("_tooth_worker.py")), "--job", str(job), "--models", status["models"], "--code", status["code"]]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
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
                raise InferenceCancelled("Individual tooth segmentation cancelled.")
            try:
                line = messages.get(timeout=.1)
                tail = (tail + [line])[-18:]
                if line.startswith("ASTER_PROGRESS:"):
                    report(line.partition(":")[2].strip())
            except queue.Empty:
                pass
            if process.poll() is not None and not reader.is_alive() and messages.empty():
                break
        if process.returncode:
            raise RuntimeError("Local ToothSeg worker failed. " + "\n".join(tail)[-1800:])
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=15)
        reader.join(timeout=3)
        process.stdout.close()


def run_tooth_segmentation(volume, fingerprint, progress=None, cancel=None, base_result=None):
    """Return source-grid individual teeth only; root merges preserved anatomy.

    A verified initial dental segmentation defines a 15 mm expanded region.
    If absent it is generated with the independent DentalSegmentator provider.
    This bounds memory; missed teeth outside that region remain a limitation.
    """
    from .segmentation import create_result
    from .session import _fingerprint, _match_volume
    from .inference import run_segmentation
    _check_volume(volume)
    source = _fingerprint(fingerprint, "Tooth AI source")
    geometry = {"shape": list(volume.data.shape), "spacing": list(volume.spacing), "origin": volume.origin.tolist(), "direction": volume.direction.tolist(), "synthetic": source["synthetic"]}
    geometry.update({k: v for k, v in source.items() if k not in geometry})
    _match_volume(source, geometry)
    if _cancelled(cancel):
        raise InferenceCancelled("Individual tooth segmentation cancelled.")
    status = detect_tooth_model_status()
    if not status["ready"]:
        raise RuntimeError(status["reason"])
    report = progress or (lambda _: None)
    if base_result is None:
        report("Finding the dental region with DentalSegmentator…")
        base_result = run_segmentation(volume, source, progress=report, cancel=cancel)
    roi, revision = tooth_roi(volume, base_result, source)
    shape = np.array([s.stop - s.start for s in roi])
    estimate = int(np.prod(np.ceil(shape * np.array(volume.spacing[::-1]) / .3)))
    if estimate > 50_000_000:
        raise ValueError("The tooth region exceeds this workstation's bounded model size. Tighten the initial tooth masks or use a smaller source field of view.")
    cropped = volume.data[roi].copy()
    if volume.valid_mask is not None:
        cropped[~volume.valid_mask[roi]] = -1000.0 if volume.is_hu else volume.data_min
    origin = volume.index_to_world([roi[2].start, roi[1].start, roi[0].start])
    job_root = Path(status["work"]) / "toothseg/jobs"
    job_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aster-teeth-", dir=job_root) as folder:
        job = Path(folder)
        np.save(job / "input.npy", cropped, allow_pickle=False)
        del cropped
        job_geometry = {"shape": shape.tolist(), "spacing": list(volume.spacing), "origin": origin.tolist(), "direction": volume.direction.tolist(),
                        "source_shape_zyx": list(volume.data.shape), "source_roi_zyx": [[s.start, s.stop] for s in roi], "margin_mm": 15.0}
        (job / "geometry.json").write_text(json.dumps(job_geometry, allow_nan=False), encoding="utf-8")
        _execute(status, job, report, cancel)
        if _cancelled(cancel):
            raise InferenceCancelled("Individual tooth segmentation cancelled.")
        if base_result.revision != revision:
            raise RuntimeError("The source segmentation changed during inference. Run again to use your latest corrections.")
        data = np.load(job / "output.npy", allow_pickle=False)
        labels = labels_from_instances(json.loads((job / "instances.json").read_text(encoding="utf-8")))
        if data.shape != tuple(shape) or data.dtype != np.uint16 or not set(np.unique(data)).issubset({0, *labels}):
            raise RuntimeError("ToothSeg returned an invalid region label map.")
        restored = np.zeros(volume.data.shape, dtype=np.uint16)
        restored[roi] = data
        if volume.valid_mask is not None:
            restored[~volume.valid_mask] = 0
        provenance = {"name": "ToothSeg", "version": "Zenodo 14893540; code " + CODE_COMMIT[:12], "model_sha256": ARCHIVE_SHA256,
                      "license": "CC-BY-4.0 (weights); Apache-2.0 (ToothSeg/nnU-Net)",
                      "method": "Learned semantic + border/core models, fold5, no TTA; explicit RPS model frame / LAS spatial prior; 15mm tooth ROI " + repr(job_geometry["source_roi_zyx"]) + "; exact source-grid restoration; all FDI suggestions need review"}
        return create_result(volume, source, restored, labels, provenance)
