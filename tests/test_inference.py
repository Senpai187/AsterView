"""Provider lifecycle and source integrity, using a real isolated test worker.

These tests do not substitute for the separately recorded real model execution.
They exercise cancellation, temporary-data cleanup and source-bound result flow
without loading multiple gigabytes of ML dependencies into the test process.
"""
import json
import subprocess
import sys
import threading

import numpy as np
import pytest

from asterview import inference
from asterview.model import Volume


def volume_and_source(*, padded=False):
    data = np.arange(7 * 9 * 11, dtype=np.float32).reshape(7, 9, 11)
    mask = np.ones_like(data, dtype=bool) if padded else None
    if mask is not None:
        mask[0] = False
    v = Volume(data, (.3, .5, 1.2), np.array([1., -3., 7.]), np.eye(3),
               {"synthetic": True}, valid_mask=mask)
    source = {"shape": list(v.data.shape), "spacing": list(v.spacing), "origin": v.origin.tolist(),
              "direction": v.direction.tolist(), "synthetic": True}
    return v, source


@pytest.fixture
def local_worker(tmp_path, monkeypatch):
    """Replace only model computation with a file-based numeric worker."""
    script = tmp_path / "worker.py"
    script.write_text("""
import argparse, json, time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(); p.add_argument('--job'); p.add_argument('--model'); a=p.parse_args()
j=Path(a.job)
source=np.load(j/'input.npy',allow_pickle=False)
geometry=json.loads((j/'geometry.json').read_text())
assert list(source.shape)==geometry['shape']
print('ASTER_PROGRESS:Model test entered',flush=True)
time.sleep(float(a.model))
result=np.full(source.shape,5,dtype=np.uint16)
np.save(j/'output.npy',result,allow_pickle=False)
""", encoding="utf-8")
    real_popen = subprocess.Popen
    state = {"delay": "0", "commands": [], "processes": []}

    def popen(command, **kwargs):
        if not isinstance(command, list) or len(command) < 3 or not str(command[2]).endswith("_inference_worker.py"):
            return real_popen(command, **kwargs)
        state["commands"].append(command)
        revised = [sys.executable, "-u", str(script)] + command[3:]
        revised[-1] = state["delay"]
        process = real_popen(revised, **kwargs)
        state["processes"].append(process)
        return process

    monkeypatch.setattr(inference.subprocess, "Popen", popen)
    monkeypatch.setattr(inference, "detect_model_status", lambda: {
        "ready": True, "python": sys.executable, "model": "0", "work": str(tmp_path)})
    return state, tmp_path


def test_label_contract_is_explicit_semantic_groups_and_canal_proxy():
    assert set(inference.LABELS) == {1, 2, 3, 4, 5}
    assert inference.LABELS[3]["kind"] == inference.LABELS[4]["kind"] == "tooth"
    assert all("fdi" not in item for item in inference.LABELS.values())
    assert "proxy" in inference.LABELS[5]["name"]
    assert len(inference.MODEL_SHA256) == 64


def test_complete_worker_result_owned_and_pending(local_worker):
    state, root = local_worker
    volume, source = volume_and_source()
    original = volume.data.copy()
    messages = []
    result = inference.run_segmentation(volume, source, progress=messages.append)
    assert result.data.shape == volume.data.shape
    assert result.data.dtype == np.uint16
    assert np.all(result.data == 5)
    assert result.source == source
    assert all(x["review"] == "pending" for x in result.labels.values())
    assert result.provenance["model_sha256"] == inference.MODEL_SHA256
    assert "Model test entered" in messages
    assert np.array_equal(volume.data, original)
    assert not list((root / "segmentation/jobs").iterdir())
    assert state["processes"][0].returncode == 0


def test_padding_cannot_receive_predicted_anatomy(local_worker):
    volume, source = volume_and_source(padded=True)
    original = volume.data.copy()
    result = inference.run_segmentation(volume, source)
    assert not np.any(result.data[0])
    assert np.all(result.data[1:] == 5)
    assert np.array_equal(volume.data, original)


def test_cancellation_kills_process_and_removes_temporary_patient_data(local_worker):
    state, root = local_worker
    state["delay"] = "60"
    volume, source = volume_and_source()
    cancelled = threading.Event()

    def progress(message):
        if message == "Model test entered":
            cancelled.set()

    with pytest.raises(inference.InferenceCancelled):
        inference.run_segmentation(volume, source, progress=progress, cancel=cancelled)
    assert state["processes"][0].poll() is not None
    assert not list((root / "segmentation/jobs").iterdir())


def test_pre_cancel_never_starts_worker(local_worker):
    state, _ = local_worker
    volume, source = volume_and_source()
    with pytest.raises(inference.InferenceCancelled):
        inference.run_segmentation(volume, source, cancel=lambda: True)
    assert not state["commands"]


@pytest.mark.parametrize("field,value", [("shape", [7, 9, 12]), ("spacing", [.4, .5, 1.2]),
                                          ("origin", [1., -3., 8.])])
def test_source_geometry_mismatch_never_starts_worker(local_worker, field, value):
    state, _ = local_worker
    volume, source = volume_and_source()
    source[field] = value
    with pytest.raises(ValueError):
        inference.run_segmentation(volume, source)
    assert not state["commands"]


@pytest.mark.parametrize("metadata", [{"geometry_calibrated": False}, {"modality": "MR"}])
def test_unsupported_source_never_starts_worker(local_worker, metadata):
    state, _ = local_worker
    volume, source = volume_and_source()
    volume.metadata.update(metadata)
    with pytest.raises(ValueError):
        inference.run_segmentation(volume, source)
    assert not state["commands"]


def test_readiness_requires_verified_runtime_marker(tmp_path, monkeypatch):
    python = tmp_path / "python.exe"
    python.touch()
    model = tmp_path / "model"
    for relative in inference.ARTIFACT_HASHES:
        path = model / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not trusted")
    monkeypatch.setattr(inference, "_locations", lambda: (tmp_path, python, model))
    assert not inference.detect_model_status()["ready"]
    marker = tmp_path / "segmentation/runtime-ready.json"
    marker.parent.mkdir()
    marker.write_text('{}')
    assert not inference.detect_model_status()["ready"]
    marker.write_text(json.dumps({"torch": "2.6.0+cu124", "nnunetv2": "2.2.1", "cuda": "12.4", "device": "Test CUDA"}))
    assert inference.detect_model_status()["ready"]
    assert not inference.detect_model_status(verify=True)["ready"]
