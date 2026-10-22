import json
import subprocess
import sys
import threading

import numpy as np
import pytest

from asterview.model import Volume
from asterview.segmentation import create_result
from asterview import tooth_inference as teeth
from asterview.inference import InferenceCancelled
from asterview._tooth_worker import prior_coordinates, model_class_to_fdi


def source_and_base():
    angle = np.deg2rad(27)
    direction = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    volume = Volume(np.full((48, 62, 84), 500., np.float32), (.8, 1.2, 1.6), np.array([7., -19., 31.]), direction,
                    {"synthetic": True})
    source = {"shape": list(volume.data.shape), "spacing": list(volume.spacing), "origin": volume.origin.tolist(),
              "direction": volume.direction.tolist(), "synthetic": True}
    mask = np.zeros(volume.data.shape, np.uint16)
    mask[20:24, 25:32, 36:43] = 3
    base = create_result(volume, source, mask, {3: {"name": "Upper teeth group", "kind": "tooth", "instance": False}},
                         {"name": "Source test", "version": "1"})
    return volume, source, base


@pytest.fixture
def model_worker(tmp_path, monkeypatch):
    state = {"jobs": [], "mutate": None}
    monkeypatch.setattr(teeth, "detect_tooth_model_status", lambda: {"ready": True, "work": str(tmp_path)})
    def execute(status, job, report, cancel):
        state["jobs"].append(job)
        geometry = json.loads((job / "geometry.json").read_text())
        state["geometry"] = geometry
        state["input"] = np.load(job / "input.npy", allow_pickle=False)
        output = np.full(geometry["shape"], 1001, np.uint16)
        output[:, :, :3] = 1002
        np.save(job / "output.npy", output, allow_pickle=False)
        (job / "instances.json").write_text(json.dumps([{"id": 1001, "predicted_fdi": 11}, {"id": 1002, "predicted_fdi": 13}]))
        if state["mutate"]:
            state["mutate"]()
    monkeypatch.setattr(teeth, "_execute", execute)
    return state, tmp_path


def test_anisotropic_region_and_oblique_origin_preserve_source_geometry():
    volume, source, base = source_and_base()
    roi, revision = teeth.tooth_roi(volume, base, source)
    assert [(x.start, x.stop) for x in roi] == [(10, 34), (12, 45), (17, 62)]
    assert revision == base.revision
    np.testing.assert_array_equal(base.data, base.snapshot())


def test_individual_result_is_source_grid_pending_and_does_not_modify_source(model_worker):
    state, root = model_worker
    volume, source, base = source_and_base()
    original, base_original = volume.data.copy(), base.data.copy()
    result = teeth.run_tooth_segmentation(volume, source, base_result=base)
    assert result.data.shape == volume.data.shape and result.data.dtype == np.uint16
    assert result.source == source
    assert set(result.labels) == {1001, 1002}
    assert all(item["instance"] and item["review"] == "pending" for item in result.labels.values())
    assert result.labels[1001]["fdi"] == 11
    assert result.labels[1002]["fdi"] == 13  # missing 12 does not shift numbering
    assert result.provenance["name"] == "ToothSeg"
    assert "15mm tooth ROI" in result.provenance["method"]
    np.testing.assert_allclose(state["geometry"]["origin"], volume.index_to_world([17, 12, 10]))
    assert not np.any(result.data[:10]) and not np.any(result.data[34:])
    np.testing.assert_array_equal(volume.data, original)
    np.testing.assert_array_equal(base.data, base_original)
    assert not list((root / "toothseg/jobs").iterdir())


def test_duplicate_suggestions_preserve_individual_objects_without_duplicate_assignment():
    labels = teeth.labels_from_instances([{"id": 1001, "predicted_fdi": 11}, {"id": 1002, "predicted_fdi": 11}, {"id": 1003, "predicted_fdi": 26}])
    assert len(labels) == 3
    assert "fdi" not in labels[1001] and "fdi" not in labels[1002]
    assert labels[1001]["predicted_fdi"] == labels[1002]["predicted_fdi"] == 11
    assert labels[1003]["fdi"] == 26


def test_learned_las_prior_coordinates_preserve_true_patient_ras_input():
    # The released ToothFairy2 prior has a positive x offset from FDI11 to21.
    # Physical RAS has the opposite sign; convert coordinates, never labels.
    ras = np.array([[4., 2., 7.], [-4., 2., 7.]])
    learned = prior_coordinates(ras)
    np.testing.assert_array_equal(learned[1] - learned[0], [8., 0., 0.])
    np.testing.assert_array_equal(ras, [[4., 2., 7.], [-4., 2., 7.]])


def test_model_class_mapping_is_unchanged_across_all_permanent_teeth():
    expected = [q * 10 + tooth for q in (1, 2, 3, 4) for tooth in range(1, 9)]
    assert [model_class_to_fdi(np.int64(i)) for i in range(1, 33)] == expected
    for invalid in (0, 33, True, 11.5):
        with pytest.raises(ValueError):
            model_class_to_fdi(invalid)


@pytest.mark.parametrize("records", [[], [{"id": 1001, "predicted_fdi": 19}], [{"id": True, "predicted_fdi": 11}],
                                    [{"id": 1001, "predicted_fdi": 51}], [{"id": 1001, "predicted_fdi": 11}, {"id": 1001, "predicted_fdi": 12}]])
def test_invalid_model_instance_metadata_is_rejected(records):
    with pytest.raises(ValueError):
        teeth.labels_from_instances(records)


def test_stale_base_revision_rejects_result_and_removes_temporary_input(model_worker):
    state, root = model_worker
    volume, source, base = source_and_base()
    state["mutate"] = lambda: base.set_review(3, "accepted")
    with pytest.raises(RuntimeError, match="changed during inference"):
        teeth.run_tooth_segmentation(volume, source, base_result=base)
    assert not list((root / "toothseg/jobs").iterdir())


def test_cancellation_before_preparation_never_starts_worker(model_worker):
    state, _ = model_worker
    volume, source, base = source_and_base()
    with pytest.raises(InferenceCancelled):
        teeth.run_tooth_segmentation(volume, source, cancel=lambda: True, base_result=base)
    assert not state["jobs"]


def test_changed_volume_identity_rejects_before_worker(model_worker):
    state, _ = model_worker
    volume, source, base = source_and_base()
    source["origin"][2] += 1
    with pytest.raises(ValueError):
        teeth.run_tooth_segmentation(volume, source, base_result=base)
    assert not state["jobs"]


def test_rejected_or_empty_groups_cannot_define_automatic_region(model_worker):
    state, _ = model_worker
    volume, source, base = source_and_base()
    base.set_review(3, "rejected")
    with pytest.raises(ValueError, match="No tooth candidates"):
        teeth.run_tooth_segmentation(volume, source, base_result=base)
    assert not state["jobs"]


def test_base_from_different_volume_is_not_used(model_worker):
    state, _ = model_worker
    volume, source, base = source_and_base()
    base.source["spacing"][0] += .1
    with pytest.raises(ValueError):
        teeth.run_tooth_segmentation(volume, source, base_result=base)
    assert not state["jobs"]


def test_bounded_region_rejects_excessive_model_allocation_before_worker(model_worker):
    state, _ = model_worker
    volume, source, base = source_and_base()
    volume.spacing = (100., 100., 100.)
    source["spacing"] = list(volume.spacing)
    base = create_result(volume, source, base.data, base.labels, base.provenance)
    with pytest.raises(ValueError, match="bounded model size"):
        teeth.run_tooth_segmentation(volume, source, base_result=base)
    assert not state["jobs"]


def test_live_tooth_worker_is_terminated_on_cancellation(tmp_path, monkeypatch):
    script = tmp_path / "wait_worker.py"
    script.write_text("import time; print('ASTER_PROGRESS:Worker waiting',flush=True); time.sleep(60)", encoding="utf-8")
    monkeypatch.setattr(teeth, "detect_tooth_model_status", lambda: {"ready": True, "work": str(tmp_path),
                      "python": sys.executable, "models": "test", "code": "test"})
    real_popen, children = subprocess.Popen, []
    def popen(command, **kwargs):
        if not isinstance(command, list) or len(command) < 3 or not str(command[2]).endswith("_tooth_worker.py"):
            return real_popen(command, **kwargs)
        child = real_popen([sys.executable, "-u", str(script)], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(teeth.subprocess, "Popen", popen)
    cancelled = threading.Event()
    volume, source, base = source_and_base()
    def progress(text):
        if text == "Worker waiting":
            cancelled.set()
    with pytest.raises(InferenceCancelled):
        teeth.run_tooth_segmentation(volume, source, progress=progress, cancel=cancelled, base_result=base)
    assert len(children) == 1 and children[0].poll() is not None
    assert not list((tmp_path / "toothseg/jobs").iterdir())
