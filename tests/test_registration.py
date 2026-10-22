import json
import subprocess
import sys
import threading

import numpy as np
import pytest

from asterview import registration as reg
from asterview.model import Volume


def fingerprint(volume):
    source = {"shape": list(volume.data.shape), "spacing": list(volume.spacing), "origin": volume.origin.tolist(),
              "direction": volume.direction.tolist(), "synthetic": bool(volume.metadata.get("synthetic", False))}
    if not source["synthetic"]:
        source.update({key: volume.metadata[key] for key in reg.IDENTITY_KEYS})
    return source


def analytic_volume(function=None, *, shape=(32, 34, 36), spacing=(.9, 1.1, 1.3), direction=None):
    direction = np.eye(3) if direction is None else np.asarray(direction)
    origin = -direction @ ((np.asarray(shape[::-1]) - 1) * spacing / 2)
    points = np.moveaxis(np.indices(shape)[::-1], 0, -1) * spacing @ direction.T + origin
    if function is None:
        function = lambda p: p[..., 0] + 2 * p[..., 1] + 3 * p[..., 2]
    return Volume(np.asarray(function(points), np.float32), spacing, origin, direction, {"synthetic": True})


def test_rigid_rotation_is_about_physical_pivot_and_translates_lps():
    center = [11, -27, 6]
    transform = reg.rigid_matrix([2, -3, 4], [0, 0, 90], center)
    np.testing.assert_allclose(reg.transform_points(center, transform), np.array(center) + [2, -3, 4])
    np.testing.assert_allclose(reg.transform_points(np.array(center) + [5, 0, 0], transform), np.array(center) + [2, 2, 4], atol=1e-12)
    points = np.array([[1., 2., 3.], [-10., 8., 7.]])
    np.testing.assert_allclose(reg.transform_points(reg.transform_points(points, transform), np.linalg.inv(transform)), points, atol=1e-12)


def test_manual_refinement_composes_in_fixed_frame_and_resets_review():
    volume = analytic_volume()
    base = reg.rigid_matrix([4, 2, 0], [0, 0, 30])
    result = reg.RegistrationResult(base, fingerprint(volume), fingerprint(volume), review="accepted")
    updated = result.refined([1, 2, 3], [0, 0, 90], [1, 0, 0])
    np.testing.assert_allclose(updated.transform, reg.rigid_matrix([1, 2, 3], [0, 0, 90], [1, 0, 0]) @ base)
    assert updated.review == "pending" and result.review == "accepted"
    np.testing.assert_array_equal(result.transform, base)


@pytest.mark.parametrize("matrix", [np.diag([2, 1, 1, 1]), np.diag([-1, 1, 1, 1]), np.zeros((4, 4)), np.ones((3, 3)),
                                   np.array([[1, .1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]), np.full((4, 4), np.nan)])
def test_nonrigid_invalid_or_reflected_transforms_are_rejected(matrix):
    with pytest.raises(ValueError):
        reg.validate_transform(matrix)


@pytest.mark.parametrize("direction", [np.eye(3), np.diag([-1., 1., 1.]), np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0.]])])
def test_registered_sampling_uses_inverse_physical_mapping_with_reflected_anisotropic_source(direction):
    transform = reg.rigid_matrix([1.3, -2.1, .7], [5, -3, 7])
    moving = analytic_volume(lambda p: (reg.transform_points(p, transform) * [1, 2, 3]).sum(axis=-1), direction=direction)
    original = moving.data.copy()
    frame = (np.zeros(3), [1, 0, 0], [0, 1, 0], (8, 9))
    sampled, valid = reg.sample_registered_plane(moving, transform, frame, (17, 19), return_valid=True)
    expected = np.linspace(-4, 4, 19)[None, :] - 2 * np.linspace(-4.5, 4.5, 17)[:, None]
    np.testing.assert_allclose(sampled, expected, atol=5e-5)
    assert valid.all()
    np.testing.assert_array_equal(moving.data, original)


def test_overlap_excludes_outside_fov_and_padding():
    moving = analytic_volume(shape=(9, 9, 9), spacing=(1, 1, 1))
    mask = np.ones(moving.data.shape, bool)
    mask[4, 4, 4] = False
    moving = Volume(moving.data, moving.spacing, moving.origin, moving.direction, {"synthetic": True}, valid_mask=mask)
    _, valid = reg.sample_registered_plane(moving, np.eye(4), ([0, 0, 0], [1, 0, 0], [0, 1, 0], (12, 12)), (13, 13), return_valid=True)
    assert not valid[6, 6] and not valid[0].any() and not valid[:, 0].any()
    assert valid[3, 3]
    _, slab_valid = reg.sample_registered_plane(moving, np.eye(4), ([0, 0, 3], [1, 0, 0], [0, 1, 0], (2, 2)), (3, 3), slab_mm=4, return_valid=True)
    assert not slab_valid.any()


def test_source_bound_json_reopens_pending_and_rejects_different_moving_source(tmp_path):
    fixed, moving = analytic_volume(), analytic_volume()
    fixed.metadata = {"series_uid": "fixed", "frame_of_reference_uid": "frame", "source_group": "a", "sop_instance_digest": "a" * 64}
    moving.metadata = {"series_uid": "moving", "frame_of_reference_uid": "frame", "source_group": "b", "sop_instance_digest": "b" * 64}
    fixed_fp, moving_fp = fingerprint(fixed), fingerprint(moving)
    result = reg.RegistrationResult(reg.rigid_matrix([2, 3, 4]), fixed_fp, moving_fp, review="accepted")
    path = tmp_path / "alignment.json"
    reg.save_registration(path, result, fixed, moving, fixed_fp, moving_fp)
    restored = reg.load_registration(path, fixed, moving, fixed_fp, moving_fp)
    assert restored.review == "pending"
    np.testing.assert_array_equal(restored.transform, result.transform)
    moving.metadata["sop_instance_digest"] = "c" * 64
    with pytest.raises(ValueError, match="different DICOM"):
        reg.load_registration(path, fixed, moving, fixed_fp, fingerprint(moving))


def test_malformed_alignment_metadata_gives_validation_error_without_changing_sources(tmp_path):
    fixed, moving = analytic_volume(), analytic_volume()
    sources = fingerprint(fixed), fingerprint(moving)
    result = reg.RegistrationResult(np.eye(4), *sources)
    path = tmp_path / "alignment.json"
    reg.save_registration(path, result, fixed, moving, *sources)
    payload = json.loads(path.read_text())
    payload["review"] = []
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        reg.load_registration(path, fixed, moving, *sources)
    np.testing.assert_array_equal(result.transform, np.eye(4))


def test_registration_metadata_owns_fingerprints():
    volume = analytic_volume()
    source = fingerprint(volume)
    result = reg.RegistrationResult(np.eye(4), source, source)
    source["origin"][0] += 1
    assert result.fixed_source["origin"] != source["origin"]


@pytest.fixture
def mock_worker(tmp_path, monkeypatch):
    state = {"calls": 0, "mutate": None}
    monkeypatch.setattr(reg, "detect_registration_status", lambda: {"ready": True, "work": str(tmp_path)})
    def execute(status, job, report, cancel):
        state["calls"] += 1
        state["config"] = json.loads((job / "config.json").read_text())
        np.testing.assert_array_equal(np.load(job / "fixed.npy"), state["expected_fixed"])
        (job / "result.json").write_text(json.dumps({"moving_to_fixed": reg.rigid_matrix([1, 2, 3]).tolist(), "metric": -.7,
                                                     "method": "Test worker", "warnings": []}))
        if state["mutate"]:
            state["mutate"]()
    monkeypatch.setattr(reg, "_run_worker", execute)
    state["jobs"] = tmp_path / "registration/jobs"
    return state


def test_provider_preserves_arrays_and_physical_initial_transform(mock_worker):
    fixed, moving = analytic_volume(), analytic_volume(direction=np.diag([-1., 1., 1.]))
    before = fixed.data.copy(), moving.data.copy()
    mock_worker["expected_fixed"] = fixed.data.copy()
    initial = reg.rigid_matrix([1, 4, 5], [2, -3, 5])
    result = reg.register_volumes(fixed, moving, fingerprint(fixed), fingerprint(moving), initial_transform=initial, bone_focus=True)
    assert result.review == "pending" and result.metric == -.7
    assert mock_worker["config"]["initial_moving_to_fixed"] == initial.tolist()
    assert mock_worker["config"]["bone_focus"] is True
    np.testing.assert_array_equal(fixed.data, before[0])
    np.testing.assert_array_equal(moving.data, before[1])
    assert not list(mock_worker["jobs"].iterdir())


def test_cancellation_prevents_start_and_nonuniform_sources_rejected(mock_worker):
    fixed, moving = analytic_volume(), analytic_volume()
    with pytest.raises(reg.RegistrationCancelled):
        reg.register_volumes(fixed, moving, fingerprint(fixed), fingerprint(moving), cancel=lambda: True)
    moving.metadata["nonuniform_spacing"] = True
    with pytest.raises(ValueError, match="Nonuniform"):
        reg.register_volumes(fixed, moving, fingerprint(fixed), fingerprint(moving))
    assert mock_worker["calls"] == 0


def test_changed_source_during_worker_rejected_and_job_cleaned(mock_worker):
    fixed, moving = analytic_volume(), analytic_volume()
    fixed_fp, moving_fp = fingerprint(fixed), fingerprint(moving)
    mock_worker["expected_fixed"] = fixed.data.copy()
    mock_worker["mutate"] = lambda: moving.origin.__setitem__(0, moving.origin[0] + 1)
    with pytest.raises(ValueError, match="geometry"):
        reg.register_volumes(fixed, moving, fixed_fp, moving_fp)
    assert not list(mock_worker["jobs"].iterdir())


def test_cancellation_terminates_only_the_owned_worker(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    processes = []
    def start(command, **kwargs):
        child = real_popen([sys.executable, "-u", "-c", "import threading; print('ASTER_PROGRESS:worker started',flush=True); threading.Event().wait(30)"], **kwargs)
        processes.append(child)
        return child
    monkeypatch.setattr(reg.subprocess, "Popen", start)
    cancelled = threading.Event()
    with pytest.raises(reg.RegistrationCancelled):
        reg._run_worker({"python": sys.executable}, tmp_path, lambda _: cancelled.set(), cancelled)
    assert len(processes) == 1 and processes[0].poll() is not None


def phantom_values(p):
    result = np.zeros(p.shape[:-1])
    for center, radii, weight in [((-12, -5, -3), (8, 13, 11), 1000), ((10, -7, 7), (6, 9, 8), 1400),
                                  ((3, 12, -12), (9, 6, 6), 1100), ((-10, 15, 11), (5, 7, 5), 700),
                                  ((19, 8, -3), (4, 5, 12), 1600)]:
        result += weight * np.exp(-1.5 * np.sum(((p - center) / radii) ** 2, axis=-1))
    return result


@pytest.mark.parametrize("direction", [np.eye(3), np.diag([-1., 1., 1.])])
def test_actual_simpleitk_recovers_known_motion_on_anisotropic_phantom(direction):
    if not reg.detect_registration_status()["ready"]:
        pytest.skip("Optional local registration runtime is not installed")
    fixed = analytic_volume(phantom_values, shape=(58, 66, 74))
    expected = reg.rigid_matrix([2, -1.3, 1.7], [2, -3, 4])
    moving = analytic_volume(lambda p: phantom_values(reg.transform_points(p, expected)), shape=(58, 66, 74), direction=direction)
    result = reg.register_volumes(fixed, moving, fingerprint(fixed), fingerprint(moving))
    landmarks = np.array([[-20, -16, -15], [20, 12, 16], [-10, 15, 11], [3, 12, -12], [19, 8, -3]])
    error = np.linalg.norm(reg.transform_points(landmarks, result.transform) - reg.transform_points(landmarks, expected), axis=1)
    assert error.max() < .6
    assert result.review == "pending"


def test_actual_simpleitk_does_not_degrade_exact_initial_alignment_of_sharp_phantom():
    if not reg.detect_registration_status()["ready"]:
        pytest.skip("Optional local registration runtime is not installed")
    from asterview.demo import make_demo
    fixed = make_demo()
    moving = Volume(fixed.data.copy(), fixed.spacing, fixed.origin + [4, -3, 2], fixed.direction,
                    {"synthetic": True}, window=fixed.window, is_hu=fixed.is_hu)
    result = reg.register_volumes(fixed, moving, fingerprint(fixed), fingerprint(moving))
    np.testing.assert_allclose(result.transform, reg.rigid_matrix([-4, 3, -2]), atol=1e-9)
    assert any("initial alignment was retained" in warning for warning in result.warnings)
