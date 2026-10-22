import copy
import numpy as np
import pytest

from asterview.model import Volume
from asterview.segmentation import create_result
from asterview.auto_dental import AutoDentalCancelled, propose_arch, propose_tmj


def fingerprint(volume):
    source = dict(shape=list(volume.data.shape), spacing=list(volume.spacing),
                  origin=volume.origin.tolist(), direction=volume.direction.tolist(), synthetic=True)
    return source


def make_case(*, instances=True, unilateral=False):
    shape = (100, 100, 120)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    volume = Volume(np.zeros(shape, dtype=np.float32), (1., 1., 1.), [-10., -20., -10.], np.eye(3), metadata={"synthetic": True})
    data = np.zeros(shape, dtype=np.uint16)
    labels = {2: {"name": "Mandible", "kind": "jaw"}, 4: {"name": "Lower teeth group", "kind": "tooth", "instance": False}}
    theta = np.linspace(-np.pi / 2, np.pi / 2, 45)
    body = np.zeros(shape[1:], dtype=bool)
    for angle in theta:
        body |= (xx[0] - (60 + 30 * np.sin(angle))) ** 2 + (yy[0] - (60 - 40 * np.cos(angle))) ** 2 <= 16
    data[(zz >= 15) & (zz <= 31) & body] = 2
    for x in (30, 90):
        ramus = (abs(xx - x) <= 5) & (abs(yy - 60) <= 5) & (zz >= 23) & (zz <= 80)
        head = ((xx - x) / 8) ** 2 + ((yy - 64) / 5) ** 2 + ((zz - 80) / 5) ** 2 <= 1
        data[ramus | head] = 2
    for i, angle in enumerate(np.linspace(-np.pi / 2, np.pi / 2, 13)):
        x, y = 60 + 23 * np.sin(angle), 51 - 29 * np.cos(angle)
        tooth = (xx - x) ** 2 + (yy - y) ** 2 <= 2.5 ** 2
        tooth = tooth & (zz >= 31) & (zz <= 43)
        key = 1001 + i if instances else 4
        if instances:
            labels[key] = {"name": f"Tooth candidate {i}", "kind": "tooth", "instance": True,
                           "predicted_fdi": 31 + i % 8}
        data[tooth] = key
    if unilateral:
        data[:, :, 60:] = 0
    result = create_result(volume, fingerprint(volume), data, labels, {"name": "Synthetic test", "version": "1"})
    return volume, result


def arch(volume, result, **kwargs):
    return propose_arch(volume, result, fingerprint(volume), plane_center_lps=volume.center,
                        right_lps=volume.direction[:, 0], up_lps=-volume.direction[:, 1], **kwargs)


def test_instance_arch_follows_occupied_u_and_current_plane_without_mutating_source():
    volume, result = make_case()
    before, labels = result.data.copy(), copy.deepcopy(result.labels)
    proposal = arch(volume, result)
    points = proposal.points_lps_mm
    assert points.shape == (13, 3)
    assert np.allclose(points[:, 2], volume.center[2])
    assert np.min(points[:, 1]) < points[0, 1] - 20
    assert points[-1, 0] > points[0, 0]
    assert proposal.review == "pending" and proposal.provenance["method"] == "tooth-instance-centroid-geodesic"
    assert proposal.source == fingerprint(volume) and proposal.warnings
    assert np.array_equal(result.data, before) and result.labels == labels


@pytest.mark.parametrize("instances", [True, False])
def test_automatic_arch_depth_tracks_teeth_instead_of_initial_skull_plane(instances):
    volume, result = make_case(instances=instances)
    before = result.data.copy()
    requested = volume.center + [0., 0., 35.]
    proposal = propose_arch(volume, result, fingerprint(volume),
                            plane_center_lps=requested, right_lps=[1, 0, 0],
                            up_lps=[0, -1, 0], depth_mode="anatomy")
    # Teeth occupy source z=31..43, translated by origin z=-10.
    np.testing.assert_allclose(proposal.points_lps_mm[:, 2], 27., atol=.5)
    assert np.min(np.abs(proposal.points_lps_mm[:, 2] - requested[2])) > 40
    assert proposal.provenance["depth_mode"] == "anatomy"
    np.testing.assert_allclose(proposal.provenance["requested_plane_center_lps_mm"], requested)
    np.testing.assert_array_equal(result.data, before)


def test_anatomy_depth_respects_oblique_orientation_and_is_independent_of_initial_height():
    volume, result = make_case()
    angle = np.deg2rad(16)
    right, up = np.array([np.cos(angle), 0, np.sin(angle)]), np.array([0, -1, 0])
    normal = np.cross(right, up)
    options = dict(right_lps=right, up_lps=up, depth_mode="anatomy")
    before = propose_arch(volume, result, fingerprint(volume),
                          plane_center_lps=volume.center, **options)
    after = propose_arch(volume, result, fingerprint(volume),
                         plane_center_lps=volume.center + normal * 40, **options)
    np.testing.assert_allclose(before.points_lps_mm, after.points_lps_mm, atol=1e-10)
    plane = np.asarray(before.provenance["plane_center_lps_mm"])
    np.testing.assert_allclose((before.points_lps_mm - plane) @ normal, 0, atol=1e-10)


def test_arch_depth_mode_rejects_unknown_strategy():
    volume, result = make_case()
    with pytest.raises(ValueError, match="depth mode"):
        arch(volume, result, depth_mode="unknown")


def test_grouped_mask_uses_occupancy_ridge_instead_of_a_template():
    volume, result = make_case(instances=False)
    proposal = arch(volume, result)
    assert len(proposal.points_lps_mm) >= 3
    assert proposal.provenance["method"] == "grouped-tooth-projection-ridge"
    assert np.ptp(proposal.points_lps_mm[:, 0]) > 30
    shifted = result.data.copy()
    shifted[:, :, 4:] = result.data[:, :, :-4]
    shifted[:, :, :4] = 0
    other = create_result(volume, fingerprint(volume), shifted, result.labels, result.provenance)
    moved = arch(volume, other)
    assert np.mean(moved.points_lps_mm[:, 0]) > np.mean(proposal.points_lps_mm[:, 0]) + 2


def test_mandible_projection_fallback_is_identified_explicitly():
    volume, result = make_case()
    for item in result.labels.values():
        if item["kind"] == "tooth":
            item["review"] = "rejected"
    proposal = arch(volume, result)
    assert proposal.provenance["method"] == "mandibular-projection-ridge"
    assert any("ramus" in text for text in proposal.warnings)


def test_large_dental_gaps_do_not_produce_a_full_jaw_bridge():
    volume, result = make_case()
    result.data[np.isin(result.data, [1006, 1007, 1008])] = 0
    proposal = arch(volume, result, max_gap_mm=8)
    assert any("gap" in text.casefold() for text in proposal.warnings)
    assert np.ptp(proposal.points_lps_mm[:, 0]) < 45


def test_upper_arch_requires_upper_evidence_and_never_uses_mandible_as_maxilla():
    volume, result = make_case()
    with pytest.raises(ValueError, match="upper dental mask"):
        arch(volume, result, preferred_jaw="upper")


def test_current_corrected_plane_and_anisotropic_geometry_are_respected():
    volume, result = make_case()
    volume = Volume(volume.data, (.7, .9, 1.4), volume.origin, volume.direction, metadata={"synthetic": True})
    result = create_result(volume, fingerprint(volume), result.data, result.labels, result.provenance)
    angle = np.deg2rad(16)
    right = np.array([np.cos(angle), 0, np.sin(angle)])
    up = np.array([0, -1, 0])
    proposal = propose_arch(volume, result, fingerprint(volume), plane_center_lps=volume.center, right_lps=right, up_lps=up)
    assert np.allclose((proposal.points_lps_mm - volume.center) @ np.cross(right, up), 0, atol=1e-9)
    assert np.isfinite(proposal.points_lps_mm).all()


@pytest.mark.parametrize("value", [None, True, "18", float("nan"), 0, 60])
def test_invalid_arch_gap_is_rejected(value):
    volume, result = make_case()
    with pytest.raises(ValueError, match="gap"):
        arch(volume, result, max_gap_mm=value)


def test_nonorthogonal_or_offsource_plane_cannot_create_a_fake_arch():
    volume, result = make_case()
    with pytest.raises(ValueError, match="orthogonal"):
        propose_arch(volume, result, fingerprint(volume), plane_center_lps=volume.center, right_lps=[1, 0, 0], up_lps=[1, 1, 0])
    with pytest.raises(ValueError, match="source coverage"):
        propose_arch(volume, result, fingerprint(volume), plane_center_lps=[0, 0, 1000], right_lps=[1, 0, 0], up_lps=[0, -1, 0])


def test_tmj_derives_bilateral_superoposterior_heads_and_patient_space_axes():
    volume, result = make_case()
    proposal = propose_tmj(volume, result, fingerprint(volume))
    assert set(proposal.centers) == {"left", "right"}
    assert proposal.centers["right"][0] < 50 < proposal.centers["left"][0]
    for side in proposal.centers:
        assert proposal.centers[side][1] > 35
        assert proposal.centers[side][2] > 65
        assert np.isclose(np.linalg.norm(proposal.axes[side]), 1)
    assert abs(proposal.midline_lps_x - 50) < 2
    assert proposal.review == "pending"
    assert any("not scanner zero" in warning for warning in proposal.warnings)


def test_tmj_unilateral_mask_does_not_invent_a_contralateral_condyle():
    volume, result = make_case(unilateral=True)
    unconfirmed = propose_tmj(volume, result, fingerprint(volume))
    assert unconfirmed.centers == {}
    confirmed = propose_tmj(volume, result, fingerprint(volume), midline_lps_x=50.)
    assert set(confirmed.centers) == {"right"}
    assert any("left" in warning for warning in confirmed.warnings)


def test_tmj_reflected_voxel_storage_preserves_physical_candidates():
    volume, result = make_case()
    before = propose_tmj(volume, result, fingerprint(volume), midline_lps_x=50)
    direction = np.eye(3); direction[0, 0] = -1
    reflected = Volume(volume.data[:, :, ::-1].copy(), volume.spacing,
                       volume.origin + [volume.dimensions[0] - 1, 0, 0], direction, metadata={"synthetic": True})
    other = create_result(reflected, fingerprint(reflected), result.data[:, :, ::-1].copy(), result.labels, result.provenance)
    after = propose_tmj(reflected, other, fingerprint(reflected), midline_lps_x=50)
    for side in before.centers:
        assert np.allclose(before.centers[side], after.centers[side])
        assert np.allclose(before.axes[side], after.axes[side])


def test_tmj_source_axis_permutation_preserves_physical_candidates():
    volume, result = make_case()
    before = propose_tmj(volume, result, fingerprint(volume), midline_lps_x=50)
    permutation = [1, 2, 0]
    other_volume = Volume(volume.data.transpose(2, 0, 1).copy(), tuple(np.asarray(volume.spacing)[permutation]),
                          volume.origin, volume.direction[:, permutation], metadata={"synthetic": True})
    other = create_result(other_volume, fingerprint(other_volume), result.data.transpose(2, 0, 1).copy(), result.labels, result.provenance)
    after = propose_tmj(other_volume, other, fingerprint(other_volume), midline_lps_x=50)
    for side in before.centers:
        assert np.allclose(before.centers[side], after.centers[side])
        assert np.allclose(before.axes[side], after.axes[side])


@pytest.mark.parametrize("midline", [True, "50", [50], float("nan"), float("inf")])
def test_tmj_invalid_confirmed_midline_is_rejected(midline):
    volume, result = make_case()
    with pytest.raises(ValueError, match="midline"):
        propose_tmj(volume, result, fingerprint(volume), midline_lps_x=midline)


def test_source_mismatch_cancel_and_stale_snapshot_cannot_supply_a_proposal(monkeypatch):
    import asterview.auto_dental as module
    volume, result = make_case()
    wrong = fingerprint(volume); wrong["origin"] = [1., 2., 3.]
    with pytest.raises(ValueError, match="geometry"):
        propose_tmj(volume, result, wrong)
    with pytest.raises(AutoDentalCancelled):
        arch(volume, result, cancel=lambda: True)
    with pytest.raises(AutoDentalCancelled):
        propose_tmj(volume, result, fingerprint(volume), cancel=lambda: True)
    original = module._graph_path
    def mutation(*args, **kwargs):
        outcome = original(*args, **kwargs)
        result.revision += 1
        return outcome
    monkeypatch.setattr(module, "_graph_path", mutation)
    with pytest.raises(ValueError, match="changed"):
        arch(volume, result)


def test_tmj_refuses_rejected_or_missing_mandibular_evidence():
    volume, result = make_case()
    result.labels[2]["review"] = "rejected"
    with pytest.raises(ValueError, match="mandible mask"):
        propose_tmj(volume, result, fingerprint(volume))


def test_tmj_does_not_select_a_taller_anterior_coronoid_like_peak():
    volume, result = make_case()
    # Deliberate superior anterior decoys must not outrank posterior heads.
    result.data[87:94, 39:45, 27:34] = 2
    result.data[87:94, 39:45, 87:94] = 2
    proposal = propose_tmj(volume, result, fingerprint(volume), midline_lps_x=50)
    assert set(proposal.centers) == {"left", "right"}
    assert all(center[1] > 35 and center[2] < 76 for center in proposal.centers.values())


def test_source_truncation_at_condylar_heads_withholds_automatic_setup():
    volume, result = make_case()
    cropped = Volume(volume.data[:83].copy(), volume.spacing, volume.origin, volume.direction, metadata={"synthetic": True})
    partial = create_result(cropped, fingerprint(cropped), result.data[:83].copy(), result.labels, result.provenance)
    proposal = propose_tmj(cropped, partial, fingerprint(cropped), midline_lps_x=50)
    assert proposal.centers == {}
    assert any("boundary" in warning for warning in proposal.warnings)


def test_declared_but_empty_tooth_instances_do_not_create_an_arch():
    volume, result = make_case()
    result.data[result.data >= 1000] = 0
    with pytest.raises(ValueError, match="nonempty"):
        arch(volume, result)


def test_inflight_cancel_during_mask_sampling_leaves_everything_untouched():
    volume, result = make_case()
    before = result.data.copy()
    count = 0
    def cancel():
        nonlocal count
        count += 1
        return count > 12
    with pytest.raises(AutoDentalCancelled):
        propose_tmj(volume, result, fingerprint(volume), cancel=cancel)
    assert np.array_equal(before, result.data)
