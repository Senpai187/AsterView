import copy
import io
import json
import zipfile

import numpy as np
import pytest

from asterview.model import Volume
from asterview.segmentation import (
    FORMAT, SegmentationResult, create_result, export_segmentation,
    read_segmentation,
)


def fingerprint(volume):
    meta = volume.metadata
    source = {"shape": list(volume.data.shape), "spacing": list(volume.spacing),
              "origin": volume.origin.tolist(), "direction": volume.direction.tolist(),
              "synthetic": bool(meta.get("synthetic", False))}
    for key in ("series_uid", "frame_of_reference_uid", "source_group", "sop_instance_digest"):
        if key in meta:
            source[key] = meta[key]
    return source


def make_volume(shape=(9, 11, 13), spacing=(.5, .8, 1.7), reflected=False, real=False):
    a, b = np.deg2rad([29, -17])
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    direction = rz @ ry
    if reflected:
        direction[:, 0] *= -1
    metadata = {"synthetic": not real}
    if real:
        metadata.update(series_uid="1.2.3", frame_of_reference_uid="1.2.4",
                        source_group="source-stack-1", sop_instance_digest="a" * 64)
    return Volume(np.zeros(shape), spacing, np.array([11.3, -7.1, 22.6]), direction, metadata=metadata)


def make_result(volume=None, data=None, labels=None, **kwargs):
    volume = make_volume() if volume is None else volume
    data = np.zeros(volume.data.shape, dtype=np.uint16) if data is None else data
    labels = {3: {"name": "Upper teeth", "kind": "tooth", "color": "#55AAEE"},
              5: {"name": "Mandibular canal proxy", "kind": "canal"}} if labels is None else labels
    result = SegmentationResult(data, labels, fingerprint(volume), {"name": "Test producer", "version": "1", "model_sha256": "b" * 64}, **kwargs)
    result.validate_source(volume, fingerprint(volume))
    return result


def axial(volume, z=4):
    return volume.index_to_world([6, 5, z]), volume.direction[:, 0], volume.direction[:, 1]


def test_result_owns_mask_and_metadata_without_copying_source_image():
    volume = make_volume()
    data = np.zeros(volume.data.shape, dtype=np.uint8)
    labels = {3: {"name": "Upper teeth", "kind": "tooth"}}
    result = create_result(volume, fingerprint(volume), data, labels, {"name": "Manual", "version": "1"})
    data[0, 0, 0] = 3
    labels[3]["name"] = "Changed outside"
    assert result.data[0, 0, 0] == 0
    assert result.labels[3]["name"] == "Upper teeth"
    assert result.data.flags.c_contiguous
    assert result.labels[3]["review"] == "pending"
    assert result.labelmap is result.data and result.producer is result.provenance
    assert result.source["shape"] == list(volume.data.shape)


@pytest.mark.parametrize("dtype", [np.float32, np.int16, np.bool_, object, np.uint32])
def test_only_bounded_integer_voxel_labels_are_accepted(dtype):
    volume = make_volume()
    with pytest.raises(ValueError, match="uint8 or uint16"):
        make_result(volume, np.zeros(volume.data.shape, dtype=dtype))


@pytest.mark.parametrize("change", ["shape", "undeclared", "zero-id", "float-id", "name", "kind", "review", "color", "fdi"])
def test_malformed_results_cannot_enter_viewer(change):
    volume = make_volume()
    result = make_result(volume)
    data, labels = result.data.copy(), copy.deepcopy(result.labels)
    if change == "shape": data = data[:3]
    if change == "undeclared": data[0, 0, 0] = 42
    if change == "zero-id": labels[0] = {"name": "Background", "kind": "other"}
    if change == "float-id": labels[1.5] = {"name": "Fraction", "kind": "other"}
    if change == "name": labels[3]["name"] = "bad\x00name"
    if change == "kind": labels[3]["kind"] = "diagnosis"
    if change == "review": labels[3]["review"] = "definitely correct"
    if change == "color": labels[3]["color"] = "blue"
    if change == "fdi": labels[3]["fdi"] = 19
    with pytest.raises(ValueError):
        make_result(volume, data, labels)


def test_real_source_identity_includes_exact_stack_and_instance_digest():
    volume = make_volume(real=True)
    result = make_result(volume)
    for key in ("series_uid", "frame_of_reference_uid", "source_group", "sop_instance_digest"):
        wrong = fingerprint(volume)
        wrong[key] = "c" * 64 if key == "sop_instance_digest" else "other"
        with pytest.raises(ValueError, match="different DICOM"):
            result.validate_source(volume, wrong)
    wrong_volume = make_volume(real=True)
    wrong_volume.metadata["source_group"] = "other-stack"
    with pytest.raises(ValueError, match="different source"):
        result.validate_source(wrong_volume, fingerprint(volume))
    with pytest.raises(ValueError, match="different source"):
        result.paint(wrong_volume, *axial(wrong_volume), 1, 3)


def test_fingerprint_cannot_lie_about_actual_volume_geometry():
    volume = make_volume()
    result = make_result(volume)
    wrong_volume = make_volume(spacing=(.5, .8, 1.8))
    with pytest.raises(ValueError, match="geometry"):
        result.validate_source(wrong_volume, fingerprint(volume))
    with pytest.raises(ValueError, match="geometry"):
        result.sample_plane(wrong_volume, (*axial(wrong_volume), (5., 8.)), (11, 13))


@pytest.mark.parametrize("reflected", [False, True])
def test_brush_uses_physical_radius_and_anisotropic_reflected_voxel_centers(reflected):
    volume = make_volume(reflected=reflected)
    result = make_result(volume)
    center, right, up = axial(volume)
    count = result.paint(volume, center, right, up, 1.01, 3)
    z, y, x = np.indices(volume.data.shape)
    expected = (z == 4) & ((x - 6) ** 2 * .5**2 + (y - 5) ** 2 * .8**2 <= 1.01**2)
    assert count == np.count_nonzero(expected)
    assert np.array_equal(result.data == 3, expected)
    assert result.undo() == count
    assert not result.data.any()


def test_oblique_cylindrical_stroke_matches_world_space_disk_and_plane_depth():
    volume = make_volume(reflected=True)
    result = make_result(volume)
    center = volume.index_to_world([6.2, 5.1, 4.3])
    right = np.array([1, 2, 0], dtype=float) / np.sqrt(5)
    normal = np.array([2, -1, 3], dtype=float) / np.sqrt(14)
    up = np.cross(normal, right)
    radius, depth = 1.7, .7
    result.paint(volume, center, right, up, radius, 5, thickness_mm=depth)
    z, y, x = np.indices(volume.data.shape)
    world = volume.index_to_world(np.stack([x, y, z], axis=-1))
    delta = world - center
    expected = ((delta @ right) ** 2 + (delta @ up) ** 2 <= radius**2 + 1e-10) & (np.abs(delta @ normal) <= depth / 2 + 1e-10)
    assert np.array_equal(result.data == 5, expected)


def test_brush_preserves_padding_and_unrelated_labels_and_erase_targets_selected_label():
    volume = make_volume()
    volume.valid_mask = np.ones(volume.data.shape, dtype=bool)
    volume.valid_mask[4, 5, 6] = False
    data = np.zeros(volume.data.shape, dtype=np.uint16)
    data[4, 5, 7] = 5
    result = make_result(volume, data)
    result.paint(volume, *axial(volume), 1.1, 3)
    assert result.data[4, 5, 6] == 0
    assert result.data[4, 5, 7] == 5
    result.paint(volume, *axial(volume), 1.1, 3, erase=True)
    assert result.data[4, 5, 7] == 5 and not np.any(result.data == 3)
    result.paint(volume, *axial(volume), 1.1, 3, overwrite=True)
    assert result.data[4, 5, 7] == 3


def test_rejection_retains_voxels_and_edit_requires_review_again():
    volume = make_volume()
    result = make_result(volume)
    result.paint(volume, *axial(volume), 1., 3)
    before = result.data.copy()
    result.set_review(3, "rejected")
    assert np.array_equal(result.data, before)
    result.set_review(3, "accepted")
    result.paint(volume, *axial(volume), 1., 3, erase=True)
    assert result.labels[3]["review"] == "pending"
    result.undo()
    assert result.labels[3]["review"] == "accepted"
    assert np.array_equal(result.data, before)


def test_gesture_aggregates_repeated_dabs_and_undo_restores_original_once():
    volume = make_volume()
    result = make_result(volume)
    result.set_review(3, "accepted")
    result.begin_stroke()
    result.paint(volume, *axial(volume), 1.2, 3)
    result.paint(volume, volume.index_to_world([7, 5, 4]), *axial(volume)[1:], 1.2, 3)
    result.paint(volume, *axial(volume), .5, 3, erase=True)
    assert result.can_undo
    count = result.end_stroke()
    assert count > 0 and len(result._undo) == 1
    assert result.undo() == count
    assert not result.data.any()
    assert result.labels[3]["review"] == "accepted"
    assert not result.can_undo and result.undo() == 0


def test_undo_limit_refuses_unrecoverable_edit_before_modifying_data():
    volume = make_volume()
    result = make_result(volume, max_undo_bytes=1)
    with pytest.raises(ValueError, match="undo memory"):
        result.paint(volume, *axial(volume), 1., 3)
    assert not result.data.any() and result.revision == 0


def test_undo_limit_evicts_oldest_strokes_but_preserves_latest_recoverable_change():
    volume = make_volume()
    result = make_result(volume, max_undo_bytes=20)
    for x in (3, 5, 7):
        result.paint(volume, volume.index_to_world([x, 5, 4]), *axial(volume)[1:], .1, 3)
    assert len(result._undo) == 2
    assert result.undo() == 1 and result.undo() == 1
    assert result.data[4, 5, 3] == 3 and not result.can_undo


@pytest.mark.parametrize("reflected", [False, True])
def test_plane_sampling_reconstructs_all_source_pixel_centers_without_interpolation(reflected):
    volume = make_volume(reflected=reflected)
    data = np.zeros(volume.data.shape, dtype=np.uint16)
    data[4, 1:10:2, 1:12:2] = 3
    data[4, 2:10:2, 2:12:2] = 5
    data[4, 0, :] = 3
    data[4, -1, :] = 5
    data[4, :, 0] = 3
    data[4, :, -1] = 5
    result = make_result(volume, data)
    frame = (*axial(volume), ((volume.dimensions[0] - 1) * .5, (volume.dimensions[1] - 1) * .8))
    sampled = result.sample_plane(volume, frame, (11, 13))
    assert sampled.dtype == data.dtype
    assert np.array_equal(sampled, data[4, ::-1, :])
    outside = (volume.index_to_world([6, 5, 100]), frame[1], frame[2], frame[3])
    assert not result.sample_plane(volume, outside, (11, 13)).any()


def test_oblique_plane_matches_nearest_source_voxel_and_never_blends_ids():
    volume = make_volume(reflected=True)
    rng = np.random.default_rng(8245)
    data = rng.choice(np.array([0, 3, 5], dtype=np.uint16), size=volume.data.shape)
    result = make_result(volume, data)
    right = np.array([1., 2., 0.]) / np.sqrt(5)
    normal = np.array([2., -1., 3.]) / np.sqrt(14)
    up = np.cross(normal, right)
    center = volume.index_to_world([6.2, 5.1, 4.3])
    rows, columns = 41, 45
    xx, yy = np.meshgrid(np.linspace(-6., 6., columns), np.linspace(5., -5., rows))
    world = center + xx[..., None] * right + yy[..., None] * up
    index = volume.world_to_index(world)
    inside = np.all((index >= -1e-7) & (index <= np.asarray(volume.dimensions) - 1 + 1e-7), axis=-1)
    nearest = np.floor(index + .5).astype(int)
    expected = np.zeros((rows, columns), dtype=np.uint16)
    expected[inside] = data[nearest[..., 2][inside], nearest[..., 1][inside], nearest[..., 0][inside]]
    sampled = result.sample_plane(volume, (center, right, up, (12., 10.)), (rows, columns))
    assert np.array_equal(sampled, expected)
    assert set(np.unique(sampled)) == {0, 3, 5}


def test_single_pixel_plane_uses_center_and_padding_labels_cannot_be_loaded():
    volume = make_volume()
    data = np.zeros(volume.data.shape, dtype=np.uint16)
    data[4, 5, 6] = 3
    result = make_result(volume, data)
    sampled = result.sample_plane(volume, (*axial(volume), (100., 100.)), (1, 1))
    assert sampled.tolist() == [[3]]
    volume.valid_mask = np.ones(volume.data.shape, dtype=bool)
    volume.valid_mask[4, 5, 6] = False
    with pytest.raises(ValueError, match="padding voxels"):
        result.validate_source(volume, fingerprint(volume))


def test_outside_or_nonfinite_brush_is_bounded_and_invalid_gesture_does_not_apply():
    volume = make_volume()
    result = make_result(volume)
    assert result.paint(volume, [1e100, -1e100, 1e100], *axial(volume)[1:], 1., 3) == 0
    for center, right, up, radius in (([np.nan, 0, 0], [1, 0, 0], [0, 1, 0], 1),
                                     ([0, 0, 0], [1, 0, 0], [2, 0, 0], 1),
                                     ([0, 0, 0], [1, 0, 0], [0, 1, 0], np.inf)):
        with pytest.raises(ValueError):
            result.paint(volume, center, right, up, radius, 3)
    assert not result.data.any() and result.revision == 0


def test_centroid_is_in_patient_space_and_snapshot_is_independent_read_only():
    volume = make_volume()
    result = make_result(volume)
    assert result.label_centroid(volume, 5) is None
    result.paint(volume, *axial(volume), 1., 3)
    assert np.allclose(result.label_centroid(volume, 3), axial(volume)[0])
    snapshot = result.snapshot()
    assert not snapshot.flags.writeable
    result.paint(volume, *axial(volume), 1., 3, erase=True)
    assert snapshot.any() and not result.data.any()


def test_file_roundtrip_preserves_exact_grid_provenance_and_explicit_review_rules(tmp_path):
    volume = make_volume(real=True)
    result = make_result(volume)
    result.paint(volume, *axial(volume), 1.1, 3)
    result.set_review(3, "accepted")
    result.set_review(5, "rejected")
    path = tmp_path / "labels.asterseg.npz"
    export_segmentation(path, result, volume, fingerprint(volume))
    with np.load(path, allow_pickle=False) as stored:
        assert set(stored.files) == {"data", "metadata"}
        assert stored["metadata"].dtype == np.uint8
        metadata = json.loads(stored["metadata"].tobytes())
        assert metadata["format"] == FORMAT
        assert not {"patient_name", "path", "filename"} & metadata["source"].keys()
    imported = read_segmentation(path, volume, fingerprint(volume))
    assert all(label["review"] == "pending" for label in imported.labels.values())
    reopened = read_segmentation(path, volume, fingerprint(volume), imported=False)
    assert reopened.labels[3]["review"] == "accepted" and reopened.labels[5]["review"] == "rejected"
    assert np.array_equal(reopened.data, result.data)
    assert reopened.provenance == result.provenance
    assert reopened.revision == result.revision
    assert not reopened.can_undo


def test_wrong_source_import_and_failed_export_do_not_mutate_live_result_or_file(tmp_path):
    volume = make_volume(real=True)
    result = make_result(volume)
    path = tmp_path / "labels.npz"
    export_segmentation(path, result, volume, fingerprint(volume))
    original_file = path.read_bytes()
    wrong = fingerprint(volume)
    wrong["source_group"] = "other"
    with pytest.raises(ValueError): read_segmentation(path, volume, wrong)
    result.data[0, 0, 0] = 999  # Simulate an unsupported external mutation.
    with pytest.raises(ValueError): export_segmentation(path, result, volume, fingerprint(volume))
    assert path.read_bytes() == original_file
    assert result.data[0, 0, 0] == 999


def test_object_arrays_and_extra_archive_entries_are_rejected(tmp_path):
    volume = make_volume()
    path = tmp_path / "unsafe.npz"
    np.savez_compressed(path, data=np.empty(volume.data.shape, dtype=object), metadata=np.zeros(1, dtype=np.uint8))
    with pytest.raises(ValueError, match="plain contiguous"):
        read_segmentation(path, volume, fingerprint(volume))
    np.savez_compressed(path, data=np.zeros(volume.data.shape, dtype=np.uint8), metadata=np.zeros(1, dtype=np.uint8), code=np.zeros(1))
    with pytest.raises(ValueError, match="only data and metadata"):
        read_segmentation(path, volume, fingerprint(volume))


def test_compressed_huge_array_header_is_rejected_before_allocation(tmp_path):
    volume = make_volume()
    bad = io.BytesIO()
    np.lib.format.write_array_header_1_0(bad, {"descr": "<u2", "fortran_order": False, "shape": (2**30, 2**30, 2**30)})
    good_meta = io.BytesIO()
    np.save(good_meta, np.frombuffer(b"{}", dtype=np.uint8), allow_pickle=False)
    path = tmp_path / "huge.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data.npy", bad.getvalue())
        archive.writestr("metadata.npy", good_meta.getvalue())
    with pytest.raises(ValueError, match="dimensions and integer type"):
        read_segmentation(path, volume, fingerprint(volume))


def test_malformed_json_and_duplicate_label_ids_are_atomic_import_failures(tmp_path):
    volume = make_volume()
    result = make_result(volume)
    path = tmp_path / "labels.npz"
    export_segmentation(path, result, volume, fingerprint(volume))
    with np.load(path, allow_pickle=False) as payload:
        data = payload["data"].copy()
        metadata = json.loads(payload["metadata"].tobytes())
    metadata["labels"].append(dict(metadata["labels"][0]))
    np.savez_compressed(path, data=data, metadata=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8))
    with pytest.raises(ValueError, match="unique"):
        read_segmentation(path, volume, fingerprint(volume))
    np.savez_compressed(path, data=data, metadata=np.frombuffer(b'{"format":1,"format":2}', dtype=np.uint8))
    with pytest.raises(ValueError, match="duplicate JSON"):
        read_segmentation(path, volume, fingerprint(volume))
    assert not result.data.any() and result.revision == 0


@pytest.mark.parametrize("field,value", [("model_sha256", "bad"), ("name", ""), ("version", 1), ("unexpected", "x")])
def test_provenance_requires_bounded_declared_model_identity(field, value):
    volume = make_volume()
    provenance = {"name": "Model", "version": "1", field: value}
    with pytest.raises(ValueError):
        create_result(volume, fingerprint(volume), np.zeros(volume.data.shape, dtype=np.uint8), {3: {"name": "Teeth", "kind": "tooth"}}, provenance)
