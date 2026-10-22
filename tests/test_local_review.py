import threading

import numpy as np
import pytest

from asterview.local_review import analyze_local, LocalReviewCancelled
from asterview.model import Volume
from test_abnormality_candidates import phantom, source


def test_inventory_visits_all_source_slices_and_excludes_padding_without_mutation(monkeypatch):
    import asterview.local_review as module
    monkeypatch.setattr(module, "CHUNK_VOXELS", 7)
    pixels = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    valid = np.ones(pixels.shape, bool)
    valid[1] = False
    valid[3, 1:3, 2:5] = False
    volume = Volume(pixels, (.5, .6, .7), [1, 2, 3], np.eye(3),
                    {"synthetic": True}, valid_mask=valid)
    before = volume.data.copy()
    progress = []
    result = analyze_local(volume, source(volume), progress=lambda percent, message: progress.append(percent))
    record = result["inventory"]
    assert record["slices_visited"] == 4
    assert record["valid_voxels"] == np.count_nonzero(valid)
    assert record["padding_voxels"] == valid.size - np.count_nonzero(valid)
    assert record["slices"][1]["minimum"] is None
    assert record["minimum"] == float(volume.data[valid].min())
    assert record["maximum"] == float(volume.data[valid].max())
    assert record["nonfinite_valid_voxels"] == 0
    assert result["batch"] is None and result["status"] == "inventory_only"
    assert result["vision_model_enabled"] is False
    assert "not a complete anatomical examination" in result["summary"]
    assert progress[-1] == 100 and progress == sorted(progress)
    np.testing.assert_array_equal(volume.data, before)


def test_real_bounded_candidates_are_pending_and_source_linked():
    volume, segmentation, expected = phantom()
    original = volume.data.copy(), segmentation.data.copy()
    result = analyze_local(volume, source(volume), segmentation)
    batch = result["batch"]
    assert result["status"] == "candidates_ready" and batch.findings
    assert all(item["review"] == "pending" and item["differential"] == "" for item in batch.findings)
    assert min(np.linalg.norm(np.asarray(item["location_lps_mm"]) - expected) for item in batch.findings) < 1
    assert batch.source == source(volume)
    assert batch.mask_revision == segmentation.revision
    np.testing.assert_array_equal(volume.data, original[0])
    np.testing.assert_array_equal(segmentation.data, original[1])


def test_rejected_mandible_cannot_enable_candidate_search():
    volume, segmentation, _ = phantom()
    segmentation.set_review(2, "rejected")
    result = analyze_local(volume, source(volume), segmentation)
    assert result["status"] == "inventory_only"
    assert result["batch"] is None


def test_nonfinite_source_is_reported_and_never_searched():
    volume, segmentation, _ = phantom()
    volume.data[0, 0, 0] = np.nan
    result = analyze_local(volume, source(volume), segmentation)
    assert result["status"] == "invalid_source"
    assert result["inventory"]["nonfinite_valid_voxels"] == 1
    assert result["batch"] is None


def test_cancellation_and_changed_masks_never_return_partial_review():
    volume, segmentation, _ = phantom()
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(LocalReviewCancelled):
        analyze_local(volume, source(volume), segmentation, cancel=cancel)
    def change(percent, message):
        if percent > 0 and segmentation.labels[2]["review"] != "accepted":
            segmentation.set_review(2, "accepted")
    with pytest.raises(RuntimeError, match="Anatomy changed"):
        analyze_local(volume, source(volume), segmentation, progress=change)


def test_wrong_source_anatomy_is_inventory_only():
    volume, segmentation, _ = phantom()
    segmentation.source["origin"][0] += 10
    result = analyze_local(volume, source(volume), segmentation)
    assert result["batch"] is None
    assert result["anatomy"]["segmentation_status"].startswith("unavailable")


@pytest.mark.parametrize("dtype", [object, np.complex64, np.bool_])
def test_non_numeric_or_non_scalar_source_dtype_is_rejected(dtype):
    volume = Volume(np.zeros((3, 4, 5), np.float32), (1, 1, 1),
                    [0, 0, 0], np.eye(3), {"synthetic": True})
    volume.data = volume.data.astype(dtype)
    with pytest.raises(ValueError, match="real numeric"):
        analyze_local(volume, source(volume))


def test_padding_shape_and_dtype_must_match_source():
    volume = Volume(np.zeros((3, 4, 5), np.float32), (1, 1, 1),
                    [0, 0, 0], np.eye(3), {"synthetic": True})
    for mask in (np.ones((3, 4, 4), bool), np.ones((3, 4, 5), np.uint8)):
        volume.valid_mask = mask
        with pytest.raises(ValueError, match="padding mask"):
            analyze_local(volume, source(volume))


def test_mid_inventory_cancellation_returns_no_partial_result():
    volume = Volume(np.zeros((8, 4, 5), np.float32), (1, 1, 1),
                    [0, 0, 0], np.eye(3), {"synthetic": True})
    canceled = threading.Event()
    def progress(percent, message):
        if percent >= 8:
            canceled.set()
    with pytest.raises(LocalReviewCancelled):
        analyze_local(volume, source(volume), progress=progress, cancel=canceled)
