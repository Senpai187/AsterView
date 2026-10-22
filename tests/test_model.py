import numpy as np
import pytest

from asterview.demo import make_demo
from asterview.model import Volume


def test_oblique_world_index_round_trip_and_center():
    angle = np.deg2rad(29)
    direction = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    volume = Volume(np.arange(24).reshape(2, 3, 4), (0.4, 0.7, 2.5),
                    np.array([11, -3, 17]), direction)
    points = np.array([[0, 0, 0], [3, 2, 1], [1.4, 0.7, 0.2]])
    assert np.allclose(volume.world_to_index(volume.index_to_world(points)), points)
    assert np.allclose(volume.center, volume.index_to_world([1.5, 1, 0.5]))
    assert volume.dimensions == (4, 3, 2)
    assert volume.data.dtype == np.float32 and volume.data.flags.c_contiguous
    assert (volume.data_min, volume.data_max) == (0, 23)


def test_bounds_use_all_oblique_corners():
    direction = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    volume = Volume(np.zeros((3, 4, 5)), (2, 3, 4), np.array([10, 20, 30]), direction)
    assert volume.bounds == (1, 10, 20, 28, 30, 38)


@pytest.mark.parametrize("data,spacing,direction", [
    (np.array([[[np.nan]]]), (1, 1, 1), np.eye(3)),
    (np.zeros((2, 2, 2)), (1, 0, 1), np.eye(3)),
    (np.zeros((2, 2, 2)), (1, 1, 1), np.ones((3, 3))),
])
def test_invalid_volume_cannot_enter_render_pipeline(data, spacing, direction):
    with pytest.raises(ValueError):
        Volume(data, spacing, np.zeros(3), direction)


def test_demo_is_finite_calibrated_and_explicitly_synthetic():
    volume = make_demo()
    assert volume.data.shape == (160, 192, 192)
    assert np.allclose(volume.center, 0)
    assert volume.metadata["synthetic"]
    assert not volume.is_hu
    assert "SYNTHETIC" in volume.warnings[0]
    assert volume.data_min == -1000 and volume.data_max == 2400


def test_padding_mask_excludes_background_from_range_and_normalizes_to_finite_sentinel():
    values = np.array([[[99999, 100], [np.nan, 200]]], dtype=np.float32)
    mask = np.array([[[False, True], [False, True]]])
    volume = Volume(values, (1, 1, 1), np.zeros(3), np.eye(3), valid_mask=mask)
    assert (volume.data_min, volume.data_max) == (100, 200)
    assert volume.padding_value < 100
    assert np.isfinite(volume.data).all()
    assert np.all(volume.data[~mask] == volume.padding_value)
    assert volume.valid_mask.dtype == np.bool_


def test_padding_only_volume_and_malformed_mask_fail():
    with pytest.raises(ValueError, match="no valid non-padding"):
        Volume(np.zeros((1, 2, 2)), (1, 1, 1), np.zeros(3), np.eye(3), valid_mask=np.zeros((1, 2, 2), dtype=bool))
    with pytest.raises(ValueError, match="boolean and match"):
        Volume(np.zeros((1, 2, 2)), (1, 1, 1), np.zeros(3), np.eye(3), valid_mask=np.ones((2, 2), dtype=bool))
