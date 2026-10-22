import numpy as np
import pytest

from asterview.model import Volume
from asterview.panoramic_range import acquisition_bounds, range_presets
from asterview.reformat import curved_planar, rotation_matrix


def test_asymmetric_range_samples_physical_upper_lower_and_roundtrips_rows():
    z, y, x = np.mgrid[:31, :21, :21]
    volume = Volume(z.astype(np.float32), (1, 1, 1), np.zeros(3), np.eye(3))
    path = [[5, 8, 15], [10, 5, 15], [15, 8, 15]]
    result = curved_planar(volume, path, [0, 0, 1], thickness_mm=0, samples=17,
                           rows=11, vertical_bounds_mm=(-3, 7))
    assert result.vertical_limits_mm == (-3, 7)
    np.testing.assert_allclose(result.image[:, 8], np.linspace(22, 12, 11), atol=1e-5)
    assert result.row_pitch_mm == 1
    rows = np.arange(11)
    np.testing.assert_allclose(result.row_at_offset(result.offset_at_row(rows)), rows)
    np.testing.assert_allclose(result.path[:, 2], 15)


def test_range_presets_follow_current_acquired_geometry_without_jaw_claims():
    rotation = rotation_matrix(17, 28, -12)
    volume = Volume(np.zeros((31, 21, 21), np.float32), (.4, .5, .8), [4, 8, -2], rotation)
    origin = volume.index_to_world([10, 10, 12])
    points = [origin + rotation[:, 0] * delta for delta in (-3, 0, 3)]
    assert acquisition_bounds(volume, points, rotation[:, 2]) == pytest.approx((-9.6, 14.4))
    presets = range_presets(volume, points, rotation[:, 2])
    assert set(presets) == {"Full scan", "Above arch", "Below arch"}
    assert presets["Above arch"] == pytest.approx((0, 14.4))
    assert presets["Below arch"] == pytest.approx((-9.6, 0))


def test_single_panorama_row_represents_the_midpoint_of_asymmetric_bounds():
    z, _, _ = np.mgrid[:31, :21, :21]
    volume = Volume(z.astype(np.float32), (1, 1, 1), np.zeros(3), np.eye(3))
    result = curved_planar(volume, [[5, 8, 15], [10, 5, 15], [15, 8, 15]],
                           [0, 0, 1], thickness_mm=0, rows=1, samples=9,
                           vertical_bounds_mm=(-3, 7))
    assert result.offset_at_row(0) == 2
    np.testing.assert_allclose(result.image, 17)


@pytest.mark.parametrize("bounds", [(2, 2), (4, -4), (0, np.inf), (np.nan, 2), (0, 1, 2)])
def test_invalid_data_range_rejects_before_reconstruction(bounds):
    volume = Volume(np.zeros((9, 9, 9), np.float32), (1, 1, 1), np.zeros(3), np.eye(3))
    with pytest.raises(ValueError):
        curved_planar(volume, [[1, 3, 4], [4, 1, 4], [7, 3, 4]], [0, 0, 1],
                      vertical_bounds_mm=bounds)
