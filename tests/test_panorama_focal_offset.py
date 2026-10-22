"""Focal-trough motion samples a different physical surface, not other columns."""
import numpy as np
import pytest

from asterview.model import Volume
from asterview.reformat import curved_planar, path_frame


@pytest.fixture
def ramp():
    z, y, x = np.mgrid[:81, :81, :81].astype(np.float32) - 40
    return Volume(2 * x + 3 * y + 5 * z, (1., 1., 1.), (-40., -40., -40.), np.eye(3))


def test_zero_offset_preserves_existing_pixels_path_and_pitch_exactly(ramp):
    points = [(-14., 10., 0.), (0., -8., 0.), (14., 10., 0.)]
    original = curved_planar(ramp, points, (0, 0, 1), height_mm=12, samples=65, rows=31)
    zero = curved_planar(ramp, points, (0, 0, 1), height_mm=12, samples=65, rows=31, focal_offset_mm=0)
    np.testing.assert_array_equal(zero.image, original.image)
    np.testing.assert_array_equal(zero.path, original.path)
    assert zero.column_pitch_mm == original.column_pitch_mm
    assert zero.row_pitch_mm == original.row_pitch_mm


@pytest.mark.parametrize("offset", [-4., 3.])
def test_straight_arch_offset_follows_local_normal_and_changes_sampled_anatomy(ramp, offset):
    points = [(0., -12., 0.), (0., 0., 0.), (0., 12., 0.)]
    source = ramp.data.copy()
    plain = curved_planar(ramp, points, (0, 0, 1), height_mm=10, thickness_mm=0, samples=65, rows=31)
    moved = curved_planar(ramp, points, (0, 0, 1), height_mm=10, thickness_mm=0, samples=65, rows=31,
                          focal_offset_mm=offset)
    # up Z cross tangent Y gives -X; source gradient along X is exactly2.
    np.testing.assert_allclose(moved.path, plain.path + (-offset, 0., 0.), atol=1e-10)
    np.testing.assert_allclose(moved.image - plain.image, -2 * offset, atol=2e-5)
    np.testing.assert_array_equal(ramp.data, source)


def test_curved_offset_pixels_and_section_frames_describe_displayed_trough(ramp):
    points = [(-14., 10., 0.), (0., -8., 0.), (14., 10., 0.)]
    source = ramp.data.copy()
    result = curved_planar(ramp, points, (0, 0, 1), height_mm=12, thickness_mm=2,
                           samples=129, rows=41, focal_offset_mm=2.5)
    vertical = np.linspace(6, -6, 41)
    expected = result.path[:, 0][None, :] * 2 + result.path[:, 1][None, :] * 3 + vertical[:, None] * 5
    np.testing.assert_allclose(result.image, expected, atol=2e-5)
    steps = np.linalg.norm(np.diff(result.path, axis=0), axis=1)
    assert np.std(steps) / np.mean(steps) < .002
    assert result.column_pitch_mm == pytest.approx(steps.mean())
    for column in (0, 64, 128):
        center, right, up = path_frame(result.path, column, (0, 0, 1))
        np.testing.assert_array_equal(center, result.path[column])
        assert np.dot(right, up) == pytest.approx(0, abs=1e-12)
    np.testing.assert_array_equal(ramp.data, source)


@pytest.mark.parametrize("offset", [np.nan, np.inf, -np.inf])
def test_nonfinite_offset_rejected(ramp, offset):
    with pytest.raises(ValueError, match="focal offset"):
        curved_planar(ramp, [(0, -12, 0), (0, 0, 0), (0, 12, 0)],
                      (0, 0, 1), samples=17, rows=17, focal_offset_mm=offset)
