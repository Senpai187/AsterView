"""Upright orbit geometry independent of Qt and GPU state."""
import numpy as np
import pytest

from asterview.camera_navigation import upright_orbit


def test_full_horizontal_orbit_retains_height_radius_and_returns_home():
    focus, position = np.array([2., 3., 4.]), np.array([12., -17., 11.])
    for x in range(0, 1601, 100):
        moved = upright_orbit(position, focus, (0, 0), (x, 0), (800, 600))
        assert moved[2] == pytest.approx(position[2])
        assert np.linalg.norm(moved - focus) == pytest.approx(np.linalg.norm(position - focus))
    np.testing.assert_allclose(upright_orbit(position, focus, (0, 0), (800, 0), (800, 600)), position)


@pytest.mark.parametrize("vertical", [-100000, 100000])
def test_vertical_drag_stops_before_poles_without_flipping(vertical):
    moved = upright_orbit([0, -10, 0], [0, 0, 0], (0, 0), (0, vertical), (800, 600))
    assert np.isfinite(moved).all() and moved[1] < 0
    assert abs(moved[2]) < 10
    assert np.linalg.norm(moved) == pytest.approx(10)


def test_drag_to_original_pointer_restores_exact_anchor():
    position = np.array([3., -7., 9.])
    np.testing.assert_array_equal(upright_orbit(position, [0, 0, 0], (10, 20), (10, 20), (800, 600)), position)
