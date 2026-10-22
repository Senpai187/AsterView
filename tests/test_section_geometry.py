"""Physical spacing and boundary behavior independent of panoramic pixel pitch."""
import numpy as np
import pytest

from asterview.section_geometry import ArchSectionGeometry


def test_irregular_columns_interpolate_exact_physical_section_positions():
    geometry = ArchSectionGeometry([[0, 0, 0], [1, 0, 0], [4, 0, 0], [10, 0, 0]], [0, 0, 2])
    anchor = geometry.nearest_distance([2.3, .5, 4])
    assert anchor == pytest.approx(2.3)
    for distance in np.arange(.15, 10, .25):
        center, right, up = geometry.frame(distance)
        np.testing.assert_allclose(center, [distance, 0, 0])
        np.testing.assert_allclose(right, [0, 1, 0])
        np.testing.assert_allclose(up, [0, 0, 1])


def test_terminal_sections_are_never_duplicated_for_outside_arch_distances():
    geometry = ArchSectionGeometry([[0, 0, 0], [5, 0, 0], [10, 0, 0]], [0, 0, 1])
    assert geometry.frame(-.1) is None
    assert geometry.frame(10.1) is None
    np.testing.assert_array_equal(geometry.frame(0)[0], [0, 0, 0])
    np.testing.assert_array_equal(geometry.frame(10)[0], [10, 0, 0])
    assert geometry.offset_limits(1, 5, 2) == (3., 5.)
    assert geometry.offset_limits(1, 12, 2) == (4., 4.)


def test_curved_oblique_sections_stay_orthonormal_and_do_not_mutate_path():
    from asterview.reformat import rotation_matrix
    rotation = rotation_matrix(15, 31, -23)
    theta = np.linspace(0, np.pi, 100)
    path = np.c_[15 * np.cos(theta), 15 * np.sin(theta), np.zeros(100)] @ rotation.T
    before = path.copy()
    geometry = ArchSectionGeometry(path, rotation[:, 2])
    for distance in np.linspace(0, geometry.length, 50):
        center, right, up = geometry.frame(distance)
        assert np.linalg.norm(right) == pytest.approx(1)
        assert np.linalg.norm(up) == pytest.approx(1)
        assert right @ up == pytest.approx(0, abs=1e-12)
        assert (center @ up) == pytest.approx(0, abs=1e-12)
    np.testing.assert_array_equal(path, before)


@pytest.mark.parametrize("path,up", [
    ([[0, 0, 0]], [0, 0, 1]),
    ([[0, 0, 0], [0, 0, 0]], [0, 0, 1]),
    ([[0, 0, 0], [0, 0, 1]], [0, 0, 1]),
    ([[0, 0, 0], [1, 0, 0]], [0, 0, 0]),
    ([[0, 0, 0], [np.nan, 0, 0]], [0, 0, 1]),
])
def test_invalid_arch_geometry_is_rejected(path, up):
    with pytest.raises(ValueError):
        ArchSectionGeometry(path, up)
