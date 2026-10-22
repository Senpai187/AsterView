"""CPR pixel-center navigation in physical LPS, without a Qt application or GPU."""
from types import SimpleNamespace

import numpy as np
import pytest

from asterview.app import MainWindow
from asterview.model import Volume
from asterview.reformat import PanoramicResult, rotation_matrix


class Slider:
    def __init__(self, callback):
        self.current = 2
        self.blocked = False
        self.callback = callback
        self.emissions = 0

    def blockSignals(self, blocked):
        before, self.blocked = self.blocked, blocked
        return before

    def setValue(self, value):
        if self.current != value:
            self.current = value
            if not self.blocked:
                self.emissions += 1
                self.callback(value)

    def value(self):
        return self.current


class NavigationHarness:
    panorama_picked = MainWindow.panorama_picked
    panorama_index_changed = MainWindow.panorama_index_changed
    locate_panorama_column = MainWindow.locate_panorama_column
    update_panorama_cursor = MainWindow.update_panorama_cursor

    def __init__(self):
        # Anisotropic, oblique and reflected source geometry. The corrected CPR
        # up axis is independent of all source voxel axes.
        direction = rotation_matrix(-9, 17, 23) @ np.diag([-1., 1., 1.])
        self.volume = Volume(np.zeros((61, 61, 61), np.float32), (.7, 1.1, 1.6),
                             np.array([-35., 15., -20.]), direction)
        corrected = rotation_matrix(13, -22, 11)
        self.pano_up = corrected[:, 2] * 3.7  # locator normalizes its direction
        self.unit_up = corrected[:, 2]
        offsets = np.array([[-6., 1., .2], [-3., -2., -.3], [0., -3., 0.],
                            [3., -2., .6], [6., 1., -.5]])
        path = self.volume.center + offsets @ corrected.T
        self.pano_result = PanoramicResult(np.zeros((7, 5), np.float32), path, 3., 1.7)
        self.crosshair = self.volume.center + corrected @ [1., 1., 2.]
        self.pano_index = Slider(self.panorama_index_changed)
        self.pano_view = SimpleNamespace(cursor_fraction=None, update=lambda: None)
        self.implant_pano_view = SimpleNamespace(cursor_fraction=None, update=lambda: None)
        self.references = self.renders = 0
        self.messages = []

    def statusBar(self):
        return SimpleNamespace(showMessage=self.messages.append)

    def refresh_references(self):
        self.references += 1

    def schedule_render(self):
        self.renders += 1

    def sync_panorama_position(self):
        pass


@pytest.fixture
def viewer():
    return NavigationHarness()


@pytest.mark.parametrize("column,row", [(0, 0), (2, 3), (4, 6), (1, 5), (3, 1)])
def test_exact_pixel_center_click_locates_oblique_lps_and_updates_all_references(viewer, column, row):
    result = viewer.pano_result
    rows, columns = result.image.shape
    viewer.panorama_picked({"fraction": ((column + .5) / columns, (row + .5) / rows)})
    expected = result.path[column] + viewer.unit_up * ((rows - 1) / 2 - row) * result.row_pitch_mm
    np.testing.assert_allclose(viewer.crosshair, expected, atol=1e-12)
    assert viewer.pano_index.value() == column
    assert viewer.references == viewer.renders == 1
    assert viewer.pano_index.emissions == 0  # click must not invoke locator twice
    assert not viewer.pano_index.blocked
    assert not viewer.messages
    expected_fraction = ((column + .5) / columns, (row + .5) / rows)
    np.testing.assert_allclose(viewer.pano_view.cursor_fraction, expected_fraction, atol=1e-12)
    np.testing.assert_allclose(viewer.implant_pano_view.cursor_fraction, expected_fraction, atol=1e-12)


@pytest.mark.parametrize("fraction,column,row", [((0., 0.), 0, 0), ((1., 1.), 4, 6)])
def test_image_edges_clamp_to_outer_pixel_centers(viewer, fraction, column, row):
    result = viewer.pano_result
    viewer.panorama_picked({"fraction": fraction})
    expected = result.path[column] + viewer.unit_up * (3 - row) * result.row_pitch_mm
    np.testing.assert_allclose(viewer.crosshair, expected, atol=1e-12)


def test_slider_preserves_physical_height_along_oblique_up_even_with_nonplanar_arch(viewer):
    original_height = float(np.dot(viewer.crosshair, viewer.unit_up))
    viewer.pano_index.setValue(4)
    base = viewer.pano_result.path[4]
    expected = base + viewer.unit_up * (original_height - np.dot(base, viewer.unit_up))
    np.testing.assert_allclose(viewer.crosshair, expected, atol=1e-12)
    assert np.dot(viewer.crosshair, viewer.unit_up) == pytest.approx(original_height, abs=1e-12)
    assert viewer.references == viewer.renders == viewer.pano_index.emissions == 1


def test_single_row_cpr_has_zero_vertical_offset(viewer):
    result = viewer.pano_result
    viewer.pano_result = PanoramicResult(np.zeros((1, 5), np.float32), result.path, 3., 0.)
    viewer.panorama_picked({"fraction": (.5, .83)})
    np.testing.assert_allclose(viewer.crosshair, result.path[2], atol=1e-12)
    np.testing.assert_allclose(viewer.pano_view.cursor_fraction, (.5, .5), atol=1e-12)


@pytest.mark.parametrize("row", [0, 2, 6])
def test_asymmetric_crop_click_and_cursor_use_completed_result_bounds(viewer, row):
    old = viewer.pano_result
    viewer.pano_result = PanoramicResult(old.image, old.path, old.column_pitch_mm,
                                         old.row_pitch_mm, (-2., 8.2))
    viewer.panorama_picked({"fraction": (.5, (row + .5) / 7)})
    expected = old.path[2] + viewer.unit_up * (8.2 - row * 1.7)
    np.testing.assert_allclose(viewer.crosshair, expected, atol=1e-12)
    np.testing.assert_allclose(viewer.pano_view.cursor_fraction, (.5, (row + .5) / 7), atol=1e-12)


@pytest.mark.parametrize("height", [-1000., 1000., np.nan, np.inf])
def test_outside_or_nonfinite_physical_candidate_is_rejected_atomically(viewer, height):
    previous = viewer.crosshair.copy()
    assert viewer.locate_panorama_column(1, height) is False
    np.testing.assert_array_equal(viewer.crosshair, previous)
    assert viewer.references == viewer.renders == 0
    if np.isfinite(height):
        assert viewer.messages


@pytest.mark.parametrize("fraction", [(), (.5,), (.5, .5, .5), (np.nan, .5), (.5, np.inf), None, ("bad", .5)])
def test_invalid_click_fraction_is_rejected_without_slider_or_view_changes(viewer, fraction):
    previous = viewer.crosshair.copy()
    viewer.panorama_picked({"fraction": fraction})
    np.testing.assert_array_equal(viewer.crosshair, previous)
    assert viewer.pano_index.value() == 2
    assert viewer.references == viewer.renders == viewer.pano_index.emissions == 0


@pytest.mark.parametrize("index", [np.nan, np.inf, None, "bad"])
def test_invalid_column_is_rejected_without_mutation(viewer, index):
    previous = viewer.crosshair.copy()
    assert viewer.locate_panorama_column(index) is False
    np.testing.assert_array_equal(viewer.crosshair, previous)
    assert viewer.references == viewer.renders == 0


@pytest.mark.parametrize("missing", ["volume", "result"])
def test_missing_source_or_result_is_an_inert_navigation_request(viewer, missing):
    previous = viewer.crosshair.copy()
    if missing == "volume":
        viewer.volume = None
    else:
        viewer.pano_result = None
    viewer.panorama_picked({"fraction": (.5, .5)})
    viewer.panorama_index_changed(4)
    assert viewer.locate_panorama_column(1, 0.) is False
    np.testing.assert_array_equal(viewer.crosshair, previous)
    assert viewer.references == viewer.renders == 0


def test_outside_acquired_click_never_moves_slider_to_unlocatable_column(viewer):
    result = viewer.pano_result
    path = result.path.copy()
    path[0] += [1000., 1000., 1000.]
    viewer.pano_result = PanoramicResult(result.image, path, result.column_pitch_mm, result.row_pitch_mm)
    previous = viewer.crosshair.copy()
    viewer.panorama_picked({"fraction": (.1, .5)})
    np.testing.assert_array_equal(viewer.crosshair, previous)
    assert viewer.pano_index.value() == 2
    assert viewer.references == viewer.renders == 0


def test_navigation_does_not_change_source_voxels_or_panoramic_result(viewer):
    volume_before = viewer.volume.data.copy()
    path_before = viewer.pano_result.path.copy()
    image_before = viewer.pano_result.image.copy()
    viewer.panorama_picked({"fraction": (.7, .5)})
    viewer.panorama_index_changed(0)
    np.testing.assert_array_equal(viewer.volume.data, volume_before)
    np.testing.assert_array_equal(viewer.pano_result.path, path_before)
    np.testing.assert_array_equal(viewer.pano_result.image, image_before)
