"""Geometry tests use analytic physical-coordinate phantoms, not screenshots."""

import unittest
from unittest.mock import patch

import numpy as np

from asterview.model import Volume
from asterview.reformat import (
    ReconstructionCancelled,
    base_frame,
    curved_planar,
    path_frame,
    resample_path,
    rotation_matrix,
    sample_plane,
)


def ramp_volume(shape=(45, 55, 60), spacing=(0.8, 1.3, 2.1), oblique=True):
    direction = rotation_matrix(12, -9, 20) if oblique else np.eye(3)
    origin = np.array((-30.0, -20.0, -40.0))
    z, y, x = np.indices(shape, dtype=float)
    index = np.stack((x, y, z), axis=-1)
    world = (index * spacing) @ direction.T + origin
    gradient = np.array((0.7, -1.1, 2.3))
    data = world @ gradient + 9
    return Volume(data, spacing, origin, direction), gradient


class PlaneTests(unittest.TestCase):
    def test_anisotropic_oblique_volume_and_plane_match_physical_ramp(self):
        volume, gradient = ramp_volume()
        right, up = base_frame("axial", rotation_matrix(7, 13, -11))
        center = volume.center + (1.1, -0.8, 0.3)
        actual = sample_plane(volume, center, right, up, (18, 16), (19, 23))
        x = np.linspace(-9, 9, 23)
        y = np.linspace(8, -8, 19)
        world = center + x[None, :, None] * right + y[:, None, None] * up
        expected = world @ gradient + 9
        np.testing.assert_allclose(actual, expected, atol=2e-5)
        self.assertEqual(actual.dtype, np.float32)

    def test_radiological_base_frames_and_top_row_orientation(self):
        axial_right, axial_up = base_frame("axial")
        coronal_right, coronal_up = base_frame("coronal")
        sagittal_right, sagittal_up = base_frame("sagittal")
        np.testing.assert_array_equal(axial_right, (1, 0, 0))
        np.testing.assert_array_equal(axial_up, (0, -1, 0))
        np.testing.assert_array_equal(coronal_right, (1, 0, 0))
        np.testing.assert_array_equal(coronal_up, (0, 0, 1))
        np.testing.assert_array_equal(sagittal_right, (0, 1, 0))
        np.testing.assert_array_equal(sagittal_up, (0, 0, 1))
        volume, gradient = ramp_volume(oblique=False)
        actual = sample_plane(volume, volume.center, axial_right, axial_up, (6, 8), (3, 3))
        self.assertAlmostEqual(float(actual[0, 1] - actual[-1, 1]),
                               float(8 * axial_up @ gradient), places=4)

    def test_mean_maximum_and_minimum_slab_in_physical_mm(self):
        volume, gradient = ramp_volume()
        right, up = base_frame("coronal", rotation_matrix(4, -6, 11))
        normal = np.cross(right, up)
        center = volume.center
        center_value = float(center @ gradient + 9)
        change = abs(float(normal @ gradient)) * 3
        for mode, expected in (("mean", center_value), ("max", center_value + change),
                               ("min", center_value - change)):
            with self.subTest(mode=mode):
                actual = sample_plane(volume, center, right, up, (4, 4), (1, 1),
                                      slab_mm=6, mode=mode)
                self.assertAlmostEqual(float(actual[0, 0]), expected, places=4)

    def test_outside_uses_minimum_and_single_pixel_uses_center(self):
        volume, gradient = ramp_volume()
        right, up = base_frame("axial")
        outside = sample_plane(volume, volume.center + 1000, right, up, shape=(3, 5))
        np.testing.assert_array_equal(outside, volume.data_min)
        center = sample_plane(volume, volume.center, right, up, (150, 150), (1, 1))
        self.assertAlmostEqual(float(center[0, 0]), float(volume.center @ gradient + 9), places=4)

    def test_tiled_sampling_equals_single_tile(self):
        volume, _ = ramp_volume()
        right, up = base_frame("axial", rotation_matrix(15, 6, 3))
        expected = sample_plane(volume, volume.center, right, up, (15, 14), (13, 17), 4, "max")
        with patch("asterview.reformat._MAX_TILE_PIXELS", 36):
            actual = sample_plane(volume, volume.center, right, up, (15, 14), (13, 17), 4, "max")
        np.testing.assert_array_equal(actual, expected)

    def test_gram_schmidt_handles_nonunit_frame(self):
        volume, _ = ramp_volume()
        expected = sample_plane(volume, volume.center, (1, 0, 0), (0, 1, 0), (8, 8), (7, 9))
        actual = sample_plane(volume, volume.center, (4, 0, 0), (1, 6, 0), (8, 8), (7, 9))
        np.testing.assert_array_equal(actual, expected)

    def test_invalid_geometry_rejected(self):
        volume, _ = ramp_volume()
        cases = [
            dict(center=(np.nan, 0, 0)),
            dict(right=(0, 0, 0)),
            dict(right=(1, 0, 0), up=(1, 0, 0)),
            dict(size_mm=(0, 5)),
            dict(shape=(0, 10)),
            dict(shape=(3.5, 5)),
            dict(slab_mm=-1),
            dict(mode="median"),
        ]
        for change in cases:
            kwargs = dict(center=volume.center, right=(1, 0, 0), up=(0, 1, 0))
            kwargs.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                sample_plane(volume, **kwargs)
        with self.assertRaises(ValueError):
            rotation_matrix(0, np.inf, 0)
        with self.assertRaises(ValueError):
            base_frame("axial", np.ones((3, 3)))

    def test_rotations_are_proper_orthonormal_lps_rotations(self):
        rotate = rotation_matrix(11, 22, 33)
        np.testing.assert_allclose(rotate.T @ rotate, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(float(np.linalg.det(rotate)), 1)
        np.testing.assert_allclose(rotation_matrix(rz=90) @ (1, 0, 0), (0, 1, 0), atol=1e-14)


class PanoramicTests(unittest.TestCase):
    def test_arc_length_resampling_preserves_endpoints_and_uniform_steps(self):
        angles = np.linspace(-1.2, 1.2, 9)
        points = np.column_stack((20 * np.sin(angles), 20 * np.cos(angles), np.full(9, 4)))
        path = resample_path(points, 301)
        np.testing.assert_allclose(path[[0, -1]], points[[0, -1]], atol=1e-12)
        step = np.linalg.norm(np.diff(path, axis=0), axis=1)
        self.assertLess(float(np.std(step) / np.mean(step)), 0.001)
        np.testing.assert_allclose(path[:, 2], 4, atol=1e-12)

    def test_cross_sections_follow_tangent_and_up(self):
        path = resample_path(((-5, -5, 0), (0, 0, 0), (5, -5, 0)), 101)
        center, right, up = path_frame(path, 50, (0, 0, 2))
        tangent = path[51] - path[49]
        tangent /= np.linalg.norm(tangent)
        np.testing.assert_array_equal(center, path[50])
        np.testing.assert_allclose(np.cross(right, up), tangent, atol=1e-12)
        self.assertAlmostEqual(float(right @ up), 0)
        self.assertAlmostEqual(float(np.linalg.norm(right)), 1)

    def test_curved_planar_matches_physical_ramp_and_measurement_pitch(self):
        volume, gradient = ramp_volume(oblique=False)
        control = volume.center + np.array(((-12, -8, 0), (0, 0, 0), (12, -8, 0)))
        result = curved_planar(volume, control, (0, 0, 1), height_mm=12,
                               thickness_mm=4, samples=65, rows=17)
        vertical = np.linspace(6, -6, 17)
        expected = result.path[None, :, :] @ gradient + 9 + vertical[:, None] * gradient[2]
        np.testing.assert_allclose(result.image, expected, atol=2e-5)
        self.assertAlmostEqual(result.row_pitch_mm, 0.75)
        self.assertAlmostEqual(result.column_pitch_mm,
                               float(np.linalg.norm(np.diff(result.path, axis=0), axis=1).mean()))
        image, path = result
        self.assertIs(image, result.image)
        self.assertIs(path, result.path)

    def test_panorama_maximum_slab_uses_local_arch_normals(self):
        volume, gradient = ramp_volume(oblique=False)
        control = volume.center + np.array(((-10, -6, 0), (0, 0, 0), (10, -6, 0)))
        result = curved_planar(volume, control, (0, 0, 1), height_mm=10,
                               thickness_mm=6, samples=41, rows=1, mode="max")
        tangent = np.gradient(result.path, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        normals = np.cross((0, 0, 1), tangent)
        expected = result.path @ gradient + 9 + np.abs(normals @ gradient) * 3
        np.testing.assert_allclose(result.image[0], expected, atol=2e-5)

    def test_duplicate_and_invalid_arch_points(self):
        valid = ((0, 0, 0), (2, 2, 0), (4, 0, 0))
        with_duplicates = ((0, 0, 0), (0, 0, 0), (2, 2, 0), (4, 0, 0))
        np.testing.assert_allclose(resample_path(valid), resample_path(with_duplicates))
        for points in (((0, 0, 0), (0, 0, 0), (1, 0, 0)),
                       ((0, 0, 0), (1, np.nan, 0), (2, 0, 0))):
            with self.assertRaises(ValueError):
                resample_path(points)
        with self.assertRaises(ValueError):
            path_frame(np.asarray(valid), 99, (0, 0, 1))
        with self.assertRaises(ValueError):
            path_frame(((0, 0, 0), (0, 0, 1), (0, 0, 2)), 1, (0, 0, 1))


def padded_volume():
    data = np.broadcast_to(np.array((0, 20, 40, 60), dtype=np.float32), (4, 4, 4)).copy()
    # A valid distant low voxel makes invalid-sample exclusion distinguishable
    # from filling missing slab samples with the display background.
    data[0, 0, 3] = 5
    valid = np.ones(data.shape, dtype=bool)
    valid[:, :, 0] = False
    return Volume(data, (1, 1, 1), np.zeros(3), np.eye(3),
                  valid_mask=valid, padding_value=-100)


class PaddingTests(unittest.TestCase):
    def test_interpolation_renormalizes_valid_neighbors(self):
        volume = padded_volume()
        for x, expected in ((0.25, 20), (0.5, 20), (0.75, 20), (1.5, 30)):
            with self.subTest(x=x):
                image = sample_plane(volume, (x, 1.5, 1.5), (0, 1, 0), (0, 0, 1),
                                     size_mm=(1, 1), shape=(1, 1))
                self.assertAlmostEqual(float(image[0, 0]), expected, places=5)

    def test_slabs_weight_valid_interpolation_and_ignore_invalid_extrema(self):
        volume = padded_volume()
        # 3mm slab at x=.5 samples x=(-1,-.25,.5,1.25,2). The three
        # valid samples have weights (.5,1,1), values (20,25,40).
        for mode, expected in (("mean", 30), ("max", 40), ("min", 20)):
            with self.subTest(mode=mode):
                image = sample_plane(volume, (.5, 1.5, 1.5), (0, 1, 0), (0, 0, 1),
                                     size_mm=(1, 1), shape=(1, 1), slab_mm=3, mode=mode)
                self.assertAlmostEqual(float(image[0, 0]), expected, places=5)

    def test_all_invalid_or_outside_samples_use_valid_minimum(self):
        volume = padded_volume()
        for x in (-3, -0.01, 0, 3.01, 9):
            for mode in ("mean", "max", "min"):
                with self.subTest(x=x, mode=mode):
                    image = sample_plane(volume, (x, 1.5, 1.5), (0, 1, 0), (0, 0, 1),
                                         size_mm=(1, 1), shape=(1, 1), mode=mode)
                    self.assertEqual(float(image[0, 0]), volume.data_min)
        image = sample_plane(volume, (-4, 1.5, 1.5), (0, 1, 0), (0, 0, 1),
                             size_mm=(1, 1), shape=(1, 1), slab_mm=2)
        self.assertEqual(float(image[0, 0]), volume.data_min)

    def test_valid_border_is_preserved_and_no_source_copy_or_mutation(self):
        volume = padded_volume()
        data, mask = volume.data, volume.valid_mask
        snapshot = data.copy()
        image = sample_plane(volume, (3, 1.5, 1.5), (0, 1, 0), (0, 0, 1),
                             size_mm=(1, 1), shape=(1, 1))
        self.assertEqual(float(image[0, 0]), 60)
        self.assertIs(volume.data, data)
        self.assertIs(volume.valid_mask, mask)
        np.testing.assert_array_equal(data, snapshot)

    def test_masked_curved_panorama_uses_same_weighted_reconstruction(self):
        volume = padded_volume()
        points = ((.5, 1, 1), (.5, 2, 1), (.5, 3, 1))
        plain = curved_planar(volume, points, (0, 0, 1), height_mm=1,
                              thickness_mm=0, rows=3, samples=9)
        np.testing.assert_allclose(plain.image, 20)
        # Local normal is -x; a 4mm slab has valid weights (1,1,.5)
        # and intensities (50,30,20), giving a weighted mean of 36.
        slab = curved_planar(volume, points, (0, 0, 1), height_mm=1,
                             thickness_mm=4, rows=3, samples=9)
        np.testing.assert_allclose(slab.image, 36)

    def test_masked_sampling_matches_unmasked_when_all_voxels_valid(self):
        plain, _ = ramp_volume()
        masked = Volume(plain.data.copy(), plain.spacing, plain.origin, plain.direction,
                        valid_mask=np.ones(plain.data.shape, dtype=bool))
        right, up = base_frame("axial", rotation_matrix(7, 3, -4))
        for mode in ("mean", "max", "min"):
            expected = sample_plane(plain, plain.center, right, up, (15, 15), (19, 23), 4, mode)
            actual = sample_plane(masked, masked.center, right, up, (15, 15), (19, 23), 4, mode)
            np.testing.assert_allclose(actual, expected, atol=2e-5)


class CancellationTests(unittest.TestCase):
    def test_ordinary_and_masked_reconstructions_cancel_between_slab_samples(self):
        from scipy.ndimage import map_coordinates

        ordinary, _ = ramp_volume()
        for volume in (ordinary, padded_volume()):
            for reconstruction in ("plane", "panorama"):
                with self.subTest(masked=volume.valid_mask is not None, kind=reconstruction):
                    checks = [0]

                    def cancel():
                        checks[0] += 1
                        return checks[0] >= 4

                    with patch("asterview.reformat.map_coordinates", wraps=map_coordinates) as sampling:
                        with self.assertRaises(ReconstructionCancelled):
                            if reconstruction == "plane":
                                sample_plane(volume, volume.center, (1, 0, 0), (0, 1, 0),
                                             (1, 1), (8, 9), slab_mm=10, cancel=cancel)
                            else:
                                points = volume.center + np.array(((0, -.4, 0), (.2, 0, 0), (0, .4, 0)))
                                curved_planar(volume, points, (0, 0, 1), height_mm=1,
                                              thickness_mm=10, rows=8, samples=9, cancel=cancel)
                    # At least one sample ran, but the remaining thick slab was
                    # never processed. A masked sample interpolates two arrays.
                    self.assertGreater(sampling.call_count, 0)
                    self.assertLessEqual(sampling.call_count, 2)

    def test_cancellation_between_tiles_without_a_slab(self):
        volume, _ = ramp_volume()
        checks = [0]

        def cancel():
            checks[0] += 1
            return checks[0] >= 4

        with patch("asterview.reformat._MAX_TILE_PIXELS", 9), self.assertRaises(ReconstructionCancelled):
            sample_plane(volume, volume.center, (1, 0, 0), (0, 1, 0), (1, 1), (8, 9), cancel=cancel)

    def test_already_canceled_request_never_samples(self):
        volume, _ = ramp_volume()
        with patch("asterview.reformat.map_coordinates") as sampling:
            with self.assertRaises(ReconstructionCancelled):
                sample_plane(volume, volume.center, (1, 0, 0), (0, 1, 0), cancel=lambda: True)
            sampling.assert_not_called()


if __name__ == "__main__":
    unittest.main()
