"""Native/reference geometry agreement and an opt-in 512-cube benchmark.

Run ``python tests/test_fast_reslice.py --benchmark`` from the project root
with the project on PYTHONPATH, or use ``python -m tests.test_fast_reslice``
when tests is importable. Ordinary pytest runs do not allocate a 512-cube.
"""

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import sys
import unittest
from unittest.mock import patch

import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy

from asterview.fast_reslice import FastReslicer
from asterview.model import Volume
from asterview.reformat import (
    ReconstructionCancelled, base_frame, rotation_matrix, sample_plane,
)


def ramp(shape=(37, 41, 47), spacing=(.8, 1.3, 2.1),
         direction=None, origin=(-30., 20., -40.)):
    direction = rotation_matrix(12, -9, 20) if direction is None else direction
    gradient = np.array((.7, -1.1, 2.3))
    coefficients = (np.asarray(spacing) * (direction.T @ gradient)).astype(np.float32)
    z = np.arange(shape[0], dtype=np.float32)[:, None, None]
    y = np.arange(shape[1], dtype=np.float32)[None, :, None]
    x = np.arange(shape[2], dtype=np.float32)[None, None, :]
    data = (x * coefficients[0] + y * coefficients[1] + z * coefficients[2]
            + np.float32(np.asarray(origin) @ gradient + 9))
    return Volume(data, spacing, origin, direction), gradient


class NativeGeometryTests(unittest.TestCase):
    def test_sixteen_anisotropic_oblique_and_edge_crossing_cases(self):
        random = np.random.default_rng(1703)
        for case in range(16):
            direction = rotation_matrix(*random.uniform(-35, 35, 3))
            if case % 4 == 0:
                direction[:, 2] *= -1  # An orthonormal reflected source basis.
            volume, _ = ramp(spacing=tuple(random.uniform(.4, 2.5, 3)),
                             direction=direction,
                             origin=tuple(random.uniform(-150, 150, 3)))
            right, up = base_frame(("axial", "coronal", "sagittal")[case % 3],
                                   rotation_matrix(*random.uniform(-30, 30, 3)))
            center = volume.center + random.uniform(-1, 1, 3)
            # Alternating compact/interior and wide/edge-crossing fields.
            size = (12., 14.) if case % 2 else (83., 77.)
            shape = (67 + case % 3, 73 + case % 4)
            with self.subTest(case=case):
                reslicer = FastReslicer(volume)
                actual = reslicer.sample_plane(center, right, up, size, shape)
                expected = sample_plane(volume, center, right, up, size, shape)
                self.assertEqual(reslicer.backend, "vtk")
                self.assertEqual(actual.dtype, np.float32)
                self.assertEqual(actual.shape, shape)
                np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=2e-6)
                # Outside samples are exactly the reference background value.
                np.testing.assert_array_equal(actual[expected == volume.data_min],
                                              expected[expected == volume.data_min])

    def test_analytic_pixel_centers_positive_up_and_gram_schmidt(self):
        volume, gradient = ramp()
        right, up = base_frame("coronal", rotation_matrix(7, -13, 11))
        center = volume.center
        actual = FastReslicer(volume).sample_plane(center, right * 4,
                                                   up * 3 + right, (12, 10), (65, 71))
        horizontal = np.linspace(-6, 6, 71)
        vertical = np.linspace(5, -5, 65)
        world = (center + right * horizontal[None, :, None]
                 + up * vertical[:, None, None])
        np.testing.assert_allclose(actual, world @ gradient + 9, atol=2e-4, rtol=2e-6)

    def test_single_pixel_dimensions_and_single_slice_sources(self):
        for source_shape in ((1, 17, 19), (13, 17, 19)):
            volume, _ = ramp(source_shape, (1, 1, 1), np.eye(3), (0, 0, 0))
            reslicer = FastReslicer(volume)
            for shape in ((1, 1), (1, 31), (29, 1), (29, 31)):
                with self.subTest(source=source_shape, output=shape):
                    expected = sample_plane(volume, volume.center, (1, 0, 0),
                                            (0, 1, 0), (10, 8), shape)
                    actual = reslicer.sample_plane(volume.center, (1, 0, 0),
                                                   (0, 1, 0), (10, 8), shape)
                    np.testing.assert_allclose(actual, expected, atol=2e-4)

    def test_strict_padding_outside_voxel_centers_not_half_voxel_border(self):
        data = np.arange(11 * 13 * 17, dtype=np.float32).reshape((11, 13, 17)) + 10
        volume = Volume(data, (1, 1, 1), np.zeros(3), np.eye(3))
        reslicer = FastReslicer(volume)
        for x in (-.49, -1e-8, 0., 1e-8, 16., 16. + 1e-8, 16.49):
            with self.subTest(x=x):
                args = ((x, 6, 5), (0, 1, 0), (0, 0, 1), (4, 4), (33, 35))
                np.testing.assert_allclose(reslicer.sample_plane(*args),
                                           sample_plane(volume, *args), atol=5e-4)

    def test_textured_data_agrees_with_linear_reference(self):
        random = np.random.default_rng(700)
        data = random.normal(300, 500, (37, 41, 47)).astype(np.float32)
        volume = Volume(data, (.7, 1.2, 1.7), (11, -15, 31), rotation_matrix(13, -17, 9))
        right, up = base_frame("axial", rotation_matrix(11, 7, -13))
        args = (volume.center, right, up, (27, 31), (97, 103))
        actual = FastReslicer(volume).sample_plane(*args)
        expected = sample_plane(volume, *args)
        # Native interpolation uses finite fractional precision; this bound is
        # intensity interpolation error, not an allowed geometric shift.
        np.testing.assert_allclose(actual, expected, atol=.04, rtol=2e-5)

    def test_source_borrowing_outputs_owned_and_no_source_mutation(self):
        volume, _ = ramp()
        original = volume.data.copy()
        reslicer = FastReslicer(volume)
        args = (volume.center, (1, 0, 0), (0, 1, 0), (12, 12), (65, 71))
        first = reslicer.sample_plane(*args)
        snapshot = first.copy()
        pipeline = reslicer._reslice
        self.assertTrue(np.shares_memory(vtk_to_numpy(reslicer._scalars), volume.data))
        self.assertTrue(first.flags.owndata)
        second = reslicer.sample_plane(volume.center + (1, 1, 1), *args[1:])
        self.assertIs(reslicer._reslice, pipeline)
        self.assertFalse(np.shares_memory(first, second))
        np.testing.assert_array_equal(first, snapshot)
        np.testing.assert_array_equal(volume.data, original)

    def test_invalid_geometry_is_rejected(self):
        volume, _ = ramp()
        reslicer = FastReslicer(volume)
        changes = [dict(center=(np.nan, 0, 0)), dict(right=(0, 0, 0)),
                   dict(up=(1, 0, 0)), dict(size_mm=(0, 5)), dict(shape=(0, 10)),
                   dict(shape=(3.5, 5)), dict(slab_mm=-1), dict(mode="median")]
        for change in changes:
            kwargs = dict(center=volume.center, right=(1, 0, 0), up=(0, 1, 0))
            kwargs.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                reslicer.sample_plane(**kwargs)


class FallbackAndCancellationTests(unittest.TestCase):
    def test_masked_padding_and_native_slab_modes_match_reference(self):
        ordinary, _ = ramp()
        mask = np.ones(ordinary.data.shape, dtype=bool)
        mask[:, :, :20] = False
        masked = Volume(ordinary.data.copy(), ordinary.spacing, ordinary.origin,
                        ordinary.direction, valid_mask=mask)
        for volume in (ordinary, masked):
            reslicer = FastReslicer(volume)
            for slab in (0, 4):
                for mode in ("mean", "min", "max"):
                    with self.subTest(masked=volume.valid_mask is not None,
                                      slab=slab, mode=mode):
                        args = (volume.center, (1, 0, 0), (0, 1, 0),
                                (43, 35), (67, 71), slab, mode)
                        actual = reslicer.sample_plane(*args)
                        expected = sample_plane(volume, *args)
                        if volume.valid_mask is not None:
                            np.testing.assert_array_equal(actual, expected)
                        else:
                            np.testing.assert_allclose(actual, expected, atol=3e-4)

    def test_native_slabs_preserve_thickness_extrema_and_owned_output(self):
        volume, gradient = ramp()
        reslicer = FastReslicer(volume)
        right, up = base_frame("coronal", rotation_matrix(7, -13, 11))
        snapshots = []
        for mode in ("mean", "min", "max"):
            args = (volume.center, right, up, (12., 14.), (89, 97), 6., mode)
            actual = reslicer.sample_plane(*args)
            expected = sample_plane(volume, *args)
            np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=2e-6)
            self.assertTrue(actual.flags.owndata)
            snapshots.append((actual, actual.copy()))
        # Distinct extrema demonstrate that a thick slab was not silently
        # replaced by its middle plane during navigation.
        self.assertGreater(float(np.mean(snapshots[2][0] - snapshots[1][0])), 1.)
        for actual, saved in snapshots:
            np.testing.assert_array_equal(actual, saved)
        self.assertTrue(np.shares_memory(vtk_to_numpy(reslicer._scalars), volume.data))

    def test_textured_oblique_slabs_crossing_edges_match_reference(self):
        rng = np.random.default_rng(516)
        for spacing, thickness in (((.5, .9, 1.6), .8), ((1., 2., .8), 4.7)):
            volume = Volume(rng.normal(700, 900, (29, 37, 41)).astype(np.float32),
                            spacing, (11., -17., 23.), rotation_matrix(17, -12, 8))
            engine = FastReslicer(volume)
            right, up = base_frame("sagittal", rotation_matrix(12, 5, -19))
            for mode in ("mean", "min", "max"):
                with self.subTest(spacing=spacing, thickness=thickness, mode=mode):
                    args = (volume.center, right, up, (80., 71.), (117, 131), thickness, mode)
                    actual = engine.sample_plane(*args)
                    expected = sample_plane(volume, *args)
                    np.testing.assert_allclose(actual, expected, atol=.07, rtol=2e-5)

    def test_noncontiguous_source_uses_reference_without_copying(self):
        volume, _ = ramp()
        volume.data = volume.data[:, ::2, :]
        reslicer = FastReslicer(volume)
        self.assertEqual(reslicer.backend, "reference")
        args = (volume.center, (1, 0, 0), (0, 1, 0), (10, 10), (17, 19))
        np.testing.assert_array_equal(reslicer.sample_plane(*args), sample_plane(volume, *args))
        self.assertIs(reslicer._data, volume.data)

    def test_already_canceled_never_creates_native_objects(self):
        volume, _ = ramp()
        reslicer = FastReslicer(volume)
        with self.assertRaises(ReconstructionCancelled):
            reslicer.sample_plane(volume.center, (1, 0, 0), (0, 1, 0), cancel=lambda: True)
        self.assertIsNone(reslicer._reslice)

    def test_cancel_after_native_update_discards_output(self):
        volume, _ = ramp()
        reslicer = FastReslicer(volume)
        checks = 0

        def cancel():
            nonlocal checks
            checks += 1
            return checks >= 3

        with self.assertRaises(ReconstructionCancelled):
            reslicer.sample_plane(volume.center, (1, 0, 0), (0, 1, 0),
                                   (8, 8), (65, 71), cancel=cancel)
        self.assertIsNotNone(reslicer._reslice)

    def test_padding_fallback_retains_cancellation(self):
        volume, _ = ramp()
        volume.valid_mask = np.ones(volume.data.shape, dtype=bool)
        volume.padding_value = volume.data_min - 1
        reslicer = FastReslicer(volume)
        calls = 0

        def cancel():
            nonlocal calls
            calls += 1
            return calls >= 5

        with patch("asterview.fast_reslice.reference_sample_plane", wraps=sample_plane) as ref:
            with self.assertRaises(ReconstructionCancelled):
                reslicer.sample_plane(volume.center, (1, 0, 0), (0, 1, 0),
                                       (8, 8), (65, 71), slab_mm=10, cancel=cancel)
            ref.assert_called_once()

    def test_lazy_native_creation_and_first_sampling_thread_confinement(self):
        volume, _ = ramp()
        reslicer = FastReslicer(volume)
        self.assertIsNone(reslicer._reslice)
        args = (volume.center, (1, 0, 0), (0, 1, 0), (8, 8), (65, 71))
        with ThreadPoolExecutor(max_workers=1) as worker:
            actual = worker.submit(reslicer.sample_plane, *args).result()
        np.testing.assert_allclose(actual, sample_plane(volume, *args), atol=3e-4)
        with self.assertRaisesRegex(RuntimeError, "first sampling thread"):
            reslicer.sample_plane(*args)


def benchmark():
    """Print warmed medians without a hardware-specific pytest time assertion."""
    volume, _ = ramp((512, 512, 512))
    reslicer = FastReslicer(volume)
    right, up = base_frame("axial", rotation_matrix(7, 13, -11))
    cases = [(volume.center + (i * .013, -i * .011, i * .007), right, up,
              (300, 300), (512, 512)) for i in range(18)]
    timings = {}
    for name, sampler in (("reference", lambda *a: sample_plane(volume, *a)),
                          ("native", reslicer.sample_plane)):
        elapsed = []
        for args in cases:
            start = perf_counter()
            sampler(*args)
            elapsed.append((perf_counter() - start) * 1000)
        timings[name] = float(np.median(elapsed[3:]))
        print(f"{name}: {timings[name]:.3f} ms median; "
              f"p95 {np.percentile(elapsed[3:], 95):.3f} ms")
    expected = sample_plane(volume, *cases[0])
    actual = reslicer.sample_plane(*cases[0])
    print(f"speedup: {timings['reference'] / timings['native']:.2f}x; "
          f"maximum intensity difference: {np.max(np.abs(actual - expected)):.8g}")


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        benchmark()
    else:
        unittest.main()
