"""Physical/display sample budgets and retained source detail, without a GPU."""
import numpy as np
import pytest

from asterview.fast_reslice import FastReslicer
from asterview.model import Volume
from asterview.sampling_quality import (
    bounded_sample_shape, display_sample_shape, panorama_sampling_plan,
)


def test_default_sampling_matches_display_instead_of_fixed_512():
    rows, columns = display_sample_shape((180., 160.), (900., 800.))
    assert (rows, columns) == (801, 901)
    assert columns > 768


def test_dpi_and_zoom_request_new_source_samples():
    ordinary = display_sample_shape((160., 100.), (512., 320.))
    retina = display_sample_shape((160., 100.), (512., 320.), device_ratio=2)
    zoomed = display_sample_shape((160., 100.), (512., 320.), zoom=2)
    assert ordinary == (321, 513)
    assert retina == zoomed == (641, 1025)


@pytest.mark.parametrize("size,viewport", [
    ((180., 180.), (3840., 2160.)),
    ((180., 5.), (3840., 2160.)),
    ((2., 180.), (100., 600.)),
])
def test_extreme_zoom_respects_explicit_memory_and_edge_caps(size, viewport):
    shape = display_sample_shape(size, viewport, zoom=12, device_ratio=2,
                                 maximum=2048, pixel_budget=2_000_000)
    assert min(shape) >= 32 and max(shape) <= 2048
    assert np.prod(shape) <= 2_000_000


def test_tiny_grid_cells_do_not_allocate_fullsize_mpr_images():
    shape = display_sample_shape((24., 50.), (100., 120.))
    assert max(shape) <= 121
    assert np.prod(shape) < 10_000


def test_panorama_uses_source_pitch_and_grows_for_display_detail():
    arch = [(-40., 15., 0.), (0., -30., 0.), (40., 15., 0.)]
    size, native = panorama_sampling_plan(arch, 90., (.2, .3, .5))
    assert native[0] >= 451
    assert native[1] >= size[0] / .2
    _, display = panorama_sampling_plan(arch, 90., (.2, .3, .5),
                                       display_shape=(900, 1400))
    assert display[0] >= 900 and display[1] >= 1400
    assert max(display) <= 2048


def test_display_sampling_retains_voxel_detail_lost_in_512_intermediate():
    # A deterministic acquired signal; the fine output reconstructs existing
    # samples exactly rather than creating detail with a sharpening filter.
    x = np.arange(1024)
    signal = (800 * np.sin(2 * np.pi * x / 4)).astype(np.float32)
    data = np.broadcast_to(signal, (1, 3, 1024)).copy()
    volume = Volume(data, (.1, .1, .1), (0., 0., 0.), np.eye(3))
    before = volume.data.copy()
    shape = display_sample_shape((102.3, .2), (1023., 100.))
    engine = FastReslicer(volume)
    args = (volume.center, (1, 0, 0), (0, 1, 0), (102.3, .2))
    native = engine.sample_plane(*args, shape=shape)[shape[0] // 2]
    old = engine.sample_plane(*args, shape=(32, 512))[16]
    displayed_old = np.interp(x, np.linspace(0, 1023, 512), old)
    np.testing.assert_allclose(native, signal, atol=.02)
    assert np.mean(np.abs(displayed_old - signal)) > 200
    np.testing.assert_array_equal(volume.data, before)


@pytest.mark.parametrize("kwargs", [
    {"size_mm": (0, 1), "pixels_per_mm": 1},
    {"size_mm": (1, 1), "pixels_per_mm": float("nan")},
    {"size_mm": (1, 1), "pixels_per_mm": 1, "pixel_budget": 12},
])
def test_invalid_sampling_requests_are_rejected(kwargs):
    with pytest.raises(ValueError):
        bounded_sample_shape(**kwargs)
