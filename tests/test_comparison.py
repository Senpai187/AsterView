import time
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMainWindow

from asterview.comparison_panel import comparison_image, MODES
from asterview.comparison_controls import ComparisonControls, volume_fingerprint
from asterview.model import Volume


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def volume():
    z, y, x = np.indices((30, 32, 34), dtype=np.float32)
    return Volume(x * 10 + y * 5 + z * 2, (1, 1.1, 1.2), np.array([-17, -17, -18]), np.eye(3), {"synthetic": True})


class Workstation(QMainWindow):
    def __init__(self):
        super().__init__()
        self.volume = volume()
        self.crosshair = self.volume.center.copy()
        self.job = None

    def volume_fingerprint(self):
        return volume_fingerprint(self.volume)

    def start_job(self, function, callback):
        return False


@pytest.mark.parametrize("mode", MODES)
def test_windowed_comparison_masks_invalid_overlap_and_preserves_inputs(mode):
    fixed = np.arange(48, dtype=np.float32).reshape(6, 8)
    moving = 2 * fixed
    before = fixed.copy(), moving.copy()
    valid = np.ones(fixed.shape, bool)
    valid[0, :3] = False
    output = comparison_image(fixed, moving, valid, mode=mode, fixed_window=(24, 48), moving_window=(48, 96), tile=2)
    assert output.shape == fixed.shape and output.dtype == np.float32
    assert np.all(output[~valid] == 0)
    if mode == "Visual difference":
        assert np.all(output[valid] == 0)  # windowed display, not a raw HU subtraction
    np.testing.assert_array_equal(fixed, before[0])
    np.testing.assert_array_equal(moving, before[1])


@pytest.mark.parametrize("mode", MODES)
def test_comparison_display_preserves_exact_windowed_formula_and_validation(mode):
    rng = np.random.default_rng(21)
    fixed = rng.uniform(-200, 800, (9, 12)).astype(np.float32)
    moving = rng.uniform(-500, 500, fixed.shape).astype(np.float32)
    valid = rng.random(fixed.shape) > .2
    fixed[~valid] = np.nan
    moving[~valid] = np.inf
    first = np.clip((fixed - 150) / 700 + .5, 0, 1)
    second = np.clip((moving - 20) / 600 + .5, 0, 1)
    yy, xx = np.indices(fixed.shape)
    expected = {"Registered moving": second, "Blend": .65 * first + .35 * second,
                "Visual difference": abs(first - second),
                "Checkerboard": np.where((xx // 3 + yy // 3) % 2 == 0, second, first),
                "Wipe": np.where(xx >= round(.35 * fixed.shape[1]), second, first)}[mode]
    output = comparison_image(fixed, moving, valid, mode=mode, alpha=.35,
                              fixed_window=(150, 700), moving_window=(20, 600), tile=3)
    np.testing.assert_array_equal(output, np.where(valid, expected * 255, 0).astype(np.float32))
    with pytest.raises(ValueError, match="finite"):
        comparison_image(fixed, moving, valid, mode=mode, fixed_window=(150, 0))
    fixed[np.argwhere(valid)[0][0], np.argwhere(valid)[0][1]] = np.nan
    with pytest.raises(ValueError, match="finite"):
        comparison_image(fixed, moving, valid, mode=mode)


def test_comparison_controls_manual_alignment_review_and_reset(qapp):
    workstation = Workstation()
    controls = ComparisonControls(workstation)
    moving = volume()
    moving.origin += [3, -4, 2]
    original = moving.data.copy()
    controls.set_moving(moving)
    assert controls.result.review == "pending"
    np.testing.assert_allclose(controls.result.transform[:3, 3], [-3, 4, -2])
    controls.review("accepted")
    assert controls.result.review == "accepted"
    controls.panel.manual[0].setValue(1.5)
    controls.apply_manual()
    assert controls.result.review == "pending"
    np.testing.assert_allclose(controls.result.transform[:3, 3], [-1.5, 4, -2])
    assert controls.panel.manual[0].value() == 0
    controls.reset_comparison()
    assert controls.moving is None and controls.result is None
    assert not controls.panel.review_button.isEnabled()
    np.testing.assert_array_equal(moving.data, original)
    controls._executor.shutdown(wait=True, cancel_futures=True)
    workstation.close()


def test_review_dialog_samples_linked_orthogonal_frames_and_discards_reset(qapp):
    workstation = Workstation()
    controls = ComparisonControls(workstation)
    controls.set_moving(volume())
    controls.open_comparison()
    deadline = time.monotonic() + 5
    while controls.dialog.cached_planes is None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(.01)
    assert controls.dialog.cached_planes is not None
    assert len(controls.dialog.cached_planes) == 3
    for frame, fixed, moving, valid in controls.dialog.cached_planes:
        np.testing.assert_allclose(fixed, moving)
        assert valid.any()
        np.testing.assert_allclose(frame[0], workstation.crosshair)
    controls.dialog.link_main.setChecked(False)
    picked = workstation.crosshair + [1, 2, 3]
    controls.dialog.views[0][1].picked.emit({"world": picked, "u": .5, "v": .5})
    np.testing.assert_allclose(controls.dialog.center, picked)
    previous = controls.dialog.center.copy()
    controls.dialog.scroll("Coronal", 1)
    np.testing.assert_allclose(controls.dialog.center, previous + [0, -1, 0])
    controls.reset_comparison()
    deadline = time.monotonic() + 2
    while controls._future is not None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(.005)
    assert controls.dialog is None and controls._future is None
    controls._executor.shutdown(wait=True, cancel_futures=True)
    workstation.close()
