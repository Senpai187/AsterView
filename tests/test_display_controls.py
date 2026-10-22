"""Visible display controls preserve exact intensity values and source identity."""
import pytest
from PySide6.QtWidgets import QApplication, QDoubleSpinBox

from asterview.display_slider import DisplaySlider
from asterview.presentation import patient_display_name, startup_joke, RADIOGRAPH_JOKES


@pytest.mark.parametrize("raw, expected", [
    ("Example^Synthetic", "Synthetic Example"),
    ("Synthetic^Example", "Example Synthetic"),
    ("Family^Given^^Dr", "Dr Given Family"),
    ("Family^Given^Middle^Dr^Jr", "Dr Given Middle Family Jr"),
    ("Alpha^Beta=Other^Script", "Beta Alpha"),
    ("=اسم^عائلة", "عائلة اسم"),
    ("  Two   spaces  ", "Two spaces"),
    (None, ""),
])
def test_patient_display_name(raw, expected):
    assert patient_display_name(raw) == expected


def test_joke_changes_between_launches():
    for previous in RADIOGRAPH_JOKES:
        assert startup_joke(previous) in set(RADIOGRAPH_JOKES) - {previous}


@pytest.fixture
def qt_app():
    return QApplication.instance() or QApplication([])


def test_brightness_increases_as_level_decreases(qt_app):
    control = QDoubleSpinBox()
    control.setRange(-10000, 10000)
    control.setValue(500)
    widget = DisplaySlider(control, "Brightness", inverted=True)
    widget.set_limits(-1000, 3000)
    widget.slider.setValue(2000)
    assert control.value() == -1000
    widget.slider.setValue(0)
    assert control.value() == 3000
    widget.close()


def test_contrast_is_logarithmic_and_exact_values_survive_slider_clamping(qt_app):
    control = QDoubleSpinBox()
    control.setRange(1, 100000)
    widget = DisplaySlider(control, "Contrast", logarithmic=True, inverted=True)
    widget.set_limits(1, 10000)
    widget.slider.setValue(1000)
    assert control.value() == pytest.approx(100)
    control.setValue(25000)
    assert widget.slider.value() == 0
    assert control.value() == 25000
    widget.close()
