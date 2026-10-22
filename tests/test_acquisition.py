import numpy as np
import pytest

from asterview.acquisition import acquisition_text, acquisition_details, format_time_12h


def test_source_parameters_format_without_inferring_missing_fields():
    text = acquisition_text({"study_date": "20261020", "study_time": "141231.125", "kvp": "90", "tube_current_ma": "7", "source_slice_thickness_mm": "0.4"})
    assert "10/20/26 2:12 PM" in text
    assert "90 kV" in text and "7 mA" in text
    assert "0.4 mm source thickness" in text
    assert "mAs" not in text and "Acquired" not in text
    assert acquisition_text({}) == ""


def test_acquisition_time_is_not_paired_with_a_different_study_date():
    text = acquisition_text({"study_date": "20261019", "study_time": "235959", "acquisition_time": "000001"})
    assert text == "12 AM"


def test_original_datetime_timezone_is_retained():
    meta = {"acquisition_datetime": "20261020141231+0300"}
    assert acquisition_text(meta) == "10/20/26 2:12 PM UTC+03:00"
    assert "20261020141231+0300" in acquisition_details(meta)


@pytest.mark.parametrize("source,expected", [
    ("000001", "12 AM"), ("120000", "12 PM"), ("100000", "10 AM"),
    ("220000", "10 PM"), ("141231.125", "2:12 PM"), ("10", "10 AM"),
    ("235959", "11:59 PM"), ("236000", ""), ("240000", ""), ("oops", ""),
])
def test_12_hour_source_time(source, expected):
    assert format_time_12h(source) == expected


def test_compact_shared_strip_and_raw_precision_tooltip():
    meta = {"study_date": "20261020", "study_time": "141231.125", "kvp": "90",
            "tube_current_ma": "7", "source_slice_thickness_mm": "0.4", "exposure_time_ms": "12287"}
    text = acquisition_text(meta, (575, 575, 689), (.4, .4, .4), np.eye(3))
    assert text == "10/20/26 2:12 PM - 23.0 × 23.0 × 27.6 cm (575 × 575 × 689) - 0.400 mm - 90 kV 7 mA 12.287 s - AXIAL"
    assert "141231.125" in acquisition_details(meta)
    assert acquisition_text({}) == ""


def test_datetime_preferred_once():
    meta = {"acquisition_datetime": "20261020141231+0300",
            "acquisition_date": "20261020", "acquisition_time": "141231"}
    assert acquisition_text(meta) == "10/20/26 2:12 PM UTC+03:00"


def test_uncalibrated_geometry_has_no_false_physical_extent():
    text = acquisition_text({"geometry_calibrated": False}, (10, 20, 30), (1, 1, 1))
    assert "uncalibrated" in text and "cm" not in text and " mm" not in text
