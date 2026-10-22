"""Report integration at the source, pixel-capture, and async boundaries."""
import time

import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from asterview.app import MainWindow
from asterview.report_images import make_attachment


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def settle(app, window):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        app.processEvents()
        if (window.job is None and window.slice_job is None and window.slice_pending is None
                and not window.render_timer.isActive() and not window.pano_refresh_timer.isActive()):
            return
        time.sleep(.003)
    raise AssertionError("Workstation did not settle")


@pytest.fixture
def window(app):
    w = MainWindow()
    w.load_demo()
    settle(app, w)
    yield w
    settle(app, w)
    w.close()
    app.processEvents()


def test_clean_mpr_capture_keeps_exact_source_and_current_zoom(app, window):
    view = window.mpr_views["Axial"]
    view.resize(540, 410)
    view.zoom = 1.7
    view.pan = QPointF(15, -8)
    expected_source = window.volume_fingerprint()
    captured = window.capture_report_views(["Axial", "Coronal", "Sagittal"])
    assert [item["title"] for item in captured] == ["Axial", "Coronal", "Sagittal"]
    assert captured[0]["view_state"]["zoom"] == 1.7
    assert captured[0]["view_state"]["pan"] == [15, -8]
    for item in captured:
        assert item["source"] == expected_source
        image = QImage.fromData(item["png"], "PNG")
        assert not image.isNull()
        make_attachment(item["png"], item["title"], expected_source, view_state=item["view_state"])
    assert view.zoom == 1.7 and view.pan == QPointF(15, -8)


def test_capture_refuses_unavailable_panorama(window):
    with pytest.raises(ValueError, match="panoramic"):
        window.capture_report_views(["Panoramic"])


def test_report_preparation_has_inventory_without_diagnostic_claims(app, window):
    wizard = window.reporting_panel.open_report()
    assert wizard is not None
    assert window.analyze_for_report()
    settle(app, window)
    panel = window.reporting_panel
    assert not panel.analysis_status["busy"]
    assert "analysis_summary" in panel.document
    assert "no diagnosis" in panel.document["analysis_summary"]["summary"].lower()
    assert panel.document["findings"] == []
    assert window._report_analysis_completed


def test_top_restore_is_a_fresh_case_but_keeps_original_voxels(app, window):
    volume = window.volume
    before = volume.data.copy()
    window.add_implant()
    window.reporting_panel.open_report()
    window.reporting_panel.report_wizard.fields["referrer"].setText("Transient synthetic note")
    window.restore_loaded_scan()
    settle(app, window)
    assert window.volume is volume
    np.testing.assert_array_equal(volume.data, before)
    assert not window.implants and window.segmentation is None
    assert window.reporting_panel.report_wizard is None
    assert window.header_thickness.value() == pytest.approx(min(volume.spacing))
    np.testing.assert_allclose(window.crosshair, volume.center)
