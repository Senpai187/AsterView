"""Actual MainWindow wiring with synthetic evidence and no exposed GPU surface."""
import time

import numpy as np


def test_report_topbutton_bookmarks_and_source_change():
    from PySide6.QtWidgets import QApplication
    from asterview.app import MainWindow
    from asterview.model import Volume
    from asterview.reporting import make_finding, new_findings

    app = QApplication.instance() or QApplication([])
    window = MainWindow()  # Intentionally unshown; no native rendering surface.
    errors = []
    window.reporting_panel._error = lambda error: errors.append(str(error))

    def settle():
        end = time.monotonic() + 12
        while time.monotonic() < end:
            app.processEvents()
            if window.slice_job is None and window.slice_pending is None and not window.render_timer.isActive():
                return
            time.sleep(.002)
        raise AssertionError("Synthetic report source slices did not settle")

    volume = Volume(np.arange(9*10*11, dtype=np.float32).reshape((9, 10, 11)),
                    (.7, .8, .9), [-4., -5., 6.], np.eye(3),
                    metadata={"synthetic": True, "modality": "CT"})
    try:
        window.set_volume(volume)
        settle()
        panel = window.reporting_panel
        document = new_findings(volume, window.volume_fingerprint())
        locations = [volume.index_to_world([2., 3., 4.]), volume.index_to_world([7., 6., 5.])]
        document["findings"] = [
            make_finding("Synthetic reviewed observation A.", location_lps_mm=locations[0], review="accepted"),
            make_finding("Synthetic reviewed observation B.", location_lps_mm=locations[1], review="accepted"),
        ]
        document["clinical_indication"] = "Synthetic workflow test."
        panel.document = document
        panel.indication.setPlainText(document["clinical_indication"])
        panel.refresh(-1)
        window.inspector.hide()
        window.report_button.click()
        app.processEvents()
        dialog = panel._report_dialog
        assert dialog is not None and dialog.export_button.isEnabled()
        assert not dialog.isWindow()
        assert not window.inspector.isHidden()
        assert window.inspector_tabs.currentWidget().isAncestorOf(panel)
        assert dialog.links.count() == 2
        dialog.locate(dialog.links.item(1))
        settle()
        np.testing.assert_allclose(window.crosshair, locations[1])
        assert panel.selected()["id"] == document["findings"][1]["id"]
        assert window.tabs.currentIndex() == 0
        assert not window.volume_view._initialized
        replacement = Volume(volume.data.copy(), volume.spacing, volume.origin + [1., 0., 0.],
                             volume.direction, metadata={"synthetic": True, "modality": "CT"})
        window.set_volume(replacement)
        settle()
        assert panel._report_dialog is None and panel.document is None
        assert not errors, errors
    finally:
        window.close()
        app.processEvents()
