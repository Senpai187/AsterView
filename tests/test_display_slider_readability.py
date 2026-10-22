"""Actual styled text geometry; no VTK render window or GPU is created."""
import pytest
from PySide6.QtWidgets import QApplication, QWidget, QDoubleSpinBox, QFormLayout

from asterview.display_slider import DisplaySlider
from asterview.ui_shell import STYLE


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("name,low,high,value,logarithmic", [
    ("Brightness", -1e6, 1e6, 500., False),
    ("Contrast", 1, 2e6, 2000., True),
])
def test_exact_slider_value_has_readable_edit_geometry(qt_app, name, low, high, value, logarithmic):
    host = QWidget()
    host.setStyleSheet(STYLE)
    host.resize(350, 100)
    layout = QFormLayout(host)
    control = QDoubleSpinBox()
    control.setRange(low, high)
    control.setDecimals(1)
    control.setValue(value)
    slider = DisplaySlider(control, name, logarithmic=logarithmic)
    layout.addRow(name, slider)
    host.show()
    qt_app.processEvents()
    try:
        editor = control.lineEdit()
        assert editor.isVisible()
        assert editor.text() == f"{value:.1f}"
        assert editor.width() >= editor.fontMetrics().horizontalAdvance(editor.text()) + 8
        assert editor.height() >= editor.fontMetrics().height()
        control.setValue(value + 125)
        assert editor.text() == f"{value + 125:.1f}"
    finally:
        host.close()
        host.deleteLater()
        qt_app.processEvents()
