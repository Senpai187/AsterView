import pytest
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QApplication

from asterview.thickness_control import ThicknessControl


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_source_pitch_is_the_minimum_and_presets_are_physical_multiples(app):
    control = ThicknessControl()
    control.configure_for_spacing((.4, .4, .4))
    assert control.combo.itemText(0) == "0.400 mm"
    assert control.value() == .4
    assert [control.combo.itemData(i) for i in range(13)] == pytest.approx([.4 * i for i in range(1, 14)])
    assert control.combo.itemText(2) == "1.200 mm"
    control.combo.setCurrentIndex(2)
    assert control.value() == 1.2
    control.combo.setCurrentIndex(0)
    assert control.value() == .4
    control.setRange(0, 30)
    control.setValue(0)
    control.setSpecialValueText("Native")
    assert control.value() == .4 and control.minimum() == .4
    assert all(control.combo.itemData(i) >= .4 for i in range(control.combo.count()))
    control.close()


def test_custom_millimeters_survive_edit_and_invalid_text_never_changes_sampling(app):
    control = ThicknessControl()
    control.configure_for_spacing((.25, .25, .5))
    emitted = []
    control.valueChanged.connect(emitted.append)
    control.combo.setEditText("2.375 mm")
    control.combo.lineEdit().editingFinished.emit()
    assert control.value() == 2.375 and emitted == [2.375]
    for text in ("-2", "nan", "42 mm", "unknown", ""):
        control.combo.setEditText(text)
        control.combo.lineEdit().editingFinished.emit()
        assert control.value() == 2.375
        assert control.combo.currentText() == "2.375 mm"
    assert emitted == [2.375]
    control.close()


def test_paired_header_and_inspector_controls_sync_without_signal_loop(app):
    header, inspector = ThicknessControl(), ThicknessControl()
    for control in (header, inspector):
        control.configure_for_spacing(.4)
    header.valueChanged.connect(inspector.setValue)
    inspector.valueChanged.connect(header.setValue)
    seen = []
    header.valueChanged.connect(seen.append)
    inspector.combo.setCurrentIndex(7)
    assert header.value() == inspector.value() == 3.2
    assert header.combo.currentText() == "3.200 mm"
    assert seen == [3.2]
    with QSignalBlocker(inspector):
        inspector.setValue(1.2)
    assert inspector.value() == 1.2 and header.value() == 3.2
    header.setValue(inspector.value())
    assert seen == [3.2, 1.2]
    header.close()
    inspector.close()


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"),
                                   (), (.4, float("nan"), .4), (.4, -1, .4)])
def test_invalid_pitch_is_rejected(app, value):
    control = ThicknessControl()
    with pytest.raises(ValueError):
        control.configure_for_spacing(value)
    control.close()
