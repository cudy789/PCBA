"""Offscreen GUI interaction tests (pytest-qt)."""

import pytest
from PySide6.QtCore import Qt

from png_to_gerber.gui.main_window import MainWindow
from png_to_gerber.model import LayerType, Side

from conftest import BOARD_COLOR, COPPER_COLOR, MASK_COLOR, SILK_COLOR


@pytest.fixture
def window(qtbot, fixture_image_path):
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(fixture_image_path)
    return win


def add_colored_layer(window, layer_type, color, side=None):
    layer = window.add_layer(layer_type)
    if side is not None:
        layer.side = side
    window.inspector.apply_color(color)
    window.flush_recompute()
    return layer


# ---- image loading ----------------------------------------------------------------


def test_load_image_sets_canvas_and_swatches(window):
    assert window.image is not None
    assert window.canvas.base_item is not None
    assert len(window.inspector._swatch_buttons) > 0


def test_add_layer_requires_image(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    win = MainWindow()
    qtbot.addWidget(win)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    assert win.add_layer(LayerType.COPPER) is None


# ---- layer CRUD / reorder / visibility ---------------------------------------------


def test_add_layer_appears_in_pane_and_selected(window):
    layer = window.add_layer(LayerType.COPPER)
    assert window.layer_pane.list.count() == 2  # base image row + the new layer
    assert window.layer_pane.current_layer_id() == layer.id
    assert window.inspector._layer is layer


def test_multiple_cut_layers_can_be_added(window):
    first = window.add_layer(LayerType.CUT)
    second = window.add_layer(LayerType.CUT)
    assert first is not None and second is not None
    assert [layer.id for layer in window.project.cut_layers()] == [first.id, second.id]


def test_delete_layer_removes_overlay(window):
    layer = add_colored_layer(window, LayerType.COPPER, COPPER_COLOR)
    assert layer.id in window.canvas.overlay_items
    window.remove_layer(layer.id)
    assert layer.id not in window.canvas.overlay_items
    assert window.layer_pane.list.count() == 1  # only the base image row remains
    assert window.project.layers == []


def test_reorder_layers(window):
    a = window.add_layer(LayerType.COPPER)
    b = window.add_layer(LayerType.SILKSCREEN)
    window.move_layer(b.id, -1)
    assert [l.id for l in window.project.layers] == [b.id, a.id]


def test_rename_layer_via_pane(window):
    layer = window.add_layer(LayerType.COPPER)
    item = window.layer_pane.layer_item(layer.id)
    item.setText("My copper art")
    assert layer.name == "My copper art"


def test_eye_toggle_hides_overlay(window):
    layer = add_colored_layer(window, LayerType.COPPER, COPPER_COLOR)
    window.layer_pane.toggle_layer_visibility(layer.id)
    assert layer.visible is False
    assert not window.canvas.overlay_items[layer.id].isVisible()
    window.layer_pane.toggle_layer_visibility(layer.id)
    assert layer.visible is True
    assert window.canvas.overlay_items[layer.id].isVisible()


def test_base_image_toggle(window):
    window.base_visible_action.setChecked(False)
    assert not window.canvas.base_item.isVisible()
    window.base_visible_action.setChecked(True)
    assert window.canvas.base_item.isVisible()


# ---- color tools -------------------------------------------------------------------


def test_hex_input_sets_color(window, qtbot):
    layer = window.add_layer(LayerType.COPPER)
    window.inspector.hex_edit.setText("#1E3CC8")
    window.inspector._on_hex()
    assert layer.color == (0x1E, 0x3C, 0xC8)
    assert layer.has_color


def test_invalid_hex_rejected(window):
    layer = window.add_layer(LayerType.COPPER)
    window.inspector.apply_color(COPPER_COLOR)
    window.inspector.hex_edit.setText("not-a-color")
    window.inspector._on_hex()
    assert layer.color == COPPER_COLOR  # unchanged


def test_swatch_click_sets_color(window, qtbot):
    layer = window.add_layer(LayerType.COPPER)
    swatch = window.inspector._swatch_buttons[0]
    qtbot.mouseClick(swatch, Qt.LeftButton)
    assert layer.has_color


def test_eyedropper_picks_pixel_color(window):
    layer = window.add_layer(LayerType.COPPER)
    window.inspector.eyedropper_button.setChecked(True)
    assert window.canvas.eyedropper_active
    window.canvas.pixel_picked.emit(40, 40)  # inside the copper rectangle
    assert layer.color == COPPER_COLOR
    assert not window.inspector.eyedropper_button.isChecked()  # auto-disarm
    assert not window.canvas.eyedropper_active


def test_side_combo_updates_layer(window):
    layer = window.add_layer(LayerType.COPPER)
    window.inspector.side_combo.setCurrentIndex(1)
    assert layer.side == Side.BACK


def test_cut_layer_hides_side_and_shows_smoothness(window):
    window.add_layer(LayerType.CUT)
    assert not window.inspector.side_combo.isVisibleTo(window.inspector)
    assert window.inspector.smooth_slider.isVisibleTo(window.inspector)
    window.add_layer(LayerType.COPPER)
    assert window.inspector.side_combo.isVisibleTo(window.inspector)
    assert not window.inspector.smooth_slider.isVisibleTo(window.inspector)


# ---- live preview ------------------------------------------------------------------


def test_color_pick_creates_overlay_after_debounce(window, qtbot):
    layer = window.add_layer(LayerType.COPPER)
    window.inspector.apply_color(COPPER_COLOR)
    qtbot.waitUntil(lambda: layer.id in window.canvas.overlay_items, timeout=3000)
    assert layer.id in window.masks
    assert window.masks[layer.id].max() == 255


def test_tolerance_change_recomputes_mask(window):
    layer = add_colored_layer(window, LayerType.MASK, MASK_COLOR)
    window.inspector.tolerance_slider.setValue(0)
    window.flush_recompute()
    strict_count = int((window.masks[layer.id] > 0).sum())
    window.inspector.tolerance_slider.setValue(30)
    window.flush_recompute()
    tolerant_count = int((window.masks[layer.id] > 0).sum())
    assert tolerant_count > strict_count


def test_cut_layer_produces_outline(window):
    layer = add_colored_layer(window, LayerType.CUT, BOARD_COLOR)
    assert layer.id in window.outlines
    assert layer.id in window.canvas.outline_items
    assert layer.id not in window.canvas.overlay_items
    assert window.canvas.outline_items[layer.id].isVisible()
    outline = window.outlines[layer.id]
    assert outline.is_valid and outline.area > 0


def test_cut_smoothness_slider_updates_outline(window):
    layer = add_colored_layer(window, LayerType.CUT, BOARD_COLOR)
    raw_len = window.outlines[layer.id].exterior.length
    window.inspector.smooth_slider.setValue(80)
    window.flush_recompute()
    smooth_len = window.outlines[layer.id].exterior.length
    assert smooth_len <= raw_len


def test_cut_with_unmatched_color_reports_error(window):
    layer = window.add_layer(LayerType.CUT)
    window.inspector.apply_color((255, 0, 255))  # matches nothing
    window.flush_recompute()
    assert layer.id not in window.outlines
    assert "Cut layer" in window.statusBar().currentMessage()


# ---- export ------------------------------------------------------------------------


def test_export_gerbers_full_flow(window, tmp_path):
    add_colored_layer(window, LayerType.COPPER, COPPER_COLOR)
    add_colored_layer(window, LayerType.SILKSCREEN, SILK_COLOR)
    add_colored_layer(window, LayerType.CUT, BOARD_COLOR)
    written = window.export_gerbers(tmp_path, "art", 60.0, 45.0)
    names = sorted(p.name for p in written)
    assert names == ["art-Edge_Cuts.gm1", "art-F_Cu.gtl", "art-F_Silkscreen.gto"]


def test_export_without_layers_raises(window, tmp_path):
    from png_to_gerber.gerber import ExportError

    with pytest.raises(ExportError):
        window.export_gerbers(tmp_path, "art", 60.0, 45.0)


def test_export_skips_hidden_layer(window, tmp_path):
    copper = add_colored_layer(window, LayerType.COPPER, COPPER_COLOR)
    add_colored_layer(window, LayerType.SILKSCREEN, SILK_COLOR)
    window.set_layer_visible(copper.id, False)
    written = window.export_gerbers(tmp_path, "art", 60.0, 45.0)
    assert [p.name for p in written] == ["art-F_Silkscreen.gto"]


# ---- export dialog validation -------------------------------------------------------


def test_export_dialog_validation(qtbot, tmp_path):
    from png_to_gerber.gui.export_dialog import ExportDialog

    dialog = ExportDialog(200, 150, "art")
    qtbot.addWidget(dialog)
    assert dialog.validate() == "Choose an output folder."
    dialog.dir_edit.setText(str(tmp_path))
    dialog.name_edit.setText("")
    assert dialog.validate() == "Enter a base name for the Gerber files."
    dialog.name_edit.setText("bad/name")
    assert "not allowed" in dialog.validate()
    dialog.name_edit.setText("art")
    assert dialog.validate() is None


def test_export_dialog_aspect_lock(qtbot):
    from png_to_gerber.gui.export_dialog import ExportDialog

    dialog = ExportDialog(200, 100, "art")  # 2:1 aspect
    qtbot.addWidget(dialog)
    dialog.width_spin.setValue(80.0)
    assert dialog.height_spin.value() == pytest.approx(40.0, abs=0.01)
    dialog.lock_aspect.setChecked(False)
    dialog.width_spin.setValue(100.0)
    assert dialog.height_spin.value() == pytest.approx(40.0, abs=0.01)
