"""Duplicate, retype, invert, and the base-image row inside the layer list."""

import json

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QMessageBox

from gerber_reader import mismatch, rasterize, read_regions
from png_to_gerber.gui.layer_pane import BASE_ITEM_ID
from png_to_gerber.gui.main_window import MainWindow
from png_to_gerber.model import LayerType, Side
from png_to_gerber.project_io import FORMAT, ProjectFileError, load_project, project_from_dict

from conftest import BOARD_COLOR, COPPER_COLOR, MASK_COLOR, SILK_COLOR


@pytest.fixture
def window(qtbot, fixture_image_path):
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(fixture_image_path)
    return win


def add(window, layer_type=LayerType.COPPER, color=COPPER_COLOR):
    layer = window.add_layer(layer_type)
    window.inspector.apply_color(color)
    window.flush_recompute()
    return layer


def warnings_of(monkeypatch):
    """Record warning dialogs and still let window teardown discard changes."""
    seen = []

    def spy(*args, **kwargs):
        seen.append(args)
        return QMessageBox.StandardButton.Discard

    monkeypatch.setattr(QMessageBox, "warning", spy)
    return seen


def choose_type(window, layer_type):
    combo = window.inspector.type_combo
    combo.setCurrentIndex(combo.findData(layer_type))


# ---- duplicate ------------------------------------------------------------------------------


def test_duplicate_copies_settings_and_sits_below_the_original(window):
    first = add(window)
    window.inspector.tolerance_slider.setValue(8)
    window.inspector.fill_holes_check.setChecked(True)
    window.inspector.invert_button.setChecked(True)
    first.side = Side.BACK
    window.flush_recompute()
    second = add(window, LayerType.SILKSCREEN, SILK_COLOR)

    copy = window.duplicate_layer(first.id)
    window.flush_recompute()

    assert copy is not None and copy.id != first.id
    assert copy.name == f"{first.name} copy"
    assert copy.layer_type == first.layer_type
    assert copy.side == Side.BACK
    assert copy.color == first.color
    assert copy.tolerance == 8
    assert copy.fill_holes is True
    assert copy.inverted is True
    assert copy.negative_of == []
    assert [layer.id for layer in window.project.layers] == [first.id, copy.id, second.id]
    assert window.layer_pane.current_layer_id() == copy.id
    assert np.array_equal(window.masks[copy.id], window.masks[first.id])
    # The base image stays the first row no matter how layers are ordered.
    assert window.layer_pane.list.item(0).data(Qt.UserRole) == BASE_ITEM_ID


def test_duplicate_button_creates_the_layer(window):
    layer = add(window)
    assert window.layer_pane.duplicate_button.isEnabled()
    window.layer_pane.duplicate_button.click()
    window.flush_recompute()
    assert len(window.project.layers) == 2
    copy = window.project.layers[1]
    assert copy.name == f"{layer.name} copy"
    assert np.array_equal(window.masks[copy.id], window.masks[layer.id])


def test_second_duplicate_gets_a_distinct_name(window):
    layer = add(window)
    window.duplicate_layer(layer.id)
    again = window.duplicate_layer(layer.id)
    assert again.name == f"{layer.name} copy 2"


def test_duplicate_of_a_negative_tracks_the_same_sources(window):
    source = add(window)
    negative = window.generate_negative([source.id], LayerType.MASK)
    window.flush_recompute()
    copy = window.duplicate_layer(negative.id)
    window.flush_recompute()
    assert copy.negative_of == [source.id]
    assert copy.negative_of is not negative.negative_of
    assert np.array_equal(window.masks[copy.id], window.masks[negative.id])

    window.layer_pane.select_layer(source.id)
    window.inspector.close_spin.setValue(0)
    window.inspector.tolerance_slider.setValue(0)
    window.flush_recompute()
    assert np.array_equal(window.masks[copy.id], window.masks[negative.id])
    assert int((window.masks[copy.id] > 0).sum()) > 0


def test_deleting_a_source_removes_a_duplicated_negative_too(window):
    source = add(window)
    negative = window.generate_negative([source.id], LayerType.COPPER)
    copy = window.duplicate_layer(negative.id)
    window.remove_layer(source.id)
    remaining = {layer.id for layer in window.project.layers}
    assert negative.id not in remaining
    assert copy.id not in remaining


def test_duplicate_cut_is_rejected(window, monkeypatch):
    seen = warnings_of(monkeypatch)
    cut = add(window, LayerType.CUT, BOARD_COLOR)
    assert window.duplicate_layer(cut.id) is None
    assert window.project.layers == [cut]
    assert seen and "cut" in seen[0][1].lower() + seen[0][2].lower()


def test_duplicate_marks_the_project_modified(window, tmp_path):
    layer = add(window)
    window.save_project(tmp_path / "d.prj")
    window.duplicate_layer(layer.id)
    assert window.is_modified


def test_hidden_duplicate_stays_hidden(window):
    layer = add(window)
    window.set_layer_visible(layer.id, False)
    copy = window.duplicate_layer(layer.id)
    window.flush_recompute()
    assert copy.visible is False
    assert not window.canvas.overlay_items[copy.id].isVisible()


# ---- change layer type -----------------------------------------------------------------------


def test_type_combo_switches_copper_to_mask_without_changing_the_mask(window, tmp_path):
    layer = add(window)
    before = window.masks[layer.id].copy()
    choose_type(window, LayerType.MASK)
    window.flush_recompute()
    assert layer.layer_type == LayerType.MASK
    assert np.array_equal(window.masks[layer.id], before)
    assert window.inspector.side_combo.isVisibleTo(window.inspector)
    assert not window.inspector.outline_combo.isVisibleTo(window.inspector)
    assert "Solder mask" in window.layer_pane.layer_item(layer.id).toolTip()

    written = {path.name for path in window.export_gerbers(tmp_path, "retype", 40, 30)}
    assert "retype-F_Mask.gts" in written
    assert "retype-F_Cu.gtl" not in written


def test_changing_to_cut_draws_an_outline_and_back_removes_it(window, tmp_path):
    layer = add(window)
    choose_type(window, LayerType.CUT)
    window.flush_recompute()
    assert layer.is_cut
    assert layer.id in window.outlines
    assert window.inspector.outline_combo.isVisibleTo(window.inspector)
    assert not window.inspector.side_combo.isVisibleTo(window.inspector)
    names = {path.name for path in window.export_gerbers(tmp_path, "cut", 40, 30)}
    assert "cut-Edge_Cuts.gm1" in names

    choose_type(window, LayerType.COPPER)
    window.flush_recompute()
    assert layer.layer_type == LayerType.COPPER
    assert layer.id not in window.outlines
    names = {path.name for path in window.export_gerbers(tmp_path, "back", 40, 30)}
    assert "back-F_Cu.gtl" in names
    assert "back-Edge_Cuts.gm1" not in names


def test_second_cut_via_the_type_combo_is_rejected_and_reverted(window, monkeypatch):
    seen = warnings_of(monkeypatch)
    add(window, LayerType.CUT, BOARD_COLOR)
    copper = add(window)
    choose_type(window, LayerType.CUT)
    assert copper.layer_type == LayerType.COPPER
    assert window.inspector.type_combo.currentData() == LayerType.COPPER
    assert window.project.cut_layer() is not copper
    assert seen


def test_a_negative_cannot_become_a_cut(window, monkeypatch):
    seen = warnings_of(monkeypatch)
    source = add(window)
    negative = window.generate_negative([source.id], LayerType.MASK)
    choose_type(window, LayerType.CUT)
    assert negative.layer_type == LayerType.MASK
    assert window.inspector.type_combo.currentData() == LayerType.MASK
    assert seen


def test_retiring_the_cut_layer_allows_a_new_one(window):
    cut = add(window, LayerType.CUT, BOARD_COLOR)
    choose_type(window, LayerType.COPPER)
    window.flush_recompute()
    replacement = window.add_layer(LayerType.CUT)
    assert replacement is not None
    assert replacement.id != cut.id
    assert window.project.cut_layer() is replacement


def test_type_change_is_saved_and_marks_modified(qtbot, window, tmp_path):
    layer = add(window)
    window.save_project(tmp_path / "before.prj")
    choose_type(window, LayerType.SILKSCREEN)
    assert window.is_modified
    path = window.save_project(tmp_path / "typed.prj")

    fresh = MainWindow()
    qtbot.addWidget(fresh)
    fresh.open_project(path)
    assert fresh.project.layers[0].layer_type == LayerType.SILKSCREEN
    assert fresh.inspector.type_combo.currentData() == LayerType.SILKSCREEN


# ---- base image inside the list --------------------------------------------------------------


def test_base_image_stays_first_when_layers_move(window):
    first = add(window)
    second = add(window, LayerType.SILKSCREEN, SILK_COLOR)
    window.move_layer(second.id, -1)
    pane = window.layer_pane
    assert pane.list.item(0).text() == "Base image"
    assert pane.list.item(0).data(Qt.UserRole) == BASE_ITEM_ID
    assert [layer.id for layer in window.project.layers] == [second.id, first.id]
    assert not pane.up_button.isEnabled()  # the top layer cannot pass the base row


def test_base_row_name_cannot_be_edited(window, tmp_path):
    window.save_project(tmp_path / "b.prj")
    window.layer_pane.list.item(0).setText("not the base")
    assert window.layer_pane.list.item(0).text() == "Base image"
    assert not window.is_modified


def test_selecting_the_base_row_hides_layer_actions(window):
    add(window)
    window.layer_pane.select_base()
    pane = window.layer_pane
    assert pane.base_is_selected()
    assert window.inspector_stack.currentWidget() is window.base_panel
    assert not pane.duplicate_button.isEnabled()
    assert not pane.delete_button.isEnabled()
    assert not pane.up_button.isEnabled()
    assert not pane.down_button.isEnabled()
    assert not pane.negative_button.isEnabled()


def test_negative_selection_ignores_the_base_row(window):
    layer = add(window)
    pane = window.layer_pane
    pane.list.clearSelection()
    pane.list.item(0).setSelected(True)
    pane.layer_item(layer.id).setSelected(True)
    assert pane.selected_ids() == [layer.id]
    negative = window.generate_negative(pane.selected_ids(), LayerType.COPPER)
    assert negative.negative_of == [layer.id]


def test_base_eye_click_does_not_select_or_hide_a_layer(window, qtbot):
    layer = add(window)
    window.show()
    qtbot.waitExposed(window)
    pane = window.layer_pane
    rect = pane.list.visualItemRect(pane.list.item(0))
    qtbot.mouseClick(
        pane.list.viewport(), Qt.LeftButton, pos=QPoint(rect.left() + 8, rect.center().y())
    )
    assert layer.visible is True
    assert not window.canvas.base_item.isVisible()
    assert pane.current_layer_id() == layer.id


# ---- invert ----------------------------------------------------------------------------------


def test_invert_flips_the_layer_and_exports_the_flipped_mask(window, tmp_path):
    layer = add(window)
    before = window.masks[layer.id].copy()
    assert before[40, 40] == 255  # inside the copper block
    assert before[20, 20] == 0  # board color, not copper

    window.inspector.invert_button.click()
    window.flush_recompute()
    inverted = window.masks[layer.id]
    assert layer.inverted is True
    assert window.inspector.invert_button.isChecked()
    assert inverted[40, 40] == 0
    assert inverted[20, 20] == 255
    assert inverted[0, 0] == 0  # transparent background is never painted
    assert "inverted" in window.layer_pane.layer_item(layer.id).toolTip()

    written = window.export_gerbers(tmp_path, "inv", 40, 30)
    copper = next(path for path in written if path.name.endswith(".gtl"))
    painted = rasterize(read_regions(copper, 200, 150, 40, 30), 200, 150)
    assert mismatch(inverted, painted) == 0

    window.inspector.invert_button.click()
    window.flush_recompute()
    assert layer.inverted is False
    assert np.array_equal(window.masks[layer.id], before)


def test_inverting_a_negative_restores_the_source_region(window):
    source = add(window)
    negative = window.generate_negative([source.id], LayerType.COPPER)
    window.flush_recompute()
    assert window.inspector.invert_button.isVisibleTo(window.inspector)
    window.inspector.invert_button.setChecked(True)
    window.flush_recompute()
    source_on = window.masks[source.id] > 0
    restored = window.masks[negative.id] > 0
    assert (source_on & restored).sum() / source_on.sum() > 0.99
    assert not restored[20, 20]  # board pixels stay out of the restored copper


def test_inverting_a_cut_layer_follows_the_flipped_mask(window):
    cut = add(window, LayerType.CUT, BOARD_COLOR)
    assert window.masks[cut.id][20, 20] == 255
    window.inspector.invert_button.setChecked(True)
    window.flush_recompute()
    assert window.masks[cut.id][20, 20] == 0
    assert window.masks[cut.id][40, 40] == 255  # the copper block is now inside the cut
    assert cut.id in window.outlines


def test_invert_is_saved_and_an_old_file_without_it_stays_uninverted(qtbot, window, tmp_path):
    layer = add(window)
    window.inspector.invert_button.setChecked(True)
    window.flush_recompute()
    path = window.save_project(tmp_path / "inv.prj")

    fresh = MainWindow()
    qtbot.addWidget(fresh)
    fresh.open_project(path)
    restored = fresh.project.layers[0]
    assert restored.inverted is True
    assert fresh.inspector.invert_button.isChecked()
    assert np.array_equal(fresh.masks[restored.id], window.masks[layer.id])

    data = json.loads(path.read_text(encoding="utf-8"))
    del data["layers"][0]["inverted"]
    legacy = tmp_path / "legacy.prj"
    legacy.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_project(legacy)
    assert loaded.layers[0].inverted is False

    data["layers"][0]["inverted"] = "yes"
    broken = tmp_path / "bad.prj"
    broken.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ProjectFileError, match="inverted"):
        load_project(broken)


def test_missing_checker_opacity_defaults_to_half_and_explicit_values_load():
    bare = project_from_dict({"format": FORMAT, "version": 1})
    assert bare.checker_opacity == 50
    full = project_from_dict(
        {"format": FORMAT, "version": 1, "view": {"checker_opacity": 100}}
    )
    assert full.checker_opacity == 100


def test_uninverted_export_still_matches_the_preview(window, tmp_path):
    """Invert defaults off, so a normal copper export is unchanged."""
    layer = add(window)
    assert layer.inverted is False
    written = window.export_gerbers(tmp_path, "plain", 40, 30)
    copper = next(path for path in written if path.name.endswith(".gtl"))
    painted = rasterize(read_regions(copper, 200, 150, 40, 30), 200, 150)
    assert mismatch(window.masks[layer.id], painted) == 0
    assert painted[40, 40]
