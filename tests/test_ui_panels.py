"""Layer-pane eyes, the base-image row/panel, and the checker-grid slider."""

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QFileDialog, QToolBar

from png_to_gerber.gui.main_window import MainWindow
from png_to_gerber.gui.canvas import _checker_brush
from png_to_gerber.model import LayerType

from conftest import COPPER_COLOR, SILK_COLOR


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


def image_bytes(icon):
    """The icon rendered as a QImage (compares pixel by pixel with ==)."""
    return icon.pixmap(18, 18).toImage().convertToFormat(QImage.Format_ARGB32)


# ---- eyes in the layer pane ----------------------------------------------------------------


def test_each_layer_row_has_an_eye_icon(window):
    add(window)
    add(window, LayerType.SILKSCREEN, SILK_COLOR)
    assert window.layer_pane.list.count() == 3  # base image + two layers
    for row in range(3):
        assert not window.layer_pane.list.item(row).icon().isNull()


def test_eye_icon_differs_between_visible_and_hidden(window):
    layer = add(window)
    item = window.layer_pane.layer_item(layer.id)
    open_eye = image_bytes(item.icon())
    window.layer_pane.toggle_layer_visibility(layer.id)
    closed_eye = image_bytes(item.icon())
    assert open_eye != closed_eye
    window.layer_pane.toggle_layer_visibility(layer.id)
    assert image_bytes(item.icon()) == open_eye


def test_eye_toggle_updates_model_and_canvas(window):
    layer = add(window)
    window.layer_pane.toggle_layer_visibility(layer.id)
    assert layer.visible is False
    assert window.layer_pane.is_layer_visible(layer.id) is False
    assert not window.canvas.overlay_items[layer.id].isVisible()


def test_hidden_layer_row_is_dimmed(window):
    layer = add(window)
    item = window.layer_pane.layer_item(layer.id)
    shown = item.foreground().color().alphaF()
    window.layer_pane.toggle_layer_visibility(layer.id)
    assert item.foreground().color().alphaF() < shown


def test_clicking_the_eye_toggles_but_clicking_the_name_selects(window, qtbot):
    a = add(window)
    b = add(window, LayerType.SILKSCREEN, SILK_COLOR)
    window.show()
    qtbot.waitExposed(window)
    pane = window.layer_pane
    first = pane.layer_item(a.id)
    rect = pane.list.visualItemRect(first)

    # Eye region: toggles visibility, does not change the selection.
    assert pane.current_layer_id() == b.id
    qtbot.mouseClick(
        pane.list.viewport(), Qt.LeftButton, pos=QPoint(rect.left() + 8, rect.center().y())
    )
    assert a.visible is False
    assert pane.current_layer_id() == b.id

    # Name region: selects, does not toggle.
    qtbot.mouseClick(
        pane.list.viewport(), Qt.LeftButton, pos=QPoint(rect.left() + 80, rect.center().y())
    )
    assert pane.current_layer_id() == a.id
    assert a.visible is False


def test_programmatic_visibility_change_updates_the_eye(window):
    layer = add(window)
    window.set_layer_visible(layer.id, False)
    assert window.layer_pane.is_layer_visible(layer.id) is False


def test_eye_states_survive_a_pane_refresh(window):
    a = add(window)
    b = add(window, LayerType.SILKSCREEN, SILK_COLOR)
    window.layer_pane.toggle_layer_visibility(b.id)
    window.move_layer(a.id, 1)  # triggers a refresh
    assert window.layer_pane.is_layer_visible(a.id) is True
    assert window.layer_pane.is_layer_visible(b.id) is False


def test_toggling_the_eye_does_not_rename_the_layer(window):
    layer = add(window)
    name = layer.name
    window.layer_pane.toggle_layer_visibility(layer.id)
    assert layer.name == name


# ---- base image row and panel ----------------------------------------------------------------


def test_base_image_is_the_first_row_of_the_layer_list(window):
    from png_to_gerber.gui.layer_pane import BASE_ITEM_ID

    add(window)
    pane = window.layer_pane
    base = pane.list.item(0)
    assert base.text() == "Base image"
    assert base.data(Qt.UserRole) == BASE_ITEM_ID
    assert pane.list.count() == 2
    assert not (base.flags() & Qt.ItemIsEditable)


def test_base_image_is_selected_at_startup_and_shows_its_options(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    assert win.layer_pane.base_is_selected()
    assert win.inspector_stack.currentWidget() is win.base_panel


def test_selecting_a_layer_swaps_to_layer_options_and_back(window):
    layer = add(window)
    assert window.inspector_stack.currentWidget() is window.inspector
    assert not window.layer_pane.base_is_selected()
    window.layer_pane.select_base()
    assert window.inspector_stack.currentWidget() is window.base_panel
    assert window.layer_pane.base_is_selected()
    assert window.layer_pane.current_layer_id() == -1
    window.layer_pane.select_layer(layer.id)
    assert window.inspector_stack.currentWidget() is window.inspector
    assert not window.layer_pane.base_is_selected()


def test_clicking_the_base_row_selects_it(window, qtbot):
    add(window)
    window.show()
    qtbot.waitExposed(window)
    pane = window.layer_pane
    rect = pane.list.visualItemRect(pane.list.item(0))
    qtbot.mouseClick(
        pane.list.viewport(), Qt.LeftButton, pos=QPoint(rect.left() + 80, rect.center().y())
    )
    assert pane.base_is_selected()
    assert window.inspector_stack.currentWidget() is window.base_panel


def test_base_eye_toggles_base_image_visibility(window):
    window.layer_pane.toggle_base_visibility()
    assert not window.canvas.base_item.isVisible()
    assert not window.base_visible_action.isChecked()
    assert window.project.base_visible is False
    window.layer_pane.toggle_base_visibility()
    assert window.canvas.base_item.isVisible()


def test_base_eye_follows_the_menu_action(window):
    window.base_visible_action.setChecked(False)
    assert window.layer_pane.is_base_visible() is False
    window.base_visible_action.setChecked(True)
    assert window.layer_pane.is_base_visible() is True


def test_base_eye_icon_changes_with_state(window):
    item = window.layer_pane.list.item(0)
    shown = image_bytes(item.icon())
    window.base_visible_action.setChecked(False)
    assert image_bytes(item.icon()) != shown


def test_hidden_base_stays_hidden_when_background_removed(window):
    window.base_visible_action.setChecked(False)
    window.remove_bg_action.setChecked(True)
    assert not window.canvas.base_item.isVisible()


def test_base_panel_shows_image_info(window):
    text = window.base_panel.info_label.text()
    assert "fixture.png" in text and "200 x 150" in text


def test_base_panel_background_checkbox_drives_removal(qtbot, tmp_path):
    img = np.zeros((40, 40, 4), dtype=np.uint8)
    img[..., :3] = 255
    img[..., 3] = 255
    img[10:30, 10:30, :3] = (0, 0, 255)
    path = tmp_path / "p.png"
    Image.fromarray(img, "RGBA").save(path)
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(path)

    win.base_panel.remove_bg_check.setChecked(True)
    assert win.image.shape[:2] == (20, 20)
    assert tuple(win.image[0, 0, :3]) == (0, 0, 255)
    assert win.project.background_removed
    win.base_panel.remove_bg_check.setChecked(False)
    assert win.image.shape[:2] == (40, 40)
    assert win.image[0, 0, 3] == 255


def test_base_panel_checkbox_follows_programmatic_changes(window):
    window.set_background_removed(True, tolerance=25)
    assert window.base_panel.remove_bg_check.isChecked()
    assert window.base_panel.tolerance_spin.value() == 25
    window.set_background_removed(False)
    assert not window.base_panel.remove_bg_check.isChecked()


def test_base_panel_tolerance_spin_updates_project(window):
    window.base_panel.tolerance_spin.setValue(33)
    assert window.project.background_tolerance == 33
    assert window.bg_tolerance_spin is window.base_panel.tolerance_spin


def test_base_panel_open_button_loads_an_image(window, tmp_path, monkeypatch):
    other = tmp_path / "other.png"
    Image.new("RGBA", (30, 20), (9, 9, 9, 255)).save(other)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(other), ""))
    window.base_panel.open_button.click()
    assert window.image_path == other
    assert "30 x 20" in window.base_panel.info_label.text()


def test_toolbar_keeps_only_the_checker_slider(window):
    toolbar = window.findChildren(QToolBar)[0]
    actions = toolbar.actions()
    assert window.remove_bg_action not in actions
    assert window.base_visible_action not in actions
    assert window.open_action not in actions
    assert window.save_action not in actions
    assert window.export_files_action not in actions
    assert window.bg_tolerance_spin.parent() is not toolbar
    assert window.checker_label.text() == "Checker opacity"
    parent = window.checker_label.parent()
    while parent is not None and parent is not toolbar:
        parent = parent.parent()
    assert parent is toolbar


def test_base_row_cannot_be_deleted_or_moved(window):
    window.layer_pane.select_base()
    assert not window.layer_pane.duplicate_button.isEnabled()
    assert not window.layer_pane.delete_button.isEnabled()
    assert not window.layer_pane.up_button.isEnabled()
    assert not window.layer_pane.down_button.isEnabled()


def test_deleting_the_last_layer_returns_to_the_base_image_options(window):
    layer = add(window)
    window.remove_layer(layer.id)
    assert window.layer_pane.base_is_selected()
    assert window.inspector_stack.currentWidget() is window.base_panel


def test_base_panel_reflects_a_reopened_project(qtbot, window, tmp_path):
    window.set_background_removed(True, tolerance=20)
    prj = window.save_project(tmp_path / "w.prj")
    fresh = MainWindow()
    qtbot.addWidget(fresh)
    fresh.open_project(prj)
    assert fresh.base_panel.remove_bg_check.isChecked()
    assert fresh.base_panel.tolerance_spin.value() == 20
    assert "fixture.png" in fresh.base_panel.info_label.text()


# ---- checker-grid slider ----------------------------------------------------------------------


def checker_pixels(opacity):
    image = _checker_brush(opacity).texture().toImage()
    return image.pixelColor(2, 2).name(), image.pixelColor(10, 2).name()


def test_checker_slider_lives_in_the_toolbar(window):
    toolbar = window.findChildren(QToolBar)[0]
    widget = window.checker_slider
    parent = widget.parent()
    while parent is not None and parent is not toolbar:
        parent = parent.parent()
    assert parent is toolbar
    assert (widget.minimum(), widget.maximum()) == (0, 100)


def test_checker_defaults_to_half_strength(window):
    assert window.checker_slider.value() == 50
    assert window.canvas.checker_opacity == 50
    assert window.project.checker_opacity == 50


def test_slider_changes_canvas_checker_opacity(window):
    window.checker_slider.setValue(35)
    assert window.canvas.checker_opacity == 35
    assert window.project.checker_opacity == 35


def test_zero_opacity_is_flat_and_full_opacity_is_checked():
    light, dark = checker_pixels(100)
    assert light != dark
    flat_a, flat_b = checker_pixels(0)
    assert flat_a == flat_b


def test_checker_contrast_grows_with_opacity():
    def contrast(opacity):
        a, b = checker_pixels(opacity)
        return abs(int(a[1:3], 16) - int(b[1:3], 16))

    assert contrast(0) < contrast(40) < contrast(100)


def test_checker_opacity_is_clamped():
    assert checker_pixels(500) == checker_pixels(100)
    assert checker_pixels(-5) == checker_pixels(0)


def test_checker_brush_is_applied_to_the_view(window):
    before = window.canvas.backgroundBrush().texture().toImage()
    window.checker_slider.setValue(0)
    after = window.canvas.backgroundBrush().texture().toImage()
    assert before != after


def test_checker_opacity_is_saved_and_restored(qtbot, window, tmp_path):
    window.checker_slider.setValue(42)
    prj = window.save_project(tmp_path / "w.prj")
    fresh = MainWindow()
    qtbot.addWidget(fresh)
    fresh.open_project(prj)
    assert fresh.checker_slider.value() == 42
    assert fresh.canvas.checker_opacity == 42
    assert fresh.project.checker_opacity == 42


def test_invalid_saved_checker_opacity_is_rejected(tmp_path):
    import json

    from png_to_gerber.project_io import ProjectFileError, load_project, project_to_dict
    from png_to_gerber.model import Project

    data = project_to_dict(Project(), tmp_path)
    data["view"]["checker_opacity"] = 150
    path = tmp_path / "bad.prj"
    path.write_text(json.dumps(data))
    with pytest.raises(ProjectFileError, match="checker_opacity"):
        load_project(path)
