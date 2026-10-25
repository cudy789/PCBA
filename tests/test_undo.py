"""Undo and redo for project edits, from the File menu and the keyboard."""

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QKeySequence, QMouseEvent
from PySide6.QtWidgets import QApplication

from png_to_gerber.gui.main_window import MainWindow
from png_to_gerber.model import BoardSettings, LayerType
from png_to_gerber.paint import Tool

from conftest import BOARD_COLOR, COPPER_COLOR, SILK_COLOR


@pytest.fixture
def window(qtbot, fixture_image_path):
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(fixture_image_path)
    return win


def test_undo_and_redo_are_at_the_top_of_the_file_menu(window):
    file_menu = window.menuBar().actions()[0].menu()
    actions = file_menu.actions()
    assert actions[0] is window.undo_action
    assert actions[1] is window.redo_action
    assert actions[2].isSeparator()
    assert window.undo_action.shortcut() == QKeySequence("Ctrl+Z")
    assert window.redo_action.shortcut() == QKeySequence("Ctrl+Shift+Z")
    assert not window.undo_action.isEnabled()
    assert not window.redo_action.isEnabled()


def test_undo_and_redo_do_not_open_a_window(window, qtbot):
    window.show()
    qtbot.waitExposed(window)
    window.activateWindow()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is window)
    window.add_layer(LayerType.COPPER)
    window.inspector.apply_color(COPPER_COLOR)
    qtbot.waitUntil(lambda: bool(window.inspector._swatch_buttons))

    def stray_windows():
        QApplication.processEvents()
        return [w for w in QApplication.topLevelWidgets() if w.isVisible() and w is not window]

    qtbot.keyClick(window, Qt.Key_Z, Qt.ControlModifier)
    assert stray_windows() == []
    qtbot.keyClick(window, Qt.Key_Z, Qt.ControlModifier | Qt.ShiftModifier)
    assert stray_windows() == []


def test_ctrl_z_and_ctrl_shift_z_undo_and_redo(window, qtbot):
    window.show()
    qtbot.waitExposed(window)
    window.activateWindow()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is window)
    window.add_layer(LayerType.COPPER)
    qtbot.keyClick(window, Qt.Key_Z, Qt.ControlModifier)
    assert window.project.layers == []
    qtbot.keyClick(window, Qt.Key_Z, Qt.ControlModifier | Qt.ShiftModifier)
    assert window.project.layers[0].layer_type == LayerType.COPPER


def test_layer_edits_undo_redo_and_the_saved_marker(window, tmp_path):
    window.add_layer(LayerType.COPPER)
    assert window.is_modified
    assert window.windowTitle().startswith("*")
    copper_id = window.project.layers[0].id

    window.undo()
    assert window.project.layers == []
    assert not window.is_modified
    assert not window.windowTitle().startswith("*")
    assert not window.undo_action.isEnabled()
    assert window.redo_action.isEnabled()

    window.redo()
    assert window.project.layers[0].id == copper_id
    assert window.is_modified

    window.save_project(tmp_path / "saved.prj")
    assert not window.is_modified
    window.project.layers[0].name = "will be replaced through the pane"
    window.layer_pane.layer_item(window.project.layers[0].id).setText("Trace")
    window.inspector.apply_color(COPPER_COLOR)
    window.inspector.side_combo.setCurrentIndex(1)
    window.inspector.fill_holes_check.setChecked(True)
    window.inspector.invert_button.setChecked(True)
    window.set_layer_visible(window.project.layers[0].id, False)
    window._store_board_settings(BoardSettings(12.0, 9.0, True, "", "board"))

    window.undo()  # board
    assert window.project.board.width_mm is None
    window.undo()  # visibility
    assert window.project.layers[0].visible
    window.undo()  # invert
    assert not window.project.layers[0].inverted
    window.undo()  # fill holes
    assert not window.project.layers[0].fill_holes
    window.undo()  # side
    assert window.project.layers[0].side.value == "front"
    window.undo()  # color
    assert not window.project.layers[0].has_color
    window.undo()  # rename
    assert window.project.layers[0].name.startswith("Copper")
    assert not window.is_modified  # back at the save
    assert window.project.layers[0].id == copper_id

    window.undo()  # the added layer, past the save
    assert window.project.layers == []
    assert window.is_modified

    window.redo()
    assert not window.is_modified
    window.add_layer(LayerType.SILKSCREEN)
    assert not window.redo_action.isEnabled()
    assert window.is_modified


def test_a_slider_drag_is_one_step_and_spin_clicks_are_separate(window):
    window.add_layer(LayerType.COPPER)
    start = window.project.layers[0].tolerance
    slider = window.inspector.tolerance_slider
    slider.sliderPressed.emit()
    slider.setValue(start + 1)
    slider.setValue(start + 4)
    slider.setValue(start + 9)
    slider.sliderReleased.emit()
    assert window.project.layers[0].tolerance == start + 9

    window.undo()
    assert window.project.layers[0].tolerance == start
    assert len(window.project.layers) == 1

    window.base_panel.tolerance_spin.setValue(20)
    window.base_panel.tolerance_spin.setValue(21)
    window.undo()
    assert window.project.images[0].background_tolerance == 20
    window.undo()
    assert window.project.images[0].background_tolerance == 12


def test_holding_a_spin_box_is_one_step(window):
    spin = window.base_panel.tolerance_spin
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(4, 4),
        QPointF(4, 4),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.NoModifier,
    )
    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(4, 4),
        QPointF(4, 4),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.NoModifier,
    )
    window.eventFilter(spin, press)
    spin.setValue(30)
    spin.setValue(31)
    assert window.eventFilter(spin, release) is False
    QApplication.processEvents()
    assert window.project.images[0].background_tolerance == 31
    window.undo()
    assert window.project.images[0].background_tolerance == 12


def test_cutoff_drag_is_one_step(window):
    window.set_flatten(True)
    bar = window.base_panel.cutoff_bar
    bar.resize(300, 80)
    before = list(window.project.images[0].flatten_cutoffs)
    y = bar.TRACK_TOP + 4
    start = bar.luminance_to_x(before[0])

    def mouse(kind, x, buttons):
        event = QMouseEvent(
            kind,
            QPointF(x, y),
            QPointF(x, y),
            Qt.MouseButton.LeftButton,
            buttons,
            Qt.NoModifier,
        )
        if kind == QEvent.Type.MouseButtonPress:
            bar.mousePressEvent(event)
        elif kind == QEvent.Type.MouseMove:
            bar.mouseMoveEvent(event)
        else:
            bar.mouseReleaseEvent(event)

    mouse(QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
    mouse(QEvent.Type.MouseMove, bar.luminance_to_x(before[0] + 10), Qt.MouseButton.LeftButton)
    mouse(QEvent.Type.MouseMove, bar.luminance_to_x(before[0] + 18), Qt.MouseButton.LeftButton)
    mouse(QEvent.Type.MouseButtonRelease, bar.luminance_to_x(before[0] + 18), Qt.MouseButton.NoButton)
    assert window.project.images[0].flatten_cutoffs != before

    window.undo()
    assert window.project.images[0].flatten_cutoffs == before
    assert window.project.images[0].flatten_enabled
    window.undo()
    assert not window.project.images[0].flatten_enabled


def test_paint_stroke_line_and_fill_are_single_steps(window):
    window.add_layer(LayerType.COPPER)
    window.set_brush_size(1)
    window.paint_stroke([(10, 10), (11, 10), (12, 10)])
    window.paint_line(20, 10, 24, 10)
    window.paint_fill(0, 0)
    assert window.project.layers[0].manual

    window.undo()  # fill
    manual_after_line = list(window.project.layers[0].manual)
    assert manual_after_line
    window.undo()  # line
    after_stroke = list(window.project.layers[0].manual)
    assert after_stroke
    assert after_stroke != manual_after_line
    window.undo()  # whole stroke, not each dab
    assert window.project.layers[0].manual == []
    assert len(window.project.layers) == 1

    window.layer_pane.select_base()
    window.set_paint_color((1, 2, 3, 255))
    window.paint_stroke([(4, 4), (5, 4)])
    painted = window.image[4, 4, :3].copy()
    assert tuple(int(v) for v in painted) == (1, 2, 3)
    window.undo()
    assert not np.array_equal(window.image[4, 4, :3], painted)


def test_view_settings_are_not_undo_steps(window):
    window.set_paint_tool(Tool.LINE)
    window.set_brush_size(9)
    window.canvas.scale(1.5, 1.5)
    window._on_compare_original(True)
    assert not window.undo_action.isEnabled()
    assert window.paint_toolbar.tool == Tool.LINE
    assert window.paint_toolbar.brush_size == 9


def test_images_background_checker_and_selection_restore(window, tmp_path, fixture_image_path):
    window.add_layer(LayerType.COPPER)
    window.inspector.apply_color(COPPER_COLOR)
    original = window.original_image.copy()
    window.set_background_removed(True)
    cropped = tuple(window.image.shape[:2])
    assert cropped != tuple(original.shape[:2])
    window.checker_slider.sliderPressed.emit()
    window.checker_slider.setValue(10)
    window.checker_slider.setValue(15)
    window.checker_slider.sliderReleased.emit()
    window.base_visible_action.setChecked(False)

    window.undo()
    assert window.project.base_visible
    window.undo()
    assert window.project.checker_opacity == 50
    assert window.checker_slider.value() == 50
    window.undo()
    assert tuple(window.image.shape[:2]) == tuple(original.shape[:2])
    assert not window.project.background_removed

    second = tmp_path / "second.png"
    second.write_bytes(fixture_image_path.read_bytes())
    window.add_image(second)
    assert len(window.project.images) == 2
    window.undo()
    assert len(window.project.images) == 1

    window.load_image(second)
    assert window.is_modified
    window.undo()
    assert np.array_equal(window.original_image, original)
    assert window.project.layers[0].has_color

    window.layer_pane.select_layer(window.project.layers[0].id)
    layer_id = window.project.layers[0].id
    window.remove_layer(layer_id)
    assert window.layer_pane.current_layer_id() < 0
    window.undo()
    assert window.layer_pane.current_layer_id() == layer_id


def test_open_project_clears_history(window, tmp_path):
    window.add_layer(LayerType.COPPER)
    path = window.save_project(tmp_path / "one.prj")
    window.add_layer(LayerType.SILKSCREEN)
    assert window.undo_action.isEnabled()
    window.open_project(path)
    assert not window.is_modified
    assert len(window.project.layers) == 1
    assert not window.undo_action.isEnabled()
    assert not window.redo_action.isEnabled()


def test_undo_of_a_stroke_keeps_masks_that_do_not_depend_on_it(window):
    copper = window.add_layer(LayerType.COPPER)
    window.inspector.apply_color(COPPER_COLOR)
    silk = window.add_layer(LayerType.SILKSCREEN)
    window.inspector.apply_color(SILK_COLOR)
    cut = window.add_layer(LayerType.CUT)
    window.inspector.apply_color(BOARD_COLOR)
    negative = window.generate_negative([copper.id], LayerType.MASK)
    window.flush_recompute()
    window.layer_pane.select_layer(copper.id)

    silk_mask = window.masks[silk.id]
    cut_outline = window.outlines[cut.id]
    copper_before = window.masks[copper.id].copy()
    negative_before = window.masks[negative.id].copy()

    window.set_brush_size(1)
    window.paint_stroke([(20, 20)])
    assert window.masks[silk.id] is silk_mask
    assert window.outlines[cut.id] is cut_outline
    assert not np.array_equal(window.masks[copper.id], copper_before)
    assert not np.array_equal(window.masks[negative.id], negative_before)
    copper_painted = window.masks[copper.id].copy()
    negative_painted = window.masks[negative.id].copy()

    window.undo()
    assert window.masks[silk.id] is silk_mask
    assert window.outlines[cut.id] is cut_outline
    assert np.array_equal(window.masks[copper.id], copper_before)
    assert np.array_equal(window.masks[negative.id], negative_before)

    window.redo()
    assert window.masks[silk.id] is silk_mask
    assert window.outlines[cut.id] is cut_outline
    assert np.array_equal(window.masks[copper.id], copper_painted)
    assert np.array_equal(window.masks[negative.id], negative_painted)


def test_undo_visibility_shows_the_overlay_without_rebuilding_its_mask(window):
    silk = window.add_layer(LayerType.SILKSCREEN)
    window.inspector.apply_color(SILK_COLOR)
    window.flush_recompute()
    mask = window.masks[silk.id]
    window.set_layer_visible(silk.id, False)
    assert not window.canvas.overlay_items[silk.id].isVisible()

    window.undo()
    assert window.project.layers[0].visible
    assert window.canvas.overlay_items[silk.id].isVisible()
    assert window.masks[silk.id] is mask


def test_undo_base_paint_rebuilds_layer_masks(window):
    layer = window.add_layer(LayerType.COPPER)
    window.inspector.apply_color(COPPER_COLOR)
    window.flush_recompute()
    before = window.masks[layer.id].copy()
    window.layer_pane.select_base()
    window.set_paint_color(COPPER_COLOR)
    window.set_brush_size(1)
    window.paint_stroke([(29, 40)])  # beside the copper block, so it is not a speck
    window.flush_recompute()
    assert window.masks[layer.id][40, 29] == 255
    assert not np.array_equal(window.masks[layer.id], before)

    window.undo()
    assert window.masks[layer.id][40, 29] == 0
    assert np.array_equal(window.masks[layer.id], before)


def test_key_repeat_on_a_slider_is_one_step(window):
    window.add_layer(LayerType.COPPER)
    slider = window.inspector.tolerance_slider
    start = slider.value()
    press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Right, Qt.NoModifier)
    repeat = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Right, Qt.NoModifier, autorep=True, count=2)
    release = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key_Right, Qt.NoModifier)
    window.eventFilter(slider, press)
    slider.setValue(start + 1)
    window.eventFilter(slider, repeat)
    slider.setValue(start + 2)
    window.eventFilter(slider, release)
    QApplication.processEvents()
    window.undo()
    assert window.project.layers[0].tolerance == start
