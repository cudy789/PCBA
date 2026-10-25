"""Paint toolbar, layer masks, and base-image painting."""

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QColorDialog, QSlider

from png_to_gerber.gui.main_window import MainWindow
from png_to_gerber.model import LayerType, Side
from png_to_gerber.paint import Tool, disk_mask

from conftest import COPPER_COLOR, SILK_COLOR


@pytest.fixture
def window(qtbot, fixture_image_path):
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(fixture_image_path)
    return win


def _mask(window, layer):
    return window.masks[layer.id]


# ---- toolbar -----------------------------------------------------------------------


def test_tools_live_on_a_left_toolbar_and_share_one_size(window, qtbot):
    assert window.toolBarArea(window.paint_toolbar_host) == Qt.LeftToolBarArea
    sliders = window.paint_toolbar.findChildren(QSlider)
    assert len(sliders) == 1

    for tool in (Tool.BRUSH, Tool.LINE, Tool.RECTANGLE, Tool.ERASE, Tool.FILL):
        button = window.paint_toolbar.button(tool)
        assert button.isEnabled()
        assert button.toolTip()

    qtbot.mouseClick(window.paint_toolbar.button(Tool.BRUSH), Qt.LeftButton)
    assert window.canvas.tool == Tool.BRUSH
    assert window.paint_toolbar.size_slider.isEnabled()

    window.set_paint_tool(Tool.FILL)
    assert not window.paint_toolbar.size_slider.isEnabled()
    window.set_paint_tool(Tool.RECTANGLE)
    assert window.paint_toolbar.size_slider.isEnabled()

    qtbot.mouseClick(window.paint_toolbar.button(Tool.RECTANGLE), Qt.LeftButton)
    assert window.canvas.tool == Tool.PAN
    assert window.paint_toolbar.size_slider.isEnabled()

    window.resize(1000, 700)
    window.show()
    qtbot.waitExposed(window)
    assert window.paint_toolbar.size_caption.text() == "Brush Size"
    assert window.paint_toolbar_host.width() <= 72


def test_brush_size_can_be_typed(window, qtbot):
    edit = window.paint_toolbar.size_edit
    assert window.paint_toolbar.size_caption.text() == "Brush Size"
    window.show()
    qtbot.waitExposed(window)
    edit.setFocus()
    qtbot.waitUntil(edit.hasFocus)
    edit.selectAll()
    qtbot.keyClicks(edit, "24")
    assert window.paint_toolbar.brush_size == 24
    assert window.paint_toolbar.size_slider.value() == 24
    assert window.canvas._brush_diameter == 24

    window.paint_toolbar.size_slider.setValue(7)
    assert edit.value() == 7
    assert window.canvas._brush_diameter == 7

    window.set_brush_size(200)
    assert edit.value() == 128
    window.set_brush_size(0)
    assert edit.value() == 1

    window.set_paint_tool(Tool.FILL)
    assert not edit.isEnabled()
    assert not window.paint_toolbar.size_slider.isEnabled()


def test_color_button_is_for_the_base_image(window, qtbot, monkeypatch):
    layer = window.add_layer(LayerType.COPPER)
    assert window.selected_layer() is layer
    assert not window.paint_toolbar.color_button.isEnabled()

    window.layer_pane.select_base()
    assert window.paint_toolbar.color_button.isEnabled()

    monkeypatch.setattr(QColorDialog, "getColor", lambda *args, **kwargs: QColor(8, 7, 6, 200))
    qtbot.mouseClick(window.paint_toolbar.color_button, Qt.LeftButton)
    assert window.paint_toolbar.color == (8, 7, 6, 200)


# ---- layer tools -------------------------------------------------------------------


def test_brush_line_and_erase_share_the_brush_size(window):
    layer = window.add_layer(LayerType.COPPER)
    window.set_brush_size(1)
    window.set_paint_tool(Tool.BRUSH)
    window.paint_stroke([(10, 10), (10, 14)])
    mask = _mask(window, layer)
    assert list(mask[10:15, 10]) == [255, 255, 255, 255, 255]
    assert mask[10, 11] == 0

    window.set_brush_size(1)
    window.paint_line(20, 10, 24, 10)
    mask = _mask(window, layer)
    assert list(mask[10, 20:25]) == [255, 255, 255, 255, 255]
    assert mask[9, 22] == 0

    window.set_brush_size(3)
    window.set_paint_tool(Tool.ERASE)
    window.paint_stroke([(22, 10)])
    mask = _mask(window, layer)
    assert mask[10, 22] == 0
    assert mask[9, 22] == 0 and mask[11, 22] == 0
    assert mask[10, 20] == 255  # outside the eraser

    window.set_brush_size(5)
    window.set_paint_tool(Tool.BRUSH)
    window.paint_stroke([(40, 120)])
    window.paint_line(60, 120, 60, 120)
    mask = _mask(window, layer)
    disk = disk_mask(5)
    height, width = disk.shape
    y0, x0 = 120 - height // 2, 40 - width // 2
    brush = mask[y0 : y0 + height, x0 : x0 + width] > 0
    y1, x1 = 120 - height // 2, 60 - width // 2
    line = mask[y1 : y1 + height, x1 : x1 + width] > 0
    assert brush.shape == disk.shape
    assert np.array_equal(brush, disk)
    assert np.array_equal(line, disk)


def test_rectangle_is_an_outline_and_uses_brush_size(window):
    layer = window.add_layer(LayerType.COPPER)
    window.set_brush_size(1)
    window.paint_rectangle(50, 20, 20, 50)
    mask = _mask(window, layer)
    assert mask[20, 35] == 255 and mask[35, 20] == 255
    assert mask[35, 35] == 0
    assert mask[35, 18] == 0

    window.set_brush_size(7)
    window.paint_rectangle(20, 80, 50, 110)
    mask = _mask(window, layer)
    assert mask[95, 35] == 0
    assert mask[80, 35] == 255
    assert mask[77, 35] == 255
    assert mask[75, 35] == 0


def test_fill_ignores_brush_size_and_stops_at_mask_walls(window):
    layer = window.add_layer(LayerType.COPPER)
    window.inspector.apply_color(COPPER_COLOR)
    window.flush_recompute()
    window.set_brush_size(1)
    window.set_paint_tool(Tool.FILL)
    window.paint_fill(5, 5)  # empty margin, connected around the artwork

    mask = _mask(window, layer)
    assert mask[5, 5] == 255
    assert mask[0, 0] == 255  # a 1px brush would not reach the corner
    assert mask[29, 60] == 255
    assert mask[55, 55] == 0  # hole inside the copper, not connected
    assert mask[40, 40] == 255

    window.paint_fill(55, 55)
    assert mask[50, 50] == 255 and mask[69, 69] == 255
    assert mask[49, 55] == 255  # copper around the hole stays on
    # The fill did not pour back out into a fresh empty pixel beyond the hole.
    # (5, 5) was already filled by the first click; check a pixel that is on
    # only because it is copper, which the hole fill must not have to change.
    assert mask[40, 40] == 255


def test_layer_paint_survives_tolerance_and_does_not_need_a_color(window):
    layer = window.add_layer(LayerType.COPPER)
    window.set_brush_size(1)
    window.set_paint_tool(Tool.BRUSH)
    window.paint_stroke([(5, 5)])
    assert layer.has_color is False
    assert _mask(window, layer)[5, 5] == 255
    assert layer in window.project.grouped_for_export()[(LayerType.COPPER, Side.FRONT)]

    window.inspector.apply_color(COPPER_COLOR)
    window.flush_recompute()
    assert _mask(window, layer)[40, 40] == 255  # matched copper
    assert _mask(window, layer)[5, 5] == 255  # still the painted pixel

    window.set_paint_tool(Tool.ERASE)
    window.paint_stroke([(40, 40)])
    layer.tolerance = 200
    layer.min_island = 2  # would delete a 1px island if paint were applied first
    window.recompute_layer(layer)
    assert _mask(window, layer)[40, 40] == 0
    assert _mask(window, layer)[5, 5] == 255
    assert _mask(window, layer)[41, 40] == 255


def test_painted_pixels_update_a_negative_and_duplicate(window):
    layer = window.add_layer(LayerType.COPPER)
    window.inspector.apply_color(COPPER_COLOR)
    window.flush_recompute()
    window.set_brush_size(1)
    window.paint_stroke([(15, 15)])
    negative = window.generate_negative([layer.id], LayerType.MASK)
    window.flush_recompute()
    assert _mask(window, negative)[15, 15] == 0
    assert _mask(window, negative)[16, 15] == 255

    window.layer_pane.select_layer(layer.id)
    copy = window.duplicate_layer(layer.id)
    window.flush_recompute()
    assert copy.manual == layer.manual
    assert _mask(window, copy)[15, 15] == 255


def test_cut_layer_can_be_painted(window):
    cut = window.add_layer(LayerType.CUT)
    cut.smoothness = 0
    window.paint_rectangle(20, 20, 70, 70)
    assert _mask(window, cut)[20, 45] == 255
    assert _mask(window, cut)[45, 45] == 0
    assert cut.id in window.outlines


def test_escape_cancels_a_rectangle_before_the_second_click(window):
    layer = window.add_layer(LayerType.COPPER)
    window.set_paint_tool(Tool.RECTANGLE)
    window.canvas._rect_anchor = (80, 10)
    window.canvas.set_stroke_preview([(80, 10), (90, 10), (90, 20), (80, 20)], closed=True)
    window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Escape, Qt.NoModifier))
    assert window.canvas._rect_anchor is None
    assert layer.id not in window.masks or _mask(window, layer)[15, 85] == 0


def test_canvas_line_drag_paints_the_layer(window, qtbot):
    layer = window.add_layer(LayerType.COPPER)
    window.set_brush_size(1)
    window.set_paint_tool(Tool.LINE)
    window.show()
    qtbot.waitExposed(window)
    view = window.canvas
    view.resetTransform()
    view.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    view.horizontalScrollBar().setValue(0)
    view.verticalScrollBar().setValue(0)

    def at(x, y):
        return view.mapFromScene(QPointF(float(x), float(y)))

    start, end = at(12, 16), at(18, 16)
    view.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, start))
    view.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, end, buttons=Qt.LeftButton))
    view.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, end, button=Qt.LeftButton))
    mask = _mask(window, layer)
    assert mask[16, 12] == 255
    assert mask[16, 18] == 255
    assert mask[16, 15] == 255


def test_rectangle_tool_commits_on_the_second_click(window, qtbot):
    layer = window.add_layer(LayerType.COPPER)
    window.set_brush_size(1)
    window.set_paint_tool(Tool.RECTANGLE)
    window.show()
    qtbot.waitExposed(window)
    view = window.canvas
    view.resetTransform()
    view.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    view.horizontalScrollBar().setValue(0)
    view.verticalScrollBar().setValue(0)

    def at(x, y):
        return view.mapFromScene(QPointF(float(x), float(y)))

    first, second = at(20, 30), at(40, 50)
    view.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, first))
    view.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, first, button=Qt.LeftButton))
    assert view._rect_anchor == (20, 30)
    assert layer.id not in window.masks

    view.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, second, button=Qt.NoButton, buttons=Qt.NoButton))
    view.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, second))
    mask = _mask(window, layer)
    assert mask[30, 20] == 255 and mask[30, 40] == 255 and mask[50, 30] == 255
    assert mask[40, 30] == 0
    assert view._rect_anchor is None


def test_brush_cursor_matches_the_diameter_and_hides_for_fill(window, qtbot):
    window.set_brush_size(9)
    window.set_paint_tool(Tool.BRUSH)
    window.show()
    qtbot.waitExposed(window)
    view = window.canvas
    view.resetTransform()
    view.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    view.horizontalScrollBar().setValue(0)
    view.verticalScrollBar().setValue(0)
    view.mouseMoveEvent(
        _mouse(QEvent.Type.MouseMove, view.mapFromScene(QPointF(40.0, 40.0)), button=Qt.NoButton, buttons=Qt.NoButton)
    )
    cursor = view._brush_cursor
    assert cursor is not None and cursor.isVisible()
    rect = cursor.rect()
    assert (rect.x(), rect.y(), rect.width(), rect.height()) == (36, 36, 9, 9)

    window.set_paint_tool(Tool.FILL)
    assert not view._brush_cursor.isVisible()


def _mouse(kind, pos, button=Qt.LeftButton, buttons=None):
    if buttons is None:
        buttons = Qt.NoButton if kind == QEvent.Type.MouseButtonRelease else button
    point = QPointF(pos)
    return QMouseEvent(kind, point, point, button, buttons, Qt.NoModifier)


# ---- base image --------------------------------------------------------------------


def test_base_image_brush_erase_and_fill_use_the_paint_color(window):
    layer = window.add_layer(LayerType.COPPER)
    window.inspector.apply_color(COPPER_COLOR)
    window.flush_recompute()
    assert _mask(window, layer)[40, 40] == 255

    window.layer_pane.select_base()
    window.set_brush_size(1)
    window.set_paint_color((1, 2, 3, 255))
    window.set_paint_tool(Tool.BRUSH)
    window.paint_stroke([(40, 40)])
    assert tuple(int(channel) for channel in window.image[40, 40]) == (1, 2, 3, 255)
    # Gap closing would fill a one-pixel hole inside the copper rectangle.
    layer.close_radius = 0
    layer.min_island = 0
    window.recompute_layer(layer)
    assert _mask(window, layer)[40, 40] == 0

    window.set_paint_color((*COPPER_COLOR, 255))
    window.paint_stroke([(5, 5)])
    assert tuple(int(channel) for channel in window.image[5, 5]) == (*COPPER_COLOR, 255)
    assert _mask(window, layer)[5, 5] == 255

    window.set_paint_tool(Tool.ERASE)
    window.paint_stroke([(41, 40)])
    assert window.image[40, 41, 3] == 0
    assert _mask(window, layer)[40, 41] == 0

    window.set_paint_color((9, 9, 9, 255))
    window.set_brush_size(64)
    window.set_paint_tool(Tool.FILL)
    window.paint_fill(120, 90)  # inside the silkscreen rectangle
    assert tuple(int(channel) for channel in window.image[80, 100]) == (9, 9, 9, 255)
    assert tuple(int(channel) for channel in window.image[109, 159]) == (9, 9, 9, 255)
    assert tuple(int(channel) for channel in window.image[79, 100, :3]) != (9, 9, 9)
    assert tuple(int(channel) for channel in window.image[90, 120, :3]) != SILK_COLOR


def test_base_image_line_and_polygon_use_the_paint_color(window):
    window.layer_pane.select_base()
    window.set_brush_size(1)
    window.set_paint_color((4, 5, 6, 255))
    window.paint_line(2, 2, 4, 2)
    assert tuple(int(channel) for channel in window.image[2, 2]) == (4, 5, 6, 255)
    assert tuple(int(channel) for channel in window.image[2, 4]) == (4, 5, 6, 255)
    assert window.image[3, 3, 3] == 0

    window.paint_rectangle(12, 120, 20, 128)
    assert tuple(int(channel) for channel in window.image[120, 16]) == (4, 5, 6, 255)
    assert tuple(int(channel) for channel in window.image[124, 16, :3]) != (4, 5, 6)
    assert window.image[124, 8, 3] == 0


def test_base_paint_survives_flatten_and_background_removal(window):
    window.layer_pane.select_base()
    window.set_brush_size(1)
    window.set_paint_tool(Tool.BRUSH)
    window.set_paint_color((7, 8, 9, 255))
    window.paint_stroke([(40, 40)])

    image = window.active_image()
    image.flatten_enabled = True
    window._rebuild_working(image)
    assert tuple(window.image[40, 40]) == (7, 8, 9, 255)

    image.flatten_enabled = False
    image.background_removed = True
    window._rebuild_working(image)
    # The fixture's opaque artwork starts at (10, 10), so the painted source
    # pixel slides up-left by that crop.
    assert tuple(window.image[30, 30]) == (7, 8, 9, 255)


def test_paint_roundtrips_through_the_project_file(window, qtbot, tmp_path):
    layer = window.add_layer(LayerType.COPPER)
    window.set_brush_size(1)
    window.paint_stroke([(6, 7)])
    window.layer_pane.select_base()
    window.set_paint_color((3, 4, 5, 255))
    window.paint_stroke([(8, 9)])

    path = window.save_project(tmp_path / "art.prj")
    restored = MainWindow()
    qtbot.addWidget(restored)
    restored.open_project(path)
    loaded = next(item for item in restored.project.layers if item.name == layer.name)
    assert restored.masks[loaded.id][7, 6] == 255
    assert tuple(restored.image[9, 8]) == (3, 4, 5, 255)
