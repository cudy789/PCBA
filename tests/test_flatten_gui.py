"""Flatten controls in the base-image panel."""

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from png_to_gerber.flatten import auto_cutoffs, even_cutoffs, histogram
from png_to_gerber.gui.cutoff_bar import CutoffBar
from png_to_gerber.gui.main_window import MainWindow
from png_to_gerber.model import LayerType


def _gradient(width=60, height=30) -> np.ndarray:
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[..., 3] = 255
    ramp = np.linspace(0, 255, width, dtype=np.uint8)
    image[..., 0] = ramp
    image[..., 1] = ramp
    image[..., 2] = ramp
    return image


def _unique(rgba: np.ndarray) -> int:
    opaque = rgba[rgba[..., 3] > 0][:, :3]
    return len(np.unique(opaque, axis=0))


@pytest.fixture
def window(qtbot, tmp_path):
    path = tmp_path / "gradient.png"
    Image.fromarray(_gradient(), "RGBA").save(path)
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(path)
    return win


def _mouse(widget, kind, x, y, buttons):
    event = QMouseEvent(
        kind,
        QPointF(x, y),
        widget.mapToGlobal(QPoint(int(x), int(y))),
        Qt.LeftButton,
        buttons,
        Qt.NoModifier,
    )
    if kind == QEvent.Type.MouseButtonPress:
        widget.mousePressEvent(event)
    elif kind == QEvent.Type.MouseMove:
        widget.mouseMoveEvent(event)
    else:
        widget.mouseReleaseEvent(event)


def test_notches_slide_and_cannot_cross(qtbot):
    bar = CutoffBar()
    bar.resize(300, 80)
    bar.set_state([80, 160], [(0, 0, 0), (128, 128, 128), (255, 255, 255)], np.arange(256))
    qtbot.addWidget(bar)
    bar.show()
    qtbot.waitExposed(bar)
    bar.repaint()
    seen = []
    bar.cutoffs_changed.connect(seen.append)

    y = CutoffBar.TRACK_TOP + 4
    start = bar.luminance_to_x(80)
    target = bar.luminance_to_x(200)
    _mouse(bar, QEvent.Type.MouseButtonPress, start, y, Qt.LeftButton)
    _mouse(bar, QEvent.Type.MouseMove, target, y, Qt.LeftButton)
    _mouse(bar, QEvent.Type.MouseButtonRelease, target, y, Qt.NoButton)
    assert bar.cutoffs == [159, 160]
    assert seen[-1] == [159, 160]


def test_flatten_checkbox_reduces_the_working_image(window):
    before = window.image.copy()
    assert not window.base_panel.cutoff_bar.isEnabled()
    window.base_panel.flatten_check.setChecked(True)
    assert window.project.images[0].flatten_enabled
    assert window.base_panel.cutoff_bar.isEnabled()
    assert window.base_panel.count_spin.isEnabled()
    assert window.base_panel.swatches[0].isEnabled()
    assert not np.array_equal(window.image, before)
    assert _unique(window.image) <= 4
    assert _unique(before) > 4


def test_color_count_changes_notches_swatches_and_keeps_overrides(window):
    window.set_flatten(True, colors=[(1, 2, 3), (4, 5, 6), None, None])
    assert len(window.base_panel.swatches) == 4
    assert len(window.base_panel.cutoff_bar.cutoffs) == 3
    window.base_panel.count_spin.setValue(3)
    assert window.project.images[0].flatten_colors == [(1, 2, 3), (4, 5, 6), None]
    assert len(window.base_panel.swatches) == 3
    assert len(window.base_panel.cutoff_bar.cutoffs) == 2
    window.base_panel.count_spin.setValue(6)
    assert window.project.images[0].flatten_colors[:3] == [(1, 2, 3), (4, 5, 6), None]
    assert window.project.images[0].flatten_colors[3:] == [None, None, None]
    assert len(window.base_panel.swatches) == 6
    assert len(window.base_panel.cutoff_bar.cutoffs) == 5


def test_right_click_restores_the_bucket_average(window):
    window.set_flatten(True, colors=[(1, 2, 3), None, None, None])
    assert np.any(np.all(window.image[..., :3] == (1, 2, 3), axis=-1))
    window.base_panel.swatches[0].customContextMenuRequested.emit(QPoint(0, 0))
    assert window.project.images[0].flatten_colors[0] is None
    assert not np.any(np.all(window.image[..., :3] == (1, 2, 3), axis=-1))


def test_compare_original_swaps_the_canvas_but_not_the_mask(window):
    window.base_panel.flatten_check.setChecked(True)
    window.flush_recompute()
    flat = window.image.copy()
    color = tuple(int(channel) for channel in flat[0, 0, :3])
    assert color != (0, 0, 0)

    layer = window.add_layer(LayerType.COPPER)
    window.inspector.apply_color(color)
    layer.tolerance = 0
    layer.close_radius = 0
    layer.min_island = 0
    window.flush_recompute()
    mask = window.masks[layer.id].copy()
    assert mask[0, 0]

    window.base_panel.compare_check.setChecked(True)
    window.flush_recompute()
    assert np.array_equal(window.image, flat)
    assert np.array_equal(window.masks[layer.id], mask)
    shown = window.canvas.base_item.pixmap().toImage().pixelColor(0, 0)
    assert (shown.red(), shown.green(), shown.blue()) == (0, 0, 0)


def test_panel_follows_programmatic_flatten_settings(window):
    window.set_flatten(True, count=3, cutoffs=[50, 150], smooth=2)
    assert window.base_panel.flatten_check.isChecked()
    assert window.base_panel.count_spin.value() == 3
    assert window.base_panel.cutoff_bar.cutoffs == [50, 150]
    assert len(window.base_panel.swatches) == 3
    assert window.base_panel.smooth_slider.value() == 2
    assert window.base_panel.smooth_value.text() == "2"


def test_auto_and_reset_buttons_move_the_cutoffs(window):
    window.set_flatten(True, count=2, cutoffs=[30], colors=[(9, 9, 9), None])
    window.base_panel.auto_button.click()
    prepared = window.original_image
    assert window.project.images[0].flatten_cutoffs == auto_cutoffs(histogram(prepared), 2)
    assert window.project.images[0].flatten_colors[0] == (9, 9, 9)

    window.set_flatten(cutoffs=[30, 90, 200], count=4)
    window.base_panel.reset_button.click()
    assert window.project.images[0].flatten_cutoffs == even_cutoffs(4)
    assert window.project.images[0].flatten_colors[0] == (9, 9, 9)


def test_save_and_open_restores_flatten(qtbot, window, tmp_path):
    window.set_flatten(
        True,
        count=5,
        cutoffs=[20, 40, 80, 160],
        colors=[(9, 8, 7), None, (4, 5, 6), None, None],
        smooth=3,
    )
    before = window.image.copy()
    path = window.save_project(tmp_path / "flat.prj")
    fresh = MainWindow()
    qtbot.addWidget(fresh)
    fresh.open_project(path)
    image = fresh.project.images[0]
    assert image.flatten_enabled is True
    assert image.flatten_count == 5
    assert image.flatten_cutoffs == [20, 40, 80, 160]
    assert image.flatten_colors == [(9, 8, 7), None, (4, 5, 6), None, None]
    assert image.flatten_smooth == 3
    assert fresh.base_panel.flatten_check.isChecked()
    assert fresh.base_panel.count_spin.value() == 5
    assert fresh.base_panel.smooth_slider.value() == 3
    assert fresh.base_panel.cutoff_bar.cutoffs == [20, 40, 80, 160]
    assert np.array_equal(fresh.image, before)


def test_replacing_an_image_resets_flatten(window, tmp_path):
    window.set_flatten(True, count=3, smooth=2)
    other = tmp_path / "other.png"
    Image.fromarray(_gradient(20, 20), "RGBA").save(other)
    window.load_image(other)
    image = window.project.images[0]
    assert image.flatten_enabled is False
    assert image.flatten_count == 4
    assert image.flatten_cutoffs == [64, 128, 192]
    assert image.flatten_smooth == 0
    assert not window.base_panel.flatten_check.isChecked()
