"""Background removal: core algorithm and GUI toggle."""

import numpy as np
import pytest
from PIL import Image

from png_to_gerber.background import (
    background_mask,
    crop_to_opaque,
    detect_background_color,
    remove_background,
    working_image,
)
from png_to_gerber.gui.main_window import MainWindow
from png_to_gerber.model import LayerType

WHITE = (255, 255, 255)
BLUE = (20, 40, 200)


def ring_image(bg=WHITE, size=60) -> np.ndarray:
    """Opaque `bg` canvas with a blue ring whose inside is also `bg`-colored."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    img[..., :3] = bg
    img[..., 3] = 255
    img[15:45, 15:45, :3] = BLUE
    img[22:38, 22:38, :3] = bg  # enclosed, same color as the background
    return img


def test_detect_background_color_is_most_common_border_color():
    img = ring_image()
    assert detect_background_color(img) == WHITE


def test_detect_returns_none_when_border_fully_transparent():
    img = ring_image()
    img[0, :, 3] = img[-1, :, 3] = img[:, 0, 3] = img[:, -1, 3] = 0
    assert detect_background_color(img) is None


def test_removes_outer_background_only():
    out = remove_background(ring_image())
    assert out[2, 2, 3] == 0  # outer background -> transparent
    assert out[30, 18, 3] == 255  # ring stays opaque
    assert out[30, 30, 3] == 255, "enclosed same-color region must be kept"


def test_does_not_modify_input():
    img = ring_image()
    before = img.copy()
    remove_background(img)
    assert np.array_equal(img, before)


def test_removes_non_white_background():
    out = remove_background(ring_image(bg=(30, 160, 60)))
    assert out[2, 2, 3] == 0
    assert out[30, 18, 3] == 255


def test_tolerance_controls_near_color_removal():
    img = ring_image()
    img[0:10, 0:10, :3] = (245, 245, 245)  # slightly off-white corner
    strict = remove_background(img, tolerance=2)
    loose = remove_background(img, tolerance=15)
    assert strict[5, 5, 3] == 255  # too different for tight tolerance
    assert loose[5, 5, 3] == 0


def test_enclosed_background_not_removed_even_when_loose():
    out = remove_background(ring_image(), tolerance=30)
    assert out[30, 30, 3] == 255


def test_only_border_connected_regions_removed():
    """Same color as bg but cut off from the border by artwork stays opaque."""
    img = ring_image()
    mask = background_mask(img)
    assert mask[2, 2]
    assert not mask[30, 30]
    assert not mask[30, 18]


def test_no_leak_through_diagonal_gap():
    img = np.zeros((7, 7, 4), dtype=np.uint8)
    img[..., :3] = WHITE
    img[..., 3] = 255
    # Black diagonal wall with the interior pixel touching the outside only
    # diagonally; 4-connectivity must not treat it as background.
    for i in range(1, 6):
        img[1, i, :3] = img[5, i, :3] = (0, 0, 0)
        img[i, 1, :3] = img[i, 5, :3] = (0, 0, 0)
    img[1, 1, :3] = WHITE  # open corner (diagonal-only connection)
    out = remove_background(img)
    assert out[3, 3, 3] == 255


def test_already_transparent_regions_do_not_block_removal():
    img = ring_image()
    img[:, :10, 3] = 0  # left strip already transparent
    out = remove_background(img)
    assert out[2, 30, 3] == 0  # opaque bg on the other side is still reached
    assert out[30, 18, 3] == 255


def test_crop_to_opaque_drops_the_transparent_margin():
    img = ring_image()
    img[:5, :, 3] = 0
    img[-5:, :, 3] = 0
    cropped = crop_to_opaque(img)
    assert cropped.shape == (50, 60, 4)
    assert cropped[..., 3].min() > 0


def test_working_image_crops_after_background_removal():
    img = ring_image(size=80)
    cropped = working_image(img, removed=True)
    assert cropped.shape[:2] == (30, 30)  # the blue ring, 15..44
    assert (cropped[..., 3] > 0).all()
    restored = working_image(img, removed=False)
    assert restored.shape == img.shape


def test_uniform_image_is_entirely_background():
    img = ring_image()
    img[:, :, :3] = BLUE  # border color is blue and everything is connected to it
    out = remove_background(img)
    assert (out[..., 3] == 0).all()


# ---- GUI -----------------------------------------------------------------------------


@pytest.fixture
def opaque_window(qtbot, tmp_path):
    path = tmp_path / "opaque.png"
    Image.fromarray(ring_image(size=80), "RGBA").save(path)
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(path)
    return win


def test_toggle_removes_and_restores_background(opaque_window):
    win = opaque_window
    assert win.image[2, 2, 3] == 255
    win.remove_bg_action.setChecked(True)
    assert win.image.shape[:2] == (30, 30)
    assert (win.image[..., 3] > 0).all()  # transparent margin was cropped away
    assert "30 x 30" in win.base_panel.info_label.text()
    assert win.project.background_removed
    assert win.original_image[2, 2, 3] == 255  # original untouched
    win.remove_bg_action.setChecked(False)
    assert win.image.shape == win.original_image.shape
    assert win.image[2, 2, 3] == 255
    assert not win.project.background_removed


def test_layers_ignore_removed_background(opaque_window):
    win = opaque_window
    win.remove_bg_action.setChecked(True)
    layer = win.add_layer(LayerType.COPPER)
    win.inspector.apply_color(WHITE)  # the old background color
    win.flush_recompute()
    mask = win.masks[layer.id]
    white = (win.image[..., :3] == WHITE).all(axis=-1)
    assert white.sum() == 16 * 16  # only the enclosed region remains
    assert (mask[white] == 255).all()
    assert mask[~white].max() == 0  # the ring is not the removed background color


def test_existing_layers_recompute_after_toggle(opaque_window):
    win = opaque_window
    layer = win.add_layer(LayerType.COPPER)
    win.inspector.apply_color(WHITE)
    win.flush_recompute()
    assert win.masks[layer.id][2, 2] == 255
    win.remove_bg_action.setChecked(True)
    win.flush_recompute()
    white = (win.image[..., :3] == WHITE).all(axis=-1)
    assert white.sum() == 16 * 16
    assert (win.masks[layer.id][white] == 255).all()
    assert win.masks[layer.id][~white].max() == 0


def test_tolerance_spin_reapplies_when_enabled(opaque_window):
    win = opaque_window
    win.original_image[0:10, 0:10, :3] = (245, 245, 245)
    win.bg_tolerance_spin.setValue(2)
    win.remove_bg_action.setChecked(True)
    assert win.image[5, 5, 3] == 255  # off-white corner is kept, so the image is not cropped past it
    win.bg_tolerance_spin.setValue(15)
    off_white = (win.image[..., :3] == (245, 245, 245)).all(axis=-1)
    assert not off_white.any()
    assert win.image.shape[0] < win.original_image.shape[0]
    assert win.project.background_tolerance == 15


def test_toggle_without_image_is_safe(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    win.remove_bg_action.setChecked(True)
    assert not win.remove_bg_action.isChecked()
    assert not win.project.background_removed


def test_loading_new_image_resets_background_removal(opaque_window, tmp_path):
    win = opaque_window
    win.remove_bg_action.setChecked(True)
    other = tmp_path / "other.png"
    Image.fromarray(ring_image(size=80), "RGBA").save(other)
    win.load_image(other)
    assert not win.remove_bg_action.isChecked()
    assert win.image[2, 2, 3] == 255


def test_dominant_colors_ignore_removed_background(opaque_window):
    from png_to_gerber.image_io import dominant_colors

    win = opaque_window
    before_white = max(n for rgb, n in dominant_colors(win.image) if rgb == WHITE)
    win.remove_bg_action.setChecked(True)
    after_white = max(
        (n for rgb, n in dominant_colors(win.image) if rgb == WHITE), default=0
    )
    # Only the enclosed white region (16x16) is left after removal.
    assert before_white > after_white
    assert after_white == 16 * 16


def test_cropped_artwork_fills_the_board_size(opaque_window, tmp_path):
    """Gerber coordinates span the requested size once the margin is cropped away."""
    import re

    win = opaque_window
    win.remove_bg_action.setChecked(True)
    layer = win.add_layer(LayerType.COPPER)
    win.inspector.apply_color(BLUE)
    win.flush_recompute()
    written = win.export_gerbers(tmp_path, "board", 30.0, 30.0)
    coords = [
        (int(x), int(y))
        for x, y in re.findall(r"X(-?\d+)Y(-?\d+)", written[0].read_text())
    ]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    assert min(xs) == 0 and min(ys) == 0
    assert max(xs) == 30_000_000
    assert max(ys) == 30_000_000
    assert layer.id in win.masks


def test_transparent_margin_is_cropped_too(qtbot, tmp_path):
    """A PNG that is already transparent around the art still resizes."""
    img = np.zeros((40, 80, 4), dtype=np.uint8)
    img[10:30, 20:60, :3] = BLUE
    img[10:30, 20:60, 3] = 255
    path = tmp_path / "padded.png"
    Image.fromarray(img, "RGBA").save(path)
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(path)
    win.project.board.width_mm = 40.0
    win.project.board.height_mm = 20.0
    win.project.board.lock_aspect = True
    win.set_background_removed(True)
    assert win.image.shape[:2] == (20, 40)  # the blue block, margin discarded
    assert win.project.board.height_mm == 20.0  # 40mm wide, half as tall
    win.project.board.lock_aspect = False
    win.project.board.height_mm = 11.0
    win.set_background_removed(False)
    win.set_background_removed(True)
    assert win.project.board.height_mm == 11.0  # unlocked height is left alone
