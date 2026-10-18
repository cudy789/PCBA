"""Tests for image loading and dominant color extraction."""

import numpy as np

from png_to_gerber.image_io import dominant_colors, load_rgba

from conftest import BOARD_COLOR, COPPER_COLOR


def test_load_rgba_preserves_alpha(fixture_image_path):
    rgba = load_rgba(fixture_image_path)
    assert rgba.shape == (150, 200, 4)
    assert rgba.dtype == np.uint8
    assert rgba[0, 0, 3] == 0  # background stays transparent
    assert rgba[20, 20, 3] == 255


def test_load_rgba_converts_rgb_images(tmp_path):
    from PIL import Image

    path = tmp_path / "rgb.png"
    Image.new("RGB", (10, 10), (255, 0, 0)).save(path)
    rgba = load_rgba(path)
    assert rgba.shape == (10, 10, 4)
    assert rgba[0, 0].tolist() == [255, 0, 0, 255]


def test_dominant_colors_ignores_transparent(fixture_rgba):
    colors = dominant_colors(fixture_rgba)
    assert colors, "expected at least one dominant color"
    # Transparent black background must not be reported.
    assert all(rgb != (0, 0, 0) for rgb, _ in colors)
    # Board color dominates the opaque pixels.
    top_rgb, top_count = colors[0]
    assert np.linalg.norm(np.subtract(top_rgb, BOARD_COLOR)) < 10
    assert top_count > 10000


def test_dominant_colors_finds_copper(fixture_rgba):
    colors = dominant_colors(fixture_rgba, count=12)
    assert any(
        np.linalg.norm(np.subtract(rgb, COPPER_COLOR)) < 10 for rgb, _ in colors
    )


def test_dominant_colors_empty_for_fully_transparent():
    img = np.zeros((10, 10, 4), dtype=np.uint8)
    assert dominant_colors(img) == []
