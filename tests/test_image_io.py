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


def test_dominant_colors_match_bin_means():
    """Each swatch is the rounded mean of the opaque pixels in that quantized bin."""
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(40, 50, 4), dtype=np.uint8)
    img[:5, :, 3] = 0
    quantize = 8
    colors = dominant_colors(img, count=6, quantize=quantize)
    flat = img.reshape(-1, 4)
    opaque = flat[flat[:, 3] > 0][:, :3]
    bins = (256 + quantize - 1) // quantize
    quantized = (opaque.astype(np.uint16) // quantize).astype(np.int32)
    keys = (quantized[:, 0] * bins + quantized[:, 1]) * bins + quantized[:, 2]
    _unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    sums = np.column_stack(
        [
            np.bincount(inverse, weights=opaque[:, channel].astype(np.float64), minlength=len(counts))
            for channel in range(3)
        ]
    )
    order = np.argsort(-counts, kind="stable")[:6]
    for (rgb, count), idx in zip(colors, order):
        mean = np.round(sums[idx] / counts[idx]).astype(int)
        assert rgb == (int(mean[0]), int(mean[1]), int(mean[2]))
        assert count == int(counts[idx])


def test_dominant_colors_empty_for_fully_transparent():
    img = np.zeros((10, 10, 4), dtype=np.uint8)
    assert dominant_colors(img) == []
