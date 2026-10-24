"""Color flattening: brightness buckets, overrides, and edge smoothing."""

import numpy as np

from png_to_gerber.background import working_image
from png_to_gerber.flatten import (
    auto_cutoffs,
    bucket_means,
    even_cutoffs,
    flatten_image,
    histogram,
    luminance,
)


def _opaque(h, w, rgb) -> np.ndarray:
    image = np.zeros((h, w, 4), dtype=np.uint8)
    image[..., :3] = rgb
    image[..., 3] = 255
    return image


def _bands() -> np.ndarray:
    """Three horizontal bands whose brightness falls in three different buckets."""
    image = np.zeros((30, 10, 4), dtype=np.uint8)
    image[..., 3] = 255
    image[0:10, :, :3] = (80, 0, 0)  # luminance ~17
    image[10:20, :, :3] = (0, 200, 0)  # luminance ~143
    image[20:30, :, :3] = (255, 255, 255)  # luminance 255
    return image


CUTOFFS = [100, 180]


def test_even_cutoffs_for_four_colors():
    assert even_cutoffs(4) == [64, 128, 192]
    assert even_cutoffs(2) == [128]


def test_pixels_take_their_bucket_mean():
    out = flatten_image(_bands(), CUTOFFS)
    assert tuple(out[5, 5, :3]) == (80, 0, 0)
    assert tuple(out[15, 5, :3]) == (0, 200, 0)
    assert tuple(out[25, 5, :3]) == (255, 255, 255)
    assert len(np.unique(out[..., :3].reshape(-1, 3), axis=0)) == 3


def test_swatch_override_replaces_the_bucket_mean():
    out = flatten_image(_bands(), CUTOFFS, overrides=[None, (1, 2, 3), None])
    assert tuple(out[15, 5, :3]) == (1, 2, 3)
    assert tuple(out[5, 5, :3]) == (80, 0, 0)
    assert tuple(out[25, 5, :3]) == (255, 255, 255)


def test_transparent_pixels_are_left_alone_and_input_is_not_mutated():
    image = _bands()
    image[0, 0, 3] = 0
    image[0, 0, :3] = (1, 2, 3)
    before = image.copy()
    out = flatten_image(image, CUTOFFS)
    assert np.array_equal(image, before)
    assert out[0, 0, 3] == 0
    assert tuple(out[0, 0, :3]) == (1, 2, 3)
    assert out[5, 5, 3] == 255


def test_transparent_pixels_do_not_affect_the_mean_or_histogram():
    image = _opaque(4, 4, (10, 10, 10))
    image[0, 0] = (255, 255, 255, 0)
    means = bucket_means(image, [128])
    assert means[0] == (10, 10, 10)
    assert histogram(image)[10] == 15
    assert histogram(image)[255] == 0
    assert int(luminance(image)[1, 1]) == 10


def test_empty_bucket_falls_back_to_midpoint_gray():
    image = _opaque(4, 4, (10, 10, 10))
    means = bucket_means(image, [80, 180])
    assert means[0] == (10, 10, 10)
    assert means[1] == (129, 129, 129)  # values 80..179
    assert means[2] == (217, 217, 217)  # values 180..255


def test_auto_cutoffs_sit_in_the_valley_of_a_bimodal_histogram():
    hist = np.zeros(256, dtype=np.int64)
    hist[20:40] = 200
    hist[210:230] = 200
    cutoff = auto_cutoffs(hist, 2)[0]
    assert 40 < cutoff < 210


def test_auto_cutoffs_fall_back_to_even_spacing_without_enough_peaks():
    hist = np.zeros(256, dtype=np.int64)
    hist[100:120] = 80
    assert auto_cutoffs(hist, 4) == [64, 128, 192]


def test_edge_smooth_removes_a_speckle_without_new_colors():
    image = _opaque(9, 9, (200, 200, 200))
    image[4, 4, :3] = (10, 10, 10)
    raw = flatten_image(image, [100], smooth=0)
    smoothed = flatten_image(image, [100], smooth=1)
    assert tuple(raw[4, 4, :3]) != tuple(raw[0, 0, :3])
    assert np.all(smoothed[..., :3] == smoothed[0, 0, :3])
    raw_colors = {tuple(color) for color in np.unique(raw[..., :3].reshape(-1, 3), axis=0)}
    smooth_colors = {tuple(color) for color in np.unique(smoothed[..., :3].reshape(-1, 3), axis=0)}
    assert smooth_colors <= raw_colors
    assert len(smooth_colors) == 1


def test_working_image_flattens_after_the_background_is_cropped():
    """The bright artwork band shares a bucket with white, so flattening first
    would recolor it and background removal would crop it away."""
    image = _opaque(10, 30, (255, 255, 255))
    image[:, 5:10, :3] = (30, 30, 30)
    image[:, 10:15, :3] = (100, 100, 100)
    image[:, 15:20, :3] = (180, 180, 180)
    image[:, 20:25, :3] = (230, 230, 230)
    out = working_image(image, True, flatten=True, cutoffs=[64, 128, 192])
    assert out.shape == (10, 20, 4)
    assert int(out[..., 3].min()) == 255
    assert len(np.unique(out[..., :3].reshape(-1, 3), axis=0)) == 4
    assert tuple(out[0, 0, :3]) == (30, 30, 30)
    assert tuple(out[0, -1, :3]) == (230, 230, 230)
