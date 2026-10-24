"""Tests for color masking and its cleanup controls."""

import numpy as np

from png_to_gerber.masking import (
    build_mask,
    color_distance,
    fill_enclosed_holes,
    invert_mask,
)

from conftest import COPPER_COLOR, MASK_COLOR, SILK_COLOR


def test_color_distance_zero_for_exact_match():
    img = np.zeros((2, 2, 4), dtype=np.uint8)
    img[..., :3] = (10, 20, 30)
    dist = color_distance(img, (10, 20, 30))
    assert np.allclose(dist, 0)


def test_color_distance_monotonic():
    img = np.zeros((1, 2, 4), dtype=np.uint8)
    img[0, 0, :3] = (100, 100, 100)
    img[0, 1, :3] = (200, 200, 200)
    dist = color_distance(img, (90, 90, 90))
    assert dist[0, 0] < dist[0, 1]


def test_build_mask_exact_color(fixture_rgba):
    mask = build_mask(fixture_rgba, SILK_COLOR, tolerance=5, close_radius=0)
    assert mask[90, 120] == 255  # inside silkscreen rect
    assert mask[20, 20] == 0  # board region not matched


def test_build_mask_excludes_transparent_pixels(fixture_rgba):
    # Transparent background is RGB(0,0,0); silk color is near-black, so a
    # generous tolerance would otherwise swallow the whole background.
    mask = build_mask(fixture_rgba, SILK_COLOR, tolerance=40, close_radius=0)
    assert mask[0, 0] == 0
    assert mask[90, 120] == 255


def test_black_does_not_match_nearly_transparent_pixels():
    """Opacity is part of the match, so a black layer ignores faint pixels.

    Low-alpha pixels are frequently stored near black. They used to fall
    inside a normal tolerance of opaque black; solid black and a mostly
    opaque edge of that color still match.
    """
    img = np.zeros((2, 4, 4), dtype=np.uint8)
    img[..., 3] = 255
    img[0, 0, :3] = (0, 0, 0)  # opaque black
    img[0, 1, :3] = (0, 0, 0)
    img[0, 1, 3] = 8  # nearly transparent, stored as black
    img[0, 2, :3] = (12, 4, 6)
    img[0, 2, 3] = 20  # faint premultiplied fringe of some other color
    img[0, 3, :3] = (0, 0, 0)
    img[0, 3, 3] = 220  # mostly opaque black edge
    img[1, 0, :3] = (21, 15, 23)  # opaque near-black artwork
    img[1, 1, :3] = (2, 3, 6)
    img[1, 1, 3] = 10  # within RGB tolerance of the artwork color, but faint
    img[1, 2, :3] = (10, 10, 10)  # opaque, within tolerance
    img[1, 3, :3] = (255, 0, 0)  # opaque red

    mask = build_mask(img, (0, 0, 0), tolerance=35, close_radius=0, min_island=0)
    assert mask[0, 0] == 255
    assert mask[0, 1] == 0
    assert mask[0, 2] == 0
    assert mask[0, 3] == 255
    assert mask[1, 2] == 255
    assert mask[1, 3] == 0

    artwork = build_mask(
        img, (21, 15, 23), tolerance=35, close_radius=0, min_island=0
    )
    assert artwork[1, 0] == 255
    assert artwork[1, 1] == 0


def test_build_mask_include_transparent_opt_in(fixture_rgba):
    mask = build_mask(
        fixture_rgba, (0, 0, 0), tolerance=10, close_radius=0,
        min_island=0, include_transparent=True,
    )
    assert mask[0, 0] == 255


def test_tolerance_closes_near_color_gaps(fixture_rgba):
    """Near-color 'anti-aliased' pixels must join the mask within tolerance."""
    strict = build_mask(fixture_rgba, MASK_COLOR, tolerance=0, close_radius=0)
    assert strict[30, 158] == 0  # the off-color pixel is excluded when strict
    tolerant = build_mask(fixture_rgba, MASK_COLOR, tolerance=15, close_radius=0)
    assert tolerant[30, 158] == 255  # ...and included with normal tolerance
    region = tolerant[30:60, 100:160]
    assert region.min() == 255, "mask region should be gap-free"


def test_close_radius_fills_pinholes():
    img = np.zeros((20, 20, 4), dtype=np.uint8)
    img[..., 3] = 255
    img[5:15, 5:15, :3] = (200, 0, 0)
    img[10, 10, :3] = (0, 200, 0)  # single-pixel pinhole
    no_close = build_mask(img, (200, 0, 0), tolerance=5, close_radius=0)
    assert no_close[10, 10] == 0
    closed = build_mask(img, (200, 0, 0), tolerance=5, close_radius=1)
    assert closed[10, 10] == 255


def test_min_island_removes_specks():
    img = np.zeros((30, 30, 4), dtype=np.uint8)
    img[..., 3] = 255
    img[5:15, 5:15, :3] = (0, 0, 250)  # 100 px blob
    img[25, 25, :3] = (0, 0, 250)  # 1 px speck
    kept = build_mask(img, (0, 0, 250), tolerance=5, close_radius=0, min_island=0)
    assert kept[25, 25] == 255
    cleaned = build_mask(img, (0, 0, 250), tolerance=5, close_radius=0, min_island=4)
    assert cleaned[25, 25] == 0
    assert cleaned[10, 10] == 255


def test_fill_holes_option(fixture_rgba):
    without = build_mask(fixture_rgba, COPPER_COLOR, tolerance=10, close_radius=0)
    assert without[60, 60] == 0  # the board-colored hole stays open
    filled = build_mask(
        fixture_rgba, COPPER_COLOR, tolerance=10, close_radius=0, fill_holes=True
    )
    assert filled[60, 60] == 255


def test_fill_enclosed_holes_keeps_open_regions_open():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:18, 2:18] = 255
    mask[5:10, 5:10] = 0  # enclosed hole
    mask[0:4, 8:12] = 0  # notch open to the border (already background)
    filled = fill_enclosed_holes(mask)
    assert filled[7, 7] == 255  # hole filled
    assert filled[0, 10] == 0  # border background untouched


def test_empty_mask_when_no_color_matches(fixture_rgba):
    mask = build_mask(fixture_rgba, (255, 0, 255), tolerance=5, close_radius=0)
    assert mask.max() == 0


def test_invert_mask_flips_opaque_pixels_only():
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    rgba[0, :, 3] = 0  # transparent row stays empty
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    inverted = invert_mask(rgba, mask)
    assert inverted[0, 1] == 0
    assert inverted[1, 1] == 0
    assert inverted[3, 0] == 255
