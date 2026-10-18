"""Tests for polygon conversion and cut outline smoothing."""

import numpy as np
import pytest
from shapely.geometry import Polygon

from png_to_gerber.geometry import CutError, cut_outline, iter_polygons, mask_to_geometry
from png_to_gerber.masking import build_mask

from conftest import BOARD_COLOR, COPPER_COLOR


def square_mask(size=100, inset=10) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[inset : size - inset, inset : size - inset] = 255
    return mask


def test_mask_to_geometry_empty():
    geom = mask_to_geometry(np.zeros((10, 10), dtype=np.uint8))
    assert geom.is_empty


def test_mask_to_geometry_square_area():
    geom = mask_to_geometry(square_mask(), simplify_px=0)
    polys = list(iter_polygons(geom))
    assert len(polys) == 1
    # 80x80 filled block; contour tracing is within a pixel of that.
    assert polys[0].area == pytest.approx(80 * 80, rel=0.05)


def test_mask_to_geometry_retains_holes(fixture_rgba):
    mask = build_mask(fixture_rgba, COPPER_COLOR, tolerance=10, close_radius=0)
    geom = mask_to_geometry(mask)
    polys = list(iter_polygons(geom))
    assert len(polys) == 1
    assert len(polys[0].interiors) == 1  # the board-colored hole


def test_mask_to_geometry_multiple_regions():
    mask = np.zeros((50, 100), dtype=np.uint8)
    mask[10:40, 10:40] = 255
    mask[10:40, 60:90] = 255
    geom = mask_to_geometry(mask)
    assert len(list(iter_polygons(geom))) == 2


def test_cut_outline_empty_mask_raises():
    with pytest.raises(CutError):
        cut_outline(np.zeros((10, 10), dtype=np.uint8))


def test_cut_outline_keeps_largest_region_only():
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[10:90, 10:120] = 255  # large region
    mask[20:40, 150:190] = 255  # smaller separate region
    outline = cut_outline(mask, smoothness=0)
    assert isinstance(outline, Polygon)
    minx, _, maxx, _ = outline.bounds
    assert maxx <= 130, "outline must only trace the largest region"


def test_cut_outline_discards_holes(fixture_rgba):
    """A cut outline is a single perimeter even if the color region has holes."""
    mask = build_mask(fixture_rgba, BOARD_COLOR, tolerance=10, close_radius=0)
    outline = cut_outline(mask, smoothness=0)
    assert len(outline.interiors) == 0


def test_cut_outline_is_closed_and_valid(fixture_rgba):
    mask = build_mask(fixture_rgba, BOARD_COLOR, tolerance=10, close_radius=0)
    for smoothness in (0, 25, 50, 100):
        outline = cut_outline(mask, smoothness=smoothness)
        assert outline.is_valid
        coords = list(outline.exterior.coords)
        assert coords[0] == coords[-1], "perimeter must be closed"
        assert outline.area > 0


def test_smoothing_reduces_jaggedness():
    """A staircase edge gets shorter (smoother) as smoothness rises."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[20:180, 20:180] = 255
    # Add a jagged staircase along the right edge.
    for i, y in enumerate(range(20, 180)):
        mask[y, 180 : 185 + (i % 7)] = 255
    raw = cut_outline(mask, smoothness=0)
    smooth = cut_outline(mask, smoothness=60)
    assert smooth.exterior.length < raw.exterior.length
    # Smoothed outline must still cover roughly the same region.
    assert smooth.area == pytest.approx(raw.area, rel=0.15)


def test_smoothness_slider_monotonic_simplification():
    mask = np.zeros((150, 150), dtype=np.uint8)
    mask[10:140, 10:140] = 255
    for y in range(10, 140, 2):
        mask[y, 140:143] = 255  # fine-toothed comb edge
    lengths = [
        cut_outline(mask, smoothness=s).exterior.length for s in (0, 30, 90)
    ]
    assert lengths[2] <= lengths[1] <= lengths[0]
