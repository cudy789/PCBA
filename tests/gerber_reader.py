"""Minimal Gerber region reader used to verify exported files.

It replays a file the way KiCad's Gerber viewer does (gerbview, rs274d.cpp):

- Every contour is filled solid. A D02 inside a region ends the current
  polygon and the next contour is a new solid polygon — an inner contour is
  NOT a hole.
- Dark polarity (%LPD*%) adds that solid polygon. Clear polarity (%LPC*%)
  subtracts it, which is how a hole has to be expressed for GerbView.
- Later dark polygons paint over earlier clears, so an island inside a hole
  survives only if it is written after the clear.

The result is a shapely geometry in *image pixel* coordinates (the inverse of
BoardTransform), so it can be compared pixel-for-pixel against the mask shown
in the app.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon

COORD = re.compile(r"X(-?\d+)Y(-?\d+)D0([123])\*")


def _solid(contour: list[tuple[float, float]]):
    """Fill one contour as a solid polygon, the way GerbView fills a region."""
    if len(contour) < 3:
        return Polygon()
    poly = Polygon(contour)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def read_regions(path: Path, image_w: int, image_h: int, width_mm: float, height_mm: float):
    """Return the image painted by all regions in `path` as a shapely geometry (px coords)."""
    text = Path(path).read_text(encoding="ascii")
    assert "%FSLAX46Y46*%" in text and "%MOMM*%" in text

    def to_px(xi: int, yi: int) -> tuple[float, float]:
        x_mm, y_mm = xi / 1e6, yi / 1e6
        return x_mm * image_w / width_mm, image_h - y_mm * image_h / height_mm

    pieces: list[tuple[str, Polygon]] = []
    polarity = "D"
    in_region = False
    ring: list[tuple[float, float]] = []

    def paint(contour: list[tuple[float, float]]) -> None:
        region = _solid(contour)
        if region.is_empty:
            return
        pieces.append((polarity, region))

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("%LP"):
            polarity = line[3]
        elif line == "G36*":
            in_region, ring = True, []
        elif line == "G37*":
            in_region = False
            paint(ring)
            ring = []
        elif in_region:
            m = COORD.fullmatch(line)
            if not m:
                continue
            point = to_px(int(m.group(1)), int(m.group(2)))
            if m.group(3) == "2":  # D02 ends the current polygon and starts another
                paint(ring)
                ring = [point]
            else:
                ring.append(point)
    return _apply_polarity(pieces)


def _apply_polarity(pieces: list[tuple[str, Polygon]]):
    """Replay dark and clear regions.

    A run of the same polarity is one union. That matches painting them one
    at a time, and it stays cheap when a layer is hundreds of separate shapes.
    """
    image = Polygon()
    start = 0
    count = len(pieces)
    while start < count:
        polarity = pieces[start][0]
        end = start + 1
        while end < count and pieces[end][0] == polarity:
            end += 1
        batch = [region for _, region in pieces[start:end]]
        merged = batch[0] if len(batch) == 1 else shapely.union_all(batch)
        if not merged.is_empty:
            image = image.union(merged) if polarity == "D" else image.difference(merged)
        start = end
    return image


def rasterize(geometry, width: int, height: int) -> np.ndarray:
    """Boolean image: is each pixel center inside `geometry`?"""
    ys, xs = np.mgrid[0:height, 0:width]
    if geometry.is_empty:
        return np.zeros((height, width), dtype=bool)
    return shapely.contains_xy(geometry, xs + 0.5, ys + 0.5)


def mismatch(mask: np.ndarray, painted: np.ndarray) -> int:
    """Number of pixels where the exported file disagrees with the mask."""
    return int(((mask > 0) != painted).sum())
