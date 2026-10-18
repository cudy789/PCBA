"""Mask -> polygon conversion and smooth board-outline derivation.

Coordinates throughout are image pixel coordinates (x right, y down).
The Gerber writer handles the flip to board coordinates.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

from .model import OutlineMode


def _mask_boxes(binary: np.ndarray) -> np.ndarray:
    """Cover a boolean mask with axis-aligned boxes (x0, y0, x1, y1), pixel-exact.

    Each row's horizontal runs become boxes, and identical runs on consecutive
    rows are merged into one taller box, so large flat regions stay cheap.
    """
    h, w = binary.shape
    padded = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1:-1] = binary
    edges = np.diff(padded, axis=1)
    run_rows, run_starts = np.nonzero(edges == 1)  # first column of each run
    _, run_ends = np.nonzero(edges == -1)  # one past the last column
    # np.nonzero is row-major, so starts and ends pair up one-to-one.
    row_bounds = np.searchsorted(run_rows, np.arange(h + 1))

    boxes: list[tuple[int, int, int, int]] = []
    active: dict[tuple[int, int], int] = {}  # (x0, x1) -> first row of the box
    for r in range(h):
        lo, hi = row_bounds[r], row_bounds[r + 1]
        current = set(zip(run_starts[lo:hi].tolist(), run_ends[lo:hi].tolist()))
        for key in [k for k in active if k not in current]:
            boxes.append((key[0], active.pop(key), key[1], r))
        for key in current:
            active.setdefault(key, r)
    for key, first_row in active.items():
        boxes.append((key[0], first_row, key[1], h))
    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def mask_to_geometry(mask: np.ndarray, simplify_px: float = 0.0):
    """Convert a binary mask into valid shapely geometry, retaining holes.

    The polygons follow the *pixel edges* (corner coordinates, pixel (x, y)
    spans x..x+1, y..y+1), so the geometry covers exactly the pixels that are
    set in the mask: nothing shrinks, one-pixel-wide lines survive, and holes
    and islands of any size are kept. `simplify_px` > 0 optionally trades that
    exactness for fewer vertices.
    """
    binary = np.asarray(mask) > 0
    if not binary.any():
        return GeometryCollection()
    boxes = _mask_boxes(binary)
    merged = shapely.union_all(shapely.box(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]))
    # Drop vertices that sit exactly on a straight edge (lossless).
    merged = merged.simplify(simplify_px, preserve_topology=True)
    if not merged.is_valid:  # defensive; unions of boxes are normally valid
        merged = merged.buffer(0)
    return merged

def iter_polygons(geom):
    """Yield every Polygon inside any shapely geometry."""
    if geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, (MultiPolygon, GeometryCollection)):
        for g in geom.geoms:
            yield from iter_polygons(g)


class CutError(ValueError):
    pass


def _smooth(base: Polygon, shape: tuple[int, int], smoothness: int) -> Polygon:
    """Round off pixel steps and corners of `base`, keeping one valid closed ring."""
    outline = base
    smoothness = max(0, min(100, int(smoothness)))
    if smoothness > 0:
        diag = float(np.hypot(shape[0], shape[1]))
        radius = (smoothness / 100.0) * diag * 0.02  # up to 2% of the diagonal
        if radius > 0:
            # Closing (dilate->erode) rounds concave notches; opening
            # (erode->dilate) rounds convex spikes. Round joins keep it smooth.
            smoothed = (
                outline.buffer(radius, join_style="round")
                .buffer(-2 * radius, join_style="round")
                .buffer(radius, join_style="round")
            )
            smoothed = smoothed.simplify(radius * 0.1, preserve_topology=True)
            candidates = [p for p in iter_polygons(smoothed) if p.area > 0]
            if candidates:
                outline = Polygon(max(candidates, key=lambda p: p.area).exterior)

    if not outline.is_valid:
        outline = outline.buffer(0)
        candidates = list(iter_polygons(outline))
        if not candidates:
            raise CutError("Smoothing collapsed the outline; lower the smoothness.")
        outline = Polygon(max(candidates, key=lambda p: p.area).exterior)

    if outline.is_empty or outline.area <= 0:
        raise CutError("Smoothing collapsed the outline; lower the smoothness.")
    return outline


def cut_outline(
    mask: np.ndarray,
    smoothness: int = 20,
    mode: OutlineMode = OutlineMode.OUTER,
) -> Polygon:
    """Derive a single smooth closed board perimeter from a cut-layer mask.

    - OUTER: the external boundary of the largest region of the selected color
      (its holes are discarded - a board profile is one contiguous perimeter).
    - INNER: the boundary of the largest area enclosed by the selected color,
      e.g. the inside edge of a frame or border line drawn around the board.

    `smoothness` maps 0..100 onto rounded-corner buffering plus simplification
    scaled to the image size, so higher values progressively remove jagged
    pixel steps and round sharp corners while the perimeter stays closed and
    valid.
    """
    mode = OutlineMode(mode)
    if mask.max() == 0:
        raise CutError("The cut layer mask is empty; no outline color matched.")

    polys = list(iter_polygons(mask_to_geometry(mask, simplify_px=0.0)))
    if not polys:
        raise CutError("No closed outline could be traced from the cut layer.")

    if mode == OutlineMode.OUTER:
        # Largest by its filled footprint, so a thin frame beats a solid speck.
        largest = max(polys, key=lambda p: Polygon(p.exterior).area)
        base = Polygon(largest.exterior)
    else:
        enclosed = [Polygon(ring) for p in polys for ring in p.interiors]
        enclosed = [h for h in enclosed if h.area > 0]
        if not enclosed:
            raise CutError(
                "The cut color does not enclose an area, so there is no inner "
                "perimeter. Close gaps in the outline (raise tolerance or gap "
                "closing) or switch to the outer perimeter."
            )
        base = max(enclosed, key=lambda h: h.area)

    return _smooth(base, mask.shape, smoothness)
