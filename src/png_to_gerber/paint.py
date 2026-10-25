"""Brush strokes for layer masks and base-image pixels.

Layer edits are a sparse set of horizontal runs that force the mask on
(paint) or off (erase) after color matching, so a stroke stays put when
tolerance or gap closing changes. Base-image paint is stored in source-image
coordinates and stamped back on after background removal and flattening, so
the color the user picked is the color the layers see.

The rectangle tool strokes the outline between two corners and leaves the
interior empty. The fill tool is a flood fill and does not use the brush size.
"""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

from .background import remove_background, working_image

PAINT = 1
ERASE = -1
BRUSH_SIZE_MIN = 1
BRUSH_SIZE_MAX = 128
BRUSH_SIZE_DEFAULT = 1


class Tool(Enum):
    PAN = "pan"
    BRUSH = "brush"
    LINE = "line"
    RECTANGLE = "rectangle"
    ERASE = "erase"
    FILL = "fill"


def disk_mask(diameter: int) -> np.ndarray:
    """Boolean disk of `diameter` pixels. 1 is a single pixel.

    Odd diameters are centered on a pixel. The disk is not cached as a view
    the caller could write into; each call returns its own array.
    """
    diameter = max(BRUSH_SIZE_MIN, int(diameter))
    if diameter == 1:
        return np.ones((1, 1), dtype=bool)
    coords = np.arange(diameter) - (diameter - 1) / 2.0
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    return xx * xx + yy * yy <= (diameter / 2.0) ** 2


def _disk_origin(x: int, y: int, diameter: int) -> tuple[int, int]:
    """Top-left of a disk centered on pixel (x, y)."""
    return int(x) - int(diameter) // 2, int(y) - int(diameter) // 2


def stamp(target: np.ndarray, x: int, y: int, diameter: int, value) -> None:
    """Write a disk centered on (x, y) into a 2-D array or an RGBA image."""
    diameter = max(BRUSH_SIZE_MIN, int(diameter))
    disk = disk_mask(diameter)
    height, width = disk.shape
    x0, y0 = _disk_origin(x, y, diameter)
    src_x0 = max(0, -x0)
    src_y0 = max(0, -y0)
    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = min(target.shape[1], x0 + width)
    dst_y1 = min(target.shape[0], y0 + height)
    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return
    src = disk[src_y0 : src_y0 + (dst_y1 - dst_y0), src_x0 : src_x0 + (dst_x1 - dst_x0)]
    view = target[dst_y0:dst_y1, dst_x0:dst_x1]
    view[src] = value


def line_points(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Integer pixels along the segment, including both endpoints."""
    x0, y0, x1, y1 = (int(v) for v in (x0, y0, x1, y1))
    steps = max(abs(x1 - x0), abs(y1 - y0))
    if steps == 0:
        return [(x0, y0)]
    xs = np.rint(np.linspace(x0, x1, steps + 1)).astype(int)
    ys = np.rint(np.linspace(y0, y1, steps + 1)).astype(int)
    points = [(int(xs[0]), int(ys[0]))]
    for x, y in zip(xs[1:], ys[1:]):
        point = (int(x), int(y))
        if point != points[-1]:
            points.append(point)
    return points


def stroke(target: np.ndarray, x0: int, y0: int, x1: int, y1: int, diameter: int, value) -> None:
    """Stamp a disk at every pixel of the segment so the stroke has no gaps."""
    for x, y in line_points(x0, y0, x1, y1):
        stamp(target, x, y, diameter, value)


def stroke_rectangle(
    target: np.ndarray, x0: int, y0: int, x1: int, y1: int, diameter: int, value
) -> None:
    """Stroke the rectangle outline between two corners. The interior is left alone.

    A larger brush thickens the outline. It only reaches the middle when the
    rectangle is smaller than the brush.
    """
    left, right = sorted((int(x0), int(x1)))
    top, bottom = sorted((int(y0), int(y1)))
    if left == right and top == bottom:
        stamp(target, left, top, diameter, value)
        return
    if left == right or top == bottom:
        stroke(target, left, top, right, bottom, diameter, value)
        return
    stroke(target, left, top, right, top, diameter, value)
    stroke(target, right, top, right, bottom, diameter, value)
    stroke(target, right, bottom, left, bottom, diameter, value)
    stroke(target, left, bottom, left, top, diameter, value)


def flood_region(on: np.ndarray, x: int, y: int) -> np.ndarray:
    """4-connected cells equal to the seed. `on` is boolean.

    Diagonal neighbors are not included, so a flood cannot slip through a
    corner where two opposite pixels touch.
    """
    height, width = on.shape
    empty = np.zeros((height, width), dtype=bool)
    if not (0 <= int(x) < width and 0 <= int(y) < height):
        return empty
    seed_on = bool(on[int(y), int(x)])
    match = on if seed_on else ~on
    labels = cv2.connectedComponents(match.astype(np.uint8), connectivity=4)[1]
    label = int(labels[int(y), int(x)])
    if label == 0:
        return empty
    return labels == label


def normalize_rgba(color) -> tuple[int, int, int, int]:
    """Accept RGB or RGBA. A missing alpha is opaque."""
    channels = tuple(int(channel) for channel in color)
    if len(channels) == 3:
        return channels[0], channels[1], channels[2], 255
    if len(channels) != 4:
        raise ValueError("A paint color needs 3 or 4 channels.")
    return channels  # type: ignore[return-value]


def flood_fill_rgba(rgba: np.ndarray, x: int, y: int, color) -> np.ndarray:
    """Replace the 4-connected exact-RGBA region with `color`. Returns a copy."""
    result = np.array(rgba, copy=True)
    height, width = result.shape[:2]
    if not (0 <= int(x) < width and 0 <= int(y) < height):
        return result
    seed = result[int(y), int(x)]
    match = np.all(result == seed, axis=-1)
    labels = cv2.connectedComponents(match.astype(np.uint8), connectivity=4)[1]
    label = int(labels[int(y), int(x)])
    if label == 0:
        return result
    result[labels == label] = np.asarray(normalize_rgba(color), dtype=np.uint8)
    return result


def apply_edits(mask: np.ndarray, edits: np.ndarray) -> np.ndarray:
    """Force mask pixels on or off. `edits` aligns to the top-left of `mask`.

    `mask` is uint8 {0, 255}. `edits` is int8: 1 paints, -1 erases, 0 leaves
    the mask alone. An edits array of a different size is clipped.
    """
    out = np.array(mask, copy=True)
    height = min(out.shape[0], edits.shape[0])
    width = min(out.shape[1], edits.shape[1])
    if height == 0 or width == 0:
        return out
    region = edits[:height, :width]
    view = out[:height, :width]
    view[region == PAINT] = 255
    view[region == ERASE] = 0
    return out


def edits_from_runs(runs, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize horizontal runs `(x, y, width, value)` into an int8 array.

    Later runs overwrite earlier ones. Runs that fall outside `shape` are clipped.
    """
    edits = np.zeros(shape, dtype=np.int8)
    height, width = shape
    for entry in runs:
        x, y, run_width, value = (int(part) for part in entry)
        if y < 0 or y >= height or run_width == 0:
            continue
        if x < 0:
            run_width += x
            x = 0
        x1 = min(width, x + run_width)
        if x < x1:
            edits[y, x:x1] = value
    return edits


def runs_from_edits(edits: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Compress non-zero pixels into horizontal runs `(x, y, width, value)`."""
    runs: list[tuple[int, int, int, int]] = []
    height, width = edits.shape
    for y in range(height):
        row = edits[y]
        x = 0
        while x < width:
            value = int(row[x])
            if value == 0:
                x += 1
                continue
            end = x + 1
            while end < width and int(row[end]) == value:
                end += 1
            runs.append((x, y, end - x, value))
            x = end
    return runs


def coalesce_paint(entries) -> list[tuple[int, int, int, int, int, int]]:
    """Last write wins for each (x, y). Order follows the first time a pixel was seen."""
    latest: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for entry in entries:
        x, y = int(entry[0]), int(entry[1])
        latest[(x, y)] = normalize_rgba(entry[2:])
    return [(x, y, r, g, b, a) for (x, y), (r, g, b, a) in latest.items()]


def apply_paint(original: np.ndarray, entries) -> np.ndarray:
    """Copy `original` and write painted source pixels. Empty paint returns it unchanged."""
    if not entries:
        return original
    out = np.array(original, copy=True)
    height, width = out.shape[:2]
    for x, y, r, g, b, a in entries:
        if 0 <= int(y) < height and 0 <= int(x) < width:
            out[int(y), int(x)] = (r, g, b, a)
    return out


def crop_origin(rgba: np.ndarray) -> tuple[int, int]:
    """Top-left of the opaque bounding box, or (0, 0) when nothing is opaque."""
    opaque = np.nonzero(rgba[..., 3] > 0)
    if opaque[0].size == 0:
        return 0, 0
    return int(opaque[1].min()), int(opaque[0].min())


def paint_origin(original: np.ndarray, entries, removed: bool, tolerance: int) -> tuple[int, int]:
    """Working-image origin of source pixels, after paint and background removal."""
    if not removed:
        return 0, 0
    return crop_origin(remove_background(apply_paint(original, entries), tolerance))


def overlay_paint(image: np.ndarray, entries, origin: tuple[int, int]) -> np.ndarray:
    """Stamp source-coordinate paint onto a working image. Always returns a copy when painting."""
    if not entries:
        return image
    out = np.array(image, copy=True)
    height, width = out.shape[:2]
    origin_x, origin_y = origin
    for x, y, r, g, b, a in entries:
        wx = int(x) - origin_x
        wy = int(y) - origin_y
        if 0 <= wy < height and 0 <= wx < width:
            out[wy, wx] = (r, g, b, a)
    return out


def compose_painted(
    original: np.ndarray,
    entries,
    *,
    removed: bool,
    tolerance: int,
    flatten: bool = False,
    cutoffs: list[int] | None = None,
    colors: list | None = None,
    smooth: int = 0,
) -> np.ndarray:
    """Working image with user paint applied on top of removal and flattening.

    Paint is written into the source first so cropping sees erased and added
    pixels, then stamped again in working coordinates so flattening cannot
    recolor a pixel the user set by hand.
    """
    if not entries:
        return working_image(
            original,
            removed,
            tolerance,
            flatten=flatten,
            cutoffs=cutoffs,
            colors=colors,
            smooth=smooth,
        )
    painted = apply_paint(original, entries)
    origin = paint_origin(original, entries, removed, tolerance)
    working = working_image(
        painted,
        removed,
        tolerance,
        flatten=flatten,
        cutoffs=cutoffs,
        colors=colors,
        smooth=smooth,
    )
    return overlay_paint(working, entries, origin)


def iter_disk(x: int, y: int, diameter: int, height: int, width: int):
    """Image pixels covered by a disk centered on (x, y), clipped to the image."""
    diameter = max(BRUSH_SIZE_MIN, int(diameter))
    disk = disk_mask(diameter)
    x0, y0 = _disk_origin(x, y, diameter)
    ys, xs = np.nonzero(disk)
    for dy, dx in zip(ys.tolist(), xs.tolist()):
        px = x0 + int(dx)
        py = y0 + int(dy)
        if 0 <= px < width and 0 <= py < height:
            yield px, py
