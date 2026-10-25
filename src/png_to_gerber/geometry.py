"""Mask -> polygon conversion and smooth board-outline derivation.

Coordinates throughout are image pixel coordinates (x right, y down).
The Gerber writer handles the flip to board coordinates.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

from .model import OutlineMode


# Heading of a directed pixel-corner edge, in counterclockwise order.
_EAST, _NORTH, _WEST, _SOUTH = 0, 1, 2, 3


def _boundary_edges(binary: np.ndarray):
    """Directed edges along pixel corners, with the filled side on the left.

    Pixel (x, y) covers [x, x+1] x [y, y+1]. An edge shared by two set pixels
    is emitted twice in opposite directions and cancels, so the remainder is
    the boundary. Diagonal neighbors only meet at a corner and stay separate.
    """
    height, width = binary.shape
    stride = width + 1

    above = np.zeros((height + 1, width), dtype=bool)
    below = np.zeros((height + 1, width), dtype=bool)
    above[1:] = binary
    below[:-1] = binary
    horizontal = above != below
    row, col = np.nonzero(horizontal)
    below_on = below[row, col]
    start_x = np.where(below_on, col, col + 1)
    end_x = np.where(below_on, col + 1, col)
    horizontal_dir = np.where(below_on, _EAST, _WEST).astype(np.int8)

    left = np.zeros((height, width + 1), dtype=bool)
    right = np.zeros((height, width + 1), dtype=bool)
    left[:, 1:] = binary
    right[:, :-1] = binary
    vertical = left != right
    v_row, v_col = np.nonzero(vertical)
    left_on = left[v_row, v_col]
    start_y = np.where(left_on, v_row, v_row + 1)
    end_y = np.where(left_on, v_row + 1, v_row)
    vertical_dir = np.where(left_on, _NORTH, _SOUTH).astype(np.int8)

    n_horizontal = int(row.size)
    n_vertical = int(v_row.size)
    count = n_horizontal + n_vertical
    start = np.empty(count, dtype=np.int32)
    end = np.empty(count, dtype=np.int32)
    direction = np.empty(count, dtype=np.int8)
    if n_horizontal:
        start[:n_horizontal] = row * stride + start_x
        end[:n_horizontal] = row * stride + end_x
        direction[:n_horizontal] = horizontal_dir
    if n_vertical:
        start[n_horizontal:] = start_y * stride + v_col
        end[n_horizontal:] = end_y * stride + v_col
        direction[n_horizontal:] = vertical_dir
    return start, end, direction, stride


def _successors(start: np.ndarray, end: np.ndarray, direction: np.ndarray, n_vertices: int):
    """Next edge around the boundary, filled pixels staying on the left.

    Most corners have one way out. A corner where two diagonal pixels meet has
    two; the smaller counterclockwise turn stays with the pixel we arrived on,
    so those pixels become two rings that touch at a point.
    """
    count = int(start.size)
    order = np.argsort(start, kind="mergesort")
    sorted_start = start[order]
    change = np.empty(count, dtype=bool)
    change[0] = True
    if count > 1:
        change[1:] = sorted_start[1:] != sorted_start[:-1]
    group_starts = np.flatnonzero(change)
    sizes = np.diff(np.append(group_starts, count))

    succ = np.full(count, -1, dtype=np.int32)
    out_count = np.bincount(start, minlength=n_vertices)

    one = sizes == 1
    if np.any(one):
        slots = group_starts[one]
        unique = np.empty(n_vertices, dtype=np.int32)
        unique[sorted_start[slots]] = order[slots]
        single_end = out_count[end] == 1
        succ[single_end] = unique[end[single_end]]

    two = np.flatnonzero(sizes == 2)
    if two.size:
        slots = group_starts[two]
        edge_a = np.empty(n_vertices, dtype=np.int32)
        edge_b = np.empty(n_vertices, dtype=np.int32)
        edge_a[sorted_start[slots]] = order[slots]
        edge_b[sorted_start[slots]] = order[slots + 1]
        pending = np.flatnonzero((succ < 0) & (out_count[end] == 2))
        heading = direction[pending].astype(np.int16)
        first = edge_a[end[pending]]
        second = edge_b[end[pending]]
        delta_first = (direction[first].astype(np.int16) - heading) & 3
        delta_second = (direction[second].astype(np.int16) - heading) & 3
        succ[pending] = np.where(delta_first <= delta_second, first, second)

    extra = np.flatnonzero(sizes > 2)
    for slot, size in zip(group_starts[extra], sizes[extra]):
        choices = order[slot : slot + size]
        vertex = int(sorted_start[slot])
        for edge in np.flatnonzero(end == vertex):
            heading = int(direction[edge])
            best = int(choices[0])
            best_delta = (int(direction[best]) - heading) & 3
            for candidate in choices[1:]:
                delta = (int(direction[candidate]) - heading) & 3
                if delta < best_delta:
                    best = int(candidate)
                    best_delta = delta
            succ[int(edge)] = best
    return succ


def _trace_rings(start, end, direction, stride: int, height: int):
    """Corner-only rings. Straight pixel runs contribute their endpoints."""
    count = int(start.size)
    if count == 0:
        return np.empty((0, 2), dtype=np.float64), []
    succ = _successors(start, end, direction, (height + 1) * stride)
    turn = direction[succ] != direction
    used = np.zeros(count, dtype=bool)
    # One closing vertex per ring on top of one corner per edge.
    cx = np.empty(count + count // 4 + 8, dtype=np.int32)
    cy = np.empty_like(cx)
    areas: list[int] = []
    offsets = [0]
    pos = 0
    for origin in range(count):
        if used[origin]:
            continue
        edge = origin
        start_pos = pos
        prev_x = prev_y = first_x = first_y = None
        area = 0
        while not used[edge]:
            used[edge] = True
            nxt = int(succ[edge])
            if turn[edge]:
                vertex = int(end[edge])
                x = vertex % stride
                y = vertex // stride
                if first_x is None:
                    first_x = x
                    first_y = y
                else:
                    area += prev_x * y - prev_y * x
                prev_x = x
                prev_y = y
                cx[pos] = x
                cy[pos] = y
                pos += 1
            edge = nxt
            if edge == origin:
                break
        if pos - start_pos >= 3:
            area += prev_x * first_y - prev_y * first_x
            cx[pos] = first_x
            cy[pos] = first_y
            pos += 1
            areas.append(area)
            offsets.append(pos)
        else:
            pos = start_pos
    if not areas:
        return np.empty((0, 2), dtype=np.float64), []
    coords = np.column_stack((cx[:pos], cy[:pos])).astype(np.float64, copy=False)
    spans = [(offsets[i], offsets[i + 1], areas[i]) for i in range(len(areas))]
    return coords, spans


def _split_pinches(coords: np.ndarray, spans):
    """Split rings that visit a corner twice.

    A filled region can touch itself at a pixel corner. The trace is still one
    loop, and that pinch is a self-intersection. Each lobe keeps the direction
    it was walked, so a shell stays positive and a hole stays negative.
    """
    pieces: list[tuple[np.ndarray, int]] = []
    for start, end, _area in spans:
        xs = coords[start:end, 0]
        ys = coords[start:end, 1]
        seen: dict[tuple[int, int], int] = {}
        current: list[tuple[int, int]] = []
        loops: list[list[tuple[int, int]]] = []
        for x, y in zip(xs[:-1].tolist(), ys[:-1].tolist()):
            key = (int(x), int(y))
            previous = seen.get(key)
            if previous is not None:
                loop = current[previous:]
                loop.append(key)
                if len(loop) >= 4:
                    loops.append(loop)
                del current[previous + 1 :]
                seen = {point: index for index, point in enumerate(current)}
            else:
                seen[key] = len(current)
                current.append(key)
        if len(current) >= 3:
            loops.append(current + [current[0]])
        for loop in loops:
            arr = np.asarray(loop, dtype=np.float64)
            xi = arr[:-1, 0].astype(np.int64)
            yi = arr[:-1, 1].astype(np.int64)
            area = int(np.sum(xi * arr[1:, 1].astype(np.int64) - yi * arr[1:, 0].astype(np.int64)))
            if area != 0:
                pieces.append((arr, area))
    if not pieces:
        return np.empty((0, 2), dtype=np.float64), []
    lengths = [len(arr) for arr, _area in pieces]
    packed = np.vstack([arr for arr, _area in pieces])
    offsets = np.cumsum([0, *lengths])
    spans_out = [
        (int(offsets[i]), int(offsets[i + 1]), area) for i, (_arr, area) in enumerate(pieces)
    ]
    return packed, spans_out


def _linearrings(coords: np.ndarray, spans, keep: np.ndarray):
    chosen = [span for span, flag in zip(spans, keep) if flag]
    if not chosen:
        return np.empty((0,), dtype=object)
    lengths = np.fromiter((end - start for start, end, _area in chosen), dtype=np.int32, count=len(chosen))
    packed = np.empty((int(lengths.sum()), 2), dtype=np.float64)
    cursor = 0
    for (start, end, _area), length in zip(chosen, lengths):
        packed[cursor : cursor + int(length)] = coords[start:end]
        cursor += int(length)
    indices = np.repeat(np.arange(len(chosen), dtype=np.int32), lengths)
    return shapely.linearrings(packed, indices=indices)


def _polygons_with_holes(shell_rings, hole_rings, parent: np.ndarray):
    grouped: list[list] = [[] for _ in range(len(shell_rings))]
    for hole_index, shell_index in enumerate(parent):
        if shell_index >= 0:
            grouped[int(shell_index)].append(hole_rings[hole_index])
    geoms = []
    indices = []
    for shell_index, holes in enumerate(grouped):
        geoms.append(shell_rings[shell_index])
        indices.append(shell_index)
        geoms.extend(holes)
        indices.extend([shell_index] * len(holes))
    return shapely.polygons(np.asarray(geoms, dtype=object), indices=np.asarray(indices, dtype=np.int32))


def _assign_holes(shell_polys, hole_rings) -> np.ndarray:
    """Parent shell of each hole: the smallest solid shell that contains it.

    Shells are still solid here, so a point in a hole lies inside its parent
    and outside any island. The probe sits just beside the hole's first edge,
    on the empty side, which keeps it off an island in that hole.
    """
    counts = shapely.get_num_coordinates(hole_rings)
    xy = shapely.get_coordinates(hole_rings)
    first = np.cumsum(counts) - counts
    x0 = xy[first, 0]
    y0 = xy[first, 1]
    dx = np.sign(xy[first + 1, 0] - x0)
    dy = np.sign(xy[first + 1, 1] - y0)
    # Right of the walk is the empty side: (dx, dy) rotated clockwise is (dy, -dx).
    probes = shapely.points(x0 + 0.5 * dx + 0.5 * dy, y0 + 0.5 * dy - 0.5 * dx)
    tree = shapely.STRtree(shell_polys)
    pairs = tree.query(probes, predicate="within")
    parent = np.full(len(hole_rings), -1, dtype=np.int32)
    best = np.full(len(hole_rings), np.inf)
    shell_areas = shapely.area(shell_polys)

    def consider(hole_index: int, shell_index: int) -> None:
        area = float(shell_areas[shell_index])
        if area < best[hole_index]:
            best[hole_index] = area
            parent[hole_index] = shell_index

    if pairs.size:
        for hole_index, shell_index in zip(pairs[0], pairs[1]):
            consider(int(hole_index), int(shell_index))
    missing = np.flatnonzero(parent < 0)
    if missing.size:
        # The edge probe can sit on a corner pinch. The hole polygon itself
        # still lies in its parent.
        fallback = shapely.polygons(hole_rings[missing])
        extra = tree.query(fallback, predicate="within")
        if extra.size:
            for local_index, shell_index in zip(extra[0], extra[1]):
                consider(int(missing[int(local_index)]), int(shell_index))
    return parent


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
    height = int(binary.shape[0])
    start, end, direction, stride = _boundary_edges(binary)
    coords, spans = _trace_rings(start, end, direction, stride, height)
    coords, spans = _split_pinches(coords, spans)
    if not spans:
        return GeometryCollection()
    areas = np.fromiter((area for _start, _end, area in spans), dtype=np.int64, count=len(spans))
    shell_flags = areas > 0
    if not np.any(shell_flags):
        return GeometryCollection()
    shell_rings = _linearrings(coords, spans, shell_flags)
    shell_polys = shapely.polygons(shell_rings)
    hole_flags = areas < 0
    if np.any(hole_flags):
        hole_rings = _linearrings(coords, spans, hole_flags)
        parent = _assign_holes(shell_polys, hole_rings)
        polys = _polygons_with_holes(shell_rings, hole_rings, parent)
    else:
        polys = shell_polys
    valid = shapely.is_valid(polys)
    if not np.all(valid):
        repaired = shapely.get_parts(shapely.buffer(polys[~valid], 0))
        repaired = repaired[~shapely.is_empty(repaired)]
        polys = np.concatenate([polys[valid], repaired]) if np.any(valid) else repaired
    if len(polys) == 0:
        return GeometryCollection()
    if len(polys) == 1:
        merged = polys[0]
    else:
        merged = shapely.multipolygons(polys, indices=np.zeros(len(polys), dtype=np.int32))[0]
    if simplify_px > 0:
        merged = merged.simplify(simplify_px, preserve_topology=True)
    if not merged.is_valid:
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
