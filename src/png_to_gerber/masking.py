"""Color masking: turn 'all pixels near this color' into a clean binary mask.

The distance metric is the "redmean" weighted RGB distance, which tracks human
color perception much better than plain Euclidean RGB. That lets a moderate
tolerance capture anti-aliased edge pixels of the same perceived color without
bleeding into genuinely different colors, so masks come out solid rather than
riddled with pixel-size gaps.
"""

from __future__ import annotations

import cv2
import numpy as np


def color_distance(rgba: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """Perceptual (redmean) distance from every pixel to `color`, as float32.

    Result scale is comparable to a 0-255 per-channel difference; the maximum
    possible value (black vs white) is around 765.
    """
    rgb = rgba[..., :3].astype(np.float32)
    cr, cg, cb = (float(c) for c in color)
    dr = rgb[..., 0] - cr
    dg = rgb[..., 1] - cg
    db = rgb[..., 2] - cb
    rmean = (rgb[..., 0] + cr) / 2.0
    dist2 = (
        (2.0 + rmean / 256.0) * dr * dr
        + 4.0 * dg * dg
        + (2.0 + (255.0 - rmean) / 256.0) * db * db
    )
    return np.sqrt(dist2)


def build_mask(
    rgba: np.ndarray,
    color: tuple[int, int, int],
    tolerance: int = 35,
    close_radius: int = 1,
    min_island: int = 4,
    fill_holes: bool = False,
    include_transparent: bool = False,
) -> np.ndarray:
    """Build a uint8 {0,255} mask of pixels whose color is within `tolerance`.

    - `tolerance`: perceptual distance threshold. The raw redmean distance is
      scaled by ~1/2 so the slider behaves like a 0-255 per-channel tolerance.
    - `close_radius`: morphological closing radius (px); fills anti-aliasing
      gaps so similar-but-not-identical edge pixels don't leave pinholes.
    - `min_island`: remove connected components smaller than this many px.
    - `fill_holes`: fill fully-enclosed holes in the mask.
    - `include_transparent`: when False, fully transparent pixels never match.
    """
    dist = color_distance(rgba, color) / 2.0
    mask = (dist <= float(tolerance)).astype(np.uint8) * 255

    if not include_transparent:
        mask[rgba[..., 3] == 0] = 0

    if close_radius > 0:
        k = int(close_radius) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask = remove_small_islands(mask, min_island)

    if fill_holes:
        mask = fill_enclosed_holes(mask)

    return mask


def remove_small_islands(mask: np.ndarray, min_island: int) -> np.ndarray:
    """Drop 8-connected components smaller than `min_island` pixels."""
    if min_island <= 0 or mask.max() == 0:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_island
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def invert_mask(rgba: np.ndarray, mask: np.ndarray, min_island: int = 0) -> np.ndarray:
    """Opaque pixels that `mask` does not cover.

    Fully transparent pixels stay empty, matching a negative layer: inversion
    flips the artwork, not the area outside the image.
    """
    inverted = (mask == 0) & (rgba[..., 3] > 0)
    return remove_small_islands(inverted.astype(np.uint8) * 255, min_island)


def negative_mask(
    rgba: np.ndarray, source_masks: list[np.ndarray], min_island: int = 0
) -> np.ndarray:
    """Mask of the area NOT covered by any of `source_masks`.

    Only opaque image pixels can be part of a negative: transparent pixels
    (outside the artwork, or a removed background) are never included.
    """
    covered = np.zeros(rgba.shape[:2], dtype=bool)
    for source in source_masks:
        covered |= source > 0
    negative = (~covered) & (rgba[..., 3] > 0)
    return remove_small_islands(negative.astype(np.uint8) * 255, min_island)


def fill_enclosed_holes(mask: np.ndarray) -> np.ndarray:
    """Fill regions of background that are fully enclosed by the mask."""
    h, w = mask.shape
    # Flood-fill the background from the border; anything still 0 is a hole.
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    inverted = cv2.bitwise_not(mask)
    cv2.floodFill(inverted, flood, (0, 0), 0)
    # Also flood from all border pixels in case (0,0) is inside the mask.
    border = np.zeros_like(mask)
    border[0, :] = border[-1, :] = 255
    border[:, 0] = border[:, -1] = 255
    seeds = cv2.bitwise_and(cv2.bitwise_not(mask), border)
    ys, xs = np.nonzero(seeds)
    inverted = cv2.bitwise_not(mask)
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if inverted[y, x]:
            cv2.floodFill(inverted, flood, (x, y), 0)
    holes = inverted  # leftover non-zero = enclosed holes
    return cv2.bitwise_or(mask, holes)
