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


def _redmean_dist2(rgba: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """Squared redmean distance from every pixel to `color`."""
    rgb = rgba[..., :3].astype(np.float32)
    cr, cg, cb = (float(c) for c in color)
    dr = rgb[..., 0] - cr
    dg = rgb[..., 1] - cg
    db = rgb[..., 2] - cb
    rmean = (rgb[..., 0] + cr) / 2.0
    return (
        (2.0 + rmean / 256.0) * dr * dr
        + 4.0 * dg * dg
        + (2.0 + (255.0 - rmean) / 256.0) * db * db
    )


def color_distance(rgba: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """Perceptual (redmean) distance from every pixel to `color`, as float32.

    Result scale is comparable to a 0-255 per-channel difference; the maximum
    possible value (black vs white) is around 765.
    """
    return np.sqrt(_redmean_dist2(rgba, color))


def pixels_within_tolerance(
    rgba: np.ndarray,
    color: tuple[int, int, int],
    tolerance: float,
    *,
    weigh_alpha: bool = False,
) -> np.ndarray:
    """Pixels whose redmean distance, halved, is within `tolerance`.

    Comparing squared distances skips the square root. With `weigh_alpha`, a
    shortfall from opaque is added on that same scale: an exact-color pixel at
    alpha 128 is 31.75 away and one at alpha 32 is 55.75 away.
    """
    if tolerance < 0 or rgba.size == 0:
        return np.zeros(rgba.shape[:2], dtype=bool)
    dist2 = _redmean_dist2(rgba, color)
    limit = (2.0 * float(tolerance)) ** 2
    if not weigh_alpha:
        return dist2 <= limit
    gap = 255.0 - rgba[..., 3].astype(np.float32)
    return dist2 + 0.25 * gap * gap <= limit


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
      Partial opacity counts toward the same tolerance: the target color is
      opaque, so a faint pixel must be correspondingly closer in RGB.
      Nearly transparent pixels are often stored near black (cleared RGB, or
      premultiplied edges of other colors), and an RGB-only test would
      select that whole fringe for a black layer.
    """
    # Quartered alpha gap, when used, sits on the tolerance scale: an
    # exact-color pixel at alpha 128 is 31.75 away (inside the default
    # tolerance of 35) and one at alpha 32 is 55.75 away (outside it).
    matched = pixels_within_tolerance(
        rgba, color, tolerance, weigh_alpha=not include_transparent
    )
    mask = matched.astype(np.uint8)
    mask *= 255

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
    lut = np.zeros(n, dtype=np.uint8)
    big = stats[:, cv2.CC_STAT_AREA] >= min_island
    big[0] = False  # label 0 is the background
    lut[big] = 255
    return lut[labels]


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
    """Fill regions of background that are fully enclosed by the mask.

    A one-pixel frame of background makes the outside a single 4-connected
    region, so one flood fill reaches every border opening. Zeros the flood
    never reaches are holes.
    """
    height, width = mask.shape
    padded = np.zeros((height + 2, width + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = np.where(mask > 0, np.uint8(255), np.uint8(0))
    # floodFill's mask argument is two pixels larger than the image.
    cv2.floodFill(padded, np.zeros((height + 4, width + 4), dtype=np.uint8), (0, 0), 128)
    holes = padded[1:-1, 1:-1] == 0
    result = np.array(mask, copy=True)
    result[holes] = 255
    return result
