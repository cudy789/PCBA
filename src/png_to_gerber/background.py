"""Background removal: make the image's outer contiguous color transparent.

The background is the color found along the image border, grown inward across
4-connected pixels of (nearly) the same color. Same-colored regions that are
enclosed by artwork (the inside of a letter, a hole in a shape) are not
connected to the border and so are left untouched.
"""

from __future__ import annotations

import cv2
import numpy as np

from .flatten import even_cutoffs, flatten_image
from .masking import color_distance


def detect_background_color(rgba: np.ndarray) -> tuple[int, int, int] | None:
    """Most common opaque color along the image border, or None if there is none."""
    border = np.concatenate(
        [rgba[0, :], rgba[-1, :], rgba[1:-1, 0], rgba[1:-1, -1]], axis=0
    )
    opaque = border[border[:, 3] > 0][:, :3]
    if opaque.size == 0:
        return None
    colors, counts = np.unique(opaque, axis=0, return_counts=True)
    r, g, b = colors[int(np.argmax(counts))]
    return int(r), int(g), int(b)


def background_mask(rgba: np.ndarray, tolerance: int = 12) -> np.ndarray:
    """Boolean mask of the border-connected background (including transparent pixels)."""
    color = detect_background_color(rgba)
    if color is None:
        return np.zeros(rgba.shape[:2], dtype=bool)
    # Same scale as layer masks: redmean distance halved ~ per-channel tolerance.
    similar = (color_distance(rgba, color) / 2.0) <= float(tolerance)
    candidate = similar | (rgba[..., 3] == 0)
    # 4-connectivity so the background cannot leak through diagonal gaps in the art.
    _, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=4)
    edge_labels = np.unique(
        np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    )
    edge_labels = edge_labels[edge_labels != 0]
    return np.isin(labels, edge_labels)


def remove_background(rgba: np.ndarray, tolerance: int = 12) -> np.ndarray:
    """Return a copy of `rgba` with the outer contiguous background made transparent."""
    result = rgba.copy()
    result[background_mask(rgba, tolerance), 3] = 0
    return result


def crop_to_opaque(rgba: np.ndarray) -> np.ndarray:
    """Crop to the bounding box of pixels that are still opaque.

    An image with no opaque pixels is returned unchanged (still a copy).
    """
    opaque = np.nonzero(rgba[..., 3] > 0)
    if opaque[0].size == 0:
        return rgba.copy()
    y0, y1 = int(opaque[0].min()), int(opaque[0].max()) + 1
    x0, x1 = int(opaque[1].min()), int(opaque[1].max()) + 1
    return rgba[y0:y1, x0:x1].copy()


def working_image(
    original: np.ndarray,
    removed: bool,
    tolerance: int = 12,
    *,
    flatten: bool = False,
    cutoffs: list[int] | None = None,
    colors: list | None = None,
    smooth: int = 0,
) -> np.ndarray:
    """The image layers and Gerber export are built from.

    With background removal on, the outer background is cleared and the result
    is cropped to the remaining artwork, so a board sized to this image matches
    the artwork instead of the transparent margin around it. Color flattening,
    when requested, runs after that crop so transparent background never
    joins a bucket.
    """
    image = original if not removed else crop_to_opaque(remove_background(original, tolerance))
    if not flatten:
        return image
    return flatten_image(
        image,
        list(cutoffs) if cutoffs else even_cutoffs(4),
        colors,
        smooth,
    )
