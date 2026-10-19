"""Image loading that preserves alpha."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgba(path: str | Path) -> np.ndarray:
    """Load an image as an HxWx4 uint8 RGBA array, preserving transparency."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA")).copy()


def canvas_size(images: list[np.ndarray]) -> tuple[int, int]:
    """`(height, width)` of the shared top-left canvas covering every image."""
    if not images:
        return (0, 0)
    return (
        max(int(image.shape[0]) for image in images),
        max(int(image.shape[1]) for image in images),
    )


def pad_top_left(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Place `image` at the top-left of a zeroed array of the given size.

    An image that is already that size is returned unchanged. Pixels outside
    the image stay 0, so a mask stays empty there and an RGBA image stays
    transparent.
    """
    if image.shape[0] == height and image.shape[1] == width:
        return image
    if image.ndim == 2:
        out = np.zeros((height, width), dtype=image.dtype)
    else:
        out = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    out[: image.shape[0], : image.shape[1]] = image
    return out


def dominant_colors(
    rgba: np.ndarray, count: int = 12, quantize: int = 8
) -> list[tuple[tuple[int, int, int], int]]:
    """Rank quantized RGB bins of opaque pixels by pixel count.

    Returns [(mean_rgb, pixel_count), ...] sorted by descending count.
    Fully transparent pixels are ignored.
    """
    flat = rgba.reshape(-1, 4)
    opaque = flat[flat[:, 3] > 0][:, :3]
    if opaque.size == 0:
        return []
    bins = (256 + quantize - 1) // quantize
    quantized = (opaque.astype(np.uint16) // quantize).astype(np.int32)
    keys = (quantized[:, 0] * bins + quantized[:, 1]) * bins + quantized[:, 2]
    _unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    order = np.argsort(counts)[::-1][:count]
    weights = opaque.astype(np.float64)
    sums = np.column_stack(
        [
            np.bincount(inverse, weights=weights[:, channel], minlength=len(counts))
            for channel in range(3)
        ]
    )
    means = np.round(sums[order] / counts[order, None]).astype(int)
    return [
        ((int(mean[0]), int(mean[1]), int(mean[2])), int(counts[idx]))
        for mean, idx in zip(means, order)
    ]
