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
    opaque = rgba[..., 3] > 0
    if not np.any(opaque):
        return []
    quantize = max(1, int(quantize))
    bins = (256 + quantize - 1) // quantize
    # Histogram the quantized bins in one pass per channel. Sorting every opaque
    # pixel dominates once the image is a few megapixels.
    quantized = rgba[..., :3].astype(np.uint16) // np.uint16(quantize)
    keys = (
        quantized[..., 0].astype(np.int32) * bins + quantized[..., 1].astype(np.int32)
    ) * bins + quantized[..., 2].astype(np.int32)
    n_bins = bins * bins * bins
    flat_keys = np.ascontiguousarray(keys).ravel()
    weight = np.ascontiguousarray(opaque).ravel().astype(np.float64)
    counts = np.bincount(flat_keys, weights=weight, minlength=n_bins)
    sums = np.empty((n_bins, 3), dtype=np.float64)
    channel = np.empty(flat_keys.shape[0], dtype=np.float64)
    for index in range(3):
        channel[:] = np.ascontiguousarray(rgba[..., index]).ravel()
        channel *= weight
        sums[:, index] = np.bincount(flat_keys, weights=channel, minlength=n_bins)
    order = np.argsort(-counts, kind="stable")
    order = order[counts[order] > 0][:count]
    if order.size == 0:
        return []
    means = np.round(sums[order] / counts[order, None]).astype(int)
    return [
        ((int(mean[0]), int(mean[1]), int(mean[2])), int(counts[idx]))
        for mean, idx in zip(means, order)
    ]
