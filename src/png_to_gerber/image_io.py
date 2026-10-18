"""Image loading that preserves alpha."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgba(path: str | Path) -> np.ndarray:
    """Load an image as an HxWx4 uint8 RGBA array, preserving transparency."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA")).copy()


def dominant_colors(
    rgba: np.ndarray, count: int = 12, quantize: int = 8
) -> list[tuple[tuple[int, int, int], int]]:
    """Rank quantized RGB bins of opaque pixels by pixel count.

    Returns [(mean_rgb, pixel_count), ...] sorted by descending count.
    Fully transparent pixels are ignored.
    """
    flat = rgba.reshape(-1, 4)
    opaque = flat[flat[:, 3] > 0][:, :3].astype(np.uint16)
    if opaque.size == 0:
        return []
    q = (opaque // quantize).astype(np.int32)
    _, inverse, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(counts)[::-1][:count]
    result = []
    for idx in order:
        pixels = opaque[inverse == idx]
        mean = np.round(pixels.mean(axis=0)).astype(int)
        result.append(((int(mean[0]), int(mean[1]), int(mean[2])), int(counts[idx])))
    return result
