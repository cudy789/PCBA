"""Flatten an image to a few brightness buckets, then optionally smooth edges.

Luminance is Rec. 709. Only opaque pixels are measured or recolored; fully
transparent pixels keep their color and alpha. Cutoffs are brightness
boundaries (1-254): N colors need N-1 strictly increasing cutoffs.
"""

from __future__ import annotations

import cv2
import numpy as np

MIN_COLORS = 2
MAX_COLORS = 12
MAX_SMOOTH = 5

_LUMA_R = 0.2126
_LUMA_G = 0.7152
_LUMA_B = 0.0722


def clamp_count(count: int) -> int:
    return min(MAX_COLORS, max(MIN_COLORS, int(count)))


def luminance(rgba: np.ndarray) -> np.ndarray:
    """Rec. 709 luminance of every pixel, rounded to 0-255."""
    rgb = rgba[..., :3].astype(np.float64)
    y = _LUMA_R * rgb[..., 0] + _LUMA_G * rgb[..., 1] + _LUMA_B * rgb[..., 2]
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)


def histogram(rgba: np.ndarray) -> np.ndarray:
    """256-bin counts of opaque-pixel luminance."""
    opaque = rgba[..., 3] > 0
    if not np.any(opaque):
        return np.zeros(256, dtype=np.int64)
    return np.bincount(luminance(rgba)[opaque], minlength=256).astype(np.int64)


def even_cutoffs(count: int) -> list[int]:
    """Evenly spaced bucket boundaries. Four colors → ``[64, 128, 192]``."""
    count = clamp_count(count)
    cutoffs: list[int] = []
    previous = 0
    for step in range(1, count):
        value = int(round(step * 256 / count))
        value = min(254, max(previous + 1, value))
        cutoffs.append(value)
        previous = value
    return cutoffs


def _smooth_histogram(hist: np.ndarray) -> np.ndarray:
    radius = 8
    sigma = 3.0
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(hist.astype(np.float64), radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def _find_peaks(smooth: np.ndarray) -> list[int]:
    """Centers of local maxima, ignoring peaks below 5% of the tallest."""
    peaks: list[int] = []
    size = int(smooth.shape[0])
    index = 0
    while index < size:
        end = index
        while end + 1 < size and smooth[end + 1] == smooth[index]:
            end += 1
        left = smooth[index - 1] if index > 0 else -1.0
        right = smooth[end + 1] if end + 1 < size else -1.0
        if smooth[index] > 0 and smooth[index] > left and smooth[index] > right:
            peaks.append((index + end) // 2)
        index = end + 1
    if not peaks:
        return []
    tallest = max(float(smooth[peak]) for peak in peaks)
    return [peak for peak in peaks if float(smooth[peak]) >= 0.05 * tallest]


def _valid_cutoffs(cutoffs: list[int], count: int) -> bool:
    if len(cutoffs) != count - 1:
        return False
    previous = 0
    for value in cutoffs:
        if not 1 <= value <= 254 or value <= previous:
            return False
        previous = value
    return True


def auto_cutoffs(hist: np.ndarray, count: int) -> list[int]:
    """Place cutoffs midway between the ``count`` strongest brightness peaks.

    Falls back to even spacing when the histogram has fewer than ``count`` peaks
    or the midpoints are not usable boundaries.
    """
    count = clamp_count(count)
    smooth = _smooth_histogram(np.asarray(hist))
    peaks = _find_peaks(smooth)
    if len(peaks) < count:
        return even_cutoffs(count)
    strongest = sorted(peaks, key=lambda peak: float(smooth[peak]), reverse=True)[:count]
    positions = sorted(strongest)
    cutoffs = [(left + right) // 2 for left, right in zip(positions, positions[1:])]
    if not _valid_cutoffs(cutoffs, count):
        return even_cutoffs(count)
    return cutoffs


def bucket_index(luma: np.ndarray, cutoffs: list[int]) -> np.ndarray:
    """Bucket id of each luminance. A value equal to a cutoff starts the next bucket."""
    return np.digitize(luma, np.asarray(cutoffs, dtype=np.int32), right=False)


def _bucket_edges(cutoffs: list[int]) -> list[tuple[int, int]]:
    bounds = [0, *cutoffs, 256]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _empty_gray(start: int, end: int) -> tuple[int, int, int]:
    """Gray at the midpoint of luminance values in ``[start, end)``."""
    mid = (start + end - 1) // 2
    return (mid, mid, mid)


def bucket_means(
    rgba: np.ndarray, cutoffs: list[int]
) -> list[tuple[int, int, int]]:
    """Mean RGB of each bucket. An empty bucket becomes a gray at its midpoint."""
    opaque = rgba[..., 3] > 0
    indices = bucket_index(luminance(rgba), cutoffs)
    rgb = rgba[..., :3].astype(np.float64)
    means: list[tuple[int, int, int]] = []
    for bucket, (start, end) in enumerate(_bucket_edges(cutoffs)):
        selected = opaque & (indices == bucket)
        if not np.any(selected):
            means.append(_empty_gray(start, end))
            continue
        average = np.rint(rgb[selected].mean(axis=0)).astype(int)
        means.append((int(average[0]), int(average[1]), int(average[2])))
    return means


def palette_for(
    rgba: np.ndarray,
    cutoffs: list[int],
    overrides: list[tuple[int, int, int] | None] | None = None,
) -> list[tuple[int, int, int]]:
    """Bucket output colors: an override when set, otherwise the bucket mean."""
    means = bucket_means(rgba, cutoffs)
    choices = list(overrides or [])
    palette: list[tuple[int, int, int]] = []
    for index, mean in enumerate(means):
        choice = choices[index] if index < len(choices) else None
        if choice is None:
            palette.append(mean)
            continue
        palette.append(tuple(max(0, min(255, int(channel))) for channel in choice[:3]))
    return palette


def _smooth_labels(labels: np.ndarray, opaque: np.ndarray, smooth: int, buckets: int) -> np.ndarray:
    """Median-filter bucket ids. Transparent pixels stay out of the median."""
    level = max(0, min(MAX_SMOOTH, int(smooth)))
    if level <= 0:
        return labels
    ksize = level * 2 + 1
    height, width = labels.shape
    while ksize >= 3 and (ksize > height or ksize > width):
        ksize -= 2
    if ksize < 3:
        return labels
    # 255 is not a real bucket (there are at most 12), so clear pixels cannot
    # vote themselves into the palette.
    work = np.where(opaque, labels, np.uint8(255))
    blurred = cv2.medianBlur(np.ascontiguousarray(work), ksize)
    use = opaque & (blurred < buckets)
    return np.where(use, blurred, labels).astype(np.uint8)


def flatten_image(
    rgba: np.ndarray,
    cutoffs: list[int],
    overrides: list[tuple[int, int, int] | None] | None = None,
    smooth: int = 0,
) -> np.ndarray:
    """Return a copy reduced to one color per brightness bucket.

    ``smooth`` 0 is off. 1..5 median-filters the bucket ids with a kernel of
    3, 5, ..., 11 pixels, which removes speckles without creating new colors.
    """
    result = np.array(rgba, copy=True)
    opaque = rgba[..., 3] > 0
    if not np.any(opaque):
        return result
    labels = bucket_index(luminance(rgba), cutoffs).astype(np.uint8)
    palette = palette_for(rgba, cutoffs, overrides)
    labels = _smooth_labels(labels, opaque, smooth, len(palette))
    colors = np.asarray(palette, dtype=np.uint8)
    result[opaque, :3] = colors[labels[opaque]]
    return result
