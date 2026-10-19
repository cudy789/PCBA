"""Export layer masks as PNG images, and an optional rendered picture of the board.

Per-layer files use the same hues as the on-screen preview, drawn solid.
A cut layer is only its perimeter. The rendered PCB is a separate picture:
green solder mask, copper showing through the mask openings, white
silkscreen on top, and transparent pixels outside the cut outline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import shapely
from PIL import Image

from .model import Layer, LayerType

# Same hues as the canvas preview, written opaque so the file is easy to see.
LAYER_TINTS = {
    LayerType.COPPER: (218, 140, 30),
    LayerType.MASK: (46, 160, 90),
    LayerType.SILKSCREEN: (120, 160, 255),
    LayerType.CUT: (230, 60, 60),
}

SOLDERMASK_RGB = (20, 92, 52)
COPPER_RGB = (218, 140, 30)
SILK_RGB = (245, 245, 245)
RENDER_NAME = "PCB"


def layer_rgba(mask: np.ndarray, layer_type: LayerType) -> np.ndarray:
    """Solid tint where `mask` is on, transparent everywhere else."""
    tint = LAYER_TINTS[layer_type]
    image = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    on = mask > 0
    image[on, 0] = tint[0]
    image[on, 1] = tint[1]
    image[on, 2] = tint[2]
    image[on, 3] = 255
    return image


def cut_perimeter_rgba(height: int, width: int, outline) -> np.ndarray:
    """One-pixel cut tint around the inside of `outline`.

    Outline coordinates lie on pixel edges: pixel (x, y) covers x..x+1, y..y+1.
    An edge on the far side of the last pixel is drawn on that pixel, so the
    bottom and right of the image are not dropped.
    """
    image = np.zeros((height, width, 4), dtype=np.uint8)
    if height <= 0 or width <= 0 or outline.is_empty:
        return image
    ys, xs = np.mgrid[0:height, 0:width]
    inside = np.asarray(shapely.contains_xy(outline, xs + 0.5, ys + 0.5), dtype=bool)
    if not inside.any():
        return image
    # Off-image neighbors count as outside, so a cut that reaches the image
    # border still paints that border pixel.
    exposed = np.zeros(inside.shape, dtype=bool)
    exposed[0, :] = exposed[-1, :] = True
    exposed[:, 0] = exposed[:, -1] = True
    exposed[1:, :] |= ~inside[:-1, :]
    exposed[:-1, :] |= ~inside[1:, :]
    exposed[:, 1:] |= ~inside[:, :-1]
    exposed[:, :-1] |= ~inside[:, 1:]
    tint = LAYER_TINTS[LayerType.CUT]
    image[inside & exposed, 0] = tint[0]
    image[inside & exposed, 1] = tint[1]
    image[inside & exposed, 2] = tint[2]
    image[inside & exposed, 3] = 255
    return image


def render_pcb(
    height: int,
    width: int,
    parts: list[tuple[Layer, np.ndarray]],
    cut_polygon=None,
) -> np.ndarray:
    """Compose `parts` into one RGBA picture of the board.

    Copper is covered by solder mask except where a mask layer opens it.
    With no mask layer in `parts`, the copper is shown in full. A cut outline
    clips the board; without one, and without a cut mask, the whole image is
    the board.
    """
    board = np.ones((height, width), dtype=bool)
    if cut_polygon is not None and not cut_polygon.is_empty:
        ys, xs = np.mgrid[0:height, 0:width]
        board = shapely.contains_xy(cut_polygon, xs + 0.5, ys + 0.5)
    else:
        cut_masks = [mask for layer, mask in parts if layer.is_cut]
        if cut_masks:
            board = np.zeros((height, width), dtype=bool)
            for mask in cut_masks:
                board |= mask > 0

    copper = np.zeros((height, width), dtype=bool)
    openings = np.zeros((height, width), dtype=bool)
    silk = np.zeros((height, width), dtype=bool)
    has_mask = False
    for layer, mask in parts:
        on = mask > 0
        if layer.layer_type == LayerType.COPPER:
            copper |= on
        elif layer.layer_type == LayerType.MASK:
            openings |= on
            has_mask = True
        elif layer.layer_type == LayerType.SILKSCREEN:
            silk |= on

    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[board, 0] = SOLDERMASK_RGB[0]
    image[board, 1] = SOLDERMASK_RGB[1]
    image[board, 2] = SOLDERMASK_RGB[2]
    image[board, 3] = 255

    visible_copper = (copper & openings) if has_mask else copper
    visible_copper &= board
    image[visible_copper, 0] = COPPER_RGB[0]
    image[visible_copper, 1] = COPPER_RGB[1]
    image[visible_copper, 2] = COPPER_RGB[2]

    silk &= board
    image[silk, 0] = SILK_RGB[0]
    image[silk, 1] = SILK_RGB[1]
    image[silk, 2] = SILK_RGB[2]
    return image


def safe_filename(name: str) -> str:
    """A single path segment made from a layer name."""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "layer"


def layer_png_name(base_name: str, layer_name: str, used: set[str]) -> str:
    """`{base}-{layer}.png`, with a numeric suffix when that stem is taken."""
    stem = f"{base_name}-{safe_filename(layer_name)}"
    candidate = stem
    number = 2
    while candidate.lower() in used:
        candidate = f"{stem}_{number}"
        number += 1
    used.add(candidate.lower())
    return f"{candidate}.png"


def save_rgba(path: Path, rgba: np.ndarray) -> None:
    Image.fromarray(rgba, "RGBA").save(path)
