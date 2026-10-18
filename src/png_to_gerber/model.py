"""Typed project/layer data model.

Layers mask a single color of the source image, or (for "negative" layers) are
derived from the inverse of other layers. Copper, solder-mask-opening, and
silkscreen layers choose a PCB side (front/back); the cut layer defines a
single contiguous board perimeter, so a project may hold at most one of them.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum

DEFAULT_BG_TOLERANCE = 12


class LayerType(Enum):
    COPPER = "copper"
    MASK = "mask"  # solder mask opening (exposed area)
    SILKSCREEN = "silkscreen"
    CUT = "cut"  # board outline / Edge.Cuts


class Side(Enum):
    FRONT = "front"
    BACK = "back"


class OutlineMode(Enum):
    """Which perimeter of the cut color becomes the board outline."""

    OUTER = "outer"  # outside edge of the color region
    INNER = "inner"  # edge of the area enclosed by the color (e.g. a frame)


SIDED_TYPES = (LayerType.COPPER, LayerType.MASK, LayerType.SILKSCREEN)

_id_counter = itertools.count(1)


def _next_layer_id() -> int:
    return next(_id_counter)


@dataclass
class Layer:
    """One color mask (or negative of other layers) plus its parameters."""

    name: str
    layer_type: LayerType
    side: Side = Side.FRONT  # ignored for CUT layers
    color: tuple[int, int, int] = (0, 0, 0)  # target RGB
    tolerance: int = 35  # perceptual color distance threshold (0-255-ish scale)
    close_radius: int = 1  # morphological closing radius, px (fills AA gaps)
    min_island: int = 4  # remove connected specks smaller than this, px
    fill_holes: bool = False  # fill fully-enclosed holes in the mask
    smoothness: int = 20  # CUT only: 0 (raw) .. 100 (very smooth)
    outline_mode: OutlineMode = OutlineMode.OUTER  # CUT only
    visible: bool = True
    has_color: bool = False  # becomes True once the user picks a color
    # Non-empty => this layer is the negative of the listed layers (by id).
    negative_of: list[int] = field(default_factory=list)
    inverted: bool = False  # paint the opaque pixels this layer does not cover
    id: int = field(default_factory=_next_layer_id)

    @property
    def is_cut(self) -> bool:
        return self.layer_type == LayerType.CUT

    @property
    def is_derived(self) -> bool:
        return bool(self.negative_of)


class ProjectError(ValueError):
    pass


@dataclass
class BoardSettings:
    """Physical board size and export preferences saved with the project."""

    width_mm: float | None = None  # None until the user chooses a size
    height_mm: float | None = None
    lock_aspect: bool = True
    output_dir: str = ""
    base_name: str = ""


@dataclass
class Project:
    """An ordered stack of layers over one source image."""

    layers: list[Layer] = field(default_factory=list)
    board: BoardSettings = field(default_factory=BoardSettings)
    image_path: str | None = None
    background_removed: bool = False
    background_tolerance: int = DEFAULT_BG_TOLERANCE
    base_visible: bool = True
    checker_opacity: int = 50  # transparency checkerboard strength, 0-100

    def add_layer(self, layer: Layer, after_id: int | None = None) -> Layer:
        if layer.layer_type == LayerType.CUT:
            if layer.negative_of:
                raise ProjectError("A cut layer cannot be a negative layer.")
            if self.cut_layer() is not None:
                raise ProjectError(
                    "The project already has a cut layer; the board outline "
                    "must be a single contiguous perimeter."
                )
        for source_id in layer.negative_of:
            if self.get_layer(source_id) is None:
                raise ProjectError(f"Negative source layer {source_id} does not exist.")
        if after_id is None:
            self.layers.append(layer)
        else:
            idx = next((i for i, existing in enumerate(self.layers) if existing.id == after_id), None)
            if idx is None:
                self.layers.append(layer)
            else:
                self.layers.insert(idx + 1, layer)
        return layer

    def remove_layer(self, layer_id: int) -> list[int]:
        """Remove a layer; returns every removed layer id.

        Negative layers drop the removed layer from their sources; one left
        with no sources is removed as well (cascading).
        """
        removed: list[int] = []
        pending = [layer_id]
        while pending:
            current = pending.pop()
            if self.get_layer(current) is None:
                continue
            self.layers = [l for l in self.layers if l.id != current]
            removed.append(current)
            for layer in self.layers:
                if current in layer.negative_of:
                    layer.negative_of = [s for s in layer.negative_of if s != current]
                    if not layer.negative_of:
                        pending.append(layer.id)
        return removed

    def get_layer(self, layer_id: int) -> Layer | None:
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        return None

    def move_layer(self, layer_id: int, delta: int) -> None:
        """Move a layer up (delta < 0) or down (delta > 0) in the stack."""
        idx = next((i for i, l in enumerate(self.layers) if l.id == layer_id), None)
        if idx is None:
            return
        new_idx = max(0, min(len(self.layers) - 1, idx + delta))
        if new_idx != idx:
            layer = self.layers.pop(idx)
            self.layers.insert(new_idx, layer)

    def cut_layer(self) -> Layer | None:
        for layer in self.layers:
            if layer.is_cut:
                return layer
        return None

    def artwork_layers(self) -> list[Layer]:
        return [l for l in self.layers if not l.is_cut]

    def dependents(self, layer_id: int) -> set[int]:
        """Ids of all layers that (transitively) derive from `layer_id`."""
        result: set[int] = set()
        frontier = [layer_id]
        while frontier:
            current = frontier.pop()
            for layer in self.layers:
                if current in layer.negative_of and layer.id not in result:
                    result.add(layer.id)
                    frontier.append(layer.id)
        return result

    def dependency_order(self, layer_ids) -> list[int]:
        """Order `layer_ids` so every layer follows the sources it derives from."""
        wanted = set(layer_ids)
        ordered: list[int] = []
        done: set[int] = set()
        visiting: set[int] = set()

        def visit(layer_id: int):
            if layer_id in done:
                return
            if layer_id in visiting:
                raise ProjectError("Negative layers form a dependency cycle.")
            layer = self.get_layer(layer_id)
            if layer is None:
                return
            visiting.add(layer_id)
            for source_id in layer.negative_of:
                if source_id in wanted:
                    visit(source_id)
            visiting.discard(layer_id)
            done.add(layer_id)
            ordered.append(layer_id)

        for layer in self.layers:  # stable, stack order for independent layers
            if layer.id in wanted:
                visit(layer.id)
        return ordered

    def grouped_for_export(self) -> dict[tuple[LayerType, Side], list[Layer]]:
        """Group enabled, colored artwork layers by (type, side) for merging."""
        groups: dict[tuple[LayerType, Side], list[Layer]] = {}
        for layer in self.artwork_layers():
            if not layer.visible or not layer.has_color:
                continue
            groups.setdefault((layer.layer_type, layer.side), []).append(layer)
        return groups


def duplicate_layer_name(name: str, existing: list[Layer]) -> str:
    """A name for a copy of `name` that does not collide with `existing`."""
    base = f"{name} copy"
    taken = {layer.name for layer in existing}
    if base not in taken:
        return base
    number = 2
    while f"{base} {number}" in taken:
        number += 1
    return f"{base} {number}"


def default_layer_name(layer_type: LayerType, existing: list[Layer]) -> str:
    base = {
        LayerType.COPPER: "Copper",
        LayerType.MASK: "Mask opening",
        LayerType.SILKSCREEN: "Silkscreen",
        LayerType.CUT: "Board outline",
    }[layer_type]
    n = sum(1 for l in existing if l.layer_type == layer_type) + 1
    return f"{base} {n}" if layer_type != LayerType.CUT else base
