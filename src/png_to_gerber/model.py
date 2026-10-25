"""Typed project/layer data model.

A project holds any number of base images. Layers mask a single color of their
own image, or (for "negative" layers) are derived from the inverse of other
layers, which may belong to any image. Copper, solder-mask-opening, and
silkscreen layers choose a PCB side (front/back). Each cut layer is one
contiguous board perimeter; a project may hold any number of them, and export
unions those perimeters into the board outline.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum

DEFAULT_BG_TOLERANCE = 12
DEFAULT_FLATTEN_COUNT = 4
FLATTEN_COUNT_MIN = 2
FLATTEN_COUNT_MAX = 12
FLATTEN_SMOOTH_MAX = 5


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

_layer_ids = itertools.count(1)
_image_ids = itertools.count(1)


def _next_layer_id() -> int:
    return next(_layer_ids)


def _next_image_id() -> int:
    return next(_image_ids)


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
    # Horizontal runs (x, y, width, value) that override the computed mask.
    # 1 forces a pixel on, -1 forces it off. Later runs win where they overlap.
    manual: list[tuple[int, int, int, int]] = field(default_factory=list)
    id: int = field(default_factory=_next_layer_id)

    @property
    def is_cut(self) -> bool:
        return self.layer_type == LayerType.CUT

    @property
    def is_derived(self) -> bool:
        return bool(self.negative_of)

    @property
    def has_artwork(self) -> bool:
        """True when export has something to draw: a color, a negative, or painted pixels."""
        return self.has_color or bool(self.negative_of) or bool(self.manual)


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


def default_flatten_cutoffs() -> list[int]:
    """Even brightness boundaries for the default four buckets."""
    return [64, 128, 192]


def default_flatten_colors() -> list:
    """One unset override per default bucket (``None`` uses the live average)."""
    return [None] * DEFAULT_FLATTEN_COUNT


@dataclass
class BaseImage:
    """One source image and the layers masked from it."""

    name: str = "Base image"
    path: str | None = None
    background_removed: bool = False
    background_tolerance: int = DEFAULT_BG_TOLERANCE
    flatten_enabled: bool = False
    flatten_count: int = DEFAULT_FLATTEN_COUNT  # 2-12 colors
    flatten_cutoffs: list[int] = field(default_factory=default_flatten_cutoffs)
    # None = use that bucket's average color. Same length as flatten_count.
    flatten_colors: list = field(default_factory=default_flatten_colors)
    flatten_smooth: int = 0  # 0 off .. 5 strongest edge median
    visible: bool = True
    # Pixels painted by hand, in source-image coordinates: (x, y, r, g, b, a).
    # Applied on top of background removal and color flattening.
    paint: list[tuple[int, int, int, int, int, int]] = field(default_factory=list)
    layers: list[Layer] = field(default_factory=list)
    id: int = field(default_factory=_next_image_id)


def reset_flatten(image: BaseImage) -> None:
    """Turn flattening off and restore the default buckets."""
    image.flatten_enabled = False
    image.flatten_count = DEFAULT_FLATTEN_COUNT
    image.flatten_cutoffs = default_flatten_cutoffs()
    image.flatten_colors = default_flatten_colors()
    image.flatten_smooth = 0


class Project:
    """Layers nested under one or more base images, plus board settings.

    `image_path`, `background_removed`, `background_tolerance`, and
    `base_visible` read and write the first image so a one-image project keeps
    the original attribute interface. `layers` is every layer in image order.
    """

    def __init__(
        self,
        image_path: str | None = None,
        *,
        background_removed: bool = False,
        background_tolerance: int = DEFAULT_BG_TOLERANCE,
        base_visible: bool = True,
        checker_opacity: int = 50,
        board: BoardSettings | None = None,
        images: list[BaseImage] | None = None,
    ):
        self.images: list[BaseImage] = list(images) if images else []
        self.board = board if board is not None else BoardSettings()
        self.checker_opacity = checker_opacity
        if images:
            return
        if (
            image_path is not None
            or background_removed
            or background_tolerance != DEFAULT_BG_TOLERANCE
            or not base_visible
        ):
            self.images.append(
                BaseImage(
                    name=_image_name(image_path),
                    path=image_path,
                    background_removed=background_removed,
                    background_tolerance=background_tolerance,
                    visible=base_visible,
                )
            )

    def add_image(
        self,
        name: str = "Base image",
        path: str | None = None,
        *,
        background_removed: bool = False,
        background_tolerance: int = DEFAULT_BG_TOLERANCE,
        visible: bool = True,
    ) -> BaseImage:
        image = BaseImage(
            name=name or _image_name(path),
            path=path,
            background_removed=background_removed,
            background_tolerance=background_tolerance,
            visible=visible,
        )
        self.images.append(image)
        return image

    def get_image(self, image_id: int) -> BaseImage | None:
        for image in self.images:
            if image.id == image_id:
                return image
        return None

    def image_of(self, layer_id: int) -> BaseImage | None:
        for image in self.images:
            if any(layer.id == layer_id for layer in image.layers):
                return image
        return None

    def remove_image(self, image_id: int) -> list[int]:
        """Drop an image and its layers. Returns every removed layer id."""
        image = self.get_image(image_id)
        if image is None:
            return []
        removed: list[int] = []
        for layer in list(image.layers):
            if self.get_layer(layer.id) is not None:
                removed.extend(self.remove_layer(layer.id))
        self.images = [existing for existing in self.images if existing.id != image_id]
        return removed

    @property
    def layers(self) -> list[Layer]:
        return [layer for image in self.images for layer in image.layers]

    def _primary(self) -> BaseImage | None:
        return self.images[0] if self.images else None

    def _ensure_primary(self) -> BaseImage:
        if not self.images:
            self.images.append(BaseImage())
        return self.images[0]

    @property
    def image_path(self) -> str | None:
        image = self._primary()
        return image.path if image is not None else None

    @image_path.setter
    def image_path(self, value: str | None) -> None:
        if not self.images:
            if value is None:
                return
            self.add_image(name=_image_name(value), path=value)
            return
        self.images[0].path = value
        if value and self.images[0].name == "Base image":
            self.images[0].name = _image_name(value)

    @property
    def background_removed(self) -> bool:
        image = self._primary()
        return image.background_removed if image is not None else False

    @background_removed.setter
    def background_removed(self, value: bool) -> None:
        self._ensure_primary().background_removed = bool(value)

    @property
    def background_tolerance(self) -> int:
        image = self._primary()
        return image.background_tolerance if image is not None else DEFAULT_BG_TOLERANCE

    @background_tolerance.setter
    def background_tolerance(self, value: int) -> None:
        self._ensure_primary().background_tolerance = int(value)

    @property
    def base_visible(self) -> bool:
        image = self._primary()
        return image.visible if image is not None else True

    @base_visible.setter
    def base_visible(self, value: bool) -> None:
        self._ensure_primary().visible = bool(value)

    def add_layer(
        self,
        layer: Layer,
        after_id: int | None = None,
        image_id: int | None = None,
    ) -> Layer:
        if layer.is_cut and layer.negative_of:
            raise ProjectError("A cut layer cannot be a negative layer.")
        for source_id in layer.negative_of:
            if self.get_layer(source_id) is None:
                raise ProjectError(f"Negative source layer {source_id} does not exist.")
        image = self._image_for_new_layer(image_id, after_id)
        if after_id is None:
            image.layers.append(layer)
            return layer
        owner = self.image_of(after_id)
        if owner is not image:
            image.layers.append(layer)
            return layer
        idx = next(i for i, existing in enumerate(image.layers) if existing.id == after_id)
        image.layers.insert(idx + 1, layer)
        return layer

    def _image_for_new_layer(self, image_id: int | None, after_id: int | None) -> BaseImage:
        if image_id is not None:
            image = self.get_image(image_id)
            if image is None:
                raise ProjectError(f"Base image {image_id} does not exist.")
            return image
        if after_id is not None:
            owner = self.image_of(after_id)
            if owner is not None:
                return owner
        if not self.images:
            return self.add_image()
        return self.images[0]

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
            self._take_layer(current)
            removed.append(current)
            for layer in self.layers:
                if current in layer.negative_of:
                    layer.negative_of = [s for s in layer.negative_of if s != current]
                    if not layer.negative_of:
                        pending.append(layer.id)
        return removed

    def _take_layer(self, layer_id: int) -> None:
        for image in self.images:
            for index, layer in enumerate(image.layers):
                if layer.id == layer_id:
                    image.layers.pop(index)
                    return

    def get_layer(self, layer_id: int) -> Layer | None:
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        return None

    def move_layer(self, layer_id: int, delta: int) -> None:
        """Move a layer up (delta < 0) or down (delta > 0) within its image."""
        image = self.image_of(layer_id)
        if image is None:
            return
        idx = next(i for i, layer in enumerate(image.layers) if layer.id == layer_id)
        new_idx = max(0, min(len(image.layers) - 1, idx + delta))
        if new_idx != idx:
            layer = image.layers.pop(idx)
            image.layers.insert(new_idx, layer)

    def cut_layers(self) -> list[Layer]:
        return [layer for layer in self.layers if layer.is_cut]

    def cut_layer(self) -> Layer | None:
        cuts = self.cut_layers()
        return cuts[0] if cuts else None

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
            if not layer.visible or not layer.has_artwork:
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
    if layer_type == LayerType.CUT and n == 1:
        return base
    return f"{base} {n}"


def _image_name(path: str | None) -> str:
    if not path:
        return "Base image"
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    return name or "Base image"
