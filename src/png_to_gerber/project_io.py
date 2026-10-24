""".prj project files: JSON snapshots of layers, board size, and view state.

A project with one base image is written as version 1: a single image path and
a flat layer list. A project with several images is version 2: each image
keeps its own layers, background removal, color flattening, and visibility.
Negative layers
refer to other layers by index in that flat list (version 2 indexes across
every image, in order), so reloading assigns fresh runtime ids without
breaking the references. Each source image is stored by absolute path plus a
path relative to the project file, so a project folder can be moved or shared
along with its images.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .flatten import even_cutoffs
from .model import (
    DEFAULT_BG_TOLERANCE,
    DEFAULT_FLATTEN_COUNT,
    FLATTEN_COUNT_MAX,
    FLATTEN_COUNT_MIN,
    FLATTEN_SMOOTH_MAX,
    BaseImage,
    BoardSettings,
    Layer,
    LayerType,
    OutlineMode,
    Project,
    ProjectError,
    Side,
    default_flatten_colors,
    default_flatten_cutoffs,
)

FORMAT = "png_to_gerber_project"
VERSION = 2
EXTENSION = ".prj"


class ProjectFileError(ValueError):
    """Raised when a project file is unreadable, malformed, or unsupported."""


# ---- saving --------------------------------------------------------------------


def _layer_to_dict(layer: Layer, index_of: dict[int, int]) -> dict[str, Any]:
    return {
        "name": layer.name,
        "type": layer.layer_type.value,
        "side": layer.side.value,
        "color": list(layer.color),
        "has_color": layer.has_color,
        "tolerance": layer.tolerance,
        "close_radius": layer.close_radius,
        "min_island": layer.min_island,
        "fill_holes": layer.fill_holes,
        "smoothness": layer.smoothness,
        "outline_mode": layer.outline_mode.value,
        "visible": layer.visible,
        "inverted": layer.inverted,
        "negative_of": [index_of[s] for s in layer.negative_of if s in index_of],
    }


def _path_dict(path: str | None, project_dir: Path | None) -> dict[str, Any]:
    image: dict[str, Any] = {"absolute": None, "relative": None}
    if not path:
        return image
    absolute = Path(path).absolute()
    image["absolute"] = str(absolute)
    if project_dir is not None:
        try:
            image["relative"] = os.path.relpath(absolute, project_dir)
        except ValueError:  # different drive on Windows
            image["relative"] = None
    return image


def _flatten_dict(image: BaseImage | None) -> dict[str, Any]:
    if image is None:
        return {
            "enabled": False,
            "count": DEFAULT_FLATTEN_COUNT,
            "cutoffs": default_flatten_cutoffs(),
            "colors": default_flatten_colors(),
            "smooth": 0,
        }
    count = image.flatten_count
    colors = list(image.flatten_colors)
    if len(colors) < count:
        colors.extend([None] * (count - len(colors)))
    stored = []
    for color in colors[:count]:
        stored.append(None if color is None else [int(color[0]), int(color[1]), int(color[2])])
    return {
        "enabled": bool(image.flatten_enabled),
        "count": int(count),
        "cutoffs": [int(value) for value in image.flatten_cutoffs],
        "colors": stored,
        "smooth": int(image.flatten_smooth),
    }


def _board_dict(board: BoardSettings) -> dict[str, Any]:
    return {
        "width_mm": board.width_mm,
        "height_mm": board.height_mm,
        "lock_aspect": board.lock_aspect,
        "output_dir": board.output_dir,
        "base_name": board.base_name,
    }


def project_to_dict(project: Project, project_dir: Path | None = None) -> dict[str, Any]:
    index_of = {layer.id: i for i, layer in enumerate(project.layers)}
    board = _board_dict(project.board)
    if len(project.images) <= 1:
        image = project.images[0] if project.images else None
        return {
            "format": FORMAT,
            "version": 1,
            "image": _path_dict(image.path if image else None, project_dir),
            "background": {
                "removed": image.background_removed if image else False,
                "tolerance": image.background_tolerance if image else DEFAULT_BG_TOLERANCE,
            },
            "flatten": _flatten_dict(image),
            "board": board,
            "view": {
                "base_visible": image.visible if image else True,
                "checker_opacity": project.checker_opacity,
            },
            "layers": [_layer_to_dict(layer, index_of) for layer in project.layers],
        }

    images = []
    for image in project.images:
        stored = _path_dict(image.path, project_dir)
        stored["name"] = image.name
        stored["background"] = {
            "removed": image.background_removed,
            "tolerance": image.background_tolerance,
        }
        stored["flatten"] = _flatten_dict(image)
        stored["visible"] = image.visible
        stored["layers"] = [_layer_to_dict(layer, index_of) for layer in image.layers]
        images.append(stored)
    return {
        "format": FORMAT,
        "version": VERSION,
        "images": images,
        "board": board,
        "view": {"checker_opacity": project.checker_opacity},
    }


def save_project(path: str | Path, project: Project) -> Path:
    """Write `project` to `path` (adds the .prj extension if missing)."""
    path = Path(path)
    if path.suffix.lower() != EXTENSION:
        path = path.with_name(path.name + EXTENSION)
    data = project_to_dict(project, path.parent.absolute())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


# ---- loading -------------------------------------------------------------------


def _get_int(data: dict, key: str, default: int, lo: int, hi: int, where: str) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProjectFileError(f"{where}: '{key}' must be an integer.")
    if not lo <= value <= hi:
        raise ProjectFileError(f"{where}: '{key}' must be between {lo} and {hi}.")
    return value


def _get_bool(data: dict, key: str, default: bool, where: str) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ProjectFileError(f"{where}: '{key}' must be true or false.")
    return value


def _get_enum(cls, data: dict, key: str, default, where: str):
    raw = data.get(key, default.value)
    try:
        return cls(raw)
    except ValueError:
        raise ProjectFileError(f"{where}: unknown {key} {raw!r}.") from None


def _get_optional_float(data: dict, key: str, where: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ProjectFileError(f"{where}: '{key}' must be a positive number.")
    return float(value)


def _parse_layer(data: Any, index: int, count: int) -> tuple[Layer, list[int]]:
    where = f"layers[{index}]"
    if not isinstance(data, dict):
        raise ProjectFileError(f"{where} must be an object.")
    raw_type = data.get("type")
    try:
        layer_type = LayerType(raw_type)
    except ValueError:
        raise ProjectFileError(f"{where}: unknown layer type {raw_type!r}.") from None

    color = data.get("color", [0, 0, 0])
    if (
        not isinstance(color, list)
        or len(color) != 3
        or any(isinstance(c, bool) or not isinstance(c, int) or not 0 <= c <= 255 for c in color)
    ):
        raise ProjectFileError(f"{where}: 'color' must be three integers 0-255.")

    sources = data.get("negative_of", [])
    if not isinstance(sources, list) or any(
        isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < count or s == index
        for s in sources
    ):
        raise ProjectFileError(f"{where}: 'negative_of' references invalid layers.")
    if sources and layer_type == LayerType.CUT:
        raise ProjectFileError(f"{where}: a cut layer cannot be a negative layer.")

    name = data.get("name", f"Layer {index + 1}")
    if not isinstance(name, str) or not name.strip():
        raise ProjectFileError(f"{where}: 'name' must be a non-empty string.")

    layer = Layer(
        name=name,
        layer_type=layer_type,
        side=_get_enum(Side, data, "side", Side.FRONT, where),
        color=(color[0], color[1], color[2]),
        tolerance=_get_int(data, "tolerance", 35, 0, 1000, where),
        close_radius=_get_int(data, "close_radius", 1, 0, 100, where),
        min_island=_get_int(data, "min_island", 4, 0, 10_000_000, where),
        fill_holes=_get_bool(data, "fill_holes", False, where),
        smoothness=_get_int(data, "smoothness", 20, 0, 100, where),
        outline_mode=_get_enum(OutlineMode, data, "outline_mode", OutlineMode.OUTER, where),
        visible=_get_bool(data, "visible", True, where),
        has_color=_get_bool(data, "has_color", bool(sources), where) or bool(sources),
        inverted=_get_bool(data, "inverted", False, where),
    )
    return layer, sources


def _resolve_image_path(image: dict, project_dir: Path | None) -> str | None:
    candidates = []
    relative, absolute = image.get("relative"), image.get("absolute")
    if isinstance(relative, str) and relative and project_dir is not None:
        candidates.append(Path(project_dir) / relative)
    if isinstance(absolute, str) and absolute:
        candidates.append(Path(absolute))
    existing = next((c for c in candidates if c.is_file()), None)
    chosen = existing or (candidates[0] if candidates else None)
    return str(chosen) if chosen is not None else None


def _parse_board(data: dict) -> BoardSettings:
    board = data.get("board") or {}
    if not isinstance(board, dict):
        raise ProjectFileError("'board' must be an object.")
    output_dir, base_name = board.get("output_dir", ""), board.get("base_name", "")
    if not isinstance(output_dir, str) or not isinstance(base_name, str):
        raise ProjectFileError("board: 'output_dir' and 'base_name' must be strings.")
    return BoardSettings(
        width_mm=_get_optional_float(board, "width_mm", "board"),
        height_mm=_get_optional_float(board, "height_mm", "board"),
        lock_aspect=_get_bool(board, "lock_aspect", True, "board"),
        output_dir=output_dir,
        base_name=base_name,
    )


def _parse_checker(data: dict) -> int:
    view = data.get("view") or {}
    if not isinstance(view, dict):
        raise ProjectFileError("'view' must be an object.")
    return _get_int(view, "checker_opacity", 50, 0, 100, "view")


def _parse_rgb(value: Any, where: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel <= 255 for channel in value)
    ):
        raise ProjectFileError(f"{where}: each color must be three integers 0-255, or null.")
    return (value[0], value[1], value[2])


def _parse_flatten(source: dict, where: str) -> dict[str, Any]:
    """Read an optional ``flatten`` object. Missing means the defaults."""
    if "flatten" not in source or source.get("flatten") is None:
        return {
            "flatten_enabled": False,
            "flatten_count": DEFAULT_FLATTEN_COUNT,
            "flatten_cutoffs": default_flatten_cutoffs(),
            "flatten_colors": default_flatten_colors(),
            "flatten_smooth": 0,
        }
    raw = source.get("flatten")
    if not isinstance(raw, dict):
        raise ProjectFileError(f"{where}: 'flatten' must be an object.")
    count = _get_int(raw, "count", DEFAULT_FLATTEN_COUNT, FLATTEN_COUNT_MIN, FLATTEN_COUNT_MAX, where)
    if "cutoffs" in raw:
        cutoffs = raw.get("cutoffs")
        if (
            not isinstance(cutoffs, list)
            or len(cutoffs) != count - 1
            or any(isinstance(value, bool) or not isinstance(value, int) for value in cutoffs)
        ):
            raise ProjectFileError(
                f"{where}: 'cutoffs' must be {count - 1} integers from 1 to 254."
            )
        previous = 0
        parsed_cutoffs: list[int] = []
        for value in cutoffs:
            if not 1 <= value <= 254 or value <= previous:
                raise ProjectFileError(
                    f"{where}: 'cutoffs' must be {count - 1} strictly increasing integers from 1 to 254."
                )
            parsed_cutoffs.append(value)
            previous = value
    else:
        parsed_cutoffs = even_cutoffs(count)
    if "colors" in raw:
        colors = raw.get("colors")
        if not isinstance(colors, list) or len(colors) != count:
            raise ProjectFileError(f"{where}: 'colors' must be {count} colors or nulls.")
        parsed_colors = []
        for entry in colors:
            if entry is None:
                parsed_colors.append(None)
            else:
                parsed_colors.append(_parse_rgb(entry, where))
    else:
        parsed_colors = [None] * count
    return {
        "flatten_enabled": _get_bool(raw, "enabled", False, where),
        "flatten_count": count,
        "flatten_cutoffs": parsed_cutoffs,
        "flatten_colors": parsed_colors,
        "flatten_smooth": _get_int(raw, "smooth", 0, 0, FLATTEN_SMOOTH_MAX, where),
    }


def _apply_flatten(image: BaseImage, source: dict, where: str) -> None:
    parsed = _parse_flatten(source, where)
    image.flatten_enabled = parsed["flatten_enabled"]
    image.flatten_count = parsed["flatten_count"]
    image.flatten_cutoffs = parsed["flatten_cutoffs"]
    image.flatten_colors = parsed["flatten_colors"]
    image.flatten_smooth = parsed["flatten_smooth"]


def _attach_layers(project: Project, images: list[tuple[BaseImage, list[tuple[Layer, list[int]]]]]) -> None:
    """Give each image its layers and remap negative indexes to fresh ids."""
    flat: list[Layer] = []
    pending: list[tuple[Layer, list[int]]] = []
    for image, parsed in images:
        for layer, sources in parsed:
            image.layers.append(layer)
            flat.append(layer)
            pending.append((layer, sources))
    for layer, sources in pending:
        layer.negative_of = [flat[s].id for s in sources]
    try:
        project.dependency_order([layer.id for layer in flat])
    except ProjectError as e:
        raise ProjectFileError(str(e)) from None


def project_from_dict(data: Any, project_dir: Path | None = None) -> Project:
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ProjectFileError("This is not a png_to_gerber project file.")
    version = data.get("version")
    if not isinstance(version, int) or version < 1 or version > VERSION:
        raise ProjectFileError(
            f"Unsupported project version {version!r} (this app reads up to {VERSION})."
        )
    if version >= 2:
        if not isinstance(data.get("images"), list):
            raise ProjectFileError(
                f"Unsupported project version {version!r} (this app reads up to {VERSION})."
            )
        return _project_from_v2(data, project_dir)
    return _project_from_v1(data, project_dir)


def _project_from_v1(data: dict, project_dir: Path | None) -> Project:
    raw_layers = data.get("layers", [])
    if not isinstance(raw_layers, list):
        raise ProjectFileError("'layers' must be a list.")
    parsed = [_parse_layer(d, i, len(raw_layers)) for i, d in enumerate(raw_layers)]

    image_info = data.get("image") or {}
    if not isinstance(image_info, dict):
        raise ProjectFileError("'image' must be an object.")
    background = data.get("background") or {}
    if not isinstance(background, dict):
        raise ProjectFileError("'background' must be an object.")
    view = data.get("view") or {}
    if not isinstance(view, dict):
        raise ProjectFileError("'view' must be an object.")

    path = _resolve_image_path(image_info, project_dir)
    has_image = path is not None or bool(parsed)
    project = Project(
        board=_parse_board(data),
        checker_opacity=_parse_checker(data),
    )
    if has_image or _get_bool(background, "removed", False, "background"):
        image = BaseImage(
            name=_image_label(path),
            path=path,
            background_removed=_get_bool(background, "removed", False, "background"),
            background_tolerance=_get_int(
                background, "tolerance", DEFAULT_BG_TOLERANCE, 0, 255, "background"
            ),
            visible=_get_bool(view, "base_visible", True, "view"),
        )
        _apply_flatten(image, data, "flatten")
        project.images.append(image)
        _attach_layers(project, [(image, parsed)])
    return project


def _project_from_v2(data: dict, project_dir: Path | None) -> Project:
    raw_images = data["images"]
    flat_raw: list[Any] = []
    spans: list[tuple[dict, int]] = []
    for index, raw in enumerate(raw_images):
        where = f"images[{index}]"
        if not isinstance(raw, dict):
            raise ProjectFileError(f"{where} must be an object.")
        raw_layers = raw.get("layers", [])
        if not isinstance(raw_layers, list):
            raise ProjectFileError(f"{where}: 'layers' must be a list.")
        spans.append((raw, len(raw_layers)))
        flat_raw.extend(raw_layers)

    parsed = [_parse_layer(entry, i, len(flat_raw)) for i, entry in enumerate(flat_raw)]
    project = Project(board=_parse_board(data), checker_opacity=_parse_checker(data))
    groups: list[tuple[BaseImage, list[tuple[Layer, list[int]]]]] = []
    offset = 0
    for image_index, (raw, count) in enumerate(spans):
        path = _resolve_image_path(raw, project_dir)
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            name = _image_label(path)
        background = raw.get("background") or {}
        if not isinstance(background, dict):
            raise ProjectFileError("'background' must be an object.")
        image = BaseImage(
            name=name.strip(),
            path=path,
            background_removed=_get_bool(background, "removed", False, "background"),
            background_tolerance=_get_int(
                background, "tolerance", DEFAULT_BG_TOLERANCE, 0, 255, "background"
            ),
            visible=_get_bool(raw, "visible", True, "images"),
        )
        _apply_flatten(image, raw, f"images[{image_index}].flatten")
        path = _resolve_image_path(raw, project_dir)
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            name = _image_label(path)
        background = raw.get("background") or {}
        if not isinstance(background, dict):
            raise ProjectFileError("'background' must be an object.")
        image = BaseImage(
            name=name.strip(),
            path=path,
            background_removed=_get_bool(background, "removed", False, "background"),
            background_tolerance=_get_int(
                background, "tolerance", DEFAULT_BG_TOLERANCE, 0, 255, "background"
            ),
            visible=_get_bool(raw, "visible", True, "images"),
        )
        _apply_flatten(image, raw, f"images[{index}].flatten")
        project.images.append(image)
        groups.append((image, parsed[offset : offset + count]))
        offset += count
    _attach_layers(project, groups)
    return project


def _image_label(path: str | None) -> str:
    if not path:
        return "Base image"
    return Path(path).name or "Base image"


def load_project(path: str | Path) -> Project:
    """Read a .prj file. Raises ProjectFileError/OSError on failure."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ProjectFileError(f"The project file is not valid JSON: {e}") from None
    except UnicodeDecodeError:
        raise ProjectFileError("The project file is not a text file.") from None
    return project_from_dict(data, path.parent.absolute())
