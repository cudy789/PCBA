""".prj project files: JSON snapshots of layers, board size, and view state.

Layers reference each other (negative layers) by their index in the saved
list, so reloading assigns fresh runtime ids without breaking the references.
The source image is stored by absolute path plus a path relative to the
project file, so a project folder can be moved or shared along with its image.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .model import (
    DEFAULT_BG_TOLERANCE,
    BoardSettings,
    Layer,
    LayerType,
    OutlineMode,
    Project,
    ProjectError,
    Side,
)

FORMAT = "png_to_gerber_project"
VERSION = 1
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


def project_to_dict(project: Project, project_dir: Path | None = None) -> dict[str, Any]:
    index_of = {layer.id: i for i, layer in enumerate(project.layers)}
    image: dict[str, Any] = {"absolute": None, "relative": None}
    if project.image_path:
        absolute = Path(project.image_path).absolute()
        image["absolute"] = str(absolute)
        if project_dir is not None:
            try:
                image["relative"] = os.path.relpath(absolute, project_dir)
            except ValueError:  # different drive on Windows
                image["relative"] = None
    board = project.board
    return {
        "format": FORMAT,
        "version": VERSION,
        "image": image,
        "background": {
            "removed": project.background_removed,
            "tolerance": project.background_tolerance,
        },
        "board": {
            "width_mm": board.width_mm,
            "height_mm": board.height_mm,
            "lock_aspect": board.lock_aspect,
            "output_dir": board.output_dir,
            "base_name": board.base_name,
        },
        "view": {
            "base_visible": project.base_visible,
            "checker_opacity": project.checker_opacity,
        },
        "layers": [_layer_to_dict(layer, index_of) for layer in project.layers],
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


def project_from_dict(data: Any, project_dir: Path | None = None) -> Project:
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ProjectFileError("This is not a png_to_gerber project file.")
    version = data.get("version")
    if not isinstance(version, int) or version < 1 or version > VERSION:
        raise ProjectFileError(
            f"Unsupported project version {version!r} (this app reads up to {VERSION})."
        )

    raw_layers = data.get("layers", [])
    if not isinstance(raw_layers, list):
        raise ProjectFileError("'layers' must be a list.")
    parsed = [_parse_layer(d, i, len(raw_layers)) for i, d in enumerate(raw_layers)]
    layers = [layer for layer, _ in parsed]
    for layer, sources in parsed:
        layer.negative_of = [layers[s].id for s in sources]

    project = Project()
    try:
        for layer in layers:
            project.layers.append(layer)
        if sum(1 for l in layers if l.is_cut) > 1:
            raise ProjectError("The project has more than one cut layer.")
        project.dependency_order([l.id for l in layers])  # rejects cycles
    except ProjectError as e:
        raise ProjectFileError(str(e)) from None

    image = data.get("image") or {}
    if not isinstance(image, dict):
        raise ProjectFileError("'image' must be an object.")
    candidates = []
    relative, absolute = image.get("relative"), image.get("absolute")
    if isinstance(relative, str) and relative and project_dir is not None:
        candidates.append(Path(project_dir) / relative)
    if isinstance(absolute, str) and absolute:
        candidates.append(Path(absolute))
    existing = next((c for c in candidates if c.is_file()), None)
    chosen = existing or (candidates[0] if candidates else None)
    project.image_path = str(chosen) if chosen is not None else None

    background = data.get("background") or {}
    if not isinstance(background, dict):
        raise ProjectFileError("'background' must be an object.")
    project.background_removed = _get_bool(background, "removed", False, "background")
    project.background_tolerance = _get_int(
        background, "tolerance", DEFAULT_BG_TOLERANCE, 0, 255, "background"
    )

    board = data.get("board") or {}
    if not isinstance(board, dict):
        raise ProjectFileError("'board' must be an object.")
    output_dir, base_name = board.get("output_dir", ""), board.get("base_name", "")
    if not isinstance(output_dir, str) or not isinstance(base_name, str):
        raise ProjectFileError("board: 'output_dir' and 'base_name' must be strings.")
    project.board = BoardSettings(
        width_mm=_get_optional_float(board, "width_mm", "board"),
        height_mm=_get_optional_float(board, "height_mm", "board"),
        lock_aspect=_get_bool(board, "lock_aspect", True, "board"),
        output_dir=output_dir,
        base_name=base_name,
    )

    view = data.get("view") or {}
    if not isinstance(view, dict):
        raise ProjectFileError("'view' must be an object.")
    project.base_visible = _get_bool(view, "base_visible", True, "view")
    project.checker_opacity = _get_int(view, "checker_opacity", 50, 0, 100, "view")
    return project


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
