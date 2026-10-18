"""Gerber X2 export compatible with KiCad 10's Gerber viewer.

Conventions:
- Millimeters, absolute coordinates, 4.6 format (FSLAX46Y46).
- Artwork is written as G36/G37 region fills, one contour per region. KiCad's
  Gerber viewer does not punch holes with a second contour inside the same
  region: a D02 there ends the current polygon and the next contour is painted
  as its own solid fill (see gerbview rs274d.cpp, D02 in polygon mode). Holes
  are therefore separate regions drawn with clear polarity (%LPC%) after the
  outline they cut. An island inside a hole is a dark region drawn after that
  clear, so the clear does not erase it.
- The board outline (Edge.Cuts) is a closed stroked path with a thin circular
  aperture and the Profile,NP file function.
- X2 attributes (.FileFunction, .FilePolarity) let KiCad identify each layer
  automatically on import.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shapely.geometry import Polygon
from shapely.ops import unary_union

from .geometry import iter_polygons
from .model import LayerType, Project, Side

# (type, side) -> (filename suffix, extension, X2 FileFunction)
GERBER_FILES: dict[tuple[LayerType, Side], tuple[str, str, str]] = {
    (LayerType.COPPER, Side.FRONT): ("F_Cu", "gtl", "Copper,L1,Top"),
    (LayerType.COPPER, Side.BACK): ("B_Cu", "gbl", "Copper,L2,Bot"),
    (LayerType.MASK, Side.FRONT): ("F_Mask", "gts", "Soldermask,Top"),
    (LayerType.MASK, Side.BACK): ("B_Mask", "gbs", "Soldermask,Bot"),
    (LayerType.SILKSCREEN, Side.FRONT): ("F_Silkscreen", "gto", "Legend,Top"),
    (LayerType.SILKSCREEN, Side.BACK): ("B_Silkscreen", "gbo", "Legend,Bot"),
}

EDGE_CUTS_FILE = ("Edge_Cuts", "gm1", "Profile,NP")

OUTLINE_WIDTH_MM = 0.10
# Some viewers refuse to plot regions until an aperture is selected, even
# though the spec says the current aperture does not affect G36/G37.
REGION_APERTURE_MM = 0.010


class ExportError(ValueError):
    pass


@dataclass(frozen=True)
class BoardTransform:
    """Maps image pixel coordinates to board millimeters (y flipped)."""

    image_w: int
    image_h: int
    width_mm: float
    height_mm: float

    def __post_init__(self):
        if self.image_w <= 0 or self.image_h <= 0:
            raise ExportError("Image dimensions must be positive.")
        if self.width_mm <= 0 or self.height_mm <= 0:
            raise ExportError("Board width and height must be positive.")

    def to_mm(self, x_px: float, y_px: float) -> tuple[float, float]:
        x = x_px * self.width_mm / self.image_w
        y = (self.image_h - y_px) * self.height_mm / self.image_h
        return x, y

    def to_gerber(self, x_px: float, y_px: float) -> tuple[int, int]:
        x_mm, y_mm = self.to_mm(x_px, y_px)
        return int(round(x_mm * 1_000_000)), int(round(y_mm * 1_000_000))


def _coord(xy: tuple[int, int]) -> str:
    return f"X{xy[0]}Y{xy[1]}"


def _header(file_function: str, polarity: str | None) -> list[str]:
    lines = [
        f"%TF.FileFunction,{file_function}*%",
    ]
    if polarity:
        lines.append(f"%TF.FilePolarity,{polarity}*%")
    lines += [
        "%TF.GenerationSoftware,png_to_gerber,png_to_gerber,0.1.0*%",
        "%FSLAX46Y46*%",
        "%MOMM*%",
        "G01*",  # linear interpolation must be selected explicitly before any D01
        "G75*",  # multi-quadrant mode (harmless for linear-only files)
        f"%ADD10C,{REGION_APERTURE_MM:.3f}*%",
        "D10*",
        "%LPD*%",
    ]
    return lines


def _closed_ring(transform: BoardTransform, ring) -> list[tuple[int, int]]:
    """Integer Gerber vertices of a closed ring, without consecutive duplicates."""
    pts: list[tuple[int, int]] = []
    for x, y in ring.coords:
        p = transform.to_gerber(x, y)
        if not pts or p != pts[-1]:
            pts.append(p)
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts.pop()  # we'll close the contour explicitly
    return pts if len(pts) >= 3 else []


def _contour_commands(pts: list[tuple[int, int]]) -> list[str]:
    """D02 to the start, D01 around the ring, D01 back to the start."""
    return (
        [f"{_coord(pts[0])}D02*"]
        + [f"{_coord(p)}D01*" for p in pts[1:]]
        + [f"{_coord(pts[0])}D01*"]
    )


def _single_region(transform: BoardTransform, ring) -> list[str]:
    """One solid G36/G37 region. KiCad fills each of these on its own."""
    pts = _closed_ring(transform, ring)
    if not pts:
        return []
    return ["G36*", *_contour_commands(pts), "G37*"]


def _paint_order(polygons: list[Polygon]) -> list[Polygon]:
    """Outside-in order, so a clear that opens a hole is painted before the island inside it."""
    if len(polygons) < 2:
        return polygons
    # contains() is False for a point in a hole, so test against the filled footprint.
    footprints = [Polygon(poly.exterior) for poly in polygons]
    depths = []
    for poly in polygons:
        probe = poly.representative_point()
        depths.append(sum(1 for foot in footprints if foot.contains(probe)) - 1)
    return [polygons[i] for i in sorted(range(len(polygons)), key=lambda i: depths[i])]


def _artwork_regions(transform: BoardTransform, geometry) -> list[str]:
    """Dark outline, then a clear region per hole, islands after the clear that surrounds them."""
    lines: list[str] = []
    for poly in _paint_order(list(iter_polygons(geometry))):
        outline = _single_region(transform, poly.exterior)
        if not outline:
            continue
        lines.append("%LPD*%")
        lines += outline
        for interior in poly.interiors:
            hole = _single_region(transform, interior)
            if hole:
                lines.append("%LPC*%")
                lines += hole
    lines.append("%LPD*%")
    return lines


def write_artwork_gerber(
    path: Path,
    geometry,
    transform: BoardTransform,
    file_function: str,
    polarity: str = "Positive",
) -> None:
    """Write filled regions (holes cleared afterwards) as one Gerber artwork file."""
    lines = _header(file_function, polarity)
    lines += _artwork_regions(transform, geometry)
    lines.append("M02*")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_outline_gerber(
    path: Path, outline: Polygon, transform: BoardTransform
) -> None:
    """Write a closed stroked perimeter as the Edge.Cuts profile."""
    pts = _closed_ring(transform, outline.exterior)
    if len(pts) < 3:
        raise ExportError("The board outline is not a closed polygon.")
    lines = _header(EDGE_CUTS_FILE[2], None)
    # Replace the dummy region aperture with the visible outline width.
    lines = [l for l in lines if not l.startswith("%ADD10") and l != "D10*"]
    lines += [
        f"%ADD10C,{OUTLINE_WIDTH_MM:.3f}*%",
        "D10*",
        f"{_coord(pts[0])}D02*",
    ]
    lines += [f"{_coord(p)}D01*" for p in pts[1:]]
    lines.append(f"{_coord(pts[0])}D01*")
    lines.append("M02*")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def export_project(
    project: Project,
    layer_geometries: dict[int, object],
    cut_outline_polygon: Polygon | None,
    out_dir: str | Path,
    base_name: str,
    transform: BoardTransform,
) -> list[Path]:
    """Export the project to Gerber files; returns the written file paths.

    `layer_geometries` maps layer id -> shapely geometry (image px coords).
    Visible artwork layers with the same (type, side) are unioned into one
    standard file. The cut outline, when provided, becomes Edge_Cuts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for (layer_type, side), layers in project.grouped_for_export().items():
        geoms = [
            layer_geometries[l.id]
            for l in layers
            if l.id in layer_geometries and not layer_geometries[l.id].is_empty
        ]
        if not geoms:
            continue
        merged = unary_union(geoms)
        if merged.is_empty:
            continue
        suffix, ext, file_function = GERBER_FILES[(layer_type, side)]
        path = out_dir / f"{base_name}-{suffix}.{ext}"
        # Mask files draw the *openings*, which X2 calls a negative image.
        file_polarity = "Negative" if layer_type == LayerType.MASK else "Positive"
        write_artwork_gerber(path, merged, transform, file_function, file_polarity)
        written.append(path)

    if cut_outline_polygon is not None:
        suffix, ext, _ = EDGE_CUTS_FILE
        path = out_dir / f"{base_name}-{suffix}.{ext}"
        write_outline_gerber(path, cut_outline_polygon, transform)
        written.append(path)

    if not written:
        raise ExportError(
            "Nothing to export: no visible layer produced any geometry."
        )
    return written
