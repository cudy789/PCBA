"""Tests for the Gerber X2 writer and export grouping."""

import numpy as np
import pytest
from shapely.geometry import Polygon

from png_to_gerber.gerber import (
    BoardTransform,
    EDGE_CUTS_FILE,
    ExportError,
    GERBER_FILES,
    export_project,
    write_artwork_gerber,
    write_outline_gerber,
)
from png_to_gerber.model import Layer, LayerType, Project, Side


def make_transform(w=100, h=50, width_mm=50.0, height_mm=25.0):
    return BoardTransform(w, h, width_mm, height_mm)


def square(x0, y0, x1, y1, holes=None):
    return Polygon(
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], holes or []
    )


def test_transform_scales_and_flips_y():
    t = make_transform()
    assert t.to_mm(0, 50) == (0.0, 0.0)  # bottom-left pixel -> origin
    assert t.to_mm(100, 0) == (50.0, 25.0)  # top-right pixel -> max mm
    x, y = t.to_mm(50, 25)
    assert (x, y) == (25.0, 12.5)


def test_transform_rejects_bad_dimensions():
    with pytest.raises(ExportError):
        BoardTransform(100, 50, 0, 25.0)
    with pytest.raises(ExportError):
        BoardTransform(0, 50, 50.0, 25.0)


def test_artwork_gerber_structure(tmp_path):
    path = tmp_path / "test.gtl"
    geom = square(10, 10, 40, 40, holes=[[(20, 20), (30, 20), (30, 30), (20, 30)]])
    write_artwork_gerber(path, geom, make_transform(), "Copper,L1,Top")
    text = path.read_text()
    assert "%TF.FileFunction,Copper,L1,Top*%" in text
    assert "%FSLAX46Y46*%" in text
    assert "%MOMM*%" in text
    assert text.count("G36*") == 2  # outline, then the hole as its own region
    assert text.count("G37*") == 2
    assert "%LPC*%" in text  # the hole is cleared after the outline is painted
    assert text.index("%LPD*%") < text.index("%LPC*%")
    assert "G01*" in text and "G75*" in text
    assert "%ADD10C,0.010*%" in text
    assert text.rstrip().endswith("M02*")


def test_artwork_region_closed(tmp_path):
    path = tmp_path / "test.gtl"
    write_artwork_gerber(path, square(0, 0, 10, 10), make_transform(), "Copper,L1,Top")
    lines = path.read_text().splitlines()
    start = next(l for l in lines if l.endswith("D02*"))
    d01s = [l for l in lines if l.endswith("D01*")]
    assert start.replace("D02*", "") == d01s[-1].replace("D01*", "")


def test_outline_gerber_structure(tmp_path):
    path = tmp_path / "test.gm1"
    write_outline_gerber(path, square(5, 5, 95, 45), make_transform())
    text = path.read_text()
    assert "%TF.FileFunction,Profile,NP*%" in text
    assert "%ADD10C,0.100*%" in text
    lines = text.splitlines()
    start = next(l for l in lines if l.endswith("D02*"))
    d01s = [l for l in lines if l.endswith("D01*")]
    assert start.replace("D02*", "") == d01s[-1].replace("D01*", "")
    assert text.rstrip().endswith("M02*")


def test_outline_requires_closed_polygon(tmp_path):
    degenerate = Polygon()
    with pytest.raises((ExportError, AttributeError, IndexError)):
        write_outline_gerber(tmp_path / "x.gm1", degenerate, make_transform())


def _project_with(*specs):
    """specs: (layer_type, side, geometry) -> (project, geometries dict)."""
    project = Project()
    geometries = {}
    for layer_type, side, geom in specs:
        layer = project.add_layer(
            Layer(name="l", layer_type=layer_type, side=side, has_color=True)
        )
        geometries[layer.id] = geom
    return project, geometries


def test_export_project_writes_expected_files(tmp_path):
    project, geoms = _project_with(
        (LayerType.COPPER, Side.FRONT, square(0, 0, 10, 10)),
        (LayerType.COPPER, Side.BACK, square(0, 0, 10, 10)),
        (LayerType.MASK, Side.FRONT, square(0, 0, 10, 10)),
        (LayerType.SILKSCREEN, Side.BACK, square(0, 0, 10, 10)),
    )
    outline = square(0, 0, 99, 49)
    written = export_project(
        project, geoms, outline, tmp_path, "board", make_transform()
    )
    names = sorted(p.name for p in written)
    assert names == [
        "board-B_Cu.gbl",
        "board-B_Silkscreen.gbo",
        "board-Edge_Cuts.gm1",
        "board-F_Cu.gtl",
        "board-F_Mask.gts",
    ]
    for path in written:
        assert path.read_text().rstrip().endswith("M02*")


def test_export_merges_same_type_and_side(tmp_path):
    project, geoms = _project_with(
        (LayerType.COPPER, Side.FRONT, square(0, 0, 10, 10)),
        (LayerType.COPPER, Side.FRONT, square(50, 20, 70, 40)),
    )
    written = export_project(project, geoms, None, tmp_path, "b", make_transform())
    assert len(written) == 1
    text = written[0].read_text()
    assert text.count("G36*") == 2  # two disjoint regions in one merged file


def test_export_skips_hidden_layers(tmp_path):
    project, geoms = _project_with(
        (LayerType.COPPER, Side.FRONT, square(0, 0, 10, 10)),
        (LayerType.SILKSCREEN, Side.FRONT, square(0, 0, 10, 10)),
    )
    project.layers[1].visible = False
    written = export_project(project, geoms, None, tmp_path, "b", make_transform())
    assert [p.suffix for p in written] == [".gtl"]


def test_export_nothing_raises(tmp_path):
    project = Project()
    with pytest.raises(ExportError):
        export_project(project, {}, None, tmp_path, "b", make_transform())


def test_export_creates_output_directory(tmp_path):
    project, geoms = _project_with(
        (LayerType.COPPER, Side.FRONT, square(0, 0, 10, 10)),
    )
    out = tmp_path / "nested" / "dir"
    written = export_project(project, geoms, None, out, "b", make_transform())
    assert written[0].parent == out


def test_gerber_files_table_covers_all_sided_types():
    for layer_type in (LayerType.COPPER, LayerType.MASK, LayerType.SILKSCREEN):
        for side in Side:
            assert (layer_type, side) in GERBER_FILES
    assert EDGE_CUTS_FILE[1] == "gm1"


def test_coordinates_written_in_4_6_format(tmp_path):
    path = tmp_path / "t.gtl"
    write_artwork_gerber(
        path, square(0, 0, 100, 50), make_transform(), "Copper,L1,Top"
    )
    # 50 mm -> 50,000,000 in 4.6 units.
    assert "X50000000" in path.read_text()
