"""The exported Gerber files must reproduce the masks shown in the app exactly.

Each test replays the written file the way a CAM tool does (see
gerber_reader.py) and compares the painted result to the mask pixel by pixel.
"""

import re

import cv2
import numpy as np
import pytest
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon, box

from gerber_reader import mismatch, rasterize, read_regions
from png_to_gerber.geometry import iter_polygons, mask_to_geometry
from png_to_gerber.gerber import (
    BoardTransform,
    export_project,
    write_artwork_gerber,
)
from png_to_gerber.gui.main_window import MainWindow
from png_to_gerber.model import Layer, LayerType, Project, Side

from conftest import BOARD_COLOR, COPPER_COLOR, MASK_COLOR, SILK_COLOR


def block(mask, y0, y1, x0, x1, value=255):
    mask[y0:y1, x0:x1] = value
    return mask


def blank(h=100, w=100):
    return np.zeros((h, w), dtype=np.uint8)


def case_solid():
    return block(blank(), 10, 90, 10, 90)


def case_ring():
    return block(block(blank(), 10, 90, 10, 90), 30, 70, 30, 70, 0)


def case_island_in_hole():
    return block(case_ring(), 45, 55, 45, 55)


def case_hole_in_island():
    return block(case_island_in_hole(), 48, 52, 48, 52, 0)


def case_three_levels_deep():
    m = blank(120, 120)
    for inset, value in ((5, 255), (15, 0), (25, 255), (35, 0), (45, 255)):
        block(m, inset, 120 - inset, inset, 120 - inset, value)
    return m


def case_one_px_frame():
    return block(block(blank(), 10, 90, 10, 90), 11, 89, 11, 89, 0)


def case_two_px_frame():
    return block(block(blank(), 10, 90, 10, 90), 12, 88, 12, 88, 0)


def case_one_px_line():
    return block(blank(), 50, 51, 10, 90)


def case_one_px_slit():
    return block(block(blank(), 10, 90, 10, 90), 40, 41, 20, 80, 0)


def case_adjacent_holes():
    m = block(blank(), 10, 90, 10, 90)
    block(m, 30, 34, 30, 34, 0)
    return block(m, 30, 34, 35, 39, 0)  # one-pixel wall between two holes


def case_hole_touching_edge_wall():
    m = block(blank(), 10, 90, 10, 90)
    return block(m, 11, 20, 11, 20, 0)  # hole separated from outside by 1px


def case_diagonal_touch():
    return block(block(blank(), 20, 40, 20, 40), 40, 60, 40, 60)


def case_diagonal_hole_pinch():
    m = block(blank(), 10, 90, 10, 90)
    block(m, 30, 40, 30, 40, 0)
    return block(m, 40, 50, 40, 50, 0)  # two holes touching at one corner


def case_circle_rings():
    m = blank(200, 200)
    cv2.circle(m, (100, 100), 90, 255, -1)
    cv2.circle(m, (100, 100), 60, 0, -1)
    cv2.circle(m, (100, 100), 30, 255, -1)
    cv2.circle(m, (100, 100), 10, 0, -1)
    return m


def case_noise():
    rng = np.random.default_rng(0)
    return (rng.random((100, 100)) > 0.5).astype(np.uint8) * 255


CASES = {
    name[5:]: fn
    for name, fn in list(globals().items())
    if name.startswith("case_") and callable(fn)
}


def export_and_replay(mask, tmp_path, width_mm=None, height_mm=None):
    h, w = mask.shape
    width_mm = width_mm or w * 0.25
    height_mm = height_mm or h * 0.25
    path = tmp_path / "t.gtl"
    write_artwork_gerber(
        path, mask_to_geometry(mask), BoardTransform(w, h, width_mm, height_mm),
        "Copper,L1,Top",
    )
    return rasterize(read_regions(path, w, h, width_mm, height_mm), w, h)


# ---- geometry is pixel-exact -----------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CASES))
def test_geometry_covers_exactly_the_mask_pixels(name):
    mask = CASES[name]()
    geom = mask_to_geometry(mask)
    assert geom.is_valid
    assert geom.area == int((mask > 0).sum())
    assert mismatch(mask, rasterize(geom, mask.shape[1], mask.shape[0])) == 0


def test_straight_edges_have_no_redundant_vertices():
    geom = mask_to_geometry(case_solid())
    assert len(list(geom.exterior.coords)) == 5  # rectangle: 4 corners + closure


def test_one_pixel_line_is_kept():
    geom = mask_to_geometry(case_one_px_line())
    assert geom.area == 80


def test_holes_and_islands_are_all_present():
    polys = list(iter_polygons(mask_to_geometry(case_hole_in_island())))
    assert len(polys) == 2  # ring and island
    assert sorted(len(p.interiors) for p in polys) == [1, 1]


# ---- written Gerber reproduces the mask ------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CASES))
def test_exported_gerber_matches_mask_exactly(name, tmp_path):
    mask = CASES[name]()
    painted = export_and_replay(mask, tmp_path)
    assert mismatch(mask, painted) == 0


@pytest.mark.parametrize("size", [(33.3, 21.7), (100.0, 100.0), (7.77, 90.1)])
def test_matches_for_awkward_board_scales(size, tmp_path):
    """Rounding to 1e-6 mm must never flip a pixel, whatever the scale."""
    mask = case_circle_rings()
    assert mismatch(mask, export_and_replay(mask, tmp_path, *size)) == 0


# ---- holes are clear regions, because GerbView fills every contour solid ----------------


def _regions(text: str) -> int:
    return text.count("G36*")


def test_hole_is_a_clear_region_after_the_outline(tmp_path):
    """GerbView paints a second contour solid, so a hole must be its own clear region."""
    path = tmp_path / "t.gtl"
    write_artwork_gerber(
        path, mask_to_geometry(case_ring()), BoardTransform(100, 100, 25, 25),
        "Copper,L1,Top",
    )
    text = path.read_text()
    assert _regions(text) == 2
    assert text.index("%LPD*%") < text.index("%LPC*%") < text.rindex("%LPD*%")
    painted = rasterize(read_regions(path, 100, 100, 25, 25), 100, 100)
    assert painted[15, 15] and not painted[50, 50]


def test_island_is_painted_after_the_clear_that_would_erase_it(tmp_path):
    path = tmp_path / "t.gtl"
    write_artwork_gerber(
        path, mask_to_geometry(case_island_in_hole()), BoardTransform(100, 100, 25, 25),
        "Copper,L1,Top",
    )
    text = path.read_text()
    # ring outline (dark), its hole (clear), then the island (dark again).
    assert _regions(text) == 3
    assert text.index("%LPC*%") < text.rindex("%LPD*%")
    painted = rasterize(read_regions(path, 100, 100, 25, 25), 100, 100)
    assert painted[50, 50], "island inside the hole must be painted"
    assert not painted[35, 35]  # the hole itself is still open
    assert painted[15, 15]  # the ring itself is dark


def test_island_listed_before_its_ring_is_still_painted(tmp_path):
    """A clear punches everything already drawn, so the island must be written after it."""
    ring = Polygon(box(10, 10, 90, 90).exterior, [box(30, 30, 70, 70).exterior])
    island = box(45, 45, 55, 55)
    adversarial = MultiPolygon([island, ring])  # island deliberately first
    path = tmp_path / "t.gtl"
    write_artwork_gerber(
        path, adversarial, BoardTransform(100, 100, 25, 25), "Copper,L1,Top"
    )
    painted = rasterize(read_regions(path, 100, 100, 25, 25), 100, 100)
    assert painted[50, 50], "island inside the hole was erased by the hole's clear"
    assert not painted[35, 35]
    assert painted[15, 15]


def test_second_contour_without_clear_polarity_would_fill_the_hole(tmp_path):
    """Documents the GerbView behavior: a contour that is not %LPC% is solid paint."""
    path = tmp_path / "t.gtl"
    write_artwork_gerber(
        path, mask_to_geometry(case_ring()), BoardTransform(100, 100, 25, 25),
        "Copper,L1,Top",
    )
    # Drop the clear command. GerbView then fills the hole contour too.
    broken = path.read_text().replace("%LPC*%\n", "")
    path.write_text(broken)
    painted = rasterize(read_regions(path, 100, 100, 25, 25), 100, 100)
    assert painted[50, 50], "without LPC the hole contour is filled"


def test_duplicate_vertices_after_rounding_are_collapsed(tmp_path):
    """A degenerate extra vertex must not be emitted (it would break the contour)."""
    poly = Polygon([(10, 10), (10, 10), (90, 10), (90, 90), (10, 90), (10, 10)])
    path = tmp_path / "t.gtl"
    write_artwork_gerber(path, poly, BoardTransform(100, 100, 25, 25), "Copper,L1,Top")
    # 4 unique corners + 1 explicit close = 1 D02 + 4 D01, not a zero-length spike.
    draws = [l for l in path.read_text().splitlines() if l.endswith("D01*") or l.endswith("D02*")]
    assert len(draws) == 5


# ---- file-level correctness ----------------------------------------------------------------------


def test_linear_interpolation_is_selected_explicitly(tmp_path):
    path = tmp_path / "t.gtl"
    write_artwork_gerber(path, mask_to_geometry(case_solid()), BoardTransform(100, 100, 25, 25), "Copper,L1,Top")
    lines = path.read_text().splitlines()
    first_draw = next(i for i, l in enumerate(lines) if "D01*" in l)
    assert "G01*" in lines[:first_draw]


def _project_with(*specs):
    project = Project()
    geoms = {}
    for layer_type, side, mask in specs:
        layer = project.add_layer(
            Layer(name="l", layer_type=layer_type, side=side, has_color=True)
        )
        geoms[layer.id] = mask_to_geometry(mask)
    return project, geoms


def test_mask_files_are_marked_negative_and_others_positive(tmp_path):
    project, geoms = _project_with(
        (LayerType.COPPER, Side.FRONT, case_solid()),
        (LayerType.MASK, Side.FRONT, case_solid()),
        (LayerType.SILKSCREEN, Side.FRONT, case_solid()),
    )
    files = export_project(project, geoms, None, tmp_path, "b", BoardTransform(100, 100, 25, 25))
    polarity = {
        p.suffix: re.search(r"FilePolarity,(\w+)", p.read_text()).group(1) for p in files
    }
    assert polarity == {".gtl": "Positive", ".gts": "Negative", ".gto": "Positive"}


def test_merged_layers_reproduce_union_of_masks(tmp_path):
    a = block(blank(), 10, 60, 10, 60)
    b = block(blank(), 40, 90, 40, 90)  # overlaps a
    c = block(blank(), 20, 30, 70, 95)  # separate
    project, geoms = _project_with(
        (LayerType.COPPER, Side.FRONT, a),
        (LayerType.COPPER, Side.FRONT, b),
        (LayerType.COPPER, Side.FRONT, c),
    )
    files = export_project(project, geoms, None, tmp_path, "b", BoardTransform(100, 100, 25, 25))
    painted = rasterize(read_regions(files[0], 100, 100, 25, 25), 100, 100)
    assert mismatch(a | b | c, painted) == 0


# ---- through the real GUI pipeline ------------------------------------------------------------------


@pytest.fixture
def art_image_path(tmp_path):
    """Concentric squares (island inside a hole), thin lines, and a plain block."""
    img = np.zeros((160, 200, 4), dtype=np.uint8)
    img[..., 3] = 255
    img[..., :3] = BOARD_COLOR
    img[10:110, 10:110, :3] = COPPER_COLOR  # ring outer
    img[25:95, 25:95, :3] = BOARD_COLOR  # hole
    img[40:80, 40:80, :3] = COPPER_COLOR  # island in the hole
    img[55:65, 55:65, :3] = BOARD_COLOR  # hole inside the island
    img[20, 120:190, :3] = SILK_COLOR  # 1px line
    img[30:32, 120:190, :3] = SILK_COLOR  # 2px line
    img[50:90, 130:190, :3] = MASK_COLOR
    img[60:70, 140:150, :3] = BOARD_COLOR  # hole in mask block
    path = tmp_path / "art.png"
    Image.fromarray(img, "RGBA").save(path)
    return path


@pytest.fixture
def art_window(qtbot, art_image_path):
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(art_image_path)
    for layer_type, color in (
        (LayerType.COPPER, COPPER_COLOR),
        (LayerType.SILKSCREEN, SILK_COLOR),
        (LayerType.MASK, MASK_COLOR),
    ):
        win.add_layer(layer_type)
        win.inspector.apply_color(color)
        win.project.layers[-1].min_island = 0  # keep the 1px lines in the mask
        win.project.layers[-1].close_radius = 0
    win.flush_recompute()
    return win


def _replay(path, win, width_mm, height_mm):
    h, w = win.image.shape[:2]
    return rasterize(read_regions(path, w, h, width_mm, height_mm), w, h)


def test_gui_export_reproduces_every_layer_exactly(art_window, tmp_path):
    win = art_window
    written = win.export_gerbers(tmp_path, "art", 80.0, 64.0)
    by_suffix = {p.suffix: p for p in written}
    expected = {
        ".gtl": win.project.layers[0],
        ".gto": win.project.layers[1],
        ".gts": win.project.layers[2],
    }
    for suffix, layer in expected.items():
        mask = win.masks[layer.id]
        assert mask.max() == 255
        assert mismatch(mask, _replay(by_suffix[suffix], win, 80.0, 64.0)) == 0, suffix


def test_gui_export_keeps_nested_island_and_thin_lines(art_window, tmp_path):
    win = art_window
    written = win.export_gerbers(tmp_path, "art", 80.0, 64.0)
    copper = _replay(next(p for p in written if p.suffix == ".gtl"), win, 80.0, 64.0)
    assert copper[60, 45], "island inside the hole must be exported"
    assert not copper[60, 60], "hole inside the island must stay open"
    assert not copper[30, 30], "hole around the island must stay open"
    silk = _replay(next(p for p in written if p.suffix == ".gto"), win, 80.0, 64.0)
    assert silk[20, 150], "1px line must survive export"
    assert silk[30, 150] and silk[31, 150], "2px line must keep both rows"
    mask = _replay(next(p for p in written if p.suffix == ".gts"), win, 80.0, 64.0)
    assert not mask[65, 145] and mask[55, 135]


def test_gui_negative_layer_exports_exactly(art_window, tmp_path):
    win = art_window
    copper = win.project.layers[0]
    neg = win.generate_negative([copper.id], LayerType.MASK)
    neg.side = Side.BACK
    win.flush_recompute()
    written = win.export_gerbers(tmp_path, "art", 80.0, 64.0)
    gbs = next(p for p in written if p.suffix == ".gbs")
    assert mismatch(win.masks[neg.id], _replay(gbs, win, 80.0, 64.0)) == 0


def test_gui_outline_vertices_match_computed_outline(art_window, tmp_path):
    win = art_window
    win.add_layer(LayerType.CUT)
    win.inspector.apply_color(COPPER_COLOR)
    win.flush_recompute()
    cut = win.project.cut_layer()
    outline = win.outlines[cut.id]
    written = win.export_gerbers(tmp_path, "art", 80.0, 64.0)
    text = next(p for p in written if p.suffix == ".gm1").read_text()
    h, w = win.image.shape[:2]
    points = [
        (int(x) / 1e6 * w / 80.0, h - int(y) / 1e6 * h / 64.0)
        for x, y in re.findall(r"X(-?\d+)Y(-?\d+)D0[12]\*", text)
    ]
    assert points[0] == pytest.approx(points[-1], abs=1e-4)  # closed
    assert Polygon(points).area == pytest.approx(outline.area, rel=1e-4)
    assert Polygon(points).symmetric_difference(outline).area < 1e-2 * outline.area
