"""Board outline: outer vs inner perimeter of the cut color."""

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon

from png_to_gerber.geometry import CutError, cut_outline
from png_to_gerber.gui.main_window import MainWindow
from png_to_gerber.model import LayerType, OutlineMode

FRAME = (200, 30, 30)
BOARD = (240, 240, 240)
ART = (20, 120, 20)


def frame_mask(size=100, outer=(10, 90), inner=(30, 70)) -> np.ndarray:
    """A solid square ring: outer edge at `outer`, hole spanning `inner`."""
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[outer[0] : outer[1], outer[0] : outer[1]] = 255
    mask[inner[0] : inner[1], inner[0] : inner[1]] = 0
    return mask


def bounds_span(poly: Polygon):
    minx, miny, maxx, maxy = poly.bounds
    return maxx - minx, maxy - miny


# ---- core ---------------------------------------------------------------------------


def test_outer_mode_traces_outside_edge_of_frame():
    outline = cut_outline(frame_mask(), smoothness=0, mode=OutlineMode.OUTER)
    w, h = bounds_span(outline)
    assert w == pytest.approx(79, abs=2) and h == pytest.approx(79, abs=2)
    assert outline.area == pytest.approx(80 * 80, rel=0.06)


def test_inner_mode_traces_inside_edge_of_frame():
    outline = cut_outline(frame_mask(), smoothness=0, mode=OutlineMode.INNER)
    w, h = bounds_span(outline)
    assert w == pytest.approx(40, abs=3) and h == pytest.approx(40, abs=3)
    assert outline.area == pytest.approx(40 * 40, rel=0.12)
    minx, miny, maxx, maxy = outline.bounds
    assert minx >= 28 and maxx <= 71  # inside the frame, away from outer edge


def test_default_mode_is_outer():
    default = cut_outline(frame_mask(), smoothness=0)
    outer = cut_outline(frame_mask(), smoothness=0, mode=OutlineMode.OUTER)
    assert default.equals(outer)


def test_inner_is_strictly_inside_outer():
    mask = frame_mask()
    outer = cut_outline(mask, 0, OutlineMode.OUTER)
    inner = cut_outline(mask, 0, OutlineMode.INNER)
    assert outer.contains(inner)


def test_mode_accepts_string_values():
    outline = cut_outline(frame_mask(), 0, "inner")
    assert outline.area < 2000


def test_inner_mode_requires_enclosed_area():
    solid = np.zeros((50, 50), dtype=np.uint8)
    solid[10:40, 10:40] = 255
    with pytest.raises(CutError, match="inner"):
        cut_outline(solid, 0, OutlineMode.INNER)


def test_inner_mode_open_frame_has_no_inner_perimeter():
    mask = frame_mask()
    mask[50, 10:30] = 0  # cut a gap through the left side of the ring
    mask[48:53, 10:30] = 0
    with pytest.raises(CutError):
        cut_outline(mask, 0, OutlineMode.INNER)
    # The outer perimeter still works (the ring is now just a C-shape).
    outer = cut_outline(mask, 0, OutlineMode.OUTER)
    assert outer.is_valid and outer.area > 0


def test_inner_mode_picks_largest_enclosed_area():
    mask = frame_mask(size=200, outer=(10, 190), inner=(30, 170))
    mask[60:80, 60:80] = 255  # island inside the board...
    mask[65:75, 65:75] = 0  # ...with its own small hole
    outline = cut_outline(mask, 0, OutlineMode.INNER)
    assert outline.area == pytest.approx(140 * 140, rel=0.08)


def test_inner_mode_picks_largest_hole_across_separate_rings():
    mask = frame_mask(size=200, outer=(10, 60), inner=(20, 50))  # 30x30 hole
    mask[100:190, 100:190] = 255
    mask[110:180, 110:180] = 0  # second ring with a larger 70x70 hole
    outline = cut_outline(mask, 0, OutlineMode.INNER)
    assert outline.bounds[0] > 100  # the big ring's hole, not the small one
    assert outline.area == pytest.approx(70 * 70, rel=0.1)


def test_outer_mode_prefers_frame_over_larger_solid_speck():
    """Outer perimeter is picked by footprint, so a thin frame beats a speck."""
    mask = frame_mask(size=200, outer=(10, 150), inner=(14, 146))  # thin frame
    mask[160:175, 160:175] = 255  # smaller solid block
    outline = cut_outline(mask, 0, OutlineMode.OUTER)
    assert outline.bounds[2] < 160


@pytest.mark.parametrize("mode", [OutlineMode.OUTER, OutlineMode.INNER])
def test_both_modes_closed_and_valid_at_any_smoothness(mode):
    for smoothness in (0, 30, 100):
        outline = cut_outline(frame_mask(), smoothness, mode)
        assert outline.is_valid and outline.area > 0
        coords = list(outline.exterior.coords)
        assert coords[0] == coords[-1]
        assert len(outline.interiors) == 0


def test_inner_mode_smoothing_reduces_jaggedness():
    mask = frame_mask(size=200, outer=(10, 190), inner=(40, 160))
    for i, y in enumerate(range(40, 160)):  # jagged teeth on the inner edge
        mask[y, 40 : 40 + (i % 5)] = 255
    raw = cut_outline(mask, 0, OutlineMode.INNER)
    smooth = cut_outline(mask, 70, OutlineMode.INNER)
    assert smooth.exterior.length < raw.exterior.length
    assert smooth.area == pytest.approx(raw.area, rel=0.2)


# ---- GUI -----------------------------------------------------------------------------


@pytest.fixture
def frame_image_path(tmp_path):
    """Board-colored artwork surrounded by a thick red frame on a white page."""
    img = np.zeros((120, 160, 4), dtype=np.uint8)
    img[..., :3] = (255, 255, 255)
    img[..., 3] = 255
    img[10:110, 10:150, :3] = FRAME
    img[20:100, 20:140, :3] = BOARD
    img[40:60, 40:80, :3] = ART
    path = tmp_path / "frame.png"
    Image.fromarray(img, "RGBA").save(path)
    return path


@pytest.fixture
def window(qtbot, frame_image_path):
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(frame_image_path)
    return win


def make_cut(window):
    layer = window.add_layer(LayerType.CUT)
    window.inspector.apply_color(FRAME)
    window.flush_recompute()
    return layer


def test_new_cut_layer_defaults_to_outer(window):
    layer = make_cut(window)
    assert layer.outline_mode == OutlineMode.OUTER
    minx, miny, maxx, maxy = window.outlines[layer.id].bounds
    assert minx < 12 and maxx > 147


def test_inspector_switches_to_inner_perimeter(window):
    layer = make_cut(window)
    outer_area = window.outlines[layer.id].area
    window.inspector.outline_combo.setCurrentIndex(1)
    window.flush_recompute()
    assert layer.outline_mode == OutlineMode.INNER
    inner = window.outlines[layer.id]
    assert inner.area < outer_area
    minx, miny, maxx, maxy = inner.bounds
    assert minx >= 18 and maxx <= 142 and miny >= 18 and maxy <= 102


def test_switching_back_to_outer_restores_outline(window):
    layer = make_cut(window)
    outer = window.outlines[layer.id]
    window.inspector.outline_combo.setCurrentIndex(1)
    window.flush_recompute()
    window.inspector.outline_combo.setCurrentIndex(0)
    window.flush_recompute()
    assert window.outlines[layer.id].equals(outer)


def test_outline_combo_visible_only_for_cut_layers(window):
    window.add_layer(LayerType.COPPER)
    assert not window.inspector.outline_combo.isVisibleTo(window.inspector)
    window.add_layer(LayerType.CUT)
    assert window.inspector.outline_combo.isVisibleTo(window.inspector)


def test_inspector_reflects_saved_mode_on_selection(window):
    layer = make_cut(window)
    layer.outline_mode = OutlineMode.INNER
    window.add_layer(LayerType.COPPER)
    window.layer_pane.select_layer(layer.id)
    assert window.inspector.outline_combo.currentData() == OutlineMode.INNER


def test_inner_mode_on_open_color_reports_error_not_crash(window):
    layer = window.add_layer(LayerType.CUT)
    window.inspector.apply_color(ART)  # solid block: encloses nothing
    window.inspector.outline_combo.setCurrentIndex(1)
    window.flush_recompute()
    assert layer.id not in window.outlines
    assert "inner" in window.statusBar().currentMessage().lower()


def test_exported_outline_uses_selected_perimeter(window, tmp_path):
    layer = make_cut(window)
    outer_files = window.export_gerbers(tmp_path / "outer", "b", 160.0, 120.0)
    window.inspector.outline_combo.setCurrentIndex(1)
    window.flush_recompute()
    inner_files = window.export_gerbers(tmp_path / "inner", "b", 160.0, 120.0)

    def xs(path):
        import re

        return [int(m) for m in re.findall(r"X(\d+)Y\d+D0", path.read_text())]

    outer_x = xs(next(p for p in outer_files if p.suffix == ".gm1"))
    inner_x = xs(next(p for p in inner_files if p.suffix == ".gm1"))
    assert min(inner_x) > min(outer_x)
    assert max(inner_x) < max(outer_x)
    assert layer.outline_mode == OutlineMode.INNER


def test_combo_options_are_labeled(window):
    texts = [window.inspector.outline_combo.itemText(i) for i in range(2)]
    assert "Outer" in texts[0] and "Inner" in texts[1]
