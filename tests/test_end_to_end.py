"""End-to-end: fixture image -> all front/back layer types -> Gerber files."""

import pytest

from png_to_gerber.gui.main_window import MainWindow
from png_to_gerber.model import LayerType, Side

from conftest import BOARD_COLOR, COPPER_COLOR, MASK_COLOR, SILK_COLOR


@pytest.fixture
def full_window(qtbot, fixture_image_path):
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(fixture_image_path)

    specs = [
        (LayerType.COPPER, Side.FRONT, COPPER_COLOR),
        (LayerType.COPPER, Side.BACK, COPPER_COLOR),
        (LayerType.MASK, Side.FRONT, MASK_COLOR),
        (LayerType.MASK, Side.BACK, MASK_COLOR),
        (LayerType.SILKSCREEN, Side.FRONT, SILK_COLOR),
        (LayerType.SILKSCREEN, Side.BACK, SILK_COLOR),
    ]
    for layer_type, side, color in specs:
        layer = win.add_layer(layer_type)
        layer.side = side
        win.inspector.set_layer(layer)  # keep inspector in sync with side
        win.inspector.apply_color(color)
    cut = win.add_layer(LayerType.CUT)
    win.inspector.apply_color(BOARD_COLOR)
    cut.smoothness = 40
    win.flush_recompute()
    return win


def test_exports_every_supported_file(full_window, tmp_path):
    written = full_window.export_gerbers(tmp_path, "full", 100.0, 75.0)
    names = sorted(p.name for p in written)
    assert names == [
        "full-B_Cu.gbl",
        "full-B_Mask.gbs",
        "full-B_Silkscreen.gbo",
        "full-Edge_Cuts.gm1",
        "full-F_Cu.gtl",
        "full-F_Mask.gts",
        "full-F_Silkscreen.gto",
    ]


def test_exported_files_are_valid_gerber(full_window, tmp_path):
    written = full_window.export_gerbers(tmp_path, "full", 100.0, 75.0)
    for path in written:
        text = path.read_text(encoding="ascii")
        assert text.startswith("%TF.FileFunction,")
        assert "%FSLAX46Y46*%" in text
        assert "%MOMM*%" in text
        assert text.rstrip().endswith("M02*")


def test_outline_is_closed_in_gerber(full_window, tmp_path):
    written = full_window.export_gerbers(tmp_path, "full", 100.0, 75.0)
    outline_file = next(p for p in written if p.name.endswith(".gm1"))
    lines = outline_file.read_text().splitlines()
    start = next(l for l in lines if l.endswith("D02*")).replace("D02*", "")
    last_draw = [l for l in lines if l.endswith("D01*")][-1].replace("D01*", "")
    assert start == last_draw, "Edge.Cuts path must return to its start point"


def test_copper_layer_contains_cleared_hole(full_window, tmp_path):
    written = full_window.export_gerbers(tmp_path, "full", 100.0, 75.0)
    copper = next(p for p in written if p.name.endswith(".gtl"))
    text = copper.read_text()
    assert text.count("G36*") == 2  # copper block, then its hole
    assert "%LPC*%" in text
    assert text.index("%LPD*%") < text.index("%LPC*%")


def test_coordinates_fit_board_size(full_window, tmp_path):
    import re

    written = full_window.export_gerbers(tmp_path, "full", 100.0, 75.0)
    pattern = re.compile(r"X(-?\d+)Y(-?\d+)D0[12]\*")
    for path in written:
        for match in pattern.finditer(path.read_text()):
            x, y = int(match.group(1)), int(match.group(2))
            assert -1_000_000 <= x <= 101_000_000  # within ~1mm of bounds
            assert -1_000_000 <= y <= 76_000_000
