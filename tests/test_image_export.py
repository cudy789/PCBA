"""Exporting layer PNGs and an optional rendered picture of the board."""

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QMessageBox
from shapely.geometry import box

from png_to_gerber.gerber import ExportError
from png_to_gerber.gui.image_export_dialog import ImageExportDialog
from png_to_gerber.gui.main_window import MainWindow
from png_to_gerber.image_export import (
    COPPER_RGB,
    LAYER_TINTS,
    RENDER_NAME,
    SILK_RGB,
    SOLDERMASK_RGB,
    cut_perimeter_rgba,
    layer_png_name,
    layer_rgba,
    render_pcb,
    safe_filename,
)
from png_to_gerber.model import Layer, LayerType, Side

from conftest import BOARD_COLOR, COPPER_COLOR, SILK_COLOR


def _layer(layer_type, name="Layer"):
    return Layer(name=name, layer_type=layer_type, has_color=True)


def _open(path):
    return np.array(Image.open(path))


def test_layer_rgba_is_solid_only_where_the_mask_is_on():
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1, 2] = 255
    image = layer_rgba(mask, LayerType.COPPER)
    assert image.shape == (4, 5, 4)
    assert tuple(image[1, 2]) == (*LAYER_TINTS[LayerType.COPPER], 255)
    assert int(image[:, :, 3].sum()) == 255


def test_cut_perimeter_is_only_the_outline():
    image = cut_perimeter_rgba(8, 8, box(2, 2, 6, 6))
    assert image.shape == (8, 8, 4)
    assert image[3, 3, 3] == 0  # interior
    assert image[0, 0, 3] == 0  # outside
    assert np.all(image[2:6, 5, 3] == 255)  # right side of the shape
    assert np.all(image[5, 2:6, 3] == 255)  # bottom side of the shape
    assert np.all(image[2:6, 6, 3] == 0)  # not the pixel past the right edge
    assert np.all(image[6, 2:6, 3] == 0)
    opaque = image[:, :, 3] > 0
    assert np.all(image[opaque][:, :3] == LAYER_TINTS[LayerType.CUT])
    assert int(opaque.sum()) == 12


def test_cut_perimeter_keeps_the_bottom_and_right_image_edges():
    image = cut_perimeter_rgba(8, 8, box(0, 0, 8, 8))
    assert image[4, 7, 3] == 255
    assert image[7, 4, 3] == 255
    assert image[7, 7, 3] == 255
    assert image[3, 3, 3] == 0


def test_render_shows_copper_through_openings_and_silk_on_top():
    size = 8
    copper = np.zeros((size, size), dtype=np.uint8)
    copper[2:6, 2:6] = 255
    copper[3, 3] = 0  # hole
    opening = np.zeros((size, size), dtype=np.uint8)
    opening[2, 2] = 255  # copper visible here
    opening[3, 3] = 255  # opening over the hole stays mask-colored
    opening[4, 4] = 255  # covered by silkscreen
    silk = np.zeros((size, size), dtype=np.uint8)
    silk[4, 4] = 255
    silk[0, 0] = 255  # outside the board, so it must not appear
    parts = [
        (_layer(LayerType.COPPER), copper),
        (_layer(LayerType.MASK), opening),
        (_layer(LayerType.SILKSCREEN), silk),
    ]
    image = render_pcb(size, size, parts, cut_polygon=box(2, 2, 6, 6))

    assert image[0, 0, 3] == 0
    assert image[1, 1, 3] == 0
    assert tuple(image[2, 2]) == (*COPPER_RGB, 255)
    assert tuple(image[3, 3]) == (*SOLDERMASK_RGB, 255)
    assert tuple(image[4, 4]) == (*SILK_RGB, 255)
    assert tuple(image[5, 5]) == (*SOLDERMASK_RGB, 255)  # copper covered by mask


def test_render_without_a_mask_layer_shows_all_of_the_copper():
    copper = np.zeros((4, 4), dtype=np.uint8)
    copper[1:3, 1:3] = 255
    image = render_pcb(4, 4, [(_layer(LayerType.COPPER), copper)])
    assert tuple(image[1, 1, :3]) == COPPER_RGB
    assert tuple(image[0, 0, :3]) == SOLDERMASK_RGB
    assert image[0, 0, 3] == 255


def test_render_falls_back_to_the_cut_mask_when_there_is_no_outline():
    cut = np.zeros((4, 4), dtype=np.uint8)
    cut[1:3, 1:3] = 255
    image = render_pcb(4, 4, [(_layer(LayerType.CUT), cut)])
    assert image[0, 0, 3] == 0
    assert tuple(image[1, 1]) == (*SOLDERMASK_RGB, 255)


def test_layer_filenames_are_safe_and_do_not_collide():
    assert safe_filename("  ") == "layer"
    used = set()
    assert layer_png_name("card", "Copper 1", used) == "card-Copper_1.png"
    assert layer_png_name("card", "PCB", {f"card-{RENDER_NAME}".lower()}) == "card-PCB_2.png"


def test_dialog_lists_layers_and_requires_a_choice(qtbot, tmp_path):
    front = Layer(name="Pads", layer_type=LayerType.COPPER, side=Side.FRONT, has_color=True)
    back = Layer(
        name="Ground",
        layer_type=LayerType.COPPER,
        side=Side.BACK,
        visible=False,
        has_color=True,
    )
    bare = Layer(name="Empty", layer_type=LayerType.MASK, has_color=False)
    dialog = ImageExportDialog(
        [(front, True), (back, True), (bare, False)], "card", "", parent=None
    )
    qtbot.addWidget(dialog)
    assert [dialog.list.item(row).text() for row in range(3)] == [
        "Pads — Copper, front",
        "Ground — Copper, back",
        "Empty — Solder mask opening, front",
    ]
    assert dialog.selected_ids() == [front.id]
    assert dialog.list.item(1).flags() & Qt.ItemIsEnabled
    assert not (dialog.list.item(2).flags() & Qt.ItemIsEnabled)
    assert not dialog.wants_render()

    assert dialog.validate() == "Choose an output folder."
    dialog.dir_edit.setText(str(tmp_path / "missing-file"))
    (tmp_path / "missing-file").write_text("x")
    dialog.dir_edit.setText(str(tmp_path / "missing-file"))
    assert "not a folder" in dialog.validate()

    dialog.dir_edit.setText(str(tmp_path))
    dialog.name_edit.setText("a/b")
    assert "not allowed" in dialog.validate()
    dialog.name_edit.setText("card")
    dialog.list.item(0).setCheckState(Qt.Unchecked)
    assert "at least one" in dialog.validate()

    dialog.render_check.setChecked(True)
    assert dialog.validate() is None
    assert dialog.wants_render()
    dialog._on_accept()
    assert dialog.result() == QDialog.Accepted
    out_dir, name, ids, render = dialog.values()
    assert (out_dir, name, ids, render) == (tmp_path, "card", [], True)


def test_dialog_refuses_a_render_when_no_layer_has_a_mask(qtbot, tmp_path):
    bare = Layer(name="Empty", layer_type=LayerType.COPPER, has_color=False)
    dialog = ImageExportDialog([(bare, False)], "card", str(tmp_path))
    qtbot.addWidget(dialog)
    dialog.render_check.setChecked(True)
    assert "color" in dialog.validate()


def test_export_image_without_an_image_explains(qtbot, monkeypatch):
    win = MainWindow()
    qtbot.addWidget(win)
    messages = []
    monkeypatch.setattr(
        QMessageBox, "information", lambda *args, **kwargs: messages.append(args[2])
    )
    win.export_image_action.trigger()
    assert messages == ["Open an image first."]


def test_export_image_action_opens_the_dialog(qtbot, fixture_image_path, monkeypatch):
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(fixture_image_path)
    copper = win.add_layer(LayerType.COPPER)
    win.inspector.apply_color(COPPER_COLOR)
    win.add_layer(LayerType.MASK)
    captured = {}

    def reject(dialog):
        captured["dialog"] = dialog
        return ImageExportDialog.Rejected

    monkeypatch.setattr(ImageExportDialog, "exec", reject)
    win.export_image_action.trigger()
    dialog = captured["dialog"]
    assert dialog.windowTitle() == "Export image"
    assert dialog.dir_edit.text() == str(fixture_image_path.parent)
    assert dialog.name_edit.text() == fixture_image_path.stem
    assert dialog.selected_ids() == [copper.id]
    assert not (dialog.list.item(1).flags() & Qt.ItemIsEnabled)


def test_accepting_the_dialog_writes_checked_layers_and_the_render(
    qtbot, fixture_image_path, tmp_path, monkeypatch
):
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(fixture_image_path)
    win.add_layer(LayerType.COPPER)
    win.inspector.apply_color(COPPER_COLOR)
    win.add_layer(LayerType.SILKSCREEN)
    win.inspector.apply_color(SILK_COLOR)

    def accept(dialog):
        dialog.list.item(1).setCheckState(Qt.Unchecked)
        dialog.render_check.setChecked(True)
        dialog.dir_edit.setText(str(tmp_path))
        dialog.name_edit.setText("art")
        return ImageExportDialog.Accepted

    messages = []
    monkeypatch.setattr(ImageExportDialog, "exec", accept)
    monkeypatch.setattr(
        QMessageBox, "information", lambda *args, **kwargs: messages.append(args[2])
    )
    win.export_image_action.trigger()

    copper_path = tmp_path / "art-Copper_1.png"
    assert copper_path.is_file()
    assert not (tmp_path / "art-Silkscreen_1.png").exists()
    rendered_path = tmp_path / "art-PCB.png"
    assert rendered_path.is_file()
    assert "art-Copper_1.png" in messages[0]
    assert "art-PCB.png" in messages[0]
    assert "Silkscreen" not in messages[0]

    layer = _open(copper_path)
    assert tuple(layer[40, 40]) == (*LAYER_TINTS[LayerType.COPPER], 255)
    assert layer[20, 20, 3] == 0
    assert layer[90, 120, 3] == 0

    rendered = _open(rendered_path)
    assert tuple(rendered[40, 40, :3]) == COPPER_RGB
    assert tuple(rendered[60, 60, :3]) == SOLDERMASK_RGB  # hole in the copper
    assert tuple(rendered[90, 120, :3]) == SILK_RGB  # unchecked silk is still rendered


def test_rendered_board_clips_to_the_cut_and_skips_unselected_files(
    qtbot, fixture_image_path, tmp_path
):
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(fixture_image_path)
    copper = win.add_layer(LayerType.COPPER)
    win.inspector.apply_color(COPPER_COLOR)
    win.set_layer_visible(copper.id, False)
    cut = win.add_layer(LayerType.CUT)
    win.inspector.apply_color(BOARD_COLOR)
    cut.smoothness = 0
    win.recompute_layer(cut)

    written = win.export_images(tmp_path, "card", [cut.id], render=True)
    assert {path.name for path in written} == {"card-Board_outline.png", "card-PCB.png"}

    outline = _open(tmp_path / "card-Board_outline.png")
    assert outline[20, 20, 3] == 0  # interior of the board is not filled
    assert outline[0, 0, 3] == 0
    opaque = outline[:, :, 3] > 0
    assert opaque.any()
    assert np.all(outline[opaque][:, :3] == LAYER_TINTS[LayerType.CUT])

    rendered = _open(tmp_path / "card-PCB.png")
    assert rendered[0, 0, 3] == 0
    assert tuple(rendered[20, 20, :3]) == SOLDERMASK_RGB
    assert tuple(rendered[40, 40, :3]) == COPPER_RGB  # hidden copper is still on the board


def test_a_layer_named_like_the_render_gets_its_own_file(qtbot, fixture_image_path, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(fixture_image_path)
    copper = win.add_layer(LayerType.COPPER)
    win.inspector.apply_color(COPPER_COLOR)
    copper.name = "PCB"
    written = win.export_images(tmp_path, "card", [copper.id], render=True)
    assert {path.name for path in written} == {"card-PCB_2.png", "card-PCB.png"}
    assert tuple(_open(tmp_path / "card-PCB.png")[40, 40, :3]) == COPPER_RGB
    assert tuple(_open(tmp_path / "card-PCB_2.png")[40, 40, :3]) == LAYER_TINTS[LayerType.COPPER]


def test_export_images_refuses_a_layer_without_a_mask(qtbot, fixture_image_path, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(fixture_image_path)
    bare = win.add_layer(LayerType.COPPER)
    out = tmp_path / "out"
    with pytest.raises(ExportError, match="no mask"):
        win.export_images(out, "card", [bare.id], render=False)
    assert not out.exists()

    with pytest.raises(ExportError, match="Nothing to render"):
        win.export_images(out, "card", [], render=True)
    assert not out.exists()
