"""Several base images: nesting, a shared top-left canvas, and unioned Gerbers."""

import json

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import Qt

from gerber_reader import mismatch, rasterize, read_regions
from png_to_gerber.gui.layer_pane import BASE_ITEM_ID, LAYER_INDENT
from png_to_gerber.gui.main_window import MainWindow
from png_to_gerber.image_io import canvas_size, pad_top_left
from png_to_gerber.model import Layer, LayerType, Project, Side
from png_to_gerber.project_io import load_project, save_project

from conftest import COPPER_COLOR


def _save(path, rgba):
    Image.fromarray(rgba, "RGBA").save(path)


def _solid(h, w, rgb, alpha=255):
    image = np.zeros((h, w, 4), dtype=np.uint8)
    image[..., :3] = rgb
    image[..., 3] = alpha
    return image


@pytest.fixture
def window(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    return win


def _color_layer(window, layer, color):
    window.layer_pane.select_layer(layer.id)
    window.inspector.apply_color(color)
    layer.close_radius = 0
    layer.min_island = 0
    layer.tolerance = 0
    window.flush_recompute()
    return layer


# ---- placement ------------------------------------------------------------------------


def test_pad_top_left_keeps_pixels_and_leaves_the_rest_empty():
    small = np.full((2, 3), 9, dtype=np.uint8)
    padded = pad_top_left(small, 4, 5)
    assert padded.shape == (4, 5)
    assert np.array_equal(padded[:2, :3], small)
    assert padded[2:, :].sum() == 0 and padded[:, 3:].sum() == 0
    assert canvas_size([small, np.zeros((4, 1), np.uint8)]) == (4, 3)


def test_layers_are_nested_under_the_image_that_created_them():
    project = Project()
    first = project.add_image("one.png", "one.png")
    second = project.add_image("two.png", "two.png")
    copper = project.add_layer(Layer(name="Copper 1", layer_type=LayerType.COPPER), image_id=first.id)
    silk = project.add_layer(
        Layer(name="Silkscreen 1", layer_type=LayerType.SILKSCREEN), image_id=second.id
    )
    project.move_layer(silk.id, -5)  # already at the top of its image
    assert first.layers == [copper]
    assert [layer.id for layer in second.layers] == [silk.id]
    assert project.image_of(copper.id) is first
    assert [layer.name for layer in project.layers] == ["Copper 1", "Silkscreen 1"]


def test_a_negative_may_cite_layers_from_other_images():
    project = Project()
    first = project.add_image("one.png")
    second = project.add_image("two.png")
    a = project.add_layer(Layer(name="A", layer_type=LayerType.COPPER, has_color=True), image_id=first.id)
    b = project.add_layer(Layer(name="B", layer_type=LayerType.COPPER, has_color=True), image_id=second.id)
    negative = project.add_layer(
        Layer(name="Neg", layer_type=LayerType.MASK, has_color=True, negative_of=[a.id, b.id]),
        image_id=first.id,
    )
    assert negative.negative_of == [a.id, b.id]
    assert project.image_of(negative.id) is first


def test_version_two_project_roundtrips_every_image(tmp_path):
    project = Project()
    first = project.add_image("left.png", str(tmp_path / "left.png"), background_removed=True, background_tolerance=9)
    first.flatten_enabled = True
    first.flatten_count = 3
    first.flatten_cutoffs = [40, 90]
    first.flatten_colors = [(8, 8, 8), None, (1, 2, 3)]
    first.flatten_smooth = 2
    second = project.add_image("right.png", str(tmp_path / "right.png"), visible=False)
    copper = project.add_layer(
        Layer(name="Copper 1", layer_type=LayerType.COPPER, has_color=True, color=(1, 2, 3)),
        image_id=first.id,
    )
    project.add_layer(
        Layer(
            name="Neg",
            layer_type=LayerType.SILKSCREEN,
            side=Side.BACK,
            has_color=True,
            negative_of=[copper.id],
        ),
        image_id=second.id,
    )
    loaded = load_project(save_project(tmp_path / "multi.prj", project))
    data = json.loads((tmp_path / "multi.prj").read_text())
    assert data["version"] == 2 and "layers" not in data
    assert [image.name for image in loaded.images] == ["left.png", "right.png"]
    assert loaded.images[0].background_removed and loaded.images[0].background_tolerance == 9
    assert loaded.images[0].flatten_enabled and loaded.images[0].flatten_count == 3
    assert loaded.images[0].flatten_cutoffs == [40, 90]
    assert loaded.images[0].flatten_colors == [(8, 8, 8), None, (1, 2, 3)]
    assert loaded.images[0].flatten_smooth == 2
    assert loaded.images[1].flatten_enabled is False
    assert loaded.images[1].flatten_cutoffs == [64, 128, 192]
    assert loaded.images[1].visible is False
    assert loaded.images[1].layers[0].negative_of == [loaded.images[0].layers[0].id]
    assert loaded.images[1].layers[0].side == Side.BACK


# ---- window ---------------------------------------------------------------------------


def test_the_layer_list_nests_layers_under_each_image(window, tmp_path):
    wide = _solid(20, 50, (10, 10, 10))
    tall = _solid(40, 20, (20, 20, 20))
    wide_path, tall_path = tmp_path / "wide.png", tmp_path / "tall.png"
    _save(wide_path, wide)
    _save(tall_path, tall)

    window.load_image(wide_path)
    window.add_image(tall_path)
    window.layer_pane.select_base(window.project.images[0].id)
    first = window.add_layer(LayerType.COPPER)
    window.layer_pane.select_base(window.project.images[1].id)
    second = window.add_layer(LayerType.SILKSCREEN)

    pane = window.layer_pane
    assert pane.list.count() == 4
    assert pane.list.item(0).data(Qt.UserRole) == BASE_ITEM_ID
    assert pane.list.item(0).text() == "wide.png"
    assert pane.list.item(1).data(Qt.UserRole) == first.id
    assert pane.list.item(1).text() == LAYER_INDENT + first.name
    assert pane.list.item(2).text() == "tall.png"
    assert pane.list.item(3).data(Qt.UserRole) == second.id
    assert window.project.image_of(first.id) is window.project.images[0]
    assert window.project.image_of(second.id) is window.project.images[1]

    window.move_layer(second.id, -1)
    assert [layer.id for layer in window.project.images[1].layers] == [second.id]
    assert window.project.image_of(first.id) is window.project.images[0]


def test_a_smaller_images_mask_stays_in_the_top_left(window, tmp_path):
    _save(tmp_path / "wide.png", _solid(20, 50, (255, 255, 255)))
    tall = _solid(40, 20, (255, 255, 255))
    tall[5:15, 5:15, :3] = COPPER_COLOR
    _save(tmp_path / "tall.png", tall)

    window.load_image(tmp_path / "wide.png")
    window.add_image(tmp_path / "tall.png")
    layer = _color_layer(window, window.add_layer(LayerType.COPPER), COPPER_COLOR)

    assert window._canvas_shape() == (40, 50)
    mask = window.masks[layer.id]
    assert mask.shape == (40, 50)
    assert mask[5:15, 5:15].min() == 255
    assert mask[:, 20:].sum() == 0  # past the narrow image
    assert mask[20:, :].sum() == 0  # the color only occupies rows 5..14
    assert set(window.canvas.base_items) == {image.id for image in window.project.images}
    assert window.canvas._image_size == (50, 40)


def test_exported_copper_is_the_union_of_every_image(window, tmp_path):
    left = _solid(40, 80, (255, 255, 255))
    left[10:30, 5:25, :3] = COPPER_COLOR
    right = _solid(40, 80, (255, 255, 255))
    right[10:30, 50:70, :3] = (180, 40, 40)
    _save(tmp_path / "left.png", left)
    _save(tmp_path / "right.png", right)

    window.load_image(tmp_path / "left.png")
    window.layer_pane.select_base(window.project.images[0].id)
    first = _color_layer(window, window.add_layer(LayerType.COPPER), COPPER_COLOR)
    window.add_image(tmp_path / "right.png")
    second = _color_layer(window, window.add_layer(LayerType.COPPER), (180, 40, 40))

    written = window.export_gerbers(tmp_path, "union", 80.0, 40.0)
    copper = next(path for path in written if path.suffix == ".gtl")
    painted = rasterize(read_regions(copper, 80, 40, 80.0, 40.0), 80, 40)
    combined = (window.masks[first.id] > 0) | (window.masks[second.id] > 0)
    assert mismatch(combined.astype(np.uint8) * 255, painted) == 0
    # Each image contributed pixels the other image's layer does not cover.
    assert painted[15, 10] and painted[15, 60]
    assert not painted[15, 40]


def test_cut_outlines_are_unioned_into_one_edge_cuts_file(window, tmp_path):
    left = _solid(30, 90, (255, 255, 255))
    left[:, :30, :3] = (240, 240, 235)
    right = _solid(30, 90, (255, 255, 255))
    right[:, 60:, :3] = (20, 20, 80)
    _save(tmp_path / "left.png", left)
    _save(tmp_path / "right.png", right)

    window.load_image(tmp_path / "left.png")
    window.layer_pane.select_base(window.project.images[0].id)
    first = _color_layer(window, window.add_layer(LayerType.CUT), (240, 240, 235))
    first.smoothness = 0
    window.add_image(tmp_path / "right.png")
    second = _color_layer(window, window.add_layer(LayerType.CUT), (20, 20, 80))
    second.smoothness = 0
    window.flush_recompute()

    written = window.export_gerbers(tmp_path, "cuts", 90.0, 30.0)
    outline = next(path for path in written if path.suffix == ".gm1").read_text()
    assert outline.count("D02*") == 2  # one closed stroke per cut layer
    xs = [int(line.split("Y")[0][1:]) for line in outline.splitlines() if line.startswith("X")]
    assert min(xs) < 1_000_000  # left outline, near x = 0 mm
    assert max(xs) > 80_000_000  # right outline, past 80 mm on a 90 mm board


def test_a_negative_uses_opaque_pixels_from_every_source_image(window, tmp_path):
    full = _solid(20, 20, (255, 0, 0))
    partial = _solid(20, 20, (0, 0, 0), alpha=0)
    partial[:, 10:, 3] = 255
    partial[:, 10:, :3] = (0, 0, 255)
    partial[:5, 10:, :3] = (0, 255, 0)  # the only colored copper
    _save(tmp_path / "full.png", full)
    _save(tmp_path / "partial.png", partial)

    window.load_image(tmp_path / "full.png")
    window.add_image(tmp_path / "partial.png")
    source = _color_layer(window, window.add_layer(LayerType.COPPER), (0, 255, 0))
    window.layer_pane.select_base(window.project.images[0].id)
    negative = window.generate_negative([source.id], LayerType.MASK)
    window.flush_recompute()

    assert window.project.image_of(negative.id) is window.project.images[0]
    mask = window.masks[negative.id]
    assert mask[:5, 10:].sum() == 0  # covered by the source layer
    assert mask[10, 15] == 255  # opaque on the source image, not covered
    assert mask[:, :10].sum() == 0  # transparent on the source image, so not in the negative


def test_removing_an_image_removes_its_layers(window, tmp_path):
    _save(tmp_path / "a.png", _solid(8, 8, (1, 1, 1)))
    _save(tmp_path / "b.png", _solid(8, 8, (2, 2, 2)))
    window.load_image(tmp_path / "a.png")
    window.add_layer(LayerType.COPPER)
    kept = window.project.images[0]
    window.add_image(tmp_path / "b.png")
    dropped = window.add_layer(LayerType.SILKSCREEN)
    window.remove_image(window.project.images[1].id)

    assert window.project.images == [kept]
    assert window.project.get_layer(dropped.id) is None
    assert len(window.project.layers) == 1
    assert window.layer_pane.list.item(0).text() == "Base image"
    assert window.layer_pane.list.count() == 2
