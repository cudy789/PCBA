"""Raster paint operations: brush size, strokes, fill, and project persistence."""

import numpy as np
import pytest

from png_to_gerber.paint import (
    ERASE,
    PAINT,
    apply_edits,
    coalesce_paint,
    compose_painted,
    disk_mask,
    edits_from_runs,
    stroke_rectangle,
    flood_fill_rgba,
    flood_region,
    runs_from_edits,
    snap_line_end,
    snap_square_corner,
    stamp,
    stroke,
)
from png_to_gerber.project_io import ProjectFileError, load_project, project_from_dict, project_to_dict, save_project
from png_to_gerber.model import Layer, LayerType, Project

from test_project_io import sample_project


def test_disk_is_round_and_diameter_one_is_a_pixel():
    assert disk_mask(1).shape == (1, 1)
    assert disk_mask(1)[0, 0]
    disk = disk_mask(5)
    assert disk.shape == (5, 5)
    assert disk[2, 2]
    assert not disk[0, 0]  # corner of the bounding box is outside the circle


def test_stamp_and_stroke_use_the_diameter_and_keep_endpoints():
    canvas = np.zeros((12, 12), dtype=np.uint8)
    stamp(canvas, 2, 2, 1, 255)
    assert canvas[2, 2] == 255
    assert canvas.sum() == 255

    canvas[:] = 0
    stroke(canvas, 1, 5, 8, 5, 1, 255)
    assert list(canvas[5, 1:9]) == [255] * 8
    assert canvas[4, 4] == 0 and canvas[6, 4] == 0

    canvas[:] = 0
    stroke(canvas, 1, 5, 8, 5, 3, 255)
    assert canvas[5, 1] == 255 and canvas[5, 8] == 255
    assert canvas[4, 4] == 255 and canvas[6, 4] == 255


def test_rectangle_strokes_the_outline_and_brush_size_thickens_it():
    canvas = np.zeros((30, 30), dtype=np.uint8)
    stroke_rectangle(canvas, 20, 5, 5, 20, 1, 255)
    assert canvas[5, 12] == 255 and canvas[12, 5] == 255
    assert canvas[20, 12] == 255 and canvas[12, 20] == 255
    assert canvas[12, 12] == 0  # interior stays empty
    assert canvas[12, 3] == 0

    thick = np.zeros_like(canvas)
    stroke_rectangle(thick, 5, 5, 20, 20, 7, 255)
    assert thick[12, 12] == 0
    assert thick[5, 12] == 255
    assert thick[2, 12] == 255  # outside the edge, inside the thicker stroke
    assert thick[0, 12] == 0

    dot = np.zeros_like(canvas)
    stroke_rectangle(dot, 8, 8, 8, 8, 1, 255)
    assert dot[8, 8] == 255
    assert dot.sum() == 255


def test_line_snap_uses_axes_and_diagonals():
    assert snap_line_end(10, 20, 30, 24) == (30, 20)  # shallow, so horizontal
    assert snap_line_end(10, 20, 14, 40) == (10, 40)  # steep, so vertical
    assert snap_line_end(10, 40, 26, 30) == (23, 27)  # nearest diagonal
    assert snap_line_end(5, 5, 15, 15) == (15, 15)  # already 45 degrees
    assert snap_line_end(8, 8, 0, 9) == (0, 8)  # shallow the other way
    assert snap_line_end(4, 4, 4, 4) == (4, 4)


def test_square_snap_matches_the_longer_side_and_follows_the_cursor():
    assert snap_square_corner(20, 30, 40, 36) == (40, 50)
    assert snap_square_corner(20, 30, 28, 34) == (28, 38)  # a smaller square
    assert snap_square_corner(20, 30, 12, 40) == (10, 40)  # left and down
    assert snap_square_corner(20, 30, 20, 30) == (20, 30)


def test_erase_overwrites_paint_and_apply_edits_forces_the_mask():
    edits = np.zeros((6, 6), dtype=np.int8)
    stamp(edits, 2, 2, 3, PAINT)
    stamp(edits, 2, 2, 1, ERASE)
    assert edits[2, 2] == ERASE
    assert edits[2, 3] == PAINT

    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[0, :] = 255
    result = apply_edits(mask, edits)
    assert result[2, 2] == 0
    assert result[2, 3] == 255
    assert result[0, 0] == 255  # untouched mask pixel
    assert result[5, 5] == 0


def test_runs_roundtrip_and_later_runs_win():
    edits = np.zeros((4, 6), dtype=np.int8)
    edits[1, 1:4] = PAINT
    edits[1, 2] = ERASE
    edits[3, 0] = ERASE
    restored = edits_from_runs(runs_from_edits(edits), edits.shape)
    assert np.array_equal(restored, edits)

    overlapped = edits_from_runs([(0, 0, 4, PAINT), (1, 0, 2, ERASE)], (2, 4))
    assert list(overlapped[0]) == [PAINT, ERASE, ERASE, PAINT]


def test_flood_fill_is_4_connected_and_exact_on_rgba():
    on = np.zeros((4, 4), dtype=bool)
    on[0, 0] = on[0, 1] = on[1, 1] = True
    on[2, 2] = True  # touches (1, 1) only at a corner
    region = flood_region(on, 0, 0)
    assert region[1, 1] and not region[2, 2]
    assert flood_region(on, 2, 2)[2, 2]
    assert not flood_region(on, 2, 2)[0, 0]

    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[0:2, 0:2] = (10, 20, 30, 255)
    rgba[2, 2] = (10, 20, 30, 255)  # touches the block only at a corner
    filled = flood_fill_rgba(rgba, 0, 0, (1, 2, 3, 255))
    assert tuple(filled[1, 1]) == (1, 2, 3, 255)
    assert tuple(filled[2, 2]) == (10, 20, 30, 255)
    assert tuple(filled[3, 3]) == (0, 0, 0, 0)


def test_coalesce_paint_keeps_the_last_color():
    pixels = coalesce_paint([(1, 2, 9, 9, 9, 255), (1, 2, 4, 5, 6, 128), (0, 0, 1, 1, 1, 255)])
    assert pixels == [(1, 2, 4, 5, 6, 128), (0, 0, 1, 1, 1, 255)]


def test_compose_keeps_painted_color_through_flatten_and_crop():
    image = np.zeros((10, 10, 4), dtype=np.uint8)
    image[2:8, 2:8] = (200, 200, 200, 255)
    painted = compose_painted(
        image,
        [(4, 4, 1, 2, 3, 255)],
        removed=True,
        tolerance=0,
        flatten=True,
        cutoffs=[128],
        colors=[(9, 9, 9), (8, 8, 8)],
        smooth=0,
    )
    assert painted.shape[:2] == (6, 6)
    assert tuple(painted[2, 2]) == (1, 2, 3, 255)
    assert tuple(painted[0, 0]) == (8, 8, 8, 255)  # flattened, not the painted color


def test_project_roundtrip_keeps_manual_runs_and_base_paint(tmp_path):
    project = sample_project()
    project.layers[0].manual = [(4, 5, 3, PAINT), (4, 6, 1, ERASE)]
    project.images[0].paint = [(8, 9, 1, 2, 3, 255), (8, 10, 0, 0, 0, 0)]
    loaded = load_project(save_project(tmp_path / "painted.prj", project))
    assert loaded.layers[0].manual == [(4, 5, 3, PAINT), (4, 6, 1, ERASE)]
    assert loaded.images[0].paint == [(8, 9, 1, 2, 3, 255), (8, 10, 0, 0, 0, 0)]


def test_multi_image_project_keeps_each_images_paint(tmp_path):
    project = Project()
    project.add_image("one")
    second = project.add_image("two")
    second.paint = [(1, 2, 3, 4, 5, 6)]
    project.add_layer(Layer(name="Copper", layer_type=LayerType.COPPER), image_id=second.id)
    loaded = load_project(save_project(tmp_path / "two.prj", project))
    assert loaded.images[0].paint == []
    assert loaded.images[1].paint == [(1, 2, 3, 4, 5, 6)]
    assert loaded.images[1].layers[0].manual == []


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda data: data["layers"][0].__setitem__("manual", [[0, 0, 0, 1]]), "manual"),
        (lambda data: data["layers"][0].__setitem__("manual", [[0, 0, 2, 2]]), "manual"),
        (lambda data: data["layers"][0].__setitem__("manual", "nope"), "manual"),
        (lambda data: data.__setitem__("paint", [[1, 2, 3]]), "paint"),
        (lambda data: data.__setitem__("paint", [[1, 2, 300, 0, 0, 255]]), "paint"),
    ],
)
def test_malformed_paint_is_rejected(mutate, match):
    data = project_to_dict(sample_project())
    mutate(data)
    with pytest.raises(ProjectFileError, match=match):
        project_from_dict(data)
