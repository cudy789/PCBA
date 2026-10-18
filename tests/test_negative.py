"""Negative layers: mask math, model dependencies, and GUI behavior."""

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from png_to_gerber.gui.main_window import MainWindow
from png_to_gerber.masking import negative_mask
from png_to_gerber.model import Layer, LayerType, Project, ProjectError, Side

from conftest import COPPER_COLOR, MASK_COLOR, SILK_COLOR


def rgba_canvas(h=10, w=10, opaque=True) -> np.ndarray:
    img = np.zeros((h, w, 4), dtype=np.uint8)
    if opaque:
        img[..., 3] = 255
    return img


# ---- negative_mask -------------------------------------------------------------------


def test_negative_is_complement_of_single_source():
    img = rgba_canvas()
    src = np.zeros((10, 10), dtype=np.uint8)
    src[2:6, 2:6] = 255
    neg = negative_mask(img, [src])
    assert neg[3, 3] == 0
    assert neg[0, 0] == 255
    assert np.array_equal(neg > 0, ~(src > 0))


def test_negative_of_multiple_sources_is_complement_of_union():
    img = rgba_canvas()
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((10, 10), dtype=np.uint8)
    a[0:4, 0:4] = 255
    b[6:10, 6:10] = 255
    neg = negative_mask(img, [a, b])
    assert neg[1, 1] == 0 and neg[8, 8] == 0
    assert neg[5, 5] == 255
    assert (neg > 0).sum() == 100 - 16 - 16


def test_negative_excludes_transparent_pixels():
    img = rgba_canvas()
    img[:, :3, 3] = 0  # transparent strip
    neg = negative_mask(img, [np.zeros((10, 10), dtype=np.uint8)])
    assert neg[5, 0] == 0
    assert neg[5, 5] == 255


def test_negative_with_no_sources_is_all_opaque_area():
    img = rgba_canvas()
    assert (negative_mask(img, []) == 255).all()


def test_negative_min_island_removes_specks():
    img = rgba_canvas(20, 20)
    src = np.full((20, 20), 255, dtype=np.uint8)
    src[5, 5] = 0  # one-pixel hole in the source => 1px negative speck
    src[10:15, 10:15] = 0  # 25px hole
    kept = negative_mask(img, [src], min_island=0)
    assert kept[5, 5] == 255
    cleaned = negative_mask(img, [src], min_island=4)
    assert cleaned[5, 5] == 0
    assert cleaned[12, 12] == 255


def test_negative_result_is_uint8_binary():
    neg = negative_mask(rgba_canvas(), [np.zeros((10, 10), dtype=np.uint8)])
    assert neg.dtype == np.uint8
    assert set(np.unique(neg)) <= {0, 255}


# ---- model ---------------------------------------------------------------------------


def make(layer_type=LayerType.COPPER, **kw):
    return Layer(name="l", layer_type=layer_type, has_color=True, **kw)


def test_add_negative_requires_existing_sources():
    project = Project()
    with pytest.raises(ProjectError):
        project.add_layer(make(negative_of=[999999]))


def test_cut_layer_cannot_be_negative():
    project = Project()
    src = project.add_layer(make())
    with pytest.raises(ProjectError):
        project.add_layer(make(LayerType.CUT, negative_of=[src.id]))


def test_is_derived_flag():
    project = Project()
    src = project.add_layer(make())
    neg = project.add_layer(make(negative_of=[src.id]))
    assert neg.is_derived and not src.is_derived


def test_dependents_are_transitive():
    project = Project()
    a = project.add_layer(make())
    b = project.add_layer(make(negative_of=[a.id]))
    c = project.add_layer(make(negative_of=[b.id]))
    assert project.dependents(a.id) == {b.id, c.id}
    assert project.dependents(c.id) == set()


def test_dependency_order_puts_sources_first_even_after_reorder():
    project = Project()
    a = project.add_layer(make())
    b = project.add_layer(make(negative_of=[a.id]))
    project.move_layer(b.id, -1)  # negative now listed before its source
    order = project.dependency_order([a.id, b.id])
    assert order.index(a.id) < order.index(b.id)


def test_removing_source_drops_it_from_remaining_negative():
    project = Project()
    a = project.add_layer(make())
    b = project.add_layer(make())
    neg = project.add_layer(make(negative_of=[a.id, b.id]))
    removed = project.remove_layer(a.id)
    assert removed == [a.id]
    assert project.get_layer(neg.id).negative_of == [b.id]


def test_removing_only_source_cascades_to_negative():
    project = Project()
    a = project.add_layer(make())
    neg = project.add_layer(make(negative_of=[a.id]))
    neg2 = project.add_layer(make(negative_of=[neg.id]))
    removed = project.remove_layer(a.id)
    assert set(removed) == {a.id, neg.id, neg2.id}
    assert project.layers == []


# ---- GUI ---------------------------------------------------------------------------


@pytest.fixture
def window(qtbot, fixture_image_path):
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(fixture_image_path)
    return win


def colored(window, layer_type, color):
    layer = window.add_layer(layer_type)
    window.inspector.apply_color(color)
    window.flush_recompute()
    return layer


def test_generate_negative_from_two_layers(window):
    copper = colored(window, LayerType.COPPER, COPPER_COLOR)
    silk = colored(window, LayerType.SILKSCREEN, SILK_COLOR)
    neg = window.generate_negative([copper.id, silk.id], LayerType.MASK)
    window.flush_recompute()

    assert neg is not None and neg.layer_type == LayerType.MASK
    assert neg.negative_of == [copper.id, silk.id]
    covered = (window.masks[copper.id] > 0) | (window.masks[silk.id] > 0)
    expected = (~covered) & (window.image[..., 3] > 0)
    # min_island removes tiny specks only; compare away from them.
    assert (window.masks[neg.id] > 0)[expected].mean() > 0.99
    assert not (window.masks[neg.id] > 0)[covered].any()
    assert neg.id in window.canvas.overlay_items


def test_negative_area_is_opposite_of_sources_pixelwise(window):
    copper = colored(window, LayerType.COPPER, COPPER_COLOR)
    neg = window.generate_negative([copper.id], LayerType.COPPER)
    window.flush_recompute()
    m = window.masks[neg.id]
    assert m[40, 40] == 0  # inside the copper block
    assert m[120, 20] == 255  # elsewhere on the board
    assert m[0, 0] == 0  # transparent background excluded


def test_negative_updates_live_when_source_changes(window):
    mask_layer = colored(window, LayerType.MASK, MASK_COLOR)
    neg = window.generate_negative([mask_layer.id], LayerType.COPPER)
    window.layer_pane.select_layer(mask_layer.id)
    # Strict source mask (no tolerance, no gap closing) misses the off-color
    # "anti-aliased" pixels, so the negative contains them.
    window.inspector.close_spin.setValue(0)
    window.inspector.tolerance_slider.setValue(0)
    window.flush_recompute()
    before = int((window.masks[neg.id] > 0).sum())
    # Relaxing the tolerance grows the source mask; the negative must follow.
    window.inspector.tolerance_slider.setValue(35)
    window.flush_recompute()
    after = int((window.masks[neg.id] > 0).sum())
    assert after < before


def test_generate_negative_via_multi_select_button(window, qtbot):
    a = colored(window, LayerType.COPPER, COPPER_COLOR)
    b = colored(window, LayerType.SILKSCREEN, SILK_COLOR)
    pane = window.layer_pane
    pane.list.clearSelection()
    pane.layer_item(a.id).setSelected(True)
    pane.layer_item(b.id).setSelected(True)
    assert pane.selected_ids() == [a.id, b.id]
    assert pane.negative_button.isEnabled()
    # Trigger the "As copper" menu action.
    action = pane.negative_button.menu().actions()[0]
    action.trigger()
    assert len(window.project.layers) == 3
    neg = window.project.layers[-1]
    assert neg.negative_of == [a.id, b.id]
    assert neg.layer_type == LayerType.COPPER


def test_negative_button_disabled_without_selection(window):
    window.layer_pane.list.clearSelection()
    assert not window.layer_pane.negative_button.isEnabled()


def test_negative_inherits_side_of_first_source(window):
    src = window.add_layer(LayerType.COPPER)
    src.side = Side.BACK
    window.inspector.apply_color(COPPER_COLOR)
    neg = window.generate_negative([src.id], LayerType.MASK)
    assert neg.side == Side.BACK


def test_negative_requires_colored_sources(window, monkeypatch):
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a))
    src = window.add_layer(LayerType.COPPER)  # no color yet
    assert window.generate_negative([src.id], LayerType.COPPER) is None
    assert warnings and len(window.project.layers) == 1


def test_negative_cannot_be_cut_type(window, monkeypatch):
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a))
    src = colored(window, LayerType.COPPER, COPPER_COLOR)
    assert window.generate_negative([src.id], LayerType.CUT) is None
    assert warnings


def test_negative_with_no_selection_rejected(window, monkeypatch):
    infos = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: infos.append(a))
    assert window.generate_negative([], LayerType.COPPER) is None
    assert infos


def test_negative_inspector_hides_color_tools(window):
    src = colored(window, LayerType.COPPER, COPPER_COLOR)
    neg = window.generate_negative([src.id], LayerType.MASK)
    insp = window.inspector
    assert insp._layer is neg
    assert not insp.hex_edit.isVisibleTo(insp)
    assert not insp.tolerance_slider.isVisibleTo(insp)
    assert insp.sources_label.isVisibleTo(insp)
    assert src.name in insp.sources_label.text()
    assert insp.island_spin.isVisibleTo(insp)
    # Color edits must be ignored on negative layers.
    insp.apply_color((1, 2, 3))
    assert neg.color != (1, 2, 3)


def test_negative_of_negative(window):
    src = colored(window, LayerType.COPPER, COPPER_COLOR)
    neg = window.generate_negative([src.id], LayerType.COPPER)
    neg2 = window.generate_negative([neg.id], LayerType.MASK)
    window.flush_recompute()
    # Double negative ~ original region (modulo small-speck cleanup).
    original = window.masks[src.id] > 0
    again = window.masks[neg2.id] > 0
    assert (original & again).sum() / original.sum() > 0.99


def test_deleting_source_updates_or_removes_negatives(window):
    a = colored(window, LayerType.COPPER, COPPER_COLOR)
    b = colored(window, LayerType.SILKSCREEN, SILK_COLOR)
    both = window.generate_negative([a.id, b.id], LayerType.MASK)
    only_a = window.generate_negative([a.id], LayerType.COPPER)
    window.flush_recompute()
    window.remove_layer(a.id)
    window.flush_recompute()
    remaining = {l.id for l in window.project.layers}
    assert only_a.id not in remaining  # lost its only source
    assert only_a.id not in window.canvas.overlay_items
    assert both.id in remaining and both.negative_of == [b.id]
    # The surviving negative now equals the negative of b alone.
    expected = (~(window.masks[b.id] > 0)) & (window.image[..., 3] > 0)
    assert (window.masks[both.id] > 0)[expected].mean() > 0.99


def test_negative_exports_like_any_layer(window, tmp_path):
    copper = colored(window, LayerType.COPPER, COPPER_COLOR)
    window.generate_negative([copper.id], LayerType.MASK)
    written = window.export_gerbers(tmp_path, "neg", 60.0, 45.0)
    names = {p.name for p in written}
    assert "neg-F_Mask.gts" in names and "neg-F_Cu.gtl" in names
    mask_text = (tmp_path / "neg-F_Mask.gts").read_text()
    assert mask_text.count("G36*") >= 2
    assert "%LPC*%" in mask_text  # the copper area is a hole in the negative


def test_hidden_negative_not_exported(window, tmp_path):
    copper = colored(window, LayerType.COPPER, COPPER_COLOR)
    neg = window.generate_negative([copper.id], LayerType.MASK)
    window.set_layer_visible(neg.id, False)
    written = window.export_gerbers(tmp_path, "neg", 60.0, 45.0)
    assert {p.name for p in written} == {"neg-F_Cu.gtl"}


def test_negative_follows_reordered_stack(window):
    src = colored(window, LayerType.COPPER, COPPER_COLOR)
    neg = window.generate_negative([src.id], LayerType.MASK)
    window.move_layer(neg.id, -1)  # negative listed above its source
    window.inspector.set_layer(src)
    window.layer_pane.select_layer(src.id)
    window.inspector.apply_color(SILK_COLOR)  # change source color
    window.flush_recompute()
    silk_area = window.masks[src.id] > 0
    assert not (window.masks[neg.id] > 0)[silk_area].any()


def test_layer_pane_tooltip_marks_negative(window):
    src = colored(window, LayerType.COPPER, COPPER_COLOR)
    window.generate_negative([src.id], LayerType.MASK)
    neg = window.project.layers[-1]
    assert "negative" in window.layer_pane.layer_item(neg.id).toolTip()
    assert "negative" not in window.layer_pane.layer_item(src.id).toolTip()


def test_layer_rows_use_eyes_not_checkboxes(window):
    layer = colored(window, LayerType.COPPER, COPPER_COLOR)
    item = window.layer_pane.layer_item(layer.id)
    assert not item.flags() & Qt.ItemIsUserCheckable
    assert not item.icon().isNull()
