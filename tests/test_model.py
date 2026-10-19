"""Tests for the project/layer model."""

import pytest

from png_to_gerber.model import (
    Layer,
    LayerType,
    Project,
    ProjectError,
    Side,
    default_layer_name,
    duplicate_layer_name,
)


def make_layer(layer_type=LayerType.COPPER, **kwargs) -> Layer:
    return Layer(name="test", layer_type=layer_type, **kwargs)


def test_layer_ids_are_unique():
    a, b = make_layer(), make_layer()
    assert a.id != b.id


def test_add_and_get_layer():
    project = Project()
    layer = project.add_layer(make_layer())
    assert project.get_layer(layer.id) is layer
    assert project.get_layer(99999) is None


def test_any_number_of_cut_layers_are_allowed():
    project = Project()
    first = project.add_layer(make_layer(LayerType.CUT))
    second = project.add_layer(make_layer(LayerType.CUT))
    assert [layer.id for layer in project.cut_layers()] == [first.id, second.id]


def test_cut_allowed_again_after_removal():
    project = Project()
    cut = project.add_layer(make_layer(LayerType.CUT))
    project.remove_layer(cut.id)
    assert project.cut_layer() is None
    project.add_layer(make_layer(LayerType.CUT))  # no error


def test_remove_layer():
    project = Project()
    a = project.add_layer(make_layer())
    b = project.add_layer(make_layer())
    project.remove_layer(a.id)
    assert [l.id for l in project.layers] == [b.id]


def test_move_layer_up_down_and_clamping():
    project = Project()
    a = project.add_layer(make_layer())
    b = project.add_layer(make_layer())
    c = project.add_layer(make_layer())
    project.move_layer(c.id, -1)
    assert [l.id for l in project.layers] == [a.id, c.id, b.id]
    project.move_layer(a.id, -1)  # already at top: no change
    assert project.layers[0].id == a.id
    project.move_layer(b.id, 5)  # clamped to bottom
    assert project.layers[-1].id == b.id


def test_artwork_layers_excludes_cut():
    project = Project()
    copper = project.add_layer(make_layer())
    project.add_layer(make_layer(LayerType.CUT))
    assert [l.id for l in project.artwork_layers()] == [copper.id]


def test_grouped_for_export_merges_same_type_and_side():
    project = Project()
    a = project.add_layer(make_layer(side=Side.FRONT, has_color=True))
    b = project.add_layer(make_layer(side=Side.FRONT, has_color=True))
    c = project.add_layer(make_layer(side=Side.BACK, has_color=True))
    hidden = project.add_layer(make_layer(side=Side.BACK, has_color=True))
    hidden.visible = False
    uncolored = project.add_layer(make_layer(side=Side.BACK))

    groups = project.grouped_for_export()
    assert {l.id for l in groups[(LayerType.COPPER, Side.FRONT)]} == {a.id, b.id}
    assert {l.id for l in groups[(LayerType.COPPER, Side.BACK)]} == {c.id}
    assert uncolored.id not in {
        l.id for layers in groups.values() for l in layers
    }


def test_duplicate_layer_name_avoids_collisions():
    layers = [Layer(name="Copper 1", layer_type=LayerType.COPPER)]
    assert duplicate_layer_name("Copper 1", layers) == "Copper 1 copy"
    layers.append(Layer(name="Copper 1 copy", layer_type=LayerType.COPPER))
    assert duplicate_layer_name("Copper 1", layers) == "Copper 1 copy 2"


def test_add_layer_can_insert_directly_after_another():
    project = Project()
    first = project.add_layer(make_layer())
    third = project.add_layer(make_layer())
    second = project.add_layer(make_layer(), after_id=first.id)
    assert [layer.id for layer in project.layers] == [first.id, second.id, third.id]


def test_new_layers_are_not_inverted_and_checker_starts_at_half():
    assert make_layer().inverted is False
    assert Project().checker_opacity == 50


def test_default_layer_names_count_per_type():
    layers = []
    name1 = default_layer_name(LayerType.COPPER, layers)
    assert name1 == "Copper 1"
    layers.append(Layer(name=name1, layer_type=LayerType.COPPER))
    assert default_layer_name(LayerType.COPPER, layers) == "Copper 2"
    assert default_layer_name(LayerType.CUT, layers) == "Board outline"
