""".prj project files: serialization, validation, and GUI save/open round-trips."""

import copy
import json
import shutil

import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QFileDialog, QMessageBox

from png_to_gerber import project_io
from png_to_gerber.gui.main_window import MainWindow
from png_to_gerber.model import (
    BoardSettings,
    Layer,
    LayerType,
    OutlineMode,
    Project,
    Side,
)
from png_to_gerber.project_io import ProjectFileError, load_project, save_project

from conftest import BOARD_COLOR, COPPER_COLOR, MASK_COLOR, SILK_COLOR


def sample_project(image_path=None) -> Project:
    project = Project(image_path=str(image_path) if image_path else None)
    copper = project.add_layer(
        Layer(
            name="My copper", layer_type=LayerType.COPPER, side=Side.BACK,
            color=(1, 2, 3), has_color=True, tolerance=44, close_radius=2,
            min_island=9, fill_holes=True, visible=False,
        )
    )
    project.add_layer(
        Layer(
            name="Outline", layer_type=LayerType.CUT, color=(240, 240, 235),
            has_color=True, smoothness=55, outline_mode=OutlineMode.INNER,
        )
    )
    project.add_layer(
        Layer(
            name="Negative", layer_type=LayerType.MASK, has_color=True,
            negative_of=[copper.id], min_island=7,
        )
    )
    project.board = BoardSettings(
        width_mm=80.5, height_mm=60.25, lock_aspect=False,
        output_dir="C:/out", base_name="art",
    )
    project.background_removed = True
    project.background_tolerance = 21
    project.base_visible = False
    return project


# ---- round trip ----------------------------------------------------------------------


def test_save_writes_json_with_format_and_version(tmp_path):
    path = save_project(tmp_path / "a.prj", sample_project())
    data = json.loads(path.read_text())
    assert data["format"] == "png_to_gerber_project"
    assert data["version"] == 1
    assert len(data["layers"]) == 3


def test_save_appends_prj_extension(tmp_path):
    path = save_project(tmp_path / "noext", Project())
    assert path.suffix == ".prj" and path.exists()


def test_save_creates_missing_directories_and_leaves_no_temp_file(tmp_path):
    path = save_project(tmp_path / "x" / "y" / "a.prj", Project())
    assert path.exists()
    assert not list(path.parent.glob("*.tmp"))


def test_roundtrip_preserves_all_layer_fields(tmp_path):
    original = sample_project()
    loaded = load_project(save_project(tmp_path / "a.prj", original))
    assert len(loaded.layers) == 3
    for a, b in zip(original.layers, loaded.layers):
        assert (a.name, a.layer_type, a.side, a.color, a.has_color) == (
            b.name, b.layer_type, b.side, b.color, b.has_color,
        )
        assert (a.tolerance, a.close_radius, a.min_island, a.fill_holes) == (
            b.tolerance, b.close_radius, b.min_island, b.fill_holes,
        )
        assert (a.smoothness, a.outline_mode, a.visible) == (
            b.smoothness, b.outline_mode, b.visible,
        )


def test_roundtrip_preserves_board_and_view_state(tmp_path):
    loaded = load_project(save_project(tmp_path / "a.prj", sample_project()))
    assert loaded.board == BoardSettings(80.5, 60.25, False, "C:/out", "art")
    assert loaded.background_removed is True
    assert loaded.background_tolerance == 21
    assert loaded.base_visible is False


def test_negative_references_are_remapped_to_new_ids(tmp_path):
    original = sample_project()
    loaded = load_project(save_project(tmp_path / "a.prj", original))
    copper, _, negative = loaded.layers
    assert negative.negative_of == [copper.id]
    assert negative.is_derived
    # New runtime ids, but references still resolve.
    assert loaded.get_layer(negative.negative_of[0]) is copper


def test_layer_order_preserved(tmp_path):
    project = sample_project()
    project.move_layer(project.layers[2].id, -2)  # negative first
    loaded = load_project(save_project(tmp_path / "a.prj", project))
    assert [l.name for l in loaded.layers] == ["Negative", "My copper", "Outline"]
    assert loaded.layers[0].negative_of == [loaded.layers[1].id]


def test_default_board_settings_roundtrip(tmp_path):
    loaded = load_project(save_project(tmp_path / "a.prj", Project()))
    assert loaded.board.width_mm is None and loaded.board.height_mm is None
    assert loaded.layers == []


# ---- image path resolution ---------------------------------------------------------------


def test_image_path_roundtrip(tmp_path):
    img = tmp_path / "pic.png"
    Image.new("RGB", (4, 4)).save(img)
    loaded = load_project(save_project(tmp_path / "a.prj", sample_project(img)))
    assert loaded.image_path == str(img)


def test_relative_image_path_survives_moving_the_project_folder(tmp_path):
    src = tmp_path / "proj"
    src.mkdir()
    Image.new("RGB", (4, 4)).save(src / "pic.png")
    save_project(src / "a.prj", sample_project(src / "pic.png"))

    moved = tmp_path / "moved"
    shutil.copytree(src, moved)
    shutil.rmtree(src)  # the absolute path no longer exists
    loaded = load_project(moved / "a.prj")
    assert loaded.image_path == str(moved / "pic.png")


def test_missing_image_keeps_a_path_for_the_error_message(tmp_path):
    save_project(tmp_path / "a.prj", sample_project(tmp_path / "gone.png"))
    loaded = load_project(tmp_path / "a.prj")
    assert loaded.image_path is not None


def test_project_without_image(tmp_path):
    loaded = load_project(save_project(tmp_path / "a.prj", Project()))
    assert loaded.image_path is None


# ---- validation ---------------------------------------------------------------------------


def valid_dict(tmp_path):
    return project_io.project_to_dict(sample_project(), tmp_path)


def write(tmp_path, data, name="bad.prj"):
    path = tmp_path / name
    path.write_text(json.dumps(data) if not isinstance(data, str) else data)
    return path


def test_invalid_json_rejected(tmp_path):
    with pytest.raises(ProjectFileError, match="JSON"):
        load_project(write(tmp_path, "{not json"))


def test_non_project_json_rejected(tmp_path):
    with pytest.raises(ProjectFileError, match="not a png_to_gerber"):
        load_project(write(tmp_path, {"hello": "world"}))
    with pytest.raises(ProjectFileError):
        load_project(write(tmp_path, "[1, 2, 3]"))


def test_binary_file_rejected(tmp_path):
    path = tmp_path / "b.prj"
    path.write_bytes(b"\xff\xfe\x00\x80\x81")
    with pytest.raises(ProjectFileError):
        load_project(path)


def test_missing_file_raises_oserror(tmp_path):
    with pytest.raises(OSError):
        load_project(tmp_path / "nope.prj")


@pytest.mark.parametrize("version", [0, 2, 99, "1", None])
def test_unsupported_version_rejected(tmp_path, version):
    data = valid_dict(tmp_path)
    data["version"] = version
    with pytest.raises(ProjectFileError, match="version"):
        load_project(write(tmp_path, data))


def mutate(tmp_path, fn):
    data = copy.deepcopy(valid_dict(tmp_path))
    fn(data)
    return write(tmp_path, data)


@pytest.mark.parametrize(
    "fn, match",
    [
        (lambda d: d["layers"][0].update(type="laser"), "layer type"),
        (lambda d: d["layers"][0].update(side="left"), "side"),
        (lambda d: d["layers"][0].update(color=[1, 2]), "color"),
        (lambda d: d["layers"][0].update(color=[1, 2, 300]), "color"),
        (lambda d: d["layers"][0].update(tolerance="high"), "tolerance"),
        (lambda d: d["layers"][0].update(tolerance=True), "tolerance"),
        (lambda d: d["layers"][0].update(smoothness=101), "smoothness"),
        (lambda d: d["layers"][0].update(visible="yes"), "visible"),
        (lambda d: d["layers"][0].update(name=""), "name"),
        (lambda d: d["layers"][1].update(outline_mode="middle"), "outline_mode"),
        (lambda d: d["layers"][2].update(negative_of=[99]), "negative_of"),
        (lambda d: d["layers"][2].update(negative_of=[2]), "negative_of"),
        (lambda d: d["layers"][1].update(negative_of=[0]), "cut layer"),
        (lambda d: d["board"].update(width_mm=-5), "width_mm"),
        (lambda d: d["board"].update(width_mm="wide"), "width_mm"),
        (lambda d: d["board"].update(output_dir=5), "output_dir"),
        (lambda d: d["background"].update(tolerance=-1), "tolerance"),
        (lambda d: d.update(layers="nope"), "layers"),
        (lambda d: d["layers"].append("not an object"), "object"),
    ],
)
def test_malformed_fields_rejected(tmp_path, fn, match):
    with pytest.raises(ProjectFileError, match=match):
        load_project(mutate(tmp_path, fn))


def test_two_cut_layers_rejected(tmp_path):
    def add_cut(d):
        d["layers"].append(dict(d["layers"][1], name="Second outline"))

    with pytest.raises(ProjectFileError, match="cut"):
        load_project(mutate(tmp_path, add_cut))


def test_negative_cycle_rejected(tmp_path):
    def make_cycle(d):
        d["layers"][0]["negative_of"] = [2]  # copper <- negative <- copper

    with pytest.raises(ProjectFileError, match="cycle"):
        load_project(mutate(tmp_path, make_cycle))


def test_optional_sections_may_be_missing(tmp_path):
    data = {"format": "png_to_gerber_project", "version": 1}
    loaded = load_project(write(tmp_path, data))
    assert loaded.layers == []
    assert loaded.board == BoardSettings()


# ---- GUI round trip -------------------------------------------------------------------------


@pytest.fixture
def window(qtbot, fixture_image_path):
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(fixture_image_path)
    return win


def build_scene(window):
    copper = window.add_layer(LayerType.COPPER)
    window.inspector.apply_color(COPPER_COLOR)
    copper.tolerance = 20
    copper.name = "Pads"
    mask = window.add_layer(LayerType.MASK)
    mask.side = Side.BACK
    window.inspector.apply_color(MASK_COLOR)
    silk = window.add_layer(LayerType.SILKSCREEN)
    window.inspector.apply_color(SILK_COLOR)
    window.set_layer_visible(silk.id, False)
    cut = window.add_layer(LayerType.CUT)
    window.inspector.apply_color(BOARD_COLOR)
    cut.smoothness = 40
    cut.outline_mode = OutlineMode.OUTER
    neg = window.generate_negative([copper.id, mask.id], LayerType.SILKSCREEN)
    window.mark = neg
    window.project.board = BoardSettings(70.0, 52.5, True, "", "roundtrip")
    window.flush_recompute()


def test_save_and_open_restores_layers_masks_and_outline(qtbot, window, fixture_image_path, tmp_path):
    build_scene(window)
    path = window.save_project(tmp_path / "work.prj")
    assert path.suffix == ".prj" and window.project_path == path

    fresh = MainWindow()
    qtbot.addWidget(fresh)
    fresh.open_project(path)

    assert [l.name for l in fresh.project.layers] == [l.name for l in window.project.layers]
    assert fresh.layer_pane.list.count() == len(window.project.layers) + 1
    for old, new in zip(window.project.layers, fresh.project.layers):
        assert np.array_equal(window.masks[old.id], fresh.masks[new.id])
        assert new.visible == old.visible
        assert new.side == old.side
    old_cut, new_cut = window.project.cut_layer(), fresh.project.cut_layer()
    assert fresh.outlines[new_cut.id].equals(window.outlines[old_cut.id])
    # Negative refs resolve to the *new* layers.
    new_neg = fresh.project.layers[-1]
    assert [fresh.project.get_layer(s).name for s in new_neg.negative_of] == ["Pads", "Mask opening 1"]
    assert fresh.project.board.width_mm == 70.0
    assert fresh.project.board.base_name == "roundtrip"
    assert fresh.windowTitle().startswith("work.prj")


def test_open_project_restores_hidden_layer_overlay_state(qtbot, window, tmp_path):
    build_scene(window)
    path = window.save_project(tmp_path / "work.prj")
    fresh = MainWindow()
    qtbot.addWidget(fresh)
    fresh.open_project(path)
    silk = next(l for l in fresh.project.layers if l.layer_type == LayerType.SILKSCREEN and not l.is_derived)
    assert not fresh.canvas.overlay_items[silk.id].isVisible()


def test_saved_background_and_base_visibility_restored(qtbot, tmp_path):
    img = np.zeros((40, 40, 4), dtype=np.uint8)
    img[..., :3] = 255
    img[..., 3] = 255
    img[10:30, 10:30, :3] = (0, 0, 255)
    path_img = tmp_path / "p.png"
    Image.fromarray(img, "RGBA").save(path_img)

    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(path_img)
    win.set_background_removed(True, tolerance=18)
    win.base_visible_action.setChecked(False)
    prj = win.save_project(tmp_path / "bg.prj")

    fresh = MainWindow()
    qtbot.addWidget(fresh)
    fresh.open_project(prj)
    assert fresh.remove_bg_action.isChecked()
    assert fresh.image.shape[:2] == (20, 20)  # cropped to the blue square
    assert (fresh.image[..., 3] == 255).all()
    assert tuple(fresh.image[0, 0, :3]) == (0, 0, 255)
    assert fresh.bg_tolerance_spin.value() == 18
    assert not fresh.base_visible_action.isChecked()
    assert not fresh.canvas.base_item.isVisible()


def test_resave_overwrites_same_file(window, tmp_path):
    build_scene(window)
    path = window.save_project(tmp_path / "work.prj")
    window.add_layer(LayerType.COPPER)
    again = window.save_project()  # no path -> reuse project_path
    assert again == path
    assert len(json.loads(path.read_text())["layers"]) == 6


def test_save_without_path_raises(window):
    with pytest.raises(ProjectFileError):
        window.save_project()


def test_board_settings_saved_from_export_dialog_values(window, tmp_path):
    window.project.board = BoardSettings(33.0, 24.75, False, str(tmp_path), "named")
    window.save_project(tmp_path / "b.prj")
    data = json.loads((tmp_path / "b.prj").read_text())
    assert data["board"] == {
        "width_mm": 33.0, "height_mm": 24.75, "lock_aspect": False,
        "output_dir": str(tmp_path), "base_name": "named",
    }


def test_export_dialog_prefills_from_saved_board(qtbot, window, tmp_path):
    from png_to_gerber.gui.export_dialog import ExportDialog

    window.project.board = BoardSettings(90.0, 40.0, False, str(tmp_path), "named")
    dialog = ExportDialog(200, 150, "x", settings=window.project.board)
    qtbot.addWidget(dialog)
    assert dialog.width_spin.value() == 90.0
    assert dialog.height_spin.value() == 40.0  # not overwritten by aspect lock
    assert not dialog.lock_aspect.isChecked()
    assert dialog.name_edit.text() == "named"
    assert dialog.dir_edit.text() == str(tmp_path)
    assert dialog.board_settings() == BoardSettings(90.0, 40.0, False, str(tmp_path), "named")


def test_board_settings_dialog_mode_needs_no_output_folder(qtbot):
    from png_to_gerber.gui.export_dialog import ExportDialog

    dialog = ExportDialog(200, 150, "x", export_mode=False)
    qtbot.addWidget(dialog)
    assert dialog.validate() is None
    dialog.name_edit.setText("bad/name")
    assert dialog.validate() is not None


def test_export_after_reopen_matches_original_export(qtbot, window, tmp_path):
    build_scene(window)
    prj = window.save_project(tmp_path / "w.prj")
    first = window.export_gerbers(tmp_path / "one", "a", 70.0, 52.5)
    fresh = MainWindow()
    qtbot.addWidget(fresh)
    fresh.open_project(prj)
    second = fresh.export_gerbers(tmp_path / "two", "a", 70.0, 52.5)
    assert sorted(p.name for p in first) == sorted(p.name for p in second)
    for a, b in zip(sorted(first), sorted(second)):
        assert a.read_text() == b.read_text()


def test_open_project_with_explicit_image_override(qtbot, window, fixture_image_path, tmp_path):
    build_scene(window)
    prj = window.save_project(tmp_path / "w.prj")
    other = tmp_path / "renamed.png"
    shutil.copy(fixture_image_path, other)
    data = json.loads(prj.read_text())
    data["image"] = {"absolute": str(tmp_path / "gone.png"), "relative": "gone.png"}
    prj.write_text(json.dumps(data))

    fresh = MainWindow()
    qtbot.addWidget(fresh)
    fresh.open_project(prj, image_path=other)
    assert fresh.image_path == other
    assert len(fresh.project.layers) == len(window.project.layers)


def test_missing_image_prompts_to_locate(qtbot, window, fixture_image_path, tmp_path, monkeypatch):
    build_scene(window)
    prj = window.save_project(tmp_path / "w.prj")
    data = json.loads(prj.read_text())
    data["image"] = {"absolute": str(tmp_path / "gone.png"), "relative": None}
    prj.write_text(json.dumps(data))

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *a, **k: (str(fixture_image_path), "")
    )
    fresh = MainWindow()
    qtbot.addWidget(fresh)
    fresh.open_project(prj)
    assert fresh.image_path == fixture_image_path
    assert fresh.project.image_path == str(fixture_image_path)


def test_missing_image_and_cancelled_dialog_leaves_window_untouched(qtbot, window, tmp_path, monkeypatch):
    build_scene(window)
    prj = window.save_project(tmp_path / "w.prj")
    data = json.loads(prj.read_text())
    data["image"] = {"absolute": str(tmp_path / "gone.png"), "relative": None}
    prj.write_text(json.dumps(data))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))

    fresh = MainWindow()
    qtbot.addWidget(fresh)
    with pytest.raises(ProjectFileError):
        fresh.open_project(prj)
    assert fresh.image is None and fresh.project.layers == []


def test_corrupt_project_leaves_current_work_intact(window, tmp_path):
    build_scene(window)
    before = len(window.project.layers)
    bad = tmp_path / "bad.prj"
    bad.write_text("{oops")
    with pytest.raises(ProjectFileError):
        window.open_project(bad)
    assert len(window.project.layers) == before
    assert window.image is not None


def test_opening_project_replaces_existing_layers(qtbot, window, tmp_path):
    build_scene(window)
    prj = window.save_project(tmp_path / "w.prj")
    other = MainWindow()
    qtbot.addWidget(other)
    other.load_image(window.image_path)
    other.add_layer(LayerType.COPPER)
    other.open_project(prj)
    assert len(other.project.layers) == len(window.project.layers)
    assert other.layer_pane.list.count() == len(window.project.layers) + 1


def test_open_project_via_menu_handler_reports_errors(window, tmp_path, monkeypatch):
    bad = tmp_path / "bad.prj"
    bad.write_text("nope")
    errors = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(bad), ""))
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a))
    window._on_open_project()
    assert errors


def test_save_as_handler_writes_file(window, tmp_path, monkeypatch):
    target = tmp_path / "viaDialog"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), "")
    )
    window._on_save_as()
    assert (tmp_path / "viaDialog.prj").exists()
    assert window.project_path == tmp_path / "viaDialog.prj"


def test_save_handler_without_path_falls_back_to_save_as(window, tmp_path, monkeypatch):
    target = tmp_path / "first.prj"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), "")
    )
    window._on_save()
    assert target.exists()


def test_save_failure_is_reported(window, tmp_path, monkeypatch):
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a))
    blocker = tmp_path / "file.txt"
    blocker.write_text("x")
    window._try_save(blocker / "inside.prj")  # parent is a file -> OSError
    assert errors
