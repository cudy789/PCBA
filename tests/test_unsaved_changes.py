"""Unsaved-changes tracking and the Save / Discard / Cancel prompts."""

import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox

from png_to_gerber.gui.main_window import MainWindow
from png_to_gerber.model import BoardSettings, LayerType

from conftest import COPPER_COLOR, SILK_COLOR

Save = QMessageBox.StandardButton.Save
Discard = QMessageBox.StandardButton.Discard
Cancel = QMessageBox.StandardButton.Cancel


@pytest.fixture
def window(qtbot, fixture_image_path):
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(fixture_image_path)
    return win


@pytest.fixture
def prompt(monkeypatch):
    """Control the answer to the unsaved-changes dialog and record its calls."""

    class Prompt:
        answer = Discard
        calls = []

    def fake_warning(parent, title, text, buttons=None, default=None):
        Prompt.calls.append((title, text))
        return Prompt.answer

    Prompt.calls = []
    Prompt.answer = Discard
    monkeypatch.setattr(QMessageBox, "warning", fake_warning)
    return Prompt


def saved_project(window, tmp_path, name="work.prj"):
    layer = window.add_layer(LayerType.COPPER)
    window.inspector.apply_color(COPPER_COLOR)
    path = window.save_project(tmp_path / name)
    assert not window.is_modified
    return layer, path


# ---- what counts as a change -------------------------------------------------------------------


def test_new_window_is_unmodified(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    assert not win.is_modified
    assert not win.windowTitle().startswith("*")


def test_just_opening_an_image_is_not_a_change(window):
    assert not window.is_modified


def test_adding_a_layer_marks_modified_and_titles_it(window):
    window.add_layer(LayerType.COPPER)
    assert window.is_modified
    assert window.windowTitle().startswith("*")


def test_saving_clears_the_modified_flag_and_star(window, tmp_path):
    _, path = saved_project(window, tmp_path)
    assert not window.windowTitle().startswith("*")
    assert path.name in window.windowTitle()


def test_opening_a_project_leaves_it_unmodified(qtbot, window, tmp_path):
    _, path = saved_project(window, tmp_path)
    fresh = MainWindow()
    qtbot.addWidget(fresh)
    fresh.open_project(path)
    assert not fresh.is_modified


def test_load_image_into_an_existing_project_is_a_change(window, tmp_path, fixture_image_path):
    saved_project(window, tmp_path)
    window.load_image(fixture_image_path)
    assert window.is_modified


def edit_add_layer(w, layer):
    w.add_layer(LayerType.SILKSCREEN)


def edit_rename(w, layer):
    w.layer_pane.layer_item(layer.id).setText("renamed")


def edit_visibility(w, layer):
    w.layer_pane.toggle_layer_visibility(layer.id)


def edit_color(w, layer):
    w.layer_pane.select_layer(layer.id)
    w.inspector.apply_color(SILK_COLOR)


def edit_tolerance(w, layer):
    w.layer_pane.select_layer(layer.id)
    w.inspector.tolerance_slider.setValue(layer.tolerance + 5)


def edit_remove(w, layer):
    w.remove_layer(layer.id)


def edit_move(w, layer):
    w.add_layer(LayerType.SILKSCREEN)
    w.flush_recompute()


def edit_negative(w, layer):
    w.generate_negative([layer.id], LayerType.MASK)


def edit_background(w, layer):
    w.set_background_removed(True)


def edit_bg_tolerance(w, layer):
    w.base_panel.tolerance_spin.setValue(30)


def edit_base_visible(w, layer):
    w.base_visible_action.setChecked(False)


def edit_checker(w, layer):
    w.checker_slider.setValue(10)


def edit_board(w, layer):
    w._store_board_settings(BoardSettings(12.0, 8.0, True, "", "x"))


@pytest.mark.parametrize(
    "edit",
    [
        edit_add_layer, edit_rename, edit_visibility, edit_color, edit_tolerance,
        edit_remove, edit_move, edit_negative, edit_background, edit_bg_tolerance,
        edit_base_visible, edit_checker, edit_board,
    ],
    ids=lambda f: f.__name__[5:],
)
def test_every_saved_setting_marks_the_project_modified(window, tmp_path, edit):
    layer, _ = saved_project(window, tmp_path)
    edit(window, layer)
    assert window.is_modified


def test_identical_values_do_not_mark_modified(window, tmp_path):
    layer, _ = saved_project(window, tmp_path)
    window.layer_pane.layer_item(layer.id).setText(layer.name)  # same name
    window.checker_slider.setValue(window.checker_slider.value())
    window.base_panel.tolerance_spin.setValue(window.project.background_tolerance)
    window.set_layer_visible(layer.id, True)  # already visible
    assert not window.is_modified


def test_recomputation_alone_is_not_a_change(window, tmp_path):
    saved_project(window, tmp_path)
    window.flush_recompute()
    window.canvas.set_checker_opacity(10)  # view-only call, not via the slider
    assert not window.is_modified


# ---- opening another project ----------------------------------------------------------------------


@pytest.fixture
def other_project(qtbot, fixture_image_path, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(fixture_image_path)
    win.add_layer(LayerType.SILKSCREEN)
    win.inspector.apply_color(SILK_COLOR)
    return win.save_project(tmp_path / "other.prj")


def pick_file(monkeypatch, path):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))


def test_new_project_clears_images_layers_and_history(window, tmp_path):
    window.add_layer(LayerType.COPPER)
    window.save_project(tmp_path / "kept.prj")
    window.add_layer(LayerType.SILKSCREEN)
    window.checker_slider.setValue(80)
    window._store_board_settings(BoardSettings(12.0, 9.0, True, "", "board"))
    assert window.undo_action.isEnabled()

    window.new_project()

    assert window.project.images == []
    assert window.project.layers == []
    assert window.project_path is None
    assert window.project.board == BoardSettings()
    assert not window.is_modified
    assert window.windowTitle().startswith("untitled")
    assert not window.undo_action.isEnabled()
    assert not window.redo_action.isEnabled()
    assert window.checker_slider.value() == 50
    assert window.canvas.base_item is None
    assert window.layer_pane.list.count() == 1


def test_new_project_without_changes_does_not_prompt(window, prompt):
    window._on_new_project()
    assert prompt.calls == []
    assert window.project.images == []


def test_new_project_prompts_when_modified(window, prompt):
    window.add_layer(LayerType.COPPER)
    window._on_new_project()
    assert len(prompt.calls) == 1
    title, text = prompt.calls[0]
    assert title == "Unsaved changes" and "starting a new project" in text
    assert window.project.images == []


def test_cancel_keeps_the_project_when_starting_new(window, prompt):
    window.add_layer(LayerType.COPPER)
    prompt.answer = Cancel
    window._on_new_project()
    assert window.is_modified
    assert len(window.project.layers) == 1
    assert window.project_path is None


def test_discard_starts_a_new_project_without_saving(window, tmp_path, prompt):
    _, mine = saved_project(window, tmp_path)
    window.add_layer(LayerType.SILKSCREEN)
    prompt.answer = Discard
    window._on_new_project()
    import json

    assert len(json.loads(mine.read_text())["layers"]) == 1
    assert window.project.images == []
    assert window.project_path is None
    assert not window.is_modified


def test_save_writes_the_project_before_starting_new(window, tmp_path, prompt, monkeypatch):
    window.add_layer(LayerType.COPPER)
    prompt.answer = Save
    target = tmp_path / "before_new.prj"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    window._on_new_project()
    assert target.exists()
    assert window.project.images == []
    assert window.project_path is None
    assert not window.is_modified


def test_cancelling_save_does_not_start_a_new_project(window, prompt, monkeypatch):
    window.add_layer(LayerType.COPPER)
    prompt.answer = Save
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    window._on_new_project()
    assert window.project_path is None
    assert window.is_modified
    assert len(window.project.layers) == 1


def test_open_project_without_changes_does_not_prompt(window, tmp_path, other_project, prompt, monkeypatch):
    saved_project(window, tmp_path)
    pick_file(monkeypatch, other_project)
    window._on_open_project()
    assert prompt.calls == []
    assert window.project_path == other_project


def test_open_project_prompts_when_modified(window, tmp_path, other_project, prompt, monkeypatch):
    saved_project(window, tmp_path)
    window.add_layer(LayerType.SILKSCREEN)
    pick_file(monkeypatch, other_project)
    window._on_open_project()
    assert len(prompt.calls) == 1
    title, text = prompt.calls[0]
    assert title == "Unsaved changes" and "opening another project" in text


def test_cancel_keeps_current_project_and_never_opens_the_dialog(window, tmp_path, other_project, prompt, monkeypatch):
    _, mine = saved_project(window, tmp_path)
    window.add_layer(LayerType.SILKSCREEN)
    prompt.answer = Cancel
    opened = []
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *a, **k: opened.append(1) or (str(other_project), "")
    )
    window._on_open_project()
    assert not opened
    assert window.project_path == mine
    assert window.is_modified and len(window.project.layers) == 2


def test_discard_opens_the_other_project_without_saving(window, tmp_path, other_project, prompt, monkeypatch):
    _, mine = saved_project(window, tmp_path)
    window.add_layer(LayerType.SILKSCREEN)  # unsaved
    prompt.answer = Discard
    pick_file(monkeypatch, other_project)
    window._on_open_project()
    assert window.project_path == other_project
    import json

    assert len(json.loads(mine.read_text())["layers"]) == 1  # unsaved edit not written
    assert not window.is_modified


def test_save_writes_the_current_project_before_opening(window, tmp_path, other_project, prompt, monkeypatch):
    _, mine = saved_project(window, tmp_path)
    window.add_layer(LayerType.SILKSCREEN)
    prompt.answer = Save
    pick_file(monkeypatch, other_project)
    window._on_open_project()
    import json

    assert len(json.loads(mine.read_text())["layers"]) == 2  # saved first
    assert window.project_path == other_project


def test_save_for_an_untitled_project_asks_for_a_file(window, tmp_path, other_project, prompt, monkeypatch):
    window.add_layer(LayerType.COPPER)
    prompt.answer = Save
    target = tmp_path / "newly_saved.prj"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    pick_file(monkeypatch, other_project)
    window._on_open_project()
    assert target.exists()
    assert window.project_path == other_project


def test_cancelling_the_save_dialog_cancels_the_open(window, tmp_path, other_project, prompt, monkeypatch):
    window.add_layer(LayerType.COPPER)
    prompt.answer = Save
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    opened = []
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *a, **k: opened.append(1) or (str(other_project), "")
    )
    window._on_open_project()
    assert not opened
    assert window.project_path is None and window.is_modified


def test_failed_save_cancels_the_open(window, tmp_path, other_project, prompt, monkeypatch):
    _, mine = saved_project(window, tmp_path)
    window.add_layer(LayerType.SILKSCREEN)
    prompt.answer = Save
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: None)
    monkeypatch.setattr(
        type(window), "save_project", lambda self, path=None: (_ for _ in ()).throw(OSError("disk full"))
    )
    pick_file(monkeypatch, other_project)
    window._on_open_project()
    assert window.project_path == mine and window.is_modified


# ---- replacing the image ----------------------------------------------------------------------


def test_open_image_without_changes_does_not_prompt(window, fixture_image_path, prompt, monkeypatch):
    pick_file(monkeypatch, fixture_image_path)
    window._on_open_image()
    assert prompt.calls == []


def test_open_image_prompts_when_modified(window, fixture_image_path, prompt, monkeypatch):
    window.add_layer(LayerType.COPPER)
    pick_file(monkeypatch, fixture_image_path)
    window._on_open_image()
    assert len(prompt.calls) == 1
    assert "opening another image" in prompt.calls[0][1]


def test_cancel_open_image_keeps_layers(window, fixture_image_path, prompt, monkeypatch):
    window.add_layer(LayerType.COPPER)
    prompt.answer = Cancel
    opened = []
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *a, **k: opened.append(1) or (str(fixture_image_path), "")
    )
    window._on_open_image()
    assert not opened
    assert len(window.project.layers) == 1 and window.is_modified


# ---- closing the window --------------------------------------------------------------------------------


def test_closing_an_unmodified_window_does_not_prompt(window, prompt):
    assert window.close() is True
    assert prompt.calls == []


def test_closing_a_modified_window_prompts(window, prompt):
    window.add_layer(LayerType.COPPER)
    prompt.answer = Discard
    assert window.close() is True
    assert "closing" in prompt.calls[0][1]


def test_cancel_keeps_the_window_open(window, prompt):
    window.add_layer(LayerType.COPPER)
    prompt.answer = Cancel
    assert window.close() is False
    assert window.is_modified


def test_save_then_close(window, tmp_path, prompt):
    _, path = saved_project(window, tmp_path)
    window.add_layer(LayerType.SILKSCREEN)
    prompt.answer = Save
    assert window.close() is True
    import json

    assert len(json.loads(path.read_text())["layers"]) == 2


def test_closing_after_saving_does_not_prompt(window, tmp_path, prompt):
    saved_project(window, tmp_path)
    assert window.close() is True
    assert prompt.calls == []


def test_cancelled_save_as_keeps_window_open(window, prompt, monkeypatch):
    window.add_layer(LayerType.COPPER)
    prompt.answer = Save
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    assert window.close() is False


def test_save_action_returns_success_flag(window, tmp_path, monkeypatch):
    window.add_layer(LayerType.COPPER)
    target = tmp_path / "s.prj"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    assert window._on_save() is True
    assert not window.is_modified
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    window.project_path = None
    assert window._on_save_as() is False
