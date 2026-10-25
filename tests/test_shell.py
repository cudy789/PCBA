"""Menus, the checker toolbar, Gerber ZIP export, and the dark theme."""

import zipfile

from PySide6.QtGui import QKeySequence, QPalette
from PySide6.QtWidgets import QApplication, QToolBar

from png_to_gerber.gui.export_dialog import ExportDialog
from png_to_gerber.gui.main_window import MainWindow
from png_to_gerber.model import BoardSettings, LayerType

from conftest import COPPER_COLOR


def test_there_is_no_image_menu_and_file_owns_export(qtbot, fixture_image_path):
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(fixture_image_path)
    titles = [action.text().replace("&", "") for action in win.menuBar().actions()]
    assert titles == ["File", "View"]

    file_menu = win.menuBar().actions()[0].menu()
    file_texts = [action.text().replace("&", "") for action in file_menu.actions()]
    assert "New project" in file_texts
    assert file_texts.index("New project") < file_texts.index("Open project...")
    assert win.new_project_action.shortcut() == QKeySequence("Ctrl+N")
    assert "Open project..." in file_texts
    assert "Save project" in file_texts
    assert "Open image..." in file_texts
    assert "Board settings..." in file_texts
    assert "Export Gerbers..." not in file_texts
    export = next(action.menu() for action in file_menu.actions() if action.menu())
    assert export.title().replace("&", "") == "Export"
    assert [action.text().replace("&", "") for action in export.actions()] == [
        "Gerber files...",
        "Gerber ZIP...",
        "Export image...",
    ]

    view_menu = win.menuBar().actions()[1].menu()
    assert win.base_visible_action in view_menu.actions()
    for top in win.menuBar().actions():
        assert win.remove_bg_action not in top.menu().actions()


def test_export_actions_open_the_dialog(qtbot, fixture_image_path, monkeypatch):
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(fixture_image_path)
    opened = []

    def reject(dialog):
        opened.append(dialog.windowTitle())
        return ExportDialog.Rejected

    monkeypatch.setattr(ExportDialog, "exec", reject)
    win.export_files_action.trigger()
    win.export_zip_action.trigger()
    assert opened == ["Export Gerber files", "Export Gerber ZIP"]


def test_gerber_zip_uses_the_project_name_and_holds_the_files(qtbot, fixture_image_path, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    win.load_image(fixture_image_path)
    win.project_path = tmp_path / "SlayCard.prj"
    assert win._default_name() == "SlayCard"
    layer = win.add_layer(LayerType.COPPER)
    win.inspector.apply_color(COPPER_COLOR)
    win.flush_recompute()

    zip_path = win.export_gerber_zip(tmp_path, "SlayCard", 40.0, 30.0)
    assert zip_path == tmp_path / "SlayCard.zip"
    assert zip_path.is_file()
    assert list(tmp_path.glob("*.gtl")) == []
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert names == {"SlayCard-F_Cu.gtl"}
    assert layer.has_color


def test_locked_dialog_height_follows_the_image_aspect(qtbot):
    dialog = ExportDialog(
        20, 20, "x", settings=BoardSettings(80.0, 40.0, True, "", "x")
    )
    qtbot.addWidget(dialog)
    assert dialog.height_spin.value() == 80.0


def test_dark_theme_is_applied(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    palette = QApplication.instance().palette()
    window = palette.color(QPalette.Window)
    text = palette.color(QPalette.WindowText)
    assert window.lightness() < 80
    assert text.lightness() > 180
    assert "#1c1c1f" in QApplication.instance().styleSheet()
    # The checker slider is the only toolbar control besides its label.
    toolbar = win.findChildren(QToolBar)[0]
    assert all(action.text() == "" for action in toolbar.actions())
