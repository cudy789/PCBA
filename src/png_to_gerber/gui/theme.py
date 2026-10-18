"""A modern dark palette for the whole application."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

# Cool neutrals with a single blue accent. Lightness stays high enough that
# text, icons, and the checkerboard remain easy to tell apart.
_WINDOW = "#1c1c1f"
_BASE = "#252529"
_ALT = "#2e2e34"
_BUTTON = "#34343b"
_TEXT = "#ececf1"
_DISABLED = "#8e8e99"
_MID = "#3f3f48"
_HIGHLIGHT = "#4c6fff"


def apply_dark_theme(app: QApplication | None) -> None:
    """Switch `app` to Fusion with a dark palette. Safe to call more than once."""
    if app is None:
        return
    app.setStyle("Fusion")
    palette = QPalette()
    window = QColor(_WINDOW)
    base = QColor(_BASE)
    alt = QColor(_ALT)
    button = QColor(_BUTTON)
    text = QColor(_TEXT)
    disabled = QColor(_DISABLED)
    mid = QColor(_MID)
    highlight = QColor(_HIGHLIGHT)

    palette.setColor(QPalette.Window, window)
    palette.setColor(QPalette.WindowText, text)
    palette.setColor(QPalette.Base, base)
    palette.setColor(QPalette.AlternateBase, alt)
    palette.setColor(QPalette.ToolTipBase, alt)
    palette.setColor(QPalette.ToolTipText, text)
    palette.setColor(QPalette.Text, text)
    palette.setColor(QPalette.Button, button)
    palette.setColor(QPalette.ButtonText, text)
    palette.setColor(QPalette.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.Highlight, highlight)
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.PlaceholderText, disabled)
    palette.setColor(QPalette.Light, QColor("#4a4a54"))
    palette.setColor(QPalette.Midlight, QColor("#3a3a44"))
    palette.setColor(QPalette.Dark, QColor("#121214"))
    palette.setColor(QPalette.Mid, mid)
    palette.setColor(QPalette.Shadow, QColor("#000000"))
    palette.setColor(QPalette.Link, highlight)

    for role, color in (
        (QPalette.WindowText, disabled),
        (QPalette.Text, disabled),
        (QPalette.ButtonText, disabled),
        (QPalette.Highlight, mid),
        (QPalette.HighlightedText, disabled),
    ):
        palette.setColor(QPalette.Disabled, role, color)

    app.setPalette(palette)
    app.setStyleSheet(
        f"""
        QToolTip {{
            color: {_TEXT};
            background: {_ALT};
            border: 1px solid {_MID};
        }}
        QMenuBar {{
            background: {_WINDOW};
            color: {_TEXT};
        }}
        QMenuBar::item:selected {{
            background: {_ALT};
        }}
        QMenu {{
            background: {_BASE};
            color: {_TEXT};
            border: 1px solid {_MID};
        }}
        QMenu::item:selected {{
            background: {_HIGHLIGHT};
            color: #ffffff;
        }}
        QToolBar {{
            background: {_WINDOW};
            border: none;
            spacing: 8px;
            padding: 2px 6px;
        }}
        QStatusBar {{
            background: {_WINDOW};
            color: {_TEXT};
        }}
        """
    )
