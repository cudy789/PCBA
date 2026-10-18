"""Application entry point."""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow
    from .theme import apply_dark_theme

    app = QApplication(sys.argv)
    app.setApplicationName("PNG to Gerber")
    apply_dark_theme(app)
    window = MainWindow()
    window.show()
    if len(sys.argv) > 1:
        window.load_image(sys.argv[1])
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
