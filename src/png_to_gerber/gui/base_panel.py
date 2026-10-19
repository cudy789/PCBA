"""Options for the base image, shown when "Base image" is selected in the layers pane."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)


class BaseImagePanel(QWidget):
    """Image info, open/replace, and background-removal controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.info_label = QLabel("No image loaded.", self)
        self.info_label.setWordWrap(True)

        self.add_button = QPushButton("Add image...", self)
        self.add_button.setToolTip("Open another base image. Its layers are listed under it.")
        self.open_button = QPushButton("Open / replace image...", self)
        self.remove_button = QPushButton("Remove image", self)
        self.remove_button.setToolTip("Remove this base image and the layers nested under it.")
        self.remove_button.setEnabled(False)

        self.remove_bg_check = QCheckBox("Remove background", self)
        self.remove_bg_check.setToolTip(
            "Make the image's outer contiguous background color transparent "
            "(untick to restore it)."
        )
        self.tolerance_spin = QSpinBox(self)
        self.tolerance_spin.setRange(0, 100)
        self.tolerance_spin.setToolTip(
            "How different from the border color a pixel may be and still "
            "count as background."
        )
        hint = QLabel(
            "The background is the color along the image border, spreading "
            "inward through touching pixels of nearly the same color. Areas "
            "of that color enclosed by artwork are kept. The image is then "
            "cropped to the remaining artwork, so the board size matches it.",
            self,
        )
        hint.setWordWrap(True)

        form = QFormLayout(self)
        form.addRow(self.info_label)
        form.addRow(self.add_button)
        form.addRow(self.open_button)
        form.addRow(self.remove_button)
        form.addRow(self.remove_bg_check)
        form.addRow("Background tolerance", self.tolerance_spin)
        form.addRow(hint)

    def set_image_info(self, name: str | None, width: int = 0, height: int = 0) -> None:
        if name is None:
            self.info_label.setText("No image loaded.")
        else:
            self.info_label.setText(f"{name}\n{width} x {height} px")

    def sync(self, background_removed: bool, tolerance: int) -> None:
        """Show state without re-triggering the controls' signals."""
        for widget, setter, value in (
            (self.remove_bg_check, self.remove_bg_check.setChecked, background_removed),
            (self.tolerance_spin, self.tolerance_spin.setValue, tolerance),
        ):
            widget.blockSignals(True)
            setter(value)
            widget.blockSignals(False)
