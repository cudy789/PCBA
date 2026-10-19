"""Dialog for choosing which layers to write as PNG images."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
)

from ..model import Layer, LayerType, Side

_TYPE_LABELS = {
    LayerType.COPPER: "Copper",
    LayerType.MASK: "Solder mask opening",
    LayerType.SILKSCREEN: "Silkscreen",
    LayerType.CUT: "Cut",
}
_SIDE_LABELS = {Side.FRONT: "front", Side.BACK: "back"}


def _layer_label(layer: Layer) -> str:
    kind = _TYPE_LABELS[layer.layer_type]
    if layer.is_cut:
        return f"{layer.name} — {kind}"
    return f"{layer.name} — {kind}, {_SIDE_LABELS[layer.side]}"


class ImageExportDialog(QDialog):
    """Pick layers to save as images, and optionally a rendered picture of the board."""

    def __init__(
        self,
        layers: list[tuple[Layer, bool]],
        default_name: str,
        output_dir: str = "",
        parent=None,
    ):
        """`layers` is `(layer, has_mask)` in stack order."""
        super().__init__(parent)
        self.setWindowTitle("Export image")
        self.setMinimumWidth(440)

        self.dir_edit = QLineEdit(output_dir, self)
        browse = QPushButton("Browse...", self)
        browse.clicked.connect(self._on_browse)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.dir_edit)
        dir_row.addWidget(browse)

        self.name_edit = QLineEdit(default_name, self)

        self.list = QListWidget(self)
        self.list.setSelectionMode(QListWidget.NoSelection)
        self.list.setMinimumHeight(140)
        for layer, has_mask in layers:
            item = QListWidgetItem(_layer_label(layer), self.list)
            item.setData(Qt.UserRole, layer.id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            if has_mask:
                checked = Qt.Checked if layer.visible else Qt.Unchecked
                item.setCheckState(checked)
            else:
                item.setCheckState(Qt.Unchecked)
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                item.setToolTip("Pick a color for this layer before exporting it.")

        self.render_check = QCheckBox("Export a fully rendered PCB image", self)
        self.render_check.setToolTip(
            "Write one picture of the finished board: green solder mask, "
            "copper showing through the openings, white silkscreen on top, "
            "clipped to the cut outline. Every layer that has a mask is "
            "included, including layers left unchecked above."
        )

        hint = QLabel(
            "Each checked layer is saved as its own PNG. The rendered image "
            "is a separate picture of the whole board.",
            self,
        )
        hint.setWordWrap(True)

        self.error_label = QLabel(self)
        self.error_label.setStyleSheet("color: #ff8b8b;")
        self.error_label.setWordWrap(True)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        self.buttons.accepted.connect(self._on_accept)
        self.buttons.rejected.connect(self.reject)

        form = QFormLayout(self)
        form.addRow("Output folder", dir_row)
        form.addRow("Base name", self.name_edit)
        form.addRow("Layers", self.list)
        form.addRow("", self.render_check)
        form.addRow(hint)
        form.addRow(self.error_label)
        form.addRow(self.buttons)

    def selected_ids(self) -> list[int]:
        ids = []
        for row in range(self.list.count()):
            item = self.list.item(row)
            if item.checkState() == Qt.Checked:
                ids.append(item.data(Qt.UserRole))
        return ids

    def wants_render(self) -> bool:
        return self.render_check.isChecked()

    def values(self) -> tuple[Path, str, list[int], bool]:
        return (
            Path(self.dir_edit.text().strip()),
            self.name_edit.text().strip(),
            self.selected_ids(),
            self.wants_render(),
        )

    def validate(self) -> str | None:
        out_dir, name, _ids, render = self.values()
        if not self.dir_edit.text().strip():
            return "Choose an output folder."
        if out_dir.exists() and not out_dir.is_dir():
            return "The output path exists but is not a folder."
        if not name:
            return "Enter a base name for the image files."
        if any(ch in name for ch in '<>:"/\\|?*'):
            return "The base name contains characters not allowed in filenames."
        if not self.selected_ids() and not render:
            return "Select at least one layer, or export the rendered PCB image."
        if render and not any(self._item_enabled(row) for row in range(self.list.count())):
            return "Add a layer with a color before rendering the PCB."
        return None

    def _item_enabled(self, row: int) -> bool:
        return bool(self.list.item(row).flags() & Qt.ItemIsEnabled)

    def _on_accept(self):
        error = self.validate()
        if error:
            self.error_label.setText(error)
            return
        self.accept()

    def _on_browse(self):
        path = QFileDialog.getExistingDirectory(self, "Output folder")
        if path:
            self.dir_edit.setText(path)
