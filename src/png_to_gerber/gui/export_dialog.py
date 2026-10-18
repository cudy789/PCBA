"""Board/export dialog: output location, base name, and physical board size."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
)

from ..model import BoardSettings

DEFAULT_WIDTH_MM = 50.0


class ExportDialog(QDialog):
    """Collects export settings; validation errors appear in `error_label`.

    With `export_mode=False` it only edits the saved board settings (no output
    folder or file name is required).
    """

    def __init__(
        self,
        image_w: int,
        image_h: int,
        default_name: str,
        parent=None,
        settings: BoardSettings | None = None,
        export_mode: bool = True,
    ):
        super().__init__(parent)
        self.export_mode = export_mode
        self.setWindowTitle("Export Gerber files" if export_mode else "Board settings")
        self._aspect = image_w / image_h if image_h else 1.0
        self._syncing = False
        settings = settings or BoardSettings()

        self.dir_edit = QLineEdit(settings.output_dir, self)
        browse = QPushButton("Browse...", self)
        browse.clicked.connect(self._on_browse)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.dir_edit)
        dir_row.addWidget(browse)

        self.name_edit = QLineEdit(settings.base_name or default_name, self)

        self.width_spin = QDoubleSpinBox(self)
        self.width_spin.setRange(0.1, 10000.0)
        self.width_spin.setDecimals(2)
        self.width_spin.setSuffix(" mm")
        self.height_spin = QDoubleSpinBox(self)
        self.height_spin.setRange(0.1, 10000.0)
        self.height_spin.setDecimals(2)
        self.height_spin.setSuffix(" mm")

        width = settings.width_mm or DEFAULT_WIDTH_MM
        # A locked aspect always follows the image currently being exported,
        # including after background removal crops that image.
        if settings.lock_aspect or settings.height_mm is None:
            height = round(width / self._aspect, 2)
        else:
            height = settings.height_mm
        self._syncing = True
        self.width_spin.setValue(width)
        self.height_spin.setValue(height)
        self._syncing = False
        self.width_spin.valueChanged.connect(self._on_width)
        self.height_spin.valueChanged.connect(self._on_height)

        self.lock_aspect = QCheckBox("Lock aspect ratio to image", self)
        self.lock_aspect.setChecked(settings.lock_aspect)

        self.error_label = QLabel(self)
        self.error_label.setStyleSheet("color: #C04040;")
        self.error_label.setWordWrap(True)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        self.buttons.accepted.connect(self._on_accept)
        self.buttons.rejected.connect(self.reject)

        form = QFormLayout(self)
        form.addRow("Output folder", dir_row)
        form.addRow("Base name", self.name_edit)
        form.addRow("Board width", self.width_spin)
        form.addRow("Board height", self.height_spin)
        form.addRow("", self.lock_aspect)
        form.addRow(self.error_label)
        form.addRow(self.buttons)

    # ---- values ---------------------------------------------------------------

    def values(self) -> tuple[Path, str, float, float]:
        return (
            Path(self.dir_edit.text().strip()),
            self.name_edit.text().strip(),
            self.width_spin.value(),
            self.height_spin.value(),
        )

    def board_settings(self) -> BoardSettings:
        """The dialog contents as saveable project board settings."""
        return BoardSettings(
            width_mm=self.width_spin.value(),
            height_mm=self.height_spin.value(),
            lock_aspect=self.lock_aspect.isChecked(),
            output_dir=self.dir_edit.text().strip(),
            base_name=self.name_edit.text().strip(),
        )

    def validate(self) -> str | None:
        """Return an error message, or None when the settings are usable."""
        out_dir, name, width, height = self.values()
        has_dir = bool(self.dir_edit.text().strip())
        if self.export_mode and not has_dir:
            return "Choose an output folder."
        if has_dir and out_dir.exists() and not out_dir.is_dir():
            return "The output path exists but is not a folder."
        if self.export_mode and not name:
            return "Enter a base name for the Gerber files."
        if any(c in name for c in '<>:"/\\|?*'):
            return "The base name contains characters not allowed in filenames."
        if width <= 0 or height <= 0:
            return "Board width and height must be positive."
        return None

    # ---- handlers ---------------------------------------------------------------

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

    def _on_width(self, value: float):
        if self._syncing or not self.lock_aspect.isChecked():
            return
        self._syncing = True
        self.height_spin.setValue(value / self._aspect)
        self._syncing = False

    def _on_height(self, value: float):
        if self._syncing or not self.lock_aspect.isChecked():
            return
        self._syncing = True
        self.width_spin.setValue(value * self._aspect)
        self._syncing = False
