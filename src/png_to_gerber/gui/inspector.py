"""Selected-layer inspector: color tools and mask/cut parameters."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QWidget,
)

from ..model import Layer, LayerType, OutlineMode, Side
from .layer_pane import LAYER_TYPE_LABELS


def _swatch_style(color: tuple[int, int, int]) -> str:
    return (
        f"background-color: rgb({color[0]},{color[1]},{color[2]});"
        "border: 1px solid #666; min-width: 24px; min-height: 24px;"
    )


class LayerInspector(QWidget):
    """Edits the currently selected layer. Emits `changed` after any edit."""

    changed = Signal()
    type_changed = Signal(object)  # LayerType
    eyedropper_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layer: Layer | None = None
        self._updating = False
        self._rows: dict[str, tuple[QWidget, QWidget]] = {}

        self.type_combo = QComboBox(self)
        for layer_type, label in LAYER_TYPE_LABELS.items():
            self.type_combo.addItem(label, layer_type)
        self.type_combo.setToolTip("Change what this layer exports as.")
        self.type_combo.currentIndexChanged.connect(self._on_type)

        self.side_combo = QComboBox(self)
        self.side_combo.addItem("Front", Side.FRONT)
        self.side_combo.addItem("Back", Side.BACK)
        self.side_combo.currentIndexChanged.connect(self._on_side)

        self.sources_label = QLabel("", self)
        self.sources_label.setWordWrap(True)

        self.color_button = QPushButton(self)
        self.color_button.setToolTip("Pick color (RGBA color dialog)")
        self.color_button.clicked.connect(self._on_color_dialog)
        self.hex_edit = QLineEdit(self)
        self.hex_edit.setPlaceholderText("#RRGGBB")
        self.hex_edit.editingFinished.connect(self._on_hex)
        self.eyedropper_button = QPushButton("Eyedropper", self)
        self.eyedropper_button.setCheckable(True)
        self.eyedropper_button.toggled.connect(self.eyedropper_toggled)

        color_row = QHBoxLayout()
        color_row.addWidget(self.color_button)
        color_row.addWidget(self.hex_edit)
        color_row.addWidget(self.eyedropper_button)

        self.swatch_row = QHBoxLayout()
        self.swatch_row.setSpacing(2)
        self._swatch_buttons: list[QPushButton] = []

        self.tolerance_slider = QSlider(Qt.Horizontal, self)
        self.tolerance_slider.setRange(0, 150)
        self.tolerance_slider.valueChanged.connect(self._on_tolerance)
        self.tolerance_value = QLabel("0", self)
        tol_row = QHBoxLayout()
        tol_row.addWidget(self.tolerance_slider)
        tol_row.addWidget(self.tolerance_value)

        self.close_spin = QSpinBox(self)
        self.close_spin.setRange(0, 10)
        self.close_spin.setToolTip(
            "Gap closing radius (px): fills pinholes left by anti-aliased "
            "pixels near the selected color."
        )
        self.close_spin.valueChanged.connect(self._on_close_radius)

        self.island_spin = QSpinBox(self)
        self.island_spin.setRange(0, 10000)
        self.island_spin.setToolTip("Remove specks smaller than this many pixels.")
        self.island_spin.valueChanged.connect(self._on_min_island)

        self.fill_holes_check = QCheckBox("Fill enclosed holes", self)
        self.fill_holes_check.toggled.connect(self._on_fill_holes)

        self.invert_button = QPushButton("Invert", self)
        self.invert_button.setCheckable(True)
        self.invert_button.setToolTip(
            "Paint the opaque pixels this layer does not cover."
        )
        self.invert_button.toggled.connect(self._on_invert)

        self.outline_combo = QComboBox(self)
        self.outline_combo.addItem("Outer perimeter of the color", OutlineMode.OUTER)
        self.outline_combo.addItem("Inner perimeter (inside the color)", OutlineMode.INNER)
        self.outline_combo.setToolTip(
            "Outer: trace the outside edge of the color region.\n"
            "Inner: trace the edge of the area the color encloses, e.g. the "
            "inside of a frame or border line."
        )
        self.outline_combo.currentIndexChanged.connect(self._on_outline_mode)

        self.smooth_slider = QSlider(Qt.Horizontal, self)
        self.smooth_slider.setRange(0, 100)
        self.smooth_slider.valueChanged.connect(self._on_smoothness)
        self.smooth_value = QLabel("0", self)
        smooth_row = QHBoxLayout()
        smooth_row.addWidget(self.smooth_slider)
        smooth_row.addWidget(self.smooth_value)

        self._form = QFormLayout(self)
        self._add_row("type", "Type", self.type_combo)
        self._add_row("side", "Side", self.side_combo)
        self._add_row("sources", "Negative of", self.sources_label)
        self._add_row("color", "Color", color_row)
        self._add_row("swatches", "Image colors", self.swatch_row)
        self._add_row("tolerance", "Tolerance", tol_row)
        self._add_row("close", "Gap closing", self.close_spin)
        self._add_row("island", "Min island px", self.island_spin)
        self._add_row("fill", "", self.fill_holes_check)
        self._add_row("invert", "", self.invert_button)
        self._add_row("outline", "Cut perimeter", self.outline_combo)
        self._add_row("smooth", "Cut smoothness", smooth_row)

        # Kept for callers/tests that address the labels directly.
        self.side_label = self._rows["side"][0]
        self.smooth_label = self._rows["smooth"][0]

        self.setEnabled(False)

    def _add_row(self, key: str, label_text: str, field) -> None:
        label = QLabel(label_text, self)
        if isinstance(field, QLayout):
            container = QWidget(self)
            container.setLayout(field)
            field = container
        self._form.addRow(label, field)
        self._rows[key] = (label, field)

    def _show_row(self, key: str, visible: bool) -> None:
        for widget in self._rows[key]:
            widget.setVisible(visible)

    # ---- public API ------------------------------------------------------------

    def set_dominant_colors(self, colors: list[tuple[tuple[int, int, int], int]]):
        """Populate the dominant-color swatch row."""
        for btn in self._swatch_buttons:
            btn.hide()
            btn.setParent(None)
            btn.deleteLater()
        self._swatch_buttons.clear()
        for rgb, count in colors[:10]:
            btn = QPushButton(self)
            btn.setStyleSheet(_swatch_style(rgb))
            btn.setFixedSize(24, 24)
            btn.setToolTip("#{:02X}{:02X}{:02X} ({:,} px)".format(*rgb, count))
            btn.clicked.connect(lambda checked=False, c=rgb: self.apply_color(c))
            self.swatch_row.addWidget(btn)
            self._swatch_buttons.append(btn)

    def set_layer(self, layer: Layer | None, source_names: list[str] | None = None) -> None:
        self._layer = layer
        self.setEnabled(layer is not None)
        if layer is None:
            return
        self._updating = True
        try:
            is_cut = layer.layer_type == LayerType.CUT
            derived = layer.is_derived
            self.type_combo.setCurrentIndex(self.type_combo.findData(layer.layer_type))
            self.sources_label.setText(", ".join(source_names or []))
            self.invert_button.setChecked(layer.inverted)

            self.side_combo.setCurrentIndex(0 if layer.side == Side.FRONT else 1)
            self._refresh_color_widgets()
            self.tolerance_slider.setValue(layer.tolerance)
            self.tolerance_value.setText(str(layer.tolerance))
            self.close_spin.setValue(layer.close_radius)
            self.island_spin.setValue(layer.min_island)
            self.fill_holes_check.setChecked(layer.fill_holes)
            self.outline_combo.setCurrentIndex(
                0 if layer.outline_mode == OutlineMode.OUTER else 1
            )
            self.smooth_slider.setValue(layer.smoothness)
            self.smooth_value.setText(str(layer.smoothness))

            self._show_row("side", not is_cut)
            self._show_row("sources", derived)
            for key in ("color", "swatches", "tolerance", "close"):
                self._show_row(key, not derived)
            self._show_row("fill", not is_cut and not derived)
            self._show_row("outline", is_cut)
            self._show_row("smooth", is_cut)
        finally:
            self._updating = False

    def apply_color(self, rgb: tuple[int, int, int]) -> None:
        """Set the layer's target color (from eyedropper, swatch, or dialog)."""
        if self._layer is None or self._layer.is_derived:
            return
        self._layer.color = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        self._layer.has_color = True
        self._refresh_color_widgets()
        self.changed.emit()

    # ---- handlers ----------------------------------------------------------------

    def _refresh_color_widgets(self):
        layer = self._layer
        if layer.has_color and not layer.is_derived:
            self.color_button.setStyleSheet(_swatch_style(layer.color))
            self.color_button.setText("")
            self.hex_edit.setText("#{:02X}{:02X}{:02X}".format(*layer.color))
        else:
            self.color_button.setStyleSheet("")
            self.color_button.setText("Pick...")
            self.hex_edit.setText("")

    def _on_color_dialog(self):
        if self._layer is None:
            return
        initial = QColor(*self._layer.color) if self._layer.has_color else QColor()
        color = QColorDialog.getColor(
            initial, self, "Layer color", QColorDialog.ShowAlphaChannel
        )
        if color.isValid():
            self.apply_color((color.red(), color.green(), color.blue()))

    def _on_hex(self):
        if self._layer is None or self._updating:
            return
        text = self.hex_edit.text().strip()
        color = QColor(text)
        if color.isValid():
            self.apply_color((color.red(), color.green(), color.blue()))
        else:
            self._refresh_color_widgets()

    def _on_type(self, _index: int):
        if self._layer is None or self._updating:
            return
        new_type = self.type_combo.currentData()
        if new_type is None or new_type == self._layer.layer_type:
            return
        self.type_changed.emit(new_type)

    def _on_invert(self, checked: bool):
        if self._layer is None or self._updating:
            return
        self._layer.inverted = checked
        self.changed.emit()

    def _on_side(self, index: int):
        if self._layer is None or self._updating:
            return
        self._layer.side = self.side_combo.currentData()
        self.changed.emit()

    def _on_tolerance(self, value: int):
        self.tolerance_value.setText(str(value))
        if self._layer is None or self._updating:
            return
        self._layer.tolerance = value
        self.changed.emit()

    def _on_close_radius(self, value: int):
        if self._layer is None or self._updating:
            return
        self._layer.close_radius = value
        self.changed.emit()

    def _on_min_island(self, value: int):
        if self._layer is None or self._updating:
            return
        self._layer.min_island = value
        self.changed.emit()

    def _on_fill_holes(self, checked: bool):
        if self._layer is None or self._updating:
            return
        self._layer.fill_holes = checked
        self.changed.emit()

    def _on_outline_mode(self, index: int):
        if self._layer is None or self._updating:
            return
        self._layer.outline_mode = self.outline_combo.currentData()
        self.changed.emit()

    def _on_smoothness(self, value: int):
        self.smooth_value.setText(str(value))
        if self._layer is None or self._updating:
            return
        self._layer.smoothness = value
        self.changed.emit()
