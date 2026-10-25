"""Options for the base image, shown when "Base image" is selected in the layers pane."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..flatten import even_cutoffs
from .cutoff_bar import CutoffBar


def _swatch_style(color: tuple[int, int, int]) -> str:
    return (
        f"background-color: rgb({color[0]},{color[1]},{color[2]});"
        "border: 1px solid #666; min-width: 24px; min-height: 24px;"
    )


class BaseImagePanel(QWidget):
    """Image info, background removal, and color flattening."""

    color_edited = Signal(int, object)  # bucket index, (r, g, b) or None to restore the average

    def __init__(self, parent=None):
        super().__init__(parent)
        self._swatches: list[QPushButton] = []
        self._swatch_colors: list[tuple[int, int, int]] = []

        self.info_label = QLabel("No image loaded.")
        self.info_label.setWordWrap(True)

        self.add_button = QPushButton("Add image...")
        self.add_button.setToolTip("Open another base image. Its layers are listed under it.")
        self.open_button = QPushButton("Open / replace image...")
        self.remove_button = QPushButton("Remove image")
        self.remove_button.setToolTip("Remove this base image and the layers nested under it.")
        self.remove_button.setEnabled(False)

        self.remove_bg_check = QCheckBox("Remove background")
        self.remove_bg_check.setToolTip(
            "Make the image's outer contiguous background color transparent "
            "(untick to restore it)."
        )
        self.tolerance_spin = QSpinBox()
        self.tolerance_spin.setRange(0, 100)
        self.tolerance_spin.setToolTip(
            "How different from the border color a pixel may be and still "
            "count as background."
        )
        hint = QLabel(
            "The background is the color along the image border, spreading "
            "inward through touching pixels of nearly the same color. Areas "
            "of that color enclosed by artwork are kept. The image is then "
            "cropped to the remaining artwork, so the board size matches it."
        )
        hint.setWordWrap(True)

        self.flatten_check = QCheckBox("Flatten colors")
        self.flatten_check.setToolTip(
            "Reduce this image to a few brightness buckets. Untick to restore "
            "the original colors. Runs after background removal."
        )
        self.count_spin = QSpinBox()
        self.count_spin.setRange(2, 12)
        self.count_spin.setValue(4)
        self.count_spin.setToolTip("How many colors to keep.")

        self.cutoff_bar = CutoffBar()
        self.cutoff_bar.setToolTip("Drag a notch to set a bucket boundary.")

        self._swatch_host = QWidget()
        self._swatch_row = QHBoxLayout(self._swatch_host)
        self._swatch_row.setContentsMargins(0, 0, 0, 0)
        self._swatch_row.setSpacing(4)

        self.auto_button = QPushButton("Auto-place cutoffs")
        self.auto_button.setToolTip("Put the notches between the strongest brightness peaks.")
        self.reset_button = QPushButton("Reset to even spacing")
        self.reset_button.setToolTip("Spread the notches evenly. Colors you picked are kept.")
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.addWidget(self.auto_button)
        actions.addWidget(self.reset_button)
        action_host = QWidget()
        action_host.setLayout(actions)

        self.smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self.smooth_slider.setRange(0, 5)
        self.smooth_slider.setToolTip(
            "Smooth jagged edges and speckles after flattening. "
            "Does not create new colors."
        )
        self.smooth_value = QLabel("0")
        self.smooth_slider.valueChanged.connect(
            lambda value: self.smooth_value.setText(str(value))
        )
        smooth_host = QWidget()
        smooth_row = QHBoxLayout(smooth_host)
        smooth_row.setContentsMargins(0, 0, 0, 0)
        smooth_row.addWidget(self.smooth_slider)
        smooth_row.addWidget(self.smooth_value)

        self.compare_check = QCheckBox("Compare original")
        self.compare_check.setToolTip(
            "Show this image before color flattening, including when its eye is off. "
            "Layer masks still use the flattened colors."
        )
        flatten_hint = QLabel(
            "Notches are the brightness boundaries. Each swatch is the color "
            "that bucket becomes — click to change it, right-click to use the "
            "average of the pixels in that bucket."
        )
        flatten_hint.setWordWrap(True)

        content = QWidget()
        form = QFormLayout(content)
        form.addRow(self.info_label)
        form.addRow(self.add_button)
        form.addRow(self.open_button)
        form.addRow(self.remove_button)
        form.addRow(self.remove_bg_check)
        form.addRow("Background tolerance", self.tolerance_spin)
        form.addRow(hint)
        form.addRow(self.flatten_check)
        form.addRow("Colors", self.count_spin)
        form.addRow(self.cutoff_bar)
        form.addRow("Bucket colors", self._swatch_host)
        form.addRow(action_host)
        form.addRow("Edge smooth", smooth_host)
        form.addRow(self.compare_check)
        form.addRow(flatten_hint)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._flatten_controls = (
            self.count_spin,
            self.cutoff_bar,
            self._swatch_host,
            self.auto_button,
            self.reset_button,
            self.smooth_slider,
            self.compare_check,
        )
        self._set_flatten_enabled(False)
        self._rebuild_swatches([(32, 32, 32), (96, 96, 96), (160, 160, 160), (224, 224, 224)])

    def sizeHint(self) -> QSize:
        return QSize(260, 360)

    def minimumSizeHint(self) -> QSize:
        return QSize(200, 140)

    def set_image_info(self, name: str | None, width: int = 0, height: int = 0) -> None:
        if name is None:
            self.info_label.setText("No image loaded.")
        else:
            self.info_label.setText(f"{name}\n{width} x {height} px")

    @property
    def swatches(self) -> list[QPushButton]:
        return list(self._swatches)

    def sync(
        self,
        background_removed: bool,
        tolerance: int,
        *,
        flatten_enabled: bool = False,
        flatten_count: int = 4,
        cutoffs: list[int] | None = None,
        colors: list | None = None,
        smooth: int = 0,
        histogram=None,
        compare: bool = False,
    ) -> None:
        """Show state without re-triggering the controls' signals."""
        shown_cutoffs = list(cutoffs) if cutoffs is not None else even_cutoffs(flatten_count)
        shown_colors = list(colors) if colors is not None else []
        for widget, setter, value in (
            (self.remove_bg_check, self.remove_bg_check.setChecked, background_removed),
            (self.tolerance_spin, self.tolerance_spin.setValue, tolerance),
            (self.flatten_check, self.flatten_check.setChecked, flatten_enabled),
            (self.count_spin, self.count_spin.setValue, flatten_count),
            (self.smooth_slider, self.smooth_slider.setValue, smooth),
            (self.compare_check, self.compare_check.setChecked, compare),
        ):
            widget.blockSignals(True)
            setter(value)
            widget.blockSignals(False)
        self.smooth_value.setText(str(smooth))
        self._rebuild_swatches(shown_colors)
        self.cutoff_bar.set_state(shown_cutoffs, shown_colors, histogram)
        self._set_flatten_enabled(flatten_enabled)

    def _set_flatten_enabled(self, enabled: bool) -> None:
        for widget in self._flatten_controls:
            widget.setEnabled(enabled)

    def _rebuild_swatches(self, colors: list) -> None:
        rgb = [tuple(int(channel) for channel in color) for color in colors]
        if len(rgb) == len(self._swatches):
            self._swatch_colors = rgb
            for button, color in zip(self._swatches, rgb):
                button.setStyleSheet(_swatch_style(color))
                button.setToolTip(self._swatch_tip(color))
            return
        for button in self._swatches:
            button.hide()
            button.setParent(None)
            button.deleteLater()
        self._swatches.clear()
        self._swatch_colors = rgb
        for index, color in enumerate(rgb):
            button = QPushButton(self._swatch_host)
            button.setFixedSize(24, 24)
            button.setStyleSheet(_swatch_style(color))
            button.setToolTip(self._swatch_tip(color))
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda checked=False, i=index: self._pick_color(i))
            button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            button.customContextMenuRequested.connect(
                lambda _pos, i=index: self.color_edited.emit(i, None)
            )
            self._swatch_row.addWidget(button)
            self._swatches.append(button)

    @staticmethod
    def _swatch_tip(color: tuple[int, int, int]) -> str:
        return "#{:02X}{:02X}{:02X} — click to change, right-click to use the average".format(*color)

    def _pick_color(self, index: int) -> None:
        current = self._swatch_colors[index] if index < len(self._swatch_colors) else (128, 128, 128)
        color = QColorDialog.getColor(QColor(*current), self, "Bucket color")
        if color.isValid():
            self.color_edited.emit(index, (color.red(), color.green(), color.blue()))
