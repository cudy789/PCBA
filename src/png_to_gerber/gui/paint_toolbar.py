"""Left-hand paint tools: brush, line, rectangle, eraser, fill, and one brush size."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QColorDialog,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..paint import BRUSH_SIZE_DEFAULT, BRUSH_SIZE_MAX, BRUSH_SIZE_MIN, Tool
from .icons import brush_icon, color_swatch_icon, eraser_icon, fill_icon, line_icon, rectangle_icon

_BUTTON_STYLE = """
QPushButton {
    background: #34343b;
    border: 1px solid #3f3f48;
    border-radius: 4px;
}
QPushButton:checked {
    background: #4c6fff;
    border: 1px solid #4c6fff;
}
QPushButton:disabled {
    background: #2a2a30;
    color: #8e8e99;
    border: 1px solid #3f3f48;
}
"""

_TOOLS = (
    (Tool.BRUSH, "Paintbrush", "Drag to add pixels.", brush_icon),
    (Tool.LINE, "Straight line", "Drag from one point to another. Hold Ctrl to snap to 45-degree steps.", line_icon),
    (
        Tool.RECTANGLE,
        "Rectangle",
        "Click one corner, then the opposite corner. Only the outline is drawn. "
        "Hold Ctrl to lock a square. Esc cancels.",
        rectangle_icon,
    ),
    (Tool.ERASE, "Eraser", "Drag to remove pixels.", eraser_icon),
    (
        Tool.FILL,
        "Fill",
        "On a layer, click an empty area to fill it. On the base image, click to replace "
        "the connected pixels of that exact color. Brush size does not apply.",
        fill_icon,
    ),
)


class PaintToolbar(QWidget):
    """Vertical tool strip. Brush size is shared by every tool except fill."""

    tool_changed = Signal(object)
    brush_size_changed = Signal(int)
    color_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("paint_toolbar")
        self._tool = Tool.PAN
        self._color = (0, 0, 0, 255)
        self._buttons: dict[Tool, QPushButton] = {}

        self.setFixedWidth(56)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        for tool, title, detail, icon in _TOOLS:
            button = QPushButton()
            button.setObjectName(f"tool_{tool.value}")
            button.setCheckable(True)
            button.setIcon(icon())
            button.setIconSize(QSize(16, 16))
            button.setFixedSize(30, 26)
            button.setToolTip(f"{title}. {detail}")
            button.setStyleSheet(_BUTTON_STYLE)
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            button.clicked.connect(lambda _checked=False, chosen=tool: self._on_clicked(chosen))
            layout.addWidget(button, alignment=Qt.AlignHCenter)
            self._buttons[tool] = button

        layout.addSpacing(4)
        self.size_caption = QLabel("Brush Size")
        self.size_caption.setWordWrap(True)
        self.size_caption.setAlignment(Qt.AlignHCenter)
        layout.addWidget(self.size_caption)

        self.size_slider = QSlider(Qt.Vertical)
        self.size_slider.setObjectName("brush_size")
        self.size_slider.setRange(BRUSH_SIZE_MIN, BRUSH_SIZE_MAX)
        self.size_slider.setValue(BRUSH_SIZE_DEFAULT)
        self.size_slider.setFixedWidth(16)
        self.size_slider.setFixedHeight(72)
        self.size_slider.setToolTip(
            "Brush diameter in pixels. Used by the paintbrush, line, rectangle, and eraser."
        )
        self.size_slider.valueChanged.connect(self._on_size)
        layout.addWidget(self.size_slider, alignment=Qt.AlignHCenter)

        self.size_edit = QSpinBox()
        self.size_edit.setObjectName("brush_size_edit")
        self.size_edit.setRange(BRUSH_SIZE_MIN, BRUSH_SIZE_MAX)
        self.size_edit.setValue(BRUSH_SIZE_DEFAULT)
        self.size_edit.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.size_edit.setAlignment(Qt.AlignCenter)
        self.size_edit.setFixedWidth(44)
        self.size_edit.setToolTip("Type a brush diameter in pixels, from 1 to 128.")
        self.size_edit.valueChanged.connect(self._on_size)
        layout.addWidget(self.size_edit, alignment=Qt.AlignHCenter)

        self.color_button = QPushButton()
        self.color_button.setObjectName("paint_color")
        self.color_button.setFixedSize(30, 20)
        self.color_button.clicked.connect(self._choose_color)
        layout.addSpacing(8)
        layout.addWidget(self.color_button, alignment=Qt.AlignHCenter)
        layout.addStretch(1)

        self._refresh_swatch()
        self.set_color_enabled(False)
        self.set_tool(Tool.PAN)

    @property
    def tool(self) -> Tool:
        return self._tool

    @property
    def brush_size(self) -> int:
        return int(self.size_slider.value())

    @property
    def color(self) -> tuple[int, int, int, int]:
        return self._color

    def button(self, tool: Tool) -> QPushButton:
        return self._buttons[tool]

    def set_tool(self, tool: Tool) -> None:
        self._tool = tool
        for known, button in self._buttons.items():
            button.setChecked(known == tool)
        # Fill ignores the diameter. Every other tool, including pan, can change it.
        uses_size = tool != Tool.FILL
        self.size_slider.setEnabled(uses_size)
        self.size_caption.setEnabled(uses_size)
        self.size_edit.setEnabled(uses_size)
        self.tool_changed.emit(tool)

    def set_brush_size(self, size: int) -> None:
        clamped = max(BRUSH_SIZE_MIN, min(BRUSH_SIZE_MAX, int(size)))
        if self.size_slider.value() != clamped:
            self.size_slider.setValue(clamped)
        elif self.size_edit.value() != clamped:
            self.size_edit.setValue(clamped)

    def set_color(self, color) -> None:
        channels = tuple(int(channel) for channel in color)
        if len(channels) == 3:
            channels = (*channels, 255)
        self._color = channels[:4]
        self._refresh_swatch()
        self.color_changed.emit(self._color)

    def set_color_enabled(self, enabled: bool) -> None:
        """The color applies only when painting the base image."""
        self.color_button.setEnabled(enabled)
        if enabled:
            self.color_button.setToolTip("Color painted onto the base image.")
        else:
            self.color_button.setToolTip(
                "Select the base image to paint with a color. "
                "Layer tools add or remove mask pixels."
            )

    def _on_clicked(self, tool: Tool) -> None:
        # Clicking the active tool returns to pan, so left-drag can move the view.
        if self._tool == tool:
            self.set_tool(Tool.PAN)
        else:
            self.set_tool(tool)

    def _on_size(self, value: int) -> None:
        value = int(value)
        if self.size_slider.value() != value:
            self.size_slider.blockSignals(True)
            self.size_slider.setValue(value)
            self.size_slider.blockSignals(False)
        if self.size_edit.value() != value:
            self.size_edit.blockSignals(True)
            self.size_edit.setValue(value)
            self.size_edit.blockSignals(False)
        self.brush_size_changed.emit(value)

    def _refresh_swatch(self) -> None:
        self.color_button.setIcon(color_swatch_icon(self._color, size=16))
        self.color_button.setIconSize(QSize(16, 16))

    def _choose_color(self) -> None:
        current = QColor(*self._color)
        chosen = QColorDialog.getColor(
            current,
            self,
            "Paint color",
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if chosen.isValid():
            self.set_color((chosen.red(), chosen.green(), chosen.blue(), chosen.alpha()))
