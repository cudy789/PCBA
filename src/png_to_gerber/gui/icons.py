"""Small programmatically drawn icons (no image assets to bundle)."""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

_ICON = QColor("#ececf1")
_ICON_SIZE = 24


def eye_icon(visible: bool, color: QColor, size: int = 18) -> QIcon:
    """An open eye when `visible`, otherwise a dimmed eye with a slash through it."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    tone = QColor(color)
    if not visible:
        tone.setAlphaF(0.55)
    painter.setPen(QPen(tone, 1.5))
    centre = size / 2
    outline = QPainterPath()
    outline.moveTo(1.5, centre)
    outline.quadTo(centre, centre - size * 0.62, size - 1.5, centre)
    outline.quadTo(centre, centre + size * 0.62, 1.5, centre)
    painter.drawPath(outline)
    if visible:
        painter.setBrush(tone)
        painter.drawEllipse(QPointF(centre, centre), size * 0.16, size * 0.16)
    else:
        painter.drawLine(QPointF(3, size - 3), QPointF(size - 3, 3))
    painter.end()
    return QIcon(pixmap)


def _tool_icon(draw) -> QIcon:
    pixmap = QPixmap(_ICON_SIZE, _ICON_SIZE)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(_ICON)
    pen.setWidthF(2.0)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    draw(painter)
    painter.end()
    return QIcon(pixmap)


def brush_icon() -> QIcon:
    """A diagonal stroke with a round tip."""

    def draw(painter: QPainter) -> None:
        painter.drawLine(QPointF(6, 18), QPointF(15, 8))
        painter.setBrush(_ICON)
        painter.drawEllipse(QPointF(16.5, 6.5), 2.4, 2.4)

    return _tool_icon(draw)


def line_icon() -> QIcon:
    def draw(painter: QPainter) -> None:
        painter.drawLine(QPointF(5, 18), QPointF(19, 6))

    return _tool_icon(draw)


def rectangle_icon() -> QIcon:
    def draw(painter: QPainter) -> None:
        painter.drawRect(5, 6, 14, 12)

    return _tool_icon(draw)


def eraser_icon() -> QIcon:
    def draw(painter: QPainter) -> None:
        painter.drawRoundedRect(6, 8, 12, 9, 1.5, 1.5)
        painter.drawLine(QPointF(10, 8), QPointF(10, 17))

    return _tool_icon(draw)


def fill_icon() -> QIcon:
    """A paint bucket tipped toward a drop."""

    def draw(painter: QPainter) -> None:
        path = QPainterPath()
        path.moveTo(5, 11)
        path.lineTo(12, 5)
        path.lineTo(19, 12)
        path.lineTo(14, 17)
        path.lineTo(8, 17)
        path.closeSubpath()
        painter.drawPath(path)
        painter.setBrush(_ICON)
        painter.drawEllipse(QPointF(8, 19), 1.4, 1.4)

    return _tool_icon(draw)


def color_swatch_icon(color: tuple[int, int, int, int], size: int = 22) -> QIcon:
    """A filled square of `color`, with a checker behind any transparency."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.fillRect(0, 0, size // 2, size // 2, QColor("#b4b4b4"))
    painter.fillRect(size // 2, size // 2, size - size // 2, size - size // 2, QColor("#b4b4b4"))
    painter.fillRect(size // 2, 0, size - size // 2, size // 2, QColor("#6c6c6c"))
    painter.fillRect(0, size // 2, size // 2, size - size // 2, QColor("#6c6c6c"))
    r, g, b, a = color
    painter.fillRect(1, 1, size - 2, size - 2, QColor(r, g, b, a))
    painter.setPen(QPen(QColor("#888888")))
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(0, 0, size - 1, size - 1)
    painter.end()
    return QIcon(pixmap)
