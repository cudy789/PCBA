"""Small programmatically drawn icons (no image assets to bundle)."""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


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
