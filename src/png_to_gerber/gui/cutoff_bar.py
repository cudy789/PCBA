"""Draggable brightness-cutoff bar with a luminance histogram."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


class CutoffBar(QWidget):
    """Brightness axis, bucket colors, histogram, and sliding cutoff notches.

    Cutoffs are luminance values from 1 to 254. A notch cannot cross its
    neighbors; they stay at least one level apart.
    """

    cutoffs_changed = Signal(object)
    drag_started = Signal()
    drag_finished = Signal()

    MARGIN = 12
    TRACK_TOP = 6
    TRACK_HEIGHT = 18
    HIST_TOP = 32
    HIST_HEIGHT = 36
    HIT_RADIUS = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cutoffs: list[int] = [64, 128, 192]
        self._colors: list[tuple[int, int, int]] = []
        self._hist = np.zeros(256, dtype=np.float64)
        self._drag: int | None = None
        self.setMinimumHeight(self.HIST_TOP + self.HIST_HEIGHT + 6)
        self.setMinimumWidth(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self._refresh_tooltip()

    def sizeHint(self) -> QSize:
        return QSize(240, self.minimumHeight())

    @property
    def cutoffs(self) -> list[int]:
        return list(self._cutoffs)

    def set_state(self, cutoffs, colors, histogram=None) -> None:
        """Show these cutoffs, bucket colors, and 256-bin histogram. Does not emit."""
        self._cutoffs = [int(value) for value in cutoffs]
        self._colors = [tuple(int(channel) for channel in color) for color in colors]
        incoming = None if histogram is None else np.asarray(histogram, dtype=np.float64).reshape(-1)
        if incoming is not None and incoming.size == 256:
            self._hist = incoming
        else:
            self._hist = np.zeros(256, dtype=np.float64)
        self._refresh_tooltip()
        self.update()

    def luminance_to_x(self, luminance: float) -> float:
        span = max(1, self.width() - 2 * self.MARGIN)
        return self.MARGIN + (float(luminance) / 255.0) * span

    def x_to_luminance(self, x: float) -> int:
        span = max(1, self.width() - 2 * self.MARGIN)
        placed = (float(x) - self.MARGIN) / span
        return int(round(min(1.0, max(0.0, placed)) * 255))

    # ---- events --------------------------------------------------------------------

    def mousePressEvent(self, event):
        if not self.isEnabled() or event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        index = self._notch_at(event.position().x())
        if index is None:
            super().mousePressEvent(event)
            return
        self._drag = index
        self.grabMouse()
        self.drag_started.emit()
        self._move_drag(event.position().x())
        event.accept()

    def mouseMoveEvent(self, event):
        if self._drag is None:
            super().mouseMoveEvent(event)
            return
        self._move_drag(event.position().x())
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._drag is None:
            super().mouseReleaseEvent(event)
            return
        self._move_drag(event.position().x())
        self._drag = None
        self.releaseMouse()
        self.drag_finished.emit()
        event.accept()

    # ---- painting ------------------------------------------------------------------

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setOpacity(1.0 if self.isEnabled() else 0.4)
        self._paint_histogram(painter)
        self._paint_track(painter)
        self._paint_notches(painter)
        painter.end()

    def _paint_histogram(self, painter: QPainter) -> None:
        peak = float(self._hist.max()) if self._hist.size else 0.0
        if peak <= 0:
            return
        bottom = self.HIST_TOP + self.HIST_HEIGHT
        for level, count in enumerate(self._hist):
            if count <= 0:
                continue
            x0 = int(round(self.luminance_to_x(level)))
            x1 = int(round(self.luminance_to_x(level + 1)))
            if x1 <= x0:
                x1 = x0 + 1
            height = max(1, int(round(self.HIST_HEIGHT * (float(count) / peak))))
            red, green, blue = self._bucket_color(self._bucket_of(level))
            painter.fillRect(
                x0,
                bottom - height,
                x1 - x0,
                height,
                QColor(
                    int(red * 0.7 + 255 * 0.3),
                    int(green * 0.7 + 255 * 0.3),
                    int(blue * 0.7 + 255 * 0.3),
                    230,
                ),
            )

    def _paint_track(self, painter: QPainter) -> None:
        edges = [0, *self._cutoffs, 255]
        top = self.TRACK_TOP
        for index in range(len(edges) - 1):
            x0 = int(round(self.luminance_to_x(edges[index])))
            x1 = int(round(self.luminance_to_x(edges[-1] if index == len(edges) - 2 else edges[index + 1])))
            if x1 <= x0:
                x1 = x0 + 1
            red, green, blue = self._bucket_color(index)
            painter.fillRect(x0, top, x1 - x0, self.TRACK_HEIGHT, QColor(red, green, blue))
        painter.setPen(QPen(self.palette().color(QPalette.Mid), 1))
        painter.drawRect(
            int(round(self.luminance_to_x(0))),
            top,
            int(round(self.luminance_to_x(255) - self.luminance_to_x(0))),
            self.TRACK_HEIGHT,
        )

    def _paint_notches(self, painter: QPainter) -> None:
        color = self.palette().color(QPalette.Text)
        for cutoff in self._cutoffs:
            x = self.luminance_to_x(cutoff)
            painter.setPen(QPen(color, 2))
            painter.drawLine(QPointF(x, self.TRACK_TOP - 3), QPointF(x, self.TRACK_TOP + self.TRACK_HEIGHT + 3))
            path = QPainterPath()
            path.moveTo(x - 5, self.TRACK_TOP - 1)
            path.lineTo(x + 5, self.TRACK_TOP - 1)
            path.lineTo(x, self.TRACK_TOP + 7)
            path.closeSubpath()
            painter.fillPath(path, color)

    # ---- internals -----------------------------------------------------------------

    def _bucket_of(self, level: int) -> int:
        bucket = 0
        for cutoff in self._cutoffs:
            if level >= cutoff:
                bucket += 1
            else:
                break
        return bucket

    def _bucket_color(self, bucket: int) -> tuple[int, int, int]:
        if 0 <= bucket < len(self._colors):
            return self._colors[bucket]
        gray = min(255, 48 + bucket * 28)
        return (gray, gray, gray)

    def _notch_at(self, x: float) -> int | None:
        found = None
        best = self.HIT_RADIUS + 1
        for index, cutoff in enumerate(self._cutoffs):
            distance = abs(x - self.luminance_to_x(cutoff))
            if distance < best:
                found = index
                best = distance
        return found

    def _move_drag(self, x: float) -> None:
        index = self._drag
        if index is None:
            return
        value = self.x_to_luminance(x)
        low = 1 if index == 0 else self._cutoffs[index - 1] + 1
        high = 254 if index == len(self._cutoffs) - 1 else self._cutoffs[index + 1] - 1
        value = max(low, min(high, value))
        if value == self._cutoffs[index]:
            return
        self._cutoffs[index] = value
        self._refresh_tooltip()
        self.update()
        self.cutoffs_changed.emit(list(self._cutoffs))

    def _refresh_tooltip(self) -> None:
        listed = ", ".join(str(value) for value in self._cutoffs)
        self.setToolTip(f"Bucket cutoffs (brightness 0-255): {listed}")
