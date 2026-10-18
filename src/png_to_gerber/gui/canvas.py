"""Zoomable/pannable image canvas with per-layer overlay items."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

# Preview tints per layer type (RGB).
TINTS = {
    "copper": (218, 140, 30),
    "mask": (46, 160, 90),
    "silkscreen": (120, 160, 255),
    "cut": (230, 60, 60),
}
OVERLAY_ALPHA = 160
CUT_PEN_WIDTH = 2.0


def _checker_brush(opacity_percent: int = 100) -> QBrush:
    """Checkerboard shown behind transparent pixels.

    0% is a flat neutral gray; 100% is full-contrast checks.
    """
    pixmap = QPixmap(16, 16)
    pixmap.fill(QColor("#909090"))  # flat tone, midway between the two checks
    painter = QPainter(pixmap)
    painter.setOpacity(max(0, min(100, opacity_percent)) / 100.0)
    painter.fillRect(0, 0, 8, 8, QColor("#b4b4b4"))
    painter.fillRect(8, 8, 8, 8, QColor("#b4b4b4"))
    painter.fillRect(8, 0, 8, 8, QColor("#6c6c6c"))
    painter.fillRect(0, 8, 8, 8, QColor("#6c6c6c"))
    painter.end()
    return QBrush(pixmap)


def rgba_to_qimage(rgba: np.ndarray) -> QImage:
    h, w = rgba.shape[:2]
    data = np.ascontiguousarray(rgba)
    img = QImage(data.data, w, h, w * 4, QImage.Format_RGBA8888)
    return img.copy()  # detach from the numpy buffer


def mask_to_overlay_pixmap(mask: np.ndarray, tint: tuple[int, int, int]) -> QPixmap:
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    on = mask > 0
    overlay[on, 0] = tint[0]
    overlay[on, 1] = tint[1]
    overlay[on, 2] = tint[2]
    overlay[on, 3] = OVERLAY_ALPHA
    return QPixmap.fromImage(rgba_to_qimage(overlay))


class ImageCanvas(QGraphicsView):
    """Displays the base image, tinted layer overlays, and the cut outline.

    In eyedropper mode a left click emits `pixel_picked(x, y)` instead of
    starting a pan drag.
    """

    pixel_picked = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.checker_opacity = 50
        self.setBackgroundBrush(_checker_brush(self.checker_opacity))  # shows transparency

        self.base_item: QGraphicsPixmapItem | None = None
        self.overlay_items: dict[int, QGraphicsPixmapItem] = {}
        self.outline_items: dict[int, QGraphicsPathItem] = {}
        self.eyedropper_active = False
        self._image_size: tuple[int, int] | None = None  # (w, h)

    # ---- content management -------------------------------------------------

    def set_image(self, rgba: np.ndarray, fit: bool = True) -> None:
        """Show a new base image. `fit=False` keeps the current zoom/pan."""
        self._scene.clear()
        self.overlay_items.clear()
        self.outline_items.clear()
        pixmap = QPixmap.fromImage(rgba_to_qimage(rgba))
        self.base_item = self._scene.addPixmap(pixmap)
        self.base_item.setZValue(0)
        h, w = rgba.shape[:2]
        same_size = self._image_size == (w, h)
        self._image_size = (w, h)
        self._scene.setSceneRect(0, 0, w, h)
        if fit or not same_size:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def set_checker_opacity(self, percent: int) -> None:
        """Set how strongly the transparency checkerboard is drawn (0-100)."""
        self.checker_opacity = max(0, min(100, int(percent)))
        self.setBackgroundBrush(_checker_brush(self.checker_opacity))

    def set_base_visible(self, visible: bool) -> None:
        if self.base_item is not None:
            self.base_item.setVisible(visible)

    def set_layer_overlay(
        self, layer_id: int, mask: np.ndarray, layer_type_value: str
    ) -> None:
        """Create or update the tinted overlay for one layer."""
        tint = TINTS.get(layer_type_value, (255, 0, 255))
        pixmap = mask_to_overlay_pixmap(mask, tint)
        item = self.overlay_items.get(layer_id)
        if item is None:
            item = self._scene.addPixmap(pixmap)
            item.setZValue(10)
            self.overlay_items[layer_id] = item
        else:
            item.setPixmap(pixmap)

    def set_layer_outline(self, layer_id: int, coords: list[tuple[float, float]]):
        """Create or update the smooth cut-outline path for a cut layer."""
        path = QPainterPath()
        if coords:
            path.moveTo(*coords[0])
            for pt in coords[1:]:
                path.lineTo(*pt)
            path.closeSubpath()
        item = self.outline_items.get(layer_id)
        if item is None:
            pen = QPen()
            pen.setColor("#FF2020")
            pen.setWidthF(CUT_PEN_WIDTH)
            pen.setCosmetic(True)
            item = self._scene.addPath(path, pen)
            item.setZValue(20)
            self.outline_items[layer_id] = item
        else:
            item.setPath(path)

    def set_layer_visible(self, layer_id: int, visible: bool) -> None:
        if layer_id in self.overlay_items:
            self.overlay_items[layer_id].setVisible(visible)
        if layer_id in self.outline_items:
            self.outline_items[layer_id].setVisible(visible)

    def remove_layer_items(self, layer_id: int) -> None:
        item = self.overlay_items.pop(layer_id, None)
        if item is not None:
            self._scene.removeItem(item)
        item = self.outline_items.pop(layer_id, None)
        if item is not None:
            self._scene.removeItem(item)

    # ---- interaction ---------------------------------------------------------

    def set_eyedropper(self, active: bool) -> None:
        self.eyedropper_active = active
        if active:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().unsetCursor()

    def mousePressEvent(self, event):
        if (
            self.eyedropper_active
            and event.button() == Qt.LeftButton
            and self._image_size is not None
        ):
            pos = self.mapToScene(event.position().toPoint())
            x, y = int(pos.x()), int(pos.y())
            w, h = self._image_size
            if 0 <= x < w and 0 <= y < h:
                self.pixel_picked.emit(x, y)
            event.accept()
            return
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
