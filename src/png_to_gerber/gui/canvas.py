"""Zoomable/pannable image canvas with per-layer overlay items."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

from ..paint import Tool

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
    starting a pan drag. A paint tool takes the left button; middle-drag pans.
    """

    pixel_picked = Signal(int, int)
    stroke_started = Signal(int, int)
    stroke_moved = Signal(int, int)
    stroke_finished = Signal()
    line_committed = Signal(int, int, int, int)
    rectangle_committed = Signal(int, int, int, int)
    fill_at = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.checker_opacity = 50
        self.setBackgroundBrush(_checker_brush(self.checker_opacity))  # shows transparency
        self.viewport().setMouseTracking(True)

        self.base_item: QGraphicsPixmapItem | None = None
        self.base_items: dict[int, QGraphicsPixmapItem] = {}
        self.overlay_items: dict[int, QGraphicsPixmapItem] = {}
        self.outline_items: dict[int, QGraphicsPathItem] = {}
        self.eyedropper_active = False
        self.tool = Tool.PAN
        self._image_size: tuple[int, int] | None = None  # (w, h)
        self._drawing = False
        self._panning = False
        self._pan_pos = QPointF()
        self._line_start: tuple[int, int] | None = None
        self._line_end: tuple[int, int] | None = None
        self._rect_anchor: tuple[int, int] | None = None
        self._preview: QGraphicsPathItem | None = None
        self._brush_cursor: QGraphicsEllipseItem | None = None
        self._brush_diameter = 1
        self._cursor_xy: tuple[int, int] | None = None

    # ---- content management -------------------------------------------------

    def set_image(self, rgba: np.ndarray, fit: bool = True) -> None:
        """Show one base image. `fit=False` keeps the current zoom/pan."""
        self.set_bases([(0, rgba)], fit=fit)

    def set_bases(self, images: list[tuple[int, np.ndarray]], fit: bool = True) -> None:
        """Show every base image, each aligned to the top-left of the canvas.

        Later images paint above earlier ones. The scene covers the largest
        width and height. `fit=False` keeps the current zoom when the canvas
        size does not change. `base_item` is the first image, for callers that
        still track a single pixmap.
        """
        self._scene.clear()
        self.overlay_items.clear()
        self.outline_items.clear()
        self.base_items = {}
        self.base_item = None
        self._preview = None
        self._brush_cursor = None
        self._drawing = False
        self._line_start = None
        self._line_end = None
        self._rect_anchor = None
        max_w = max((int(rgba.shape[1]) for _, rgba in images), default=0)
        max_h = max((int(rgba.shape[0]) for _, rgba in images), default=0)
        for z, (image_id, rgba) in enumerate(images):
            item = self._scene.addPixmap(QPixmap.fromImage(rgba_to_qimage(rgba)))
            item.setPos(0, 0)
            item.setZValue(z)
            self.base_items[image_id] = item
            if self.base_item is None:
                self.base_item = item
        same_size = self._image_size == (max_w, max_h) and max_w > 0
        self._image_size = (max_w, max_h) if images else None
        self._scene.setSceneRect(0, 0, max(max_w, 1), max(max_h, 1))
        if images and (fit or not same_size):
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._place_brush_cursor()

    def set_checker_opacity(self, percent: int) -> None:
        """Set how strongly the transparency checkerboard is drawn (0-100)."""
        self.checker_opacity = max(0, min(100, int(percent)))
        self.setBackgroundBrush(_checker_brush(self.checker_opacity))

    def set_base_pixels(self, image_id: int, rgba: np.ndarray) -> None:
        """Replace one base image's pixmap without rebuilding the scene."""
        item = self.base_items.get(image_id)
        if item is not None:
            item.setPixmap(QPixmap.fromImage(rgba_to_qimage(rgba)))

    def set_base_visible(self, visible: bool, image_id: int | None = None) -> None:
        """Hide or show one base image. With no id, the first image is used."""
        if image_id is None:
            item = self.base_item
        else:
            item = self.base_items.get(image_id)
        if item is not None:
            item.setVisible(visible)

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

    def clear_layer_overlay(self, layer_id: int) -> None:
        """Remove a layer's filled tint, leaving any outline in place."""
        item = self.overlay_items.pop(layer_id, None)
        if item is not None:
            self._scene.removeItem(item)

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

    def set_stroke_preview(self, points: list[tuple[float, float]], closed: bool = False) -> None:
        """Dashed overlay for a line or rectangle that has not been committed."""
        if self._preview is None:
            pen = QPen(QColor("#ffffff"))
            pen.setCosmetic(True)
            pen.setWidthF(1.5)
            pen.setStyle(Qt.DashLine)
            self._preview = self._scene.addPath(QPainterPath(), pen)
            self._preview.setZValue(30)
        path = QPainterPath()
        if points:
            path.moveTo(points[0][0] + 0.5, points[0][1] + 0.5)
            for x, y in points[1:]:
                path.lineTo(x + 0.5, y + 0.5)
            if closed and len(points) > 2:
                path.closeSubpath()
        self._preview.setPath(path)

    def clear_stroke_preview(self) -> None:
        if self._preview is not None:
            self._preview.setPath(QPainterPath())

    # ---- interaction ---------------------------------------------------------

    def set_tool(self, tool: Tool) -> None:
        self.tool = tool
        self._drawing = False
        self._line_start = None
        self._line_end = None
        self.cancel_rectangle()
        self._apply_cursor()
        self._place_brush_cursor()

    def set_eyedropper(self, active: bool) -> None:
        self.eyedropper_active = active
        self._apply_cursor()
        self._place_brush_cursor()

    def set_brush_diameter(self, diameter: int) -> None:
        """Scene-space ring that shows where the next stamp will land."""
        self._brush_diameter = max(1, int(diameter))
        self._place_brush_cursor()

    def cancel_rectangle(self) -> None:
        self._rect_anchor = None
        self.clear_stroke_preview()

    def _shows_brush_cursor(self) -> bool:
        return not self.eyedropper_active and self.tool in (
            Tool.BRUSH,
            Tool.LINE,
            Tool.RECTANGLE,
            Tool.ERASE,
        )

    def _place_brush_cursor(self) -> None:
        """Ellipse matching the stamp bounds, centered on the pixel under the cursor."""
        if not self._shows_brush_cursor() or self._cursor_xy is None or self._image_size is None:
            if self._brush_cursor is not None:
                self._brush_cursor.setVisible(False)
            return
        x, y = self._cursor_xy
        diameter = self._brush_diameter
        x0 = x - diameter // 2
        y0 = y - diameter // 2
        if self._brush_cursor is None:
            pen = QPen(QColor("#ffffff"))
            pen.setCosmetic(True)
            pen.setWidthF(1.25)
            self._brush_cursor = self._scene.addEllipse(x0, y0, diameter, diameter, pen)
            self._brush_cursor.setZValue(40)
            self._brush_cursor.setAcceptedMouseButtons(Qt.NoButton)
            self._brush_cursor.setBrush(QBrush(Qt.NoBrush))
            return
        self._brush_cursor.setRect(x0, y0, diameter, diameter)
        self._brush_cursor.setVisible(True)

    def _apply_cursor(self) -> None:
        drawing = self.tool != Tool.PAN or self.eyedropper_active
        if drawing:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().unsetCursor()

    def _image_xy(self, event) -> tuple[int, int] | None:
        if self._image_size is None:
            return None
        pos = self.mapToScene(event.position().toPoint())
        x, y = int(pos.x()), int(pos.y())
        width, height = self._image_size
        if 0 <= x < width and 0 <= y < height:
            return x, y
        return None

    def _preview_rectangle(self, x: int, y: int) -> None:
        if self._rect_anchor is None:
            return
        x0, y0 = self._rect_anchor
        self.set_stroke_preview([(x0, y0), (x, y0), (x, y), (x0, y)], closed=True)

    def mousePressEvent(self, event):
        if (
            self.eyedropper_active
            and event.button() == Qt.LeftButton
            and self._image_size is not None
        ):
            xy = self._image_xy(event)
            if xy is not None:
                self.pixel_picked.emit(*xy)
            event.accept()
            return
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_pos = event.position()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self.tool != Tool.PAN:
            xy = self._image_xy(event)
            if xy is None:
                event.accept()
                return
            x, y = xy
            if self.tool == Tool.FILL:
                self.fill_at.emit(x, y)
            elif self.tool == Tool.LINE:
                self._line_start = (x, y)
                self._line_end = (x, y)
                self.set_stroke_preview([(x, y), (x, y)])
            elif self.tool == Tool.RECTANGLE:
                if self._rect_anchor is None:
                    self._rect_anchor = (x, y)
                    self._preview_rectangle(x, y)
                else:
                    x0, y0 = self._rect_anchor
                    self._rect_anchor = None
                    self.clear_stroke_preview()
                    self.rectangle_committed.emit(x0, y0, x, y)
            else:
                self._drawing = True
                self.stroke_started.emit(x, y)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        xy = self._image_xy(event)
        self._cursor_xy = xy
        self._place_brush_cursor()
        if self._panning:
            delta = event.position() - self._pan_pos
            self._pan_pos = event.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - delta.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - delta.y()))
            event.accept()
            return
        if self._drawing and xy is not None:
            self.stroke_moved.emit(*xy)
            event.accept()
            return
        if self._line_start is not None and xy is not None:
            self._line_end = xy
            self.set_stroke_preview([self._line_start, xy])
            event.accept()
            return
        if self.tool == Tool.RECTANGLE and self._rect_anchor is not None and xy is not None:
            self._preview_rectangle(*xy)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() == Qt.MiddleButton:
            self._panning = False
            self._apply_cursor()
            event.accept()
            return
        if self._line_start is not None and event.button() == Qt.LeftButton:
            end = self._image_xy(event) or self._line_end or self._line_start
            start = self._line_start
            self._line_start = None
            self._line_end = None
            self.clear_stroke_preview()
            self.line_committed.emit(start[0], start[1], end[0], end[1])
            event.accept()
            return
        if self._drawing and event.button() == Qt.LeftButton:
            xy = self._image_xy(event)
            self._drawing = False
            if xy is not None:
                self.stroke_moved.emit(*xy)
            self.stroke_finished.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def viewportEvent(self, event):
        if event.type() == QEvent.Type.Leave:
            self._cursor_xy = None
            self._place_brush_cursor()
        return super().viewportEvent(event)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
