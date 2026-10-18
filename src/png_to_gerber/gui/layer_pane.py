"""Layer list pane: base image row, layers with eye toggles, negatives, reordering."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..model import Layer, LayerType
from .icons import eye_icon

LAYER_TYPE_LABELS = {
    LayerType.COPPER: "Copper",
    LayerType.MASK: "Solder mask opening",
    LayerType.SILKSCREEN: "Silkscreen",
    LayerType.CUT: "Cut (board outline)",
}

NEGATIVE_TYPES = (LayerType.COPPER, LayerType.MASK, LayerType.SILKSCREEN)

# Stored on the base-image row so it can live in the list without being a layer.
BASE_ITEM_ID = "__base_image__"
VISIBLE_ROLE = Qt.UserRole + 1
EYE_HIT_WIDTH = 28  # px from the item's left edge that toggles visibility
BASE_ITEM_TEXT = "Base image"


def _layer_tooltip(layer: Layer) -> str:
    label = LAYER_TYPE_LABELS[layer.layer_type]
    notes = []
    if layer.is_derived:
        notes.append("negative")
    if layer.inverted:
        notes.append("inverted")
    if notes:
        return f"{label} ({', '.join(notes)})"
    return label


class _LayerList(QListWidget):
    """List whose left-hand eye icon toggles visibility instead of selecting."""

    eye_clicked = Signal(object)  # QListWidgetItem

    def mousePressEvent(self, event):
        point = event.position().toPoint()
        item = self.itemAt(point)
        if item is not None and event.button() == Qt.LeftButton:
            if point.x() - self.visualItemRect(item).left() <= EYE_HIT_WIDTH:
                self.eye_clicked.emit(item)
                event.accept()
                return
        super().mousePressEvent(event)


class LayerPane(QWidget):
    add_requested = Signal(object)  # LayerType
    negative_requested = Signal(list, object)  # source layer ids, LayerType
    duplicate_requested = Signal(int)  # layer id
    remove_requested = Signal(int)  # layer id
    move_requested = Signal(int, int)  # layer id, delta
    selected = Signal(int)  # current layer id (-1 for none)
    base_selected = Signal()  # the "Base image" row was selected
    base_visibility_changed = Signal(bool)
    visibility_changed = Signal(int, bool)
    renamed = Signal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._eye_color = self.palette().color(QPalette.Text)
        self._base_visible = True

        self.list = _LayerList(self)
        self.list.setEditTriggers(QListWidget.DoubleClicked)
        self.list.setIconSize(QSize(18, 18))
        # Ctrl/Shift-click selects several layers (used by "Generate negative").
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.currentItemChanged.connect(self._on_selection)
        self.list.itemChanged.connect(self._on_item_changed)
        self.list.itemSelectionChanged.connect(self._update_buttons)
        self.list.eye_clicked.connect(self._toggle_item)

        self.add_button = QPushButton("Add layer", self)
        menu = QMenu(self.add_button)
        for layer_type, label in LAYER_TYPE_LABELS.items():
            action = menu.addAction(label)
            action.triggered.connect(
                lambda checked=False, lt=layer_type: self.add_requested.emit(lt)
            )
        self.add_button.setMenu(menu)

        self.negative_button = QPushButton("Generate negative", self)
        self.negative_button.setToolTip(
            "Select one or more layers (Ctrl/Shift-click), then create a layer "
            "covering everything they do not."
        )
        neg_menu = QMenu(self.negative_button)
        for layer_type in NEGATIVE_TYPES:
            action = neg_menu.addAction(f"As {LAYER_TYPE_LABELS[layer_type].lower()}")
            action.triggered.connect(
                lambda checked=False, lt=layer_type: self._on_negative(lt)
            )
        self.negative_button.setMenu(neg_menu)

        self.duplicate_button = QPushButton("Duplicate", self)
        self.duplicate_button.setToolTip("Create a new layer with the same settings.")
        self.duplicate_button.clicked.connect(self._on_duplicate)
        self.delete_button = QPushButton("Delete", self)
        self.delete_button.clicked.connect(self._on_delete)
        self.up_button = QPushButton("Up", self)
        self.up_button.clicked.connect(lambda: self._on_move(-1))
        self.down_button = QPushButton("Down", self)
        self.down_button.clicked.connect(lambda: self._on_move(1))

        top = QHBoxLayout()
        top.addWidget(self.add_button)
        top.addWidget(self.negative_button)
        buttons = QHBoxLayout()
        buttons.addWidget(self.duplicate_button)
        buttons.addWidget(self.delete_button)
        buttons.addWidget(self.up_button)
        buttons.addWidget(self.down_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addLayout(buttons)
        layout.addWidget(self.list)
        self.refresh([])

    # ---- sync from model ------------------------------------------------------

    def refresh(self, layers: list[Layer], selected_id: int | None = None) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        base = self._make_base_item()
        self.list.addItem(base)
        chosen = base
        for layer in layers:
            item = QListWidgetItem(layer.name)
            item.setData(Qt.UserRole, layer.id)
            item.setFlags((item.flags() | Qt.ItemIsEditable) & ~Qt.ItemIsUserCheckable)
            item.setToolTip(_layer_tooltip(layer))
            self._show_visibility(item, layer.visible)
            self.list.addItem(item)
            if layer.id == selected_id:
                chosen = item
        self.list.setCurrentItem(chosen)
        self.list.blockSignals(False)
        self._update_buttons()
        self._emit_current()

    def current_layer_id(self) -> int:
        item = self.list.currentItem()
        if item is None or self._is_base(item):
            return -1
        return item.data(Qt.UserRole)

    def selected_ids(self) -> list[int]:
        """Ids of all selected layers, in stack order. The base image is skipped."""
        rows = sorted(self.list.row(item) for item in self.list.selectedItems())
        ids = []
        for row in rows:
            item = self.list.item(row)
            if not self._is_base(item):
                ids.append(item.data(Qt.UserRole))
        return ids

    def layer_item(self, layer_id: int) -> QListWidgetItem | None:
        return self._item_for(layer_id)

    def select_layer(self, layer_id: int) -> None:
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.data(Qt.UserRole) == layer_id:
                self.list.setCurrentItem(item)
                return

    def select_base(self) -> None:
        """Select the base image row."""
        item = self._base_item()
        if item is None:
            self.base_selected.emit()
            return
        if self.list.currentItem() is item:
            self._update_buttons()
            self.base_selected.emit()
            return
        self.list.setCurrentItem(item)

    def base_is_selected(self) -> bool:
        item = self.list.currentItem()
        return item is not None and self._is_base(item)

    # ---- visibility (eye) ------------------------------------------------------

    def is_base_visible(self) -> bool:
        return self._base_visible

    def set_base_visible(self, visible: bool) -> None:
        """Reflect base-image visibility without emitting a signal."""
        self._base_visible = bool(visible)
        item = self._base_item()
        if item is not None and bool(item.data(VISIBLE_ROLE)) != self._base_visible:
            self.list.blockSignals(True)
            self._show_visibility(item, self._base_visible)
            self.list.blockSignals(False)

    def toggle_base_visibility(self) -> None:
        item = self._base_item()
        if item is not None:
            self._toggle_item(item)

    def is_layer_visible(self, layer_id: int) -> bool:
        item = self._item_for(layer_id)
        return bool(item.data(VISIBLE_ROLE)) if item is not None else False

    def show_layer_visibility(self, layer_id: int, visible: bool) -> None:
        """Update a layer's eye to match the model, without emitting signals."""
        item = self._item_for(layer_id)
        if item is not None and bool(item.data(VISIBLE_ROLE)) != visible:
            self.list.blockSignals(True)
            self._show_visibility(item, visible)
            self.list.blockSignals(False)

    def toggle_layer_visibility(self, layer_id: int) -> None:
        item = self._item_for(layer_id)
        if item is not None:
            self._toggle_item(item)

    def note_layer_changed(self, layer: Layer) -> None:
        """Refresh the row tooltip after an in-place edit such as Invert."""
        item = self._item_for(layer.id)
        if item is not None:
            item.setToolTip(_layer_tooltip(layer))

    def _item_for(self, layer_id: int) -> QListWidgetItem | None:
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.data(Qt.UserRole) == layer_id:
                return item
        return None

    def _base_item(self) -> QListWidgetItem | None:
        if self.list.count() == 0:
            return None
        item = self.list.item(0)
        return item if self._is_base(item) else None

    def _is_base(self, item: QListWidgetItem) -> bool:
        return item.data(Qt.UserRole) == BASE_ITEM_ID

    def _make_base_item(self) -> QListWidgetItem:
        item = QListWidgetItem(BASE_ITEM_TEXT)
        item.setData(Qt.UserRole, BASE_ITEM_ID)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        item.setToolTip("Source image. Click to open it or remove the background.")
        self._show_visibility(item, self._base_visible)
        return item

    def _show_visibility(self, item: QListWidgetItem, visible: bool) -> None:
        item.setData(VISIBLE_ROLE, visible)
        item.setIcon(eye_icon(visible, self._eye_color))
        dim = QColor(self._eye_color)
        dim.setAlphaF(1.0 if visible else 0.5)
        item.setForeground(dim)

    def _toggle_item(self, item: QListWidgetItem) -> None:
        visible = not bool(item.data(VISIBLE_ROLE))
        self.list.blockSignals(True)
        self._show_visibility(item, visible)
        self.list.blockSignals(False)
        if self._is_base(item):
            self._base_visible = visible
            self.base_visibility_changed.emit(visible)
        else:
            self.visibility_changed.emit(item.data(Qt.UserRole), visible)

    # ---- internal -------------------------------------------------------------

    def _update_buttons(self):
        layer_id = self.current_layer_id()
        row = self.list.currentRow()
        is_layer = layer_id != -1
        self.negative_button.setEnabled(bool(self.selected_ids()))
        self.duplicate_button.setEnabled(is_layer)
        self.delete_button.setEnabled(is_layer)
        # Row 0 is the base image, which stays put. The first real layer is row 1.
        self.up_button.setEnabled(is_layer and row > 1)
        self.down_button.setEnabled(is_layer and 0 < row < self.list.count() - 1)

    def _emit_current(self):
        item = self.list.currentItem()
        if item is None or self._is_base(item):
            self.base_selected.emit()
        else:
            self.selected.emit(item.data(Qt.UserRole))

    def _on_selection(self, current, _previous):
        self._emit_current()

    def _on_item_changed(self, item: QListWidgetItem):
        if self._is_base(item):
            if item.text() != BASE_ITEM_TEXT:
                self.list.blockSignals(True)
                item.setText(BASE_ITEM_TEXT)
                self.list.blockSignals(False)
            return
        self.renamed.emit(item.data(Qt.UserRole), item.text())

    def _on_negative(self, layer_type: LayerType):
        ids = self.selected_ids()
        if ids:
            self.negative_requested.emit(ids, layer_type)

    def _on_duplicate(self):
        layer_id = self.current_layer_id()
        if layer_id != -1:
            self.duplicate_requested.emit(layer_id)

    def _on_delete(self):
        layer_id = self.current_layer_id()
        if layer_id != -1:
            self.remove_requested.emit(layer_id)

    def _on_move(self, delta: int):
        layer_id = self.current_layer_id()
        if layer_id != -1:
            self.move_requested.emit(layer_id, delta)
