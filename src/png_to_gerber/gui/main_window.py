"""Main window: wires the canvas, layer pane, inspector, project files, and export."""

from __future__ import annotations

import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import UnidentifiedImageError
from shapely.ops import unary_union
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QAction, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSlider,
    QSplitter,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .. import project_io
from ..background import crop_to_opaque, remove_background
from ..flatten import auto_cutoffs, even_cutoffs, histogram, palette_for
from ..gerber import BoardTransform, ExportError, export_project
from ..geometry import CutError, cut_outline, mask_to_geometry
from ..image_export import (
    RENDER_NAME,
    cut_perimeter_rgba,
    layer_png_name,
    layer_rgba,
    render_pcb,
    save_rgba,
)
from ..history import EditHistory, capture_state, clone_project
from ..image_io import canvas_size as shared_canvas_size
from ..image_io import dominant_colors, load_rgba, pad_top_left
from ..masking import build_mask, invert_mask, negative_mask
from ..paint import (
    ERASE,
    PAINT,
    Tool,
    apply_edits,
    coalesce_paint,
    compose_painted,
    edits_from_runs,
    stroke_rectangle,
    flood_fill_rgba,
    flood_region,
    iter_disk,
    line_points,
    paint_origin,
    runs_from_edits,
    stamp,
    stroke as raster_stroke,
)
from ..model import (
    FLATTEN_COUNT_MAX,
    FLATTEN_COUNT_MIN,
    FLATTEN_SMOOTH_MAX,
    Layer,
    LayerType,
    Project,
    ProjectError,
    Side,
    default_layer_name,
    duplicate_layer_name,
    reset_flatten,
)
from ..project_io import EXTENSION, ProjectFileError
from .base_panel import BaseImagePanel
from .canvas import ImageCanvas
from .export_dialog import ExportDialog
from .image_export_dialog import ImageExportDialog
from .inspector import LayerInspector
from .layer_pane import NEGATIVE_TYPES, LayerPane
from .paint_toolbar import PaintToolbar
from .theme import apply_dark_theme

RECOMPUTE_DEBOUNCE_MS = 120
APP_TITLE = "PNG to Gerber - PCB Art"
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp);;All files (*)"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        apply_dark_theme(QApplication.instance())
        self.resize(1200, 800)

        self.project = Project()
        self.project_path: Path | None = None
        self._originals: dict[int, np.ndarray] = {}  # image id -> file pixels
        self._working: dict[int, np.ndarray] = {}  # image id -> pixels layers mask
        self._palettes: dict[int, list] = {}  # image id -> dominant colors of its working image
        self._preview_cache: dict[int, tuple] = {}  # image id -> flatten histogram inputs
        self._comparing: set[int] = set()  # image ids showing the pre-flatten canvas
        self.masks: dict[int, np.ndarray] = {}
        self.outlines: dict[int, object] = {}  # layer id -> shapely Polygon
        self._dirty_layers: set[int] = set()
        self._modified = False
        self._stroke: dict | None = None
        self._history = EditHistory()
        self._edit_depth = 0
        self._edit_pending = False
        self._open_gesture: str | None = None
        self._restoring = False
        self._history_locked = False
        self._gesture_names: dict[QWidget, str] = {}

        self._recompute_timer = QTimer(self)
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.setInterval(RECOMPUTE_DEBOUNCE_MS)
        self._recompute_timer.timeout.connect(self._recompute_dirty)

        # Widgets
        self.canvas = ImageCanvas(self)
        self.canvas.pixel_picked.connect(self._on_pixel_picked)
        self.canvas.stroke_started.connect(self.begin_paint_stroke)
        self.canvas.stroke_moved.connect(self.continue_paint_stroke)
        self.canvas.stroke_finished.connect(self.end_paint_stroke)
        self.canvas.line_committed.connect(self.paint_line)
        self.canvas.rectangle_committed.connect(self.paint_rectangle)
        self.canvas.fill_at.connect(self.paint_fill)
        self.setCentralWidget(self.canvas)

        self.paint_toolbar = PaintToolbar(self)
        self.paint_toolbar.tool_changed.connect(self._on_paint_tool)
        self.paint_toolbar.brush_size_changed.connect(self.canvas.set_brush_diameter)
        self.canvas.set_brush_diameter(self.paint_toolbar.brush_size)
        self.paint_toolbar_host = QToolBar("Paint", self)
        self.paint_toolbar_host.setObjectName("paint_toolbar_host")
        self.paint_toolbar_host.setMovable(False)
        self.paint_toolbar_host.setOrientation(Qt.Vertical)
        self.paint_toolbar_host.setStyleSheet("QToolBar { padding: 0px; spacing: 0px; }")
        self.paint_toolbar_host.addWidget(self.paint_toolbar)
        self.addToolBar(Qt.LeftToolBarArea, self.paint_toolbar_host)

        self.layer_pane = LayerPane(self)
        self.layer_pane.add_requested.connect(self.add_layer)
        self.layer_pane.negative_requested.connect(self.generate_negative)
        self.layer_pane.duplicate_requested.connect(self.duplicate_layer)
        self.layer_pane.remove_requested.connect(self.remove_layer)
        self.layer_pane.move_requested.connect(self.move_layer)
        self.layer_pane.selected.connect(self._on_layer_selected)
        self.layer_pane.base_selected.connect(self._on_base_selected)
        self.layer_pane.visibility_changed.connect(self.set_layer_visible)
        self.layer_pane.renamed.connect(self._on_layer_renamed)

        self.inspector = LayerInspector(self)
        self.inspector.changed.connect(self._on_layer_edited)
        self.inspector.type_changed.connect(self.change_layer_type)
        self.inspector.eyedropper_toggled.connect(self.canvas.set_eyedropper)

        self.base_panel = BaseImagePanel(self)
        self.bg_tolerance_spin = self.base_panel.tolerance_spin

        # The lower half shows either the selected layer's options or the base image's.
        self.inspector_stack = QStackedWidget(self)
        self.inspector_stack.addWidget(self.inspector)
        self.inspector_stack.addWidget(self.base_panel)

        side = QWidget(self)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Vertical, side)
        splitter.addWidget(self.layer_pane)
        splitter.addWidget(self.inspector_stack)
        side_layout.addWidget(splitter)
        dock = QDockWidget("Layers", self)
        dock.setWidget(side)
        dock.setFeatures(QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self._build_actions()
        self._bind_gestures()
        self._connect_base_controls()
        self.layer_pane.select_base()
        self._history.reset(self._capture())
        self._update_undo_actions()
        self._update_title()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self.statusBar().showMessage("Open an image or a project to begin.")

    # ---- UI construction ------------------------------------------------------------

    def _build_actions(self):
        self.undo_action = QAction("&Undo", self)
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.triggered.connect(self.undo)
        self.redo_action = QAction("&Redo", self)
        self.redo_action.setShortcut("Ctrl+Shift+Z")
        self.redo_action.triggered.connect(self.redo)
        self.new_project_action = QAction("&New project", self)
        self.new_project_action.setShortcut("Ctrl+N")
        self.new_project_action.triggered.connect(self._on_new_project)
        self.open_project_action = QAction("Open &project...", self)
        self.open_project_action.setShortcut("Ctrl+Shift+O")
        self.open_project_action.triggered.connect(self._on_open_project)
        self.save_action = QAction("&Save project", self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.triggered.connect(self._on_save)
        self.save_as_action = QAction("Save project &as...", self)
        self.save_as_action.setShortcut("Ctrl+Shift+S")
        self.save_as_action.triggered.connect(self._on_save_as)
        self.add_image_action = QAction("&Add image...", self)
        self.add_image_action.setShortcut("Ctrl+I")
        self.add_image_action.triggered.connect(self._on_add_image)
        self.open_action = QAction("&Open image...", self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self._on_open_image)
        self.board_action = QAction("&Board settings...", self)
        self.board_action.triggered.connect(self._on_board_settings)
        self.export_files_action = QAction("&Gerber files...", self)
        self.export_files_action.setShortcut("Ctrl+E")
        self.export_files_action.triggered.connect(self._on_export_files)
        self.export_zip_action = QAction("Gerber &ZIP...", self)
        self.export_zip_action.setShortcut("Ctrl+Shift+E")
        self.export_zip_action.triggered.connect(self._on_export_zip)
        self.export_image_action = QAction("Export &image...", self)
        self.export_image_action.triggered.connect(self._on_export_image)

        self.base_visible_action = QAction("Show &base image", self)
        self.base_visible_action.setCheckable(True)
        self.base_visible_action.setChecked(True)
        self.base_visible_action.toggled.connect(self._on_base_visible_toggled)

        self.remove_bg_action = QAction("Remove &background", self)
        self.remove_bg_action.setCheckable(True)
        self.remove_bg_action.setToolTip(
            "Make the image's outer contiguous background color transparent "
            "(toggle off to restore it)."
        )
        self.remove_bg_action.toggled.connect(self.set_background_removed)

        self.checker_slider = QSlider(Qt.Horizontal, self)
        self.checker_slider.setRange(0, 100)
        self.checker_slider.setValue(self.project.checker_opacity)
        self.checker_slider.setFixedWidth(110)
        self.checker_slider.setToolTip("Transparency checkerboard opacity")
        self.checker_slider.valueChanged.connect(self._on_checker_opacity)

        file_menu = self.menuBar().addMenu("&File")
        for action in (
            self.undo_action,
            self.redo_action,
            None,
            self.new_project_action,
            self.open_project_action,
            self.save_action,
            self.save_as_action,
            None,
            self.add_image_action,
            self.open_action,
            self.board_action,
            None,
        ):
            file_menu.addSeparator() if action is None else file_menu.addAction(action)
        self.export_menu = QMenu("E&xport", self)
        self.export_menu.addAction(self.export_files_action)
        self.export_menu.addAction(self.export_zip_action)
        self.export_menu.addAction(self.export_image_action)
        file_menu.addMenu(self.export_menu)
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.base_visible_action)

        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("main_toolbar")
        toolbar.setMovable(False)
        self.main_toolbar = toolbar
        self.checker_label = QLabel("Checker opacity", self)
        toolbar.addWidget(self.checker_label)
        toolbar.addWidget(self.checker_slider)

    def _bind_gestures(self):
        """A drag or a held spin arrow is one undo step; a click is its own."""
        self._gesture_names = {
            self.inspector.tolerance_slider: "tolerance",
            self.inspector.smooth_slider: "smoothness",
            self.inspector.close_spin: "close-radius",
            self.inspector.island_spin: "min-island",
            self.base_panel.tolerance_spin: "bg-tolerance",
            self.base_panel.count_spin: "flatten-count",
            self.base_panel.smooth_slider: "flatten-smooth",
            self.base_panel.cutoff_bar: "flatten-cutoffs",
            self.checker_slider: "checker",
        }
        for widget, name in self._gesture_names.items():
            if isinstance(widget, QSlider):
                widget.sliderPressed.connect(lambda n=name: self._begin_gesture(n))
                widget.sliderReleased.connect(self._finish_gesture)
        self.base_panel.cutoff_bar.drag_started.connect(
            lambda: self._begin_gesture("flatten-cutoffs")
        )
        self.base_panel.cutoff_bar.drag_finished.connect(self._finish_gesture)

    def _connect_base_controls(self):
        """Keep the base-image row/panel and the menu actions in step."""
        self.layer_pane.base_visibility_changed.connect(self._on_base_eye)
        self.base_panel.add_button.clicked.connect(self._on_add_image)
        self.base_panel.open_button.clicked.connect(lambda: self._on_open_image())
        self.base_panel.remove_button.clicked.connect(self._on_remove_image)
        self.base_panel.remove_bg_check.toggled.connect(self.remove_bg_action.setChecked)
        self.base_panel.tolerance_spin.valueChanged.connect(self._on_bg_tolerance)
        self.base_panel.tolerance_spin.setValue(self.project.background_tolerance)
        self.base_panel.flatten_check.toggled.connect(self._on_flatten_toggled)
        self.base_panel.count_spin.valueChanged.connect(self._on_flatten_count)
        self.base_panel.cutoff_bar.cutoffs_changed.connect(self._on_flatten_cutoffs)
        self.base_panel.auto_button.clicked.connect(self._on_auto_cutoffs)
        self.base_panel.reset_button.clicked.connect(self._on_reset_cutoffs)
        self.base_panel.smooth_slider.valueChanged.connect(self._on_flatten_smooth)
        self.base_panel.compare_check.toggled.connect(self._on_compare_original)
        self.base_panel.color_edited.connect(self._on_flatten_color)

    def _sync_base_controls(self):
        image = self.active_image()
        self.base_panel.remove_button.setEnabled(image is not None)
        if image is None:
            self.base_panel.set_image_info(None)
            self.base_panel.sync(False, self.project.background_tolerance)
            return
        working = self._working.get(image.id)
        name = Path(image.path).name if image.path else image.name
        if working is not None:
            self.base_panel.set_image_info(name, int(working.shape[1]), int(working.shape[0]))
        else:
            self.base_panel.set_image_info(name)
        cutoffs, colors, hist = self._flatten_preview(image)
        self.base_panel.sync(
            image.background_removed,
            image.background_tolerance,
            flatten_enabled=image.flatten_enabled,
            flatten_count=image.flatten_count,
            cutoffs=cutoffs,
            colors=colors,
            smooth=image.flatten_smooth,
            histogram=hist,
            compare=image.id in self._comparing and image.flatten_enabled,
        )
        self._set_checked(self.base_visible_action, image.visible)
        self._set_checked(self.remove_bg_action, image.background_removed)
        if working is not None:
            self.inspector.set_dominant_colors(self._colors_for(image.id))

    @staticmethod
    def _set_checked(action: QAction, checked: bool) -> None:
        action.blockSignals(True)
        action.setChecked(checked)
        action.blockSignals(False)

    # ---- modified tracking and undo ------------------------------------------------------

    @property
    def is_modified(self) -> bool:
        return self._modified

    def _mark_modified(self):
        if self._restoring or self._history_locked:
            return
        self._edit_pending = True
        if self._edit_depth == 0:
            self._flush_edit()

    def _set_unmodified(self):
        self._modified = False
        self._update_title()

    def _sync_modified(self):
        modified = not self._history.at_saved
        if modified != self._modified:
            self._modified = modified
            self._update_title()

    @contextmanager
    def _edit(self):
        """Group every mutation inside one user action into a single history step."""
        self._edit_depth += 1
        flag = {"mode": "edit"}
        try:
            yield flag
        finally:
            self._edit_depth -= 1
            if self._edit_depth == 0:
                self._flush_edit(mode=flag["mode"])

    def _flush_edit(self, mode: str = "edit"):
        if self._restoring or self._history_locked:
            self._edit_pending = False
            return
        if mode == "baseline":
            self._open_gesture = None
            self._history.reset(self._capture())
            self._edit_pending = False
            self._sync_modified()
            self._update_undo_actions()
            return
        if not self._edit_pending and mode != "checkpoint":
            return
        self._edit_pending = False
        self._history.commit(
            self._capture(),
            gesture=self._open_gesture,
            saved=mode == "checkpoint",
        )
        self._sync_modified()
        self._update_undo_actions()

    def _capture(self):
        layer_id, image_id = self._selection()
        return capture_state(self.project, self._originals, layer_id, image_id)

    def _selection(self) -> tuple[int | None, int | None]:
        layer_id = self.layer_pane.current_layer_id()
        image_id = self.layer_pane.current_image_id()
        if layer_id is None or layer_id < 0:
            layer_id = None
        return layer_id, image_id

    def _note_selection(self):
        if self._restoring or self._history_locked or self._edit_depth or not self._history.states:
            return
        layer_id, image_id = self._selection()
        self._history.set_selection(layer_id, image_id)

    def _begin_gesture(self, name: str) -> None:
        self._open_gesture = name

    def _finish_gesture(self) -> None:
        if self._edit_depth:
            QTimer.singleShot(0, self, self._finish_gesture)
            return
        self._open_gesture = None
        self._history.end_gesture()
        self._sync_modified()
        self._update_undo_actions()

    def _schedule_gesture_finish(self) -> None:
        QTimer.singleShot(0, self, self._finish_gesture)

    def _update_undo_actions(self) -> None:
        self.undo_action.setEnabled(self._history.can_undo)
        self.redo_action.setEnabled(self._history.can_redo)

    def undo(self) -> None:
        """Return to the previous project state. Ctrl+Z and File > Undo."""
        self._open_gesture = None
        state = self._history.undo()
        if state is None:
            self._sync_modified()
            self._update_undo_actions()
            return
        self._restore(state)
        self.statusBar().showMessage("Undone.")

    def redo(self) -> None:
        """Reapply the state undone last. Ctrl+Shift+Z and File > Redo."""
        self._open_gesture = None
        state = self._history.redo()
        if state is None:
            self._sync_modified()
            self._update_undo_actions()
            return
        self._restore(state)
        self.statusBar().showMessage("Redone.")

    def _image_pixels_key(self, image) -> tuple:
        """Inputs of the composed working image. Equal keys mean the same pixels."""
        original = self._originals.get(image.id)
        if image.flatten_enabled:
            flatten = self._flatten_signature(image)
        else:
            flatten = (False,)
        return (
            None if original is None else id(original),
            bool(image.background_removed),
            int(image.background_tolerance) if image.background_removed else 0,
            flatten,
            tuple(tuple(int(value) for value in entry) for entry in image.paint),
        )

    def _layer_mask_key(self, layer: Layer) -> tuple:
        """Inputs of one layer mask. Equal keys mean the cached mask is still valid."""
        parent = self.project.image_of(layer.id)
        working = None if parent is None else self._working.get(parent.id)
        source_working = []
        for source_id in layer.negative_of:
            owner = self.project.image_of(source_id)
            if owner is not None and owner.id in self._working:
                source_working.append(id(self._working[owner.id]))
        return (
            None if parent is None else parent.id,
            None if working is None else id(working),
            self._canvas_shape(),
            layer.layer_type.value,
            bool(layer.has_color),
            tuple(int(channel) for channel in layer.color),
            int(layer.tolerance),
            int(layer.close_radius),
            int(layer.min_island),
            bool(layer.fill_holes),
            bool(layer.inverted),
            int(layer.smoothness),
            layer.outline_mode.value,
            tuple(int(source_id) for source_id in layer.negative_of),
            tuple(source_working),
            tuple(tuple(int(value) for value in run) for run in layer.manual),
        )

    def _restore(self, state) -> None:
        self._restoring = True
        try:
            # A brush stroke changes one layer's runs. Rebuilding every base
            # image and every mask made undo much slower than drawing the stroke.
            self._recompute_timer.stop()
            self._dirty_layers.clear()
            self._stroke = None
            self.canvas.cancel_rectangle()
            previous_pixels = {
                image.id: self._image_pixels_key(image) for image in self.project.images
            }
            previous_masks = {
                layer.id: self._layer_mask_key(layer) for layer in self.project.layers
            }
            previous_visible = {layer.id: layer.visible for layer in self.project.layers}
            previous_ids = [image.id for image in self.project.images]
            previous_canvas = self._canvas_shape()
            kept_working = dict(self._working)

            self.project = clone_project(state.project)
            self._originals = dict(state.originals)
            alive_images = {image.id for image in self.project.images}
            self._comparing &= alive_images

            new_pixels: dict[int, tuple] = {}
            rebuilt: dict[int, np.ndarray] = {}
            for image in self.project.images:
                if image.id not in self._originals:
                    continue
                key = self._image_pixels_key(image)
                new_pixels[image.id] = key
                cached = kept_working.get(image.id)
                if cached is not None and key == previous_pixels.get(image.id):
                    rebuilt[image.id] = cached
                else:
                    rebuilt[image.id] = self._compose_working(image)
                    self._palettes.pop(image.id, None)
            self._working = rebuilt
            for image_id in list(self._palettes):
                if image_id not in self._working:
                    self._palettes.pop(image_id, None)
            for image_id in list(self._preview_cache):
                if image_id not in self._working:
                    self._preview_cache.pop(image_id, None)

            alive_layers = {layer.id for layer in self.project.layers}
            for layer_id in list(self.masks):
                if layer_id not in alive_layers:
                    self.masks.pop(layer_id, None)
                    self.outlines.pop(layer_id, None)
                    self.canvas.remove_layer_items(layer_id)
            for layer_id in list(self.outlines):
                if layer_id not in alive_layers:
                    self.outlines.pop(layer_id, None)
                    self.canvas.remove_layer_items(layer_id)

            dirty: set[int] = set()
            for layer in self.project.layers:
                changed = self._layer_mask_key(layer) != previous_masks.get(layer.id)
                missing = layer.has_artwork and layer.id not in self.masks
                if changed or missing:
                    dirty.add(layer.id)
            canvas_match = (
                [image.id for image in self.project.images] == previous_ids
                and self._canvas_shape() == previous_canvas
            )
            self.checker_slider.blockSignals(True)
            self.checker_slider.setValue(self.project.checker_opacity)
            self.checker_slider.blockSignals(False)
            self.canvas.set_checker_opacity(self.project.checker_opacity)
            if canvas_match:
                for image in self.project.images:
                    if image.id not in self._working:
                        continue
                    if new_pixels.get(image.id) != previous_pixels.get(image.id):
                        self.canvas.set_base_pixels(image.id, self._pixels_for_canvas(image))
                    self.canvas.set_base_visible(self._base_shown(image), image.id)
                for layer in self.project.layers:
                    if layer.id in dirty:
                        continue
                    if previous_visible.get(layer.id) != layer.visible:
                        self.canvas.set_layer_visible(layer.id, layer.visible)
            else:
                # The scene is cleared, so every surviving mask has to be drawn again.
                self._refresh_canvas(fit=False, mark_dirty=False, sync_base=False)
                for layer in self.project.layers:
                    if layer.has_artwork or layer.id in self.masks:
                        dirty.add(layer.id)
            for layer_id in list(dirty):
                dirty |= self.project.dependents(layer_id)
            for layer_id in self.project.dependency_order(dirty):
                layer = self.project.get_layer(layer_id)
                if layer is not None:
                    self.recompute_layer(layer)
            self._sync_base_controls()
            if state.selected_layer is not None and self.project.get_layer(state.selected_layer):
                self._refresh_layers(selected_id=state.selected_layer)
            elif state.selected_image is not None and self.project.get_image(state.selected_image):
                self._refresh_layers(selected_image_id=state.selected_image)
            else:
                self._refresh_layers()
                if self.project.images:
                    self.layer_pane.select_base(self.project.images[0].id)
            self._sync_modified()
            self._update_undo_actions()
        finally:
            self._restoring = False

    def eventFilter(self, watched, event) -> bool:
        """Coalesce a held arrow or a pointer drag on a continuous control."""
        event_type = event.type()
        if event_type == QEvent.Type.MouseButtonPress and isinstance(event, QMouseEvent):
            if event.button() == Qt.MouseButton.LeftButton:
                name = self._gesture_name_for(watched)
                if name:
                    self._begin_gesture(name)
        elif event_type == QEvent.Type.MouseButtonRelease and isinstance(event, QMouseEvent):
            if event.button() == Qt.MouseButton.LeftButton and self._open_gesture:
                self._schedule_gesture_finish()
        elif event_type == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            if (
                not event.isAutoRepeat()
                and event.key()
                in (
                    Qt.Key.Key_Up,
                    Qt.Key.Key_Down,
                    Qt.Key.Key_Left,
                    Qt.Key.Key_Right,
                    Qt.Key.Key_PageUp,
                    Qt.Key.Key_PageDown,
                )
                and self._gesture_name_for(watched)
            ):
                self._begin_gesture(self._gesture_name_for(watched))
        elif event_type == QEvent.Type.KeyRelease and isinstance(event, QKeyEvent):
            if not event.isAutoRepeat() and self._open_gesture:
                self._schedule_gesture_finish()
        return False

    def _gesture_name_for(self, obj) -> str | None:
        widget = obj if isinstance(obj, QWidget) else None
        while widget is not None:
            name = self._gesture_names.get(widget)
            if name:
                return name
            widget = widget.parentWidget()
        return None

    def _update_title(self):
        name = self.project_path.name if self.project_path else "untitled"
        star = "*" if self._modified else ""
        self.setWindowTitle(f"{star}{name} - {APP_TITLE}")

    def _confirm_discard_changes(self, doing: str) -> bool:
        """Ask what to do with unsaved changes; False means the user cancelled."""
        if not self._modified:
            return True
        answer = QMessageBox.warning(
            self,
            "Unsaved changes",
            f"The project has unsaved changes.\n\nSave them before {doing}?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if answer == QMessageBox.Save:
            return self._save_interactive()
        return answer == QMessageBox.Discard

    def closeEvent(self, event):
        if self._confirm_discard_changes("closing"):
            event.accept()
        else:
            event.ignore()

    # ---- images -----------------------------------------------------------------

    def active_image(self):
        """The base image that owns the current row, or the first image."""
        image_id = self.layer_pane.current_image_id()
        if image_id is not None:
            found = self.project.get_image(image_id)
            if found is not None:
                return found
        return self.project.images[0] if self.project.images else None

    @property
    def image(self):
        """Working pixels of the active image (what its color layers mask)."""
        current = self.active_image()
        if current is None:
            return None
        return self._working.get(current.id)

    @property
    def original_image(self):
        """File pixels of the active image, before background removal."""
        current = self.active_image()
        if current is None:
            return None
        return self._originals.get(current.id)

    @property
    def image_path(self) -> Path | None:
        current = self.active_image()
        if current is None or not current.path:
            return None
        return Path(current.path)

    def _canvas_shape(self) -> tuple[int, int]:
        arrays = [
            self._working[image.id]
            for image in self.project.images
            if image.id in self._working
        ]
        return shared_canvas_size(arrays)

    def load_image(self, path: str | Path) -> None:
        """Replace the active image (or open the first one). Layers stay put."""
        original = load_rgba(path)
        path = Path(path)
        had_pixels = bool(self._originals)
        fresh = not self._modified and not self._history.can_undo
        with self._edit() as edit:
            image = self.active_image()
            if image is None:
                image = self.project.add_image(name=path.name, path=str(path))
            else:
                image.name = path.name
                image.path = str(path)
                image.background_removed = False
                image.paint = []
                for layer in image.layers:
                    layer.manual = []
                reset_flatten(image)
                self._comparing.discard(image.id)
            self._originals[image.id] = original
            self._working[image.id] = original
            self._palettes.pop(image.id, None)
            self._set_checked(self.remove_bg_action, False)
            self._refresh_canvas(fit=True)
            self._refresh_layers(selected_image_id=image.id)
            # A bare image with no work in it isn't worth saving. The first one
            # becomes the baseline; replacing that clean image stays undoable.
            clean = not self.project.layers and self.project_path is None and not self._modified
            if clean and not had_pixels and fresh:
                edit["mode"] = "baseline"
            elif clean:
                edit["mode"] = "checkpoint"
                self._edit_pending = True
            else:
                self._mark_modified()
            self.statusBar().showMessage(
                f"Loaded {path.name} ({original.shape[1]} x {original.shape[0]} px)"
            )

    def add_image(self, path: str | Path):
        """Open another base image. Its layers will be nested under it."""
        original = load_rgba(path)
        path = Path(path)
        had_pixels = bool(self._originals)
        fresh = not self._modified and not self._history.can_undo
        with self._edit() as edit:
            image = self.project.add_image(name=path.name, path=str(path))
            self._originals[image.id] = original
            self._working[image.id] = original
            self._refresh_canvas(fit=True)
            self._refresh_layers(selected_image_id=image.id)
            clean = (
                len(self.project.images) <= 1
                and not self.project.layers
                and self.project_path is None
                and not self._modified
            )
            if clean and not had_pixels and fresh:
                edit["mode"] = "baseline"
            elif len(self.project.images) > 1 or self.project.layers or self.project_path is not None or self._modified:
                self._mark_modified()
            else:
                edit["mode"] = "checkpoint"
                self._edit_pending = True
            self.statusBar().showMessage(
                f"Added {path.name} ({original.shape[1]} x {original.shape[0]} px)"
            )
        return image

    def remove_image(self, image_id: int) -> None:
        """Remove one base image and the layers nested under it."""
        with self._edit():
            existed = self.project.get_image(image_id) is not None
            removed = self.project.remove_image(image_id)
            self._originals.pop(image_id, None)
            self._working.pop(image_id, None)
            self._palettes.pop(image_id, None)
            self._comparing.discard(image_id)
            for layer_id in removed:
                self.masks.pop(layer_id, None)
                self.outlines.pop(layer_id, None)
                self._dirty_layers.discard(layer_id)
                self.canvas.remove_layer_items(layer_id)
            self._refresh_canvas(fit=True)
            self._refresh_layers()
            if existed:
                self._mark_modified()

    def _refresh_canvas(self, fit: bool, *, mark_dirty: bool = True, sync_base: bool = True) -> None:
        """Redraw every base image on the shared top-left canvas."""
        pairs = [
            (image.id, self._pixels_for_canvas(image))
            for image in self.project.images
            if image.id in self._working
        ]
        self.canvas.set_bases(pairs, fit=fit)
        for image in self.project.images:
            self.canvas.set_base_visible(self._base_shown(image), image.id)
        self._sync_locked_board_height()
        if sync_base:
            self._sync_base_controls()
        if mark_dirty:
            self._mark_dirty(*[layer.id for layer in self.project.layers])

    def _refresh_layers(self, selected_id: int | None = None, selected_image_id: int | None = None) -> None:
        self.layer_pane.refresh(
            images=self.project.images,
            selected_id=selected_id,
            selected_image_id=selected_image_id,
        )

    def _store_working(self, image_id: int, pixels: np.ndarray, fit: bool) -> None:
        self._working[image_id] = pixels
        self._palettes.pop(image_id, None)
        self._refresh_canvas(fit=fit)

    def _prepared_image(self, image):
        """Pixels after background removal and before color flattening."""
        original = self._originals.get(image.id)
        if original is None:
            return None
        if not image.background_removed:
            return original
        return crop_to_opaque(remove_background(original, image.background_tolerance))

    def _compose_working(self, image) -> np.ndarray:
        original = self._originals[image.id]
        return compose_painted(
            original,
            image.paint,
            removed=image.background_removed,
            tolerance=image.background_tolerance,
            flatten=image.flatten_enabled,
            cutoffs=image.flatten_cutoffs,
            colors=image.flatten_colors,
            smooth=image.flatten_smooth,
        )

    def _rebuild_working(self, image, *, fit: bool | None = None) -> np.ndarray | None:
        """Recompute one image's working pixels from its file and settings."""
        if image.id not in self._originals:
            return None
        pixels = self._compose_working(image)
        previous = self._working.get(image.id)
        size_changed = previous is None or previous.shape[:2] != pixels.shape[:2]
        self._store_working(image.id, pixels, fit=size_changed if fit is None else fit)
        return pixels

    def _pixels_for_canvas(self, image) -> np.ndarray:
        """What the canvas draws for one image.

        Compare-original shows the image after background removal and paint,
        and before color flattening. Paint can change the crop, so this is
        built the same way as the working image with flattening left off.
        Layer masks keep using the flattened working image.
        """
        if image.id in self._comparing and image.flatten_enabled:
            original = self._originals.get(image.id)
            if original is not None:
                return compose_painted(
                    original,
                    image.paint,
                    removed=image.background_removed,
                    tolerance=image.background_tolerance,
                    flatten=False,
                )
        return self._working[image.id]

    def _flatten_signature(self, image):
        return (
            bool(image.flatten_enabled),
            int(image.flatten_count),
            tuple(int(value) for value in image.flatten_cutoffs),
            tuple(None if color is None else tuple(color) for color in image.flatten_colors),
            int(image.flatten_smooth),
        )

    def _align_flatten(self, image) -> None:
        """Keep bucket count, cutoffs, and overrides the same length."""
        count = min(FLATTEN_COUNT_MAX, max(FLATTEN_COUNT_MIN, int(image.flatten_count)))
        image.flatten_count = count
        cutoffs = [int(value) for value in image.flatten_cutoffs]
        valid = len(cutoffs) == count - 1 and all(
            1 <= cutoffs[index] <= 254 and (index == 0 or cutoffs[index] > cutoffs[index - 1])
            for index in range(len(cutoffs))
        )
        if not valid:
            prepared = self._prepared_image(image) if image.flatten_enabled else None
            if prepared is not None and np.any(prepared[..., 3] > 0):
                cutoffs = auto_cutoffs(histogram(prepared), count)
            else:
                cutoffs = even_cutoffs(count)
        image.flatten_cutoffs = cutoffs
        colors = list(image.flatten_colors)
        if len(colors) < count:
            colors.extend([None] * (count - len(colors)))
        image.flatten_colors = colors[:count]
        image.flatten_smooth = min(FLATTEN_SMOOTH_MAX, max(0, int(image.flatten_smooth)))

    def _flatten_preview(self, image):
        count = image.flatten_count
        cutoffs = [int(value) for value in image.flatten_cutoffs]
        if len(cutoffs) != count - 1:
            cutoffs = even_cutoffs(count)
        original = self._originals.get(image.id)
        key = (
            None if original is None else id(original),
            bool(image.background_removed),
            int(image.background_tolerance) if image.background_removed else 0,
            tuple(cutoffs),
            tuple(
                None if color is None else tuple(int(channel) for channel in color)
                for color in image.flatten_colors
            ),
        )
        cached = self._preview_cache.get(image.id)
        if cached is not None and cached[0] == key:
            return cached[1]
        prepared = self._prepared_image(image)
        if prepared is None:
            grays = [(30 + index * 18, 30 + index * 18, 30 + index * 18) for index in range(count)]
            result = cutoffs, grays, np.zeros(256, dtype=np.int64)
        else:
            result = cutoffs, palette_for(prepared, cutoffs, image.flatten_colors), histogram(prepared)
        self._preview_cache[image.id] = (key, result)
        return result

    def set_flatten(
        self,
        enabled: bool | None = None,
        *,
        count: int | None = None,
        cutoffs: list[int] | None = None,
        colors: list | None = None,
        smooth: int | None = None,
    ) -> None:
        """Update the active image's flatten settings and rebuild its working pixels."""
        with self._edit():
            image = self.active_image()
            if image is None:
                self.statusBar().showMessage("Open an image first.")
                self._sync_base_controls()
                return
            before = self._flatten_signature(image)
            if count is not None and int(count) != image.flatten_count:
                count = int(count)
                if cutoffs is None:
                    cutoffs = even_cutoffs(count)
                if colors is None:
                    old = list(image.flatten_colors)
                    colors = [old[index] if index < len(old) else None for index in range(count)]
                image.flatten_count = count
            elif count is not None:
                image.flatten_count = int(count)
            if cutoffs is not None:
                image.flatten_cutoffs = [int(value) for value in cutoffs]
            if colors is not None:
                image.flatten_colors = [
                    None if color is None else (int(color[0]), int(color[1]), int(color[2]))
                    for color in colors
                ]
            if smooth is not None:
                image.flatten_smooth = int(smooth)
            if enabled is not None:
                image.flatten_enabled = bool(enabled)
                if not image.flatten_enabled:
                    self._comparing.discard(image.id)
            self._align_flatten(image)
            changed = self._flatten_signature(image) != before
            if changed:
                self._mark_modified()
            if changed and image.id in self._originals:
                self._rebuild_working(image)
            else:
                self._sync_base_controls()

    def _on_flatten_toggled(self, enabled: bool) -> None:
        image = self.active_image()
        if image is None:
            if enabled:
                self.base_panel.flatten_check.blockSignals(True)
                self.base_panel.flatten_check.setChecked(False)
                self.base_panel.flatten_check.blockSignals(False)
                self.statusBar().showMessage("Open an image first.")
            return
        if enabled == image.flatten_enabled:
            return
        self.set_flatten(enabled)

    def _on_flatten_count(self, count: int) -> None:
        image = self.active_image()
        if image is None or count == image.flatten_count:
            return
        self.set_flatten(count=count)

    def _on_flatten_cutoffs(self, cutoffs) -> None:
        image = self.active_image()
        if image is None:
            return
        values = [int(value) for value in cutoffs]
        if values == list(image.flatten_cutoffs):
            return
        self.set_flatten(cutoffs=values)

    def _on_auto_cutoffs(self) -> None:
        image = self.active_image()
        prepared = self._prepared_image(image) if image is not None else None
        if image is None or prepared is None:
            self.statusBar().showMessage("Open an image first.")
            return
        cutoffs = auto_cutoffs(histogram(prepared), image.flatten_count)
        if cutoffs == list(image.flatten_cutoffs):
            self.statusBar().showMessage("Cutoffs already follow the histogram.")
            return
        self.set_flatten(cutoffs=cutoffs)

    def _on_reset_cutoffs(self) -> None:
        image = self.active_image()
        if image is None:
            return
        cutoffs = even_cutoffs(image.flatten_count)
        if cutoffs == list(image.flatten_cutoffs):
            self.statusBar().showMessage("Cutoffs are already evenly spaced.")
            return
        self.set_flatten(cutoffs=cutoffs)

    def _on_flatten_smooth(self, value: int) -> None:
        image = self.active_image()
        if image is None or value == image.flatten_smooth:
            return
        self.set_flatten(smooth=value)

    def _base_shown(self, image) -> bool:
        """The base image is drawn when its eye is on, and also while comparing."""
        return bool(image.visible) or (
            image.id in self._comparing and image.flatten_enabled
        )

    def _on_compare_original(self, checked: bool) -> None:
        image = self.active_image()
        if image is None:
            return
        if checked and image.flatten_enabled:
            self._comparing.add(image.id)
        else:
            self._comparing.discard(image.id)
        self._refresh_canvas(fit=False)

    def _on_flatten_color(self, index: int, rgb) -> None:
        image = self.active_image()
        if image is None or not 0 <= index < image.flatten_count:
            return
        colors = list(image.flatten_colors)
        if len(colors) < image.flatten_count:
            colors.extend([None] * (image.flatten_count - len(colors)))
        colors = colors[: image.flatten_count]
        new = None if rgb is None else (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        if colors[index] == new:
            return
        colors[index] = new
        self.set_flatten(colors=colors)

    def _colors_for(self, image_id: int) -> list:
        """Dominant colors of one working image, reused until that image changes."""
        colors = self._palettes.get(image_id)
        if colors is None:
            working = self._working.get(image_id)
            if working is None:
                return []
            colors = dominant_colors(working)
            self._palettes[image_id] = colors
        return colors

    def set_background_removed(self, enabled: bool, tolerance: int | None = None) -> None:
        """Make the active image's outer background transparent (or restore it)."""
        with self._edit():
            image = self.active_image()
            original = self._originals.get(image.id) if image is not None else None
            if image is not None and tolerance is not None:
                image.background_tolerance = int(tolerance)
            if original is None or image is None:
                self._set_checked(self.remove_bg_action, False)
                self._sync_base_controls()
                self.statusBar().showMessage("Open an image first.")
                return
            if enabled != image.background_removed or tolerance is not None:
                self._mark_modified()
            image.background_removed = enabled
            self._set_checked(self.remove_bg_action, enabled)
            if enabled:
                cleared = remove_background(original, image.background_tolerance)
                removed_px = int(((original[..., 3] > 0) & (cleared[..., 3] == 0)).sum())
            pixels = self._rebuild_working(image)
            if pixels is None:
                return
            if enabled:
                message = (
                    f"Background removed ({removed_px:,} px made transparent). "
                    f"Image cropped to {pixels.shape[1]} x {pixels.shape[0]} px."
                )
            else:
                message = "Background restored."
            self.statusBar().showMessage(message)

    def _on_bg_tolerance(self, value: int):
        with self._edit():
            image = self.active_image()
            if image is None:
                return
            if value != image.background_tolerance:
                image.background_tolerance = value
                self._mark_modified()
            if image.background_removed:
                self.set_background_removed(True)

    def _on_base_eye(self, image_id, visible: bool):
        if image_id is None:
            return
        self._set_image_visible(image_id, bool(visible))

    def _set_image_visible(self, image_id: int, visible: bool) -> None:
        with self._edit():
            image = self.project.get_image(image_id)
            if image is None:
                return
            if image.visible != visible:
                image.visible = visible
                self._mark_modified()
            self.canvas.set_base_visible(self._base_shown(image), image_id)
            self.layer_pane.show_image_visibility(image_id, visible)
            if self.active_image() is image:
                self._set_checked(self.base_visible_action, visible)

    def _on_base_visible_toggled(self, visible: bool):
        image = self.active_image()
        if image is None:
            self.layer_pane.set_base_visible(visible)
            return
        self._set_image_visible(image.id, visible)

    def _on_checker_opacity(self, percent: int):
        with self._edit():
            self.canvas.set_checker_opacity(percent)
            if percent != self.project.checker_opacity:
                self.project.checker_opacity = percent
                self._mark_modified()

    def _on_add_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Add image", "", IMAGE_FILTER)
        if path:
            try:
                self.add_image(path)
            except (OSError, UnidentifiedImageError) as e:
                QMessageBox.critical(self, "Add image", f"Could not open image:\n{e}")

    def _on_remove_image(self):
        image = self.active_image()
        if image is not None:
            self.remove_image(image.id)

    def _on_open_image(self):
        if not self._confirm_discard_changes("opening another image"):
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", IMAGE_FILTER)
        if path:
            try:
                self.load_image(path)
            except (OSError, UnidentifiedImageError) as e:  # unreadable/corrupt file
                QMessageBox.critical(self, "Open image", f"Could not open image:\n{e}")

    # ---- layers -------------------------------------------------------------------

    def add_layer(self, layer_type: LayerType) -> Layer | None:
        with self._edit():
            if self.image is None:
                QMessageBox.information(
                    self, "Add layer", "Open an image before adding layers."
                )
                return None
            name = default_layer_name(layer_type, self.project.layers)
            image = self.active_image()
            try:
                layer = self.project.add_layer(
                    Layer(name=name, layer_type=layer_type),
                    image_id=image.id if image is not None else None,
                )
            except ProjectError as e:
                QMessageBox.warning(self, "Add layer", str(e))
                return None
            self._refresh_layers(selected_id=layer.id)
            self._mark_modified()
            return layer

    def duplicate_layer(self, layer_id: int) -> Layer | None:
        """Add a new layer with the same settings, placed just below the original."""
        with self._edit():
            source = self.project.get_layer(layer_id)
            if source is None:
                return None
            copy = Layer(
                name=duplicate_layer_name(source.name, self.project.layers),
                layer_type=source.layer_type,
                side=source.side,
                color=source.color,
                tolerance=source.tolerance,
                close_radius=source.close_radius,
                min_island=source.min_island,
                fill_holes=source.fill_holes,
                smoothness=source.smoothness,
                outline_mode=source.outline_mode,
                visible=source.visible,
                has_color=source.has_color,
                negative_of=list(source.negative_of),
                inverted=source.inverted,
                manual=[tuple(run) for run in source.manual],
            )
            owner = self.project.image_of(source.id)
            try:
                self.project.add_layer(
                    copy, after_id=source.id, image_id=owner.id if owner is not None else None
                )
            except ProjectError as e:
                QMessageBox.warning(self, "Duplicate layer", str(e))
                return None
            self._refresh_layers(selected_id=copy.id)
            if copy.has_artwork:
                self._mark_dirty(copy.id)
            self._mark_modified()
            return copy

    def change_layer_type(self, new_type: LayerType) -> bool:
        """Retype the selected layer. Each cut layer stays one closed perimeter."""
        with self._edit():
            layer = self.selected_layer()
            if layer is None or layer.layer_type == new_type:
                return False
            if new_type == LayerType.CUT and layer.is_derived:
                QMessageBox.warning(
                    self,
                    "Layer type",
                    "A negative layer cannot be a cut layer.",
                )
                self.inspector.set_layer(layer, self._source_names(layer))
                return False
            layer.layer_type = new_type
            self._refresh_layers(selected_id=layer.id)
            self._mark_dirty(layer.id)
            self._mark_modified()
            return True

    def generate_negative(
        self, source_ids: list[int], layer_type: LayerType = LayerType.COPPER
    ) -> Layer | None:
        """Create a layer covering everything the given layers do not."""
        with self._edit():
            if self.image is None:
                QMessageBox.information(self, "Generate negative", "Open an image first.")
                return None
            if layer_type not in NEGATIVE_TYPES:
                QMessageBox.warning(
                    self,
                    "Generate negative",
                    "A negative can be copper, solder mask opening, or silkscreen.",
                )
                return None
            sources = [self.project.get_layer(i) for i in source_ids]
            sources = [s for s in sources if s is not None]
            if not sources:
                QMessageBox.information(
                    self, "Generate negative", "Select at least one layer first."
                )
                return None
            uncolored = [
                s.name for s in sources if not s.is_derived and not s.has_color and not s.manual
            ]
            if uncolored:
                QMessageBox.warning(
                    self,
                    "Generate negative",
                    "Pick a color for these layers first:\n" + "\n".join(uncolored),
                )
                return None

            name = "Negative of " + ", ".join(s.name for s in sources)
            sided = next((s for s in sources if not s.is_cut), None)
            layer = Layer(
                name=name,
                layer_type=layer_type,
                side=sided.side if sided is not None else Side.FRONT,
                min_island=4,
                has_color=True,
                negative_of=[s.id for s in sources],
            )
            image = self.active_image()
            self.project.add_layer(layer, image_id=image.id if image is not None else None)
            self._refresh_layers(selected_id=layer.id)
            self._mark_dirty(layer.id)
            self._mark_modified()
            return layer

    def remove_layer(self, layer_id: int) -> None:
        with self._edit():
            affected = self.project.dependents(layer_id)
            removed = self.project.remove_layer(layer_id)
            for rid in removed:
                self.masks.pop(rid, None)
                self.outlines.pop(rid, None)
                self._dirty_layers.discard(rid)
                self.canvas.remove_layer_items(rid)
            self._refresh_layers()
            if self.selected_layer() is None:
                self.layer_pane.select_base()
            remaining = [a for a in affected if a not in removed]
            if remaining:
                self._mark_dirty(*remaining)
            if removed:
                self._mark_modified()
            if len(removed) > 1:
                self.statusBar().showMessage(
                    f"Also removed {len(removed) - 1} negative layer(s) that depended "
                    "only on the deleted layer."
                )

    def move_layer(self, layer_id: int, delta: int) -> None:
        with self._edit():
            before = [layer.id for layer in self.project.layers]
            self.project.move_layer(layer_id, delta)
            self._refresh_layers(selected_id=layer_id)
            if [layer.id for layer in self.project.layers] != before:
                self._mark_modified()

    def set_layer_visible(self, layer_id: int, visible: bool) -> None:
        with self._edit():
            layer = self.project.get_layer(layer_id)
            if layer is not None:
                if layer.visible != visible:
                    layer.visible = visible
                    self._mark_modified()
                self.canvas.set_layer_visible(layer_id, visible)
                self.layer_pane.show_layer_visibility(layer_id, visible)  # keep the eye in sync

    def selected_layer(self) -> Layer | None:
        return self.project.get_layer(self.layer_pane.current_layer_id())

    def _source_names(self, layer: Layer | None) -> list[str]:
        if layer is None:
            return []
        names = (self.project.get_layer(s) for s in layer.negative_of)
        return [l.name for l in names if l is not None]

    def _on_layer_selected(self, layer_id: int):
        layer = self.project.get_layer(layer_id)
        self.inspector.set_layer(layer, self._source_names(layer))
        self.inspector_stack.setCurrentWidget(self.inspector)
        self._sync_paint_controls()
        parent = self.project.image_of(layer_id) if layer is not None else None
        if parent is not None and parent.id in self._working:
            self.inspector.set_dominant_colors(self._colors_for(parent.id))
        self._note_selection()

    def _on_base_selected(self, image_id=None):
        self.inspector_stack.setCurrentWidget(self.base_panel)
        self._sync_base_controls()
        self._sync_paint_controls()
        self._note_selection()

    def _on_layer_renamed(self, layer_id: int, name: str):
        with self._edit():
            layer = self.project.get_layer(layer_id)
            if layer is not None and name.strip() and layer.name != name.strip():
                layer.name = name.strip()
                self.layer_pane.show_layer_name(layer.id, layer.name)
                self._mark_modified()

    def _on_layer_edited(self):
        with self._edit():
            layer = self.selected_layer()
            if layer is not None:
                self.layer_pane.note_layer_changed(layer)
                self._mark_dirty(layer.id)
                self._mark_modified()

    def _on_pixel_picked(self, x: int, y: int):
        image = self.image
        if image is None or not (0 <= y < image.shape[0] and 0 <= x < image.shape[1]):
            return
        r, g, b = (int(v) for v in image[y, x, :3])
        self.inspector.apply_color((r, g, b))
        self.inspector.eyedropper_button.setChecked(False)

    # ---- paint tools ------------------------------------------------------------------

    def _sync_paint_controls(self) -> None:
        """The color swatch paints the base image. Layers are on/off masks."""
        editing_base = self.selected_layer() is None and self.image is not None
        self.paint_toolbar.set_color_enabled(editing_base)

    def _on_paint_tool(self, tool: Tool) -> None:
        self.canvas.set_tool(tool)
        self.statusBar().showMessage(
            {
                Tool.PAN: "Drag to pan. Scroll to zoom. Middle-drag pans while a paint tool is active.",
                Tool.BRUSH: "Paintbrush: drag to add pixels.",
                Tool.LINE: "Straight line: drag from one point to another. Hold Ctrl to snap to 45-degree steps.",
                Tool.RECTANGLE: (
                    "Rectangle: click one corner, then the opposite corner. "
                    "Only the outline is drawn. Hold Ctrl to lock a square. Esc cancels."
                ),
                Tool.ERASE: "Eraser: drag to remove pixels.",
                Tool.FILL: "Fill: click an area. Brush size does not apply.",
            }.get(tool, "")
        )

    def set_paint_tool(self, tool: Tool) -> None:
        self.paint_toolbar.set_tool(tool)

    def set_brush_size(self, size: int) -> None:
        self.paint_toolbar.set_brush_size(size)

    def set_paint_color(self, color) -> None:
        self.paint_toolbar.set_color(color)

    def _paint_target(self):
        """The selected layer, or the selected base image when no layer is selected."""
        layer = self.selected_layer()
        if layer is not None:
            return "layer", layer
        image = self.active_image()
        if image is not None and image.id in self._working:
            return "image", image
        return None

    def paint_stroke(self, points, value: int | None = None) -> None:
        """Paint or erase through `points`. The eraser tool erases; other tools paint."""
        if not points:
            return
        self.begin_paint_stroke(int(points[0][0]), int(points[0][1]), value=value)
        for x, y in points[1:]:
            self.continue_paint_stroke(int(x), int(y))
        self.end_paint_stroke()

    def paint_line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self.begin_paint_stroke(int(x0), int(y0), value=PAINT)
        if self._stroke is None:
            return
        if (int(x0), int(y0)) != (int(x1), int(y1)):
            self._stamp_segment((int(x0), int(y0)), (int(x1), int(y1)))
        self.end_paint_stroke()

    def paint_rectangle(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self.begin_paint_stroke(int(x0), int(y0), value=PAINT)
        if self._stroke is None:
            return
        self._paint_rectangle_body(int(x0), int(y0), int(x1), int(y1))
        self.end_paint_stroke()

    def paint_fill(self, x: int, y: int) -> None:
        target = self._paint_target()
        if target is None:
            self.statusBar().showMessage("Open an image before painting.")
            return
        kind, obj = target
        if kind == "layer":
            self._fill_layer(obj, int(x), int(y))
        else:
            self._fill_image(obj, int(x), int(y))

    def begin_paint_stroke(self, x: int, y: int, value: int | None = None) -> None:
        if value is None:
            value = ERASE if self.paint_toolbar.tool == Tool.ERASE else PAINT
        target = self._paint_target()
        if target is None:
            self._stroke = None
            self.statusBar().showMessage("Open an image before painting.")
            return
        kind, obj = target
        size = self.paint_toolbar.brush_size
        if kind == "layer":
            height, width = self._canvas_shape()
            if height <= 0 or width <= 0:
                self._stroke = None
                return
            mask = self.masks.get(obj.id)
            if mask is None or mask.shape[:2] != (height, width):
                mask = np.zeros((height, width), dtype=np.uint8)
            else:
                mask = np.array(mask, copy=True)
            self._stroke = {
                "kind": "layer",
                "layer": obj,
                "edits": edits_from_runs(obj.manual, (height, width)),
                "mask": mask,
                "value": int(value),
                "size": size,
                "last": (int(x), int(y)),
            }
        else:
            working = self._working.get(obj.id)
            if working is None:
                self._stroke = None
                return
            color = (0, 0, 0, 0) if value == ERASE else self.paint_toolbar.color
            self._stroke = {
                "kind": "image",
                "image": obj,
                "pixels": np.array(working, copy=True),
                "entries": [],
                "origin": paint_origin(
                    self._originals[obj.id],
                    obj.paint,
                    obj.background_removed,
                    obj.background_tolerance,
                ),
                "color": color,
                "value": int(value),
                "size": size,
                "last": (int(x), int(y)),
            }
        self._stamp_dab(int(x), int(y))

    def continue_paint_stroke(self, x: int, y: int) -> None:
        if self._stroke is None:
            return
        point = (int(x), int(y))
        last = self._stroke["last"]
        if point != last:
            self._stamp_segment(last, point)
            self._stroke["last"] = point

    def end_paint_stroke(self) -> None:
        with self._edit():
            stroke = self._stroke
            self._stroke = None
            if stroke is None:
                return
            if stroke["kind"] == "layer":
                layer = stroke["layer"]
                manual = runs_from_edits(stroke["edits"])
                if manual == list(layer.manual):
                    return
                layer.manual = manual
                self._mark_modified()
                self._recompute_layer_and_dependents(layer)
                return
            if not stroke["entries"]:
                return
            image = stroke["image"]
            image.paint = coalesce_paint([*image.paint, *stroke["entries"]])
            self._rebuild_working(image)
            self.flush_recompute()
            self._mark_modified()

    def _stamp_dab(self, x: int, y: int) -> None:
        stroke = self._stroke
        if stroke is None:
            return
        if stroke["kind"] == "layer":
            shown = 255 if stroke["value"] == PAINT else 0
            stamp(stroke["edits"], x, y, stroke["size"], stroke["value"])
            stamp(stroke["mask"], x, y, stroke["size"], shown)
            layer = stroke["layer"]
            self.canvas.set_layer_overlay(layer.id, stroke["mask"], layer.layer_type.value)
            return
        self._stamp_image_disk(x, y)
        self.canvas.set_base_pixels(stroke["image"].id, stroke["pixels"])

    def _stamp_segment(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        stroke = self._stroke
        if stroke is None:
            return
        if stroke["kind"] == "layer":
            shown = 255 if stroke["value"] == PAINT else 0
            raster_stroke(
                stroke["edits"], start[0], start[1], end[0], end[1], stroke["size"], stroke["value"]
            )
            raster_stroke(
                stroke["mask"], start[0], start[1], end[0], end[1], stroke["size"], shown
            )
            layer = stroke["layer"]
            self.canvas.set_layer_overlay(layer.id, stroke["mask"], layer.layer_type.value)
            return
        for x, y in line_points(start[0], start[1], end[0], end[1]):
            self._stamp_image_disk(x, y)
        self.canvas.set_base_pixels(stroke["image"].id, stroke["pixels"])

    def _stamp_image_disk(self, x: int, y: int) -> None:
        stroke = self._stroke
        pixels = stroke["pixels"]
        height, width = pixels.shape[:2]
        origin_x, origin_y = stroke["origin"]
        color = stroke["color"]
        for px, py in iter_disk(x, y, stroke["size"], height, width):
            stroke["entries"].append((px + origin_x, py + origin_y, *color))
        stamp(pixels, x, y, stroke["size"], color)

    def _paint_rectangle_body(self, x0: int, y0: int, x1: int, y1: int) -> None:
        stroke = self._stroke
        if stroke is None:
            return
        if stroke["kind"] == "layer":
            stroke_rectangle(stroke["edits"], x0, y0, x1, y1, stroke["size"], PAINT)
            stroke_rectangle(stroke["mask"], x0, y0, x1, y1, stroke["size"], 255)
            layer = stroke["layer"]
            self.canvas.set_layer_overlay(layer.id, stroke["mask"], layer.layer_type.value)
            return
        before = np.array(stroke["pixels"], copy=True)
        stroke_rectangle(stroke["pixels"], x0, y0, x1, y1, stroke["size"], stroke["color"])
        changed = np.any(stroke["pixels"] != before, axis=-1)
        origin_x, origin_y = stroke["origin"]
        color = stroke["color"]
        ys, xs = np.nonzero(changed)
        for x, y in zip(xs.tolist(), ys.tolist()):
            stroke["entries"].append((x + origin_x, y + origin_y, *color))
        self.canvas.set_base_pixels(stroke["image"].id, stroke["pixels"])

    def _fill_layer(self, layer: Layer, x: int, y: int) -> None:
        with self._edit():
            height, width = self._canvas_shape()
            if height <= 0 or not (0 <= x < width and 0 <= y < height):
                return
            mask = self.masks.get(layer.id)
            if mask is None or mask.shape[:2] != (height, width):
                mask = np.zeros((height, width), dtype=np.uint8)
            # Clicking an area that is already on would freeze that whole region
            # into the manual override. Leave it to the color mask instead.
            if mask[y, x] > 0:
                return
            region = flood_region(mask > 0, x, y)
            edits = edits_from_runs(layer.manual, (height, width))
            edits[region] = PAINT
            layer.manual = runs_from_edits(edits)
            self._mark_modified()
            self._recompute_layer_and_dependents(layer)

    def _fill_image(self, image, x: int, y: int) -> None:
        with self._edit():
            working = self._working.get(image.id)
            if working is None or not (0 <= y < working.shape[0] and 0 <= x < working.shape[1]):
                return
            color = self.paint_toolbar.color
            filled = flood_fill_rgba(working, x, y, color)
            if np.array_equal(filled, working):
                return
            changed = np.any(filled != working, axis=-1)
            origin = paint_origin(
                self._originals[image.id],
                image.paint,
                image.background_removed,
                image.background_tolerance,
            )
            origin_x, origin_y = origin
            entries = list(image.paint)
            ys, xs = np.nonzero(changed)
            for px, py in zip(xs.tolist(), ys.tolist()):
                entries.append((px + origin_x, py + origin_y, *color))
            image.paint = coalesce_paint(entries)
            self._rebuild_working(image)
            self.flush_recompute()
            self._mark_modified()

    def _recompute_layer_and_dependents(self, layer: Layer) -> None:
        self.recompute_layer(layer)
        for layer_id in self.project.dependency_order(self.project.dependents(layer.id)):
            dependent = self.project.get_layer(layer_id)
            if dependent is not None:
                self.recompute_layer(dependent)

    def keyPressEvent(self, event):
        if self.canvas.tool == Tool.RECTANGLE and event.key() == Qt.Key_Escape:
            self.canvas.cancel_rectangle()
            event.accept()
            return
        if event.key() == Qt.Key_Control and not event.isAutoRepeat():
            self.canvas.set_constrained(True)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control and not event.isAutoRepeat():
            self.canvas.set_constrained(False)
        super().keyReleaseEvent(event)

    # ---- recompute engine ------------------------------------------------------------

    def _mark_dirty(self, *layer_ids: int):
        self._dirty_layers.update(layer_ids)
        self._recompute_timer.start()

    def _recompute_dirty(self):
        if not self._working:
            return
        ids = set(self._dirty_layers)
        for layer_id in list(ids):
            ids |= self.project.dependents(layer_id)  # negatives follow their sources
        self._dirty_layers.clear()
        for layer_id in self.project.dependency_order(ids):
            layer = self.project.get_layer(layer_id)
            if layer is not None:
                self.recompute_layer(layer)

    def flush_recompute(self):
        """Recompute immediately (used by export and tests)."""
        self._recompute_timer.stop()
        self._recompute_dirty()

    def _alpha_domain(self, images: list[np.ndarray]) -> np.ndarray:
        """RGBA whose alpha is the union of `images`, on the shared canvas."""
        height, width = self._canvas_shape()
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        for image in images:
            h, w = image.shape[:2]
            rgba[:h, :w, 3] = np.maximum(rgba[:h, :w, 3], image[..., 3])
        return rgba

    def recompute_layer(self, layer: Layer) -> None:
        parent = self.project.image_of(layer.id)
        if parent is None or parent.id not in self._working:
            return
        working = self._working[parent.id]
        height, width = self._canvas_shape()
        if height <= 0 or width <= 0:
            return
        native = None
        if layer.is_derived:
            source_ids = [s for s in layer.negative_of if s in self.masks]
            source_images = []
            for source_id in source_ids:
                owner = self.project.image_of(source_id)
                if owner is not None and owner.id in self._working:
                    source_images.append(self._working[owner.id])
            domain = self._alpha_domain(source_images or [working])
            mask = negative_mask(
                domain, [self.masks[s] for s in source_ids], min_island=layer.min_island
            )
            if layer.inverted:
                mask = invert_mask(domain, mask, min_island=layer.min_island)
        elif layer.has_color:
            native = build_mask(
                working,
                layer.color,
                tolerance=layer.tolerance,
                close_radius=layer.close_radius,
                min_island=layer.min_island,
                fill_holes=layer.fill_holes and not layer.is_cut,
            )
            if layer.inverted:
                native = invert_mask(working, native, min_island=layer.min_island)
            mask = pad_top_left(native, height, width)
        elif layer.manual:
            mask = np.zeros((height, width), dtype=np.uint8)
        else:
            self.masks.pop(layer.id, None)
            self.outlines.pop(layer.id, None)
            self.canvas.remove_layer_items(layer.id)
            return
        # Hand-drawn pixels are applied last so they survive tolerance, gap
        # closing, and invert. Paint forces a pixel on; erase forces it off.
        if layer.manual:
            mask = apply_edits(mask, edits_from_runs(layer.manual, (height, width)))
        self.masks[layer.id] = mask
        if layer.is_cut:
            try:
                outline = cut_outline(
                    mask,
                    layer.smoothness,
                    layer.outline_mode,
                )
                self.outlines[layer.id] = outline
                self.canvas.set_layer_outline(layer.id, list(outline.exterior.coords))
                self.statusBar().clearMessage()
            except CutError as e:
                self.outlines.pop(layer.id, None)
                self.canvas.set_layer_outline(layer.id, [])
                self.statusBar().showMessage(f"Cut layer: {e}")
            self.canvas.clear_layer_overlay(layer.id)
        else:
            if layer.id in self.canvas.outline_items:
                self.outlines.pop(layer.id, None)
                self.canvas.set_layer_outline(layer.id, [])
            self.canvas.set_layer_overlay(layer.id, mask, layer.layer_type.value)
        self.canvas.set_layer_visible(layer.id, layer.visible)

    # ---- project files ------------------------------------------------------------------

    def _sync_project_state(self):
        self.project.checker_opacity = self.checker_slider.value()

    def save_project(self, path: str | Path | None = None) -> Path:
        """Write the current project to a .prj file; returns the path written."""
        target = Path(path) if path is not None else self.project_path
        if target is None:
            raise ProjectFileError("No project file path has been chosen yet.")
        self._sync_project_state()
        written = project_io.save_project(target, self.project)
        self.project_path = written
        self._history.mark_saved()
        self._set_unmodified()
        self._update_undo_actions()
        self.statusBar().showMessage(f"Saved {written}")
        return written

    def new_project(self) -> None:
        """Replace the open project with a blank one."""
        self._history_locked = True
        try:
            self._recompute_timer.stop()
            self._stroke = None
            self.canvas.cancel_rectangle()
            self._open_gesture = None
            self.project = Project()
            self.project_path = None
            self.masks.clear()
            self.outlines.clear()
            self._dirty_layers.clear()
            self._originals.clear()
            self._working.clear()
            self._palettes.clear()
            self._preview_cache.clear()
            self._comparing.clear()
            self.checker_slider.blockSignals(True)
            self.checker_slider.setValue(self.project.checker_opacity)
            self.checker_slider.blockSignals(False)
            self.canvas.set_checker_opacity(self.project.checker_opacity)
            self._set_checked(self.base_visible_action, True)
            self._set_checked(self.remove_bg_action, False)
            self.inspector.eyedropper_button.setChecked(False)
            self.inspector.set_layer(None)
            self._refresh_canvas(fit=True)
            self._refresh_layers()
            self.layer_pane.select_base()
            self._set_unmodified()
        finally:
            self._history_locked = False
        self._history.reset(self._capture())
        self._sync_modified()
        self._update_undo_actions()
        self.statusBar().showMessage("Open an image or a project to begin.")

    def _on_new_project(self) -> None:
        if not self._confirm_discard_changes("starting a new project"):
            return
        self.new_project()

    def open_project(self, path: str | Path, image_path: str | Path | None = None) -> None:
        """Load a .prj file, restoring layers, board settings, and view state."""
        path = Path(path)
        project = project_io.load_project(path)
        # Resolve every image before replacing the window, so a missing file
        # leaves the project that is already open untouched.
        loaded: list[tuple] = []
        for image in project.images:
            override = Path(image_path) if image_path and len(project.images) == 1 else None
            candidate = override or (Path(image.path) if image.path else None)
            if candidate is None or not candidate.is_file():
                candidate = self._locate_image(image.path)
                if candidate is None:
                    raise ProjectFileError("The project's source image could not be found.")
            original = load_rgba(candidate)
            loaded.append((image, original, candidate))

        self._history_locked = True
        try:
            self._stroke = None
            self.canvas.cancel_rectangle()
            self._open_gesture = None
            self.project = project
            self.project_path = path
            self.masks.clear()
            self.outlines.clear()
            self._dirty_layers.clear()
            self._originals.clear()
            self._working.clear()
            self._palettes.clear()
            self._preview_cache.clear()
            self._comparing.clear()
            for image, original, candidate in loaded:
                image.path = str(candidate)
                self._originals[image.id] = original
                self._working[image.id] = self._compose_working(image)

            self.checker_slider.blockSignals(True)
            self.checker_slider.setValue(project.checker_opacity)
            self.checker_slider.blockSignals(False)
            self.canvas.set_checker_opacity(project.checker_opacity)

            self._refresh_canvas(fit=True)
            first = project.layers[0].id if project.layers else None
            if first is None:
                self.inspector.set_layer(None)
                self._refresh_layers()
                self.layer_pane.select_base()
            else:
                self._refresh_layers(selected_id=first)
            self.flush_recompute()
            self._set_unmodified()
        finally:
            self._history_locked = False
        self._history.reset(self._capture())
        self._sync_modified()
        self._update_undo_actions()
        self.statusBar().showMessage(f"Opened {path.name}")

    def _locate_image(self, missing: str | None) -> Path | None:
        QMessageBox.warning(
            self,
            "Open project",
            "The project's source image was not found"
            + (f":\n{missing}" if missing else ".")
            + "\n\nChoose the image to use instead.",
        )
        found, _ = QFileDialog.getOpenFileName(
            self, "Locate source image", "", IMAGE_FILTER
        )
        return Path(found) if found else None

    def _on_open_project(self):
        if not self._confirm_discard_changes("opening another project"):
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open project", "", f"Projects (*{EXTENSION});;All files (*)"
        )
        if not path:
            return
        try:
            self.open_project(path)
        except (ProjectFileError, OSError, UnidentifiedImageError) as e:
            QMessageBox.critical(self, "Open project", f"Could not open project:\n{e}")

    def _on_save(self) -> bool:
        return self._save_interactive()

    def _save_interactive(self) -> bool:
        """Save to the current file (asking for one if needed); False if not saved."""
        if self.project_path is None:
            return self._on_save_as()
        return self._try_save(self.project_path)

    def _on_save_as(self) -> bool:
        suggested = str(self.project_path) if self.project_path else (
            str(self.image_path.with_suffix(EXTENSION)) if self.image_path else ""
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save project", suggested, f"Projects (*{EXTENSION})"
        )
        return bool(path) and self._try_save(Path(path))

    def _try_save(self, path: Path) -> bool:
        try:
            self.save_project(path)
        except OSError as e:
            QMessageBox.critical(self, "Save project", f"Could not save project:\n{e}")
            return False
        return True

    # ---- board settings & export ------------------------------------------------------------

    def _store_board_settings(self, settings) -> None:
        with self._edit():
            if settings != self.project.board:
                self.project.board = settings
                self._mark_modified()

    def _on_board_settings(self):
        if not self._working:
            QMessageBox.information(self, "Board settings", "Open an image first.")
            return
        h, w = self._canvas_shape()
        dialog = ExportDialog(
            w, h, self._default_name(), self,
            settings=self.project.board, export_mode=False,
        )
        if dialog.exec() == ExportDialog.Accepted:
            self._store_board_settings(dialog.board_settings())

    def _default_name(self) -> str:
        """Project file name when one is saved, otherwise the image name."""
        if self.project_path is not None:
            return self.project_path.stem
        return self.image_path.stem if self.image_path else "board"

    def _sync_locked_board_height(self) -> None:
        """Keep a locked board height matched to the shared canvas aspect."""
        board = self.project.board
        if not self._working or not board.lock_aspect or board.width_mm is None:
            return
        height_px, width_px = self._canvas_shape()
        if width_px <= 0:
            return
        height_mm = round(board.width_mm * height_px / width_px, 2)
        if board.height_mm != height_mm:
            board.height_mm = height_mm
            self._mark_modified()

    def _on_export_files(self) -> None:
        self._on_export(False)

    def _on_export_zip(self) -> None:
        self._on_export(True)

    def _on_export(self, zip_output: bool = False):
        if not self._working:
            QMessageBox.information(self, "Export", "Open an image first.")
            return
        self.flush_recompute()
        h, w = self._canvas_shape()
        settings = self.project.board
        dialog = ExportDialog(w, h, self._default_name(), self, settings=settings)
        dialog.setWindowTitle("Export Gerber ZIP" if zip_output else "Export Gerber files")
        if not dialog.dir_edit.text() and self.image_path is not None:
            dialog.dir_edit.setText(str(self.image_path.parent))
        if dialog.exec() != ExportDialog.Accepted:
            return
        self._store_board_settings(dialog.board_settings())
        out_dir, name, width_mm, height_mm = dialog.values()
        try:
            if zip_output:
                written = [self.export_gerber_zip(out_dir, name, width_mm, height_mm)]
            else:
                written = self.export_gerbers(out_dir, name, width_mm, height_mm)
        except (ExportError, CutError, OSError) as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return
        QMessageBox.information(
            self,
            "Export complete",
            "Wrote:\n"
            + "\n".join(p.name for p in written)
            + "\n\nReview the layers in the KiCad Gerber Viewer before "
            "fabrication. These files are artwork and a board profile, not "
            "routed or DRC-validated electronics.",
        )

    def export_gerbers(
        self, out_dir: Path, name: str, width_mm: float, height_mm: float
    ) -> list[Path]:
        """Build geometries from current masks and write Gerber files.

        Masks already share one top-left canvas, so layers from every image
        union by (type, side). Every visible cut outline is unioned into
        Edge.Cuts.
        """
        height, width = self._canvas_shape()
        if width <= 0 or height <= 0:
            raise ExportError("No image is loaded.")
        self.flush_recompute()
        transform = BoardTransform(width, height, width_mm, height_mm)

        geometries = {}
        for layer in self.project.artwork_layers():
            if not layer.visible or not layer.has_artwork:
                continue
            mask = self.masks.get(layer.id)
            if mask is None:
                continue
            geom = mask_to_geometry(mask)
            if not geom.is_empty:
                geometries[layer.id] = geom

        outline = self._cut_union()

        return export_project(
            self.project, geometries, outline, out_dir, name, transform
        )

    def export_gerber_zip(
        self, out_dir: Path, name: str, width_mm: float, height_mm: float
    ) -> Path:
        """Write the Gerber files into `{name}.zip` and return that path.

        The loose Gerber files are not left in the output folder. `name` is
        the project name from the export dialog.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_path = out_dir / f"{name}.zip"
        with tempfile.TemporaryDirectory() as tmp:
            files = self.export_gerbers(Path(tmp), name, width_mm, height_mm)
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for file in files:
                    archive.write(file, file.name)
        return zip_path

    def _cut_union(self):
        """Union of every visible cut outline, or None when there is no cut."""
        outlines = []
        for cut in self.project.cut_layers():
            if not cut.visible or not cut.has_artwork:
                continue
            outline = self.outlines.get(cut.id)
            if outline is None:
                raise ExportError(
                    f"{cut.name} has no computed outline; pick its color or paint it first."
                )
            outlines.append(outline)
        if not outlines:
            return None
        return unary_union(outlines)

    def _on_export_image(self) -> None:
        if not self._working:
            QMessageBox.information(self, "Export", "Open an image first.")
            return
        self.flush_recompute()
        layers = [(layer, layer.id in self.masks) for layer in self.project.layers]
        dialog = ImageExportDialog(
            layers, self._default_name(), self.project.board.output_dir, self
        )
        if not dialog.dir_edit.text() and self.image_path is not None:
            dialog.dir_edit.setText(str(self.image_path.parent))
        if dialog.exec() != ImageExportDialog.Accepted:
            return
        out_dir, name, layer_ids, render = dialog.values()
        try:
            written = self.export_images(out_dir, name, layer_ids, render)
        except (ExportError, OSError) as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return
        QMessageBox.information(
            self,
            "Export complete",
            "Wrote:\n" + "\n".join(path.name for path in written),
        )

    def export_images(
        self, out_dir: Path, name: str, layer_ids: list[int], render: bool
    ) -> list[Path]:
        """Write one PNG per selected layer, and a rendered board if requested.

        The rendered image includes every layer that has a mask, including
        layers that were not selected for their own file, and clips to the
        union of the cut outlines when the project has any.
        """
        if not self._working:
            raise ExportError("No image is loaded.")
        self.flush_recompute()
        out_dir = Path(out_dir)

        wanted = set(layer_ids)
        by_id = {layer.id: layer for layer in self.project.layers}
        missing = wanted - set(by_id)
        if missing:
            raise ExportError("A selected layer is no longer in the project.")

        used: set[str] = set()
        if render:
            used.add(f"{name}-{RENDER_NAME}".lower())

        selected: list[tuple[Layer, np.ndarray]] = []
        for layer in self.project.layers:
            if layer.id not in wanted:
                continue
            mask = self.masks.get(layer.id)
            if mask is None:
                raise ExportError(
                    f"{layer.name} has no mask yet. Pick its color or paint on it first."
                )
            selected.append((layer, mask))

        if render:
            render_parts = [
                (layer, self.masks[layer.id])
                for layer in self.project.layers
                if layer.id in self.masks
            ]
            if not render_parts:
                raise ExportError("Nothing to render: no layer has a mask yet.")
        else:
            render_parts = []

        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for layer, mask in selected:
            path = out_dir / layer_png_name(name, layer.name, used)
            if layer.is_cut:
                outline = self.outlines.get(layer.id)
                if outline is None:
                    raise ExportError(
                        f"{layer.name} has no computed outline; pick its color first."
                    )
                rgba = cut_perimeter_rgba(mask.shape[0], mask.shape[1], outline)
            else:
                rgba = layer_rgba(mask, layer.layer_type)
            save_rgba(path, rgba)
            written.append(path)

        if render:
            parts = render_parts
            polygon = self._cut_union()
            height, width = self._canvas_shape()
            path = out_dir / f"{name}-{RENDER_NAME}.png"
            save_rgba(path, render_pcb(height, width, parts, cut_polygon=polygon))
            written.append(path)
        return written
