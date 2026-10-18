"""Main window: wires the canvas, layer pane, inspector, project files, and export."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import UnidentifiedImageError
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
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
    QVBoxLayout,
    QWidget,
)

from .. import project_io
from ..background import crop_to_opaque, remove_background, working_image
from ..gerber import BoardTransform, ExportError, export_project
from ..geometry import CutError, cut_outline, mask_to_geometry
from ..image_io import dominant_colors, load_rgba
from ..masking import build_mask, invert_mask, negative_mask
from ..model import (
    Layer,
    LayerType,
    Project,
    ProjectError,
    Side,
    default_layer_name,
    duplicate_layer_name,
)
from ..project_io import EXTENSION, ProjectFileError
from .base_panel import BaseImagePanel
from .canvas import ImageCanvas
from .export_dialog import ExportDialog
from .inspector import LayerInspector
from .layer_pane import NEGATIVE_TYPES, LayerPane
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
        self.original_image: np.ndarray | None = None  # as loaded from disk
        self.image: np.ndarray | None = None  # what layers are masked against
        self.image_path: Path | None = None
        self.masks: dict[int, np.ndarray] = {}
        self.outlines: dict[int, object] = {}  # layer id -> shapely Polygon
        self._dirty_layers: set[int] = set()
        self._modified = False

        self._recompute_timer = QTimer(self)
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.setInterval(RECOMPUTE_DEBOUNCE_MS)
        self._recompute_timer.timeout.connect(self._recompute_dirty)

        # Widgets
        self.canvas = ImageCanvas(self)
        self.canvas.pixel_picked.connect(self._on_pixel_picked)
        self.setCentralWidget(self.canvas)

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
        self._connect_base_controls()
        self.layer_pane.select_base()
        self._update_title()
        self.statusBar().showMessage("Open an image or a project to begin.")

    # ---- UI construction ------------------------------------------------------------

    def _build_actions(self):
        self.open_project_action = QAction("Open &project...", self)
        self.open_project_action.setShortcut("Ctrl+Shift+O")
        self.open_project_action.triggered.connect(self._on_open_project)
        self.save_action = QAction("&Save project", self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.triggered.connect(self._on_save)
        self.save_as_action = QAction("Save project &as...", self)
        self.save_as_action.setShortcut("Ctrl+Shift+S")
        self.save_as_action.triggered.connect(self._on_save_as)
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
            self.open_project_action,
            self.save_action,
            self.save_as_action,
            None,
            self.open_action,
            self.board_action,
            None,
        ):
            file_menu.addSeparator() if action is None else file_menu.addAction(action)
        self.export_menu = QMenu("E&xport", self)
        self.export_menu.addAction(self.export_files_action)
        self.export_menu.addAction(self.export_zip_action)
        file_menu.addMenu(self.export_menu)
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.base_visible_action)

        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        self.checker_label = QLabel("Checker opacity", self)
        toolbar.addWidget(self.checker_label)
        toolbar.addWidget(self.checker_slider)

    def _connect_base_controls(self):
        """Keep the base-image row/panel and the menu actions in step."""
        self.layer_pane.base_visibility_changed.connect(self.base_visible_action.setChecked)
        self.base_panel.open_button.clicked.connect(lambda: self._on_open_image())
        self.base_panel.remove_bg_check.toggled.connect(self.remove_bg_action.setChecked)
        self.base_panel.tolerance_spin.valueChanged.connect(self._on_bg_tolerance)
        self.base_panel.tolerance_spin.setValue(self.project.background_tolerance)

    def _sync_base_controls(self):
        self.layer_pane.set_base_visible(self.base_visible_action.isChecked())
        self.base_panel.sync(
            self.project.background_removed, self.project.background_tolerance
        )

    # ---- modified tracking ---------------------------------------------------------------

    @property
    def is_modified(self) -> bool:
        return self._modified

    def _mark_modified(self):
        if not self._modified:
            self._modified = True
            self._update_title()

    def _set_unmodified(self):
        self._modified = False
        self._update_title()

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

    # ---- image ------------------------------------------------------------------

    def load_image(self, path: str | Path) -> None:
        """Load a fresh source image (keeps existing layers, resets background removal)."""
        original = load_rgba(path)
        self.original_image = original
        self.image_path = Path(path)
        self.project.image_path = str(self.image_path)
        self.project.background_removed = False
        self.remove_bg_action.blockSignals(True)
        self.remove_bg_action.setChecked(False)
        self.remove_bg_action.blockSignals(False)
        self._apply_image(original, fit=True)
        if self.project.layers or self.project_path is not None:
            self._mark_modified()  # a bare image with no work in it isn't worth saving
        self.statusBar().showMessage(
            f"Loaded {self.image_path.name} "
            f"({original.shape[1]} x {original.shape[0]} px)"
        )

    def _apply_image(self, image: np.ndarray, fit: bool) -> None:
        """Make `image` the masking source and refresh everything derived from it."""
        self.image = image
        self.canvas.set_image(image, fit=fit)
        self.canvas.set_base_visible(self.base_visible_action.isChecked())
        self.inspector.set_dominant_colors(dominant_colors(image))
        if self.image_path is not None:
            self.base_panel.set_image_info(
                self.image_path.name, int(image.shape[1]), int(image.shape[0])
            )
        self._sync_locked_board_height()
        self._sync_base_controls()
        self._mark_dirty(*[l.id for l in self.project.layers])

    def set_background_removed(self, enabled: bool, tolerance: int | None = None) -> None:
        """Make the image's outer contiguous background transparent (or restore it)."""
        if tolerance is not None:
            self.project.background_tolerance = int(tolerance)
        if self.original_image is None:
            self.remove_bg_action.blockSignals(True)
            self.remove_bg_action.setChecked(False)
            self.remove_bg_action.blockSignals(False)
            self._sync_base_controls()
            self.statusBar().showMessage("Open an image first.")
            return
        if enabled != self.project.background_removed or tolerance is not None:
            self._mark_modified()
        self.project.background_removed = enabled
        self.remove_bg_action.blockSignals(True)
        self.remove_bg_action.setChecked(enabled)
        self.remove_bg_action.blockSignals(False)
        if enabled:
            cleared = remove_background(
                self.original_image, self.project.background_tolerance
            )
            removed = int(((self.original_image[..., 3] > 0) & (cleared[..., 3] == 0)).sum())
            image = crop_to_opaque(cleared)
            message = (
                f"Background removed ({removed:,} px made transparent). "
                f"Image cropped to {image.shape[1]} x {image.shape[0]} px."
            )
        else:
            image = self.original_image
            message = "Background restored."
        size_changed = self.image is None or self.image.shape[:2] != image.shape[:2]
        self._apply_image(image, fit=size_changed)
        self.statusBar().showMessage(message)

    def _on_bg_tolerance(self, value: int):
        if value != self.project.background_tolerance:
            self.project.background_tolerance = value
            self._mark_modified()
        if self.project.background_removed:
            self.set_background_removed(True)

    def _on_base_visible_toggled(self, visible: bool):
        self.canvas.set_base_visible(visible)
        self.layer_pane.set_base_visible(visible)
        self.project.base_visible = visible
        self._mark_modified()

    def _on_checker_opacity(self, percent: int):
        self.canvas.set_checker_opacity(percent)
        if percent != self.project.checker_opacity:
            self.project.checker_opacity = percent
            self._mark_modified()

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
        if self.image is None:
            QMessageBox.information(
                self, "Add layer", "Open an image before adding layers."
            )
            return None
        name = default_layer_name(layer_type, self.project.layers)
        try:
            layer = self.project.add_layer(Layer(name=name, layer_type=layer_type))
        except ProjectError as e:
            QMessageBox.warning(self, "Add layer", str(e))
            return None
        self.layer_pane.refresh(self.project.layers, selected_id=layer.id)
        self._mark_modified()
        return layer

    def duplicate_layer(self, layer_id: int) -> Layer | None:
        """Add a new layer with the same settings, placed just below the original."""
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
        )
        try:
            self.project.add_layer(copy, after_id=source.id)
        except ProjectError as e:
            QMessageBox.warning(self, "Duplicate layer", str(e))
            return None
        self.layer_pane.refresh(self.project.layers, selected_id=copy.id)
        if copy.has_color or copy.is_derived:
            self._mark_dirty(copy.id)
        self._mark_modified()
        return copy

    def change_layer_type(self, new_type: LayerType) -> bool:
        """Retype the selected layer. A project still has at most one cut layer."""
        layer = self.selected_layer()
        if layer is None or layer.layer_type == new_type:
            return False
        if new_type == LayerType.CUT:
            if layer.is_derived:
                QMessageBox.warning(
                    self,
                    "Layer type",
                    "A negative layer cannot be a cut layer.",
                )
                self.inspector.set_layer(layer, self._source_names(layer))
                return False
            existing = self.project.cut_layer()
            if existing is not None and existing.id != layer.id:
                QMessageBox.warning(
                    self,
                    "Layer type",
                    "The project already has a cut layer; the board outline "
                    "must be a single contiguous perimeter.",
                )
                self.inspector.set_layer(layer, self._source_names(layer))
                return False
        layer.layer_type = new_type
        self.layer_pane.refresh(self.project.layers, selected_id=layer.id)
        self._mark_dirty(layer.id)
        self._mark_modified()
        return True

    def generate_negative(
        self, source_ids: list[int], layer_type: LayerType = LayerType.COPPER
    ) -> Layer | None:
        """Create a layer covering everything the given layers do not."""
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
        uncolored = [s.name for s in sources if not s.is_derived and not s.has_color]
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
        self.project.add_layer(layer)
        self.layer_pane.refresh(self.project.layers, selected_id=layer.id)
        self._mark_dirty(layer.id)
        self._mark_modified()
        return layer

    def remove_layer(self, layer_id: int) -> None:
        affected = self.project.dependents(layer_id)
        removed = self.project.remove_layer(layer_id)
        for rid in removed:
            self.masks.pop(rid, None)
            self.outlines.pop(rid, None)
            self._dirty_layers.discard(rid)
            self.canvas.remove_layer_items(rid)
        self.layer_pane.refresh(self.project.layers)
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
        self.project.move_layer(layer_id, delta)
        self.layer_pane.refresh(self.project.layers, selected_id=layer_id)
        self._mark_modified()

    def set_layer_visible(self, layer_id: int, visible: bool) -> None:
        layer = self.project.get_layer(layer_id)
        if layer is not None:
            if layer.visible != visible:
                self._mark_modified()
            layer.visible = visible
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

    def _on_base_selected(self):
        self.inspector_stack.setCurrentWidget(self.base_panel)

    def _on_layer_renamed(self, layer_id: int, name: str):
        layer = self.project.get_layer(layer_id)
        if layer is not None and name.strip() and layer.name != name.strip():
            layer.name = name.strip()
            self._mark_modified()

    def _on_layer_edited(self):
        layer = self.selected_layer()
        if layer is not None:
            self.layer_pane.note_layer_changed(layer)
            self._mark_dirty(layer.id)
            self._mark_modified()

    def _on_pixel_picked(self, x: int, y: int):
        if self.image is None:
            return
        r, g, b = (int(v) for v in self.image[y, x, :3])
        self.inspector.apply_color((r, g, b))
        self.inspector.eyedropper_button.setChecked(False)

    # ---- recompute engine ------------------------------------------------------------

    def _mark_dirty(self, *layer_ids: int):
        self._dirty_layers.update(layer_ids)
        self._recompute_timer.start()

    def _recompute_dirty(self):
        if self.image is None:
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

    def recompute_layer(self, layer: Layer) -> None:
        if self.image is None:
            return
        if layer.is_derived:
            sources = [self.masks[s] for s in layer.negative_of if s in self.masks]
            mask = negative_mask(self.image, sources, min_island=layer.min_island)
        elif layer.has_color:
            mask = build_mask(
                self.image,
                layer.color,
                tolerance=layer.tolerance,
                close_radius=layer.close_radius,
                min_island=layer.min_island,
                fill_holes=layer.fill_holes and not layer.is_cut,
            )
        else:
            return
        if layer.inverted:
            mask = invert_mask(self.image, mask, min_island=layer.min_island)
        self.masks[layer.id] = mask
        if layer.is_cut:
            try:
                outline = cut_outline(mask, layer.smoothness, layer.outline_mode)
                self.outlines[layer.id] = outline
                self.canvas.set_layer_outline(layer.id, list(outline.exterior.coords))
                self.statusBar().clearMessage()
            except CutError as e:
                self.outlines.pop(layer.id, None)
                self.canvas.set_layer_outline(layer.id, [])
                self.statusBar().showMessage(f"Cut layer: {e}")
        elif layer.id in self.canvas.outline_items:
            self.outlines.pop(layer.id, None)
            self.canvas.set_layer_outline(layer.id, [])
        self.canvas.set_layer_overlay(layer.id, mask, layer.layer_type.value)
        self.canvas.set_layer_visible(layer.id, layer.visible)

    # ---- project files ------------------------------------------------------------------

    def _sync_project_state(self):
        self.project.base_visible = self.base_visible_action.isChecked()
        self.project.checker_opacity = self.checker_slider.value()
        if self.image_path is not None:
            self.project.image_path = str(self.image_path.absolute())

    def save_project(self, path: str | Path | None = None) -> Path:
        """Write the current project to a .prj file; returns the path written."""
        target = Path(path) if path is not None else self.project_path
        if target is None:
            raise ProjectFileError("No project file path has been chosen yet.")
        self._sync_project_state()
        written = project_io.save_project(target, self.project)
        self.project_path = written
        self._set_unmodified()
        self.statusBar().showMessage(f"Saved {written}")
        return written

    def open_project(self, path: str | Path, image_path: str | Path | None = None) -> None:
        """Load a .prj file, restoring layers, board settings, and view state."""
        path = Path(path)
        project = project_io.load_project(path)
        candidate = Path(image_path) if image_path else (
            Path(project.image_path) if project.image_path else None
        )
        if candidate is None or not candidate.is_file():
            candidate = self._locate_image(project.image_path)
            if candidate is None:
                raise ProjectFileError("The project's source image could not be found.")
        original = load_rgba(candidate)  # may raise before we touch current state

        self.project = project
        self.project_path = path
        self.project.image_path = str(candidate)
        self.masks.clear()
        self.outlines.clear()
        self._dirty_layers.clear()
        self.original_image = original
        self.image_path = candidate

        for action, state in (
            (self.base_visible_action, project.base_visible),
            (self.remove_bg_action, project.background_removed),
        ):
            action.blockSignals(True)
            action.setChecked(state)
            action.blockSignals(False)
        self.checker_slider.blockSignals(True)
        self.checker_slider.setValue(project.checker_opacity)
        self.checker_slider.blockSignals(False)
        self.canvas.set_checker_opacity(project.checker_opacity)

        image = working_image(
            original, project.background_removed, project.background_tolerance
        )
        self._apply_image(image, fit=True)
        first = project.layers[0].id if project.layers else None
        if first is None:
            self.inspector.set_layer(None)
            self.layer_pane.refresh(project.layers)
            self.layer_pane.select_base()
        else:
            self.layer_pane.refresh(project.layers, selected_id=first)
        self.flush_recompute()
        self._set_unmodified()
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
        if settings != self.project.board:
            self.project.board = settings
            self._mark_modified()

    def _on_board_settings(self):
        if self.image is None:
            QMessageBox.information(self, "Board settings", "Open an image first.")
            return
        h, w = self.image.shape[:2]
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
        """Keep a locked board height matched to the current image aspect."""
        board = self.project.board
        if self.image is None or not board.lock_aspect or board.width_mm is None:
            return
        height_px, width_px = self.image.shape[:2]
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
        if self.image is None:
            QMessageBox.information(self, "Export", "Open an image first.")
            return
        self.flush_recompute()
        h, w = self.image.shape[:2]
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
        """Build geometries from current masks and write Gerber files."""
        if self.image is None:
            raise ExportError("No image is loaded.")
        self.flush_recompute()
        h, w = self.image.shape[:2]
        transform = BoardTransform(w, h, width_mm, height_mm)

        geometries = {}
        for layer in self.project.artwork_layers():
            if not layer.visible or not layer.has_color:
                continue
            mask = self.masks.get(layer.id)
            if mask is None:
                continue
            geom = mask_to_geometry(mask)
            if not geom.is_empty:
                geometries[layer.id] = geom

        outline = None
        cut = self.project.cut_layer()
        if cut is not None and cut.visible and cut.has_color:
            outline = self.outlines.get(cut.id)
            if outline is None:
                mask = self.masks.get(cut.id)
                if mask is None:
                    raise ExportError(
                        "The cut layer has no computed mask; pick its color first."
                    )
                outline = cut_outline(mask, cut.smoothness, cut.outline_mode)

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
