"""Main window: wires the canvas, layer pane, inspector, project files, and export."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import UnidentifiedImageError
from shapely.ops import unary_union
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
from ..image_io import canvas_size as shared_canvas_size
from ..image_io import dominant_colors, load_rgba, pad_top_left
from ..masking import build_mask, invert_mask, negative_mask
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
        self._comparing: set[int] = set()  # image ids showing the pre-flatten canvas
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
        toolbar.setMovable(False)
        self.checker_label = QLabel("Checker opacity", self)
        toolbar.addWidget(self.checker_label)
        toolbar.addWidget(self.checker_slider)

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
        image = self.active_image()
        if image is None:
            image = self.project.add_image(name=path.name, path=str(path))
        else:
            image.name = path.name
            image.path = str(path)
            image.background_removed = False
            reset_flatten(image)
            self._comparing.discard(image.id)
        self._originals[image.id] = original
        self._working[image.id] = original
        self._palettes.pop(image.id, None)
        self._set_checked(self.remove_bg_action, False)
        self._refresh_canvas(fit=True)
        self._refresh_layers(selected_image_id=image.id)
        if self.project.layers or self.project_path is not None:
            self._mark_modified()  # a bare image with no work in it isn't worth saving
        self.statusBar().showMessage(
            f"Loaded {path.name} ({original.shape[1]} x {original.shape[0]} px)"
        )

    def add_image(self, path: str | Path):
        """Open another base image. Its layers will be nested under it."""
        original = load_rgba(path)
        path = Path(path)
        image = self.project.add_image(name=path.name, path=str(path))
        self._originals[image.id] = original
        self._working[image.id] = original
        if len(self.project.images) > 1 or self.project.layers or self.project_path is not None:
            self._mark_modified()
        self._refresh_canvas(fit=True)
        self._refresh_layers(selected_image_id=image.id)
        self.statusBar().showMessage(
            f"Added {path.name} ({original.shape[1]} x {original.shape[0]} px)"
        )
        return image

    def remove_image(self, image_id: int) -> None:
        """Remove one base image and the layers nested under it."""
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
        if removed or image_id is not None:
            self._mark_modified()

    def _refresh_canvas(self, fit: bool) -> None:
        """Redraw every base image on the shared top-left canvas."""
        pairs = [
            (image.id, self._pixels_for_canvas(image))
            for image in self.project.images
            if image.id in self._working
        ]
        self.canvas.set_bases(pairs, fit=fit)
        for image in self.project.images:
            self.canvas.set_base_visible(image.visible, image.id)
        self._sync_locked_board_height()
        self._sync_base_controls()
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
        return working_image(
            original,
            image.background_removed,
            image.background_tolerance,
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

        Compare-original shows the post-background image while layers keep
        masking the flattened working image.
        """
        working = self._working[image.id]
        if image.id in self._comparing and image.flatten_enabled:
            prepared = self._prepared_image(image)
            if prepared is not None and prepared.shape[:2] == working.shape[:2]:
                return prepared
        return working

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
        prepared = self._prepared_image(image)
        if prepared is None:
            grays = [(30 + index * 18, 30 + index * 18, 30 + index * 18) for index in range(count)]
            return cutoffs, grays, np.zeros(256, dtype=np.int64)
        return cutoffs, palette_for(prepared, cutoffs, image.flatten_colors), histogram(prepared)

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
        image = self.project.get_image(image_id)
        if image is None:
            return
        if image.visible != visible:
            image.visible = visible
            self._mark_modified()
        self.canvas.set_base_visible(visible, image_id)
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
        owner = self.project.image_of(source.id)
        try:
            self.project.add_layer(
                copy, after_id=source.id, image_id=owner.id if owner is not None else None
            )
        except ProjectError as e:
            QMessageBox.warning(self, "Duplicate layer", str(e))
            return None
        self._refresh_layers(selected_id=copy.id)
        if copy.has_color or copy.is_derived:
            self._mark_dirty(copy.id)
        self._mark_modified()
        return copy

    def change_layer_type(self, new_type: LayerType) -> bool:
        """Retype the selected layer. Each cut layer stays one closed perimeter."""
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
        image = self.active_image()
        self.project.add_layer(layer, image_id=image.id if image is not None else None)
        self._refresh_layers(selected_id=layer.id)
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
        self.project.move_layer(layer_id, delta)
        self._refresh_layers(selected_id=layer_id)
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
        parent = self.project.image_of(layer_id) if layer is not None else None
        if parent is not None and parent.id in self._working:
            self.inspector.set_dominant_colors(self._colors_for(parent.id))

    def _on_base_selected(self, image_id=None):
        self.inspector_stack.setCurrentWidget(self.base_panel)
        self._sync_base_controls()

    def _on_layer_renamed(self, layer_id: int, name: str):
        layer = self.project.get_layer(layer_id)
        if layer is not None and name.strip() and layer.name != name.strip():
            layer.name = name.strip()
            self.layer_pane.show_layer_name(layer.id, layer.name)
            self._mark_modified()

    def _on_layer_edited(self):
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
        else:
            return
        self.masks[layer.id] = mask
        if layer.is_cut:
            try:
                outline = cut_outline(
                    native if native is not None else mask,
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
        self._set_unmodified()
        self.statusBar().showMessage(f"Saved {written}")
        return written

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

        self.project = project
        self.project_path = path
        self.masks.clear()
        self.outlines.clear()
        self._dirty_layers.clear()
        self._originals.clear()
        self._working.clear()
        self._palettes.clear()
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
            if not layer.visible or not layer.has_color:
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
            if not cut.visible or not cut.has_color:
                continue
            outline = self.outlines.get(cut.id)
            if outline is None:
                raise ExportError(
                    f"{cut.name} has no computed outline; pick its color first."
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
                raise ExportError(f"{layer.name} has no mask yet. Pick its color first.")
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
