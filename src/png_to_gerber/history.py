"""Undo and redo snapshots of a project.

A snapshot is anything with ``same_document``. Equality ignores view-only
details such as which row is selected. One slider drag or paint stroke shares
a gesture name so those updates replace a single step instead of stacking.
"""

from __future__ import annotations

import copy

from .model import BaseImage, Layer, Project


class DocumentState:
    """One restorable project, plus the file pixels and the selection then current."""

    def __init__(
        self,
        project: Project,
        originals: dict[int, object],
        selected_layer: int | None,
        selected_image: int | None,
    ):
        self.project = project
        # File pixels are replaced, never edited in place, so snapshots share them.
        self.originals = originals
        self.selected_layer = selected_layer
        self.selected_image = selected_image

    def same_document(self, other: DocumentState) -> bool:
        return project_signature(self.project) == project_signature(other.project) and (
            _originals_signature(self.originals) == _originals_signature(other.originals)
        )


def clone_project(project: Project) -> Project:
    """Deep copy so later edits do not change a snapshot."""
    return copy.deepcopy(project)


def capture_state(
    project: Project,
    originals: dict[int, object],
    selected_layer: int | None,
    selected_image: int | None,
) -> DocumentState:
    return DocumentState(
        clone_project(project),
        dict(originals),
        selected_layer,
        selected_image,
    )


def project_signature(project: Project):
    """Comparable record of everything the project file would remember."""
    board = project.board
    return (
        int(project.checker_opacity),
        (
            board.width_mm,
            board.height_mm,
            bool(board.lock_aspect),
            board.output_dir,
            board.base_name,
        ),
        tuple(_image_signature(image) for image in project.images),
    )


def _image_signature(image: BaseImage):
    return (
        image.id,
        image.name,
        image.path,
        bool(image.background_removed),
        int(image.background_tolerance),
        bool(image.flatten_enabled),
        int(image.flatten_count),
        tuple(int(value) for value in image.flatten_cutoffs),
        tuple(_color_signature(color) for color in image.flatten_colors),
        int(image.flatten_smooth),
        bool(image.visible),
        tuple(tuple(entry) for entry in image.paint),
        tuple(_layer_signature(layer) for layer in image.layers),
    )


def _color_signature(color):
    if color is None:
        return None
    return (int(color[0]), int(color[1]), int(color[2]))


def _layer_signature(layer: Layer):
    return (
        layer.id,
        layer.name,
        layer.layer_type,
        layer.side,
        tuple(int(channel) for channel in layer.color),
        int(layer.tolerance),
        int(layer.close_radius),
        int(layer.min_island),
        bool(layer.fill_holes),
        int(layer.smoothness),
        layer.outline_mode,
        bool(layer.visible),
        bool(layer.has_color),
        tuple(int(source) for source in layer.negative_of),
        bool(layer.inverted),
        tuple(tuple(int(part) for part in run) for run in layer.manual),
    )


def _originals_signature(originals: dict[int, object]):
    return tuple((image_id, id(originals[image_id])) for image_id in sorted(originals))


class EditHistory:
    """Unlimited undo stack. ``saved_index`` is the clean checkpoint."""

    def __init__(self):
        self.states: list = []
        self.index = -1
        self.saved_index = -1
        self.gesture: str | None = None
        self._tip_is_gesture = False

    def reset(self, state) -> None:
        """Replace the stack with one clean state. Used when a document is opened."""
        self.states = [state]
        self.index = 0
        self.saved_index = 0
        self.gesture = None
        self._tip_is_gesture = False

    @property
    def can_undo(self) -> bool:
        return self.index > 0

    @property
    def can_redo(self) -> bool:
        return self.index + 1 < len(self.states)

    @property
    def at_saved(self) -> bool:
        return self.index == self.saved_index

    def mark_saved(self) -> None:
        self.saved_index = self.index

    def set_selection(self, layer_id: int | None, image_id: int | None) -> None:
        """Remember the current row without creating a step."""
        if not self.states:
            return
        state = self.states[self.index]
        state.selected_layer = layer_id
        state.selected_image = image_id

    def commit(self, state, *, gesture: str | None = None, saved: bool = False) -> bool:
        """Append ``state``, or replace the open gesture's step.

        Returns True when the stored document changed. An identical document
        does not grow the stack. ``saved`` marks the resulting step clean.
        """
        if gesture != self.gesture:
            self.end_gesture()
        if (
            gesture is not None
            and self._tip_is_gesture
            and gesture == self.gesture
            and self.index >= 0
        ):
            changed = not self.states[self.index].same_document(state)
            self.states[self.index] = state
            self._drop_redo()
            if saved:
                self.saved_index = self.index
            return changed
        if self.index >= 0 and self.states[self.index].same_document(state):
            self.states[self.index] = state
            self.gesture = None
            self._tip_is_gesture = False
            if saved:
                self.saved_index = self.index
            return False
        self._drop_redo()
        self.states.append(state)
        self.index += 1
        self.gesture = gesture
        self._tip_is_gesture = gesture is not None
        if saved:
            self.saved_index = self.index
        return True

    def _drop_redo(self) -> None:
        """Forget states past the current one. A discarded save stays unsaved."""
        if self.saved_index > self.index:
            self.saved_index = -1
        del self.states[self.index + 1 :]

    def end_gesture(self) -> bool:
        """Close the open gesture. A drag that lands where it started is dropped.

        Returns True when that empty step was removed.
        """
        if not self._tip_is_gesture:
            self.gesture = None
            return False
        self._tip_is_gesture = False
        self.gesture = None
        if self.index > 0 and self.states[self.index].same_document(self.states[self.index - 1]):
            dropped = self.index
            self.states.pop()
            self.index -= 1
            if self.saved_index == dropped:
                self.saved_index = self.index
            return True
        return False

    def undo(self):
        """Step back one stored document. A no-op gesture is not a step."""
        self.end_gesture()
        if self.index <= 0:
            return None
        self.index -= 1
        return self.states[self.index]

    def redo(self):
        self.end_gesture()
        if not self.can_redo:
            return None
        self.index += 1
        return self.states[self.index]
