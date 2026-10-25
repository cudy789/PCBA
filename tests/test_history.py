"""Undo stack: steps, gestures, and the saved checkpoint."""

from png_to_gerber.history import EditHistory, clone_project, project_signature
from png_to_gerber.model import Layer, LayerType, Project


class _State:
    def __init__(self, value, selection=None):
        self.value = value
        self.selection = selection
        self.selected_layer = None
        self.selected_image = None

    def same_document(self, other) -> bool:
        return self.value == other.value


def test_undo_and_redo_walk_distinct_steps():
    history = EditHistory()
    history.reset(_State(0))
    history.commit(_State(1))
    history.commit(_State(2))
    assert history.can_undo and not history.can_redo

    assert history.undo().value == 1
    assert history.undo().value == 0
    assert history.undo() is None
    assert history.redo().value == 1
    assert history.redo().value == 2
    assert history.redo() is None


def test_a_new_step_drops_the_redo_branch():
    history = EditHistory()
    history.reset(_State(0))
    history.commit(_State(1))
    history.undo()
    history.commit(_State(9))
    assert history.states[history.index].value == 9
    assert not history.can_redo
    assert [state.value for state in history.states] == [0, 9]


def test_identical_documents_do_not_grow_the_stack():
    history = EditHistory()
    history.reset(_State(0))
    assert history.commit(_State(0)) is False
    assert not history.can_undo


def test_a_gesture_is_one_step_and_a_return_to_the_start_is_none():
    history = EditHistory()
    history.reset(_State(0))
    history.commit(_State(1))
    history.commit(_State(2), gesture="slider")
    history.commit(_State(3), gesture="slider")
    history.commit(_State(4), gesture="slider")
    assert [state.value for state in history.states] == [0, 1, 4]

    assert history.undo().value == 1

    history.commit(_State(5), gesture="slider")
    history.commit(_State(1), gesture="slider")
    history.end_gesture()
    assert [state.value for state in history.states] == [0, 1]
    assert history.undo().value == 0


def test_undo_of_an_open_noop_gesture_undoes_the_previous_step():
    history = EditHistory()
    history.reset(_State(0))
    history.commit(_State(1))
    history.commit(_State(2), gesture="slider")
    history.commit(_State(1), gesture="slider")
    assert history.undo().value == 0


def test_saved_checkpoint_and_a_branch_that_discards_it():
    history = EditHistory()
    history.reset(_State("base"))
    history.commit(_State("saved"))
    history.mark_saved()
    history.commit(_State("later"))
    assert not history.at_saved
    history.undo()
    assert history.at_saved

    history.undo()
    history.commit(_State("other"))
    assert not history.at_saved
    assert not history.can_redo
    assert [state.value for state in history.states] == ["base", "other"]


def test_selection_updates_do_not_add_a_step():
    history = EditHistory()
    history.reset(_State(0))
    history.commit(_State(1))
    history.set_selection(7, 3)
    assert not history.can_redo
    assert history.states[history.index].selected_layer == 7
    assert history.states[history.index - 1].selected_layer is None


def test_history_keeps_every_step():
    history = EditHistory()
    history.reset(_State(0))
    for value in range(1, 41):
        history.commit(_State(value))
    assert history.index == 40
    for value in range(39, -1, -1):
        assert history.undo().value == value


def test_project_clone_is_independent_and_signatures_see_edits():
    project = Project()
    image = project.add_image("art.png")
    image.layers.append(Layer("Copper", LayerType.COPPER, has_color=True))
    before = project_signature(project)
    clone = clone_project(project)
    clone.images[0].layers[0].name = "Renamed"
    clone.images[0].layers[0].tolerance = 12
    assert project.images[0].layers[0].name == "Copper"
    assert project_signature(project) == before
    assert project_signature(clone) != before
