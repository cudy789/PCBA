"""Shared fixtures: sys.path setup, offscreen Qt, and a synthetic test image."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

@pytest.fixture(autouse=True)
def _auto_discard_unsaved_prompt(monkeypatch):
    """Never let the "unsaved changes" dialog block a test.

    pytest-qt closes every window at teardown, and a modified window asks what
    to do. Answer "Discard" by default; tests of the prompt itself override this.
    """
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Discard
    )


# Colors used by the synthetic fixture image.
BOARD_COLOR = (240, 240, 235)  # cut layer target
COPPER_COLOR = (30, 60, 200)
MASK_COLOR = (40, 180, 90)
SILK_COLOR = (20, 20, 20)
NEAR_COPPER = (36, 66, 205)  # within tolerance of COPPER_COLOR


@pytest.fixture
def fixture_rgba() -> np.ndarray:
    """A 200x150 RGBA test image.

    - Transparent background.
    - A light 'board' rectangle from (10,10) to (189,139)  -> cut layer.
    - A copper rectangle (30..89, 30..89) with a board-colored hole
      (50..69, 50..69) -> copper layer with an interior hole.
    - A mask rectangle (100..159, 30..59) whose rightmost column alternates
      between the exact color and a near-color to simulate anti-aliasing.
    - A silkscreen rectangle (100..159, 80..109).
    """
    img = np.zeros((150, 200, 4), dtype=np.uint8)

    def fill(x0, y0, x1, y1, rgb):
        img[y0 : y1 + 1, x0 : x1 + 1, :3] = rgb
        img[y0 : y1 + 1, x0 : x1 + 1, 3] = 255

    fill(10, 10, 189, 139, BOARD_COLOR)
    fill(30, 30, 89, 89, COPPER_COLOR)
    fill(50, 50, 69, 69, BOARD_COLOR)  # hole in the copper region
    fill(100, 30, 159, 59, MASK_COLOR)
    # Simulated anti-aliasing: sprinkle near-color pixels along one column so
    # an exact-color match would leave pixel-size gaps there.
    for y in range(30, 60, 2):
        img[y, 158, :3] = (44, 176, 94)  # slightly off the exact mask color
    fill(100, 80, 159, 109, SILK_COLOR)
    return img


@pytest.fixture
def fixture_image_path(tmp_path, fixture_rgba) -> Path:
    from PIL import Image

    path = tmp_path / "fixture.png"
    Image.fromarray(fixture_rgba, "RGBA").save(path)
    return path
