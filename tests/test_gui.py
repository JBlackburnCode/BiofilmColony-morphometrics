"""Tests for src/gui.py: parameter wiring and per-image override state.

Exercises ColonyReviewApp's logic directly (bypassing the file-open dialog by
setting image_paths/input_dir as open_folder would) against a real Tk root --
Tkinter widgets need one to exist, but no window is ever shown or mainloop'd.
"""

import tkinter as tk
from pathlib import Path

import numpy as np
import pytest
from skimage import draw, io

from src.calibrate import CalibrationResult
from src.gui import (
    DEFAULT_DISH_DIAMETER_MM,
    DEFAULT_MIN_OBJECT_SIZE,
    DEFAULT_THRESHOLD_OFFSET,
    ColonyReviewApp,
)


def _make_plate_image() -> np.ndarray:
    """A minimal synthetic plate photograph: dish + colony, no real data needed."""
    shape = (300, 300)
    image = np.full((*shape, 3), 40, dtype=np.uint8)
    rr, cc = draw.disk((150, 150), 130, shape=shape)
    image[rr, cc] = 190
    rr, cc = draw.disk((150, 150), 50, shape=shape)
    image[rr, cc] = 100
    return image


@pytest.fixture(scope="module")
def root():
    # Module-scoped: repeatedly creating/destroying Tk() roots in quick
    # succession is flaky on this machine's Tk install (intermittent
    # "couldn't read file ...msgs/en_gb.msg" from Tcl's msgcat, even though
    # the file exists on disk) -- one root shared across this file's tests
    # avoids that churn.
    tk_root = tk.Tk()
    tk_root.withdraw()
    yield tk_root
    tk_root.destroy()


@pytest.fixture(scope="module")
def app(root):
    # Module-scoped for the same reason as `root`: building a fresh
    # ColonyReviewApp (and its widget tree) per test was itself enough Tk/Tcl
    # I/O to trigger the same intermittent TclError. One app is built once
    # and reset between tests instead.
    return ColonyReviewApp(root)


@pytest.fixture(autouse=True)
def _reset_app_state(app):
    app.image_paths = []
    app.current_index = -1
    app.overrides = {}
    app.threshold_offset_var.set(DEFAULT_THRESHOLD_OFFSET)
    app.min_object_size_var.set(DEFAULT_MIN_OBJECT_SIZE)
    app.dish_diameter_var.set(DEFAULT_DISH_DIAMETER_MM)
    app.colony_brighter_var.set(False)
    app.status_label.configure(text="")


@pytest.fixture
def plate_paths(tmp_path) -> list[Path]:
    paths = [tmp_path / "plate01.png", tmp_path / "plate02.png"]
    for path in paths:
        io.imsave(path, _make_plate_image())
    return paths


def test_current_segment_kwargs_reflects_slider_values(app):
    app.threshold_offset_var.set(0.05)
    app.min_object_size_var.set(800)
    app.colony_brighter_var.set(True)

    assert app._current_segment_kwargs() == {
        "threshold_offset": 0.05,
        "min_object_size": 800,
        "colony_darker_than_background": False,
    }


def test_accept_parameters_stores_override_for_current_image(app, plate_paths):
    app.image_paths = plate_paths
    app.current_index = 1
    app.threshold_offset_var.set(0.1)
    app.min_object_size_var.set(900)

    app.accept_parameters()

    assert app.overrides[plate_paths[1].name] == app._current_segment_kwargs()
    assert plate_paths[1].name in app.status_label.cget("text")


def test_load_current_image_applies_saved_override(app, plate_paths):
    app.image_paths = plate_paths
    app.overrides = {
        plate_paths[1].name: {
            "threshold_offset": 0.2,
            "min_object_size": 1200,
            "colony_darker_than_background": False,
        }
    }

    app.current_index = 1
    app._load_current_image()

    assert app.threshold_offset_var.get() == pytest.approx(0.2)
    assert app.min_object_size_var.get() == 1200
    assert app.colony_brighter_var.get() is True


def test_load_current_image_leaves_sliders_unchanged_without_override(app, plate_paths):
    app.image_paths = plate_paths
    app.threshold_offset_var.set(0.15)

    app.current_index = 0
    app._load_current_image()

    # No saved override for plate01 -- the slider keeps its current value
    # rather than resetting to a default.
    assert app.threshold_offset_var.get() == pytest.approx(0.15)


def test_nav_buttons_disabled_at_start_and_end(app, plate_paths):
    app.image_paths = plate_paths

    app.current_index = 0
    app._update_nav_buttons()
    assert str(app.prev_button.cget("state")) == "disabled"
    assert str(app.next_button.cget("state")) == "normal"

    app.current_index = len(plate_paths) - 1
    app._update_nav_buttons()
    assert str(app.prev_button.cget("state")) == "normal"
    assert str(app.next_button.cget("state")) == "disabled"


def test_set_controls_enabled_toggles_widget_state(app):
    app._set_controls_enabled(True)
    assert str(app.threshold_scale.cget("state")) == "normal"

    app._set_controls_enabled(False)
    assert str(app.threshold_scale.cget("state")) == "disabled"


def test_update_metrics_panel_shows_pixel_units_when_uncalibrated(app):
    metrics = {
        "area": 123.456,
        "perimeter": 45.6,
        "equivalent_diameter": 12.3,
        "circularity": 0.789,
        "solidity": 0.912,
        "texture_contrast": 1.23,
        "texture_entropy": 2.34,
    }
    cal = CalibrationResult(
        mm_per_pixel=None,
        dish_center=None,
        dish_radius_px=None,
        calibrated=False,
        warning="No dish detected",
        source=None,
    )

    app._update_metrics_panel(metrics, cal)

    text = app.metrics_text.get("1.0", tk.END)
    assert "Area: 123.5 px^2" in text
    assert "Calibrated: no (px units)" in text
    assert "Warning: No dish detected" in text
