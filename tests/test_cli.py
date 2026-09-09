"""Tests for src/cli.py: argument parsing and wiring into run_batch."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from skimage import draw, io

from src import cli


def _make_plate_image() -> np.ndarray:
    """A minimal synthetic plate photograph: dish + colony, no real data needed."""
    shape = (300, 300)
    image = np.full((*shape, 3), 40, dtype=np.uint8)
    rr, cc = draw.disk((150, 150), 130, shape=shape)
    image[rr, cc] = 190
    rr, cc = draw.disk((150, 150), 50, shape=shape)
    image[rr, cc] = 100
    return image


@pytest.fixture
def fake_run_batch(monkeypatch):
    """Replace cli.run_batch with a stub that records how it was called."""
    captured = {}

    def _fake(input_dir, output_dir, dish_diameter_mm, segment_kwargs):
        captured["input_dir"] = input_dir
        captured["output_dir"] = output_dir
        captured["dish_diameter_mm"] = dish_diameter_mm
        captured["segment_kwargs"] = segment_kwargs
        return pd.DataFrame({"filename": ["a.png"], "warning": [""]})

    monkeypatch.setattr(cli, "run_batch", _fake)
    return captured


def _run_cli(monkeypatch, args: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["cli", *args])
    cli.main()


def test_cli_default_flags_produce_expected_segment_kwargs(monkeypatch, fake_run_batch):
    _run_cli(monkeypatch, ["--input", "in", "--output", "out"])

    assert fake_run_batch["dish_diameter_mm"] == 90.0
    assert fake_run_batch["segment_kwargs"] == {
        "colony_darker_than_background": True,
        "threshold_offset": 0.0,
        "min_object_size": 500,
        "morph_kernel_size": 5,
    }


def test_cli_colony_brighter_flag_inverts_darker_than_background(monkeypatch, fake_run_batch):
    _run_cli(
        monkeypatch, ["--input", "in", "--output", "out", "--colony-brighter-than-background"]
    )

    assert fake_run_batch["segment_kwargs"]["colony_darker_than_background"] is False


def test_cli_forwards_custom_tuning_flags(monkeypatch, fake_run_batch):
    _run_cli(
        monkeypatch,
        [
            "--input", "in",
            "--output", "out",
            "--dish-diameter", "60",
            "--threshold-offset", "0.1",
            "--min-object-size", "800",
            "--morph-kernel-size", "7",
        ],
    )

    assert fake_run_batch["dish_diameter_mm"] == 60.0
    assert fake_run_batch["segment_kwargs"]["threshold_offset"] == 0.1
    assert fake_run_batch["segment_kwargs"]["min_object_size"] == 800
    assert fake_run_batch["segment_kwargs"]["morph_kernel_size"] == 7


def test_cli_failure_count_ignores_no_colony_detected_warnings(monkeypatch, capsys):
    df = pd.DataFrame(
        {
            "filename": ["a.png", "b.png", "c.png", "d.png"],
            "warning": [
                "Could not read image: bad file",
                "Processing failed: boom",
                "No colony detected",
                "",
            ],
        }
    )
    monkeypatch.setattr(cli, "run_batch", lambda *a, **k: df)
    _run_cli(monkeypatch, ["--input", "in", "--output", "out"])

    out = capsys.readouterr().out
    # Only the two genuine IO/processing failures should be counted --
    # "No colony detected" is a warning about the result, not a failed run.
    assert "Processed 4 image(s); 2 failed." in out


def test_cli_end_to_end_writes_measurements_csv(tmp_path, monkeypatch, capsys):
    plates_dir = tmp_path / "plates"
    plates_dir.mkdir()
    io.imsave(plates_dir / "plate01.png", _make_plate_image())
    output_dir = tmp_path / "results"

    _run_cli(monkeypatch, ["--input", str(plates_dir), "--output", str(output_dir)])

    assert (output_dir / "measurements.csv").exists()
    out = capsys.readouterr().out
    assert f"Results written to {output_dir}/measurements.csv" in out
