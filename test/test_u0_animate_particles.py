"""Tests for animation helper functions (no display)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import u0_animate_particles as anim


def test_project_xy():
    anim.PLANE = "xy"
    df = pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]})
    x, y, labels = anim.project(df)
    assert np.allclose(x, [1.0]) and np.allclose(y, [2.0])
    assert labels == ("x", "y")


def test_color_values_r():
    anim.COLOR_BY = "r"
    df = pd.DataFrame({"r": [0.1, 0.2]})
    c = anim.color_values(df)
    assert len(c) == 2


def test_load_frame_parquet(tmp_path: Path):
    p = tmp_path / "f.parquet"
    pd.DataFrame({"id": [1], "x": [0.0], "y": [0.0], "z": [0.0], "r": [0.1]}).to_parquet(p, index=False)
    df = anim.load_frame_parquet(p)
    assert set(df.columns) >= {"id", "x", "y", "z", "r"}
