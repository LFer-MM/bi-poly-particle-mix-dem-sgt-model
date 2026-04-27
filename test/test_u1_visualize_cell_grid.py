"""Tests for cell-grid visualization (no display)."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

import u1_visualize_cell_grid as vis


def test_plot_particles_with_grid_runs(tmp_path, monkeypatch):
    p = tmp_path / "f.parquet"
    n = 20
    rng = np.random.default_rng(1)
    pd.DataFrame(
        {
            "x": rng.uniform(-1, 1, n),
            "y": rng.uniform(-1, 1, n),
            "r": np.where(np.arange(n) % 2 == 0, 0.1, 0.2),
        }
    ).to_parquet(p, index=False)

    with patch("matplotlib.pyplot.show"):
        vis.plot_particles_with_grid(str(p), cell_size=0.5, use_equal_aspect=True)
