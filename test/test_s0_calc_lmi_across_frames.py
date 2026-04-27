"""Tests for Lacey mixing index utilities."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import s0_calc_lmi_across_frames as lacey


def test_extract_frame_index():
    idx = lacey.extract_frame_index(r"C:\data\frame_00042.parquet", lacey.GT_FRAME_RE)
    assert idx == 42


def test_detect_tracer_radius():
    r = np.array([0.1, 0.1, 0.2, 0.2, 0.2])
    assert lacey.detect_tracer_radius(r) == pytest.approx(0.2)


def test_detect_tracer_radius_raises():
    with pytest.raises(ValueError):
        lacey.detect_tracer_radius(np.array([0.1, 0.1, 0.1]))


def test_lacey_index_well_mixed():
    n = 8000
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": rng.uniform(0, 4, n),
            "y": rng.uniform(0, 4, n),
            "z": rng.uniform(0, 4, n),
            "r": rng.choice([0.1, 0.2], size=n),
        }
    )
    M, n_cells, p_glob, mean_n = lacey.lacey_index_for_frame(
        df, cell_size=0.5, tracer_radius=0.2, min_particles_per_cell=5
    )
    assert n_cells > 0
    assert 0.0 <= M <= 1.0 or np.isnan(M)
    assert 0.0 < p_glob < 1.0
