"""Tests for particle radius count aggregation (module filename contains a space)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
VERIFY_PATH = REPO / "1_data_processing" / "s1_verify particle_integrity.py"


@pytest.fixture(scope="module")
def verify_mod():
    spec = importlib.util.spec_from_file_location("verify_particle_integrity", VERIFY_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_particle_radius_counts_per_file(tmp_path, verify_mod):
    d = tmp_path / "d"
    d.mkdir()
    pd.DataFrame({"r": [0.1, 0.1, 0.2]}).to_parquet(d / "a.parquet", index=False)
    df = verify_mod.particle_radius_counts_per_file(str(d), size_column="r")
    assert df.shape[0] == 1
    assert df.to_numpy().sum() > 0


def test_particle_radius_counts_empty_raises(tmp_path, verify_mod):
    with pytest.raises(FileNotFoundError):
        verify_mod.particle_radius_counts_per_file(str(tmp_path / "empty"), size_column="r")
