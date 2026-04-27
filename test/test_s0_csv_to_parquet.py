"""Tests for CSV→Parquet conversion helpers."""
from __future__ import annotations

import os

import pandas as pd

from s0_csv_to_parquet import convert_csv_file_to_parquet, convert_folder_csv_to_parquet


def test_convert_csv_file_to_parquet(tmp_path):
    csv_p = tmp_path / "a.csv"
    pq_p = tmp_path / "a.parquet"
    pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}).to_csv(csv_p, index=False)
    convert_csv_file_to_parquet(str(csv_p), str(pq_p))
    df = pd.read_parquet(pq_p)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_convert_folder_csv_to_parquet(tmp_path):
    inp = tmp_path / "in"
    out = tmp_path / "out"
    inp.mkdir()
    pd.DataFrame({"x": [1]}).to_csv(inp / "f.csv", index=False)
    convert_folder_csv_to_parquet(str(inp), str(out))
    assert os.path.isfile(out / "f.parquet")
