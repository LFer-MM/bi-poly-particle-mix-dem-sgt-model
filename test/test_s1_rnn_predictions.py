"""Tests for RNN prediction helpers."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import s1_rnn_predictions as pred


def test_sorted_frame_files(tmp_path):
    (tmp_path / "frame_00002.parquet").write_bytes(b"")
    (tmp_path / "frame_00001.parquet").write_bytes(b"")
    files = pred.sorted_frame_files(str(tmp_path), "frame_*.parquet")
    assert len(files) == 2
    assert "00001" in files[0]


def test_sorted_frame_files_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        pred.sorted_frame_files(str(tmp_path), "*.none")


def test_infer_xyz_shapes():
    y = np.random.randn(5, 3).astype(np.float32)
    assert pred.infer_xyz(y).shape == (5, 3)
    y2 = np.random.randn(5, 1, 4).astype(np.float32)
    out = pred.infer_xyz(y2)
    assert out.shape == (5, 3)
    assert np.allclose(out, y2[:, 0, :3])


def test_infer_xyz_bad_shape():
    with pytest.raises(ValueError):
        pred.infer_xyz(np.zeros((3,)))


def test_align_to_base_ids_ok():
    base = np.array([1, 2], dtype=np.int64)
    df = pd.DataFrame({"id": [1, 2], "x": [0.0, 1.0]})
    out = pred.align_to_base_ids(df, base)
    assert list(out["id"]) == [1, 2]


def test_align_to_base_ids_mismatch():
    base = np.array([1, 2])
    df = pd.DataFrame({"id": [1, 3], "x": [0.0, 1.0]})
    with pytest.raises(ValueError):
        pred.align_to_base_ids(df, base)


def test_load_frame_requires_id(tmp_path):
    p = os.path.join(tmp_path, "a.parquet")
    pd.DataFrame({"x": [1.0]}).to_parquet(p, index=False)
    with pytest.raises(ValueError):
        pred.load_frame(p, cols=["x"])
