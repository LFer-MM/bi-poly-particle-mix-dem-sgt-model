"""Frame loading and supervised-dataset construction for the RNN surrogate."""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from .config import FEATURE_COLS, ID_COL, TARGET_COLS


def sorted_frame_files(frames_dir, pattern="frame_*.parquet"):
    """Sorted parquet paths under frames_dir matching pattern.

    Args:
        frames_dir: Directory containing frame parquet files.
        pattern: Glob pattern relative to ``frames_dir`` (default
            ``frame_*.parquet``).

    Returns:
        list[str]: Lexicographically sorted matching file paths.
    """
    return sorted(glob.glob(os.path.join(str(frames_dir), pattern)))


def load_frame(path, cols=None):
    """Read one parquet frame sorted by id.

    Args:
        path: Path to a single frame parquet file.
        cols: Optional column subset to read; ``None`` reads all columns.

    Returns:
        pd.DataFrame: Frame rows sorted by particle ``id`` with a reset index.
    """
    return pd.read_parquet(path, columns=cols).sort_values(ID_COL).reset_index(drop=True)


def load_frames_stacked(frames_dir, pattern="frame_*.parquet", feature_cols=None):
    """Load all frames (sorted by id) and stack positions/radius over time.

    Args:
        frames_dir: Directory of parquet frames.
        pattern: Glob for frame files.
        feature_cols: Feature column names; defaults to ``FEATURE_COLS``
            (``x``, ``y``, ``z``, ``r``).

    Returns:
        tuple: ``(pos, rad, base_ids)`` where ``pos`` is ``[T, N, 3]``,
        ``rad`` is ``[T, N, 1]``, and ``base_ids`` is length ``N``.
    """
    feature_cols = feature_cols or FEATURE_COLS
    cols = [ID_COL] + [c for c in feature_cols if c != ID_COL]

    frames = [pd.read_parquet(f, columns=cols).sort_values(ID_COL) for f in sorted_frame_files(frames_dir, pattern)]
    base_ids = frames[0][ID_COL].to_numpy()

    pos = np.stack([df[TARGET_COLS].to_numpy(np.float32) for df in frames])  # [T, N, 3]
    rad = np.stack([df[["r"]].to_numpy(np.float32) for df in frames])  # [T, N, 1]
    return pos, rad, base_ids


def build_supervised_dataset(pos, rad, frames_in):
    """Sliding-window (X, y): frames_in steps of [x,y,z,r] -> next [x,y,z].

    Args:
        pos: Position tensor of shape ``[T, N, 3]``.
        rad: Radius tensor of shape ``[T, N, 1]``.
        frames_in: Number of input frames in each supervised window.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: ``X`` of shape
        ``[(T - frames_in) * N, frames_in, 4]`` and ``y`` of shape
        ``[(T - frames_in) * N, 3]``.
    """
    Xs, Ys = [], []
    for t0 in range(pos.shape[0] - frames_in):
        t1 = t0 + frames_in
        x_seq = np.concatenate([pos[t0:t1], rad[t0:t1]], axis=-1)  # [frames_in, N, 4]
        Xs.append(np.transpose(x_seq, (1, 0, 2)))  # [N, frames_in, 4]
        Ys.append(pos[t1])  # [N, 3]
    return np.concatenate(Xs), np.concatenate(Ys)


def train_test_split(X, y, val_fraction=0.1, seed=0):
    """Shuffle and split arrays into train/validation subsets.

    Args:
        X: Feature array (first axis is samples).
        y: Target array aligned with ``X``.
        val_fraction: Fraction of samples reserved for validation.
        seed: RNG seed for the shuffle.

    Returns:
        tuple: ``(X_train, y_train, X_val, y_val)``.
    """
    idx = np.arange(X.shape[0])
    np.random.default_rng(seed).shuffle(idx)
    split = int((1.0 - val_fraction) * len(idx))
    tr, te = idx[:split], idx[split:]
    return X[tr], y[tr], X[te], y[te]
