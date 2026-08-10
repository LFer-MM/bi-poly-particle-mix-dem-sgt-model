"""Particle-size consistency checks across a directory of Parquet frames."""

from __future__ import annotations

import glob
import os

import pandas as pd


def particle_radius_counts_per_file(folder_path, size_column="r"):
    """Stack the value counts of size_column per parquet file.

    Args:
        folder_path: Directory of ``*.parquet`` frames.
        size_column: Column holding particle radius (default ``"r"``).

    Returns:
        pd.DataFrame: Rows are files, columns are distinct radius values,
        cells are particle counts (missing filled with 0).
    """
    all_counts = []
    for file in glob.glob(os.path.join(folder_path, "*.parquet")):
        counts = pd.read_parquet(file)[size_column].value_counts()
        counts.name = os.path.basename(file)
        all_counts.append(counts)
    return pd.DataFrame(all_counts).fillna(0)


def report_particle_integrity(folder_path, size_column="r"):
    """Print counts/mean/std of radii across frames as an integrity check.

    A healthy bidisperse DEM export should have near-zero std of counts per
    size class across frames (same inventory every frame).

    Args:
        folder_path: Directory of ``*.parquet`` frames.
        size_column: Column holding particle radius (default ``"r"``).

    Returns:
        pd.DataFrame: Per-file radius count table from
        :func:`particle_radius_counts_per_file`.
    """
    counts_df = particle_radius_counts_per_file(folder_path, size_column)

    print("\nCounts per file:\n")
    print(counts_df)
    print("\nAverage particle count per size (per file):\n")
    print(counts_df.mean())
    print("\nStandard deviation per size (should be ~0 if identical):\n")
    print(counts_df.std())
    return counts_df
