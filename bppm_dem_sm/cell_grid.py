"""Single-frame particle scatter with an overlaid square cell grid."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_particles_with_grid(frame_path, cell_size, use_equal_aspect=True):
    """Scatter the two species on XY with a square grid overlay."""
    df = pd.read_parquet(frame_path)
    x, y, r = df["x"].values, df["y"].values, df["r"].values

    r_small, r_large = np.unique(r).min(), np.unique(r).max()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x[r == r_small], y[r == r_small], s=5, alpha=0.7, c="red", label="Rock")
    ax.scatter(x[r == r_large], y[r == r_large], s=5, alpha=0.7, c="blue", label="Ball")

    pad = cell_size * 0.5
    xmin, xmax = -6 - pad, 6 + pad
    ymin, ymax = -6 - pad, 6 + pad
    for xl in np.arange(xmin, xmax + cell_size, cell_size):
        ax.axvline(xl, linewidth=0.8, color="black", alpha=0.8)
    for yl in np.arange(ymin, ymax + cell_size, cell_size):
        ax.axhline(yl, linewidth=0.8, color="black", alpha=0.8)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if use_equal_aspect:
        ax.set_aspect("equal")
    ax.set_title(f"Particle Positions with Grid (cell size = {cell_size} m)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend()

    plt.tight_layout()
    plt.show()
