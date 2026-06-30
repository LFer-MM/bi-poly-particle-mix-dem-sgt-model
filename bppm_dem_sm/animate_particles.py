"""Standalone 2D particle animation from a directory of parquet frames.

For the pipeline-integrated variant see
:func:`bppm_dem_sm.run_visualization.animate_frames`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LARGE_COLOR = "#1f77b4"
SMALL_COLOR = "#d62728"

_PLANE_AXES = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}


def radius_colors(r):
    """Map a bidisperse radius array to small/large category colors."""
    small_r = np.unique(r).min()
    return np.where(r == small_r, SMALL_COLOR, LARGE_COLOR)


def animate_particles(
    frames_dir,
    glob_pattern="frame_*.parquet",
    plane="xy",
    every_nth_frame=1,
    fps=30,
    marker_size=4.0,
    save_path=None,
):
    """Build a 2D scatter animation colored by particle radius."""
    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt

    ax_x, ax_y = _PLANE_AXES[plane]
    files = sorted(Path(frames_dir).glob(glob_pattern))[::every_nth_frame]
    df0 = pd.read_parquet(files[0])

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(ax_x)
    ax.set_ylabel(ax_y)
    ax.set_xlim(-6.2, 6.2)
    ax.set_ylim(-6.2, 6.2)

    sc = ax.scatter(
        df0[ax_x].to_numpy(), df0[ax_y].to_numpy(),
        c=radius_colors(df0["r"].to_numpy()), s=marker_size, alpha=0.75,
    )
    title = ax.set_title(files[0].name)

    def update(i):
        df = pd.read_parquet(files[i])
        sc.set_offsets(np.column_stack([df[ax_x].to_numpy(), df[ax_y].to_numpy()]))
        sc.set_color(radius_colors(df["r"].to_numpy()))
        title.set_text(files[i].name)
        return sc, title

    anim = FuncAnimation(fig, update, frames=len(files), interval=int(1000 / fps), blit=False)

    if save_path:
        anim.save(str(save_path), dpi=140, fps=fps)
        print(f"Saved MP4: {save_path}")
    else:
        plt.show()
    return anim
