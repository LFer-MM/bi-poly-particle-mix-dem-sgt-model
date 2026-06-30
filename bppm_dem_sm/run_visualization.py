"""Pipeline-integrated plots: cell-grid frame and prediction animation."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import data_io
from .cell_grid import plot_particles_with_grid
from .config import FIGURES_DIR, PipelineConfig

_PLANE_AXES = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}

SMALL_COLOR = "#d62728"
LARGE_COLOR = "#1f77b4"


def _radius_colors(r):
    """Map a bidisperse radius array to small/large category colors."""
    small_r = np.unique(r).min()
    return np.where(r == small_r, SMALL_COLOR, LARGE_COLOR)


def animate_frames(frames_dir, config: PipelineConfig, pattern="frame_*.parquet", save_path=None):
    """Build a 2D scatter animation colored by particle radius."""
    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt

    ax_x, ax_y = _PLANE_AXES[config.plane]
    files = data_io.sorted_frame_files(frames_dir, pattern)[:: config.every_nth_frame]

    df0 = pd.read_parquet(files[0])
    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(ax_x)
    ax.set_ylabel(ax_y)
    ax.set_xlim(-6.2, 6.2)
    ax.set_ylim(-6.2, 6.2)

    sc = ax.scatter(
        df0[ax_x].to_numpy(), df0[ax_y].to_numpy(),
        c=_radius_colors(df0["r"].to_numpy()), s=config.marker_size, alpha=0.75,
    )
    title = ax.set_title(os.path.basename(files[0]))

    def update(i):
        df = pd.read_parquet(files[i])
        sc.set_offsets(np.column_stack([df[ax_x].to_numpy(), df[ax_y].to_numpy()]))
        sc.set_color(_radius_colors(df["r"].to_numpy()))
        title.set_text(os.path.basename(files[i]))
        return sc, title

    anim = FuncAnimation(fig, update, frames=len(files), interval=int(1000 / config.fps), blit=False)

    if save_path:
        anim.save(str(save_path), dpi=140, fps=config.fps)
        print(f"Saved animation: {save_path}")
    elif config.show_plots:
        plt.show()
    return anim


def plot_frame_grid(frame_path, config: PipelineConfig):
    """Render a single frame with the Lacey cell grid overlaid."""
    plot_particles_with_grid(str(frame_path), config.cell_size)


def generate_visualizations(config: PipelineConfig) -> dict:
    """Render the cell-grid frame and the prediction animation."""
    gt_files = data_io.sorted_frame_files(config.data_dir, config.frame_glob)
    if config.show_plots:
        plot_frame_grid(gt_files[0], config)

    save_path = None
    if config.save_figures:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / "pred_animation.mp4"

    anim = animate_frames(config.pred_frames_dir, config, "pred_frame_*.parquet", save_path)
    return {"grid_frame": gt_files[0], "animation": anim}
