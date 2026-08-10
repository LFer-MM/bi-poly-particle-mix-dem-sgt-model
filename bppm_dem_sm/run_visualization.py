"""Pipeline-integrated plots: cell-grid frame and prediction animation."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import data_io
from .cell_grid import plot_particles_with_grid
from .config import INTERIM_DIR, PipelineConfig

_PLANE_AXES = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}
VIZ_DIR = INTERIM_DIR / "figures"

SMALL_COLOR = "#d62728"
LARGE_COLOR = "#1f77b4"


def _radius_colors(r):
    """Map a bidisperse radius array to small/large category colors.

    Args:
        r: 1-D array of particle radii (two distinct values expected).

    Returns:
        numpy.ndarray: Per-particle color codes (small → red, large → blue).
    """
    small_r = np.unique(r).min()
    return np.where(r == small_r, SMALL_COLOR, LARGE_COLOR)


def animate_frames(frames_dir, config: PipelineConfig, pattern="frame_*.parquet", save_path=None):
    """Build a 2D scatter animation colored by particle radius.

    Uses ``config.plane``, ``fps``, ``marker_size``, and ``every_nth_frame``.
    When saving MP4 without ffmpeg, falls back to GIF via Pillow.

    Args:
        frames_dir: Directory of parquet frames to animate.
        config: Visualization settings from ``PipelineConfig``.
        pattern: Glob for frame files (e.g. ``pred_frame_*.parquet``).
        save_path: Optional output path (``.mp4`` or ``.gif``); ``None`` shows
            interactively when ``config.show_plots`` is true.

    Returns:
        tuple: ``(anim, resolved_save_path)`` where ``resolved_save_path`` may
        differ from ``save_path`` if MP4 was rewritten as GIF, or is ``None``
        when not saving.
    """
    from matplotlib.animation import FuncAnimation, writers
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
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Pillow can write GIF/APNG, not MP4; use ffmpeg only when available.
        if save_path.suffix.lower() == ".mp4" and not writers.is_available("ffmpeg"):
            save_path = save_path.with_suffix(".gif")
            print("ffmpeg not available; saving animation as GIF instead.")
        writer = "ffmpeg" if save_path.suffix.lower() == ".mp4" else "pillow"
        anim.save(str(save_path), dpi=140, fps=config.fps, writer=writer)
        print(f"Saved animation: {save_path}")
        if not config.show_plots:
            plt.close(fig)
        return anim, save_path
    elif config.show_plots:
        plt.show()
    return anim, None


def plot_frame_grid(frame_path, config: PipelineConfig, save_path=None, show=True):
    """Render a single frame with the Lacey cell grid overlaid.

    Args:
        frame_path: Parquet frame to scatter-plot.
        config: Supplies ``cell_size`` for the grid overlay.
        save_path: Optional path to save the figure.
        show: Whether to display the figure interactively.

    Returns:
        matplotlib.figure.Figure: Figure from :func:`plot_particles_with_grid`.
    """
    return plot_particles_with_grid(
        str(frame_path),
        config.cell_size,
        save_path=save_path,
        show=show,
    )


def generate_visualizations(config: PipelineConfig) -> dict:
    """Render the cell-grid frame and the prediction animation.

    Plots the first ground-truth frame with the Lacey cell grid and animates
    predicted frames under ``config.pred_frames_dir``.

    Args:
        config: Pipeline settings controlling show/save and animation options.

    Returns:
        dict: Paths and artists (``grid_frame``, optional ``grid_figure``,
        ``animation``, and save paths when ``save_figures`` is true).
    """
    gt_files = data_io.sorted_frame_files(config.data_dir, config.frame_glob)
    artifacts: dict = {"grid_frame": gt_files[0]}

    grid_save = None
    anim_save = None
    if config.save_figures:
        VIZ_DIR.mkdir(parents=True, exist_ok=True)
        grid_save = VIZ_DIR / "cell_grid_frame.png"
        anim_save = VIZ_DIR / "pred_animation.mp4"

    if config.show_plots or config.save_figures:
        artifacts["grid_figure"] = plot_frame_grid(
            gt_files[0],
            config,
            save_path=grid_save,
            show=config.show_plots,
        )

    anim, resolved_anim_path = animate_frames(
        config.pred_frames_dir, config, "pred_frame_*.parquet", anim_save
    )
    artifacts["animation"] = anim
    if config.save_figures:
        artifacts["grid_save_path"] = grid_save
        artifacts["animation_save_path"] = resolved_anim_path
    return artifacts
