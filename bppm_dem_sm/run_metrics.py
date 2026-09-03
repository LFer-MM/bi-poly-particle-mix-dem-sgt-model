"""Lacey mixing-index over directories of ground-truth and predicted frames."""

from __future__ import annotations

import os

import pandas as pd

from . import data_io
from . import lacey_mixing_index as lacey
from .config import FIGURES_DIR, PipelineConfig
from .lacey_mixing_index import GT_FRAME_RE, PRED_FRAME_RE


def compute_lacey_over_dir(frames_dir, pattern, frame_re, tracer_r, config, out_name, label):
    """Compute the Lacey index per frame in a directory; save a summary parquet.

    Args:
        frames_dir: Directory of parquet frames.
        pattern: Glob for frame files.
        frame_re: Regex used by :func:`extract_frame_index`.
        tracer_r: Tracer (large) particle radius.
        config: Supplies ``metrics.cell_size``, ``metrics.min_particles_per_cell``,
            and ``metrics.metrics_dt``.
        out_name: Filename for the summary parquet written into ``frames_dir``.
        label: Short label for log messages (e.g. ``"GT"``, ``"PRED"``).

    Returns:
        pd.DataFrame: Per-frame Lacey summary sorted by ``frame``, with columns
        ``frame``, ``time``, ``lacey``, ``n_cells_used``,
        ``tracer_fraction_global``, ``mean_particles_per_cell``.
    """
    rows = []
    for pth in data_io.sorted_frame_files(frames_dir, pattern):
        frame_idx = lacey.extract_frame_index(pth, frame_re)
        df = pd.read_parquet(pth)
        M, n_cells, p_global, mean_n = lacey.lacey_index_for_frame(
            df, config.metrics.cell_size, tracer_r, config.metrics.min_particles_per_cell
        )
        rows.append(
            {
                "frame": frame_idx,
                "time": frame_idx * config.metrics.metrics_dt,
                "lacey": M,
                "n_cells_used": n_cells,
                "tracer_fraction_global": p_global,
                "mean_particles_per_cell": mean_n,
            }
        )

    out = pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)
    out.to_parquet(os.path.join(str(frames_dir), out_name), index=False)
    print(f"[{label}] Saved Lacey summary ({len(out)} frames)")
    return out


def compute_metrics(config: PipelineConfig) -> dict[str, pd.DataFrame]:
    """Compute the Lacey index for ground-truth and predicted frames.

    Detects the tracer radius from the first ground-truth frame, then runs
    :func:`compute_lacey_over_dir` on DEM frames and, when present, on
    predicted frames under ``config.prediction.pred_frames_dir``.

    Args:
        config: Pipeline settings for data paths and Lacey cell parameters.

    Returns:
        dict[str, pd.DataFrame]: ``"gt"`` summary always; ``"pred"`` when
        predicted frames exist.
    """
    gt_paths = data_io.sorted_frame_files(config.data_dir, config.frame_glob)
    tracer_r = lacey.detect_tracer_radius(pd.read_parquet(gt_paths[0])["r"].to_numpy())
    print(f"Detected tracer (large) radius r = {tracer_r}")

    results = {
        "gt": compute_lacey_over_dir(
            config.data_dir, config.frame_glob, GT_FRAME_RE, tracer_r, config,
            "lacey_over_time.parquet", "GT",
        )
    }

    if data_io.sorted_frame_files(config.prediction.pred_frames_dir, "pred_frame_*.parquet"):
        results["pred"] = compute_lacey_over_dir(
            config.prediction.pred_frames_dir, "pred_frame_*.parquet", PRED_FRAME_RE, tracer_r, config,
            "lacey_over_time_pred.parquet", "PRED",
        )

    return results


def plot_lacey_comparison(metrics: dict[str, pd.DataFrame], config: PipelineConfig, show=True):
    """Plot ground-truth vs surrogate Lacey index over time.

    Args:
        metrics: Mapping from :func:`compute_metrics` (``gt`` / optional ``pred``).
        config: If ``visualization.save_figures``, writes ``lacey_comparison.png``
            under ``FIGURES_DIR``.
        show: If ``True``, display the figure interactively.

    Returns:
        matplotlib.figure.Figure: Lacey-vs-time comparison figure.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(metrics["gt"]["time"], metrics["gt"]["lacey"], label="Ground Truth (DEM)", c="red")
    if "pred" in metrics:
        plt.plot(metrics["pred"]["time"], metrics["pred"]["lacey"], label="Surrogate Model (RNN)", c="blue")

    plt.xlabel("Time (s)")
    plt.ylabel("Lacey's Mixing Index")
    plt.title("LMI vs. Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if config.visualization.save_figures:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "lacey_comparison.png", dpi=140)
    if show:
        plt.show()
    return fig
