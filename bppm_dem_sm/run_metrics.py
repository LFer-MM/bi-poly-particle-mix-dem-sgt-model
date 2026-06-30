"""Lacey mixing-index over directories of ground-truth and predicted frames."""

from __future__ import annotations

import os

import pandas as pd

from . import data_io
from . import lacey_mixing_index as lacey
from .config import FIGURES_DIR, PipelineConfig
from .lacey_mixing_index import GT_FRAME_RE, PRED_FRAME_RE


def compute_lacey_over_dir(frames_dir, pattern, frame_re, tracer_r, config, out_name, label):
    """Compute the Lacey index per frame in a directory; save a summary parquet."""
    rows = []
    for pth in data_io.sorted_frame_files(frames_dir, pattern):
        frame_idx = lacey.extract_frame_index(pth, frame_re)
        df = pd.read_parquet(pth)
        M, n_cells, p_global, mean_n = lacey.lacey_index_for_frame(
            df, config.cell_size, tracer_r, config.min_particles_per_cell
        )
        rows.append(
            {
                "frame": frame_idx,
                "time": frame_idx * config.metrics_dt,
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
    """Compute the Lacey index for ground-truth and predicted frames."""
    gt_paths = data_io.sorted_frame_files(config.data_dir, config.frame_glob)
    tracer_r = lacey.detect_tracer_radius(pd.read_parquet(gt_paths[0])["r"].to_numpy())
    print(f"Detected tracer (large) radius r = {tracer_r}")

    results = {
        "gt": compute_lacey_over_dir(
            config.data_dir, config.frame_glob, GT_FRAME_RE, tracer_r, config,
            "lacey_over_time.parquet", "GT",
        )
    }

    if data_io.sorted_frame_files(config.pred_frames_dir, "pred_frame_*.parquet"):
        results["pred"] = compute_lacey_over_dir(
            config.pred_frames_dir, "pred_frame_*.parquet", PRED_FRAME_RE, tracer_r, config,
            "lacey_over_time_pred.parquet", "PRED",
        )

    return results


def plot_lacey_comparison(metrics: dict[str, pd.DataFrame], config: PipelineConfig, show=True):
    """Plot ground-truth vs surrogate Lacey index over time."""
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

    if config.save_figures:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "lacey_comparison.png", dpi=140)
    if show:
        plt.show()
    return fig
