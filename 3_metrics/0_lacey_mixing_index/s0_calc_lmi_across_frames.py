import os
import re
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass

FRAMES_DIR = r"C:\Users\Fernando\OneDrive - electro controles del noroeste\MCD\trabajo_terminal\s0_data\d0_data\sic_dataset_20s_dt0p0001_parquet"
GT_GLOB = "frame_*.parquet"
GT_FRAME_RE = re.compile(r"frame_(\d+)\.parquet$", re.IGNORECASE)

PRED_FRAMES_DIR = r"C:\Users\Fernando\OneDrive - electro controles del noroeste\MCD\trabajo_terminal\s0_data\d1_rnn_prediction_data\pred_frames"
PRED_GLOB = "pred_frame_*.parquet"
PRED_FRAME_RE = re.compile(r"pred_frame_(\d+)\.parquet$", re.IGNORECASE)

CELL_SIZE = 0.4732
DT = 0.05

MIN_PARTICLES_PER_CELL = 15
MAX_FRAMES = None

OUTPUT_PARQUET_NAME = "lacey_over_time.parquet"
OUTPUT_PRED_PARQUET_NAME = "lacey_over_time_pred.parquet"

USE_GT_TRACER_FOR_PRED = True

@dataclass
class LaceyResult:
    """One row of Lacey metrics over time. Fields: frame, time, lacey, cell stats."""
    frame: int
    time: float
    lacey: float
    n_cells_used: int
    tracer_fraction_global: float
    mean_particles_per_cell: float


def extract_frame_index(path: str, frame_re: re.Pattern) -> int:
    """Parse frame index from filename using frame_re. In: path, compiled regex. Out: int."""
    m = frame_re.search(os.path.basename(path))
    if not m:
        raise ValueError(f"File does not match expected pattern: {path}\nRegex: {frame_re.pattern}")
    return int(m.group(1))


def detect_tracer_radius(r_values: np.ndarray) -> float:
    """Pick larger of two most frequent rounded radii as tracer. In: r array. Out: float tracer radius."""
    r_rounded = np.round(r_values.astype(float), 12)
    uniq = np.unique(r_rounded)

    if len(uniq) < 2:
        raise ValueError(
            f"Expected at least 2 distinct radii for binary system; got {len(uniq)}: {uniq}"
        )

    # If more than 2 exist (noise / extra sizes), take top-2 most frequent.
    counts = {u: int(np.sum(r_rounded == u)) for u in uniq}
    top2 = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:2]
    radii = sorted([top2[0][0], top2[1][0]])
    tracer_r = radii[-1]
    return float(tracer_r)


def lacey_index_for_frame(
    df: pd.DataFrame,
    cell_size: float,
    tracer_radius: float,
    min_particles_per_cell: int = 5,
) -> tuple[float, int, float, float]:
    """Lacey M on 3D cell grid for one frame. In: df (x,y,z,r), cell_size, tracer r, min count. Out: M, n_cells, p_global, mean_n."""
    if cell_size <= 0:
        raise ValueError("CELL_SIZE must be > 0")

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)

    r = np.round(df["r"].to_numpy(dtype=float), 12)
    tracer = (r == np.round(tracer_radius, 12)).astype(np.int32)

    p = float(tracer.mean())
    if p <= 0.0 or p >= 1.0:
        return 1.0, 0, p, 0.0

    # grid origin (per-frame bounding box)
    x0, y0, z0 = float(x.min()), float(y.min()), float(z.min())
    ix = np.floor((x - x0) / cell_size).astype(np.int64)
    iy = np.floor((y - y0) / cell_size).astype(np.int64)
    iz = np.floor((z - z0) / cell_size).astype(np.int64)

    # hash cell ids for fast grouping
    h = ix * 73856093 + iy * 19349663 + iz * 83492791
    order = np.argsort(h)
    h_sorted = h[order]
    tracer_sorted = tracer[order]

    # split into runs
    boundaries = np.flatnonzero(np.diff(h_sorted)) + 1
    splits = np.split(np.arange(len(h_sorted)), boundaries)

    n_list = []
    p_list = []

    for idxs in splits:
        n_i = len(idxs)
        if n_i < min_particles_per_cell:
            continue
        t_i = int(tracer_sorted[idxs].sum())
        p_i = t_i / n_i
        n_list.append(n_i)
        p_list.append(p_i)

    if not n_list:
        return np.nan, 0, p, 0.0

    n_i = np.asarray(n_list, dtype=float)
    p_i = np.asarray(p_list, dtype=float)

    # weighted variance around global p
    S2 = float(np.sum(n_i * (p_i - p) ** 2) / np.sum(n_i))

    S0_2 = float(p * (1.0 - p))
    Sr2 = float(np.mean(p * (1.0 - p) / n_i))

    denom = (S0_2 - Sr2)
    if denom <= 0:
        return np.nan, int(len(n_i)), p, float(n_i.mean())

    M = (S0_2 - S2) / denom
    M = float(np.clip(M, 0.0, 1.0))

    return M, int(len(n_i)), p, float(n_i.mean())


def compute_lacey_over_dir(
    frames_dir: str,
    file_glob: str,
    frame_re: re.Pattern,
    tracer_r: float,
    out_parquet_name: str,
    label: str,
) -> pd.DataFrame:
    """Compute Lacey per file in directory; write summary parquet. In: dirs, glob, tracer_r, names, label. Out: DataFrame."""
    paths = sorted(glob.glob(os.path.join(frames_dir, file_glob)))
    if not paths:
        raise FileNotFoundError(f"No files found with glob '{file_glob}' in: {frames_dir}")

    if MAX_FRAMES is not None:
        paths = paths[:MAX_FRAMES]

    results: list[LaceyResult] = []

    for pth in paths:
        frame_idx = extract_frame_index(pth, frame_re)
        df = pd.read_parquet(pth)

        M, n_cells, p_global, mean_n = lacey_index_for_frame(
            df=df,
            cell_size=CELL_SIZE,
            tracer_radius=tracer_r,
            min_particles_per_cell=MIN_PARTICLES_PER_CELL,
        )

        t = (frame_idx * DT) if (DT is not None) else None
        results.append(
            LaceyResult(
                frame=frame_idx,
                time=t,
                lacey=M,
                n_cells_used=n_cells,
                tracer_fraction_global=p_global,
                mean_particles_per_cell=mean_n,
            )
        )

        print(
            f"[{label}] frame {frame_idx:05d} | M={M:.4f} | cells={n_cells} | p={p_global:.3f} | n̄={mean_n:.1f}"
        )

    out = pd.DataFrame([r.__dict__ for r in results]).sort_values("frame")
    out_path = os.path.join(frames_dir, out_parquet_name)
    out.to_parquet(out_path, index=False)
    print(f"[{label}] Saved: {out_path}")

    return out


def main():
    """CLI: GT + optional PRED Laceys and plot. In: module constants. Out: None."""
    import matplotlib.pyplot as plt

    gt_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, GT_GLOB)))
    if not gt_paths:
        raise FileNotFoundError(f"No files found with glob '{GT_GLOB}' in: {FRAMES_DIR}")

    df0 = pd.read_parquet(gt_paths[0])
    tracer_r_gt = detect_tracer_radius(df0["r"].to_numpy())
    print(f"Detected tracer (large) radius r = {tracer_r_gt} (from GT first frame)")

    gt_df = compute_lacey_over_dir(
        frames_dir=FRAMES_DIR,
        file_glob=GT_GLOB,
        frame_re=GT_FRAME_RE,
        tracer_r=tracer_r_gt,
        out_parquet_name=OUTPUT_PARQUET_NAME,
        label="GT",
    )

    # SM
    pred_df = None
    if PRED_FRAMES_DIR is not None:
        pred_paths = sorted(glob.glob(os.path.join(PRED_FRAMES_DIR, PRED_GLOB)))
        if not pred_paths:
            raise FileNotFoundError(f"No prediction files found with glob '{PRED_GLOB}' in: {PRED_FRAMES_DIR}")

        if USE_GT_TRACER_FOR_PRED:
            tracer_r_pred = tracer_r_gt
            print(f"Using GT tracer radius for PRED: r = {tracer_r_pred}")
        else:
            dfp0 = pd.read_parquet(pred_paths[0])
            tracer_r_pred = detect_tracer_radius(dfp0["r"].to_numpy())
            print(f"Detected tracer radius r = {tracer_r_pred} (from PRED first file)")

        pred_df = compute_lacey_over_dir(
            frames_dir=PRED_FRAMES_DIR,
            file_glob=PRED_GLOB,
            frame_re=PRED_FRAME_RE,
            tracer_r=tracer_r_pred,
            out_parquet_name=OUTPUT_PRED_PARQUET_NAME,
            label="PRED",
        )

    x_label = "Time (s)" if DT is not None else "Frame"

    plt.figure()

    x_gt = gt_df["time"] if DT is not None else gt_df["frame"]
    plt.plot(x_gt, gt_df["lacey"], label="Ground Truth (DEM)", c="red")

    if pred_df is not None:
        x_pr = pred_df["time"] if DT is not None else pred_df["frame"]
        plt.plot(x_pr, pred_df["lacey"], label="Surrogate Model (RNN)", c="blue")

    plt.xlabel(x_label)
    plt.ylabel("Lacey's Mixing Index")
    plt.title("LMI vs. Time")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()