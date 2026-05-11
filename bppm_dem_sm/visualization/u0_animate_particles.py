# animate_particles_2d_static.py
# Put this script in any folder, set FRAMES_DIR, then run:
#   python animate_particles_2d_static.py
#
# Expects parquets like: frame_00000.parquet, frame_00001.parquet, ...
# Columns: id, x, y, z, r, m, vx, vy, vz (with header) OR same order w/out header.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# STATIC SETTINGS (edit these)
# -----------------------------
FRAMES_DIR = Path(r"C:\Users\Fernando\OneDrive - electro controles del noroeste\MCD\trabajo_terminal\s0_data\d1_rnn_prediction_data\pred_frames")
GLOB_PATTERN = "pred_frame_*.parquet"

PLANE = "xy"          # "xy", "xz", or "yz"
EVERY_NTH_FRAME = 1   # 2 = skip every other frame (faster)
FPS = 30              # playback speed
MARKER_SIZE = 4.0
ALPHA = 0.75

# Color options: "r", "vz", "speed", or "none"
COLOR_BY = "r"

LARGE_COLOR = "#1f77b4"
SMALL_COLOR = "#d62728"
R_TOL = 1e-12

# If you have tons of particles, cap points for speed (0 = no downsample)
MAX_POINTS = 0

# Optional: save mp4 (requires ffmpeg). Leave empty to just show.
SAVE_MP4_PATH = ""
DPI = 140
# -----------------------------

def list_frame_files():
    """Sorted frame paths under FRAMES_DIR respecting EVERY_NTH_FRAME. In: module globals. Out: list[Path]."""
    files = sorted(FRAMES_DIR.glob(GLOB_PATTERN))
    if not files:
        raise FileNotFoundError(f"No files matching '{GLOB_PATTERN}' in: {FRAMES_DIR}")
    return files[::max(1, EVERY_NTH_FRAME)]


def load_frame_parquet(path: Path) -> pd.DataFrame:
    """Load parquet with id,x,y,z,r (fallback read if columns missing). In: path. Out: DataFrame."""
    df = pd.read_parquet(path)

    expected = {"id", "x", "y", "z", "r"}
    if not expected.issubset(set(df.columns)):
        # No header case: assume fixed column order
        df = pd.read_parquet(
            path,
            columns=["id", "x", "y", "z", "r"]
        )
    return df


def project(df: pd.DataFrame):
    """2D coordinates and axis names for PLANE. In: df with x,y,z. Out: (x_arr, y_arr, (xlabel, ylabel))."""
    if PLANE == "xy":
        return df["x"].to_numpy(), df["y"].to_numpy(), ("x", "y")
    if PLANE == "xz":
        return df["x"].to_numpy(), df["z"].to_numpy(), ("x", "z")
    if PLANE == "yz":
        return df["y"].to_numpy(), df["z"].to_numpy(), ("y", "z")
    raise ValueError(f"Unknown PLANE: {PLANE}")


def color_values(df: pd.DataFrame):
    """Colors or scalar array for scatter per COLOR_BY. In: df. Out: None | str array | float array."""
    if COLOR_BY == "none":
        return None

    if COLOR_BY == "r":
        r = df["r"].to_numpy()

        # Detect unique radii (should be 2 for bidisperse)
        unique_r = np.unique(r)

        if len(unique_r) != 2:
            raise ValueError(f"Expected 2 unique radii, found {len(unique_r)}")

        small_r = unique_r.min()
        large_r = unique_r.max()

        colors = np.where(
            np.abs(r - small_r) < R_TOL,
            SMALL_COLOR,
            LARGE_COLOR
        )

        return colors

    if COLOR_BY == "vz":
        return df["vz"].to_numpy()

    if COLOR_BY == "speed":
        v = df[["vx", "vy", "vz"]].to_numpy()
        return np.linalg.norm(v, axis=1)

    raise ValueError(f"Unknown COLOR_BY: {COLOR_BY}")

def main():
    """Build matplotlib FuncAnimation from frame parquets; optional MP4. In: globals. Out: None."""
    files = list_frame_files()

    # Load first frame to set up plot + optional downsample indices
    df0 = load_frame_parquet(files[0])

    keep_idx = None
    if MAX_POINTS and len(df0) > MAX_POINTS:
        rng = np.random.default_rng(42)  # deterministic downsample
        keep_idx = rng.choice(len(df0), size=MAX_POINTS, replace=False)

    def slice_df(df):
        return df if keep_idx is None else df.iloc[keep_idx]

    df0 = slice_df(df0)
    x0, y0, (xl, yl) = project(df0)
    c0 = color_values(df0)

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)

    # Fixed limits so the view doesn't jump around
    xmin, xmax = -6, 6
    ymin, ymax = -6, 6
    dx = (xmax - xmin) * 0.03 + 1e-12
    dy = (ymax - ymin) * 0.03 + 1e-12
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)

    if c0 is None:
        sc = ax.scatter(x0, y0, s=MARKER_SIZE, alpha=ALPHA)
        cbar = None
    else:
        sc = ax.scatter(x0, y0, c=c0, s=MARKER_SIZE, alpha=ALPHA)

        # Only show colorbar for continuous values
        if COLOR_BY in ["vz", "speed"]:
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(COLOR_BY)
        else:
            cbar = None

    title = ax.set_title(files[0].name)

    def update(i):
        df = slice_df(load_frame_parquet(files[i]))
        x, y, _ = project(df)
        sc.set_offsets(np.column_stack([x, y]))

        c = color_values(df)
        if c is not None:
            if COLOR_BY in ["vz", "speed"]:
                sc.set_array(c)           # numeric colormap mode
            elif COLOR_BY == "r":
                sc.set_color(c)           # explicit color strings

                title.set_text(files[i].name)
                return sc, title

    interval_ms = int(1000 / max(1, FPS))
    anim = FuncAnimation(fig, update, frames=len(files), interval=interval_ms, blit=False)

    if SAVE_MP4_PATH:
        anim.save(SAVE_MP4_PATH, dpi=DPI, fps=FPS)
        print(f"Saved MP4: {SAVE_MP4_PATH}")
    else:
        plt.show()

if __name__ == "__main__":
    main()