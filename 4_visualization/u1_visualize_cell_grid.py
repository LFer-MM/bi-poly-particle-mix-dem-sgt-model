import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_particles_with_grid(
    frame_path: str,
    cell_size: float,
    use_equal_aspect: bool = True,
):
    """Scatter particles on XY with square grid overlay; two species colored by radius. In: path, cell_size, aspect flag. Out: None (shows plot)."""
    df = pd.read_parquet(frame_path)

    x = df["x"].values
    y = df["y"].values
    r = df["r"].values

    fig, ax = plt.subplots(figsize=(8, 8))

    unique_r = np.unique(r)

    if len(unique_r) == 2:
        r_small, r_large = np.min(unique_r), np.max(unique_r)

        mask_small = r == r_small
        mask_large = r == r_large

        ax.scatter(x[mask_small], y[mask_small], s=5, alpha=0.7, c="red", label="Rock")
        ax.scatter(x[mask_large], y[mask_large], s=5, alpha=0.7, c="blue", label="Ball")

    else:
        sc = ax.scatter(x, y, c=r, cmap="viridis", s=5, alpha=0.7)
        plt.colorbar(sc, label="Radius")

    xmin, xmax = -6, 6
    ymin, ymax = -6, 6

    pad = cell_size * 0.5
    xmin -= pad
    xmax += pad
    ymin -= pad
    ymax += pad

    x_lines = np.arange(xmin, xmax + cell_size, cell_size)
    y_lines = np.arange(ymin, ymax + cell_size, cell_size)

    for xl in x_lines:
        ax.axvline(x=xl, linewidth=0.8, color="black", alpha=0.8)

    for yl in y_lines:
        ax.axhline(y=yl, linewidth=0.8, color="black", alpha=0.8)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if use_equal_aspect:
        ax.set_aspect("equal")

    ax.set_title(f"Particle Positions with Grid (cell size = {cell_size} m)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    if len(unique_r) == 2:
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    frame_path = r"C:\Users\Fernando\OneDrive - electro controles del noroeste\MCD\trabajo_terminal\s0_data\d0_data\sic_dataset_20s_dt0p0001_parquet\frame_00000.parquet"
    cell_size = 0.4732
    plot_particles_with_grid(frame_path, cell_size, use_equal_aspect=True)
