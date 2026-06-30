"""Configuration and path resolution for the RNN surrogate pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DEFAULT_DATASET = "sic_dataset_20s_dt0p0001_parquet"
DEFAULT_TRAIN_DATASET = "sic_training_dataset_3s_4s_parquet"
DEFAULT_MODEL_NAME = "rnn_gru_sic_model"

ID_COL = "id"
FEATURE_COLS = ["x", "y", "z", "r"]
TARGET_COLS = ["x", "y", "z"]

# --- DEM simulation (YADE) ---
ROCK_COUNT = 19888  # 9%
ROCK_DIAM_M = 0.06985  # 2.75 inches
BALL_COUNT = 4696  # 17%
BALL_DIAM_M = 0.1397  # 5.5 inches
SAGMILL_STL_PATH = "sag_mill_40ft_m.stl"

MATERIALS = {
    "steel": {
        "density": 7850,
        "young": 155709722558.42664,
        "poisson": 0.292,
        "frictionAngle": math.atan(0.5),
        "label": "steel",
    },
    "rock": {
        "density": 2650,
        "young": 13468135026.041664,
        "poisson": 0.25,
        "frictionAngle": math.atan(0.5),
        "label": "rock",
    },
}


def build_material_interactions():
    """Restitution MatchMaker for steel/rock contacts (idx 0 steel, 1 rock; needs YADE)."""
    from yade import MatchMaker

    return {
        "restitution": MatchMaker(matches=[
        (0, 0, 0.8),   # steel-steel
        (0, 1, 0.5),   # steel-rock
        (1, 0, 0.5),   # rock-steel
        (1, 1, 0.3)    # rock-rock
    ])
    }


@dataclass
class PipelineConfig:
    """Knobs for the end-to-end surrogate pipeline."""

    # data
    data_dir: Path = PROCESSED_DIR / DEFAULT_DATASET
    train_data_dir: Path = PROCESSED_DIR / DEFAULT_TRAIN_DATASET
    frame_glob: str = "frame_*.parquet"
    feature_cols: list[str] = field(default_factory=lambda: list(FEATURE_COLS))

    # model
    model_path: Path = MODELS_DIR / f"{DEFAULT_MODEL_NAME}.keras"
    frames_in: int = 15

    # training
    do_train: bool = False
    epochs: int = 20
    batch_size: int = 500
    learning_rate: float = 0.01
    val_fraction: float = 0.1
    seed: int = 0
    gru_units: int = 20
    dense_units: int = 15

    # prediction
    do_predict: bool = True
    start_frame: int = 66
    autoregressive: bool = False
    predict_until_end: bool = True
    max_steps: int = 200
    predict_batch_size: int = 2048
    dt0: float = 4.05
    dt_step: float = 0.05
    pred_out_dir: Path = INTERIM_DIR / "rnn_predictions"

    # metrics
    do_metrics: bool = True
    cell_size: float = 0.4732
    min_particles_per_cell: int = 15
    metrics_dt: float = 0.05

    # visualization
    do_visualization: bool = True
    plane: str = "xy"
    fps: int = 30
    marker_size: float = 4.0
    every_nth_frame: int = 1
    save_figures: bool = False
    show_plots: bool = True

    @property
    def pred_frames_dir(self) -> Path:
        """Directory holding one parquet per predicted frame."""
        return Path(self.pred_out_dir) / "pred_frames"

    @property
    def pred_combined_parquet(self) -> Path:
        """Path to the combined predictions table."""
        return Path(self.pred_out_dir) / "predictions_all.parquet"
