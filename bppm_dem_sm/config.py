"""Configuration and path resolution for the RNN surrogate pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
from typing import Any, get_type_hints

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
    """Restitution MatchMaker for steel/rock contacts (idx 0 steel, 1 rock; needs YADE).

    Returns:
        dict: Mapping with a ``"restitution"`` key whose value is a YADE
        ``MatchMaker`` of pairwise restitution coefficients.
    """
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
    """Knobs for the end-to-end surrogate pipeline.

    Groups data paths, GRU training hyperparameters, sliding-window prediction
    settings, Lacey metrics, and visualization flags used by
    :func:`bppm_dem_sm.pipeline.run_pipeline`.
    """

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
        """Directory holding one parquet per predicted frame.

        Returns:
            Path: ``pred_out_dir / "pred_frames"``.
        """
        return Path(self.pred_out_dir) / "pred_frames"

    @property
    def pred_combined_parquet(self) -> Path:
        """Path to the combined predictions table.

        Returns:
            Path: ``pred_out_dir / "predictions_all.parquet"``.
        """
        return Path(self.pred_out_dir) / "predictions_all.parquet"

    def to_dict(self) -> dict[str, Any]:
        """Serialize config fields; ``Path`` values become strings.

        Returns:
            dict[str, Any]: Field name to JSON-serializable value.
        """
        raw = asdict(self)
        for name, typ in _field_types().items():
            if typ is Path and raw.get(name) is not None:
                raw[name] = str(raw[name])
        return raw

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Build a config from a mapping (e.g. parsed JSON). Unknown keys raise.

        Args:
            data: Mapping of ``PipelineConfig`` field names to values. Paths
                may be strings; bools may be common string forms.

        Returns:
            PipelineConfig: Coerced configuration instance.

        Raises:
            ValueError: If ``data`` contains keys not defined on the dataclass.
            TypeError: If a bool field cannot be coerced.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown PipelineConfig keys: {sorted(unknown)}")

        coerced: dict[str, Any] = {}
        type_by_name = _field_types()
        for key, value in data.items():
            typ = type_by_name[key]
            if typ is Path and value is not None:
                coerced[key] = Path(value)
            elif typ is bool and not isinstance(value, bool):
                coerced[key] = _coerce_bool(value, key)
            else:
                coerced[key] = value
        return cls(**coerced)

    @classmethod
    def from_json(cls, path: Path | str) -> PipelineConfig:
        """Load config from a JSON file.

        Args:
            path: Path to a JSON object whose keys are ``PipelineConfig`` fields.

        Returns:
            PipelineConfig: Configuration built via :meth:`from_dict`.

        Raises:
            ValueError: If the JSON root is not an object.
            OSError: If the file cannot be read.
        """
        path = Path(path)
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"Config JSON must be an object, got {type(data).__name__}")
        return cls.from_dict(data)


def _field_types() -> dict[str, type]:
    """Map field name to concrete type used for coercion.

    Returns:
        dict[str, type]: Field name to ``Path``, ``bool``, or ``object``.
    """
    hints = get_type_hints(PipelineConfig)
    mapping: dict[str, type] = {}
    for name, hint in hints.items():
        origin = getattr(hint, "__origin__", None)
        if hint is Path or origin is Path:
            mapping[name] = Path
        elif hint is bool:
            mapping[name] = bool
        else:
            mapping[name] = object
    return mapping


def _coerce_bool(value: Any, key: str) -> bool:
    """Coerce a common string/truthy form into a bool for config loading.

    Args:
        value: Value to coerce (typically a string such as ``"true"`` / ``"0"``).
        key: Field name used only in the error message.

    Returns:
        bool: Parsed boolean.

    Raises:
        TypeError: If ``value`` is not a recognized boolean string form.
    """
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise TypeError(f"Cannot coerce {key!r}={value!r} to bool")
