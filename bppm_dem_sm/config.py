"""Configuration and path resolution for the RNN surrogate pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
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
class TrainingOptions:
    """GRU training hyperparameters."""

    epochs: int = 20
    batch_size: int = 500
    learning_rate: float = 0.01
    val_fraction: float = 0.1
    seed: int = 0
    gru_units: int = 20
    dense_units: int = 15


@dataclass
class PredictionOptions:
    """Sliding-window prediction settings and output paths."""

    start_frame: int = 66
    autoregressive: bool = False
    predict_until_end: bool = True
    max_steps: int = 200
    predict_batch_size: int = 2048
    dt0: float = 4.05
    dt_step: float = 0.05
    pred_out_dir: Path = INTERIM_DIR / "rnn_predictions"

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


@dataclass
class MetricsOptions:
    """Lacey mixing-index cell grid and time-axis settings."""

    cell_size: float = 0.4732
    min_particles_per_cell: int = 15
    metrics_dt: float = 0.05


@dataclass
class VisualizationOptions:
    """Animation and figure display / save settings."""

    plane: str = "xy"
    fps: int = 30
    marker_size: float = 4.0
    every_nth_frame: int = 1
    save_figures: bool = False
    show_plots: bool = True


@dataclass
class PipelineConfig:
    """Knobs for the end-to-end surrogate pipeline.

    Core fields are data paths, model identity, and stage toggles. Training,
    prediction, metrics, and visualization knobs live on nested option
    dataclasses (defaults apply when a group is omitted). Used by
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

    # stages
    do_train: bool = False
    do_predict: bool = True
    do_metrics: bool = True
    do_visualization: bool = True

    training: TrainingOptions = field(default_factory=TrainingOptions)
    prediction: PredictionOptions = field(default_factory=PredictionOptions)
    metrics: MetricsOptions = field(default_factory=MetricsOptions)
    visualization: VisualizationOptions = field(default_factory=VisualizationOptions)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config fields; ``Path`` values become strings.

        Nested option groups are nested objects. Omitted groups in
        :meth:`from_dict` keep dataclass defaults.

        Returns:
            dict[str, Any]: Field name to JSON-serializable value.
        """
        return _to_plain_dict(self)

    def with_overrides(self, **overrides: Any) -> PipelineConfig:
        """Return a copy with top-level or nested leaf fields replaced.

        Nested group objects may be passed by name (``training=TrainingOptions(...)``)
        or individual leaf names may be passed flat (``epochs=5``), matching CLI
        flags. Leaf overrides apply after whole-group replacements.

        Args:
            **overrides: ``PipelineConfig`` field names, option-group names, or
                leaf names from a nested options dataclass.

        Returns:
            PipelineConfig: New instance with overrides applied.

        Raises:
            ValueError: If an override name is not a known field.
            TypeError: If a group override is not the matching options class.
        """
        group_types = _option_group_types()
        leaf_to_group = _leaf_to_group()
        top_names = {f.name for f in fields(self)} - set(group_types)

        top: dict[str, Any] = {}
        nested_updates: dict[str, dict[str, Any]] = {}
        for key, value in overrides.items():
            if key in group_types:
                expected = group_types[key]
                if not isinstance(value, expected):
                    raise TypeError(
                        f"{key} override must be {expected.__name__}, "
                        f"got {type(value).__name__}"
                    )
                top[key] = value
            elif key in top_names:
                top[key] = value
            elif key in leaf_to_group:
                nested_updates.setdefault(leaf_to_group[key], {})[key] = value
            else:
                raise ValueError(f"Unknown PipelineConfig override: {key!r}")

        updated = replace(self, **top) if top else self
        for group, leafs in nested_updates.items():
            current = getattr(updated, group)
            updated = replace(updated, **{group: replace(current, **leafs)})
        return updated

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Build a config from a mapping (e.g. parsed JSON). Unknown keys raise.

        Nested groups are objects keyed ``training``, ``prediction``,
        ``metrics``, and ``visualization``. Paths may be strings; bools may be
        common string forms.

        Args:
            data: Mapping of ``PipelineConfig`` field names to values.

        Returns:
            PipelineConfig: Coerced configuration instance.

        Raises:
            ValueError: If ``data`` contains keys not defined on this class or
                a nested options class.
            TypeError: If a bool field cannot be coerced, or a nested group
                is not a mapping.
        """
        return _from_plain_dict(cls, data)

    @classmethod
    def from_json(cls, path: Path | str) -> PipelineConfig:
        """Load config from a JSON file.

        Args:
            path: Path to a JSON object whose keys are ``PipelineConfig``
                fields (nested option groups as objects).

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


def _option_group_types() -> dict[str, type]:
    """Map ``PipelineConfig`` field name to nested options dataclass."""
    return {
        "training": TrainingOptions,
        "prediction": PredictionOptions,
        "metrics": MetricsOptions,
        "visualization": VisualizationOptions,
    }


def _leaf_to_group() -> dict[str, str]:
    """Map nested options field name to its group name on ``PipelineConfig``."""
    mapping: dict[str, str] = {}
    for group, cls in _option_group_types().items():
        for f in fields(cls):
            mapping[f.name] = group
    return mapping


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    """Recursively serialize a config dataclass; ``Path`` values become strings."""
    cls = type(obj)
    hints = get_type_hints(cls)
    raw: dict[str, Any] = {}
    for f in fields(cls):
        val = getattr(obj, f.name)
        hint = hints[f.name]
        if isinstance(hint, type) and is_dataclass(hint):
            raw[f.name] = _to_plain_dict(val)
        elif hint is Path and val is not None:
            raw[f.name] = str(val)
        else:
            raw[f.name] = val
    return raw


def _from_plain_dict(cls: type, data: dict[str, Any]) -> Any:
    """Coerce a mapping into ``cls``, recursing into nested dataclasses."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")

    hints = get_type_hints(cls)
    coerced: dict[str, Any] = {}
    for key, value in data.items():
        hint = hints[key]
        if isinstance(hint, type) and is_dataclass(hint):
            if not isinstance(value, dict):
                raise TypeError(
                    f"{key} must be an object, got {type(value).__name__}"
                )
            coerced[key] = _from_plain_dict(hint, value)
        elif hint is Path and value is not None:
            coerced[key] = Path(value)
        elif hint is bool and not isinstance(value, bool):
            coerced[key] = _coerce_bool(value, key)
        else:
            coerced[key] = value
    return cls(**coerced)


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
