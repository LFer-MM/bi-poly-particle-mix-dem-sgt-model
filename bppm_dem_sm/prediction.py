"""Sliding-window frame prediction using a saved GRU surrogate model."""

from __future__ import annotations

from collections import deque
import os

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from . import data_io
from .config import ID_COL, PipelineConfig


class _SavedModelWrapper:
    """Thin adapter for legacy TensorFlow SavedModel exports (Keras 3 cannot load these directly)."""

    def __init__(self, path: Path):
        from .tf_quiet import silence_tensorflow

        silence_tensorflow()
        import tensorflow as tf

        fn = tf.saved_model.load(str(path)).signatures["serving_default"]
        _, input_spec = fn.structured_input_signature
        output_spec = fn.structured_outputs
        self._input_key = next(iter(input_spec))
        self._output_key = next(iter(output_spec))
        self._fn = fn
        self.input_shape = tuple(input_spec[self._input_key].shape.as_list())
        self.output_shape = tuple(output_spec[self._output_key].shape.as_list())

    def predict(self, x, batch_size=32, verbose=0):
        import numpy as np

        parts = []
        for start in range(0, x.shape[0], batch_size):
            batch = x[start : start + batch_size]
            out = self._fn(**{self._input_key: batch})
            if isinstance(out, dict):
                out = out[self._output_key]
            parts.append(out.numpy())
        return np.concatenate(parts, axis=0)


def _resolve_model_path(path: Path) -> Path:
    """Return an existing `.keras`, `.h5`, or SavedModel directory path."""
    candidates = [path]
    if path.suffix in {".keras", ".h5"}:
        candidates.append(path.with_suffix(""))
    else:
        candidates.extend([path.with_suffix(".keras"), path.with_suffix(".h5")])

    for candidate in candidates:
        if candidate.is_dir() and (candidate / "saved_model.pb").exists():
            return candidate
        if candidate.is_file() and candidate.suffix in {".keras", ".h5"}:
            return candidate

    raise FileNotFoundError(f"No Keras or SavedModel artifact found for: {path}")


def load_model(path):
    """Load a saved Keras model (``.keras``/``.h5``) or legacy SavedModel directory."""
    from .tf_quiet import silence_tensorflow

    silence_tensorflow()
    import keras

    resolved = _resolve_model_path(Path(path))
    if resolved.is_dir():
        return _SavedModelWrapper(resolved)
    return keras.models.load_model(str(resolved))


def predict_frames(config: PipelineConfig, model=None) -> pd.DataFrame:
    """Slide a window over frames, predict next positions, and save parquets."""
    config.pred_frames_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = config.feature_cols
    cols_needed = [ID_COL] + feature_cols
    frame_files = data_io.sorted_frame_files(config.data_dir, config.frame_glob)
    T = len(frame_files)
    start, seq_len = config.start_frame, config.frames_in

    if model is None:
        model = load_model(config.model_path)
    print("Model input_shape:", model.input_shape, "output_shape:", model.output_shape)

    base_df = data_io.load_frame(frame_files[start], cols_needed)
    base_ids = base_df[ID_COL].to_numpy()
    base_r = base_df["r"].to_numpy()

    window: deque = deque(maxlen=seq_len)
    for k in range(seq_len):
        df = data_io.load_frame(frame_files[start + k], cols_needed)
        window.append(df[feature_cols].to_numpy(np.float32))

    steps = T - (start + seq_len) if config.predict_until_end else config.max_steps

    all_rows = []
    for step in tqdm(range(steps), desc="Predicted & saved frames", unit="frame"):
        target_frame_idx = start + seq_len + step
        x_in = np.stack(window, axis=1)  # (N, seq_len, F)

        yhat = model.predict(x_in, batch_size=config.predict_batch_size, verbose=0)
        xyz = yhat[:, :3].astype(np.float32)

        pred_df = pd.DataFrame(
            {
                "id": base_ids,
                "x": xyz[:, 0],
                "y": xyz[:, 1],
                "z": xyz[:, 2],
                "dt": config.dt0 + step * config.dt_step,
                "r": base_r,
            }
        )
        pred_path = os.path.join(str(config.pred_frames_dir), f"pred_frame_{target_frame_idx:05d}.parquet")
        pred_df.to_parquet(pred_path, index=False)

        row = pred_df.copy()
        row.insert(0, "frame_pred", target_frame_idx)
        row.insert(1, "step", step)
        all_rows.append(row)

        if config.autoregressive:
            last_feats = window[-1].copy()
            last_feats[:, :3] = xyz  # x, y, z are the first 3 feature columns
            window.append(last_feats)
        else:
            next_df = data_io.load_frame(frame_files[target_frame_idx], cols_needed)
            window.append(next_df[feature_cols].to_numpy(np.float32))

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_parquet(str(config.pred_combined_parquet), index=False)
    print(f"Saved predictions under: {config.pred_out_dir}")
    return combined
