# predict_and_save_frames_static.py
import os
import glob
from collections import deque

import numpy as np
import pandas as pd

# =========================
# STATIC CONFIG (EDIT THIS)
# =========================

FRAMES_DIR = r"C:\Users\Fernando\OneDrive - electro controles del noroeste\MCD\trabajo_terminal\s0_data\d0_data\sic_dataset_20s_dt0p0001_parquet"             # folder with frame_*.parquet (ground truth)
MODEL_PATH = r"C:\Users\Fernando\OneDrive - electro controles del noroeste\MCD\trabajo_terminal\s1_model\m0_RNN\rnn_gru_sic_model"   # SavedModel dir OR .keras/.h5 file

FRAME_GLOB = "frame_*.parquet"

START_FRAME = 66      # 0-based index within sorted frame list
SEQ_LEN = 15          # previous frames used as model input

# Must match training input features (order matters)
# Typical from your case: x,y,z,r
FEATURE_COLS = ["x", "y", "z", "r"]

# Output folders
OUT_DIR = r"C:\Users\Fernando\OneDrive - electro controles del noroeste\MCD\trabajo_terminal\s0_data\d1_rnn_prediction_data"
OUT_PRED_FRAMES_DIR = os.path.join(OUT_DIR, "pred_frames")   # one parquet per predicted frame
OUT_COMBINED_PARQUET = os.path.join(OUT_DIR, "predictions_all.parquet")

# Time column: dt_value = DT0 + step*DT_STEP
DT0 = 4.05
DT_STEP = 0.05

# Prediction mode:
#   False => teacher forcing / ground-truth windows (use real frames to slide window)
#   True  => autoregressive (feed predicted xyz back into window)
AUTOREGRESSIVE = False

# Predict until end of dataset (recommended for comparison with ground truth)
PREDICT_UNTIL_END = True

# If you want a hard limit (ignored if PREDICT_UNTIL_END=True)
MAX_STEPS = 200

# Batch size used in model.predict
BATCH_SIZE = 2048

# =========================
# END CONFIG
# =========================


def sorted_frame_files(folder: str, pattern: str):
    """Sorted parquet paths under folder matching pattern. In: folder, glob pattern. Out: list[str] or raises FileNotFoundError."""
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise FileNotFoundError(f"No files found in: {os.path.join(folder, pattern)}")
    return files


def load_model_any(path: str):
    """Load Keras/SavedModel from file or directory path. In: path str. Out: tf.keras.Model."""
    import tensorflow as tf

    return tf.keras.models.load_model(path)


def load_frame(path: str, cols):
    """Read parquet with selected columns; requires id. In: path, cols list. Out: DataFrame."""
    df = pd.read_parquet(path, columns=cols)
    if "id" not in df.columns:
        raise ValueError(f"'id' column missing in {path}")
    return df


def align_to_base_ids(df: pd.DataFrame, base_ids: np.ndarray):
    """Sort by id and assert id column matches base_ids. In: df, base_ids array. Out: aligned DataFrame."""
    df2 = df.sort_values("id").reset_index(drop=True)
    ids = df2["id"].to_numpy()
    if len(ids) != len(base_ids) or not np.array_equal(ids, base_ids):
        raise ValueError("Frame ids don't match base frame ids after sorting. Dataset not consistent.")
    return df2


def infer_xyz(yhat: np.ndarray) -> np.ndarray:
    """Normalize model output to (N,3) xyz (handles squeeze and D>=3). In: yhat array. Out: float array shape (N,3)."""
    y = np.asarray(yhat)
    if y.ndim == 3 and y.shape[1] == 1:
        y = y[:, 0, :]
    if y.ndim != 2 or y.shape[1] < 3:
        raise ValueError(f"Unexpected model output shape: {yhat.shape}. Expected (N, >=3).")
    return y[:, :3]


def ensure_dirs():
    """Create OUT_DIR and OUT_PRED_FRAMES_DIR. In: module constants. Out: None."""
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_PRED_FRAMES_DIR, exist_ok=True)


def main():
    """Run sliding-window RNN predictions per CONFIG; writes parquets. In: globals. Out: None."""
    ensure_dirs()

    frame_files = sorted_frame_files(FRAMES_DIR, FRAME_GLOB)
    T = len(frame_files)

    if START_FRAME < 0 or START_FRAME >= T:
        raise ValueError(f"START_FRAME out of range: {START_FRAME} (T={T})")

    if START_FRAME + SEQ_LEN >= T and not AUTOREGRESSIVE:
        raise ValueError(
            f"Not enough frames for teacher-forcing mode. Need START_FRAME+SEQ_LEN < T. "
            f"Got START_FRAME={START_FRAME}, SEQ_LEN={SEQ_LEN}, T={T}."
        )

    print(f"Loading model: {MODEL_PATH}")
    model = load_model_any(MODEL_PATH)
    print("Model loaded.")
    print("Model input_shape:", model.input_shape, "output_shape:", model.output_shape)

    cols_needed = ["id"] + FEATURE_COLS

    # Establish base id ordering from the first window frame
    base_df = load_frame(frame_files[START_FRAME], cols=cols_needed).sort_values("id").reset_index(drop=True)
    base_ids = base_df["id"].to_numpy()

    # Preserve particle radius (or species marker) for all predictions
    if "r" in FEATURE_COLS:
        base_r = base_df["r"].to_numpy()
    else:
        base_r = None

    # Prime the window with SEQ_LEN frames
    window = deque(maxlen=SEQ_LEN)
    for k in range(SEQ_LEN):
        idx = START_FRAME + k
        df = load_frame(frame_files[idx], cols=cols_needed)
        df = align_to_base_ids(df, base_ids)
        window.append(df[FEATURE_COLS].to_numpy(dtype=np.float32))

    # Decide how many steps
    if PREDICT_UNTIL_END:
        if AUTOREGRESSIVE:
            steps = T - (START_FRAME + SEQ_LEN)  # match dataset length for comparison
        else:
            steps = T - (START_FRAME + SEQ_LEN)  # each prediction corresponds to an existing target frame
    else:
        steps = MAX_STEPS

    if steps <= 0:
        raise RuntimeError("No prediction steps available with the current START_FRAME/SEQ_LEN.")

    all_rows = []

    # Main loop
    for step in range(steps):
        # Target frame index we are predicting "for" (matches ground truth index if teacher-forcing)
        target_frame_idx = START_FRAME + SEQ_LEN + step

        # Build model input: (N, SEQ_LEN, F)
        x_in = np.stack(window, axis=1)

        yhat = model.predict(x_in, batch_size=BATCH_SIZE, verbose=0)
        xyz = infer_xyz(yhat).astype(np.float32)

        dt_val = DT0 + step * DT_STEP

        # Per-frame predicted DF (this is what you’ll feed to Lacey’s index)
        pred_dict = {
            "id": base_ids,
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            "dt": dt_val,
        }

        if base_r is not None:
            pred_dict["r"] = base_r

        pred_df = pd.DataFrame(pred_dict)

        # Save one parquet per predicted frame
        pred_path = os.path.join(OUT_PRED_FRAMES_DIR, f"pred_frame_{target_frame_idx:05d}.parquet")
        pred_df.to_parquet(pred_path, index=False)

        # Also store a combined table (with frame index & step)
        pred_df2 = pred_df.copy()
        pred_df2.insert(0, "frame_pred", target_frame_idx)
        pred_df2.insert(1, "step", step)
        all_rows.append(pred_df2)

        # Advance window
        if AUTOREGRESSIVE:
            # Feed prediction back into features for next step.
            # If FEATURE_COLS contains extra features (like r), carry them forward from last frame.
            last_feats = window[-1].copy()  # (N, F)
            feat_map = {c: i for i, c in enumerate(FEATURE_COLS)}

            for comp, col in enumerate(["x", "y", "z"]):
                if col in feat_map:
                    last_feats[:, feat_map[col]] = xyz[:, comp]
            window.append(last_feats)

        else:
            # Teacher forcing: slide by reading the REAL next frame (ground truth)
            if target_frame_idx >= T:
                break
            next_df = load_frame(frame_files[target_frame_idx], cols=cols_needed)
            next_df = align_to_base_ids(next_df, base_ids)
            window.append(next_df[FEATURE_COLS].to_numpy(dtype=np.float32))

        if (step + 1) % 10 == 0:
            print(f"Predicted & saved {step+1}/{steps} frames...")

    # Save combined parquet
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_parquet(OUT_COMBINED_PARQUET, index=False)

    print("Done.")
    print(f"Per-frame predictions saved in: {OUT_PRED_FRAMES_DIR}")
    print(f"Combined predictions saved to:  {OUT_COMBINED_PARQUET}")
    print(combined.head())

if __name__ == "__main__":
    main()