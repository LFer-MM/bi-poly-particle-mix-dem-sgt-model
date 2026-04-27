import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

FRAMES_IN = 15
DATA_GLOB = r"C:\Users\Fernando\OneDrive - electro controles del noroeste\MCD\trabajo_terminal\s0_data\d0_data\sic_training_dataset_3s_4s_parquet\frame_*.parquet"
EPOCHS = 20
BATCH = 500

COLS = ["id", "x", "y", "z", "r"]

frame_files = sorted(glob.glob(DATA_GLOB))

frames = [pd.read_parquet(f, columns=COLS) for f in frame_files]

base_ids = frames[0]["id"].to_numpy()
for k, df in enumerate(frames[1:], start=1):
    ids = df["id"].to_numpy()

    if len(ids) != len(base_ids) or not np.array_equal(ids, base_ids):
        frames[k] = df.sort_values("id").reset_index(drop=True)

frames[0] = frames[0].sort_values("id").reset_index(drop=True)
base_ids = frames[0]["id"].to_numpy()

N = len(base_ids)

# Stack arrays: positions[t, i, 3], radius[t, i, 1]
pos = np.stack([df[["x","y","z"]].to_numpy(np.float32) for df in frames], axis=0)  # [T, N, 3]
rad = np.stack([df[["r"]].to_numpy(np.float32) for df in frames], axis=0)          # [T, N, 1]

# ----------------------------
# Build supervised dataset
# X: 15 frames of [x,y,z,r] per particle
# y: next frame position [x,y,z] per particle
# ----------------------------
T = pos.shape[0]
Xs, Ys = [], []
for t0 in range(0, T - FRAMES_IN):
    t1 = t0 + FRAMES_IN

    # input sequence frames t0..t1-1
    x_seq = np.concatenate([pos[t0:t1], rad[t0:t1]], axis=-1)   # [15, N, 4]
    y_next = pos[t1]                                            # [N, 3]

    # make each particle a sample
    Xs.append(np.transpose(x_seq, (1, 0, 2)))  # [N, 15, 4]
    Ys.append(y_next)                          # [N, 3]

X = np.concatenate(Xs, axis=0)  # [num_samples, 15, 4]
y = np.concatenate(Ys, axis=0)  # [num_samples, 3]

# Simple train/test split
n = X.shape[0]
idx = np.arange(n)
np.random.seed(0)
np.random.shuffle(idx)
split = int(0.9 * n)
tr, te = idx[:split], idx[split:]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

model = tf.keras.Sequential([
tf.keras.layers.Input(shape=(15, 4)),
    tf.keras.layers.GRU(20),
    tf.keras.layers.Dense(15, activation="tanh"),
    tf.keras.layers.Dense(3, activation="linear"),
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse")

model.summary()
history = model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=EPOCHS, batch_size=BATCH, verbose=2)

model.save(r"C:\Users\Fernando\OneDrive - electro controles del noroeste\MCD\trabajo_terminal\s1_model\m0_RNN\rnn_gru_sic_model")

plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training History")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()