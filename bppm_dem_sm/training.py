"""GRU surrogate model definition and training (TensorFlow imported lazily)."""

from __future__ import annotations

from pathlib import Path

from . import data_io
from .config import PipelineConfig


def build_model(frames_in, n_features=4, gru_units=20, dense_units=15, learning_rate=0.01):
    """Build and compile the GRU -> Dense regression model.

    Architecture: ``Input(frames_in, n_features)`` → GRU → Dense(tanh) →
    Dense(3, linear) predicting next ``(x, y, z)``. Compiled with Adam and MSE.

    Args:
        frames_in: Temporal window length (input sequence length).
        n_features: Features per timestep (typically 4: ``x, y, z, r``).
        gru_units: Hidden size of the GRU layer.
        dense_units: Units in the intermediate Dense layer.
        learning_rate: Adam optimizer learning rate.

    Returns:
        keras.Model: Compiled sequential GRU surrogate model.
    """
    from .tf_quiet import silence_tensorflow

    silence_tensorflow()
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(frames_in, n_features)),
            tf.keras.layers.GRU(gru_units),
            tf.keras.layers.Dense(dense_units, activation="tanh"),
            tf.keras.layers.Dense(3, activation="linear"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def train_and_save(config: PipelineConfig, plot_history=True):
    """Train the GRU surrogate on config.train_data_dir and save it.

    Builds a supervised sliding-window dataset, fits the model, writes a
    ``.keras`` artifact to ``config.model_path``, and optionally plots loss.

    Args:
        config: Pipeline settings (train data dir, epochs, batch size, etc.).
        plot_history: If ``True``, plot train/validation MSE curves.

    Returns:
        tuple: ``(model, history)`` — the trained Keras model and its
        ``History`` object.
    """
    pos, rad, _ = data_io.load_frames_stacked(config.train_data_dir, config.frame_glob, config.feature_cols)
    X, y = data_io.build_supervised_dataset(pos, rad, config.frames_in)
    Xtr, ytr, Xval, yval = data_io.train_test_split(X, y, config.val_fraction, config.seed)

    model = build_model(
        config.frames_in,
        n_features=X.shape[-1],
        gru_units=config.gru_units,
        dense_units=config.dense_units,
        learning_rate=config.learning_rate,
    )
    model.summary()

    history = model.fit(
        Xtr, ytr,
        validation_data=(Xval, yval),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=2,
    )

    save_path = Path(config.model_path)
    if save_path.suffix != ".keras":
        save_path = save_path.with_suffix(".keras")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"Saved model to: {save_path}")

    if plot_history:
        plot_training_history(history, show=config.show_plots)
    return model, history


def plot_training_history(history, show=True):
    """Plot train/validation loss curves from a keras History.

    Args:
        history: Keras ``History`` from ``model.fit`` (expects ``loss`` and
            ``val_loss`` keys).
        show: If ``True``, call ``plt.show()``.

    Returns:
        matplotlib.figure.Figure: The loss-curve figure.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training History")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if show:
        plt.show()
    return fig
