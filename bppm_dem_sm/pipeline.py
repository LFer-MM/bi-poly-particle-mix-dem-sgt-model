"""End-to-end orchestration for the RNN surrogate model.

``run_pipeline`` reads frames, optionally trains, predicts, computes Lacey
metrics, and renders visualizations. Each stage is also importable on its own.
"""

from __future__ import annotations

from . import prediction, run_metrics, run_visualization, training
from .config import PipelineConfig
from .tf_quiet import silence_tensorflow


def run_pipeline(config: PipelineConfig | None = None, **overrides):
    """Run the configurable surrogate pipeline; returns a dict of artifacts.

    Stages (each gated by the corresponding ``do_*`` flag on ``config``):

    1. Train the GRU surrogate
    2. Predict frames with a sliding window
    3. Compute Lacey mixing-index metrics
    4. Generate cell-grid and animation visualizations

    Args:
        config: Base pipeline configuration; ``None`` uses defaults.
        **overrides: Field overrides applied via
            :meth:`PipelineConfig.with_overrides`. Accepts core fields
            (``do_train=True``), nested leaf names (``epochs=5``), or whole
            option groups (``training=TrainingOptions(epochs=5)``).

    Returns:
        dict: Artifacts keyed by stage (``config``, and optionally ``model``,
        ``history``, ``predictions``, ``metrics``, ``visualizations``).
    """
    silence_tensorflow()
    config = (config or PipelineConfig()).with_overrides(**overrides)

    results: dict = {"config": config}
    model = None

    if config.do_train:
        print("STAGE 1/4: Training")
        model, results["history"] = training.train_and_save(config)
        results["model"] = model

    if config.do_predict:
        print("STAGE 2/4: Prediction")
        results["predictions"] = prediction.predict_frames(config, model=model)

    if config.do_metrics:
        print("STAGE 3/4: Metrics")
        metrics = run_metrics.compute_metrics(config)
        results["metrics"] = metrics
        viz = config.visualization
        if viz.show_plots or viz.save_figures:
            run_metrics.plot_lacey_comparison(metrics, config, show=viz.show_plots)

    if config.do_visualization:
        print("STAGE 4/4: Visualization")
        results["visualizations"] = run_visualization.generate_visualizations(config)

    print("Pipeline complete.")
    return results
