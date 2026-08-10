"""End-to-end orchestration for the RNN surrogate model.

``run_pipeline`` reads frames, optionally trains, predicts, computes Lacey
metrics, and renders visualizations. Each stage is also importable on its own.
"""

from __future__ import annotations

from dataclasses import replace

from . import prediction, run_metrics, run_visualization, training
from .config import PipelineConfig
from .tf_quiet import silence_tensorflow


def run_pipeline(config: PipelineConfig | None = None, **overrides):
    """Run the configurable surrogate pipeline; returns a dict of artifacts."""
    silence_tensorflow()
    config = replace(config or PipelineConfig(), **overrides)

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
        if config.show_plots or config.save_figures:
            run_metrics.plot_lacey_comparison(metrics, config, show=config.show_plots)

    if config.do_visualization:
        print("STAGE 4/4: Visualization")
        results["visualizations"] = run_visualization.generate_visualizations(config)

    print("Pipeline complete.")
    return results
