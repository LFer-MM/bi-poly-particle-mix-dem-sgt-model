"""Tests for PipelineConfig JSON loading and CLI argument resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bppm_dem_sm.cli import build_parser, config_from_args
from bppm_dem_sm.config import REPO_ROOT, PipelineConfig, PredictionOptions, TrainingOptions
from bppm_dem_sm.pipeline import run_pipeline


def test_from_dict_coerces_paths_and_bools():
    cfg = PipelineConfig.from_dict(
        {
            "data_dir": "data/processed/foo",
            "do_train": True,
            "frames_in": 10,
        }
    )
    assert cfg.data_dir == Path("data/processed/foo")
    assert cfg.do_train is True
    assert cfg.frames_in == 10
    assert cfg.training.epochs == 20
    assert cfg.prediction.start_frame == 66


def test_from_dict_nested_groups():
    cfg = PipelineConfig.from_dict(
        {
            "do_train": True,
            "training": {"epochs": 3, "batch_size": 32},
            "prediction": {"start_frame": 7, "pred_out_dir": "data/interim/preds"},
            "visualization": {"show_plots": False},
        }
    )
    assert cfg.do_train is True
    assert cfg.training.epochs == 3
    assert cfg.training.batch_size == 32
    assert cfg.training.learning_rate == 0.01
    assert cfg.prediction.start_frame == 7
    assert cfg.prediction.pred_out_dir == Path("data/interim/preds")
    assert cfg.visualization.show_plots is False


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown PipelineConfig keys"):
        PipelineConfig.from_dict({"not_a_field": 1})


def test_from_dict_rejects_flat_nested_keys():
    with pytest.raises(ValueError, match="Unknown PipelineConfig keys"):
        PipelineConfig.from_dict({"epochs": 5})


def test_from_dict_rejects_unknown_nested_keys():
    with pytest.raises(ValueError, match="Unknown TrainingOptions keys"):
        PipelineConfig.from_dict({"training": {"not_a_field": 1}})


def test_from_json_roundtrip(tmp_path: Path):
    original = PipelineConfig(
        do_train=True,
        do_predict=False,
        prediction=PredictionOptions(start_frame=42),
        data_dir=Path("data/processed/example"),
    )
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps(original.to_dict()), encoding="utf-8")

    loaded = PipelineConfig.from_json(path)
    assert loaded.do_train is True
    assert loaded.do_predict is False
    assert loaded.prediction.start_frame == 42
    assert loaded.data_dir == Path("data/processed/example")
    assert "prediction" in original.to_dict()
    assert original.to_dict()["prediction"]["start_frame"] == 42


def test_cli_config_json_ignores_other_flags(tmp_path: Path):
    path = tmp_path / "cfg.json"
    path.write_text(
        json.dumps(
            {
                "do_train": True,
                "do_predict": False,
                "prediction": {"start_frame": 99},
            }
        ),
        encoding="utf-8",
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--config",
            str(path),
            "--do-train",
            "--start-frame",
            "1",
            "--do-predict",
        ]
    )
    cfg = config_from_args(args)
    assert cfg.do_train is True
    assert cfg.prediction.start_frame == 99
    assert cfg.do_predict is False


def test_cli_flags_override_defaults():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--do-train",
            "--no-show-plots",
            "--start-frame",
            "10",
            "--epochs",
            "5",
            "--feature-cols",
            "x",
            "y",
            "z",
        ]
    )
    cfg = config_from_args(args)
    assert cfg.do_train is True
    assert cfg.visualization.show_plots is False
    assert cfg.prediction.start_frame == 10
    assert cfg.training.epochs == 5
    assert cfg.feature_cols == ["x", "y", "z"]


def test_with_overrides_flat_leaves_and_groups():
    cfg = PipelineConfig().with_overrides(epochs=8, start_frame=3)
    assert cfg.training.epochs == 8
    assert cfg.prediction.start_frame == 3

    cfg2 = cfg.with_overrides(training=TrainingOptions(epochs=1), batch_size=16)
    assert cfg2.training.epochs == 1
    assert cfg2.training.batch_size == 16


def test_example_pipeline_json_loads():
    cfg = PipelineConfig.from_json(REPO_ROOT / "configs" / "pipeline_example.json")
    assert cfg.prediction.start_frame == 66
    assert cfg.training.epochs == 20
    assert cfg.visualization.save_figures is True


def test_run_pipeline_accepts_flat_overrides(monkeypatch):
    monkeypatch.setattr("bppm_dem_sm.pipeline.silence_tensorflow", lambda: None)
    results = run_pipeline(
        PipelineConfig(do_train=False, do_predict=False, do_metrics=False, do_visualization=False),
        epochs=9,
    )
    assert results["config"].training.epochs == 9
    assert results["config"].do_predict is False
