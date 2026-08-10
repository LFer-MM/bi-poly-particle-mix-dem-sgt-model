"""Tests for PipelineConfig JSON loading and CLI argument resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bppm_dem_sm.cli import build_parser, config_from_args
from bppm_dem_sm.config import PipelineConfig


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


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown PipelineConfig keys"):
        PipelineConfig.from_dict({"not_a_field": 1})


def test_from_json_roundtrip(tmp_path: Path):
    original = PipelineConfig(
        do_train=True,
        do_predict=False,
        start_frame=42,
        data_dir=Path("data/processed/example"),
    )
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps(original.to_dict()), encoding="utf-8")

    loaded = PipelineConfig.from_json(path)
    assert loaded.do_train is True
    assert loaded.do_predict is False
    assert loaded.start_frame == 42
    assert loaded.data_dir == Path("data/processed/example")


def test_cli_config_json_ignores_other_flags(tmp_path: Path):
    path = tmp_path / "cfg.json"
    path.write_text(
        json.dumps({"do_train": True, "start_frame": 99, "do_predict": False}),
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
    assert cfg.start_frame == 99
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
    assert cfg.show_plots is False
    assert cfg.start_frame == 10
    assert cfg.epochs == 5
    assert cfg.feature_cols == ["x", "y", "z"]
