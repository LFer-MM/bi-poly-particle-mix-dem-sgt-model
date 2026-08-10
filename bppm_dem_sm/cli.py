"""Command-line interface for the end-to-end surrogate pipeline.

Config can be supplied either as individual flags or as a JSON file via
``--config``. When ``--config`` is set, all other pipeline config flags are
ignored.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Sequence, get_type_hints

from .config import PipelineConfig
from .pipeline import run_pipeline

_BOOL_FIELDS = frozenset(
    name for name, hint in get_type_hints(PipelineConfig).items() if hint is bool
)
_PATH_FIELDS = frozenset(
    name for name, hint in get_type_hints(PipelineConfig).items() if hint is Path
)
_SKIP_CLI_FIELDS = frozenset({"feature_cols"})


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for pipeline configuration.

    Emits one flag per ``PipelineConfig`` field (except ``feature_cols``, which
    uses ``--feature-cols``), plus ``--config`` for JSON loading.

    Returns:
        argparse.ArgumentParser: Configured parser for ``bppm-pipeline``.
    """
    parser = argparse.ArgumentParser(
        prog="bppm-pipeline",
        description=(
            "Run the RNN surrogate pipeline (train / predict / metrics / "
            "visualization). Pass ``--config path.json`` to load settings from "
            "JSON; when set, other pipeline flags are ignored."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        metavar="JSON",
        help=(
            "JSON file with PipelineConfig fields. When provided, all other "
            "pipeline config arguments are ignored."
        ),
    )

    type_by_name = get_type_hints(PipelineConfig)
    for f in fields(PipelineConfig):
        if f.name in _SKIP_CLI_FIELDS:
            continue
        flag = f"--{f.name.replace('_', '-')}"
        hint = type_by_name[f.name]
        default_repr = f.default
        if f.name in _BOOL_FIELDS:
            parser.add_argument(
                flag,
                action=argparse.BooleanOptionalAction,
                default=None,
                help=f"Override {f.name} (default: {default_repr}).",
            )
        elif f.name in _PATH_FIELDS:
            parser.add_argument(
                flag,
                type=Path,
                default=None,
                help=f"Override {f.name} (default: {default_repr}).",
            )
        elif hint is int:
            parser.add_argument(
                flag,
                type=int,
                default=None,
                help=f"Override {f.name} (default: {default_repr}).",
            )
        elif hint is float:
            parser.add_argument(
                flag,
                type=float,
                default=None,
                help=f"Override {f.name} (default: {default_repr}).",
            )
        else:
            parser.add_argument(
                flag,
                type=str,
                default=None,
                help=f"Override {f.name} (default: {default_repr}).",
            )

    parser.add_argument(
        "--feature-cols",
        nargs="+",
        default=None,
        metavar="COL",
        help="Override feature_cols (space-separated column names).",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Resolve ``PipelineConfig`` from parsed CLI args.

    If ``args.config`` is set, load only from that JSON file; all other
    pipeline flags are ignored.

    Args:
        args: Namespace produced by :func:`build_parser`.

    Returns:
        PipelineConfig: Defaults with any non-``None`` CLI overrides applied,
        or the JSON-loaded config when ``--config`` is present.
    """
    if args.config is not None:
        return PipelineConfig.from_json(args.config)

    overrides: dict[str, Any] = {}
    for f in fields(PipelineConfig):
        value = getattr(args, f.name, None)
        if value is not None:
            overrides[f.name] = value
    return replace(PipelineConfig(), **overrides) if overrides else PipelineConfig()


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI args, run the pipeline, return process exit code.

    Args:
        argv: Optional argument list (as for ``ArgumentParser.parse_args``);
            ``None`` uses ``sys.argv``.

    Returns:
        int: ``0`` on success.
    """
    from .tf_quiet import silence_tensorflow

    silence_tensorflow()
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    run_pipeline(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
