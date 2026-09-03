"""Command-line interface for the end-to-end surrogate pipeline.

Config can be supplied either as individual flags or as a JSON file via
``--config``. When ``--config`` is set, all other pipeline config flags are
ignored. Flags stay flat (``--epochs``); they are mapped onto nested option
groups on :class:`~bppm_dem_sm.config.PipelineConfig`.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any, Sequence, get_type_hints

from .config import PipelineConfig, _option_group_types
from .pipeline import run_pipeline

_SKIP_CLI_FIELDS = frozenset({"feature_cols"})


def _bool_fields(cls: type) -> frozenset[str]:
    return frozenset(name for name, hint in get_type_hints(cls).items() if hint is bool)


def _path_fields(cls: type) -> frozenset[str]:
    return frozenset(name for name, hint in get_type_hints(cls).items() if hint is Path)


def _add_config_flag(container: Any, f, owning_cls: type) -> None:
    """Add one typed flag for a dataclass field; dest is the leaf field name."""
    flag = f"--{f.name.replace('_', '-')}"
    hint = get_type_hints(owning_cls)[f.name]
    default_repr = f.default
    if f.name in _bool_fields(owning_cls):
        container.add_argument(
            flag,
            action=argparse.BooleanOptionalAction,
            default=None,
            help=f"Override {f.name} (default: {default_repr}).",
        )
    elif f.name in _path_fields(owning_cls):
        container.add_argument(
            flag,
            type=Path,
            default=None,
            help=f"Override {f.name} (default: {default_repr}).",
        )
    elif hint is int:
        container.add_argument(
            flag,
            type=int,
            default=None,
            help=f"Override {f.name} (default: {default_repr}).",
        )
    elif hint is float:
        container.add_argument(
            flag,
            type=float,
            default=None,
            help=f"Override {f.name} (default: {default_repr}).",
        )
    else:
        container.add_argument(
            flag,
            type=str,
            default=None,
            help=f"Override {f.name} (default: {default_repr}).",
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for pipeline configuration.

    Emits one flag per core ``PipelineConfig`` field and per nested options
    leaf (except ``feature_cols``, which uses ``--feature-cols``), plus
    ``--config`` for JSON loading. Help text is grouped to match the nested
    JSON layout; flag names stay independent (``--epochs``, not
    ``--training-epochs``).

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
            "JSON file with PipelineConfig fields (nested option groups). "
            "When provided, all other pipeline config arguments are ignored."
        ),
    )

    core = parser.add_argument_group("pipeline")
    group_types = _option_group_types()
    for f in fields(PipelineConfig):
        if f.name in _SKIP_CLI_FIELDS or f.name in group_types:
            continue
        _add_config_flag(core, f, PipelineConfig)

    core.add_argument(
        "--feature-cols",
        nargs="+",
        default=None,
        metavar="COL",
        dest="feature_cols",
        help="Override feature_cols (space-separated column names).",
    )

    for group_name, group_cls in group_types.items():
        arg_group = parser.add_argument_group(group_name)
        for sf in fields(group_cls):
            _add_config_flag(arg_group, sf, group_cls)

    return parser


def _leaf_override_names() -> list[str]:
    """CLI dest names that map onto ``PipelineConfig.with_overrides``."""
    names = []
    group_types = _option_group_types()
    for f in fields(PipelineConfig):
        if f.name in group_types:
            continue
        names.append(f.name)
    for group_cls in group_types.values():
        names.extend(sf.name for sf in fields(group_cls))
    return names


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Resolve ``PipelineConfig`` from parsed CLI args.

    If ``args.config`` is set, load only from that JSON file; all other
    pipeline flags are ignored. Otherwise apply any non-``None`` flags via
    :meth:`PipelineConfig.with_overrides` (flat leaf names).

    Args:
        args: Namespace produced by :func:`build_parser`.

    Returns:
        PipelineConfig: Defaults with any non-``None`` CLI overrides applied,
        or the JSON-loaded config when ``--config`` is present.
    """
    if args.config is not None:
        return PipelineConfig.from_json(args.config)

    overrides: dict[str, Any] = {}
    for name in _leaf_override_names():
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value
    return PipelineConfig().with_overrides(**overrides) if overrides else PipelineConfig()


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
