"""Command-line presentation layer for the segmentation lab."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import yaml

from . import __version__
from .config import ExperimentConfig, load_config, to_dict


def _parse_set(overrides: list[list[str]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, raw_value in overrides:
        try:
            parsed[key] = json.loads(raw_value)
        except json.JSONDecodeError:
            parsed[key] = raw_value
    return parsed


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument(
        "--set",
        nargs=2,
        action="append",
        default=[],
        metavar=("KEY", "VALUE"),
        help="override a dotted config key",
    )


def _resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    return load_config(args.config, overrides=_parse_set(args.set))


def _show_config(args: argparse.Namespace) -> int:
    print(yaml.safe_dump(to_dict(_resolve_config(args)), sort_keys=False), end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and translate expected domain errors into exit code 2."""
    parser = argparse.ArgumentParser(prog="segment")
    parser.add_argument("--version", action="version", version=f"image-segmenter {__version__}")
    parser.add_argument("--debug", action="store_true", help="show tracebacks for command errors")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show = subparsers.add_parser("show-config", help="print the resolved YAML configuration")
    _add_config_args(show)
    show.set_defaults(func=_show_config)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        if args.debug:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
