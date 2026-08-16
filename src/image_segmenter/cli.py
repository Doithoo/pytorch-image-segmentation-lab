"""Command-line presentation layer for the segmentation lab."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import yaml

from . import __version__
from .config import ExperimentConfig, load_config, to_dict
from .data import ManifestError, prepare_paired_manifests
from .training import train_from_config


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


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="override the artifact root")


def _resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    config = load_config(args.config, overrides=_parse_set(args.set))
    if getattr(args, "device", None) is not None:
        config.device = args.device
    if getattr(args, "output_dir", None) is not None:
        config.output_dir = args.output_dir
    return config


def _show_config(args: argparse.Namespace) -> int:
    print(yaml.safe_dump(to_dict(_resolve_config(args)), sort_keys=False), end="")
    return 0


def _prepare_data(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    data_dir = args.data_dir or config.data.data_dir
    manifest_dir = args.manifest_dir or config.data.manifest_dir
    paths = prepare_paired_manifests(
        data_dir,
        manifest_dir,
        config.data.split_ratios,
        config.data.seed,
        config.data.label_map,
        set(config.data.ignore_values),
    )
    for split in ("train", "valid", "test"):
        print(f"{split}: {paths[split]}")
    return 0


def _train(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    run_dir = train_from_config(config, resume=args.resume, dry_run=args.dry_run)
    if run_dir is not None:
        print(f"run artifacts: {run_dir}")
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

    prepare = subparsers.add_parser("prepare-data", help="validate paired data and create manifests")
    _add_config_args(prepare)
    prepare.add_argument("--data-dir", type=str, default=None, help="root containing images/ and masks/")
    prepare.add_argument("--manifest-dir", type=str, default=None, help="manifest output directory")
    prepare.set_defaults(func=_prepare_data)

    train = subparsers.add_parser("train", help="train a configured segmentation model")
    _add_config_args(train)
    _add_runtime_args(train)
    train.add_argument("--dry-run", action="store_true", help="verify one complete optimization batch")
    train.add_argument("--resume", type=str, default=None, help="last checkpoint to resume")
    train.set_defaults(func=_train)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ManifestError, ValueError, KeyError) as exc:
        if args.debug:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
