"""Command-line presentation layer for the segmentation lab."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .config import ExperimentConfig, load_config, to_dict
from .data import ManifestError, prepare_oxford_manifests, prepare_paired_manifests
from .evaluation.evaluate import evaluate_checkpoint
from .inference import Predictor
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
    source_format = args.source_format
    if source_format == "auto":
        annotations = Path(data_dir) / "annotations"
        source_format = "oxford-pet" if (annotations / "trainval.txt").is_file() and (annotations / "test.txt").is_file() else "paired"
    if source_format == "oxford-pet":
        paths = prepare_oxford_manifests(
            data_dir,
            manifest_dir,
            config.data.seed,
            config.data.label_map,
            set(config.data.ignore_values),
        )
    else:
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


def _evaluate(args: argparse.Namespace) -> int:
    output = evaluate_checkpoint(
        args.checkpoint,
        split=args.split,
        device=args.device,
        output_dir=args.output_dir,
        sample_limit=args.sample_limit,
        error_limit=args.error_limit,
        plot=args.plot,
    )
    print(f"evaluation artifacts: {output}")
    return 0


def _predict(args: argparse.Namespace) -> int:
    predictor = Predictor(args.checkpoint, device=args.device)
    source = Path(args.image)
    if source.is_dir():
        results = predictor.predict_directory(source, args.output)
        print(f"predicted {len(results)} images into {Path(args.output)}")
    else:
        paths = predictor.predict_path(source, args.output)
        print(f"raw mask: {paths['raw']}")
        print(f"overlay: {paths['overlay']}")
    return 0


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


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
    prepare.add_argument(
        "--source-format",
        choices=("auto", "paired", "oxford-pet"),
        default="auto",
        help="split convention (default: detect Oxford official files)",
    )
    prepare.set_defaults(func=_prepare_data)

    train = subparsers.add_parser("train", help="train a configured segmentation model")
    _add_config_args(train)
    _add_runtime_args(train)
    train.add_argument("--dry-run", action="store_true", help="verify one complete optimization batch")
    train.add_argument("--resume", type=str, default=None, help="last checkpoint to resume")
    train.set_defaults(func=_train)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a checkpoint on a manifest split")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=("train", "valid", "test"), default="test")
    evaluate.add_argument("--output-dir", default=None)
    evaluate.add_argument("--sample-limit", type=_nonnegative_int, default=8)
    evaluate.add_argument("--error-limit", type=_nonnegative_int, default=12)
    evaluate.add_argument("--plot", action="store_true")
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    evaluate.set_defaults(func=_evaluate)

    predict = subparsers.add_parser("predict", help="predict masks for an image or directory")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--image", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    predict.set_defaults(func=_predict)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ManifestError, RuntimeError, ValueError, KeyError) as exc:
        if args.debug:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
