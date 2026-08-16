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
from .data import (
    ManifestError,
    available_datasets,
    dataset_status,
    get_dataset_spec,
    inspect_prepared_data,
    prepare_dataset,
)
from .evaluation.evaluate import evaluate_checkpoint
from .inference import Predictor
from .models import available_models, create_model, get_model_spec, model_status
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
    provider = config.data.name
    if source_format == "auto" and provider in {"oxford_pet", "paired_masks"}:
        annotations = Path(data_dir) / "annotations"
        provider = (
            "oxford_pet"
            if (annotations / "trainval.txt").is_file() and (annotations / "test.txt").is_file()
            else "paired_masks"
        )
    elif source_format != "auto":
        provider = {"oxford-pet": "oxford_pet", "paired": "paired_masks"}[source_format]
    paths = prepare_dataset(
        provider,
        data_dir=data_dir,
        manifest_dir=manifest_dir,
        split_ratios=config.data.split_ratios,
        seed=config.data.seed,
        label_map=config.data.label_map,
        ignore_values=set(config.data.ignore_values),
        params=config.data.params,
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


def _list_models(args: argparse.Namespace) -> int:
    print("name\tprovider\tavailable\tpretrained")
    for name in available_models(provider=args.provider):
        spec = get_model_spec(name)
        status = model_status(name)
        print(
            f"{name}\t{spec.provider}\t{'yes' if status.available else 'no'}\t"
            f"{'yes' if spec.supports_pretrained else 'no'}"
        )
    return 0


def _model_info(args: argparse.Namespace) -> int:
    spec = get_model_spec(args.name)
    status = model_status(args.name)
    print(f"name: {spec.name}")
    print(f"provider: {spec.provider}")
    print(f"available: {'yes' if status.available else 'no'}")
    print(f"pretrained: {'yes' if spec.supports_pretrained else 'no'}")
    if spec.dependency is not None:
        print(f"dependency: {spec.dependency}")
    if status.reason is not None:
        print(f"reason: {status.reason}")
    if status.install_hint is not None:
        print(f"install: {status.install_hint}")
    if spec.parameters:
        print("parameters:")
        for name, description in sorted(spec.parameters.items()):
            print(f"  {name}: {description}")
    if spec.input_constraints:
        print("constraints:")
        for constraint in spec.input_constraints:
            print(f"  - {constraint}")
    if status.available:
        model = create_model(
            spec.name,
            in_channels=3,
            num_classes=2,
            pretrained=False,
            params={},
        )
        print(f"parameter_count: {sum(parameter.numel() for parameter in model.parameters())}")
        print(
            "trainable_parameter_count: "
            f"{sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}"
        )
    return 0


def _list_datasets(_args: argparse.Namespace) -> int:
    print("name\tprovider\tavailable")
    for name in available_datasets():
        spec = get_dataset_spec(name)
        status = dataset_status(name)
        print(f"{name}\t{spec.provider}\t{'yes' if status.available else 'no'}")
    return 0


def _dataset_info(args: argparse.Namespace) -> int:
    spec = get_dataset_spec(args.name)
    status = dataset_status(args.name)
    print(f"name: {spec.name}")
    print(f"provider: {spec.provider}")
    print(f"available: {'yes' if status.available else 'no'}")
    if status.reason is not None:
        print(f"reason: {status.reason}")
    if status.install_hint is not None:
        print(f"install: {status.install_hint}")
    if spec.parameters:
        print("parameters:")
        for name, description in sorted(spec.parameters.items()):
            print(f"  {name}: {description}")
    return 0


def _inspect_data(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    manifest_dir = args.manifest_dir or config.data.manifest_dir
    report = inspect_prepared_data(
        manifest_dir,
        split=args.split,
        limit=args.limit,
        preview=args.preview,
    )
    print(yaml.safe_dump(report, sort_keys=False), end="")
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

    list_models = subparsers.add_parser("list-models", help="list registered segmentation models")
    list_models.add_argument("--provider", default=None, help="show only one model provider")
    list_models.set_defaults(func=_list_models)

    model_info = subparsers.add_parser("model-info", help="inspect one registered model")
    model_info.add_argument("name", help="registered model name")
    model_info.set_defaults(func=_model_info)

    list_datasets = subparsers.add_parser("list-datasets", help="list registered dataset providers")
    list_datasets.set_defaults(func=_list_datasets)

    dataset_info = subparsers.add_parser("dataset-info", help="inspect one dataset provider")
    dataset_info.add_argument("name", help="registered dataset provider name")
    dataset_info.set_defaults(func=_dataset_info)

    inspect_data = subparsers.add_parser("inspect-data", help="inspect prepared labels and class ratios")
    _add_config_args(inspect_data)
    inspect_data.add_argument("--manifest-dir", default=None)
    inspect_data.add_argument("--split", choices=("train", "valid", "test"), default="train")
    inspect_data.add_argument("--limit", type=int, default=16)
    inspect_data.add_argument("--preview", default=None)
    inspect_data.set_defaults(func=_inspect_data)

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
