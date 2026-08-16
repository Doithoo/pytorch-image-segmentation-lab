"""Plot training metrics from a completed or running experiment."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from image_segmenter.evaluation.evaluate import plot_loss_curve


def plot_metrics(metrics_csv: str | Path, output: str | Path) -> Path:
    """Create the standard train and validation loss curve."""
    return plot_loss_curve(metrics_csv, output)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot train and validation loss from metrics.csv.")
    parser.add_argument("metrics_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("loss_curve.png"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = plot_metrics(args.metrics_csv, args.output)
    print(f"Saved metric plot to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
