"""Restore preprocessing and model state, then save a prediction overlay.

Production counterpart: ``image_segmenter.inference.predictor.Predictor``.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from image_segmenter.inference.predictor import Predictor


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict one image from a saved checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("prediction.png"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    outputs = Predictor(args.checkpoint, device=args.device).predict_path(args.image, args.output)
    for kind, path in outputs.items():
        print(f"{kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
