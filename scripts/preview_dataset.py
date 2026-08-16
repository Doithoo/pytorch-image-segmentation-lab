"""Create a visual preview from a prepared segmentation manifest."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from image_segmenter.data.inspection import create_dataset_preview


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview samples from a segmentation manifest.")
    parser.add_argument("manifest", type=Path, help="CSV manifest to preview")
    parser.add_argument("--output", type=Path, default=Path("preview.png"))
    parser.add_argument("--limit", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = create_dataset_preview(args.manifest, args.output, limit=args.limit)
    print(f"Saved dataset preview to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
