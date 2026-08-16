"""Follow raw labels through mapping, logits, loss, and class prediction.

Production counterparts: ``image_segmenter.data.dataset.map_mask`` and
``image_segmenter.training.losses.SegmentationLoss``.
"""

from collections.abc import Sequence

import torch

from image_segmenter.data.dataset import map_mask


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise ValueError("this example takes no arguments")
    raw_mask = torch.tensor(
        [[1, 1, 3, 2], [1, 2, 2, 2], [2, 2, 1, 1], [3, 2, 1, 1]]
    )
    target = map_mask(raw_mask, {1: 1, 2: 0}, [3], ignore_index=255)
    logits = torch.randn(1, 2, 4, 4, generator=torch.Generator().manual_seed(7))
    loss = torch.nn.functional.cross_entropy(logits, target.unsqueeze(0), ignore_index=255)
    prediction = logits.argmax(dim=1)
    print(f"raw={tuple(raw_mask.shape)} target={tuple(target.shape)}")
    print(f"logits={tuple(logits.shape)} prediction={tuple(prediction.shape)} loss={loss.item():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
