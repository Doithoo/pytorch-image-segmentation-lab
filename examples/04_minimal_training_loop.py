"""Run one complete forward, loss, backward, and optimizer step.

Production counterparts: ``image_segmenter.models.unet.SmallUNet`` and
``image_segmenter.training.trainer.Trainer``.
"""

from collections.abc import Sequence

import torch

from image_segmenter.models.unet import SmallUNet
from image_segmenter.training.losses import SegmentationLoss


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise ValueError("this example takes no arguments")
    torch.manual_seed(7)
    model = SmallUNet(base_channels=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = SegmentationLoss(cross_entropy_weight=1.0, dice_weight=1.0, ignore_index=255)
    images = torch.rand(2, 3, 32, 32)
    targets = torch.randint(0, 2, (2, 32, 32))
    logits = model(images)
    loss, components = criterion(logits, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(
        f"logits={tuple(logits.shape)} loss={loss.item():.4f} "
        f"ce={components['cross_entropy'].item():.4f} dice={components['dice'].item():.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
