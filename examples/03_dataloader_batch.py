"""Inspect how segmentation samples become a mini-batch.

Production counterpart: ``image_segmenter.data.dataset.SegmentationDataset``.
"""

from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader, TensorDataset


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise ValueError("this example takes no arguments")
    images = torch.rand(4, 3, 32, 32, generator=torch.Generator().manual_seed(7))
    masks = torch.randint(0, 2, (4, 32, 32), generator=torch.Generator().manual_seed(8))
    loader = DataLoader(TensorDataset(images, masks), batch_size=2, shuffle=False)
    image_batch, mask_batch = next(iter(loader))
    print(f"images [B,C,H,W] = {tuple(image_batch.shape)}")
    print(f"masks  [B,H,W]   = {tuple(mask_batch.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
