"""Apply the same geometric transform to an image and its pixel mask.

Production counterpart: ``image_segmenter.data.transforms.PairedTransform``.
"""

from collections.abc import Sequence

import numpy as np
from PIL import Image

from image_segmenter.data.transforms import PairedTransform


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise ValueError("this example takes no arguments")
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :16, 0] = 255
    mask = np.full((32, 32), 2, dtype=np.uint8)
    mask[:, :16] = 1
    transform = PairedTransform(32, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], horizontal_flip=1.0)
    image_tensor, mask_tensor = transform(Image.fromarray(image), Image.fromarray(mask))
    red_left = image_tensor[0, :, :16].mean()
    red_right = image_tensor[0, :, 16:].mean()
    assert red_right > red_left
    torch_equal = bool((mask_tensor[:, 16:] == 1).all())
    assert torch_equal
    print("foreground before=left after=right")
    print(f"image={tuple(image_tensor.shape)} mask={tuple(mask_tensor.shape)} aligned={torch_equal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
