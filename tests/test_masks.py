import numpy as np
import pytest
import torch
from PIL import Image

from image_segmenter.data import ClassDefinition, LabelSchema, MaskDecoder


@pytest.fixture
def schema() -> LabelSchema:
    return LabelSchema(
        classes=(
            ClassDefinition(0, "background", (0, 0, 0)),
            ClassDefinition(1, "road", (128, 64, 128)),
            ClassDefinition(2, "vehicle", (0, 0, 142)),
        ),
        ignore_index=65535,
    )


@pytest.mark.parametrize("mode", ["L", "P", "I;16"])
def test_decoder_preserves_indexed_mask_modes(mode: str, schema: LabelSchema) -> None:
    values = np.array([[10, 11, 20, 255]], dtype=np.uint16 if mode == "I;16" else np.uint8)
    mask = Image.fromarray(values)
    if mode == "P":
        mask = mask.convert("P")
    assert mask.mode == mode
    decoder = MaskDecoder(schema, indexed_map={10: 1, 11: 1, 20: 2}, ignore_values=(255,))

    decoded = decoder(mask)

    assert decoded.dtype == torch.long
    assert decoded.tolist() == [[1, 1, 2, 65535]]


def test_decoder_maps_explicit_rgb_colors(schema: LabelSchema) -> None:
    values = np.array([[[0, 0, 0], [128, 64, 128], [0, 0, 142], [255, 0, 255]]], dtype=np.uint8)
    decoder = MaskDecoder(
        schema,
        rgb_map={(0, 0, 0): 0, (128, 64, 128): 1, (0, 0, 142): 2},
        ignore_colors=((255, 0, 255),),
    )

    assert decoder(Image.fromarray(values)).tolist() == [[0, 1, 2, 65535]]


def test_decoder_lists_every_unknown_index_in_stable_order(schema: LabelSchema) -> None:
    decoder = MaskDecoder(schema, indexed_map={10: 1})

    with pytest.raises(ValueError, match=r"unknown indexed mask values: \[8, 9\]"):
        decoder(Image.fromarray(np.array([[10, 9, 8]], dtype=np.uint8)))


def test_decoder_lists_every_unknown_rgb_color(schema: LabelSchema) -> None:
    decoder = MaskDecoder(schema, rgb_map={(0, 0, 0): 0})
    values = np.array([[[2, 3, 4], [1, 2, 3], [0, 0, 0]]], dtype=np.uint8)

    with pytest.raises(ValueError, match=r"unknown RGB mask colors: \[\(1, 2, 3\), \(2, 3, 4\)\]"):
        decoder(Image.fromarray(values))


def test_decoder_rejects_conflicting_rgb_mapping_entries(schema: LabelSchema) -> None:
    with pytest.raises(ValueError, match="RGB color mapping collision"):
        MaskDecoder(
            schema,
            rgb_map=[((0, 0, 0), 0), ((0, 0, 0), 1)],
        )


def test_decoder_rejects_mapping_to_unknown_class(schema: LabelSchema) -> None:
    with pytest.raises(ValueError, match="unknown class id 3"):
        MaskDecoder(schema, indexed_map={10: 3})
