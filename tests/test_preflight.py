import pytest
import torch
from torch import nn

from image_segmenter.data import ClassDefinition, LabelSchema
from image_segmenter.preflight import validate_batch_contract


@pytest.fixture
def schema() -> LabelSchema:
    return LabelSchema(
        classes=(
            ClassDefinition(0, "background", (0, 0, 0)),
            ClassDefinition(1, "road", (128, 64, 128)),
            ClassDefinition(2, "vehicle", (0, 0, 142)),
        ),
        ignore_index=255,
    )


class TinyModel(nn.Module):
    def __init__(self, out_channels: int = 3, scale: int = 1) -> None:
        super().__init__()
        self.head = nn.Conv2d(3, out_channels, 1)
        self.scale = scale

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.head(image)
        return output[:, :, :: self.scale, :: self.scale]


def _batch() -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    return torch.randn(2, 3, 16, 16), torch.randint(0, 3, (2, 16, 16)), ["a", "b"]


@pytest.mark.parametrize(
    ("images", "message"),
    [
        (torch.randn(3, 16, 16), r"images must be float \[B,C,H,W\]"),
        (torch.ones(2, 3, 16, 16, dtype=torch.int64), "images must be float"),
        (torch.randn(2, 1, 16, 16), "expected 3 image channels"),
    ],
)
def test_preflight_rejects_invalid_images(images: torch.Tensor, message: str, schema: LabelSchema) -> None:
    with pytest.raises(ValueError, match=message):
        validate_batch_contract(
            TinyModel(), (images, torch.zeros(2, 16, 16, dtype=torch.long), ["a", "b"]), schema, expected_in_channels=3
        )


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (torch.zeros(2, 1, 16, 16, dtype=torch.long), r"targets must be long \[B,H,W\]"),
        (torch.zeros(2, 16, 16), "targets must be long"),
        (torch.full((2, 16, 16), 4, dtype=torch.long), r"target values.*\[4\].*background.*vehicle"),
        (torch.full((2, 16, 16), 255, dtype=torch.long), "all pixels use ignore_index"),
    ],
)
def test_preflight_rejects_invalid_targets(target: torch.Tensor, message: str, schema: LabelSchema) -> None:
    with pytest.raises(ValueError, match=message):
        validate_batch_contract(TinyModel(), (_batch()[0], target, ["a", "b"]), schema, expected_in_channels=3)


def test_preflight_rejects_mapping_output_without_adapter(schema: LabelSchema) -> None:
    class MappingModel(nn.Module):
        def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"out": image}

    with pytest.raises(ValueError, match="model returned a mapping.*output adapter"):
        validate_batch_contract(MappingModel(), _batch(), schema, expected_in_channels=3)


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (TinyModel(out_channels=2), "logits class channels.*expected 3"),
        (TinyModel(scale=2), "logits spatial shape.*target"),
    ],
)
def test_preflight_rejects_invalid_logits(model: nn.Module, message: str, schema: LabelSchema) -> None:
    with pytest.raises(ValueError, match=message):
        validate_batch_contract(model, _batch(), schema, expected_in_channels=3)


def test_preflight_returns_valid_multiclass_report(schema: LabelSchema) -> None:
    report = validate_batch_contract(TinyModel(), _batch(), schema, expected_in_channels=3)

    assert report.image_shape == (2, 3, 16, 16)
    assert report.logits_shape == (2, 3, 16, 16)
    assert report.class_names == ("background", "road", "vehicle")
    assert set(report.target_values) <= {0, 1, 2}
