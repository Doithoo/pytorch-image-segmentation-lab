from pathlib import Path

import pytest

from image_segmenter.data import ClassDefinition, LabelSchema


def test_label_schema_round_trips() -> None:
    schema = LabelSchema(
        classes=(
            ClassDefinition(0, "background", (0, 0, 0)),
            ClassDefinition(1, "road", (128, 64, 128)),
            ClassDefinition(2, "vehicle", (0, 0, 142)),
        ),
        ignore_index=255,
    )

    assert LabelSchema.from_dict(schema.to_dict()) == schema
    assert schema.num_classes == 3
    assert schema.class_id("vehicle") == 2


@pytest.mark.parametrize(
    "classes",
    [
        (ClassDefinition(1, "road", (1, 2, 3)),),
        (
            ClassDefinition(0, "same", (1, 2, 3)),
            ClassDefinition(1, "same", (4, 5, 6)),
        ),
        (
            ClassDefinition(0, "a", (1, 2, 3)),
            ClassDefinition(1, "b", (1, 2, 3)),
        ),
    ],
)
def test_label_schema_rejects_noncanonical_classes(classes: tuple[ClassDefinition, ...]) -> None:
    with pytest.raises(ValueError):
        LabelSchema(classes=classes, ignore_index=255)


@pytest.mark.parametrize("color", [(-1, 0, 0), (0, 0, 256), (0, 0), (0, 0, 1.5)])
def test_class_definition_rejects_invalid_rgb(color: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="color"):
        ClassDefinition(0, "background", color)  # type: ignore[arg-type]


def test_label_schema_rejects_ignore_index_used_by_a_class() -> None:
    with pytest.raises(ValueError, match="ignore_index"):
        LabelSchema(classes=(ClassDefinition(0, "background", (0, 0, 0)),), ignore_index=0)


def test_label_schema_reads_and_writes_yaml(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "labels.yaml"
    schema = LabelSchema(
        classes=(
            ClassDefinition(0, "background", (0, 0, 0)),
            ClassDefinition(1, "object", (255, 255, 255)),
        ),
        ignore_index=65535,
    )

    schema.write_yaml(path)

    assert LabelSchema.read_yaml(path) == schema
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_label_schema_reports_unknown_class_name() -> None:
    schema = LabelSchema(classes=(ClassDefinition(0, "background", (0, 0, 0)),), ignore_index=255)

    with pytest.raises(KeyError, match="unknown class name: missing"):
        schema.class_id("missing")
