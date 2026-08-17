from pathlib import Path

import pytest

from image_segmenter.config import config_from_dict, dump_config, load_config, to_dict


def test_defaults_match_binary_segmentation_contract() -> None:
    cfg = load_config()
    assert cfg.config_version == 2
    assert cfg.data.name == "oxford_pet"
    assert cfg.data.image_size == 128
    assert cfg.data.label_map == {1: 1, 2: 0}
    assert cfg.data.ignore_values == [3]
    assert cfg.loss.ignore_index == 255
    assert cfg.model.name == "unet_small"
    assert cfg.model.num_classes == 2


def test_v1_model_fields_migrate_to_v2() -> None:
    cfg = config_from_dict({"model": {"name": "unet_small", "base_channels": 8, "num_classes": 2}})

    assert cfg.config_version == 2
    assert cfg.model.params == {"base_channels": 8}
    assert cfg.model.expected_num_classes == 2
    assert "base_channels" not in to_dict(cfg)["model"]
    assert "num_classes" not in to_dict(cfg)["model"]


def test_v2_accepts_provider_parameters_and_amp() -> None:
    cfg = config_from_dict(
        {
            "config_version": 2,
            "model": {"name": "fcn_resnet50", "pretrained": False, "params": {}},
            "data": {"name": "oxford_pet", "params": {}},
            "train": {"amp": True, "best_metric": "class_iou:pet"},
        }
    )

    assert cfg.train.amp is True
    assert cfg.train.best_metric == "class_iou:pet"


@pytest.mark.parametrize("version", [0, 3, "2", True])
def test_unknown_or_invalid_config_version_is_rejected(version: object) -> None:
    with pytest.raises(ValueError, match="config_version"):
        config_from_dict({"config_version": version})


def test_v2_rejects_legacy_model_fields() -> None:
    with pytest.raises(ValueError, match="model.base_channels"):
        config_from_dict({"config_version": 2, "model": {"base_channels": 8}})


def test_nested_parameter_override_updates_v2_config() -> None:
    cfg = load_config(overrides={"model.params.base_channels": 12, "data.params.source": "local"})

    assert cfg.model.params == {"base_channels": 12}
    assert cfg.data.params == {"source": "local"}


def test_yaml_then_dotted_override_precedence(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("train:\n  epochs: 5\n")
    cfg = load_config(path, {"train.epochs": 2, "model.base_channels": 8})
    assert cfg.train.epochs == 2
    assert cfg.model.base_channels == 8


def test_yaml_label_map_keys_become_integers(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text('data:\n  label_map: {"10": 1, "20": 0}\n')
    assert load_config(path).data.label_map == {10: 1, 20: 0}


def test_multiclass_label_map_uses_contiguous_target_ids() -> None:
    config = config_from_dict(
        {
            "config_version": 2,
            "data": {"label_map": {0: 0, 10: 1, 20: 2}, "ignore_values": [255]},
            "model": {"expected_num_classes": 3},
        }
    )

    assert config.data.label_map == {0: 0, 10: 1, 20: 2}


@pytest.mark.parametrize("label_map", [{0: 0, 2: 2}, {0: -1, 1: 0}, {0: False, 1: 1}])
def test_label_map_rejects_noncontiguous_or_boolean_target_ids(label_map: dict[int, int]) -> None:
    with pytest.raises(ValueError, match="invalid data.label_map"):
        config_from_dict({"config_version": 2, "data": {"label_map": label_map}})


def test_ignore_index_must_not_overlap_a_multiclass_target() -> None:
    with pytest.raises(ValueError, match="loss.ignore_index must differ from trainable classes"):
        config_from_dict(
            {
                "config_version": 2,
                "data": {"label_map": {0: 0, 1: 1, 2: 2}, "ignore_values": [255]},
                "model": {"expected_num_classes": 3},
                "loss": {"ignore_index": 2},
            }
        )


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("train:\n  epohs: 2\n")
    with pytest.raises(ValueError, match="unknown config key: train.epohs"):
        load_config(path)


@pytest.mark.parametrize("size", [0, 127, 130])
def test_unet_image_size_must_be_positive_and_divisible_by_16(size: int) -> None:
    with pytest.raises(ValueError, match="data.image_size"):
        load_config(overrides={"data.image_size": size})


def test_dumped_config_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "resolved.yaml"
    original = load_config(overrides={"run_name": "round-trip", "data.image_size": 32})
    dump_config(original, path)
    restored = load_config(path)
    assert restored == original


def test_config_dictionary_round_trips() -> None:
    original = load_config(overrides={"data.image_size": 32, "model.base_channels": 8})
    assert config_from_dict(to_dict(original)) == original


def test_config_from_dict_rejects_unknown_nested_key() -> None:
    raw = to_dict(load_config())
    raw["model"]["mystery"] = True
    with pytest.raises(ValueError, match="unknown config key: model.mystery"):
        config_from_dict(raw)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("model.in_channels", True),
        ("model.base_channels", True),
        ("train.epochs", True),
        ("train.batch_size", True),
        ("data.seed", True),
        ("data.seed", -1),
        ("data.seed", 2**32),
        ("train.seed", "42"),
    ],
)
def test_integer_fields_reject_booleans_and_invalid_seeds(key: str, value: object) -> None:
    with pytest.raises(ValueError, match=key):
        load_config(overrides={key: value})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("data.data_dir", 1),
        ("data.manifest_dir", None),
        ("model.name", "missing"),
        ("model.num_classes", 2.0),
        ("loss.name", []),
        ("loss.ignore_index", "255"),
        ("loss.ignore_index", []),
        ("train.scheduler", []),
        ("device", []),
        ("log_level", []),
    ],
)
def test_invalid_field_types_raise_clear_value_errors(key: str, value: object) -> None:
    with pytest.raises(ValueError, match=key):
        load_config(overrides={key: value})
