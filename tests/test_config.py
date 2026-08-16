from pathlib import Path

import pytest

from image_segmenter.config import config_from_dict, dump_config, load_config, to_dict


def test_defaults_match_binary_segmentation_contract() -> None:
    cfg = load_config()
    assert cfg.data.image_size == 128
    assert cfg.data.label_map == {1: 1, 2: 0}
    assert cfg.data.ignore_values == [3]
    assert cfg.loss.ignore_index == 255
    assert cfg.model.name == "unet_small"
    assert cfg.model.num_classes == 2


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
