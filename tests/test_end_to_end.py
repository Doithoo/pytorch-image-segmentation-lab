from pathlib import Path

from image_segmenter.cli import main


def test_prepare_dry_run_and_train_through_cli(paired_data: Path, tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    artifacts = tmp_path / "artifacts"
    assert (
        main(
            [
                "prepare-data",
                "--data-dir",
                str(paired_data),
                "--manifest-dir",
                str(manifests),
                "--set",
                "data.split_ratios",
                "[0.8, 0.1, 0.1]",
            ]
        )
        == 0
    )
    common = [
        "--set",
        "data.manifest_dir",
        str(manifests),
        "--set",
        "data.image_size",
        "32",
        "--set",
        "data.max_train_samples",
        "4",
        "--set",
        "data.max_valid_samples",
        "1",
        "--set",
        "model.base_channels",
        "4",
        "--set",
        "train.batch_size",
        "2",
        "--set",
        "train.epochs",
        "1",
        "--set",
        "output_dir",
        str(artifacts),
        "--set",
        "run_name",
        "cli-run",
        "--device",
        "cpu",
    ]
    assert main(["train", "--dry-run", *common]) == 0
    assert main(["train", *common]) == 0
    assert (artifacts / "cli-run" / "best.pt").is_file()
