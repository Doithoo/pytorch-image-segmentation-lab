import hashlib
import io
import tarfile
from pathlib import Path

import pytest
from scripts import download_data
from scripts.download_data import download_file, install_oxford_archives, safe_extract_tar


def test_official_archive_sha256_values_are_pinned() -> None:
    assert download_data.IMAGE_SHA256 == "67195c5e1c01f1ab5f9b6a5d22b8c27a580d896ece458917e61d459337fa318d"
    assert download_data.ANNOTATION_SHA256 == "52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91"


def test_checksum_failure_does_not_replace_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"verified bytes")
    destination = tmp_path / "download.bin"
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_file(source.as_uri(), destination, expected_sha256="0" * 64)
    assert not destination.exists()
    assert not destination.with_suffix(".bin.part").exists()


def test_verified_local_download_is_atomic(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"verified bytes")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = download_file(source.as_uri(), tmp_path / "download.bin", expected)
    assert destination.read_bytes() == source.read_bytes()


def test_force_replaces_an_existing_archive(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"new bytes")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "download.bin"
    destination.write_bytes(b"old bytes")
    download_file(source.as_uri(), destination, expected, force=True)
    assert destination.read_bytes() == b"new bytes"


def test_safe_extract_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("../escape.txt")
        payload = b"no"
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="unsafe archive member"):
        safe_extract_tar(archive, tmp_path / "extract")


def test_install_rearranges_official_archive_layout(tmp_path: Path) -> None:
    image_archive = tmp_path / "images.tar.gz"
    annotation_archive = tmp_path / "annotations.tar.gz"
    with tarfile.open(image_archive, "w:gz") as handle:
        for name, payload in {
            "images/pet_01.jpg": b"image",
            "images/._pet_01.jpg": b"resource fork",
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    with tarfile.open(annotation_archive, "w:gz") as handle:
        for name, payload in {
            "annotations/trimaps/pet_01.png": b"mask",
            "annotations/trimaps/._pet_01.png": b"resource fork",
            "annotations/trainval.txt": b"pet_01 1 1 1\n",
            "annotations/test.txt": b"pet_02 1 1 1\n",
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    output = install_oxford_archives(image_archive, annotation_archive, tmp_path / "raw")
    assert (output / "images" / "pet_01.jpg").read_bytes() == b"image"
    assert (output / "masks" / "pet_01.png").read_bytes() == b"mask"
    assert (output / "annotations" / "trainval.txt").is_file()
    assert not (output / "images" / "._pet_01.jpg").exists()
    assert not (output / "masks" / "._pet_01.png").exists()


def test_verify_only_checks_cached_archives_without_installing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "raw"
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    images = downloads / "images.tar.gz"
    annotations = downloads / "annotations.tar.gz"
    images.write_bytes(b"images")
    annotations.write_bytes(b"annotations")
    monkeypatch.setattr(download_data, "IMAGE_SHA256", hashlib.sha256(b"images").hexdigest())
    monkeypatch.setattr(
        download_data,
        "ANNOTATION_SHA256",
        hashlib.sha256(b"annotations").hexdigest(),
    )

    assert download_data.main(["--data-dir", str(data_dir), "--verify-only"]) == 0
    assert not data_dir.exists()
