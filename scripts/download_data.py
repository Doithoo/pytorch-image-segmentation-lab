"""Download and install the Oxford-IIIT Pet dataset."""

import argparse
import hashlib
import shutil
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path
from urllib.request import urlopen

COPY_BUFFER_SIZE = 1024 * 1024
IMAGE_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTATION_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
IMAGE_SHA256 = "67195c5e1c01f1ab5f9b6a5d22b8c27a580d896ece458917e61d459337fa318d"
ANNOTATION_SHA256 = "52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(COPY_BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"archive not found: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(f"checksum mismatch for {path}: expected {expected_sha256.lower()}, got {actual_sha256}")


def download_file(
    url: str,
    destination: Path,
    expected_sha256: str,
    force: bool = False,
) -> Path:
    destination = Path(destination)
    expected_sha256 = expected_sha256.lower()
    if destination.is_file() and not force:
        _verify_file(destination, expected_sha256)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    try:
        with urlopen(url) as response, partial.open("wb") as output:  # noqa: S310
            shutil.copyfileobj(response, output, length=COPY_BUFFER_SIZE)
        actual_sha256 = _sha256(partial)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"checksum mismatch for {url}: expected {expected_sha256}, got {actual_sha256}")
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return destination


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            member_path = (root / member.name).resolve()
            if member_path != root and root not in member_path.parents or member.issym() or member.islnk():
                raise ValueError(f"unsafe archive member: {member.name}")
        handle.extractall(destination, filter="data")


def install_oxford_archives(
    image_archive: Path,
    annotation_archive: Path,
    data_dir: Path,
) -> Path:
    data_dir = Path(data_dir)
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="oxford-pet-", dir=data_dir.parent) as temporary:
        extracted = Path(temporary)
        safe_extract_tar(image_archive, extracted / "images-archive")
        safe_extract_tar(annotation_archive, extracted / "annotations-archive")

        images = extracted / "images-archive" / "images"
        annotations = extracted / "annotations-archive" / "annotations"
        masks = annotations / "trimaps"
        if not images.is_dir() or not masks.is_dir():
            raise ValueError("archives do not contain the expected Oxford-IIIT Pet layout")

        staged = extracted / "installed"
        ignore_metadata = shutil.ignore_patterns("._*", ".DS_Store", "__MACOSX")
        shutil.copytree(images, staged / "images", ignore=ignore_metadata)
        shutil.copytree(masks, staged / "masks", ignore=ignore_metadata)
        metadata_dir = staged / "annotations"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        for source in annotations.iterdir():
            if source.name == "trimaps" or source.name.startswith("._"):
                continue
            target = metadata_dir / source.name
            if source.is_dir():
                shutil.copytree(source, target, ignore=ignore_metadata)
            else:
                shutil.copy2(source, target)

        if data_dir.exists():
            previous = extracted / "previous"
            data_dir.replace(previous)
            try:
                staged.replace(data_dir)
            except BaseException:
                previous.replace(data_dir)
                raise
        else:
            staged.replace(data_dir)
    return data_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download, verify, and install the Oxford-IIIT Pet dataset.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="installation directory (default: data/raw)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="download archives again even when verified copies exist",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify cached archives without downloading or installing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    archive_dir = args.data_dir.parent / "downloads"
    image_archive = archive_dir / "images.tar.gz"
    annotation_archive = archive_dir / "annotations.tar.gz"

    if args.verify_only:
        _verify_file(image_archive, IMAGE_SHA256)
        _verify_file(annotation_archive, ANNOTATION_SHA256)
        print(f"Verified Oxford-IIIT Pet archives in {archive_dir}")
        return 0

    download_file(IMAGE_URL, image_archive, IMAGE_SHA256, force=args.force)
    download_file(
        ANNOTATION_URL,
        annotation_archive,
        ANNOTATION_SHA256,
        force=args.force,
    )
    output = install_oxford_archives(image_archive, annotation_archive, args.data_dir)
    print(f"Installed Oxford-IIIT Pet in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
