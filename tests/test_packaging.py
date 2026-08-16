from importlib.metadata import version

import image_segmenter


def test_package_and_distribution_versions_match() -> None:
    assert image_segmenter.__version__ == "0.1.0"
    assert version("image-segmenter") == image_segmenter.__version__
