# Dataset formats

[中文](dataset-formats.zh-CN.md) | [Using your data](../guides/using-your-data.md) | [Documentation](../README.md)

Run `uv run segment list-datasets` and `dataset-info NAME` before preparing data.

| Provider | Expected source |
| --- | --- |
| `oxford_pet` | Official Oxford images, trimaps, and split files |
| `paired_masks` | Matching `images/` and `masks/` trees |
| `split_folders` | Existing `train/valid/test` folders |
| `manifest` | Existing CSV files for all splits |
| `voc2012` | Extracted VOC 2012 layout |

Prepared data always has `dataset.yaml` plus train, valid, and test manifests. The metadata stores
the label schema, path policy, counts, and hashes. Built-in providers accept indexed `L/P/I;16`
masks. RGB color masks currently require an external Dataset factory that converts colors to class
IDs. Geometry options are `stretch`, `letterbox`, and optional paired crop; masks always use
nearest-neighbor interpolation.

For code-defined datasets, see [adding datasets](../guides/adding-datasets.md).
