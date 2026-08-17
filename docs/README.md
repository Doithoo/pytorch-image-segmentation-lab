# Documentation

[简体中文](README.zh-CN.md) | [Project README](../README.md)

Use the project README for the shortest runnable path. Use this page when you
want a structured learning route or need to look up one part of the project.

## Start here

- [Learning path](tutorial/learning-path.md): move from tensor shapes to a
  controlled model comparison.
- [Tutorial index](tutorial/README.md): choose one of the six practical chapters
  by the question you are trying to answer.

## Guides

These pages describe tasks you perform in the repository.

| Task | Guide |
| --- | --- |
| Switch architectures | [Using models](guides/using-models.md) |
| Prepare another dataset | [Using your data](guides/using-your-data.md) |
| Connect a Python model factory | [Adding models](guides/adding-models.md) |
| Connect a custom Dataset | [Adding datasets](guides/adding-datasets.md) |
| Compare runs fairly | [Experiments](guides/experiments.md) |
| Diagnose a failure | [Troubleshooting](guides/troubleshooting.md) |

## Reference

Use these pages to look up supported names, fields, and formats.

- [Model zoo](reference/model-zoo.md)
- [Dataset formats](reference/dataset-formats.md)
- [Configuration fields](reference/config-reference.md)
- [VOC 2012 provider](reference/voc2012.md)

## Concepts

Use these pages when you want to understand why the code is structured this
way.

- [How segmentation works](concepts/how-it-works.md)
- [Code tour](concepts/code-tour.md)
- [Configuration flow](concepts/configuration-flow.md)

## Recorded run

The [recorded U-Net run](recorded-run/README.md) keeps one measured result with
its configuration, environment, metrics, and visual evidence. Other model pages
do not present unverified performance claims.

## Architecture

[Architecture decision 0001](architecture/0001-extensible-models-and-datasets.md)
records why models, datasets, label schemas, AMP, and run comparison use their
current extension contracts.
