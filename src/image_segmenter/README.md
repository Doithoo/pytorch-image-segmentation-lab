# How to Read the Package

[简体中文](README.zh-CN.md) | [Code tour](../../docs/concepts/code-tour.md)

```text
segment train -> CLI -> resolved config -> train_from_config
              -> Trainer.fit -> save best.pt / last.pt
```

| Module | Input | Output | Read when |
| --- | --- | --- | --- |
| `config.py` | defaults/YAML/overrides | resolved config | settings surprise you |
| `data/` | manifests and pairs | tensor batches | learning data flow |
| `models/` | model settings | logits model | studying U-Net/models |
| `training/` | config/loaders/model | checkpoints/metrics | studying optimization |
| `evaluation/` | checkpoint/labeled split | reports/errors | analyzing results |
| `inference/` | checkpoint/new images | mask outputs | using a model |
| `cli.py` | terminal args | calls above | studying application entry |

Read dataset, transforms, U-Net, losses, then Trainer. Return to strict config,
manifest audit, RNG persistence, and argparse details when a question requires
them. Keep masks as integer classes, models returning logits, and geometric
randomness shared between image and mask.
