# Reproducible Segmentation Experiments

[简体中文](experiments.zh-CN.md) | [Back to README](../README.md)

Write one question, a prediction, fixed settings, the single change, and the
validation evidence before running. The test split is reserved for the final
selected setting.

## Baseline and focused comparisons

```bash
uv run segment train --config configs/learning_minimal.yaml \
  --set run_name baseline-ce
```

Compare CE against CE+Dice:

```bash
uv run segment train --config configs/learning_minimal.yaml \
  --set run_name exp-ce
uv run segment train --config configs/learning_minimal.yaml \
  --set loss.cross_entropy_weight 0.5 \
  --set loss.dice_weight 0.5 \
  --set run_name exp-ce-dice
```

Other useful single changes are `model.base_channels: 8` versus 16 and
`data.horizontal_flip: 0.0` versus 0.5. Keep manifests, seed, epochs, limits,
and preprocessing fixed. Compare validation pet IoU, curves, runtime, and the
same sample overlays. Loss values are not directly comparable when the loss
definition changes.

The default U-Net and pretrained DeepLab configs are complete-system
comparisons, not pure architecture ablations: initialization, batch, LR, and
epochs differ. State that limitation.

Avoid changed memberships, multiple simultaneous changes, selecting by test,
keeping only a lucky seed, and universal claims from one run. A report should
include manifest hashes, resolved configs, validation results, final test only
after selection, matched visual cases, environment/runtime, and limitations.

See the [reference run](reference-run.md) for a complete record.
