# Experiments

[简体中文](experiments.zh-CN.md) | [Back to README](../README.md)

An experiment answers one written question. Keep data membership, seed,
evaluation protocol, and all unrelated fields fixed.

## Establish A Baseline

Run the learning configuration unchanged first. Save its resolved config,
manifest hashes, metrics, and best checkpoint. A failed or slow baseline is
useful information; do not conceal it by changing several knobs at once.

```bash
uv run segment show-config --config configs/learning_minimal.yaml
uv run segment train --config configs/learning_minimal.yaml
```

## Change One Variable

Examples of focused questions:

- Does foreground Dice improve pet IoU over cross-entropy alone?
- Does a wider U-Net help enough to justify time and memory?
- Does horizontal flipping improve validation performance?
- How does the handwritten U-Net compare with DeepLabV3 under clearly recorded
  pretrained-weight conditions?

Give each run a distinct name. Use `--set` for a compact one-variable change,
then inspect the printed effective config.

```bash
uv run segment train --config configs/learning_minimal.yaml \
  --set loss.cross_entropy_weight 0.5 \
  --set loss.dice_weight 0.5 --set run_name ce-dice
```

## Compare Evidence

Compare validation pet IoU for model selection, training/validation curves for
optimization behavior, and worst-case overlays for error character. Record run
time and device. Do not report test results for every attempted setting: that
turns the test split into a validation set.

Global pet IoU and mean per-image pet IoU answer different questions. Always
name the statistic. An empty-foreground image is excluded from the per-image
mean when both target and prediction are empty.

## Avoid Invalid Comparisons

Do not compare runs with different split membership, hidden preprocessing,
sample limits, or pretrained initialization as if only the named model changed.
Do not select a result because its test score is highest. Do not infer a general
claim from one seed without stating that limitation.

Use [Reference Run](reference-run.md) as the reporting template and
[Configuration Flow](configuration-flow.md) to audit overrides.
