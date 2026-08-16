# Reference Run

[简体中文](reference-run.zh-CN.md) | [Back to README](../README.md)

This page records measured evidence for `configs/reference_unet.yaml`. Values
are populated only from committed artifacts after the full run and untouched
official test evaluation complete.

## Protocol

- Dataset: Oxford-IIIT Pet official archives with pinned SHA-256 checksums.
- Split: official test unchanged; trainval split 80/20 within each breed using
  the configured seed.
- Labels: raw 1 to pet, raw 2 to background, raw 3 to ignored index 255.
- Selection: highest validation pet foreground IoU over 25 epochs.
- Test use: one final evaluation of the selected checkpoint.

```bash
uv run segment train --config configs/reference_unet.yaml
uv run segment evaluate --checkpoint artifacts/reference-unet/best.pt \
  --split test --plot
```

## Environment

Status: pending measured reference execution. The release record will include
OS, processor or accelerator, Python, PyTorch, torchvision, package revision,
wall-clock time, and device limitations.

## Measured Results

Status: pending measured reference execution. No score is claimed before the
committed `metrics.json` and manifest checksums exist. Release acceptance
requires test pet IoU at least 0.70.

The final table will name global pet IoU, pet Dice, mean IoU, pixel accuracy,
valid/ignored pixels, sample count, and mean per-image pet IoU separately.

## Interpretation

The result will demonstrate that the checked-in pipeline can learn useful pet
masks under the recorded conditions. It will not establish state of the art,
guarantee an identical score across hardware, or validate other datasets.

Inspect the loss curve, representative overlays, and worst cases alongside the
score. Background leakage, missed thin structures, and ignored trimap regions
can reveal limitations hidden by one aggregate number.

## Reproduce Or Audit

Use `source.yaml` and manifest SHA-256 values to confirm split identity. Compare
the saved effective `config.yaml` rather than trusting the config filename.
Check checkpoint metadata for package and Git revisions. See
[Experiments](experiments.md) before changing a variable.
