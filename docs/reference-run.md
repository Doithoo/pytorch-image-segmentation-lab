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

The run used macOS 26.5.2 on Apple arm64 with MPS, Python 3.11.15, PyTorch
2.13.0, and torchvision 0.28.0. The training code revision was
`1deff1a03ba2d4549c538b4eef36df3868abe92b`.

Training started at `2026-08-16T04:35:04Z` and finished at
`2026-08-16T09:55:57Z`. End-to-end wall time was 19,253 seconds (5:20:53),
including one restart. The resumed epoch 15-25 process took 7,910.98 seconds;
the final test evaluation took 122.98 seconds.

The first MPS process completed epochs 1-14, then hit the Metal high-watermark
limit before epoch 15. It exited without corrupting artifacts and resumed from
the atomic epoch-14 `last.pt`. The safety limit was not disabled. This device
limitation can change runtime. The recorded revision restored model, optimizer,
scheduler, and global random states, but its separate DataLoader shuffle
generator was not checkpointed, so the resumed process restarted that shuffle
sequence. A post-run regression fix removed the uncheckpointed generator. The
measured result remains tied to the recorded revision and should not be treated
as a bitwise-equivalent uninterrupted run.

## Measured Results

The best checkpoint was epoch 24 with validation pet IoU `0.902836`. It was
evaluated once on all 3,669 official test samples and exceeded the required
test pet IoU of 0.70.

| Metric | Measured value |
| --- | ---: |
| Test pet IoU, global | 0.906669 |
| Test pet Dice, global | 0.951050 |
| Test background IoU, global | 0.950423 |
| Test mean IoU, global | 0.928546 |
| Test pixel accuracy | 0.966539 |
| Test pet IoU, mean per image | 0.905472 |
| Valid pixels | 210,899,866 |
| Ignored pixels | 29,551,718 |
| Test samples | 3,669 |
| Empty-target samples | 7 |

The committed evidence includes the [effective configuration](../artifacts/reference-unet/config.yaml),
[run metadata](../artifacts/reference-unet/run.yaml), [25-epoch metrics](../artifacts/reference-unet/metrics.csv),
[best checkpoint](../artifacts/reference-unet/best.pt), [evaluation summary](../artifacts/reference-unet/evaluation/metrics.json),
[per-class table](../artifacts/reference-unet/evaluation/per_class_metrics.csv),
[per-image table](../artifacts/reference-unet/evaluation/per_image_metrics.csv),
[loss curve](../artifacts/reference-unet/evaluation/loss_curve.png), and
[dataset preview](../artifacts/reference-unet/dataset_preview.png). Bounded
[sample overlays](../artifacts/reference-unet/evaluation/sample_overlays) and
[worst cases](../artifacts/reference-unet/evaluation/worst_cases) support visual
inspection.

## Interpretation

The result demonstrates that the checked-in pipeline learns useful pet masks
under the recorded conditions. It does not establish state of the art,
guarantee an identical score across hardware, or validate other datasets.

Inspect the loss curve, representative overlays, and worst cases alongside the
score. Background leakage, missed thin structures, and ignored trimap regions
reveal limitations hidden by one aggregate number. Seven official test masks
contain only raw value 2, so they have no target foreground. Visual inspection
also found inverted or spatially inconsistent upstream trimaps among the lowest
per-image IoU cases. These samples remain untouched and are included in every
reported aggregate.

## Reproduce Or Audit

Use [source.yaml](../data/manifests/source.yaml) and its manifest SHA-256 values
to confirm split identity: train `f7e7af...737171`, valid `872ed4...a5897d`,
and test `619642...f757c`. Full artifact hashes are recorded in `run.yaml`.
Compare the saved effective `config.yaml` rather than trusting the config
filename. Check checkpoint metadata for package and Git revisions. See
[Experiments](experiments.md) before changing a variable.
