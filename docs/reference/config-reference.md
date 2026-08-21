# Configuration Reference

[简体中文](config-reference.zh-CN.md) | [Documentation](../README.md)

Run an existing config first. Use `segment show-config` and `config.py` for exact
defaults; this page explains decisions.

`data` controls paths, input size, RGB normalization, raw-value mapping to
contiguous class IDs, ignored values, generic split ratios, seed, workers, flip
probability, and optional sample limits. `resize_mode` supports `stretch` and
`letterbox`; `crop_size` applies paired training crops. Actual generic splits
must all be non-empty. Images use bilinear resize; masks use nearest-neighbor.

`model.name` selects a registered model from `segment list-models`. The first
path uses RGB input, but the prepared label schema can contain multiple classes.
U-Net width is `model.params.base_channels`; `model.expected_num_classes` is
checked against `dataset.yaml`. Optional SMP models require `model-zoo`.

`loss` combines non-negative CE and foreground Dice weights, with at least one
positive. `ignore_index` must differ from every mapped class and match prepared
dataset metadata.

`train` contains epochs, batch, AdamW LR/weight decay, `none` or `cosine`
scheduler, optional gradient clipping, AMP, selected metric, and seed. A cosine
checkpoint cannot be resumed with a changed planned epoch total.

Top-level `device` selects the runtime, while `output_dir` and `run_name` control
artifact paths. Give every experiment a distinct run name. `log_level` values
`debug` and `info` show per-epoch progress; `warning`, `error`, and `critical`
suppress that progress while preserving command results and errors.
