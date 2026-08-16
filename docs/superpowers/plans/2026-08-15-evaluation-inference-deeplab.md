# Evaluation, Inference, And DeepLabV3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the stable U-Net training core with trustworthy evaluation artifacts, checkpoint-only prediction, and a torchvision DeepLabV3 comparison model.

**Architecture:** Reconstruct runtime configuration from self-contained checkpoints, accumulate global and per-image metrics without retaining the dataset, and keep rendering separate from metric calculation. Adapt torchvision dictionary outputs at the model boundary so the trainer, evaluator, and predictor continue to consume direct `[B,2,H,W]` logits.

**Tech Stack:** Python 3.10-3.12, PyTorch, torchvision, NumPy, Pillow, optional Matplotlib, pytest, Ruff

---

### Task 1: Restore Typed Configuration From Checkpoints

**Files:**
- Modify: `src/image_segmenter/config.py`
- Modify: `tests/test_config.py`

- [ ] Write `test_config_from_dict_rejects_unknown_nested_key` and a round-trip
  test using `to_dict(load_config())`; run it and verify import failure for
  `config_from_dict`.
- [ ] Implement `config_from_dict(raw)` through the same `_from_dict`, override,
  and `_validate_config` path as YAML loading. Refactor `load_config` to parse
  YAML and delegate, so file and checkpoint dictionaries cannot drift.
- [ ] Run `uv run pytest tests/test_config.py -v` and Ruff; expect all tests and
  lint to pass.
- [ ] Commit with `feat(config): Restore typed checkpoint configuration`.

### Task 2: Per-Image Metrics And Serializable Results

**Files:**
- Modify: `src/image_segmenter/evaluation/metrics.py`
- Modify: `src/image_segmenter/evaluation/__init__.py`
- Modify: `tests/test_metrics.py`

- [ ] Add failing tests for `per_image_scores(prediction, target, 255)` covering
  one false positive, ignored pixels, and both-empty foreground returning NaN.
- [ ] Add a failing test that `serializable_metrics` recursively converts NaN to
  `None` while preserving finite values and integer counts.
- [ ] Implement per-image pet intersection/union/Dice, foreground ratios, valid
  and ignored counts. Return one dictionary per batch item with keys
  `pet_iou`, `pet_dice`, `target_foreground_ratio`,
  `predicted_foreground_ratio`, `valid_pixels`, and `ignored_pixels`.
- [ ] Implement recursive JSON normalization for dictionaries and lists.
- [ ] Run focused tests and Ruff; commit with
  `feat(evaluation): Add per-image segmentation metrics`.

### Task 3: Error Overlays And Comparison Grids

**Files:**
- Create: `src/image_segmenter/evaluation/visualization.py`
- Modify: `src/image_segmenter/evaluation/__init__.py`
- Create: `tests/test_visualization.py`

- [ ] Write failing tests with a 4 x 4 synthetic image/mask pair. Assert raw
  class mask output contains only 0/1, color output is RGB, overlay distinguishes
  true positive, false positive, false negative, and ignored pixels, and the
  four-panel comparison has stable dimensions.
- [ ] Implement `colorize_mask`, `error_overlay`, `prediction_overlay`, and
  `comparison_grid` with Pillow and NumPy. Use a documented color palette and a
  fixed legend band; never mutate inputs.
- [ ] Implement `save_prediction_outputs(raw_path, original, prediction)` to
  write raw class indices plus `<stem>.color.png` and `<stem>.overlay.png`.
- [ ] Run focused tests and Ruff; commit with
  `feat(evaluation): Render inspectable mask errors`.

### Task 4: Complete Checkpoint Evaluation

**Files:**
- Create: `src/image_segmenter/evaluation/evaluate.py`
- Modify: `src/image_segmenter/evaluation/__init__.py`
- Create: `tests/test_evaluation.py`

- [ ] Add a fixture helper that trains the existing tiny run, then write a
  failing test for `evaluate_checkpoint(best.pt, split="valid")`. Assert
  `metrics.json`, `per_class_metrics.csv`, `per_image_metrics.csv`,
  `sample_overlays/`, and `worst_cases/` exist.
- [ ] Implement checkpoint model reconstruction, manifest-backed DataLoader,
  global streaming metrics, per-image rows, and deterministic sample IDs.
- [ ] Write global metrics as strict JSON with NaN converted to null. Write
  per-class rows for background/pet and per-image rows sorted by sample ID.
- [ ] Save up to `sample_limit` deterministic grids and `error_limit` lowest
  finite-IoU grids. Use transformed/denormalized images at network resolution;
  ignored pixels must never be counted as errors.
- [ ] Add lazy `plot_loss_curve(metrics.csv, output.png)` using Matplotlib only
  when `plot=True`; report installation of `[plot]` when unavailable.
- [ ] Run focused tests and all existing training tests; commit with
  `feat(evaluation): Evaluate checkpoints with visual evidence`.

### Task 5: Checkpoint-Only Prediction

**Files:**
- Modify: `src/image_segmenter/data/transforms.py`
- Create: `src/image_segmenter/inference/__init__.py`
- Create: `src/image_segmenter/inference/predictor.py`
- Create: `tests/test_inference.py`

- [ ] Add failing tests for an `ImageTransform` that returns `[3,H,W]`, and for
  `Predictor(best.pt).predict_path(image, output)` restoring the original image
  dimensions and writing raw/color/overlay PNGs.
- [ ] Implement image-only resize/normalize from checkpoint config. Implement a
  predictor that loads model/config/state on the requested device, uses argmax,
  restores raw mask size with nearest-neighbor interpolation, and writes three
  distinct outputs.
- [ ] Implement directory prediction preserving relative input paths and
  rejecting output paths that would overwrite input files.
- [ ] Run focused tests and Ruff; commit with
  `feat(inference): Predict masks from self-contained checkpoints`.

### Task 6: Torchvision DeepLabV3 Adapter

**Files:**
- Create: `src/image_segmenter/models/deeplab.py`
- Modify: `src/image_segmenter/models/registry.py`
- Modify: `src/image_segmenter/models/__init__.py`
- Modify: `tests/test_models.py`
- Create: `configs/reference_unet.yaml`
- Create: `configs/deeplabv3_resnet50.yaml`

- [ ] Add a failing offline test that creates `deeplabv3_resnet50` with
  `pretrained=False`, runs `[1,3,32,32]` in eval mode, and receives direct
  `[1,2,32,32]` logits. Assert registry ordering contains both model names.
- [ ] Implement `DeepLabV3Adapter`. Use
  `DeepLabV3_ResNet50_Weights.DEFAULT` only when pretrained; replace the final
  classifier convolution with a two-class 1 x 1 convolution and return
  `output["out"]` from `forward`.
- [ ] Extend the registry without trainer branches. Create the exact 25-epoch
  U-Net and 15-epoch DeepLab reference configurations from the approved design.
- [ ] Run model tests offline and Ruff; commit with
  `feat(models): Add pretrained DeepLabV3 comparison`.

### Task 7: Public Evaluate And Predict CLI

**Files:**
- Modify: `src/image_segmenter/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_end_to_end.py`

- [ ] Add failing CLI tests for
  `evaluate --checkpoint artifacts/tiny-run/best.pt --split valid` and
  `predict --checkpoint artifacts/tiny-run/best.pt --image sample.png --output prediction.png`
  using the tiny trained checkpoint fixture.
- [ ] Wire `evaluate` options `--checkpoint`, `--split`, `--output-dir`,
  `--sample-limit`, `--error-limit`, `--plot`, and `--device` directly to
  `evaluate_checkpoint`.
- [ ] Wire `predict` options `--checkpoint`, `--image`, `--output`, and
  `--device` directly to `Predictor`; support both a file and directory.
- [ ] Run `uv run pytest`, `uv run ruff check .`, and
  `uv run python -m build`; expect all checks to pass.
- [ ] Commit with `feat(cli): Expose evaluation and mask prediction`.

## Phase Completion Check

Run the complete suite without network access. Verify the tiny CLI workflow now
continues through evaluate and predict, all saved raw masks contain only 0/1,
JSON has no non-standard NaN tokens, visual files decode with Pillow, DeepLabV3
constructs with `weights=None`, and `git status --short` is empty. Do not
download pretrained weights or Oxford data in this phase.
