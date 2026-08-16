# Oxford Learning Experience And Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the verified segmentation engine into a complete bilingual beginner lab using official Oxford-IIIT Pet data, measured reference evidence, CI, and a published GitHub repository.

**Architecture:** Keep Oxford-specific archive and split knowledge in scripts/data preparation while the Dataset continues to consume portable manifests. Treat examples and documents as tested product surfaces, and publish only measured metrics tied to committed configuration and manifest checksums.

**Tech Stack:** Existing Python stack, urllib/tarfile, Pillow, Matplotlib, GitHub Actions, Markdown

---

### Task 1: Oxford-Aware Manifests And Metadata

- [ ] Add failing tests using synthetic `trainval.txt` and `test.txt` files for
  deterministic within-breed 80/20 train/valid splits and untouched official
  test membership.
- [ ] Extend manifest columns with `breed` and `source_split`; keep blank values
  for generic paired data. Refactor validation so generic and Oxford splitters
  share all image/mask checks.
- [ ] Write `source.yaml` and `summary.txt` with source type, label semantics,
  seed, counts, exact-file duplicate audit, and SHA-256 manifest checksums.
- [ ] Add `--source-format auto|paired|oxford-pet` to `prepare-data`; auto-detect
  official split files. Run data, CLI, and end-to-end tests; commit.

### Task 2: Atomic Official Dataset Download

- [ ] Add failing tests for checksum verification, safe tar extraction, and an
  interrupted/invalid download never replacing a complete archive.
- [ ] Implement `scripts/download_data.py` with pinned official image and
  annotation URLs and SHA-256 values. Extract into `data/raw/images`,
  `data/raw/masks`, and `data/raw/annotations`, preserving official split files.
- [ ] Make reruns idempotent and add `--data-dir`, `--force`, and `--verify-only`.
  Run tests and commit.

### Task 3: Dataset Preview And Metric Plot Scripts

- [ ] Add failing tests that create a preview PNG from synthetic data and a loss
  curve from a tiny metrics CSV.
- [ ] Implement `scripts/preview_dataset.py` as a Pillow grid showing image, raw
  mask, mapped mask, and overlay. Implement `scripts/plot_metrics.py` as a thin
  wrapper around the tested plotting function.
- [ ] Add bilingual script directory guides. Run tests and commit.

### Task 4: Five Runnable Learning Examples

- [ ] Add tests that run examples 01-04 without external data and exercise
  example 05 with the tiny checkpoint fixture.
- [ ] Implement mask/logit flow, paired transform, DataLoader batch, minimal
  optimization loop, and checkpoint overlay examples. Keep each focused and
  link its production counterpart in the module docstring.
- [ ] Run all example tests and commit.

### Task 5: English Learning And Contributor Documentation

- [ ] Create README, learning path, code tour, principles, configuration flow,
  config reference, custom data, experiments, reference run, six tutorial
  chapters, directory guides, and contribution guide in English.
- [ ] Ensure commands match the installed CLI and explain Oxford raw label 3 as
  unclassified/ignored, foreground-IoU checkpoint selection, validation/test
  separation, empty-foreground behavior, and CPU versus reference configs.
- [ ] Add a documentation link checker and README command assertions; run tests
  and commit.

### Task 6: Simplified Chinese Documentation Parity

- [ ] Create `.zh-CN.md` counterparts with the same headings, commands, links,
  limitations, and learning sequence as every core English document.
- [ ] Add a parity test for required file pairs, headings, and fenced commands.
- [ ] Run documentation tests and commit.

### Task 7: CI, Templates, And Packaging Metadata

- [ ] Add failing packaging assertions for README metadata, typed package data,
  all config files, and CLI help.
- [ ] Complete pyproject metadata and dev plotting dependency. Add GitHub Actions
  for Python 3.10-3.12 running Ruff, pytest, build, and twine check without data
  or model downloads.
- [ ] Add bilingual-friendly issue forms, pull request template, and Makefile
  `check` target. Run local equivalents and commit.

### Task 8: Measured Oxford Reference Evidence

- [ ] Download and verify Oxford-IIIT Pet, generate official manifests, and
  inspect the preview and summary before training.
- [ ] Run CPU minimal dry-run and two-epoch learning workflow. Run the approved
  full 25-epoch U-Net reference configuration on MPS/CUDA when available.
- [ ] Evaluate the untouched official test set, generate curves and overlays,
  verify pet IoU is at least 0.70, and record exact environment and measured
  values in both reference-run documents.
- [ ] Commit bounded reference artifacts: effective config, run metadata,
  metrics, best checkpoint, evaluation tables, curve, and selected overlays.

### Task 9: Release Audit And GitHub Publication

- [ ] Run the full requirements audit against the approved design, all tests
  with deprecations as errors, Ruff, build, twine check, README commands, and a
  clean-checkout smoke workflow.
- [ ] Review tracked files for datasets, caches, secrets, oversized accidental
  artifacts, and stale generated outputs. Confirm repository license and data
  attribution.
- [ ] Merge the implementation branch into `main`, create the public
  `Doithoo/pytorch-image-segmentation-lab` repository, push `main`, and verify
  the remote default branch and files.
