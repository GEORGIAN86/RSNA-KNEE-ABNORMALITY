# Dockerized Kaggle Training Initialization Design

## Goal

Provide one reproducible Docker Compose command that reuses local RSNA knee
DICOM data when present, otherwise downloads the competition data securely from
Kaggle, normalizes it into the pipeline layout, inspects the NVIDIA GPU, and
starts the existing adaptive end-to-end DINO/SAM/fusion training workflow.

## User interface

The primary command is:

```bash
docker compose run --rm trainer
```

The Compose service loads `KAGGLE_API_TOKEN` from `.env`, mounts persistent
`data`, `weights`, and `outputs` directories, requests an NVIDIA GPU, and runs
an idempotent initialization entrypoint. `TRAIN_MODE=plan` runs
`TrainEnsemble.py auto --plan-only`; the default `TRAIN_MODE=train` runs the
full `auto` workflow.

The repository contains a placeholder `.env` and `.env.example`. Neither file
contains a real secret, and `.env` is excluded from Docker build context and
Git tracking.

## Container architecture

The image uses a CUDA-enabled PyTorch base that supports V100-class GPUs. It
installs the application dependencies plus the official Kaggle CLI, copies only
source/configuration files, and declares an entrypoint at
`docker/entrypoint.sh`. Training data, model weights, caches, checkpoints, and
outputs remain bind-mounted and are never baked into an image layer.

Docker Compose requests all available NVIDIA GPUs and configures adequate
shared memory for PyTorch data loading. The adaptive planner remains the sole
authority for batch size, SAM image size/slices, activation checkpointing,
gradient accumulation, workers, and trainable DINO/SAM blocks.

## Initialization and data flow

The entrypoint follows this sequence:

1. Check both `data/train_series` and `data/test_series` recursively for at
   least one `.dcm` file.
2. If both splits contain DICOMs, skip all Kaggle authentication and download
   work.
3. Otherwise require `KAGGLE_API_TOKEN`, verify the Kaggle CLI can authenticate,
   and download competition `rsna-knee-abnormality-detection` into a temporary
   directory.
4. Extract archives into a staging directory. Never overwrite a populated DICOM
   split. Copy missing metadata files and missing train/test DICOM trees into
   the canonical pipeline paths.
5. Map the competition filenames to `data/train.csv`, `data/test.csv`,
   `data/train_series.csv`, `data/test_series.csv`, and
   `data/sample_submission.csv`. Preserve the user-provided
   `data/llm_labels_v4_blend.csv` as the configured weak-label file.
6. Confirm required metadata, both DICOM splits, and external knee/SAM weights
   exist. Fail with a targeted message when any input is absent.
7. Print `nvidia-smi` output and verify PyTorch reports CUDA.
8. Execute `TrainEnsemble.py auto` or its `--plan-only` variant. That command
   prepares caches, profiles hardware/data, runs real forward/backward probes,
   retries OOM candidates, writes the selected configuration, and starts
   training only after a safe candidate succeeds.

Downloads are performed in staging and promoted only after successful
extraction, so a failed or interrupted download does not look like a complete
dataset on the next run.

## Configuration

`.env` exposes:

```dotenv
KAGGLE_API_TOKEN=
KAGGLE_COMPETITION=rsna-knee-abnormality-detection
TRAIN_MODE=train
```

`KAGGLE_COMPETITION` is overridable for testing but defaults to the approved
competition. Legacy `KAGGLE_USERNAME`/`KAGGLE_KEY` variables may be forwarded
when present, but the token is the documented primary method.

The production YAML will use canonical, container-stable paths rather than the
locally duplicated filenames containing `(1)` or `(2)`.

## Error handling and safety

- Missing DICOM data plus missing credentials exits before training.
- Kaggle authorization/download errors remain visible and mention that the
  competition rules must be accepted in the browser.
- Existing DICOM trees are never removed or overwritten.
- Temporary download/extraction content is cleaned on process exit.
- Missing NVIDIA runtime, inaccessible GPU, missing checkpoints, or malformed
  data fails before a full training run.
- Secrets are passed at runtime and never printed.
- `auto` retains its 85% VRAM acceptance limit and safe no-candidate behavior.

## Testing

Shell-level tests invoke the entrypoint with fake `kaggle`, `nvidia-smi`, and
Python commands in temporary directories. They verify:

- existing train/test DICOMs skip Kaggle entirely;
- missing data without credentials fails clearly;
- missing data downloads and normalizes staged competition files;
- plan mode appends `--plan-only`;
- training mode invokes `TrainEnsemble.py auto`;
- credentials are not echoed.

Python tests continue to cover the adaptive planner and end-to-end randomized
fusion pipeline. Docker configuration is also validated structurally, and the
image entrypoint is syntax-checked before completion.
