# One-Command Docker Training and Prediction Design

## Goal

Provide one Docker Compose command that reads Kaggle credentials from `.env`, downloads missing competition metadata and DICOM files, persistently preprocesses the data, adaptively trains all configured fusion folds on CUDA, predicts the test set with the exact resolved training configuration, and writes `outputs/submission.csv`.

The user-facing command is:

```bash
docker compose run --rm --build trainer
```

## Scope

This change covers the container bootstrap and orchestration path. It does not change DICOM normalization, target construction, fold assignment, model architecture, loss weighting, or checkpoint formats.

The existing Python commands remain the source of truth for adaptive planning, preprocessing, training, checkpoint resume, prediction, and submission validation.

## Architecture

The existing responsibility boundary remains intact:

- Docker Compose loads `.env`, enables all GPUs, and mounts persistent host directories.
- `docker/entrypoint.sh` handles data acquisition, canonical file naming, prerequisite checks, and sequencing of the Python commands.
- `TrainEnsemble.py auto` profiles the GPU, builds or reuses the image cache, selects a VRAM-safe configuration, and trains all configured folds.
- `TrainEnsemble.py predict` loads the trained fold checkpoints using the exact configuration written by the adaptive run and creates the final prediction artifacts.

The normal execution sequence is:

```text
.env and mounted volumes
        ↓
normalize local metadata aliases
        ↓
reuse existing DICOMs or download and extract from Kaggle
        ↓
validate CSVs, DICOM trees, source checkpoints, and CUDA
        ↓
TrainEnsemble.py auto --config config/training.yaml
        ↓
verify outputs/auto_training.yaml
        ↓
TrainEnsemble.py predict --config outputs/auto_training.yaml
        ↓
verify outputs/submission.csv
```

## Persistent Data

Compose mounts three host directories:

- `./data:/app/data` stores downloaded metadata, raw DICOMs, and reusable `.npz` caches.
- `./weights:/app/weights` stores supplied DINO/SAM weights and trained fusion checkpoints.
- `./outputs:/app/outputs` stores resource planning reports, the resolved adaptive configuration, metrics, OOF predictions, test predictions, and the final submission.

Container removal must not delete any of these artifacts. Re-running the one command reuses valid DICOMs and caches. The pipeline's existing `checkpoint.fusion.resume: auto` setting resumes compatible `fold_<n>_last.pth` checkpoints.

## Kaggle Authentication and Download

Docker Compose supplies environment variables from `.env`. Authentication accepts either:

- `KAGGLE_API_TOKEN`, or
- both `KAGGLE_USERNAME` and `KAGGLE_KEY`.

The competition defaults to `rsna-knee-abnormality-detection` and remains configurable through `KAGGLE_COMPETITION`.

If both mounted DICOM trees already contain at least one `.dcm` file, the entrypoint skips Kaggle entirely. Otherwise it requires credentials, downloads the configured competition into a temporary directory, extracts downloaded ZIP archives, locates the five canonical metadata CSVs, and locates train and test DICOM trees under known competition directory names.

Downloaded files are copied into the persistent data mount. Temporary download and extraction files are removed when the bootstrap stage exits. Credentials must never be printed by the entrypoint or included in error output.

Known duplicate local metadata names are promoted to canonical names before deciding whether a download is required:

- `train (1).csv` to `train.csv`
- `test (1).csv` to `test.csv`
- `train_series (1).csv` to `train_series.csv`
- `test_series (1).csv` to `test_series.csv`
- `sample_submission (2).csv` to `sample_submission.csv`

## Training and Prediction Modes

The default `TRAIN_MODE=train` performs the complete adaptive training and prediction workflow.

After prerequisites pass, the entrypoint runs:

```bash
python TrainEnsemble.py auto --config config/training.yaml
```

A successful adaptive run must leave a non-empty `outputs/auto_training.yaml`. Prediction then uses that exact file:

```bash
python TrainEnsemble.py predict --config outputs/auto_training.yaml
```

The entrypoint then requires a non-empty `outputs/submission.csv` before returning success. This prevents a zero exit status from implying that the requested final artifact exists when only training completed.

The existing `TRAIN_MODE=plan` remains available. It runs adaptive planning with `--plan-only`, requires `outputs/auto_training.yaml`, and exits without prediction because no fold checkpoints are guaranteed to exist.

## Failure Handling

The entrypoint exits nonzero with a concise actionable message when any of the following occurs:

- DICOMs are missing and no supported Kaggle credentials are configured.
- The Kaggle CLI download fails.
- The downloaded archive lacks required metadata or train/test DICOM trees.
- Required mounted metadata, DICOMs, DINO fold checkpoints, or the SAM checkpoint are missing.
- `nvidia-smi` is unavailable.
- PyTorch cannot access CUDA.
- Adaptive resource planning cannot select a safe configuration.
- Training fails or does not produce a non-empty resolved configuration.
- Prediction fails or does not produce a non-empty submission.

Because the shell uses strict error handling, prediction cannot begin after a failed training command.

## Testing Strategy

Entrypoint tests use temporary project trees and fake `kaggle`, `nvidia-smi`, and `python` executables. Tests assert observable command sequencing and persistent artifacts without downloading real data, starting CUDA, or training models.

Required behavioral coverage is:

1. Existing DICOMs skip Kaggle and run adaptive training followed by prediction.
2. Prediction receives `outputs/auto_training.yaml`, not the static input configuration.
3. Missing DICOMs require credentials before invoking Kaggle.
4. Credentialed download normalizes metadata and DICOM directory names without exposing secrets.
5. A failed adaptive training command prevents prediction.
6. Missing `outputs/auto_training.yaml` after a nominal training command is rejected.
7. Missing `outputs/submission.csv` after a nominal prediction command is rejected.
8. Plan mode adds `--plan-only` and never invokes prediction.
9. Compose continues to load `.env`, mount `data`, `weights`, and `outputs`, and enable all GPUs.

Verification includes Bash syntax checking, Docker Compose configuration rendering when Docker is available, focused Docker/entrypoint tests, and the complete Pytest suite.

## Success Criteria

The work is complete when:

- `docker compose run --rm --build trainer` is the documented single command.
- Missing Kaggle DICOM data is downloaded using `.env` credentials and persisted under `./data`.
- Existing data, valid caches, and resumable checkpoints are reused.
- The adaptive CUDA training command completes before prediction begins.
- Prediction uses `outputs/auto_training.yaml`.
- A successful container exit guarantees a non-empty `outputs/submission.csv`.
- Failure paths are covered by automated tests and do not leak credentials.
