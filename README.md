# Knee MRI DINO–SAM Transformer Fusion

This project provides one local/VM command-line pipeline for preprocessing knee
MRI DICOM studies and jointly fine-tuning DINOv3, SAM ViT-B, and a transformer
fusion decoder across five folds. Twelve learned label queries attend to DINO
and SAM study embeddings and directly produce the submission predictions.

## Project layout

```text
config/training.yaml
data/
  train.csv
  test.csv
  train_series.csv
  test_series.csv
  sample_submission.csv
  labels.csv                         # optional weak labels
  train_series/<study>/<series>/*.dcm
  test_series/<study>/<series>/*.dcm
  cache/train/*.npz
  cache/test/*.npz
weights/
  checkpoints/knee/m_f0.pt             # supplied DINOv3 warm start
  checkpoints/sam/submissions_epoch_8_step_11550/ # supplied extracted SAM
  checkpoints/fusion/fold_0_best.pth
outputs/
```

`labels.csv` is optional. When present, it supplies weak targets; complete gold
rows in `train.csv` override matching weak rows and receive the configured gold
weight. Without it, training uses complete gold rows only.

## Environment

Python 3.10–3.12 is recommended. A project-local environment can be created
with `uv`:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Or use a normal virtual environment:

```bash
python -m pip install -r requirements.txt
```

Place the five supplied knee checkpoints at `weights/checkpoints/knee/m_f0.pt`
through `m_f4.pt`, and the extracted SAM checkpoint at the path configured by
`paths.external_sam_checkpoint`. The external SAM state includes the full
ViT-B encoder, so a separate SAM base checkpoint is not required.

## Commands

Run commands with the environment's Python:

```bash
python TrainEnsemble.py validate --config config/training.yaml
python TrainEnsemble.py preprocess --config config/training.yaml
python TrainEnsemble.py train --config config/training.yaml
python TrainEnsemble.py predict --config config/training.yaml
python TrainEnsemble.py all --config config/training.yaml
python TrainEnsemble.py auto --config config/training.yaml
python TrainEnsemble.py auto --plan-only --config config/training.yaml
```

Use a subset of configured folds when needed:

```bash
python TrainEnsemble.py train --config config/training.yaml --folds 0,1
```

`train.py` remains as a compatibility entry point for the `train` command.

## Pipeline behavior

`preprocess` creates one fingerprinted cache entry per study. Each entry stores
six ordered acquisition slots:

1. Sagittal fluid-sensitive
2. Sagittal non-fluid-sensitive
3. Coronal fluid-sensitive
4. Coronal non-fluid-sensitive
5. Axial fluid-sensitive
6. Axial non-fluid-sensitive

The DINOv3 knee model samples sixteen slices per slot as its image channels.
SAM independently samples two slices per slot from the same cache. Evaluation sampling is
deterministic.

`train` assigns complete fusion models to report-hash folds. Fold `n` loads the
encoder from `m_f<n>.pt` and initializes its full SAM encoder from the shared
SAM checkpoint. DINO produces a pooled study embedding, SAM produces a pooled
study embedding, and twelve learned queries attend to those two tokens through
a two-layer transformer decoder. One weighted BCE loss trains the configured
last DINO/SAM blocks and the fusion decoder end-to-end. Macro ROC AUC selects the best native
checkpoint under `weights/checkpoints/fusion/`:

```text
fold_<n>_best.pth
fold_<n>_last.pth
fold_<n>_history.json
```

Checkpoints contain the model, optimizer, scheduler, scaler, epoch, global
step, label order, branch/fold identity, cache fingerprint, fold-map
fingerprint, metrics, and model configuration.

`predict` loads each complete fusion fold, averages its twelve sigmoid
probabilities, and writes the result directly. There is no fixed DINO/SAM
probability blend.

## Resume and checkpoint loading

Configure fusion resume under `checkpoint`:

```yaml
checkpoint:
  strict: true
  fusion:
    resume: auto       # fresh, auto, or explicit
    path: null
    initialize: external
```

- `fresh`: ignore existing last checkpoints.
- `auto`: resume `fold_<n>_last.pth` when it exists.
- `explicit`: load `path`; use this with a single selected fold.

Strict mode rejects incompatible branches, folds, labels, cache settings, fold
maps, or model state shapes before training resumes.

## Important configuration groups

- `paths`: raw data, cache, pretrained weights, checkpoints, and outputs.
- `data`: folds, image size, physical crop, cache slices, workers, and label
  weights.
- `knee`: DINOv3 encoder recipe and sixteen-slice input.
- `sam`: SAM ViT-B encoder construction and slice encoding settings.
- `fusion`: decoder dimensions, batch size, optimizer, epochs, accumulation,
  and clipping.
- `checkpoint`: load/resume policy and compatibility strictness.
- `runtime`: device, seed, deterministic behavior, and precision.

`knee.trainable_blocks` and `sam.trainable_blocks` select how many trailing
encoder blocks are fine-tuned (`0` freezes an encoder and `-1` unfreezes all
blocks). Joint training defaults to batch size one and accumulation eight.

## Automatic resource planning

`auto` profiles CPU, RAM, CUDA VRAM, and the train/test tables, prepares the
cache, then tries balanced configurations using a real fusion forward,
backward, and optimizer step. A candidate is accepted only below 85% of total
VRAM. OOM failures automatically reduce activation checkpointing, SAM input
size/slices, and finally the number of trainable SAM/DINO blocks. The selected
configuration is written to `outputs/auto_training.yaml`; all estimates, probe
attempts, and measured peaks are recorded in `outputs/resource_report.json`.
Use `--plan-only` to stop after selection. Without CUDA, the report is still
written and training exits safely.

## Outputs

```text
outputs/folds.json
outputs/oof_fusion.csv
outputs/metrics.json
outputs/resolved_config.yaml
outputs/test_fusion.csv
outputs/submission.csv
```

The final validator requires unique study IDs, exact label order and coverage,
finite numeric probabilities, and values inside `[0, 1]`. Submission values are
clipped to `[1e-6, 1 - 1e-6]` only after structural validation.

## Tests

The tests do not require real DICOM data, CUDA, model downloads, or production
weights:

```bash
.venv/bin/python -m pytest -q
```

The deterministic and randomized smoke tests train, checkpoint, reload, and
fold-average tiny fusion substitutes through the same production orchestration.
