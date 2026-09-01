# Dual-Model Knee MRI Training and Ensemble Design

## Objective

Build a portable local/VM command-line pipeline that preprocesses knee MRI
studies, trains and validates both a six-slot knee classifier and a partially
fine-tuned SAM classifier across the same folds, saves resumable `.pth`
checkpoints, blends both branches, and validates the final predictions.

The implementation will reuse and repair the existing helper, model, ensemble,
and validator code rather than maintain the current incompatible standalone
paths.

## Command-Line Interface

`TrainEnsemble.py` is the thin orchestration entry point:

```bash
python TrainEnsemble.py <preprocess|train|validate|predict|all> \
  --config config/training.yaml
```

Commands have these responsibilities:

- `preprocess`: validate raw inputs and build or reuse the study cache.
- `train`: train both branches for the configured folds and write validation
  predictions and checkpoints.
- `validate`: validate inputs, caches, checkpoints, and any existing prediction
  artifacts without training.
- `predict`: load compatible fold checkpoints, predict the test set, blend the
  branches, and write a validated submission.
- `all`: run validation, preprocessing, training, test prediction, blending,
  and final output validation in order.

## Repository Layout

```text
config/training.yaml
data/
weights/
  pretrained/
  checkpoints/
    knee/
    sam/
outputs/
Helpers/
  __init__.py
  cache.py
  normalization.py
  preprocess.py
Models/
  __init__.py
  Model.py
  SlotHead.py
  SAMClassifier.py
Training/
  __init__.py
  dataset.py
  trainer.py
  checkpoints.py
  ensemble.py
Validators/
  __init__.py
  validator.py
  test_validator.py
TrainEnsemble.py
```

The current `Helpers/__init__.py/` directory is an invalid package layout. Its
three modules will move directly under `Helpers/`, and `Helpers/__init__.py`
will become a normal package file.

## Input Data Contract

The default repository-relative data paths are:

```text
data/train.csv
data/test.csv
data/train_series.csv
data/test_series.csv
data/train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
data/test_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
data/labels.csv                         # optional weak labels
data/cache/train/*.npz
data/cache/test/*.npz
```

All paths are configurable for local, VM, or Kaggle-mounted data. The pipeline
uses the optional weak-label CSV when present. Gold labels from `train.csv`
override weak labels for the same study and receive the configured gold weight.
When the optional file is absent, training uses complete gold-label rows from
`train.csv` only.

The target order is fixed and checked everywhere:

1. ACL
2. MCL
3. Medial Meniscus
4. Lateral Meniscus
5. Medial OA
6. Lateral OA
7. PF OA
8. Effusion
9. Synovitis
10. Baker's
11. Contusion
12. Fracture

Fold assignment is deterministic. Studies sharing the same normalized report
hash remain in the same fold to limit report-duplicate leakage. Both branches
use exactly the same saved fold mapping.

## Preprocessing and Shared Cache

The helpers will expose importable functions for header probing, series
annotation, slot selection, anatomical slice ordering, physical-size cropping,
intensity normalization, and laterality normalization.

Each study maps to six ordered slots:

```text
Sagittal fluid-sensitive
Sagittal non-fluid-sensitive
Coronal fluid-sensitive
Coronal non-fluid-sensitive
Axial fluid-sensitive
Axial non-fluid-sensitive
```

Each `.npz` cache entry contains:

- `images`: `uint8 [6, max_slices_per_slot, H, W]`;
- `slot_mask`: validity for the six slots;
- `slice_mask`: validity for retained or padded slices;
- study and preprocessing metadata needed for validation.

A sidecar manifest records a preprocessing fingerprint derived from image size,
crop size, slice bands, slot rules, normalization, and cache schema version.
Training and prediction reject a cache whose fingerprint does not match the
resolved configuration.

The knee branch samples three ordered slices per slot and treats them as the
three input channels expected by `Models/Model.py`. The SAM branch selects a
larger configurable, evenly distributed set of valid slices from the same
cache. Training selection may be randomized with a fixed worker seed;
validation and test selection are deterministic.

## Models

### Knee branch

`Models/Model.py` is the canonical knee model and uses the configured pretrained
image backbone. It encodes one three-slice image per available slot and passes
the six features and slot mask to `Models/SlotHead.py`.

`SlotHead` projects features, adds slot embeddings, applies label-specific
attention over valid slots, and returns 12 logits. All globals currently assumed
by the two fragments become explicit constructor arguments or shared constants.
No model module performs filesystem access or starts training on import.

### SAM branch

`Models/SAMClassifier.py` contains the classifier extracted from
`TrainEnsemble.py`. It uses the SAM image encoder, aggregates valid slice and
slot features to study level, and returns 12 logits through its classification
head.

By default, all SAM parameters are frozen except the classifier and the final
configured number of image-encoder blocks. The configuration may instead freeze
the entire encoder or fine-tune it fully. The base SAM checkpoint defaults to:

```text
weights/pretrained/sam_vit_b_01ec64.pth
```

The trainer reports trainable and total parameter counts before the first
epoch.

## Training and Validation

Both branches train independently for every configured fold. Each uses weighted
binary cross-entropy, automatic mixed precision on supported CUDA devices,
gradient scaling, configurable gradient accumulation and clipping, and a
configurable optimizer and scheduler.

After every epoch, deterministic validation produces probabilities and computes
per-label ROC AUC plus macro AUC over labels with both classes present. Macro AUC
is the checkpoint-selection metric. The trainer saves the latest resumable state
and replaces the best checkpoint only when the selection metric improves.

Validation outputs preserve `StudyInstanceUID`, fold, targets, branch
probabilities, and metrics. Combined out-of-fold files contain exactly one
prediction from each branch for every usable training study.

## Checkpoints and Resume Behavior

Default checkpoint paths are:

```text
weights/checkpoints/knee/fold_0_best.pth
weights/checkpoints/knee/fold_0_last.pth
weights/checkpoints/sam/fold_0_best.pth
weights/checkpoints/sam/fold_0_last.pth
```

Each training checkpoint contains:

- branch and fold identity;
- model state;
- optimizer, scheduler, and scaler states;
- completed epoch and global step;
- best and current validation metrics;
- ordered targets;
- resolved model and preprocessing configuration;
- preprocessing fingerprint and fold-map fingerprint;
- random seed information.

The YAML supports `fresh`, `auto`, and explicit resume modes independently for
the knee and SAM branches. `auto` resumes a compatible last checkpoint when it
exists. Explicit mode loads the configured path. Inference always loads best
checkpoints. Strict compatibility validation rejects mismatched branches,
architectures, labels, folds, or preprocessing fingerprints before state is
loaded.

## Ensemble and Prediction Outputs

Each branch averages test probabilities from its compatible best fold
checkpoints. The ensemble then blends knee and SAM predictions by label. The
default is percentile-rank blending with a SAM weight of `0.2` and a knee weight
of `0.8`; YAML may select probability blending or change the fixed weight.

The pipeline retains:

```text
outputs/oof_knee.csv
outputs/oof_sam.csv
outputs/oof_blended.csv
outputs/metrics.json
outputs/test_knee.csv
outputs/test_sam.csv
outputs/submission.csv
```

The implementation will not tune the blend weight on the same out-of-fold data
by default, avoiding an implicit extra optimization step. Blend-weight search
is outside this scope.

## Validators and Error Handling

`Validators/validator.py` provides preflight and artifact validation. It checks:

- the configuration schema and value ranges;
- required CSVs, columns, unique study IDs, and target order;
- raw series directories and referenced studies/series;
- non-empty folds and train/validation isolation;
- cache schemas, tensor shapes, masks, and fingerprints;
- checkpoint metadata and requested fold coverage.

`Validators/test_validator.py` validates prediction and submission artifacts. It
checks exact study coverage, one row per study, exact target columns and order,
finite numeric probabilities, and values in the valid probability range. Final
probabilities are clipped to `[1e-6, 1 - 1e-6]` only after structural validation.

Unreadable individual DICOM slices are recorded and skipped if the study still
has usable data. Required CSVs, empty training folds, incompatible artifacts,
and studies with no usable imaging fail with an actionable validation summary.
The cache builder writes atomically so an interrupted cache file is never
treated as complete.

## Configuration

`config/training.yaml` contains these sections:

- `paths`: data, cache, pretrained weights, checkpoints, and outputs;
- `data`: labels, folds, image geometry, slice selection, and workers;
- `knee`: backbone, pooling, slot head, optimizer, scheduler, and epochs;
- `sam`: model type, trainable final blocks, slice count, optimizer, scheduler,
  and epochs;
- `checkpoint`: strictness and per-branch resume modes/paths;
- `ensemble`: blending kind and SAM weight;
- `runtime`: device, seed, determinism, precision, and logging.

Repository-relative paths resolve relative to the config file or project root,
not the caller's current directory. The resolved configuration is written to the
output directory for reproducibility.

## Testing and Completion Criteria

Automated tests will cover:

- config loading and path resolution;
- target construction with and without weak labels;
- deterministic folds and isolation;
- cache schema and slice sampling;
- knee and SAM forward tensor contracts using lightweight test doubles;
- SAM partial-freezing parameter selection;
- weighted loss and macro AUC edge cases;
- checkpoint save/load/resume and incompatibility rejection;
- fold averaging, rank/probability blending, and UID alignment;
- submission validation and malformed-artifact failures;
- a small end-to-end smoke pipeline using synthetic cached studies.

The work is complete when all commands expose help without side effects, all
automated tests pass, a synthetic end-to-end run writes both branch checkpoints
and a validated blended submission, and no production module relies on
notebook-only globals or hard-coded `/kaggle` paths.
