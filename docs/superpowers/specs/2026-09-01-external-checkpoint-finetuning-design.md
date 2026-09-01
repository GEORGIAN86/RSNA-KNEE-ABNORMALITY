# External Checkpoint Fine-Tuning Design

## Objective

Update the knee MRI pipeline to warm-start five fold-specific knee models from
the supplied `m_f0.pt` through `m_f4.pt` checkpoints, and warm-start one global
SAM model from the supplied extracted PyTorch checkpoint. Knee training remains
five-fold; SAM training uses all available training studies without a validation
fold. Prediction averages the five knee models and blends that result with the
single global SAM model.

## Constraints Established From the Supplied Checkpoints

The knee checkpoints contain `state_dict`, `cfg`, and `fold`. Their declared
backbone is `vit_small_patch16_dinov3.lvd1689m`; they are DINOv3 checkpoints,
not DINOv2 checkpoints. Their model consumes 16 MRI slices as input channels
and includes custom conditional-token and label-specific readout layers. They
do not contain optimizer, scheduler, scaler, epoch, or training-history state,
so they support model warm-starting rather than exact optimizer resumption.

The SAM checkpoint is an extracted PyTorch ZIP serialization directory. When
repacked with the directory as the archive root, it loads as a training payload
with model, optimizer, scaler, epoch, step, and best-score fields. Its model
state uses abbreviated names: `e.*` for the image encoder, `h.*` for the head,
and `m`/`s` for normalization buffers. The model structure matches the existing
SAM classifier closely enough to support an explicit key translation.

## Architecture

### External checkpoint compatibility layer

A focused checkpoint compatibility module will discover and read the two
external formats. It will:

- Resolve knee checkpoint `m_f<n>.pt` for fold `n`.
- Load external files with PyTorch's weights-only mode.
- Validate the knee fold number, 12-label order, architecture configuration,
  state keys, and tensor shapes before applying weights.
- Read an extracted SAM serialization directory without permanently modifying
  user-provided files.
- Translate SAM model keys into the native `SAMClassifier` key namespace.
- Produce clear compatibility errors instead of silently ignoring mismatches.

The external sources remain immutable. Fine-tuning outputs continue to use the
pipeline's native atomic checkpoint format.

### Knee model

The knee branch will use a model matching the supplied DINOv3 checkpoint:

- DINOv3 small, patch size 16.
- Six ordered acquisition slots from the shared cache.
- Sixteen sampled slices per study input as required by the checkpoint.
- The checkpoint's conditional-token and label-specific `xcodex` readout.
- Twelve logits in the exact checkpoint label order.

Each configured fold warm-starts from its matching external checkpoint, then
uses the existing train/validation partition, weighted BCE, optimizer,
scheduler, AMP, macro-AUC selection, and native best/last checkpoint outputs.
Native `fold_<n>_last.pth` checkpoints take precedence when automatic resume is
enabled, because they contain full optimizer and progress state. The external
checkpoint is used only when no compatible native resume checkpoint exists.

### Global SAM model

The SAM branch will no longer iterate over folds. It will:

1. Construct the existing SAM ViT-B classifier directly from the full encoder
   weights present in the external checkpoint, without requiring a separate
   SAM base checkpoint for this path.
2. Translate and strictly load the external model state.
3. Build one loader containing every available training record.
4. Fine-tune for the configured number of epochs with weighted BCE.
5. Save `global_last.pth` atomically after each epoch.
6. Reload `global_last.pth` for automatic native resume when available.
7. Save `global_final.pth` after the configured training schedule completes.

There is no validation loader, macro-AUC selection, OOF SAM prediction, or
`global_best.pth`. Training history records epoch loss, learning rate, epoch,
and global step.

## Data and Command Flow

`preprocess` remains unchanged and creates the shared six-slot study cache.

`train` performs these independent branches:

1. Build targets and the deterministic knee fold map.
2. Train five knee folds, using native resume checkpoints first and supplied
   fold checkpoints as warm starts second.
3. Train one global SAM model on all target-bearing studies.
4. Write knee OOF predictions and branch training metrics.

Since the global SAM model sees every training study, the pipeline will not
write `oof_sam.csv` or `oof_blended.csv`, and will not report an OOF ensemble
score. Existing stale versions of those artifacts are not treated as outputs
of the new training run.

`predict` performs:

1. Load and predict with each native knee `fold_<n>_best.pth` checkpoint.
2. Average knee probabilities across folds.
3. Load and predict with native SAM `global_final.pth`.
4. Blend the knee average with the single SAM prediction using the configured
   rank or probability blend and `sam_weight`.
5. Validate and write `submission.csv` as before.

## Configuration

Configuration will explicitly identify external initialization sources and the
global SAM mode. Defaults will point at the files currently present under
`weights/checkpoints`. Native output paths remain under the branch checkpoint
directories. Configuration validation will distinguish external initialization
requirements from native resume requirements.

The knee configuration will reflect the supplied model's DINOv3 backbone,
16-slice input, image size, conditional-token mode, and readout. The SAM
configuration will identify its extracted checkpoint directory and global
training mode. A required DINOv3 implementation dependency will be added to the
project environment definition.

## Error Handling

Training stops before optimization when any of these conditions occurs:

- A requested knee fold lacks its external checkpoint and has no native resume.
- A knee checkpoint declares the wrong fold or label order.
- External architecture settings disagree across knee folds.
- Model state keys or shapes are incompatible.
- The SAM directory lacks required PyTorch serialization members.
- SAM key translation is incomplete or produces shape mismatches.
- A native resume checkpoint is incompatible with the current cache, labels,
  architecture, or training mode.

No fallback to partial or non-strict loading is allowed for external weights.

## Verification

Tests will use small synthetic models and checkpoints rather than production
weights. They will cover:

- Knee fold checkpoint discovery and fold/label validation.
- Strict external model-state loading and shape mismatch errors.
- Extracted SAM archive loading and key translation.
- Native resume precedence over external warm-start initialization.
- Global SAM training over all records and final checkpoint creation.
- Prediction with five knee checkpoints plus one global SAM checkpoint.
- Absence of misleading SAM and blended OOF artifacts.
- Existing cache, dataset, trainer, ensemble, and smoke behavior not affected
  outside the intentionally changed branch orchestration.

The focused tests will run first, followed by the complete pytest suite.
