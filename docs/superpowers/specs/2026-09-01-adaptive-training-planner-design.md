# Adaptive Hardware-Aware Training Planner Design

## Objective

Add an `auto` pipeline command that inspects available compute and the actual
dataset, chooses a balanced preprocessing/loading/training configuration,
verifies that configuration with a real forward/backward memory probe, writes
an auditable resolved configuration, and starts preprocessing and five-fold
fusion training only after the probe succeeds.

The planner optimizes for a balanced tradeoff between expected model quality,
training time, and out-of-memory safety. It preserves the original
`config/training.yaml` and writes its decisions under `outputs/`.

## Command Flow

```text
TrainEnsemble.py auto --config config/training.yaml
        │
        ├── input and storage preflight
        ├── hardware profile
        ├── data profile
        ├── candidate generation and balanced scoring
        ├── cache creation when required
        ├── real forward/backward memory probes
        ├── outputs/resource_report.json
        ├── outputs/auto_training.yaml
        └── training with the selected in-memory configuration
```

An optional `--plan-only` flag performs profiling, candidate selection, cache
preparation when needed for a representative probe, and probing, but does not
start full fold training. The default `auto` behavior starts training after a
candidate passes.

## Hardware Profile

Collect:

- Runtime device availability and selected device.
- GPU name, compute capability, total VRAM, free VRAM before model creation,
  CUDA version, and AMP support.
- System RAM total and currently available.
- Logical CPU count.
- Free disk space for the project data/cache/output volume.
- Current PyTorch deterministic and precision capabilities.

CUDA information comes from PyTorch. RAM, CPU, and disk statistics use the
Python standard library so no additional runtime dependency is required.

The automatic command requires CUDA. If CUDA is unavailable, it writes a
resource report and stops rather than starting full DINO–SAM training on CPU or
MPS. Existing explicit `train` and `predict` commands retain their current
device behavior.

## Data Profile

Before preprocessing, inspect train/test study and series CSVs and DICOM folder
coverage. Collect:

- Study and series counts per split.
- Series count per study percentiles.
- DICOM slice count per series percentiles using filesystem counts.
- Available acquisition-slot count per study.
- Estimated raw DICOM bytes.
- Existing cache fingerprint, study coverage, and cache bytes.
- Gold/weak-label coverage and complete gold row count.

The profile samples DICOM headers only when metadata required for slot coverage
is absent from the series CSV. It does not decode all pixels merely to plan.

The planner verifies that available disk can hold the estimated cache plus a
20% safety reserve and expected checkpoints. It stops before preprocessing if
the estimate exceeds available disk.

## Separate DINO and SAM Views

The cache continues to hold up to sixteen slices for each of six slots. The
dataset returns two views from the same cache:

```text
dino_images:     [6, 16, H, W]
dino_slice_mask: [6, 16]
sam_images:      [6, selected_sam_slices, H, W]
sam_slice_mask:  [6, selected_sam_slices]
slot_mask:       [6]
```

DINO remains fixed at sixteen slices because the supplied checkpoint has a
sixteen-channel patch projection. SAM slice count is configurable from one to
four. Training selects SAM slices randomly from valid cached indices;
validation/probe/prediction use evenly spaced deterministic indices.

The joint fusion forward interface consumes both views. DINO never processes
the reduced SAM view, and SAM never retains graphs for unused DINO slices.

## Configurable Encoder Freezing

Add `trainable_blocks` to both encoders with identical semantics:

- `-1`: train the complete encoder.
- `0`: freeze the complete encoder.
- Positive `N`: train only the final `N` transformer blocks.

For positive DINO values, the final `N` blocks and final encoder normalization
are trainable; patch embedding, slot embedding, earlier blocks, and other
encoder parameters are frozen. For positive SAM values, the final `N` blocks
and encoder neck are trainable; patch embedding, positional embedding, and
earlier blocks are frozen. Fusion projections, modality embeddings, label
queries, decoder, normalization, and label outputs are always trainable.

Defaults are four DINO blocks and two SAM blocks.

## Candidate Space

The planner starts from the balanced preferred candidate:

```yaml
knee:
  slices_per_slot: 16
  trainable_blocks: 4
sam:
  slices_per_slot: 2
  input_size: 512
  trainable_blocks: 2
fusion:
  batch_size: 1
  gradient_accumulation: 8
runtime:
  precision: amp
```

Candidate dimensions are intentionally bounded:

- Physical batch size: `1`, then `2` only when a probe shows substantial room.
- SAM input size: `512`, `384`, `256`.
- SAM slices per slot: `4`, `2`, `1`.
- DINO trainable blocks: `6`, `4`, `2`, `0`.
- SAM trainable blocks: `4`, `2`, `1`, `0`.
- Gradient checkpointing: off/on per encoder.
- Gradient accumulation: selected to target effective batch size eight.
- DataLoader workers: zero through the lesser of eight and half the logical
  CPUs, bounded by available RAM.

DINO slice count and image size remain checkpoint-compatible and are not
automatically reduced.

## Balanced Candidate Scoring

Candidates receive a deterministic score rewarding, in order:

1. At least two SAM slices per slot.
2. More trainable DINO blocks up to four.
3. More trainable SAM blocks up to two.
4. SAM input size 512, with a modest penalty for 384 and larger penalty for 256.
5. Physical batch size above one only after the preferred representation and
   block counts are preserved.
6. Lower estimated wall time and DataLoader overhead.

Configurations above the preferred block counts receive only a small quality
bonus because the policy is balanced rather than maximum-quality. Candidate
ordering is stable and recorded in the report.

## Analytical Memory Estimate

Estimate persistent memory from trainable and frozen parameter counts,
precision, gradient storage, AdamW moments, and scaler overhead. Estimate
activation pressure from DINO slot count, DINO tokens, SAM slice count, SAM
input size, trainable block count, physical batch, and checkpointing status.

The estimate is a filter and ordering aid, not final proof. Candidates whose
estimate exceeds 85% of free VRAM are skipped unless no lower estimate exists.

## Real Memory Probe

The real probe uses one representative high-load cached study chosen by maximum
valid slot/slice count. It constructs the production fusion model, applies the
candidate freezing/checkpointing policy, loads external fold-zero encoder
weights, and executes:

1. One training-mode forward pass under candidate precision.
2. Weighted BCE on a valid twelve-target batch.
3. One backward pass.
4. Gradient clipping.
5. One optimizer step and zeroing.

Before each attempt, delete the previous model and optimizer, run garbage
collection, clear the CUDA cache, and reset peak-memory statistics. Record peak
allocated and reserved VRAM, elapsed time, parameter counts, and success or
failure.

A candidate succeeds only when it completes and peak reserved memory is at
most 85% of total VRAM. CUDA OOM is caught only around the isolated probe;
non-OOM runtime errors stop planning as code/data failures rather than causing
silent configuration degradation.

## OOM Fallback Order

Failed candidates follow this ordered degradation:

1. Reduce physical batch to one.
2. Enable DINO and SAM gradient checkpointing.
3. Reduce SAM input from 512 to 384, then 256.
4. Reduce SAM slices from four to two, then one.
5. Reduce SAM trainable blocks from four to two, one, then zero.
6. Reduce DINO trainable blocks from six to four, two, then zero.

Gradient accumulation is increased as physical batch falls so the effective
batch remains approximately eight. If every bounded candidate fails, write the
report and stop without training.

## Gradient Checkpointing

Add explicit configuration flags for DINO and SAM. When enabled, transformer
blocks are evaluated through PyTorch checkpointing during training and normally
during gradient-enabled probes. Evaluation and prediction do not checkpoint.
Checkpointing must preserve the slot-token/rope calling convention for DINO
and the SAM block calling convention.

## Generated Outputs

`outputs/resource_report.json` contains:

- Hardware and data profiles.
- Cache/disk estimates.
- Every attempted candidate and score.
- Analytical estimate and real peak memory.
- Failure category/message.
- Selected candidate and reasons.
- Trainable/frozen parameter counts by component.

`outputs/auto_training.yaml` contains the complete resolved configuration with
absolute or project-relative paths suitable for a later explicit command. The
original configuration is never overwritten.

The generated configuration and resource report are written atomically.

## Training Startup and Recovery

After a successful probe, release all probe resources and start normal
preprocessing if the required cache fingerprint is missing or incomplete.
Then call the normal joint fusion training path with the selected in-memory
configuration. Prediction is not started automatically; this keeps `auto`
focused on safe model training. A later `predict` command can use
`outputs/auto_training.yaml`.

If full training encounters a CUDA OOM despite a successful probe, save no
partial optimizer step, write the failure into the resource report, and stop.
The pipeline does not silently change architecture after native fold
checkpoints have been created because doing so would invalidate resume
compatibility.

## Testing

Tests use synthetic hardware profiles, cache summaries, and tiny fusion models.
They cover:

- Hardware and data-profile schema.
- Candidate generation, stable balanced scoring, and fallback ordering.
- RAM/CPU-based worker selection.
- Disk-space rejection.
- Separate 16-slice DINO and two-slice SAM dataset views.
- Deterministic evaluation and randomized training sampling.
- `-1`, `0`, and positive trainable-block semantics for both encoders.
- Trainable/frozen parameter accounting.
- Successful probe selection, simulated CUDA OOM fallback, and non-OOM error
  propagation.
- Atomic generated report/config files.
- `--plan-only` versus automatic training dispatch.
- Existing randomized fusion training and full regression suite.
