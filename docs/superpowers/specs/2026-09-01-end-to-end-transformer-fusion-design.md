# End-to-End DINO–SAM Transformer Fusion Design

## Objective

Replace independently trained DINO and SAM classifiers plus fixed probability
blending with one end-to-end trainable fusion model. DINOv3 and SAM produce
study-level embeddings rather than twelve branch logits. A transformer decoder
uses twelve learned label queries to attend to the two modality embeddings and
produce the final twelve logits.

## Fold Topology

Training uses five joint folds. Within fold `n`, the training partition updates
the DINO encoder, SAM encoder, modality projections, transformer decoder, and
output parameters together. Validation uses only the held-out partition for
that fold. This removes the leakage that would result from combining a
fold-specific DINO model with a globally trained SAM model.

Each fold starts from:

- DINO checkpoint `m_f<n>.pt` for the matching fold.
- The shared extracted SAM checkpoint directory.

Native `fusion/fold_<n>_last.pth` takes precedence during automatic resume and
contains the complete joint model plus optimizer, scheduler, scaler, epoch,
history, and compatibility metadata.

## Model Architecture

### DINO study embedding

The existing checkpoint-compatible DINOv3 encoder continues to consume each
present acquisition slot as one sixteen-channel image. It injects the learned
slot token and produces CLS and patch tokens. For fusion, the legacy `xcodex`
twelve-logit readout is bypassed after external initialization.

For each study, CLS embeddings from all present slots are reduced with mean and
maximum pooling. The concatenated 768-dimensional representation is normalized
and projected to the configured 256-dimensional fusion space. Missing slots are
excluded from the reduction.

### SAM study embedding

The existing SAM classifier preprocessing remains unchanged: each valid
grayscale slice is expanded to three channels, resized, normalized, and encoded
by SAM ViT-B. Slice features are averaged across all valid slices and slots for
the study, producing a 256-dimensional representation. The legacy SAM
twelve-logit head is bypassed after external initialization.

All SAM image-encoder parameters are trainable. Since the fusion path uses the
encoder embedding directly, the old SAM classifier head is not part of the
joint model state.

### Fusion decoder

The DINO and SAM study representations are separately normalized and projected
to `fusion_dim: 256`. Learned modality embeddings are added, producing a memory
sequence with two tokens per study:

```text
[DINO study token, SAM study token]
```

Twelve learned label-query tokens form the decoder target sequence. A two-layer
`TransformerDecoder` with eight attention heads and a 1024-dimensional
feed-forward block lets every label query attend to both modality tokens. The
decoder uses GELU activation, pre-normalization, and configured dropout.

After final layer normalization, each label query is converted to one logit by
a label-specific weight and bias. The output shape is `[batch, 12]`.

## Data Flow

The shared `StudyDataset` supplies one cache item containing six slots. The
joint model receives `images`, `slot_mask`, and `slice_mask` in one forward
call. DINO uses all sixteen cached slices as its channel dimension. SAM uses the
same cached slices individually. No duplicate cache or separate branch loader
is required.

The trainer invokes:

```text
logits = model(images, slot_mask, slice_mask)
loss = weighted_bce(logits, targets, weights)
```

One backward pass updates every trainable fusion component end-to-end.

## Training and Memory Controls

The default joint batch size is one. Gradient accumulation supplies a larger
effective batch without retaining multiple studies simultaneously. AMP remains
enabled on CUDA. Gradient clipping and cosine scheduling remain supported.

All SAM encoder blocks and the encoder neck are trainable. All DINO encoder
weights are trainable unless a future configuration explicitly freezes them.
Folds train sequentially, and model/CUDA memory is released between folds.

Macro ROC AUC on the held-out fold selects `fold_<n>_best.pth`. The trainer also
writes `fold_<n>_last.pth` and `fold_<n>_history.json`. OOF fusion predictions
and metrics are written using the existing validated artifact conventions.

## Prediction

Prediction loads `fusion/fold_<n>_best.pth` for every configured fold, runs each
complete fusion model on the test cache, and averages the resulting sigmoid
probabilities. The averaged frame is validated and written directly to
`outputs/submission.csv`.

The following old inference artifacts are no longer produced:

- `test_knee.csv`
- `test_sam.csv`
- Fixed rank/probability 80/20 blending
- Global SAM prediction checkpoints

The optional diagnostic `outputs/test_fusion.csv` contains the same un-clipped
fold-averaged predictions before final submission clipping.

## Checkpoint Compatibility

External initialization is strict for every encoder tensor. The supplied DINO
checkpoint also contains legacy readout tensors; those tensors are validated
against a temporary checkpoint-compatible DINO model before only its encoder
state is copied into the fusion model. The supplied SAM state is translated as
before, then only encoder and normalization tensors are copied. Legacy SAM head
tensors are validated but are not retained by the fusion model.

Native fusion checkpoints contain an architecture fingerprint covering encoder
recipes, slice counts, fusion dimension, decoder layers, attention heads,
feed-forward dimension, dropout, and label order. Resume rejects mismatched
folds, caches, fold maps, labels, or architecture.

## Configuration Migration

Add a `fusion` section containing enablement, dimensions, decoder parameters,
batch size, epochs, optimizer settings, scheduler, accumulation, and clipping.
The knee and SAM sections retain encoder construction settings. SAM defaults to
full encoder training. The old `ensemble` blend settings remain accepted only
for backward-compatible configuration loading but are unused in fusion mode.

## Error Handling

Training stops before optimization for missing external checkpoints, wrong
DINO fold identity, label mismatch, encoder tensor mismatch, malformed SAM
serialization, empty train/validation partitions, studies without valid slots
or slices, incompatible native fusion checkpoints, or non-finite model output.

## Testing

Tests use tiny encoder substitutes and synthetic caches. They cover:

- DINO and SAM embedding extraction shapes and missing-slot masking.
- Two modality memory tokens and twelve label-query decoder outputs.
- Gradient flow into both encoders and the decoder from one weighted BCE loss.
- Strict external encoder initialization and exclusion of legacy classifier
  heads from the fusion state.
- Five-fold joint training topology and native resume precedence.
- Fold-averaged fusion prediction and final submission validation.
- The randomized in-memory pipeline test migrated from separate branch models
  to one tiny fusion factory.
- Full regression coverage for caching, targets, folds, and checkpoint safety.
