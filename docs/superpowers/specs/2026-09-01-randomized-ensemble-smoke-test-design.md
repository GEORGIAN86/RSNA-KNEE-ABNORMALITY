# Randomized Ensemble Smoke Test Design

## Objective

Add a fast, deterministic, in-memory-style integration test for the training
and prediction ensemble. The test generates all tabular inputs and fingerprinted
study caches inside pytest's temporary directory, injects tiny knee and SAM
models, and calls the real `run_train` and `run_predict` orchestration.

## Test Data

Use a fixed NumPy random generator seed. Generate enough training studies for
two non-empty report-hash folds and a small test set. Each cached study contains
six slots, three slices per slot, random uint8 pixels, valid slot and slice
masks, and the cache fingerprint expected by the configuration. Gold targets
contain all twelve canonical labels and ensure both classes occur.

The test writes the minimum required train, test, series, and sample-submission
CSVs plus cache manifests. It does not decode DICOM files, download model
weights, or depend on CUDA.

## Models and Configuration

Inject tiny PyTorch knee and SAM factories through the production factory
override boundary. Both models reduce cached pixels to a scalar feature and
apply a trainable linear head with twelve outputs. Configure two folds, one
epoch, CPU float32 execution, zero data-loader workers, fresh native training,
and no external checkpoint initialization.

## Execution and Assertions

Call `run_train(config, model_factories=factories)`, then
`run_predict(config, model_factories=factories)`.

Verify:

- Both native knee fold best and last checkpoints exist.
- SAM `global_last.pth` and `global_final.pth` exist.
- Knee OOF predictions contain every training UID exactly once.
- Metrics contain knee OOF metrics and global SAM history, without misleading
  SAM or blended OOF metrics.
- Knee and SAM test prediction files contain every test UID exactly once.
- The final submission has exact label order, finite values, and probabilities
  in `[0, 1]`.
- Rebuilding fold assignments from the same seeded input produces the same
  mapping, demonstrating deterministic structure without requiring identical
  floating-point training outputs.

## Verification

Run the new test alone first and observe its pre-implementation failure. After
implementation, run the focused test and then the complete pytest suite.
