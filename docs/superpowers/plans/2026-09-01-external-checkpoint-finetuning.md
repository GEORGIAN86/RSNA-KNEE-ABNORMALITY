# External Checkpoint Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Warm-start five DINOv3 knee folds from the supplied fold checkpoints, train one global SAM model from the extracted SAM checkpoint, and blend five-fold knee predictions with one SAM prediction.

**Architecture:** Add a strict external-checkpoint boundary, an exact DINOv3 knee model matching the supplied state dictionary, and global-training primitives for SAM. Preserve the native checkpoint format for all new outputs so native resumes remain strict and atomic.

**Tech Stack:** Python 3.12, PyTorch, timm, segment-anything, pandas, NumPy, pytest.

---

### Task 1: External checkpoint readers

**Files:**
- Create: `Training/external_checkpoints.py`
- Test: `tests/test_external_checkpoints.py`

- [ ] Write tests constructing small external knee payloads and extracted PyTorch archives, asserting fold/label validation, safe directory loading, SAM key translation, and explicit incompatibility errors.
- [ ] Run `.venv/bin/python -m pytest tests/test_external_checkpoints.py -q` and confirm failure because the compatibility module does not exist.
- [ ] Implement `resolve_external_knee_checkpoint`, `load_external_knee_state`, `load_extracted_torch_checkpoint`, and `translate_external_sam_state`, using `torch.load(..., weights_only=True)` and temporary ZIP reconstruction for extracted archives.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Checkpoint-compatible DINOv3 knee model

**Files:**
- Create: `Models/DINOv3Knee.py`
- Modify: `Models/__init__.py`
- Modify: `requirements.txt`
- Test: `tests/test_dinov3_knee.py`

- [ ] Write tests for the 16-channel patch input, conditional-token encoder, label-specific attention/readout, output shape, and strict loading of a representative supplied-style state dictionary.
- [ ] Run `.venv/bin/python -m pytest tests/test_dinov3_knee.py -q` and confirm the new model imports fail.
- [ ] Implement the checkpoint-compatible encoder/readout with state names matching `enc.vit.*`, `enc.tok.*`, and `readout.*`; build the DINOv3 backbone through timm and adapt its patch projection to 16 channels.
- [ ] Add the pinned-compatible timm requirement and export the model builder.
- [ ] Run the focused model tests and confirm they pass.

### Task 3: Global SAM trainer

**Files:**
- Modify: `Training/trainer.py`
- Test: `tests/test_trainer.py`

- [ ] Write a failing test proving global training consumes all records, writes `global_last.pth` per epoch and `global_final.pth` at completion, and resumes from the native last checkpoint.
- [ ] Run the focused global-trainer test and confirm the missing API failure.
- [ ] Add `GlobalTrainResult` and `train_global`, reusing weighted BCE, scheduler, scaler, atomic checkpoint saving, history writing, and strict native resume metadata.
- [ ] Run `tests/test_trainer.py` and confirm all trainer tests pass.

### Task 4: Configuration and validation

**Files:**
- Modify: `config/training.yaml`
- Modify: `Training/config.py`
- Modify: `Training/checkpoints.py`
- Modify: `Validators/validator.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_validation_and_checkpoints.py`

- [ ] Write failing tests for external knee path templates, the extracted SAM directory, global SAM checkpoint naming, DINOv3 architecture fingerprints, and command-specific validation.
- [ ] Run the focused configuration and validation tests and verify the expected failures.
- [ ] Add explicit external initialization settings, update the knee and SAM defaults, extend architecture fingerprints, and validate native-resume-versus-external-initialization requirements.
- [ ] Run both focused test modules and confirm they pass.

### Task 5: Training orchestration

**Files:**
- Modify: `TrainEnsemble.py`
- Modify: `tests/test_ensemble.py`
- Modify: `tests/test_smoke_pipeline.py`

- [ ] Write failing orchestration tests proving each knee fold receives its matching warm start, native resume takes precedence, SAM trains exactly once on all records, and SAM/blended OOF files are not produced.
- [ ] Run the focused tests and verify behavior failures.
- [ ] Split branch orchestration into five-fold knee training and global SAM training, load external states strictly only when no native resume exists, save knee OOF metrics, and report SAM loss history without OOF AUC.
- [ ] Run the focused tests and confirm they pass.

### Task 6: Prediction orchestration

**Files:**
- Modify: `TrainEnsemble.py`
- Modify: `Validators/validator.py`
- Modify: `tests/test_ensemble.py`
- Modify: `tests/test_smoke_pipeline.py`

- [ ] Write failing tests proving prediction averages all knee folds, loads only `sam/global_final.pth`, blends both branches, and validates the resulting submission.
- [ ] Run the focused tests and verify they fail for the old fold-based SAM path.
- [ ] Update prediction and checkpoint coverage validation for branch-specific checkpoint topology.
- [ ] Run focused orchestration and smoke tests and confirm they pass.

### Task 7: Documentation and full verification

**Files:**
- Modify: `README.md`

- [ ] Update commands, checkpoint layout, warm-start precedence, global SAM behavior, output artifacts, and dependency/setup notes.
- [ ] Run `.venv/bin/python -m pytest -q` and resolve every regression.
- [ ] Run `python TrainEnsemble.py validate --config config/training.yaml` to verify production-path diagnostics against the supplied files; record any missing data separately from code failures.
