# End-to-End DINO–SAM Transformer Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace separate knee/global-SAM training and fixed blending with five jointly trained DINO–SAM transformer-fusion folds.

**Architecture:** DINO produces a pooled 768-dimensional study representation and SAM produces a 256-dimensional study representation. Separate projections create two 256-dimensional memory tokens; twelve learned label queries pass through a two-layer transformer decoder and label-specific outputs generate twelve logits trained with one weighted BCE loss.

**Tech Stack:** Python 3.12, PyTorch, timm, segment-anything, NumPy, pandas, pytest.

---

### Task 1: Embedding extractors and fusion decoder

**Files:**
- Create: `Models/FusionModel.py`
- Modify: `Models/DINOv3Knee.py`
- Modify: `Models/SAMClassifier.py`
- Test: `tests/test_fusion_model.py`

- [ ] Write failing tests using tiny differentiable encoders to assert two memory tokens, `[batch, 12]` logits, missing-slot behavior, and gradients in DINO, SAM, decoder, and label outputs.
- [ ] Run `.venv/bin/python -m pytest tests/test_fusion_model.py -q` and confirm import/API failures.
- [ ] Expose DINO and SAM study embeddings and implement the two-token, twelve-query transformer decoder with label-specific scalar outputs.
- [ ] Run focused tests to green.

### Task 2: Strict external encoder initialization

**Files:**
- Modify: `Training/external_checkpoints.py`
- Modify: `Models/FusionModel.py`
- Modify: `tests/test_external_checkpoints.py`
- Modify: `tests/test_fusion_model.py`

- [ ] Write failing tests that copy only encoder tensors from validated legacy DINO/SAM models and reject missing or mismatched encoder state.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement strict fusion initialization from fold-specific DINO and shared extracted SAM checkpoints, excluding legacy classifier heads from fusion state.
- [ ] Verify focused tests pass and supplied production encoders initialize strictly.

### Task 3: Fusion configuration, dataset, and trainer routing

**Files:**
- Modify: `config/training.yaml`
- Modify: `Training/config.py`
- Modify: `Training/checkpoints.py`
- Modify: `Training/dataset.py`
- Modify: `Training/trainer.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_dataset.py`
- Modify: `tests/test_trainer.py`

- [ ] Write failing tests for fusion configuration validation, fusion slice loading, `forward_batch(..., "fusion", ...)`, and architecture fingerprints.
- [ ] Run focused tests and confirm failures.
- [ ] Add fusion defaults and validation, allow fusion datasets to use knee slice count, route all three input tensors to the joint model, and fingerprint decoder plus encoder recipes.
- [ ] Run focused tests to green.

### Task 4: Five-fold joint training orchestration

**Files:**
- Modify: `TrainEnsemble.py`
- Modify: `Validators/validator.py`
- Modify: `tests/test_smoke_pipeline.py`
- Modify: `tests/test_randomized_ensemble_pipeline.py`

- [ ] Migrate synthetic tests to one tiny fusion factory and write failing assertions for five/two joint folds, fusion OOF coverage, fusion checkpoint topology, and absence of old branch artifacts.
- [ ] Run both smoke tests and confirm failures against the old orchestration.
- [ ] Replace separate branch loops with one fusion fold loop; apply native resume precedence and strict external initialization; write `oof_fusion.csv` and fusion metrics.
- [ ] Run both smoke tests to green.

### Task 5: Fold-averaged fusion prediction

**Files:**
- Modify: `TrainEnsemble.py`
- Modify: `Validators/validator.py`
- Modify: `tests/test_smoke_pipeline.py`
- Modify: `tests/test_randomized_ensemble_pipeline.py`

- [ ] Add failing assertions for `test_fusion.csv`, direct fold averaging, submission validation, and removal of test-knee/test-SAM artifacts.
- [ ] Run focused tests and confirm prediction topology failures.
- [ ] Load every fusion best checkpoint, average sigmoid predictions, write diagnostic and clipped submission files, and update validation.
- [ ] Run focused tests to green.

### Task 6: Documentation and verification

**Files:**
- Modify: `README.md`

- [ ] Document joint-fold architecture, external initialization, full SAM fine-tuning, memory settings, artifacts, resume behavior, and commands.
- [ ] Run `.venv/bin/python -m pytest -q` and resolve regressions.
- [ ] Run Python compilation for all modified modules.
- [ ] Strictly initialize the production fusion encoders from one supplied DINO fold and the supplied SAM checkpoint and report the result.
