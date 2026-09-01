# Randomized Ensemble Smoke Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic randomized integration test that trains and predicts through the real mixed knee-fold/global-SAM ensemble pipeline using tiny injected models.

**Architecture:** A dedicated pytest module creates temporary CSVs and valid cache artifacts, supplies CPU-only tiny model factories, calls `run_train` and `run_predict`, and validates every checkpoint and prediction artifact. Production code is unchanged unless the test exposes a genuine orchestration defect.

**Tech Stack:** Python, pytest, PyTorch, NumPy, pandas.

---

### Task 1: Randomized fixture and failing integration test

**Files:**
- Create: `tests/test_randomized_ensemble_pipeline.py`

- [ ] Implement tiny knee/SAM modules and helpers that create a two-fold configuration, seeded random CSV labels, fingerprinted random `.npz` caches, manifests, and a sample submission.
- [ ] Add one test calling `run_train` and `run_predict`, asserting two knee fold checkpoints, global SAM checkpoints, complete knee OOF coverage, deterministic folds, complete branch predictions, and a valid submission.
- [ ] Temporarily assert one expected randomized-test marker that production does not create, run `.venv/bin/python -m pytest tests/test_randomized_ensemble_pipeline.py -q`, and confirm the intended failure proves the new test executes the complete flow.
- [ ] Replace the temporary marker with the specified artifact assertions and rerun the focused test to green.

### Task 2: Regression verification

**Files:**
- Modify only if a genuine orchestration bug is found by Task 1.

- [ ] Run `.venv/bin/python -m pytest -q` and confirm the complete suite passes.
- [ ] Run `.venv/bin/python -m py_compile tests/test_randomized_ensemble_pipeline.py` and confirm compilation succeeds.
