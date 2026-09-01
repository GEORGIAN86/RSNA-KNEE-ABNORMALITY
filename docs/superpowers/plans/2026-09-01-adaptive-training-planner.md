# Adaptive Hardware-Aware Training Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CUDA-aware `auto` command that profiles hardware/data, generates and probes balanced candidates, writes an auditable resolved config, and optionally starts safe fusion training.

**Architecture:** A focused `Training/resource_planner.py` owns immutable profile/candidate/probe records and deterministic policy. Dataset/model changes provide separate DINO/SAM views and configurable encoder freezing; CLI orchestration supplies production profiling, probing, atomic reports, and training dispatch.

**Tech Stack:** Python standard library, PyTorch, pandas, NumPy, PyYAML, pytest.

---

### Task 1: Separate DINO/SAM dataset views

**Files:**
- Modify: `Training/dataset.py`
- Modify: `Training/trainer.py`
- Modify: `Models/FusionModel.py`
- Test: `tests/test_dataset.py`
- Test: `tests/test_fusion_model.py`

- [ ] Write failing tests for 16-slice DINO and two-slice SAM tensors, random training sampling, deterministic evaluation sampling, and joint forward routing.
- [ ] Run focused tests and confirm shape/API failures.
- [ ] Return `images`, `sam_images`, `slice_mask`, and `sam_slice_mask`; route both views into fusion forward.
- [ ] Run focused tests to green.

### Task 2: Configurable encoder freezing

**Files:**
- Modify: `Models/DINOv3Knee.py`
- Modify: `Models/SAMClassifier.py`
- Modify: `Models/FusionModel.py`
- Modify: `Training/config.py`
- Modify: `Training/checkpoints.py`
- Modify: `config/training.yaml`
- Test: `tests/test_fusion_model.py`
- Test: `tests/test_config.py`

- [ ] Write failing tests for `-1`, `0`, and positive block semantics and invalid block counts.
- [ ] Run focused tests and confirm missing configuration behavior.
- [ ] Implement freezing helpers, defaults DINO=4/SAM=2, separate slices, batch one, accumulation eight, and fingerprint fields.
- [ ] Run focused tests to green.

### Task 3: Pure resource-planning policy

**Files:**
- Create: `Training/resource_planner.py`
- Test: `tests/test_resource_planner.py`

- [ ] Write failing tests for hardware-profile schema, stable candidate generation/scoring, worker selection, analytical estimates, fallback ordering, and simulated OOM selection.
- [ ] Run focused tests and confirm import failures.
- [ ] Implement dataclasses and pure policy functions with injected probe callbacks.
- [ ] Run focused tests to green.

### Task 4: Data profiling and atomic outputs

**Files:**
- Modify: `Training/resource_planner.py`
- Test: `tests/test_resource_planner.py`

- [ ] Write failing temporary-project tests for CSV/UID/series/slice/cache/disk summaries and atomic JSON/YAML writing.
- [ ] Run focused tests and confirm failures.
- [ ] Implement standard-library/Pandas profiling and report/config writers.
- [ ] Run focused tests to green.

### Task 5: Production CUDA probe and auto orchestration

**Files:**
- Modify: `TrainEnsemble.py`
- Modify: `README.md`
- Modify: `tests/test_cli.py`
- Test: `tests/test_resource_planner.py`

- [ ] Write failing tests for `auto`, `--plan-only`, no-CUDA reporting, successful injected selection, and training dispatch.
- [ ] Run focused tests and confirm CLI failures.
- [ ] Implement hardware detection, representative cached batch construction, isolated CUDA forward/backward/optimizer probe, candidate application, atomic outputs, cache preparation, and optional training dispatch.
- [ ] Document command, selection policy, outputs, and recovery behavior.
- [ ] Run focused tests to green.

### Task 6: Full verification

**Files:**
- Modify only for genuine regressions.

- [ ] Run `.venv/bin/python -m pytest -q`.
- [ ] Run Python compilation over modified modules.
- [ ] Run `TrainEnsemble.py auto --plan-only` and confirm the current non-CUDA environment writes a report and exits safely without training.
