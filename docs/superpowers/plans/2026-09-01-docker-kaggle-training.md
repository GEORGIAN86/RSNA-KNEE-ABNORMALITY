# Dockerized Kaggle Training Initialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a GPU Docker workflow that conditionally downloads the RSNA knee competition data and starts adaptive fusion training.

**Architecture:** A small shell entrypoint owns idempotent data initialization and delegates all resource decisions to `TrainEnsemble.py auto`. Docker Compose supplies GPU access, persistent bind mounts, and runtime credentials; shell tests isolate external commands with fakes.

**Tech Stack:** Docker, Docker Compose, Bash, Kaggle CLI, CUDA PyTorch, pytest.

---

### Task 1: Entrypoint contract tests

**Files:**
- Create: `tests/test_docker_entrypoint.py`
- Create: `docker/entrypoint.sh`

- [ ] Write subprocess tests using temporary project/data directories and fake `kaggle`, `nvidia-smi`, and Python executables.
- [ ] Verify tests fail because the entrypoint is absent.
- [ ] Implement DICOM detection, token validation, staged Kaggle download/extraction, canonical metadata mapping, input validation, CUDA validation, and train/plan dispatch.
- [ ] Run the focused tests to green.

### Task 2: Container definitions and secrets

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.dockerignore`
- Create: `.env`
- Create: `.env.example`
- Create/Modify: `.gitignore`
- Modify: `requirements.txt`
- Test: `tests/test_docker_config.py`

- [ ] Write structural tests for the CUDA image, Kaggle dependency, GPU reservation, persistent mounts, environment variables, and ignore rules.
- [ ] Verify structural tests fail before files exist.
- [ ] Add the minimal CUDA image and Compose configuration satisfying the contract.
- [ ] Run Docker configuration tests and shell syntax validation.

### Task 3: Canonical production paths and documentation

**Files:**
- Modify: `config/training.yaml`
- Modify: `README.md`
- Test: `tests/test_docker_config.py`

- [ ] Write a failing assertion that production paths are canonical.
- [ ] Replace duplicated local CSV filenames with canonical container paths.
- [ ] Document credential setup, competition-rule acceptance, build/run/plan commands, mounted files, skip behavior, and troubleshooting.
- [ ] Run focused tests to green.

### Task 4: Verification

**Files:**
- Modify only for genuine regressions.

- [ ] Run `.venv/bin/python -m pytest -q`.
- [ ] Run `bash -n docker/entrypoint.sh`.
- [ ] Run `docker compose config` when Docker is installed.
- [ ] Confirm `.env` contains placeholders only and no credential is printed by tests.
