# One-Command Docker Training and Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `docker compose run --rm --build trainer` download missing Kaggle DICOM data from `.env`, adaptively train every configured fold on CUDA, predict with the resolved training configuration, and return success only when `outputs/submission.csv` exists.

**Architecture:** Keep Docker Compose responsible for environment injection, GPU access, and persistent volumes. Keep `docker/entrypoint.sh` responsible for download/bootstrap and command sequencing, while the existing `TrainEnsemble.py auto` and `predict` commands remain responsible for preprocessing, adaptive training, checkpoint reuse, inference, and submission validation.

**Tech Stack:** Docker Compose, Bash, Python 3, Pytest, PyYAML, PyTorch CUDA pipeline

---

## File Structure

- Modify `docker/entrypoint.sh`: sequence adaptive training and prediction, validate required output artifacts, and preserve plan-only behavior.
- Modify `tests/test_docker_entrypoint.py`: simulate adaptive outputs and cover command order, failure propagation, missing artifacts, download behavior, and plan-only behavior.
- Modify `tests/test_docker_config.py`: require the documented one-command interface and accurate `.env.example` description in addition to existing GPU/volume assertions.
- Modify `.env.example`: describe `TRAIN_MODE=train` as training plus prediction.
- Modify `README.md`: document `.env` setup, prerequisites, the one command, persistent outputs, resume behavior, and plan-only usage.
- Do not modify `Dockerfile`, `docker-compose.yml`, model code, training code, data preprocessing, or checkpoint formats unless a failing verification test proves an existing declaration is incompatible with the approved design.

### Task 1: Run Prediction After Adaptive Training

**Files:**
- Modify: `tests/test_docker_entrypoint.py:10-32`
- Modify: `tests/test_docker_entrypoint.py:67-75`
- Modify: `docker/entrypoint.sh:112-116`

- [ ] **Step 1: Extend the fake Python executable to simulate real pipeline artifacts**

Replace the `python` entry in `_fake_tools` with this executable body so successful fake `auto` and `predict` commands leave the artifacts that the entrypoint will later validate:

```python
"python": '''#!/usr/bin/env bash
echo "python $*" >> "$COMMAND_LOG"
case "$*" in
  *"TrainEnsemble.py auto "*)
    fake_exit="${FAKE_AUTO_EXIT:-0}"
    [[ "$fake_exit" == "0" ]] || exit "$fake_exit"
    if [[ "${FAKE_SKIP_AUTO_CONFIG:-0}" != "1" ]]; then
      mkdir -p "${APP_ROOT}/outputs"
      printf 'runtime:\n  device: cuda\n' > "${APP_ROOT}/outputs/auto_training.yaml"
    fi
    ;;
  *"TrainEnsemble.py predict "*)
    fake_exit="${FAKE_PREDICT_EXIT:-0}"
    [[ "$fake_exit" == "0" ]] || exit "$fake_exit"
    if [[ "${FAKE_SKIP_SUBMISSION:-0}" != "1" ]]; then
      mkdir -p "${APP_ROOT}/outputs"
      printf 'StudyInstanceUID,ACL\nstudy,0.5\n' > "${APP_ROOT}/outputs/submission.csv"
    fi
    ;;
esac
''',
```

- [ ] **Step 2: Write the failing default-workflow test**

Replace `test_existing_dicoms_skip_kaggle_and_start_auto_training` with:

```python
def test_existing_dicoms_skip_kaggle_and_run_training_then_prediction(tmp_path: Path):
    project = _project(tmp_path)

    result = _run(tmp_path, project)

    assert result.returncode == 0, result.stderr
    commands = (tmp_path / "commands.log").read_text().splitlines()
    assert not any(line.startswith("kaggle ") for line in commands)
    auto = "python TrainEnsemble.py auto --config config/training.yaml"
    predict = "python TrainEnsemble.py predict --config outputs/auto_training.yaml"
    assert auto in commands
    assert predict in commands
    assert commands.index(auto) < commands.index(predict)
```

- [ ] **Step 3: Run the test and verify the RED state**

Run:

```bash
.venv/bin/python -m pytest tests/test_docker_entrypoint.py::test_existing_dicoms_skip_kaggle_and_run_training_then_prediction -q
```

Expected: FAIL because the current `TRAIN_MODE=train` branch invokes only `TrainEnsemble.py auto` and never logs the `predict` command.

- [ ] **Step 4: Implement the minimal training-then-prediction sequence**

Replace the final mode switch in `docker/entrypoint.sh` with:

```bash
case "$TRAIN_MODE" in
  train)
    "$PYTHON_BIN" TrainEnsemble.py auto --config config/training.yaml
    "$PYTHON_BIN" TrainEnsemble.py predict --config outputs/auto_training.yaml
    ;;
  plan)
    exec "$PYTHON_BIN" TrainEnsemble.py auto --plan-only --config config/training.yaml
    ;;
  *) fail "TRAIN_MODE must be train or plan" ;;
esac
```

- [ ] **Step 5: Run the focused entrypoint test and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_docker_entrypoint.py::test_existing_dicoms_skip_kaggle_and_run_training_then_prediction -q
```

Expected: `1 passed`.

- [ ] **Step 6: Run the complete entrypoint test module**

Run:

```bash
.venv/bin/python -m pytest tests/test_docker_entrypoint.py -q
```

Expected: all existing download, alias normalization, credential, and plan-mode tests pass with the artifact-producing fake Python executable.

- [ ] **Step 7: Commit the behavior**

```bash
git add docker/entrypoint.sh tests/test_docker_entrypoint.py
git commit -m "feat: run prediction after adaptive docker training"
```

### Task 2: Reject Missing Training and Prediction Artifacts

**Files:**
- Modify: `tests/test_docker_entrypoint.py`
- Modify: `docker/entrypoint.sh:10-11`
- Modify: `docker/entrypoint.sh:112-121`

- [ ] **Step 1: Add failure-propagation and artifact tests**

Add these tests after the default-workflow test:

```python
def test_training_failure_prevents_prediction(tmp_path: Path):
    project = _project(tmp_path)

    result = _run(tmp_path, project, FAKE_AUTO_EXIT="7")

    assert result.returncode == 7
    commands = (tmp_path / "commands.log").read_text()
    assert "TrainEnsemble.py auto --config config/training.yaml" in commands
    assert "TrainEnsemble.py predict" not in commands


def test_missing_resolved_config_fails_before_prediction(tmp_path: Path):
    project = _project(tmp_path)

    result = _run(tmp_path, project, FAKE_SKIP_AUTO_CONFIG="1")

    assert result.returncode != 0
    assert "adaptive configuration was not created" in result.stderr
    assert "TrainEnsemble.py predict" not in (tmp_path / "commands.log").read_text()


def test_missing_submission_fails_the_container(tmp_path: Path):
    project = _project(tmp_path)

    result = _run(tmp_path, project, FAKE_SKIP_SUBMISSION="1")

    assert result.returncode != 0
    assert "submission was not created" in result.stderr
    commands = (tmp_path / "commands.log").read_text()
    assert "TrainEnsemble.py predict --config outputs/auto_training.yaml" in commands
```

Replace the existing plan-mode test with:

```python
def test_plan_mode_writes_plan_without_prediction(tmp_path: Path):
    project = _project(tmp_path)

    result = _run(tmp_path, project, TRAIN_MODE="plan")

    assert result.returncode == 0, result.stderr
    commands = (tmp_path / "commands.log").read_text()
    assert "TrainEnsemble.py auto --plan-only --config config/training.yaml" in commands
    assert "TrainEnsemble.py predict" not in commands
    assert (project / "outputs" / "auto_training.yaml").is_file()
```

- [ ] **Step 2: Run the new artifact tests and verify the RED state**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_docker_entrypoint.py::test_missing_resolved_config_fails_before_prediction \
  tests/test_docker_entrypoint.py::test_missing_submission_fails_the_container -q
```

Expected: both tests FAIL because the current entrypoint neither checks `auto_training.yaml` nor checks `submission.csv`.

- [ ] **Step 3: Add a reusable non-empty artifact check**

Add this function next to `log` and `fail` in `docker/entrypoint.sh`:

```bash
require_nonempty_file() {
  local path="$1" message="$2"
  [[ -s "$path" ]] || fail "$message: $path"
}
```

- [ ] **Step 4: Enforce artifacts in both execution modes**

Replace the final mode switch with:

```bash
case "$TRAIN_MODE" in
  train)
    "$PYTHON_BIN" TrainEnsemble.py auto --config config/training.yaml
    require_nonempty_file \
      "${APP_ROOT}/outputs/auto_training.yaml" \
      "adaptive configuration was not created"
    "$PYTHON_BIN" TrainEnsemble.py predict --config outputs/auto_training.yaml
    require_nonempty_file \
      "${APP_ROOT}/outputs/submission.csv" \
      "submission was not created"
    log "training and prediction completed: ${APP_ROOT}/outputs/submission.csv"
    ;;
  plan)
    "$PYTHON_BIN" TrainEnsemble.py auto --plan-only --config config/training.yaml
    require_nonempty_file \
      "${APP_ROOT}/outputs/auto_training.yaml" \
      "adaptive configuration was not created"
    log "resource plan completed: ${APP_ROOT}/outputs/auto_training.yaml"
    ;;
  *) fail "TRAIN_MODE must be train or plan" ;;
esac
```

- [ ] **Step 5: Run all entrypoint tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_docker_entrypoint.py -q
```

Expected: every entrypoint test passes, including download, no-secret-output, command ordering, failure propagation, artifact enforcement, and plan-only behavior.

- [ ] **Step 6: Check Bash syntax**

Run:

```bash
bash -n docker/entrypoint.sh
```

Expected: exit code 0 with no output.

- [ ] **Step 7: Commit artifact enforcement**

```bash
git add docker/entrypoint.sh tests/test_docker_entrypoint.py
git commit -m "fix: require docker training output artifacts"
```

### Task 3: Document the One-Command Contract

**Files:**
- Modify: `tests/test_docker_config.py`
- Modify: `.env.example`
- Modify: `README.md`

- [ ] **Step 1: Write the failing documentation-contract test**

Add this test to `tests/test_docker_config.py`:

```python
def test_readme_documents_one_command_training_and_prediction():
    readme = (ROOT / "README.md").read_text()
    env_example = (ROOT / ".env.example").read_text()

    assert "docker compose run --rm --build trainer" in readme
    assert "outputs/submission.csv" in readme
    assert "training and prediction" in env_example.lower()
```

- [ ] **Step 2: Run the test and verify the RED state**

Run:

```bash
.venv/bin/python -m pytest tests/test_docker_config.py::test_readme_documents_one_command_training_and_prediction -q
```

Expected: FAIL because the README does not contain the single Compose command and `.env.example` describes training only.

- [ ] **Step 3: Update `.env.example`**

Replace its contents with:

```dotenv
# Copy this file to .env and paste a non-interactive Kaggle API token.
KAGGLE_API_TOKEN=
KAGGLE_COMPETITION=rsna-knee-abnormality-detection
# train performs adaptive training and prediction; plan stops after resource planning.
TRAIN_MODE=train
```

- [ ] **Step 4: Add the Docker workflow to the README**

Insert this section after environment setup and before the general Python command list:

````markdown
## One-command Docker training and prediction

The Docker workflow requires Docker Compose, the NVIDIA Container Toolkit, a CUDA-capable NVIDIA GPU, accepted Kaggle competition rules, and the supplied DINO/SAM checkpoints under `weights/checkpoints/`.

Create the local environment file and add a non-interactive Kaggle API token:

```bash
cp .env.example .env
```

Run the complete workflow:

```bash
docker compose run --rm --build trainer
```

The container downloads missing metadata and DICOMs from Kaggle, persists them under `data/`, builds or reuses `data/cache/`, adaptively trains all configured folds, and predicts the test set with `outputs/auto_training.yaml`. A successful exit produces `outputs/submission.csv`. Training checkpoints remain under `weights/checkpoints/fusion/`, so compatible interrupted runs resume automatically.

To profile resources and prepare data without training or prediction, set `TRAIN_MODE=plan` in `.env` and run the same Compose command.
````

- [ ] **Step 5: Run Docker configuration tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_docker_config.py -q
```

Expected: every Docker configuration and documentation-contract test passes.

- [ ] **Step 6: Render the Compose configuration**

Run:

```bash
docker compose config --quiet
```

Expected: exit code 0, confirming valid Compose syntax, `.env` loading, volume declarations, and GPU service configuration.

- [ ] **Step 7: Commit documentation**

```bash
git add .env.example README.md tests/test_docker_config.py
git commit -m "docs: add one-command docker training instructions"
```

### Task 4: Complete Verification

**Files:**
- Verify only; no planned production changes

- [ ] **Step 1: Run whitespace and shell checks**

Run:

```bash
git diff --check
bash -n docker/entrypoint.sh
```

Expected: both commands exit 0 with no diagnostics.

- [ ] **Step 2: Run focused Docker tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_docker_entrypoint.py tests/test_docker_config.py -q
```

Expected: all focused tests pass with zero failures.

- [ ] **Step 3: Run the complete test suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: the full repository test suite passes with zero failures.

- [ ] **Step 4: Validate the resolved Compose service**

Run:

```bash
docker compose config --quiet
docker compose config | sed -n '/trainer:/,/^[^ ]/p'
```

Expected: configuration rendering exits 0 and the trainer service includes the CUDA image build, `.env`, all three persistent mounts, and `gpus: all`. The focused Dockerfile test separately confirms `/app/docker/entrypoint.sh` is the image entrypoint.

- [ ] **Step 5: Build the trainer image**

Run:

```bash
docker compose build trainer
```

Expected: exit code 0 after installing the pinned CUDA-compatible dependencies and copying the application plus entrypoint into the image. Do not run the actual one-command workflow during verification because it downloads the full competition dataset and starts multi-fold GPU training.

- [ ] **Step 6: Review the final changes against the design**

Run:

```bash
git status --short
git log -4 --oneline
git diff HEAD~3 -- docker/entrypoint.sh docker-compose.yml Dockerfile .env.example README.md tests/test_docker_entrypoint.py tests/test_docker_config.py
```

Expected: only the approved entrypoint, test, and documentation changes are present; `Dockerfile` and `docker-compose.yml` remain unchanged unless verification exposed an incompatibility; the three implementation commits follow the approved design.
