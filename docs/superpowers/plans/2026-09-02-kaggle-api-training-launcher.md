# Kaggle API Training Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one local command that publishes reusable private Kaggle inputs, launches the existing pipeline on a private T4 GPU kernel, waits for completion, and downloads all training and prediction artifacts.

**Architecture:** A standard-library Python launcher packages deterministic inputs and treats the public `kaggle` CLI as its remote boundary. A separate Kaggle runner maps read-only competition/source inputs and temporary cache/weight storage into a generated training config, then invokes the existing `auto` and `predict` commands without changing model behavior.

**Tech Stack:** Python 3.12 standard library, Kaggle CLI 2.x, PyYAML, Pytest, Kaggle managed CUDA/PyTorch runtime

---

## File Structure

- Create `scripts/kaggle_train.py`: local configuration, deterministic packaging, private dataset synchronization, kernel submission, status polling, redaction, and output download.
- Create `kaggle_job/run_training.py`: remote path discovery, extraction, Kaggle-specific config generation, dependency installation, CUDA validation, training, prediction, and artifact validation.
- Create `requirements-kaggle.txt`: Kaggle additions that deliberately omit PyTorch and torchvision.
- Create `tests/test_kaggle_launcher.py`: unit and fake-CLI integration tests for all local behavior.
- Create `tests/test_kaggle_job.py`: isolated tests for remote config mapping and command sequencing.
- Modify `.env.example`: document the required username as well as API token.
- Modify `.gitignore`: ignore local launcher state and downloaded Kaggle run directories.
- Modify `README.md`: document setup, first-run upload, launch options, output location, privacy, and limitations.

### Task 1: Launcher Configuration and Deterministic Bundles

**Files:**
- Create: `scripts/kaggle_train.py`
- Create: `tests/test_kaggle_launcher.py`

- [ ] **Step 1: Write failing configuration and packaging tests**

Create `tests/test_kaggle_launcher.py` with temporary repository helpers and these tests:

```python
import json
import zipfile
from pathlib import Path

import pytest

from scripts.kaggle_train import (
    LauncherError,
    build_source_archive,
    load_dotenv,
    sha256_file,
    validate_weights_archive,
)


def test_load_dotenv_does_not_override_existing_environment(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("KAGGLE_API_TOKEN=file-token\nKAGGLE_USERNAME=file-user\n")
    env = {"KAGGLE_API_TOKEN": "process-token"}

    load_dotenv(env_file, env)

    assert env == {
        "KAGGLE_API_TOKEN": "process-token",
        "KAGGLE_USERNAME": "file-user",
    }


def test_source_archive_is_deterministic_and_excludes_runtime_data(tmp_path: Path):
    root = tmp_path / "project"
    for relative, content in {
        "Helpers/cache.py": "CACHE = True\n",
        "Models/model.py": "MODEL = True\n",
        "Training/train.py": "TRAIN = True\n",
        "Validators/validator.py": "VALID = True\n",
        "config/training.yaml": "runtime: {}\n",
        "TrainEnsemble.py": "print('train')\n",
        "preprocess.py": "print('preprocess')\n",
        "train.py": "print('compat')\n",
        "requirements-kaggle.txt": "pydicom\n",
        "data/llm_labels_v4_blend.csv": "StudyInstanceUID\na\n",
        ".env": "KAGGLE_API_TOKEN=secret\n",
        "weights/checkpoint.pt": "weight",
        "outputs/result.csv": "result",
        "data/train.csv": "private raw data",
        "tests/test_unused.py": "assert True\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    build_source_archive(root, first)
    build_source_archive(root, second)

    assert sha256_file(first) == sha256_file(second)
    with zipfile.ZipFile(first) as archive:
        names = set(archive.namelist())
    assert "Helpers/cache.py" in names
    assert "data/llm_labels_v4_blend.csv" in names
    assert ".env" not in names
    assert "weights/checkpoint.pt" not in names
    assert "outputs/result.csv" not in names
    assert "data/train.csv" not in names
    assert "tests/test_unused.py" not in names


def test_weight_archive_requires_every_external_checkpoint(tmp_path: Path):
    archive_path = tmp_path / "weights.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for fold in range(4):
            archive.writestr(f"weights/checkpoints/knee/m_f{fold}.pt", b"weight")
        archive.writestr(
            "weights/checkpoints/sam/submissions_epoch_8_step_11550/data.pkl",
            b"sam",
        )

    with pytest.raises(LauncherError, match="m_f4.pt"):
        validate_weights_archive(archive_path)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_kaggle_launcher.py -q
```

Expected: collection fails because `scripts.kaggle_train` does not exist.

- [ ] **Step 3: Implement configuration loading and deterministic packaging**

Create `scripts/kaggle_train.py` with:

```python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Mapping, MutableMapping, Sequence


SCHEMA_VERSION = 1
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
SOURCE_DIRECTORIES = ("Helpers", "Models", "Training", "Validators", "config")
SOURCE_FILES = (
    "TrainEnsemble.py",
    "preprocess.py",
    "train.py",
    "requirements-kaggle.txt",
    "data/llm_labels_v4_blend.csv",
)
TERMINAL_SUCCESS = {"complete", "completed"}
TERMINAL_FAILURE = {"error", "failed", "failure", "cancelled", "canceled"}
ACTIVE_STATES = {"queued", "pending", "running"}


class LauncherError(RuntimeError):
    pass


def load_dotenv(path: Path, environment: MutableMapping[str, str]) -> None:
    if not path.is_file():
        return
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise LauncherError(f"invalid .env entry on line {number}")
        name, value = line.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise LauncherError(f"invalid .env variable on line {number}")
        environment.setdefault(name, value.strip().strip("'\""))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory in SOURCE_DIRECTORIES:
        base = root / directory
        if not base.is_dir():
            raise LauncherError(f"required source directory is missing: {base}")
        paths.extend(path for path in base.rglob("*") if path.is_file())
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise LauncherError(f"required source file is missing: {path}")
        paths.append(path)
    return sorted(set(paths), key=lambda path: path.relative_to(root).as_posix())


def build_source_archive(root: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in _source_paths(root):
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def validate_weights_archive(path: Path) -> None:
    if not path.is_file():
        raise LauncherError(f"weights archive is missing: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile as exc:
        raise LauncherError(f"invalid weights archive: {path}") from exc
    required = {f"weights/checkpoints/knee/m_f{fold}.pt" for fold in range(5)}
    missing = sorted(required - names)
    sam_prefix = "weights/checkpoints/sam/submissions_epoch_8_step_11550/"
    if not any(name.startswith(sam_prefix) and not name.endswith("/") for name in names):
        missing.append(sam_prefix)
    if missing:
        raise LauncherError("weights archive is missing: " + ", ".join(missing))
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_kaggle_launcher.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit launcher foundations**

```bash
git add scripts/kaggle_train.py tests/test_kaggle_launcher.py
git commit -m "feat: package Kaggle training inputs deterministically"
```

### Task 2: Private Dataset Synchronization and Kernel Metadata

**Files:**
- Modify: `scripts/kaggle_train.py`
- Modify: `tests/test_kaggle_launcher.py`

- [ ] **Step 1: Add failing dataset and metadata tests**

Append tests that inject a fake CLI runner and assert the public command boundary:

```python
from scripts.kaggle_train import build_kernel_metadata, sync_dataset


class FakeCli:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def run(self, args, *, check=True):
        self.calls.append(list(args))
        return self.responses.pop(0)


def completed(returncode=0, stdout="", stderr=""):
    return __import__("subprocess").CompletedProcess([], returncode, stdout, stderr)


def test_sync_dataset_creates_missing_private_dataset(tmp_path: Path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "manifest.json").write_text(json.dumps({"sha256": "abc"}))
    cli = FakeCli([completed(1, stderr="404 - Not Found"), completed()])

    action = sync_dataset("owner/private-source", stage, cli, tmp_path / "probe")

    assert action == "created"
    assert cli.calls[-1] == ["datasets", "create", "-p", str(stage), "--quiet"]


def test_sync_dataset_reuses_matching_remote_manifest(tmp_path: Path):
    stage = tmp_path / "stage"
    stage.mkdir()
    manifest = {"schema_version": 1, "sha256": "abc"}
    (stage / "manifest.json").write_text(json.dumps(manifest))
    probe = tmp_path / "probe"

    def run(args, *, check=True):
        if args[:2] == ["datasets", "files"]:
            return completed()
        probe.mkdir(parents=True, exist_ok=True)
        (probe / "manifest.json").write_text(json.dumps(manifest))
        return completed()

    cli = type("Cli", (), {"run": staticmethod(run)})()
    assert sync_dataset("owner/private-source", stage, cli, probe) == "reused"


def test_sync_dataset_versions_changed_content(tmp_path: Path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "manifest.json").write_text(json.dumps({"sha256": "new"}))
    probe = tmp_path / "probe"

    def run(args, *, check=True):
        if args[:2] == ["datasets", "download"]:
            probe.mkdir(parents=True, exist_ok=True)
            (probe / "manifest.json").write_text(json.dumps({"sha256": "old"}))
        return completed()

    cli = type("Cli", (), {"run": staticmethod(run)})()
    action = sync_dataset("owner/private-source", stage, cli, probe)

    assert action == "versioned"


def test_dataset_probe_does_not_hide_authentication_failure(tmp_path: Path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "manifest.json").write_text("{}")
    cli = FakeCli([completed(1, stderr="401 Unauthorized")])

    with pytest.raises(LauncherError, match="probe private dataset"):
        sync_dataset("owner/private-source", stage, cli, tmp_path / "probe")


def test_kernel_metadata_is_private_t4_job_with_expected_sources():
    metadata = build_kernel_metadata(
        kernel="owner/rsna-knee-training",
        source_dataset="owner/source",
        weights_dataset="owner/weights",
        competition="rsna-knee-abnormality-detection",
        accelerator="NvidiaTeslaT4",
    )

    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == ["owner/source", "owner/weights"]
    assert metadata["competition_sources"] == ["rsna-knee-abnormality-detection"]
```

- [ ] **Step 2: Run these tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_kaggle_launcher.py -q
```

Expected: collection fails because `sync_dataset` and `build_kernel_metadata` do not exist.

- [ ] **Step 3: Implement private synchronization and metadata generation**

Add these public functions and small JSON helpers to `scripts/kaggle_train.py`:

```python
def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LauncherError(f"invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise LauncherError(f"JSON object required: {path}")
    return value


def _is_not_found(result: subprocess.CompletedProcess[str]) -> bool:
    message = f"{result.stdout}\n{result.stderr}".lower()
    return result.returncode != 0 and any(
        marker in message for marker in ("404", "not found", "does not exist")
    )


def sync_dataset(slug: str, stage: Path, cli, probe: Path) -> str:
    result = cli.run(["datasets", "files", slug, "--page-size", "1"], check=False)
    if _is_not_found(result):
        cli.run(["datasets", "create", "-p", str(stage), "--quiet"])
        return "created"
    if result.returncode != 0:
        raise LauncherError(f"failed to probe private dataset {slug}: {result.stderr.strip()}")
    probe.mkdir(parents=True, exist_ok=True)
    download = cli.run(
        [
            "datasets", "download", slug, "-f", "manifest.json",
            "-p", str(probe), "--force", "--quiet",
        ],
        check=False,
    )
    remote_path = probe / "manifest.json"
    if download.returncode == 0 and remote_path.is_file():
        if _read_json(remote_path) == _read_json(stage / "manifest.json"):
            return "reused"
    elif download.returncode != 0 and not _is_not_found(download):
        raise LauncherError(f"failed to read manifest for {slug}: {download.stderr.strip()}")
    cli.run(
        [
            "datasets", "version", "-p", str(stage),
            "--message", "Update launcher input bundle", "--quiet",
        ]
    )
    return "versioned"


def build_kernel_metadata(
    *, kernel: str, source_dataset: str, weights_dataset: str,
    competition: str, accelerator: str,
) -> dict:
    return {
        "id": kernel,
        "title": kernel.split("/", 1)[1].replace("-", " ").title(),
        "code_file": "run_training.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": accelerator,
        "dataset_sources": [source_dataset, weights_dataset],
        "competition_sources": [competition],
        "kernel_sources": [],
        "model_sources": [],
    }
```

Add the staging helpers exactly as follows; the hard-link fallback avoids a
second 941 MB copy when the staging directory shares a filesystem:

```python
def _write_json(path: Path, value: Mapping) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stage_dataset(
    *, stage: Path, slug: str, title: str, bundle: Path, bundle_name: str,
) -> dict:
    stage.mkdir(parents=True, exist_ok=False)
    destination = stage / bundle_name
    try:
        os.link(bundle, destination)
    except OSError:
        shutil.copy2(bundle, destination)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "bundle": bundle_name,
        "sha256": sha256_file(bundle),
    }
    _write_json(stage / "manifest.json", manifest)
    _write_json(
        stage / "dataset-metadata.json",
        {
            "title": title,
            "id": slug,
            "licenses": [{"name": "unknown"}],
        },
    )
    return manifest
```

- [ ] **Step 4: Run launcher tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_kaggle_launcher.py -q
```

Expected: all launcher tests pass.

- [ ] **Step 5: Commit dataset synchronization**

```bash
git add scripts/kaggle_train.py tests/test_kaggle_launcher.py
git commit -m "feat: synchronize private Kaggle training datasets"
```

### Task 3: Submission, Polling, Redaction, and Output Download

**Files:**
- Modify: `scripts/kaggle_train.py`
- Modify: `tests/test_kaggle_launcher.py`

- [ ] **Step 1: Write failing status and end-to-end fake CLI tests**

Add direct status, polling, timeout, failure, and redaction tests:

```python
from scripts.kaggle_train import parse_kernel_status, redact, wait_for_kernel


class StatusCli:
    def __init__(self, statuses):
        self.statuses = iter(statuses)
        self.calls = []

    def run(self, args, *, check=True):
        self.calls.append(args)
        status = next(self.statuses)
        return completed(stdout=f"Kernel status: {status}\n")


def test_wait_for_kernel_polls_until_complete():
    cli = StatusCli(["queued", "running", "complete"])

    assert wait_for_kernel(
        cli, "owner/job", poll_seconds=0, timeout_seconds=10,
        monotonic=iter([0, 1, 2]).__next__, sleep=lambda _: None,
    ) == "complete"
    assert len(cli.calls) == 3


def test_wait_for_kernel_rejects_failed_job():
    with pytest.raises(LauncherError, match="failed with status error"):
        wait_for_kernel(
            StatusCli(["error"]), "owner/job", poll_seconds=0,
            timeout_seconds=10, monotonic=iter([0]).__next__, sleep=lambda _: None,
        )


def test_wait_for_kernel_times_out_without_cancelling():
    cli = StatusCli(["running", "running"])
    with pytest.raises(LauncherError, match="timed out"):
        wait_for_kernel(
            cli, "owner/job", poll_seconds=0, timeout_seconds=1,
            monotonic=iter([0, 2]).__next__, sleep=lambda _: None,
        )
    assert not any("delete" in call for call in cli.calls)


def test_redact_replaces_every_nonempty_secret():
    assert redact("token=abc user=name", ["abc", ""]) == "token=*** user=name"
```

Use a fake executable for two subprocess-level tests. It records each operation
in `FAKE_KAGGLE_LOG`; its status command consumes lines from
`FAKE_KAGGLE_STATUSES`; its output command creates
`artifacts/outputs/submission.csv`. The synchronous test must assert this
ordering:

```python
assert operations == [
    "datasets files",
    "datasets create",
    "datasets files",
    "datasets create",
    "kernels push",
    "kernels status",
    "kernels status",
    "kernels status",
    "kernels output",
]
```

The asynchronous test must assert that `kernels push` is the last operation and
that no local output directory is created. Invoke `main([...])` directly with a
temporary repository root option so no real CLI, network, or credentials are
used. Pass `--folds 0,2` and inspect the generated `job-config.json` copied by
the fake push command to assert `{"folds": "0,2"}`. Make the fake CLI fail with
stderr containing the test token and assert the captured launcher error contains
`***` but not the token.

Add direct parser assertions:

```python
from scripts.kaggle_train import parse_kernel_status


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Kernel status: queued", "queued"),
        ("Kernel status: running", "running"),
        ("Kernel status: complete", "complete"),
        ("status: error", "error"),
    ],
)
def test_parse_kernel_status(text, expected):
    assert parse_kernel_status(text) == expected
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_kaggle_launcher.py -q
```

Expected: failures identify missing parser, orchestration, and CLI entry point.

- [ ] **Step 3: Implement the CLI boundary and orchestration**

Complete `scripts/kaggle_train.py` with:

```python
def parse_kernel_status(text: str) -> str:
    match = re.search(r"(?:kernel\s+)?status\s*:\s*([a-z_]+)", text, re.I)
    if not match:
        raise LauncherError(f"could not parse Kaggle kernel status: {text.strip()}")
    return match.group(1).lower()


def redact(text: str, secrets: Sequence[str]) -> str:
    result = text
    for secret in secrets:
        if secret:
            result = result.replace(secret, "***")
    return result
```

Implement `KaggleCli.run(args, check=True)` so it invokes
`[executable, *args]` with the prepared environment, captures text output, and
raises `LauncherError` containing the redacted operation and stderr on checked
failure. Do not include token values in argv.

Implement `wait_for_kernel()` with this exact interface and state handling:

```python
def wait_for_kernel(
    cli,
    kernel: str,
    *,
    poll_seconds: float,
    timeout_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    started = monotonic()
    while True:
        result = cli.run(["kernels", "status", kernel])
        status = parse_kernel_status(f"{result.stdout}\n{result.stderr}")
        if status in TERMINAL_SUCCESS:
            return status
        if status in TERMINAL_FAILURE:
            raise LauncherError(f"Kaggle kernel {kernel} failed with status {status}")
        if status not in ACTIVE_STATES:
            raise LauncherError(f"Kaggle kernel {kernel} returned unknown status {status}")
        if monotonic() - started >= timeout_seconds:
            raise LauncherError(f"waiting for Kaggle kernel {kernel} timed out")
        sleep(poll_seconds)
```

Build an argument parser with these options and defaults:

```text
--competition rsna-knee-abnormality-detection
--source-dataset <username>/rsna-knee-training-source
--weights-dataset <username>/rsna-knee-training-weights
--kernel <username>/rsna-knee-training
--accelerator NvidiaTeslaT4
--folds (unset)
--poll-seconds 30
--timeout-seconds 86400
--output-dir (new outputs/kaggle/YYYYMMDDTHHMMSSZ)
--no-wait false
```

The top-level sequence must:

```python
load_dotenv(root / ".env", environment)
validate credentials, username, slugs, CLI, source files, and weights.zip
build source.zip and both private dataset stages in TemporaryDirectory()
sync source, then weights
copy kaggle_job/run_training.py into the kernel stage
write job-config.json with source/weight/competition slugs and optional folds
write kernel-metadata.json
cli.run(["kernels", "push", "-p", str(kernel_stage), "--accelerator", accelerator])
return immediately when --no-wait is set
wait_for_kernel(...)
create a previously absent output directory
cli.run(["kernels", "output", kernel_slug, "-p", str(output), "--quiet"])
require output/artifacts/outputs/submission.csv
```

Wrap `main()` so `LauncherError` prints one concise redacted error to stderr and
returns exit code 2.

- [ ] **Step 4: Run launcher tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_kaggle_launcher.py -q
```

Expected: all packaging, synchronization, polling, failure, no-wait, and secret
tests pass.

- [ ] **Step 5: Commit orchestration**

```bash
git add scripts/kaggle_train.py tests/test_kaggle_launcher.py
git commit -m "feat: launch and monitor Kaggle GPU training"
```

### Task 4: Kaggle Remote Runner

**Files:**
- Create: `kaggle_job/run_training.py`
- Create: `requirements-kaggle.txt`
- Create: `tests/test_kaggle_job.py`

- [ ] **Step 1: Write failing remote config and sequencing tests**

Create `tests/test_kaggle_job.py`:

```python
from pathlib import Path

import yaml

from kaggle_job.run_training import build_kaggle_config, execute_pipeline


def test_build_kaggle_config_maps_inputs_scratch_and_artifacts(tmp_path: Path):
    source = tmp_path / "training.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "data_dir": "data",
                    "cache_dir": "data/cache",
                    "train_csv": "data/train.csv",
                    "test_csv": "data/test.csv",
                    "train_series_csv": "data/train_series.csv",
                    "test_series_csv": "data/test_series.csv",
                    "sample_submission_csv": "data/sample_submission.csv",
                    "labels_csv": "data/llm_labels_v4_blend.csv",
                    "external_knee_checkpoint": "weights/checkpoints/knee/m_f{fold}.pt",
                    "external_sam_checkpoint": "weights/checkpoints/sam/model",
                    "checkpoints_dir": "weights/checkpoints",
                    "outputs_dir": "outputs",
                }
            }
        )
    )
    project = tmp_path / "project"
    competition = tmp_path / "competition"
    scratch = tmp_path / "scratch"
    artifacts = tmp_path / "artifacts"

    result = build_kaggle_config(source, project, competition, scratch, artifacts)

    paths = result["paths"]
    assert paths["train_csv"] == str(competition / "train.csv")
    assert paths["train_series_csv"] == str(competition / "train_series.csv")
    assert paths["labels_csv"] == str(project / "data/llm_labels_v4_blend.csv")
    assert paths["cache_dir"] == str(scratch / "cache")
    assert paths["external_knee_checkpoint"] == str(
        scratch / "weights/checkpoints/knee/m_f{fold}.pt"
    )
    assert paths["checkpoints_dir"] == str(artifacts / "checkpoints")
    assert paths["outputs_dir"] == str(artifacts / "outputs")


def test_execute_pipeline_trains_before_predicting_and_forwards_folds(tmp_path: Path):
    config = tmp_path / "kaggle_training.yaml"
    config.write_text("paths: {}\n")
    resolved = tmp_path / "artifacts/outputs/auto_training.yaml"
    submission = tmp_path / "artifacts/outputs/submission.csv"
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if "auto" in args:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text("runtime: {}\n")
        else:
            submission.write_text("StudyInstanceUID,ACL\na,0.5\n")

    execute_pipeline(
        python="python",
        project=tmp_path,
        config=config,
        artifacts=tmp_path / "artifacts",
        folds="0,2",
        run=run,
    )

    assert calls == [
        ["python", "TrainEnsemble.py", "auto", "--config", str(config), "--folds", "0,2"],
        ["python", "TrainEnsemble.py", "predict", "--config", str(resolved), "--folds", "0,2"],
    ]
```

Add these failure tests:

```python
import subprocess

import pytest


def test_execute_pipeline_stops_when_training_fails(tmp_path: Path):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        raise subprocess.CalledProcessError(7, args)

    with pytest.raises(subprocess.CalledProcessError):
        execute_pipeline(
            python="python", project=tmp_path, config=tmp_path / "config.yaml",
            artifacts=tmp_path / "artifacts", folds=None, run=run,
        )
    assert len(calls) == 1


def test_execute_pipeline_requires_resolved_config_before_prediction(tmp_path: Path):
    calls = []

    with pytest.raises(RuntimeError, match="adaptive configuration"):
        execute_pipeline(
            python="python", project=tmp_path, config=tmp_path / "config.yaml",
            artifacts=tmp_path / "artifacts", folds=None,
            run=lambda args, **kwargs: calls.append(args),
        )
    assert len(calls) == 1


def test_execute_pipeline_requires_submission_after_prediction(tmp_path: Path):
    artifacts = tmp_path / "artifacts"

    def run(args, **kwargs):
        if "auto" in args:
            resolved = artifacts / "outputs/auto_training.yaml"
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text("runtime: {}\n")

    with pytest.raises(RuntimeError, match="submission"):
        execute_pipeline(
            python="python", project=tmp_path, config=tmp_path / "config.yaml",
            artifacts=artifacts, folds=None, run=run,
        )
```

- [ ] **Step 2: Run remote-runner tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_kaggle_job.py -q
```

Expected: collection fails because `kaggle_job.run_training` does not exist.

- [ ] **Step 3: Implement the remote runner**

Create `kaggle_job/run_training.py` with import-safe helpers and a `main()` that:

```python
job_config = json.loads(Path(__file__).with_name("job-config.json").read_text())
input_root = Path("/kaggle/input")
working = Path("/kaggle/working")
scratch = Path("/kaggle/temp/rsna-knee-training")
source_input = input_root / job_config["source_dataset"].split("/", 1)[1]
weights_input = input_root / job_config["weights_dataset"].split("/", 1)[1]
competition = input_root / job_config["competition"]
project = working / "project"
artifacts = working / "artifacts"
```

Add `safe_extract_zip(archive, destination)` and test it with an archive member
named `../escape.txt`; it must raise `RuntimeError("unsafe archive member")` and
must not create the escaped file. `main()` must reject missing inputs, extract
both archives with that helper, install `requirements-kaggle.txt` using
`[sys.executable, "-m", "pip", "install", "-r", requirements]`, verify CUDA
using `[sys.executable, "-c", "import torch; assert torch.cuda.is_available()"]`,
generate the config through `build_kaggle_config`, write it with
`yaml.safe_dump`, and call `execute_pipeline`.

`build_kaggle_config()` must replace every path named in the design while
leaving all model/training settings unchanged. `execute_pipeline()` must use
`subprocess.run(..., cwd=project, check=True)`, require a non-empty resolved
config before prediction, and require a non-empty submission afterward.

Create `requirements-kaggle.txt`:

```text
transformers
timm>=1.0.20,<2
segment-anything
pydicom
opencv-python-headless
PyYAML
scikit-learn
tqdm
```

Do not add `torch` or `torchvision`.

- [ ] **Step 4: Run remote tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_kaggle_job.py -q
```

Expected: all remote config, sequencing, safe extraction, and failure tests pass.

- [ ] **Step 5: Commit the Kaggle runner**

```bash
git add kaggle_job/run_training.py requirements-kaggle.txt tests/test_kaggle_job.py
git commit -m "feat: run training pipeline in Kaggle kernels"
```

### Task 5: User Configuration and Documentation

**Files:**
- Modify: `.env.example`
- Modify: `.gitignore`
- Modify: `README.md`
- Modify: `tests/test_kaggle_launcher.py`

- [ ] **Step 1: Write a failing documentation contract test**

Add:

```python
def test_readme_documents_one_command_kaggle_job():
    root = Path(__file__).parents[1]
    readme = (root / "README.md").read_text()
    env_example = (root / ".env.example").read_text()
    gitignore = (root / ".gitignore").read_text().splitlines()

    assert "python scripts/kaggle_train.py" in readme
    assert "KAGGLE_USERNAME=" in env_example
    assert ".kaggle/" in gitignore
    assert "outputs/kaggle/" in gitignore
```

- [ ] **Step 2: Run the contract test and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_kaggle_launcher.py::test_readme_documents_one_command_kaggle_job -q
```

Expected: failure because none of the new launcher documentation exists.

- [ ] **Step 3: Update configuration and ignored runtime paths**

Add `KAGGLE_USERNAME=` to `.env.example`. Add these lines to `.gitignore`:

```gitignore
.kaggle/
outputs/kaggle/
```

- [ ] **Step 4: Document the Kaggle API workflow**

Add a README section containing these exact commands:

```bash
python -m pip install "kaggle>=2,<3"
cp .env.example .env
python scripts/kaggle_train.py
python scripts/kaggle_train.py --folds 0
python scripts/kaggle_train.py --no-wait
```

Explain that the user must set `KAGGLE_API_TOKEN` and `KAGGLE_USERNAME`, accept
competition rules, allow a private 941 MB first upload, and have available GPU
quota. Explain T4/internet selection, private resource creation, immutable input
reuse, timestamped `outputs/kaggle/` downloads, status timeout behavior, and
Kaggle session-duration risk for all five folds.

- [ ] **Step 5: Run the documentation contract and focused suites**

Run:

```bash
.venv/bin/python -m pytest tests/test_kaggle_launcher.py tests/test_kaggle_job.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit documentation**

```bash
git add .env.example .gitignore README.md tests/test_kaggle_launcher.py
git commit -m "docs: add one-command Kaggle GPU workflow"
```

### Task 6: Complete Verification

**Files:**
- Verify only; change production files only in response to a reproduced failing test.

- [ ] **Step 1: Compile new Python entry points**

Run:

```bash
.venv/bin/python -m py_compile scripts/kaggle_train.py kaggle_job/run_training.py
```

Expected: exit code 0 with no output.

- [ ] **Step 2: Run focused Kaggle tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_kaggle_launcher.py tests/test_kaggle_job.py -q
```

Expected: all focused tests pass with zero warnings or failures.

- [ ] **Step 3: Run existing Docker tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_docker_config.py tests/test_docker_entrypoint.py -q
```

Expected: existing Docker behavior remains green.

- [ ] **Step 4: Run the complete suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: the complete repository suite passes.

- [ ] **Step 5: Check diffs and secret exclusions**

Run:

```bash
git diff --check
git status --short
git diff -- .env .env.example .gitignore README.md scripts/kaggle_train.py kaggle_job/run_training.py requirements-kaggle.txt tests/test_kaggle_launcher.py tests/test_kaggle_job.py
```

Expected: no whitespace errors, `.env` is unchanged and untracked/ignored, and
only approved launcher, runner, requirements, tests, and documentation changes
are present.

- [ ] **Step 6: Review against the approved design**

Confirm each success criterion in
`docs/superpowers/specs/2026-09-02-kaggle-api-training-launcher-design.md` maps to
a passing focused test or an explicit manual-only live launch step. Do not push
a real kernel during verification.
