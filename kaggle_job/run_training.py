#!/usr/bin/env python3
"""Bootstrap and run the training pipeline inside a Kaggle script kernel."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Callable


# The local launcher replaces this exact assignment in the uploaded script.
JOB_CONFIG_JSON = "{}"


def _find_required_file(root: Path, name: str) -> Path:
    direct = root / name
    if direct.is_file():
        return direct.resolve()
    matches = sorted(path.resolve() for path in root.rglob(name) if path.is_file())
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(matches)}")
    return matches[0]


def _find_dicom_directory(root: Path, split: str) -> Path:
    names = (f"{split}_series", f"{split}_images", split)
    for name in names:
        direct = root / name
        if direct.is_dir():
            return direct.resolve()
    for name in names:
        matches = sorted(path.resolve() for path in root.rglob(name) if path.is_dir())
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(f"multiple candidate {split} DICOM directories found under {root}")
    raise RuntimeError(f"{split} DICOM directory not found under {root}")


def prepare_data_root(competition: Path, scratch: Path) -> Path:
    """Expose competition DICOM roots under canonical writable-layout names."""
    data_root = scratch / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        source = _find_dicom_directory(competition, split)
        destination = data_root / f"{split}_series"
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source:
                raise RuntimeError(f"conflicting canonical DICOM path: {destination}")
            continue
        os.symlink(source, destination, target_is_directory=True)
    return data_root


def build_kaggle_config(
    source: Path,
    project: Path,
    competition: Path,
    scratch: Path,
    artifacts: Path,
    *,
    data_root: Path,
) -> dict:
    """Map the production YAML onto Kaggle input, scratch, and output storage."""
    import yaml

    loaded = yaml.safe_load(source.read_text()) or {}
    if not isinstance(loaded, dict) or not isinstance(loaded.get("paths"), dict):
        raise RuntimeError(f"invalid production config: {source}")
    config = deepcopy(loaded)
    paths = config["paths"]
    paths.update(
        {
            "data_dir": str(data_root),
            "cache_dir": str(scratch / "cache"),
            "train_csv": str(_find_required_file(competition, "train.csv")),
            "test_csv": str(_find_required_file(competition, "test.csv")),
            "train_series_csv": str(_find_required_file(competition, "train_series.csv")),
            "test_series_csv": str(_find_required_file(competition, "test_series.csv")),
            "sample_submission_csv": str(
                _find_required_file(competition, "sample_submission.csv")
            ),
            "labels_csv": str(project / "data" / "llm_labels_v4_blend.csv"),
            "sam_base_checkpoint": str(scratch / "weights" / "pretrained" / "sam_vit_b_01ec64.pth"),
            "external_knee_checkpoint": str(
                scratch / "weights" / "checkpoints" / "knee" / "m_f{fold}.pt"
            ),
            "external_sam_checkpoint": str(
                scratch
                / "weights"
                / "checkpoints"
                / "sam"
                / "submissions_epoch_8_step_11550"
            ),
            "checkpoints_dir": str(artifacts / "checkpoints"),
            "outputs_dir": str(artifacts / "outputs"),
        }
    )
    return config


def safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extract regular ZIP members while rejecting traversal and symlinks."""
    root = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(f"unsafe archive member: {member.filename}") from exc
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"unsafe archive member: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as output:
                while block := source.read(1024 * 1024):
                    output.write(block)


def _require_nonempty(path: Path, description: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{description} was not created: {path}")


def execute_pipeline(
    *,
    python: str,
    project: Path,
    config: Path,
    artifacts: Path,
    folds: str | None,
    run: Callable = subprocess.run,
) -> None:
    """Run adaptive training and prediction with required artifact gates."""
    fold_args = ["--folds", folds] if folds else []
    run(
        [python, "TrainEnsemble.py", "auto", "--config", str(config), *fold_args],
        cwd=project,
        check=True,
    )
    resolved = artifacts / "outputs" / "auto_training.yaml"
    _require_nonempty(resolved, "adaptive configuration")
    run(
        [python, "TrainEnsemble.py", "predict", "--config", str(resolved), *fold_args],
        cwd=project,
        check=True,
    )
    _require_nonempty(artifacts / "outputs" / "submission.csv", "submission")


def _dataset_directory(input_root: Path, slug: str) -> Path:
    name = slug.split("/", 1)[-1]
    path = input_root / name
    if not path.is_dir():
        raise RuntimeError(f"attached Kaggle dataset is missing: {slug}")
    return path


def main() -> int:
    job = json.loads(JOB_CONFIG_JSON)
    required = {"competition", "source_dataset", "weights_dataset", "folds"}
    if not isinstance(job, dict) or not required.issubset(job):
        raise RuntimeError("uploaded Kaggle job configuration is missing required fields")

    input_root = Path("/kaggle/input")
    working = Path("/kaggle/working")
    scratch = Path("/kaggle/temp/rsna-knee-training")
    project = working / "project"
    artifacts = working / "artifacts"
    source_input = _dataset_directory(input_root, str(job["source_dataset"]))
    weights_input = _dataset_directory(input_root, str(job["weights_dataset"]))
    competition = input_root / str(job["competition"])
    if not competition.is_dir():
        raise RuntimeError(f"attached Kaggle competition is missing: {job['competition']}")

    print("Extracting project source and pretrained weights")
    safe_extract_zip(_find_required_file(source_input, "source.zip"), project)
    safe_extract_zip(_find_required_file(weights_input, "weights.zip"), scratch)

    requirements = project / "requirements-kaggle.txt"
    _require_nonempty(requirements, "Kaggle requirements")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-c", "import torch; assert torch.cuda.is_available()"],
        check=True,
    )

    data_root = prepare_data_root(competition, scratch)
    config = build_kaggle_config(
        project / "config" / "training.yaml",
        project,
        competition,
        scratch,
        artifacts,
        data_root=data_root,
    )
    artifacts.mkdir(parents=True, exist_ok=True)
    config_path = project / "kaggle_training.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    execute_pipeline(
        python=sys.executable,
        project=project,
        config=config_path,
        artifacts=artifacts,
        folds=job["folds"],
    )
    print(f"Training and prediction completed: {artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
