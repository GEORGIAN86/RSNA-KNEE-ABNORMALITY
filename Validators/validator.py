"""Preflight validation for configuration, data, caches, folds, and checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from Helpers.cache import CacheError, load_cached_study
from Training.constants import CACHE_SCHEMA_VERSION, LABELS
from Validators.test_validator import validate_predictions


class ValidationError(ValueError):
    """Raised when a pipeline stage cannot safely start."""


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self) -> None:
        if self.errors:
            raise ValidationError("validation failed:\n- " + "\n- ".join(self.errors))


def _check_csv(path: Path, columns: list[str], report: ValidationReport) -> pd.DataFrame | None:
    if not path.is_file():
        report.errors.append(f"missing CSV: {path}")
        return None
    try:
        frame = pd.read_csv(path, nrows=100)
    except Exception as exc:
        report.errors.append(f"could not read {path}: {exc}")
        return None
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        report.errors.append(f"{path.name} is missing columns: {missing}")
    return frame


def validate_input_data(config: dict[str, Any], command: str) -> ValidationReport:
    report = ValidationReport()
    paths = config["paths"]
    needs_train = command in {"preprocess", "train", "all"}
    needs_test = command in {"preprocess", "predict", "all"}
    if needs_train:
        _check_csv(Path(paths["train_csv"]), ["StudyInstanceUID", *LABELS], report)
        _check_csv(
            Path(paths["train_series_csv"]),
            ["StudyInstanceUID", "SeriesInstanceUID"],
            report,
        )
        train_root = Path(paths["data_dir"]) / "train_series"
        if not train_root.is_dir():
            report.errors.append(f"missing train DICOM directory: {train_root}")
    if needs_test:
        _check_csv(Path(paths["test_csv"]), ["StudyInstanceUID"], report)
        _check_csv(
            Path(paths["test_series_csv"]),
            ["StudyInstanceUID", "SeriesInstanceUID"],
            report,
        )
        test_root = Path(paths["data_dir"]) / "test_series"
        if not test_root.is_dir():
            report.errors.append(f"missing test DICOM directory: {test_root}")
    if command in {"train", "all"}:
        if config["checkpoint"]["fusion"].get("initialize") == "external":
            for fold in config["data"]["folds"]:
                knee_path = Path(str(paths["external_knee_checkpoint"]).format(fold=int(fold)))
                if not knee_path.is_file():
                    report.errors.append(f"missing external knee checkpoint: {knee_path}")
        if config["checkpoint"]["fusion"].get("initialize") == "external":
            sam_path = Path(paths["external_sam_checkpoint"])
            if not sam_path.is_dir():
                report.errors.append(f"missing extracted SAM checkpoint: {sam_path}")
    labels_path = Path(paths["labels_csv"])
    if labels_path.exists():
        _check_csv(labels_path, ["StudyInstanceUID", *LABELS], report)
    else:
        report.warnings.append(f"weak labels not found; gold-only mode will be used: {labels_path}")
    return report


def validate_folds(folds: dict[str, int], n_folds: int) -> None:
    if not folds:
        raise ValidationError("fold map is empty")
    invalid = {uid: fold for uid, fold in folds.items() if fold < 0 or fold >= n_folds}
    if invalid:
        raise ValidationError(f"fold map has invalid assignments: {invalid}")
    missing = sorted(set(range(n_folds)).difference(folds.values()))
    if missing:
        raise ValidationError(f"fold map has empty folds: {missing}")


def validate_cache_manifest(path: str | Path, expected_fingerprint: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ValidationError(f"cache manifest not found: {source}")
    try:
        manifest = json.loads(source.read_text())
    except (OSError, ValueError) as exc:
        raise ValidationError(f"invalid cache manifest {source}: {exc}") from exc
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValidationError("cache manifest schema mismatch")
    if manifest.get("fingerprint") != expected_fingerprint:
        raise ValidationError("cache manifest fingerprint mismatch")
    return manifest


def validate_cached_studies(
    cache_dir: str | Path,
    uids: list[str],
    expected_fingerprint: str,
) -> ValidationReport:
    """Validate cached study contents before constructing a data loader."""

    report = ValidationReport()
    root = Path(cache_dir)
    for raw_uid in uids:
        uid = str(raw_uid)
        path = root / f"{uid}.npz"
        try:
            cached = load_cached_study(path, expected_fingerprint=expected_fingerprint)
            if cached.uid != uid:
                raise CacheError(
                    f"cache UID mismatch: expected {uid}, found {cached.uid}"
                )
            if not cached.slot_mask.any():
                raise CacheError(f"study {uid} has no usable MRI slots in the cache")
        except CacheError as exc:
            report.errors.append(str(exc))
    return report


def validate_checkpoint_coverage(directory: str | Path, folds: list[int]) -> dict[int, Path]:
    root = Path(directory)
    result = {fold: root / f"fold_{fold}_best.pth" for fold in folds}
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise ValidationError("missing best checkpoints: " + ", ".join(missing))
    return result


def validate_prediction_artifacts(
    outputs_dir: str | Path,
    *,
    train_uids: list[str],
    test_uids: list[str],
    labels: list[str] | tuple[str, ...],
) -> ValidationReport:
    """Validate every prediction artifact that currently exists."""

    report = ValidationReport()
    root = Path(outputs_dir)
    specifications = {
        "oof_fusion.csv": train_uids,
        "oof_knee.csv": train_uids,
        "oof_sam.csv": train_uids,
        "oof_blended.csv": train_uids,
        "test_knee.csv": test_uids,
        "test_sam.csv": test_uids,
        "submission.csv": test_uids,
        "test_fusion.csv": test_uids,
    }
    required = ["StudyInstanceUID", *labels]
    for filename, expected_uids in specifications.items():
        path = root / filename
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})
            missing = [column for column in required if column not in frame.columns]
            if missing:
                raise ValueError(f"missing columns: {missing}")
            validate_predictions(frame.loc[:, required], expected_uids, labels)
        except (OSError, ValueError) as exc:
            report.errors.append(f"invalid {filename}: {exc}")
    return report


__all__ = [
    "ValidationError",
    "ValidationReport",
    "validate_cache_manifest",
    "validate_cached_studies",
    "validate_checkpoint_coverage",
    "validate_folds",
    "validate_input_data",
    "validate_prediction_artifacts",
]
