"""Strict readers for externally supplied knee and SAM checkpoints."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Any

import torch


class ExternalCheckpointError(ValueError):
    """Raised when an external checkpoint is missing or incompatible."""


def resolve_external_knee_checkpoint(path_template: str | Path, fold: int) -> Path:
    source = Path(str(path_template).format(fold=int(fold)))
    if not source.is_file():
        raise ExternalCheckpointError(f"external knee checkpoint not found for fold {fold}: {source}")
    return source


def load_external_knee_state(
    path: str | Path,
    *,
    expected_fold: int,
    labels: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ExternalCheckpointError(f"could not read external knee checkpoint {source}: {exc}") from exc
    if not isinstance(payload, dict) or not {"state_dict", "cfg", "fold"}.issubset(payload):
        raise ExternalCheckpointError("external knee checkpoint must contain state_dict, cfg, and fold")
    if int(payload["fold"]) != int(expected_fold):
        raise ExternalCheckpointError(
            f"external knee checkpoint fold mismatch: found {payload['fold']}, expected {expected_fold}"
        )
    config = payload["cfg"]
    if not isinstance(config, dict):
        raise ExternalCheckpointError("external knee checkpoint cfg must be a mapping")
    if list(config.get("labels", [])) != list(labels):
        raise ExternalCheckpointError("external knee checkpoint labels do not match configured labels")
    if not isinstance(payload["state_dict"], dict):
        raise ExternalCheckpointError("external knee checkpoint state_dict must be a mapping")
    return payload


def load_extracted_torch_checkpoint(directory: str | Path) -> dict[str, Any]:
    """Load a PyTorch ZIP serialization that was extracted to a directory."""

    source = Path(directory)
    required = {"data.pkl", "version", "byteorder", "data"}
    if not source.is_dir() or not required.issubset({entry.name for entry in source.iterdir()}):
        raise ExternalCheckpointError(f"invalid extracted PyTorch checkpoint directory: {source}")
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            root = source.name
            with zipfile.ZipFile(handle.name, "w", compression=zipfile.ZIP_STORED) as archive:
                for path in sorted(source.rglob("*")):
                    if path.is_file():
                        archive.write(path, Path(root) / path.relative_to(source))
            payload = torch.load(handle.name, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        raise ExternalCheckpointError(f"could not read extracted PyTorch checkpoint {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExternalCheckpointError("extracted PyTorch checkpoint payload must be a mapping")
    return payload


def translate_external_sam_state(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    translated: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            raise ExternalCheckpointError(f"SAM model state value is not a tensor: {key}")
        if key == "m":
            destination = "pixel_mean"
        elif key == "s":
            destination = "pixel_std"
        elif key.startswith("e."):
            destination = "image_encoder." + key[2:]
        elif key.startswith("h."):
            destination = "head." + key[2:]
        else:
            raise ExternalCheckpointError(f"unknown SAM model key: {key}")
        translated[destination] = value
    return translated


__all__ = [
    "ExternalCheckpointError",
    "load_external_knee_state",
    "load_extracted_torch_checkpoint",
    "resolve_external_knee_checkpoint",
    "translate_external_sam_state",
]
