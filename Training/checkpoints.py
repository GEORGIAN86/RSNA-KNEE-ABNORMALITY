"""Atomic training checkpoints with strict compatibility metadata."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class CheckpointError(ValueError):
    """Raised when a checkpoint is missing, malformed, or incompatible."""


_ARCHITECTURE_DEFAULTS: dict[str, dict[str, Any]] = {
    "knee": {
        "backbone": None,
        "cond": "token",
        "pool": "xcodex",
        "stem": "native",
        "n_slice": 16,
        "n_meta": 0,
        "img": 336,
        "norm": "none",
        "slices_per_slot": 16,
    },
    "sam": {
        "model_type": "vit_b",
        "feature_dim": 256,
        "input_size": 512,
        "dropout": 0.2,
        "trainable_blocks": 2,
        "slices_per_slot": 16,
    },
    "fusion": {
        "fusion_dim": 256,
        "decoder_layers": 2,
        "attention_heads": 8,
        "feedforward_dim": 1024,
        "dropout": 0.2,
        "dino_backbone": None,
        "dino_slices": 16,
        "dino_trainable_blocks": 4,
        "dino_gradient_checkpointing": False,
        "sam_model_type": "vit_b",
        "sam_input_size": 512,
        "sam_slices": 2,
        "sam_trainable_blocks": 2,
        "sam_gradient_checkpointing": False,
    },
}


def architecture_fingerprint(branch: str, config: dict[str, Any]) -> str:
    """Hash settings that determine model structure and checkpoint semantics.

    Optimizer, scheduler, epoch, batch-size, and execution settings are omitted so
    a compatible checkpoint can be reused after changing the training schedule.
    """

    if branch not in _ARCHITECTURE_DEFAULTS:
        raise CheckpointError(f"unknown checkpoint branch: {branch}")
    values = {
        key: config.get(key, default)
        for key, default in _ARCHITECTURE_DEFAULTS[branch].items()
    }
    payload = {"branch": branch, "architecture": values}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    *,
    metadata: dict[str, Any],
    epoch: int = 0,
    global_step: int = 0,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "metadata": metadata,
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".pth", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _check_metadata(metadata: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, value in expected.items():
        if key not in metadata:
            raise CheckpointError(f"checkpoint metadata is missing {key}")
        if metadata[key] != value:
            raise CheckpointError(
                f"checkpoint {key} mismatch: found {metadata[key]!r}, expected {value!r}"
            )


def _read_checkpoint(path: str | Path, map_location: str | torch.device) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise CheckpointError(f"checkpoint not found: {source}")
    try:
        payload = torch.load(source, map_location=map_location, weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise CheckpointError(f"could not read checkpoint {source}: {exc}") from exc
    required = {"model_state", "metadata", "epoch", "global_step"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise CheckpointError(f"unsupported checkpoint structure: {source}")
    if not isinstance(payload["metadata"], dict):
        raise CheckpointError("checkpoint metadata must be a mapping")
    return payload


def read_checkpoint_metadata(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    return dict(_read_checkpoint(path, map_location)["metadata"])


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    *,
    expected: dict[str, Any] | None = None,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = _read_checkpoint(path, map_location)
    metadata = payload["metadata"]
    if expected:
        _check_metadata(metadata, expected)
    try:
        model.load_state_dict(payload["model_state"], strict=strict)
        if optimizer is not None and payload.get("optimizer_state") is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None and payload.get("scheduler_state") is not None:
            scheduler.load_state_dict(payload["scheduler_state"])
        if scaler is not None and payload.get("scaler_state") is not None:
            scaler.load_state_dict(payload["scaler_state"])
    except (RuntimeError, ValueError, KeyError) as exc:
        raise CheckpointError(f"checkpoint state is incompatible: {exc}") from exc
    return payload


def resolve_resume_checkpoint(
    directory: str | Path,
    fold: int,
    mode: str,
    explicit_path: str | Path | None = None,
) -> Path | None:
    if mode == "fresh":
        return None
    if mode == "auto":
        candidate = Path(directory) / f"fold_{fold}_last.pth"
        return candidate if candidate.is_file() else None
    if mode == "explicit":
        if explicit_path is None:
            raise CheckpointError("explicit resume requires a checkpoint path")
        candidate = Path(explicit_path)
        if not candidate.is_file():
            raise CheckpointError(f"explicit checkpoint not found: {candidate}")
        return candidate
    raise CheckpointError(f"unknown resume mode: {mode}")


__all__ = [
    "CheckpointError",
    "architecture_fingerprint",
    "load_checkpoint",
    "read_checkpoint_metadata",
    "resolve_resume_checkpoint",
    "save_checkpoint",
]
