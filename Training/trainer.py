"""Training and evaluation primitives shared by both model branches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from .checkpoints import architecture_fingerprint, load_checkpoint, save_checkpoint


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def macro_auc(
    targets: np.ndarray,
    predictions: np.ndarray,
    labels: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    if targets.shape != predictions.shape or targets.ndim != 2:
        raise ValueError("targets and predictions must have matching two-dimensional shapes")
    names = list(labels) if labels is not None else [str(index) for index in range(targets.shape[1])]
    if len(names) != targets.shape[1]:
        raise ValueError("label count does not match prediction width")
    per_label: dict[str, float] = {}
    for index, name in enumerate(names):
        column = (targets[:, index] > 0.5).astype(np.int8)
        per_label[name] = (
            float(roc_auc_score(column, predictions[:, index]))
            if np.unique(column).size > 1
            else float("nan")
        )
    finite = [value for value in per_label.values() if np.isfinite(value)]
    return {
        "macro_auc": float(np.mean(finite)) if finite else float("nan"),
        "per_label_auc": per_label,
    }


def weighted_bce(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape or targets.shape != weights.shape:
        raise ValueError("logits, targets, and weights must have identical shapes")
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    denominator = weights.sum()
    if denominator.item() <= 0:
        raise ValueError("at least one target weight must be positive")
    return (losses * weights).sum() / denominator


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def forward_batch(model: nn.Module, branch: str, batch: dict[str, Any]) -> torch.Tensor:
    if branch == "knee":
        return model(batch["images"], batch["slot_mask"])
    if branch == "sam":
        return model(batch["images"], batch["slot_mask"], batch["slice_mask"])
    if branch == "fusion":
        return model(
            batch["images"],
            batch["slot_mask"],
            batch["slice_mask"],
            batch["sam_images"],
            batch["sam_slice_mask"],
        )
    raise ValueError(f"unknown branch: {branch}")


def _amp_enabled(device: torch.device, precision: str) -> bool:
    return device.type == "cuda" and precision == "amp"


def train_one_epoch(
    model: nn.Module,
    branch: str,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    gradient_accumulation: int,
    gradient_clip: float,
    precision: str,
) -> tuple[float, int]:
    if len(loader) == 0:
        raise ValueError("training loader is empty")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    updates = 0
    for batch_index, raw_batch in enumerate(loader):
        batch = _to_device(raw_batch, device)
        enabled = _amp_enabled(device, precision)
        with torch.amp.autocast(device_type=device.type, enabled=enabled):
            loss = weighted_bce(
                forward_batch(model, branch, batch),
                batch["targets"],
                batch["weights"],
            )
            scaled_loss = loss / gradient_accumulation
        scaler.scale(scaled_loss).backward()
        should_step = (batch_index + 1) % gradient_accumulation == 0 or batch_index + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        total_loss += float(loss.detach().cpu())
    return total_loss / len(loader), updates


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    branch: str,
    loader: DataLoader,
    device: torch.device,
    *,
    precision: str = "amp",
) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    if len(loader) == 0:
        raise ValueError("prediction loader is empty")
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    uids: list[str] = []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        enabled = _amp_enabled(device, precision)
        with torch.amp.autocast(device_type=device.type, enabled=enabled):
            logits = forward_batch(model, branch, batch)
        predictions.append(torch.sigmoid(logits.float()).cpu().numpy())
        uids.extend(str(uid) for uid in raw_batch["uid"])
        if "targets" in batch:
            targets.append(batch["targets"].float().cpu().numpy())
    target_array = np.concatenate(targets) if targets else None
    return np.concatenate(predictions), uids, target_array


@dataclass(frozen=True)
class TrainResult:
    predictions: np.ndarray
    targets: np.ndarray
    uids: list[str]
    metrics: dict[str, object]
    history: list[dict[str, object]]
    best_path: Path
    last_path: Path


@dataclass(frozen=True)
class GlobalTrainResult:
    history: list[dict[str, object]]
    final_path: Path
    last_path: Path


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler | None:
    kind = str(config.get("scheduler", "cosine"))
    if kind == "none":
        return None
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(config["epochs"])),
        )
    raise ValueError(f"unsupported scheduler: {kind}")


def train_fold(
    model: nn.Module,
    branch: str,
    fold: int,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    checkpoint_dir: str | Path,
    config: dict[str, Any],
    device: torch.device,
    labels: list[str] | tuple[str, ...],
    cache_fingerprint: str,
    fold_fingerprint: str,
    *,
    resume_path: str | Path | None = None,
    strict_checkpoint: bool = True,
    precision: str = "amp",
) -> TrainResult:
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    best_path = directory / f"fold_{fold}_best.pth"
    last_path = directory / f"fold_{fold}_last.pth"
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError(f"{branch} model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    scheduler = _make_scheduler(optimizer, config)
    enabled = _amp_enabled(device, precision)
    scaler = torch.amp.GradScaler("cuda", enabled=enabled)
    model.to(device)
    metadata_base: dict[str, Any] = {
        "branch": branch,
        "fold": int(fold),
        "labels": list(labels),
        "cache_fingerprint": cache_fingerprint,
        "fold_fingerprint": fold_fingerprint,
        "architecture_fingerprint": architecture_fingerprint(branch, config),
        "model_config": dict(config),
    }
    expected = {
        key: metadata_base[key]
        for key in (
            "branch",
            "fold",
            "labels",
            "cache_fingerprint",
            "fold_fingerprint",
            "architecture_fingerprint",
        )
    }
    start_epoch = 0
    global_step = 0
    best_metric = float("-inf")
    history: list[dict[str, object]] = []
    if resume_path is not None:
        restored = load_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler,
            scaler,
            expected=expected,
            strict=strict_checkpoint,
            map_location=device,
        )
        start_epoch = int(restored["epoch"]) + 1
        global_step = int(restored["global_step"])
        restored_metadata = restored["metadata"]
        best_metric = float(restored_metadata.get("best_macro_auc", best_metric))
        history = list(restored_metadata.get("history", []))

    epochs = int(config["epochs"])
    if start_epoch >= epochs:
        if not best_path.is_file():
            raise ValueError("resume checkpoint has completed training but no best checkpoint exists")
    for epoch in range(start_epoch, epochs):
        loss, updates = train_one_epoch(
            model,
            branch,
            train_loader,
            optimizer,
            scaler,
            device,
            gradient_accumulation=int(config.get("gradient_accumulation", 1)),
            gradient_clip=float(config.get("gradient_clip", 0.0)),
            precision=precision,
        )
        global_step += updates
        if scheduler is not None:
            scheduler.step()
        predictions, uids, targets = predict_loader(
            model,
            branch,
            validation_loader,
            device,
            precision=precision,
        )
        if targets is None:
            raise ValueError("validation loader does not provide targets")
        metrics = macro_auc(targets, predictions, labels)
        current = float(metrics["macro_auc"])
        epoch_record = {"epoch": epoch, "loss": loss, **metrics}
        history.append(epoch_record)
        improved = not best_path.is_file() or (np.isfinite(current) and current > best_metric)
        if improved:
            best_metric = current
        metadata = {
            **metadata_base,
            "best_macro_auc": best_metric,
            "validation_metrics": metrics,
            "history": history,
        }
        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            scaler,
            metadata=metadata,
            epoch=epoch,
            global_step=global_step,
        )
        if improved:
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                scaler,
                metadata=metadata,
                epoch=epoch,
                global_step=global_step,
            )
        history_path = directory / f"fold_{fold}_history.json"
        temporary = history_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(_json_compatible(history), indent=2, allow_nan=False))
        temporary.replace(history_path)

    load_checkpoint(
        best_path,
        model,
        expected=expected,
        strict=strict_checkpoint,
        map_location=device,
    )
    predictions, uids, targets = predict_loader(
        model,
        branch,
        validation_loader,
        device,
        precision=precision,
    )
    if targets is None:
        raise ValueError("validation loader does not provide targets")
    final_metrics = macro_auc(targets, predictions, labels)
    return TrainResult(
        predictions,
        targets,
        uids,
        final_metrics,
        history,
        best_path,
        last_path,
    )


def train_global(
    model: nn.Module,
    branch: str,
    train_loader: DataLoader,
    checkpoint_dir: str | Path,
    config: dict[str, Any],
    device: torch.device,
    labels: list[str] | tuple[str, ...],
    cache_fingerprint: str,
    *,
    resume_path: str | Path | None = None,
    strict_checkpoint: bool = True,
    precision: str = "amp",
) -> GlobalTrainResult:
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    last_path = directory / "global_last.pth"
    final_path = directory / "global_final.pth"
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError(f"{branch} model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    scheduler = _make_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=_amp_enabled(device, precision))
    model.to(device)
    metadata_base: dict[str, Any] = {
        "branch": branch,
        "training_mode": "global",
        "labels": list(labels),
        "cache_fingerprint": cache_fingerprint,
        "architecture_fingerprint": architecture_fingerprint(branch, config),
        "model_config": dict(config),
    }
    expected = {
        key: metadata_base[key]
        for key in (
            "branch",
            "training_mode",
            "labels",
            "cache_fingerprint",
            "architecture_fingerprint",
        )
    }
    start_epoch = 0
    global_step = 0
    history: list[dict[str, object]] = []
    if resume_path is not None:
        restored = load_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler,
            scaler,
            expected=expected,
            strict=strict_checkpoint,
            map_location=device,
        )
        start_epoch = int(restored["epoch"]) + 1
        global_step = int(restored["global_step"])
        history = list(restored["metadata"].get("history", []))

    epochs = int(config["epochs"])
    for epoch in range(start_epoch, epochs):
        loss, updates = train_one_epoch(
            model,
            branch,
            train_loader,
            optimizer,
            scaler,
            device,
            gradient_accumulation=int(config.get("gradient_accumulation", 1)),
            gradient_clip=float(config.get("gradient_clip", 0.0)),
            precision=precision,
        )
        global_step += updates
        if scheduler is not None:
            scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "loss": loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "global_step": global_step,
            }
        )
        metadata = {**metadata_base, "history": history}
        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            scaler,
            metadata=metadata,
            epoch=epoch,
            global_step=global_step,
        )
        history_path = directory / "global_history.json"
        temporary = history_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(_json_compatible(history), indent=2, allow_nan=False))
        temporary.replace(history_path)

    if not last_path.is_file():
        raise ValueError("global training produced no last checkpoint")
    restored = load_checkpoint(
        last_path,
        model,
        expected=expected,
        strict=strict_checkpoint,
        map_location=device,
    )
    save_checkpoint(
        final_path,
        model,
        None,
        None,
        None,
        metadata=restored["metadata"],
        epoch=int(restored["epoch"]),
        global_step=int(restored["global_step"]),
    )
    return GlobalTrainResult(history, final_path, last_path)


__all__ = [
    "GlobalTrainResult",
    "TrainResult",
    "forward_batch",
    "macro_auc",
    "predict_loader",
    "train_fold",
    "train_global",
    "train_one_epoch",
    "weighted_bce",
]
