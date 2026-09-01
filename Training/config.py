"""YAML configuration loading with deterministic project-relative paths."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a pipeline configuration is structurally invalid."""


def default_config() -> dict[str, Any]:
    return {
        "paths": {
            "data_dir": "data",
            "cache_dir": "data/cache",
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "train_series_csv": "data/train_series.csv",
            "test_series_csv": "data/test_series.csv",
            "sample_submission_csv": "data/sample_submission.csv",
            "labels_csv": "data/labels.csv",
            "sam_base_checkpoint": "weights/pretrained/sam_vit_b_01ec64.pth",
            "external_knee_checkpoint": "weights/checkpoints/knee/m_f{fold}.pt",
            "external_sam_checkpoint": "weights/checkpoints/sam/submissions_epoch_8_step_11550",
            "checkpoints_dir": "weights/checkpoints",
            "outputs_dir": "outputs",
        },
        "data": {
            "n_folds": 5,
            "folds": [0, 1, 2, 3, 4],
            "image_size": 336,
            "crop_mm": 130.0,
            "max_slices_per_slot": 16,
            "slice_band": [0.12, 0.88],
            "gold_weight": 8.0,
            "silent_value": 0.25,
            "silent_weight": 0.05,
            "workers": 4,
            "cache_schema_version": 1,
        },
        "knee": {
            "enabled": True,
            "backbone": "vit_small_patch16_dinov3.lvd1689m",
            "cond": "token",
            "pool": "xcodex",
            "stem": "native",
            "n_slice": 16,
            "n_meta": 0,
            "img": 336,
            "norm": "none",
            "slices_per_slot": 16,
            "trainable_blocks": 4,
            "gradient_checkpointing": False,
            "epochs": 12,
            "batch_size": 4,
            "lr": 1.0e-4,
            "weight_decay": 1.0e-4,
            "gradient_accumulation": 1,
            "gradient_clip": 1.0,
            "scheduler": "cosine",
        },
        "sam": {
            "enabled": True,
            "model_type": "vit_b",
            "feature_dim": 256,
            "input_size": 512,
            "encode_chunk": 6,
            "trainable_blocks": 2,
            "gradient_checkpointing": False,
            "slices_per_slot": 2,
            "epochs": 8,
            "batch_size": 1,
            "lr": 2.0e-5,
            "weight_decay": 1.0e-4,
            "gradient_accumulation": 1,
            "gradient_clip": 1.0,
            "scheduler": "cosine",
        },
        "fusion": {
            "enabled": True,
            "fusion_dim": 256,
            "decoder_layers": 2,
            "attention_heads": 8,
            "feedforward_dim": 1024,
            "dropout": 0.2,
            "epochs": 8,
            "batch_size": 1,
            "lr": 1.0e-5,
            "weight_decay": 1.0e-4,
            "gradient_accumulation": 8,
            "gradient_clip": 1.0,
            "scheduler": "cosine",
        },
        "checkpoint": {
            "strict": True,
            "knee": {"resume": "auto", "path": None, "initialize": "external"},
            "sam": {"resume": "auto", "path": None, "initialize": "external"},
            "fusion": {"resume": "auto", "path": None, "initialize": "external"},
        },
        "ensemble": {"kind": "rank", "sam_weight": 0.2},
        "runtime": {
            "device": "auto",
            "seed": 2026,
            "deterministic": True,
            "precision": "amp",
            "pin_memory": True,
        },
    }


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _positive_int(cfg: dict[str, Any], section: str, key: str) -> None:
    value = cfg[section][key]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{section}.{key} must be a positive integer")


def validate_config(cfg: dict[str, Any]) -> None:
    _positive_int(cfg, "data", "n_folds")
    if cfg["data"]["n_folds"] < 2:
        raise ConfigError("data.n_folds must be at least 2")
    for section, key in (
        ("data", "image_size"),
        ("data", "max_slices_per_slot"),
        ("knee", "slices_per_slot"),
        ("sam", "slices_per_slot"),
        ("knee", "epochs"),
        ("sam", "epochs"),
        ("fusion", "fusion_dim"),
        ("fusion", "decoder_layers"),
        ("fusion", "attention_heads"),
        ("fusion", "feedforward_dim"),
        ("fusion", "epochs"),
        ("fusion", "batch_size"),
    ):
        _positive_int(cfg, section, key)
    if cfg["knee"]["slices_per_slot"] > cfg["data"]["max_slices_per_slot"]:
        raise ConfigError("knee.slices_per_slot exceeds the cache capacity")
    if cfg["knee"]["slices_per_slot"] != int(cfg["knee"].get("n_slice", 16)):
        raise ConfigError("knee.slices_per_slot must match knee.n_slice")
    if cfg["sam"]["slices_per_slot"] > cfg["data"]["max_slices_per_slot"]:
        raise ConfigError("sam.slices_per_slot exceeds the cache capacity")
    folds = cfg["data"]["folds"]
    if not isinstance(folds, list) or not folds:
        raise ConfigError("data.folds must be a non-empty list")
    invalid_folds = [fold for fold in folds if not isinstance(fold, int) or fold < 0 or fold >= cfg["data"]["n_folds"]]
    if invalid_folds:
        raise ConfigError(f"data.folds contains invalid fold indices: {invalid_folds}")
    if cfg["knee"]["pool"] not in {"xcodex", "cls_mean", "cls_mean_focal"}:
        raise ConfigError("knee.pool must be xcodex, cls_mean, or cls_mean_focal")
    blocks = cfg["sam"]["trainable_blocks"]
    if not isinstance(blocks, int) or blocks < -1:
        raise ConfigError("sam.trainable_blocks must be -1, 0, or a positive integer")
    knee_blocks = cfg["knee"]["trainable_blocks"]
    if not isinstance(knee_blocks, int) or knee_blocks < -1 or knee_blocks > 12:
        raise ConfigError("knee.trainable_blocks must be -1 or between 0 and 12")
    if blocks > 12:
        raise ConfigError("sam.trainable_blocks cannot exceed 12 for ViT-B")
    if cfg["fusion"]["fusion_dim"] % cfg["fusion"]["attention_heads"]:
        raise ConfigError("fusion.fusion_dim must be divisible by fusion.attention_heads")
    kind = cfg["ensemble"]["kind"]
    if kind not in {"rank", "probability"}:
        raise ConfigError("ensemble.kind must be rank or probability")
    weight = cfg["ensemble"]["sam_weight"]
    if not isinstance(weight, (int, float)) or not 0 <= float(weight) <= 1:
        raise ConfigError("ensemble.sam_weight must be between 0 and 1")
    for branch in ("knee", "sam", "fusion"):
        initialize = cfg["checkpoint"][branch].get("initialize", "none")
        if initialize not in {"none", "external"}:
            raise ConfigError(f"checkpoint.{branch}.initialize must be none or external")
        resume = cfg["checkpoint"][branch]["resume"]
        if resume not in {"fresh", "auto", "explicit"}:
            raise ConfigError(f"checkpoint.{branch}.resume must be fresh, auto, or explicit")
        if resume == "explicit" and not cfg["checkpoint"][branch].get("path"):
            raise ConfigError(f"checkpoint.{branch}.path is required for explicit resume")
    if cfg["runtime"]["device"] not in {"auto", "cpu", "cuda", "mps"}:
        raise ConfigError("runtime.device must be auto, cpu, cuda, or mps")
    if cfg["runtime"]["precision"] not in {"amp", "float32"}:
        raise ConfigError("runtime.precision must be amp or float32")


def load_config(path: str | Path, project_root: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"configuration file not found: {config_path}")
    root = Path(project_root).expanduser().resolve() if project_root else config_path.parent.parent
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError("the YAML root must be a mapping")
    cfg = deep_merge(default_config(), raw)
    cfg["project_root"] = root
    cfg["config_path"] = config_path
    for key, value in cfg["paths"].items():
        candidate = Path(value).expanduser()
        cfg["paths"][key] = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    validate_config(cfg)
    return cfg


__all__ = ["ConfigError", "deep_merge", "default_config", "load_config", "validate_config"]
