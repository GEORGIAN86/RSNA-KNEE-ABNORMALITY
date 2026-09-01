"""Command-line orchestration for the dual-model knee MRI pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from Helpers.cache import build_cache_split, cache_fingerprint
from Models.DINOv3Knee import build_dinov3_knee_model
from Models.FusionModel import build_fusion_model
from Models.SAMClassifier import build_sam_model, parameter_counts
from Training.checkpoints import (
    CheckpointError,
    architecture_fingerprint,
    load_checkpoint,
    resolve_resume_checkpoint,
)
from Training.config import ConfigError, load_config
from Training.constants import LABELS
from Training.dataset import (
    StudyDataset,
    StudyRecord,
    build_fold_map,
    build_targets,
    fold_fingerprint,
)
from Training.ensemble import average_fold_predictions, blend_predictions, prediction_frame
from Training.external_checkpoints import (
    load_external_knee_state,
    load_extracted_torch_checkpoint,
    resolve_external_knee_checkpoint,
    translate_external_sam_state,
)
from Training.resource_planner import (
    Candidate,
    apply_candidate,
    estimate_candidate_memory,
    generate_balanced_candidates,
    profile_data,
    profile_hardware,
    select_candidate,
    write_planner_outputs,
)
from Training.trainer import (
    TrainResult,
    forward_batch,
    macro_auc,
    predict_loader,
    train_fold,
    train_global,
    weighted_bce,
)
from Validators.test_validator import validate_submission
from Validators.validator import (
    ValidationError,
    validate_cache_manifest,
    validate_cached_studies,
    validate_checkpoint_coverage,
    validate_folds,
    validate_input_data,
    validate_prediction_artifacts,
)


LOGGER = logging.getLogger("knee_pipeline")
ModelFactory = Callable[[dict[str, Any], int], torch.nn.Module]


@dataclass(frozen=True)
class TrainArtifacts:
    checkpoint_folds: dict[str, dict[int, Path]]
    oof_paths: dict[str, Path]
    metrics_path: Path


@dataclass(frozen=True)
class PredictArtifacts:
    fusion_path: Path
    submission: Path


@dataclass(frozen=True)
class PipelineResult:
    submission: Path
    fusion_checkpoints: dict[int, Path]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def _cache_settings(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    return {
        "image_size": int(data["image_size"]),
        "crop_mm": float(data["crop_mm"]),
        "max_slices_per_slot": int(data["max_slices_per_slot"]),
        "slice_band": list(data["slice_band"]),
    }


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValidationError("CUDA was requested but is not available")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ValidationError("MPS was requested but is not available")
    return device


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False


def _production_factories(config: dict[str, Any]) -> dict[str, ModelFactory]:
    sam_checkpoint = config["paths"]["sam_base_checkpoint"]

    def knee_factory(branch_config: dict[str, Any], n_targets: int) -> torch.nn.Module:
        return build_dinov3_knee_model(branch_config, n_targets=n_targets)

    def sam_factory(branch_config: dict[str, Any], n_targets: int) -> torch.nn.Module:
        source = sam_checkpoint if Path(sam_checkpoint).is_file() else None
        return build_sam_model(branch_config, source, n_targets=n_targets)

    def fusion_factory(branch_config: dict[str, Any], n_targets: int) -> torch.nn.Module:
        return build_fusion_model(config, n_targets=n_targets)

    return {"knee": knee_factory, "sam": sam_factory, "fusion": fusion_factory}


def _factories(
    config: dict[str, Any],
    overrides: dict[str, ModelFactory] | None,
) -> dict[str, ModelFactory]:
    result = _production_factories(config)
    if overrides:
        result.update(overrides)
    return result


def _make_loader(
    records: list[StudyRecord],
    config: dict[str, Any],
    fingerprint: str,
    branch: str,
    *,
    training: bool,
) -> DataLoader:
    branch_config = config[branch]
    dataset = StudyDataset(
        records,
        config["paths"]["cache_dir"] / ("train" if records and records[0].targets is not None else "test"),
        fingerprint,
        branch=branch,
        training=training,
        knee_slices=int(config["knee"]["slices_per_slot"]),
        sam_slices=int(config["sam"]["slices_per_slot"]),
    )
    workers = int(config["data"].get("workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(branch_config["batch_size"]),
        shuffle=training,
        num_workers=workers,
        pin_memory=bool(config["runtime"].get("pin_memory", True)),
        persistent_workers=workers > 0,
    )


def run_preprocess(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    validate_input_data(config, "preprocess").raise_if_errors()
    settings = _cache_settings(config)
    paths = config["paths"]
    manifests: dict[str, dict[str, Any]] = {}
    for split in ("train", "test"):
        LOGGER.info("building %s cache", split)
        manifests[split] = build_cache_split(
            paths[f"{split}_series_csv"],
            paths["data_dir"] / f"{split}_series",
            paths["cache_dir"] / split,
            settings,
            split=split,
        )
        LOGGER.info(
            "%s cache: %s built, %s reused",
            split,
            manifests[split]["built"],
            manifests[split]["reused"],
        )
    return manifests


def run_validate(
    config: dict[str, Any],
    *,
    model_factories: dict[str, ModelFactory] | None = None,
) -> None:
    report = validate_input_data(config, "all")
    report.raise_if_errors()
    fingerprint = cache_fingerprint(_cache_settings(config))
    for split in ("train", "test"):
        manifest = config["paths"]["cache_dir"] / split / "manifest.json"
        if manifest.exists():
            validate_cache_manifest(manifest, fingerprint)
        else:
            LOGGER.warning("cache not built yet: %s", manifest)
    train_frame = pd.read_csv(
        config["paths"]["train_csv"],
        dtype={"StudyInstanceUID": str},
    )
    test_frame = pd.read_csv(
        config["paths"]["test_csv"],
        dtype={"StudyInstanceUID": str},
    )
    train_uids = train_frame["StudyInstanceUID"].astype(str).tolist()
    test_uids = test_frame["StudyInstanceUID"].astype(str).tolist()
    for split, uids in (("train", train_uids), ("test", test_uids)):
        cache_report = validate_cached_studies(
            Path(config["paths"]["cache_dir"]) / split,
            uids,
            fingerprint,
        )
        report.errors.extend(cache_report.errors)
        report.warnings.extend(cache_report.warnings)
    fold_hash: str | None = None
    folds_path = Path(config["paths"]["outputs_dir"]) / "folds.json"
    if folds_path.is_file():
        try:
            folds_payload = json.loads(folds_path.read_text())
            folds = {str(uid): int(fold) for uid, fold in folds_payload["folds"].items()}
            fold_hash = str(folds_payload["fingerprint"])
            if fold_hash != fold_fingerprint(folds):
                raise ValueError("fingerprint mismatch")
            requested_folds = set(int(fold) for fold in config["data"]["folds"])
            train_uids = [uid for uid in train_uids if folds.get(uid) in requested_folds]
        except (KeyError, OSError, TypeError, ValueError) as exc:
            report.errors.append(f"invalid folds.json: {exc}")
    artifact_report = validate_prediction_artifacts(
        config["paths"]["outputs_dir"],
        train_uids=train_uids,
        test_uids=test_uids,
        labels=LABELS,
    )
    report.errors.extend(artifact_report.errors)
    report.warnings.extend(artifact_report.warnings)
    factories = _factories(config, model_factories)
    for branch in ("fusion",):
        branch_dir = Path(config["paths"]["checkpoints_dir"]) / branch
        if not any(branch_dir.glob("*.pth")):
            continue
        try:
            expected_common = {
                "branch": branch,
                "labels": list(LABELS),
                "cache_fingerprint": fingerprint,
                "architecture_fingerprint": architecture_fingerprint(
                    branch, _fusion_training_config(config)
                ),
            }
            checkpoint_paths = validate_checkpoint_coverage(
                branch_dir,
                [int(fold) for fold in config["data"]["folds"]],
            )
            if fold_hash is None:
                raise ValidationError("checkpoint validation requires outputs/folds.json")
            expected_common["fold_fingerprint"] = fold_hash
            for fold, checkpoint_path in checkpoint_paths.items():
                branch_config = _fusion_training_config(config)
                model = factories[branch](branch_config, len(LABELS))
                load_checkpoint(
                    checkpoint_path,
                    model,
                    expected={**expected_common, "fold": fold},
                    strict=bool(config["checkpoint"].get("strict", True)),
                    map_location="cpu",
                )
        except (CheckpointError, ValidationError) as exc:
            report.errors.append(f"invalid {branch} checkpoints: {exc}")
    report.raise_if_errors()
    LOGGER.info("input validation passed")


def _load_training_records(
    config: dict[str, Any],
) -> tuple[list[StudyRecord], dict[str, int], str, str]:
    paths = config["paths"]
    train = pd.read_csv(paths["train_csv"], dtype={"StudyInstanceUID": str})
    labels_path = Path(paths["labels_csv"])
    weak = pd.read_csv(labels_path, dtype={"StudyInstanceUID": str}) if labels_path.is_file() else None
    target_table = build_targets(
        train,
        weak,
        gold_weight=float(config["data"]["gold_weight"]),
        silent_value=float(config["data"]["silent_value"]),
        silent_weight=float(config["data"]["silent_weight"]),
    )
    fingerprint = cache_fingerprint(_cache_settings(config))
    validate_cache_manifest(paths["cache_dir"] / "train" / "manifest.json", fingerprint)
    validate_cached_studies(
        paths["cache_dir"] / "train",
        target_table.uids,
        fingerprint,
    ).raise_if_errors()
    folds = build_fold_map(train, target_table.uids, int(config["data"]["n_folds"]))
    validate_folds(folds, int(config["data"]["n_folds"]))
    fold_hash = fold_fingerprint(folds)
    records = [
        StudyRecord(uid, target_table.targets[index], target_table.weights[index], folds[uid])
        for index, uid in enumerate(target_table.uids)
    ]
    return records, folds, fingerprint, fold_hash


def _resume_path(config: dict[str, Any], branch: str, fold: int, directory: Path) -> Path | None:
    resume = config["checkpoint"][branch]
    explicit = resume.get("path")
    if explicit is not None:
        explicit_path = Path(explicit).expanduser()
        explicit = explicit_path if explicit_path.is_absolute() else config["project_root"] / explicit_path
    return resolve_resume_checkpoint(directory, fold, str(resume["resume"]), explicit)


def _global_resume_path(config: dict[str, Any], branch: str, directory: Path) -> Path | None:
    resume = config["checkpoint"][branch]
    mode = str(resume["resume"])
    if mode == "fresh":
        return None
    if mode == "auto":
        candidate = directory / "global_last.pth"
        return candidate if candidate.is_file() else None
    explicit = resume.get("path")
    if explicit is None:
        raise CheckpointError("explicit global resume requires a checkpoint path")
    candidate = Path(explicit).expanduser()
    if not candidate.is_absolute():
        candidate = config["project_root"] / candidate
    if not candidate.is_file():
        raise CheckpointError(f"explicit checkpoint not found: {candidate}")
    return candidate


def _fusion_training_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        **config["fusion"],
        "dino_backbone": config["knee"]["backbone"],
        "dino_slices": int(config["knee"]["slices_per_slot"]),
        "dino_trainable_blocks": int(config["knee"]["trainable_blocks"]),
        "dino_gradient_checkpointing": bool(config["knee"].get("gradient_checkpointing", False)),
        "sam_model_type": config["sam"]["model_type"],
        "sam_input_size": int(config["sam"]["input_size"]),
        "sam_slices": int(config["sam"]["slices_per_slot"]),
        "sam_trainable_blocks": int(config["sam"]["trainable_blocks"]),
        "sam_gradient_checkpointing": bool(config["sam"].get("gradient_checkpointing", False)),
    }


def _initialize_external_fusion(model: torch.nn.Module, config: dict[str, Any], fold: int) -> None:
    knee_path = resolve_external_knee_checkpoint(config["paths"]["external_knee_checkpoint"], fold)
    knee_payload = load_external_knee_state(knee_path, expected_fold=fold, labels=LABELS)
    legacy_knee = build_dinov3_knee_model(knee_payload["cfg"], n_targets=len(LABELS))
    legacy_knee.load_state_dict(knee_payload["state_dict"], strict=True)
    model.dino.enc.load_state_dict(legacy_knee.enc.state_dict(), strict=True)

    sam_payload = load_extracted_torch_checkpoint(config["paths"]["external_sam_checkpoint"])
    sam_state = sam_payload.get("m")
    if not isinstance(sam_state, dict):
        raise CheckpointError("external SAM checkpoint is missing model state 'm'")
    legacy_sam = build_sam_model(config["sam"], None, n_targets=len(LABELS))
    legacy_sam.load_state_dict(translate_external_sam_state(sam_state), strict=True)
    model.sam.image_encoder.load_state_dict(legacy_sam.image_encoder.state_dict(), strict=True)
    model.sam.pixel_mean.copy_(legacy_sam.pixel_mean)
    model.sam.pixel_std.copy_(legacy_sam.pixel_std)


def run_train(
    config: dict[str, Any],
    *,
    model_factories: dict[str, ModelFactory] | None = None,
) -> TrainArtifacts:
    validate_input_data(config, "train").raise_if_errors()
    seed_everything(int(config["runtime"]["seed"]), bool(config["runtime"]["deterministic"]))
    device = resolve_device(str(config["runtime"]["device"]))
    paths = config["paths"]
    outputs = Path(paths["outputs_dir"])
    outputs.mkdir(parents=True, exist_ok=True)
    records, folds, fingerprint, fold_hash = _load_training_records(config)
    _write_json(outputs / "folds.json", {"folds": folds, "fingerprint": fold_hash})
    factories = _factories(config, model_factories)
    branch = "fusion"
    branch_config = _fusion_training_config(config)
    branch_dir = Path(paths["checkpoints_dir"]) / branch
    folds_to_run = [int(fold) for fold in config["data"]["folds"]]
    metrics: dict[str, dict[str, object]] = {branch: {}}
    checkpoint_folds: dict[str, dict[int, Path]] = {branch: {}}
    branch_results: list[pd.DataFrame] = []
    for fold in folds_to_run:
        training_records = [record for record in records if record.fold != fold]
        validation_records = [record for record in records if record.fold == fold]
        if not training_records or not validation_records:
            raise ValidationError(f"fold {fold} has an empty fusion train or validation partition")
        train_loader = _make_loader(training_records, config, fingerprint, branch, training=True)
        validation_loader = _make_loader(validation_records, config, fingerprint, branch, training=False)
        model = factories[branch](branch_config, len(LABELS))
        resume_path = _resume_path(config, branch, fold, branch_dir)
        if resume_path is None and config["checkpoint"][branch].get("initialize") == "external":
            _initialize_external_fusion(model, config, fold)
        result = train_fold(
            model,
            branch,
            fold,
            train_loader,
            validation_loader,
            branch_dir,
            branch_config,
            device,
            LABELS,
            fingerprint,
            fold_hash,
            resume_path=resume_path,
            strict_checkpoint=bool(config["checkpoint"].get("strict", True)),
            precision=str(config["runtime"].get("precision", "amp")),
        )
        frame = prediction_frame(result.uids, result.predictions, LABELS)
        frame.insert(1, "fold", fold)
        branch_results.append(frame)
        metrics[branch][str(fold)] = result.metrics
        checkpoint_folds[branch][fold] = result.best_path
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    oof = pd.concat(branch_results, ignore_index=True)
    expected_uids = [record.uid for record in records if record.fold in folds_to_run]
    if set(oof["StudyInstanceUID"]) != set(expected_uids) or oof["StudyInstanceUID"].duplicated().any():
        raise ValidationError("fusion OOF predictions do not cover each requested study once")
    target_by_uid = {record.uid: np.asarray(record.targets, dtype=np.float32) for record in records}
    oof_targets = np.stack([target_by_uid[str(uid)] for uid in oof["StudyInstanceUID"]])
    metrics[branch]["oof"] = macro_auc(oof_targets, oof.loc[:, LABELS].to_numpy(float), LABELS)
    for index, label in enumerate(LABELS):
        oof[f"target_{label}"] = oof_targets[:, index]
    oof_path = outputs / "oof_fusion.csv"
    oof.to_csv(oof_path, index=False)
    for stale_name in ("oof_knee.csv", "oof_sam.csv", "oof_blended.csv"):
        stale = outputs / stale_name
        if stale.is_file():
            stale.unlink()
    metrics_path = outputs / "metrics.json"
    _write_json(metrics_path, metrics)
    resolved = {key: value for key, value in config.items() if key not in {"project_root", "config_path"}}
    (outputs / "resolved_config.yaml").write_text(yaml.safe_dump(_jsonable(resolved), sort_keys=False))
    return TrainArtifacts(checkpoint_folds, {branch: oof_path}, metrics_path)


def _load_fold_map(config: dict[str, Any]) -> tuple[dict[str, int], str]:
    path = Path(config["paths"]["outputs_dir"]) / "folds.json"
    if not path.is_file():
        raise ValidationError(f"fold map not found: {path}; run training first")
    payload = json.loads(path.read_text())
    folds = {str(uid): int(fold) for uid, fold in payload["folds"].items()}
    fingerprint = str(payload["fingerprint"])
    if fingerprint != fold_fingerprint(folds):
        raise ValidationError("saved fold-map fingerprint is invalid")
    return folds, fingerprint


def _test_records(config: dict[str, Any]) -> list[StudyRecord]:
    frame = pd.read_csv(config["paths"]["test_csv"], dtype={"StudyInstanceUID": str})
    if "StudyInstanceUID" not in frame.columns:
        raise ValidationError("test.csv is missing StudyInstanceUID")
    uids = frame["StudyInstanceUID"].astype(str).tolist()
    if len(uids) != len(set(uids)):
        raise ValidationError("test.csv contains duplicate StudyInstanceUID values")
    return [StudyRecord(uid) for uid in uids]


def run_predict(
    config: dict[str, Any],
    *,
    model_factories: dict[str, ModelFactory] | None = None,
) -> PredictArtifacts:
    validate_input_data(config, "predict").raise_if_errors()
    seed_everything(int(config["runtime"]["seed"]), bool(config["runtime"]["deterministic"]))
    device = resolve_device(str(config["runtime"]["device"]))
    paths = config["paths"]
    outputs = Path(paths["outputs_dir"])
    outputs.mkdir(parents=True, exist_ok=True)
    fingerprint = cache_fingerprint(_cache_settings(config))
    validate_cache_manifest(paths["cache_dir"] / "test" / "manifest.json", fingerprint)
    _, fold_hash = _load_fold_map(config)
    records = _test_records(config)
    validate_cached_studies(
        paths["cache_dir"] / "test",
        [record.uid for record in records],
        fingerprint,
    ).raise_if_errors()
    factories = _factories(config, model_factories)
    branch = "fusion"
    branch_config = _fusion_training_config(config)
    folds_to_run = [int(fold) for fold in config["data"]["folds"]]
    checkpoint_paths = validate_checkpoint_coverage(paths["checkpoints_dir"] / branch, folds_to_run)
    fold_frames: list[pd.DataFrame] = []
    for fold, checkpoint in checkpoint_paths.items():
        model = factories[branch](branch_config, len(LABELS)).to(device)
        load_checkpoint(
            checkpoint,
            model,
            expected={
                "branch": branch,
                "fold": fold,
                "labels": list(LABELS),
                "cache_fingerprint": fingerprint,
                "fold_fingerprint": fold_hash,
                "architecture_fingerprint": architecture_fingerprint(branch, branch_config),
            },
            strict=bool(config["checkpoint"].get("strict", True)),
            map_location=device,
        )
        loader = _make_loader(records, config, fingerprint, branch, training=False)
        predictions, uids, _ = predict_loader(
            model,
            branch,
            loader,
            device,
            precision=str(config["runtime"].get("precision", "amp")),
        )
        fold_frames.append(prediction_frame(uids, predictions, LABELS))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    final = average_fold_predictions(fold_frames, LABELS)
    fusion_path = outputs / "test_fusion.csv"
    final.to_csv(fusion_path, index=False)
    for stale_name in ("test_knee.csv", "test_sam.csv"):
        stale = outputs / stale_name
        if stale.is_file():
            stale.unlink()
    sample_path = Path(paths["sample_submission_csv"])
    expected_uids = [record.uid for record in records]
    if sample_path.is_file():
        sample = pd.read_csv(sample_path, dtype={"StudyInstanceUID": str})
        if "StudyInstanceUID" not in sample:
            raise ValidationError("sample_submission.csv is missing StudyInstanceUID")
        expected_uids = sample["StudyInstanceUID"].astype(str).tolist()
    final = validate_submission(final, expected_uids, LABELS)
    final.loc[:, LABELS] = final.loc[:, LABELS].clip(1e-6, 1 - 1e-6)
    submission = outputs / "submission.csv"
    final.to_csv(submission, index=False)
    return PredictArtifacts(fusion_path, submission)


def run_pipeline(
    config: dict[str, Any],
    *,
    model_factories: dict[str, ModelFactory] | None = None,
) -> PipelineResult:
    validate_input_data(config, "all").raise_if_errors()
    run_preprocess(config)
    run_validate(config, model_factories=model_factories)
    trained = run_train(config, model_factories=model_factories)
    predicted = run_predict(config, model_factories=model_factories)
    return PipelineResult(
        predicted.submission,
        trained.checkpoint_folds.get("fusion", {}),
    )


def _probe_fusion_candidate(config: dict[str, Any]) -> dict[str, Any]:
    """Run one production fusion optimizer step and report CUDA reserved memory."""
    device = torch.device("cuda")
    model: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    try:
        records, _, fingerprint, _ = _load_training_records(config)
        loader = _make_loader(records[:1], config, fingerprint, "fusion", training=True)
        batch = next(iter(loader))
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        model = build_fusion_model(config, n_targets=len(LABELS))
        _initialize_external_fusion(model, config, int(config["data"]["folds"][0]))
        model.to(device).train()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(config["fusion"]["lr"]),
            weight_decay=float(config["fusion"]["weight_decay"]),
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = weighted_bce(
                forward_batch(model, "fusion", batch),
                batch["targets"].float(),
                batch["weights"].float(),
            )
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        return {
            "success": True,
            "oom": False,
            "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
            "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
        }
    except torch.OutOfMemoryError as exc:
        return {"success": False, "oom": True, "peak_reserved": 0, "error": str(exc)}
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            return {"success": False, "oom": True, "peak_reserved": 0, "error": str(exc)}
        return {"success": False, "oom": False, "peak_reserved": 0, "error": str(exc)}
    finally:
        del optimizer, model
        torch.cuda.empty_cache()


def run_auto(
    config: dict[str, Any],
    *,
    plan_only: bool = False,
    hardware_fn: Callable[[], dict[str, Any]] = profile_hardware,
    data_fn: Callable[[dict[str, Any]], dict[str, Any]] = profile_data,
    candidates: list[Candidate] | None = None,
    prepare_fn: Callable[[dict[str, Any]], Any] = run_preprocess,
    probe_fn: Callable[[dict[str, Any]], dict[str, Any]] = _probe_fusion_candidate,
    train_fn: Callable[[dict[str, Any]], Any] = run_train,
) -> dict[str, Any]:
    hardware = hardware_fn()
    data_profile = data_fn(config)
    report: dict[str, Any] = {
        "hardware": hardware,
        "data": data_profile,
        "policy": "balanced",
        "status": "planning",
        "attempts": [],
    }
    if not hardware.get("cuda_available", False):
        report["status"] = "cuda_unavailable"
        write_planner_outputs(config, report, config)
        return {"status": report["status"], "selected": None, "report": report}

    prepare_fn(config)
    recipes = candidates or generate_balanced_candidates()
    for recipe in recipes:
        report.setdefault("estimates", []).append(
            {"candidate": asdict(recipe), **estimate_candidate_memory(recipe)}
        )

    def probe(recipe: Candidate) -> dict[str, Any]:
        return probe_fn(apply_candidate(config, recipe, hardware))

    selected, attempts = select_candidate(
        recipes,
        probe,
        total_vram=int(hardware["vram_total"]),
    )
    report["attempts"] = attempts
    if selected is None:
        report["status"] = "no_safe_candidate"
        write_planner_outputs(config, report, config)
        return {"status": report["status"], "selected": None, "report": report}
    resolved = apply_candidate(config, selected, hardware)
    report["status"] = "selected"
    report["selected"] = asdict(selected)
    write_planner_outputs(config, report, resolved)
    if not plan_only:
        train_fn(resolved)
        report["status"] = "training_completed"
        write_planner_outputs(config, report, resolved)
    return {"status": report["status"], "selected": asdict(selected), "report": report}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and ensemble knee MRI classifiers",
    )
    parser.add_argument(
        "command",
        choices=("preprocess", "train", "validate", "predict", "all", "auto"),
    )
    parser.add_argument("--config", default="config/training.yaml")
    parser.add_argument(
        "--folds",
        default=None,
        help="optional comma-separated fold override",
    )
    parser.add_argument("--plan-only", action="store_true", help="probe and write the resolved config without training")
    return parser


def _apply_fold_override(config: dict[str, Any], value: str | None) -> dict[str, Any]:
    if value is None:
        return config
    result = deepcopy(config)
    try:
        folds = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ConfigError("--folds must contain comma-separated integers") from exc
    if not folds:
        raise ConfigError("--folds cannot be empty")
    invalid = [fold for fold in folds if fold < 0 or fold >= result["data"]["n_folds"]]
    if invalid:
        raise ConfigError(f"--folds contains invalid folds: {invalid}")
    result["data"]["folds"] = folds
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        config = _apply_fold_override(load_config(args.config), args.folds)
        if args.command == "auto":
            result = run_auto(config, plan_only=args.plan_only)
            if result["status"] in {"cuda_unavailable", "no_safe_candidate"}:
                LOGGER.error("automatic planning stopped: %s", result["status"])
                return 2
        elif args.command == "validate":
            run_validate(config)
        elif args.command == "preprocess":
            run_preprocess(config)
        elif args.command == "train":
            run_train(config)
        elif args.command == "predict":
            run_predict(config)
        else:
            run_pipeline(config)
    except (ConfigError, ValidationError, ValueError, FileNotFoundError) as exc:
        LOGGER.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
