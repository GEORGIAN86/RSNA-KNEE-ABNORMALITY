"""Hardware/data profiling and deterministic balanced training selection."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
import yaml


@dataclass(frozen=True)
class Candidate:
    batch_size: int
    sam_input_size: int
    sam_slices: int
    dino_blocks: int
    sam_blocks: int
    checkpointing: bool
    gradient_accumulation: int = 8
    score: float = 0.0


def generate_balanced_candidates() -> list[Candidate]:
    recipes = [
        (512, 2, 4, 2, False),
        (512, 2, 4, 2, True),
        (384, 2, 4, 2, True),
        (256, 2, 4, 2, True),
        (256, 1, 4, 2, True),
        (256, 1, 4, 1, True),
        (256, 1, 4, 0, True),
        (256, 1, 2, 0, True),
        (256, 1, 0, 0, True),
    ]
    result = []
    for index, (size, slices, dino, sam, checkpointing) in enumerate(recipes):
        result.append(Candidate(1, size, slices, dino, sam, checkpointing, 8, 100.0 - index))
    return result


def estimate_candidate_memory(candidate: Candidate, *, total_parameters: int = 114_917_260) -> dict[str, int]:
    """Conservative relative estimate used to order probes, not to certify fit."""
    parameter_bytes = int(total_parameters * 4)
    trainable_fraction = (candidate.dino_blocks / 12) * 0.23 + (candidate.sam_blocks / 12) * 0.78
    trainable_bytes = int(parameter_bytes * max(0.0, min(1.0, trainable_fraction)))
    sam_activation_bytes = int(candidate.sam_slices * 6 * candidate.sam_input_size**2 * 96)
    dino_activation_bytes = int(6 * 16 * 336**2 * 10)
    activation_bytes = sam_activation_bytes + dino_activation_bytes
    if candidate.checkpointing:
        activation_bytes = int(activation_bytes * 0.62)
    return {
        "estimated_parameter_bytes": parameter_bytes,
        "estimated_trainable_bytes": trainable_bytes,
        "estimated_activation_bytes": activation_bytes,
        "estimated_peak_bytes": parameter_bytes + 3 * trainable_bytes + activation_bytes,
    }


def select_workers(*, cpu_count: int, available_ram: int) -> int:
    cpu_limit = max(0, min(8, cpu_count // 2))
    ram_limit = max(0, int(available_ram // (2 * 1024**3)))
    return min(cpu_limit, ram_limit)


def profile_hardware() -> dict[str, Any]:
    page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
    pages = os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else 0
    try:
        available_pages = os.sysconf("SC_AVPHYS_PAGES") if hasattr(os, "sysconf") else pages
    except (OSError, ValueError):
        available_pages = pages
    profile: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count() or 1,
        "ram_total": int(page_size * pages),
        "ram_available": int(page_size * available_pages),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        profile.update(
            {
                "device": "cuda",
                "gpu_name": properties.name,
                "compute_capability": [int(properties.major), int(properties.minor)],
                "vram_total": int(total),
                "vram_free": int(free),
                "amp_supported": True,
            }
        )
    else:
        profile.update({"device": "cpu", "vram_total": 0, "vram_free": 0, "amp_supported": False})
    return profile


def profile_data(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    result: dict[str, Any] = {}
    for split in ("train", "test"):
        studies = pd.read_csv(paths[f"{split}_csv"], dtype={"StudyInstanceUID": str})
        series = pd.read_csv(
            paths[f"{split}_series_csv"],
            dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
        )
        counts = series.groupby("StudyInstanceUID").size()
        root = Path(paths["data_dir"]) / f"{split}_series"
        dicom_files = list(root.rglob("*.dcm")) if root.is_dir() else []
        result[split] = {
            "studies": int(studies["StudyInstanceUID"].nunique()),
            "series": int(len(series)),
            "series_per_study": {
                "median": float(counts.median()) if len(counts) else 0.0,
                "p95": float(counts.quantile(0.95)) if len(counts) else 0.0,
                "max": int(counts.max()) if len(counts) else 0,
            },
            "dicom_files": len(dicom_files),
            "dicom_bytes": int(sum(path.stat().st_size for path in dicom_files)),
        }
    cache = Path(paths["cache_dir"])
    result["cache_bytes"] = int(sum(path.stat().st_size for path in cache.rglob("*") if path.is_file())) if cache.exists() else 0
    disk = shutil.disk_usage(Path(paths["data_dir"]))
    result["disk_free"] = int(disk.free)
    return result


def select_candidate(
    candidates: list[Candidate],
    probe: Callable[[Candidate], dict[str, Any]],
    *,
    total_vram: int,
) -> tuple[Candidate | None, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        outcome = dict(probe(candidate))
        record = {"candidate": asdict(candidate), **outcome}
        records.append(record)
        if outcome.get("success") and int(outcome.get("peak_reserved", 0)) <= int(total_vram * 0.85):
            return candidate, records
        if not outcome.get("oom", False) and not outcome.get("success", False):
            raise RuntimeError(str(outcome.get("error", "resource probe failed")))
    return None, records


def apply_candidate(config: dict[str, Any], candidate: Candidate, hardware: dict[str, Any]) -> dict[str, Any]:
    from copy import deepcopy

    resolved = deepcopy(config)
    resolved["knee"]["trainable_blocks"] = candidate.dino_blocks
    resolved["knee"]["gradient_checkpointing"] = candidate.checkpointing
    resolved["sam"]["trainable_blocks"] = candidate.sam_blocks
    resolved["sam"]["slices_per_slot"] = candidate.sam_slices
    resolved["sam"]["input_size"] = candidate.sam_input_size
    resolved["sam"]["gradient_checkpointing"] = candidate.checkpointing
    resolved["fusion"]["batch_size"] = candidate.batch_size
    resolved["fusion"]["gradient_accumulation"] = candidate.gradient_accumulation
    resolved["data"]["workers"] = select_workers(
        cpu_count=int(hardware["cpu_count"]), available_ram=int(hardware["ram_available"])
    )
    resolved["runtime"]["device"] = "cuda"
    resolved["runtime"]["precision"] = "amp"
    return resolved


def _serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def write_planner_outputs(
    config: dict[str, Any], report: dict[str, Any], resolved: dict[str, Any]
) -> tuple[Path, Path]:
    outputs = Path(config["paths"]["outputs_dir"])
    outputs.mkdir(parents=True, exist_ok=True)
    report_path = outputs / "resource_report.json"
    report_tmp = report_path.with_suffix(".json.tmp")
    report_tmp.write_text(json.dumps(_serializable(report), indent=2, sort_keys=True))
    report_tmp.replace(report_path)
    config_path = outputs / "auto_training.yaml"
    config_tmp = config_path.with_suffix(".yaml.tmp")
    payload = {key: value for key, value in resolved.items() if key not in {"project_root", "config_path"}}
    config_tmp.write_text(yaml.safe_dump(_serializable(payload), sort_keys=False))
    config_tmp.replace(config_path)
    return report_path, config_path


__all__ = [
    "Candidate",
    "apply_candidate",
    "estimate_candidate_memory",
    "generate_balanced_candidates",
    "profile_data",
    "profile_hardware",
    "select_candidate",
    "select_workers",
    "write_planner_outputs",
]
