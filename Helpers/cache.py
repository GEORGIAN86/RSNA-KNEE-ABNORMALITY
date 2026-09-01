"""Atomic, fingerprinted six-slot study cache."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Training.constants import CACHE_SCHEMA_VERSION, SLOTS

from .normalization import choose_slot_records, ordered_slice_files, render_dicom
from .preprocess import annotate_series


class CacheError(ValueError):
    """Raised for an invalid or incompatible cached study."""


@dataclass(frozen=True)
class CachedStudy:
    uid: str
    images: np.ndarray
    slot_mask: np.ndarray
    slice_mask: np.ndarray
    fingerprint: str


def cache_fingerprint(settings: dict[str, Any]) -> str:
    payload = {"schema_version": CACHE_SCHEMA_VERSION, **settings}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_arrays(images: np.ndarray, slot_mask: np.ndarray, slice_mask: np.ndarray) -> None:
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[0] != len(SLOTS):
        raise CacheError("images must be uint8 with shape [6, slices, height, width]")
    if slot_mask.shape != (len(SLOTS),):
        raise CacheError("slot_mask must have shape [6]")
    if slice_mask.shape != images.shape[:2]:
        raise CacheError("slice_mask must match the image slot/slice dimensions")
    if np.any(np.asarray(slot_mask, dtype=bool) != np.asarray(slice_mask, dtype=bool).any(axis=1)):
        raise CacheError("slot_mask and slice_mask disagree")


def save_cached_study(
    path: str | Path,
    uid: str,
    images: np.ndarray,
    slot_mask: np.ndarray,
    slice_mask: np.ndarray,
    fingerprint: str,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    slot_mask = np.asarray(slot_mask, dtype=bool)
    slice_mask = np.asarray(slice_mask, dtype=bool)
    _validate_arrays(images, slot_mask, slice_mask)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle,
                uid=np.asarray(str(uid)),
                images=images,
                slot_mask=slot_mask,
                slice_mask=slice_mask,
                fingerprint=np.asarray(str(fingerprint)),
                schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int64),
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_cached_study(path: str | Path, *, expected_fingerprint: str | None = None) -> CachedStudy:
    source = Path(path)
    if not source.is_file():
        raise CacheError(f"cache file not found: {source}")
    try:
        with np.load(source, allow_pickle=False) as payload:
            schema = int(payload["schema_version"])
            fingerprint = str(payload["fingerprint"].item())
            uid = str(payload["uid"].item())
            images = payload["images"].copy()
            slot_mask = payload["slot_mask"].astype(bool, copy=True)
            slice_mask = payload["slice_mask"].astype(bool, copy=True)
    except (KeyError, OSError, ValueError) as exc:
        raise CacheError(f"invalid cache file {source}: {exc}") from exc
    if schema != CACHE_SCHEMA_VERSION:
        raise CacheError(f"cache schema mismatch: found {schema}, expected {CACHE_SCHEMA_VERSION}")
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise CacheError(f"cache fingerprint mismatch for {source.name}")
    _validate_arrays(images, slot_mask, slice_mask)
    return CachedStudy(uid, images, slot_mask, slice_mask, fingerprint)


def build_study_cache(
    uid: str,
    series: pd.DataFrame,
    series_root: str | Path,
    output_path: str | Path,
    settings: dict[str, Any],
    fingerprint: str,
) -> tuple[str, list[str]]:
    image_size = int(settings["image_size"])
    max_slices = int(settings["max_slices_per_slot"])
    crop_mm = float(settings["crop_mm"])
    band_low, band_high = map(float, settings.get("slice_band", (0.12, 0.88)))
    images = np.zeros((len(SLOTS), max_slices, image_size, image_size), dtype=np.uint8)
    slice_mask = np.zeros((len(SLOTS), max_slices), dtype=bool)
    failures: list[str] = []
    annotated = annotate_series(series)
    for slot_index, record in enumerate(choose_slot_records(annotated)):
        if record is None:
            continue
        plane = SLOTS[slot_index][0]
        series_uid = str(record["SeriesInstanceUID"])
        series_dir = Path(series_root) / str(uid) / series_uid
        files = ordered_slice_files(series_dir)
        if not files:
            failures.append(f"{uid}/{series_uid}: no readable slice headers")
            continue
        start = int(round(band_low * (len(files) - 1)))
        stop = int(round(band_high * (len(files) - 1)))
        available = np.arange(start, stop + 1, dtype=int)
        if len(available) > max_slices:
            available = available[np.linspace(0, len(available) - 1, max_slices).round().astype(int)]
        write_index = 0
        for file_index in available:
            rendered = render_dicom(files[int(file_index)], plane=plane, crop_mm=crop_mm, image_size=image_size)
            if rendered is None:
                failures.append(f"{uid}/{series_uid}/{files[int(file_index)].name}: decode failed")
                continue
            images[slot_index, write_index] = rendered
            slice_mask[slot_index, write_index] = True
            write_index += 1
    slot_mask = slice_mask.any(axis=1)
    if not slot_mask.any():
        raise CacheError(f"study {uid} has no usable MRI slots after preprocessing")
    save_cached_study(output_path, uid, images, slot_mask, slice_mask, fingerprint)
    return str(uid), failures


def build_cache_split(
    series_csv: str | Path,
    series_root: str | Path,
    output_dir: str | Path,
    settings: dict[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    frame = pd.read_csv(series_csv, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    required = {"StudyInstanceUID", "SeriesInstanceUID"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise CacheError(f"{Path(series_csv).name} is missing columns: {missing}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = cache_fingerprint(settings)
    failures: list[str] = []
    built = 0
    reused = 0
    for uid, records in frame.groupby("StudyInstanceUID", sort=True):
        path = output / f"{uid}.npz"
        if path.is_file():
            try:
                cached = load_cached_study(path, expected_fingerprint=fingerprint)
                if cached.uid != str(uid):
                    raise CacheError(
                        f"cache UID mismatch: expected {uid}, found {cached.uid}"
                    )
                if not cached.slot_mask.any():
                    raise CacheError(f"study {uid} has no usable MRI slots in the cache")
                reused += 1
                continue
            except CacheError:
                pass
        _, study_failures = build_study_cache(str(uid), records, series_root, path, settings, fingerprint)
        failures.extend(study_failures)
        built += 1
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "split": split,
        "studies": int(frame["StudyInstanceUID"].nunique()),
        "built": built,
        "reused": reused,
        "failures": failures,
    }
    manifest_path = output / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temporary.replace(manifest_path)
    return manifest


__all__ = [
    "CacheError",
    "CachedStudy",
    "build_cache_split",
    "build_study_cache",
    "cache_fingerprint",
    "load_cached_study",
    "save_cached_study",
]
