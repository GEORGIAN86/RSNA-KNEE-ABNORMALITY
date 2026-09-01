"""Series selection and DICOM pixel normalization for the six-slot cache."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom

from Training.constants import SLOTS


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "fat", "fs"}
    return bool(value)


def choose_slot_records(series: pd.DataFrame) -> list[dict[str, object] | None]:
    required = {"Anatomical_Plane", "Fat_Suppression", "SeriesInstanceUID"}
    missing = sorted(required.difference(series.columns))
    if missing:
        raise ValueError(f"series metadata is missing columns: {missing}")
    frame = series.copy()
    if "n_slices" not in frame:
        frame["n_slices"] = 0
    frame["_fat"] = frame["Fat_Suppression"].map(_as_bool)
    frame["n_slices"] = pd.to_numeric(frame["n_slices"], errors="coerce").fillna(0)
    chosen: list[dict[str, object] | None] = []
    for plane, fat_suppressed in SLOTS:
        candidates = frame[
            (frame["Anatomical_Plane"] == plane) & (frame["_fat"] == fat_suppressed)
        ]
        if candidates.empty:
            chosen.append(None)
        else:
            chosen.append(candidates.sort_values("n_slices").iloc[-1].drop(labels=["_fat"]).to_dict())
    return chosen


def _natural_key(path: Path) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name))


def ordered_slice_files(series_dir: str | Path) -> list[Path]:
    files = sorted(Path(series_dir).glob("*.dcm"), key=_natural_key)
    records: list[tuple[Path, float | None, float | None]] = []
    for path in files:
        geometry: float | None = None
        instance: float | None = None
        try:
            dataset = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True,
                specific_tags=["ImagePositionPatient", "ImageOrientationPatient", "InstanceNumber"],
            )
            position = getattr(dataset, "ImagePositionPatient", None)
            orientation = getattr(dataset, "ImageOrientationPatient", None)
            if position is not None and orientation is not None and len(position) >= 3 and len(orientation) >= 6:
                ipp = np.asarray(position[:3], dtype=np.float64)
                iop = np.asarray(orientation[:6], dtype=np.float64)
                if np.isfinite(ipp).all() and np.isfinite(iop).all():
                    geometry = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
            raw_instance = getattr(dataset, "InstanceNumber", None)
            if raw_instance is not None:
                instance = float(raw_instance)
        except Exception:
            pass
        records.append((path, geometry, instance))
    minimum = max(2, int(np.ceil(0.8 * len(records)))) if records else 0
    if sum(value is not None for _, value, _ in records) >= minimum:
        return [item[0] for item in sorted(records, key=lambda item: (float("inf") if item[1] is None else item[1], _natural_key(item[0])))]
    if sum(value is not None for _, _, value in records) >= minimum:
        return [item[0] for item in sorted(records, key=lambda item: (float("inf") if item[2] is None else item[2], _natural_key(item[0])))]
    return files


def read_dicom_array(path: str | Path) -> tuple[np.ndarray, object] | None:
    try:
        dataset = pydicom.dcmread(str(path), force=True)
        image = dataset.pixel_array.astype(np.float32)
        image = image * float(getattr(dataset, "RescaleSlope", 1) or 1)
        image = image + float(getattr(dataset, "RescaleIntercept", 0) or 0)
        if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            image = image.max() - image
        return image, dataset
    except Exception:
        return None


def right_knee(dataset: object) -> bool:
    for name in ("ImageLaterality", "Laterality"):
        value = str(getattr(dataset, name, "") or "").strip().upper()
        if value[:1] in {"L", "R"}:
            return value.startswith("R")
    position = getattr(dataset, "ImagePositionPatient", None)
    try:
        return position is not None and float(position[0]) < 0
    except (TypeError, ValueError, IndexError):
        return False


def crop_and_window(
    image: np.ndarray,
    *,
    pixel_spacing: float | None,
    crop_mm: float,
    image_size: int,
    flip_horizontal: bool,
) -> np.ndarray:
    spacing = pixel_spacing if pixel_spacing and np.isfinite(pixel_spacing) and pixel_spacing > 0 else crop_mm / max(image.shape)
    half = max(1, int(round(crop_mm / spacing / 2)))
    center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
    cropped = image[
        max(0, center_y - half) : min(image.shape[0], center_y + half),
        max(0, center_x - half) : min(image.shape[1], center_x + half),
    ]
    if cropped.size == 0:
        raise ValueError("physical crop is empty")
    sample = cropped[::4, ::4] if min(cropped.shape) >= 4 else cropped
    low, high = np.percentile(sample, [1, 99])
    scaled = np.clip((cropped - low) / max(float(high - low), 1e-6), 0, 1)
    resized = cv2.resize(scaled, (image_size, image_size), interpolation=cv2.INTER_AREA)
    if flip_horizontal:
        resized = resized[:, ::-1].copy()
    return np.rint(resized * 255).clip(0, 255).astype(np.uint8)


def render_dicom(path: str | Path, *, plane: str, crop_mm: float, image_size: int) -> np.ndarray | None:
    decoded = read_dicom_array(path)
    if decoded is None:
        return None
    image, dataset = decoded
    spacing: float | None = None
    try:
        spacing = float(dataset.PixelSpacing[0])
    except (AttributeError, TypeError, ValueError, IndexError):
        pass
    return crop_and_window(
        image,
        pixel_spacing=spacing,
        crop_mm=crop_mm,
        image_size=image_size,
        flip_horizontal=plane != "Sagittal" and right_knee(dataset),
    )


__all__ = [
    "choose_slot_records",
    "crop_and_window",
    "ordered_slice_files",
    "read_dicom_array",
    "render_dicom",
    "right_knee",
]
