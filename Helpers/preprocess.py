"""DICOM series probing and metadata normalization."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import pydicom


HEADER_TAGS = (
    "SeriesDescription",
    "SequenceName",
    "ScanOptions",
    "ScanningSequence",
    "Laterality",
    "ImageLaterality",
    "PixelSpacing",
    "ImagePositionPatient",
    "ImageOrientationPatient",
)


def _plain_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        return "|".join(str(part) for part in value)
    return str(value)


def probe_series(
    series_dir: str | Path,
    *,
    study_uid: str | None = None,
    series_uid: str | None = None,
) -> dict[str, object]:
    """Read one representative header without decoding pixel data."""

    path = Path(series_dir)
    files = sorted(path.glob("*.dcm"))
    row: dict[str, object] = {
        "StudyInstanceUID": str(study_uid or path.parent.name),
        "SeriesInstanceUID": str(series_uid or path.name),
        "series_dir": str(path),
        "n_slices": len(files),
    }
    if not files:
        row["error"] = "no DICOM files"
        return row
    try:
        dataset = pydicom.dcmread(
            str(files[len(files) // 2]),
            stop_before_pixels=True,
            force=True,
            specific_tags=list(HEADER_TAGS),
        )
        for tag in HEADER_TAGS:
            row[tag] = _plain_value(getattr(dataset, tag, None))
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:240]
    return row


def walk_series(series_root: str | Path) -> pd.DataFrame:
    root = Path(series_root)
    rows: list[dict[str, object]] = []
    if not root.is_dir():
        return pd.DataFrame(rows)
    for study in sorted((entry for entry in os.scandir(root) if entry.is_dir()), key=lambda item: item.name):
        for series in sorted((entry for entry in os.scandir(study.path) if entry.is_dir()), key=lambda item: item.name):
            rows.append(probe_series(series.path, study_uid=study.name, series_uid=series.name))
    return annotate_series(pd.DataFrame(rows))


def _plane_from_text(value: object) -> str | None:
    text = str(value or "").lower()
    if re.search(r"\bsag(?:ittal)?\b", text):
        return "Sagittal"
    if re.search(r"\bcor(?:onal)?\b", text):
        return "Coronal"
    if re.search(r"\baxi(?:al)?\b|\btra(?:nsverse)?\b", text):
        return "Axial"
    return None


def _fat_suppressed_text(value: object) -> bool:
    text = str(value or "").lower()
    return bool(re.search(r"fat\s*sat|fat\s*supp|\bfs\b|stir|spair", text))


def annotate_series(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize competition metadata or infer plane/fat suppression from text."""

    result = frame.copy()
    if result.empty:
        for column in ("Anatomical_Plane", "Fat_Suppression", "n_slices"):
            if column not in result:
                result[column] = pd.Series(dtype=object)
        return result
    description = result.get("SeriesDescription", pd.Series("", index=result.index)).fillna("")
    sequence = result.get("SequenceName", pd.Series("", index=result.index)).fillna("")
    combined = description.astype(str) + " " + sequence.astype(str)
    if "Anatomical_Plane" not in result:
        result["Anatomical_Plane"] = combined.map(_plane_from_text)
    else:
        inferred = combined.map(_plane_from_text)
        result["Anatomical_Plane"] = result["Anatomical_Plane"].where(
            result["Anatomical_Plane"].notna(), inferred
        )
    if "Fat_Suppression" not in result:
        result["Fat_Suppression"] = combined.map(_fat_suppressed_text)
    if "n_slices" not in result:
        result["n_slices"] = 0
    result["n_slices"] = pd.to_numeric(result["n_slices"], errors="coerce").fillna(0).astype(int)
    return result


__all__ = ["HEADER_TAGS", "annotate_series", "probe_series", "walk_series"]
