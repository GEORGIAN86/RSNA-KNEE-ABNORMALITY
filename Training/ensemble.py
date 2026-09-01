"""UID-safe fold averaging and knee/SAM prediction blending."""

from __future__ import annotations

import numpy as np
import pandas as pd

from Validators.test_validator import validate_predictions


def prediction_frame(
    uids: list[str],
    predictions: np.ndarray,
    labels: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    values = np.asarray(predictions, dtype=float)
    if values.shape != (len(uids), len(labels)):
        raise ValueError("prediction matrix shape does not match UIDs and labels")
    frame = pd.DataFrame(values, columns=list(labels))
    frame.insert(0, "StudyInstanceUID", [str(uid) for uid in uids])
    return validate_predictions(frame, [str(uid) for uid in uids], labels)


def average_fold_predictions(
    frames: list[pd.DataFrame],
    labels: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    if not frames:
        raise ValueError("at least one fold prediction frame is required")
    order = frames[0]["StudyInstanceUID"].astype(str).tolist()
    aligned = [validate_predictions(frame, order, labels) for frame in frames]
    values = np.mean([frame.loc[:, labels].to_numpy(dtype=float) for frame in aligned], axis=0)
    return prediction_frame(order, values, labels)


def blend_predictions(
    knee: pd.DataFrame,
    sam: pd.DataFrame,
    labels: list[str] | tuple[str, ...],
    *,
    sam_weight: float,
    kind: str,
) -> pd.DataFrame:
    if not 0 <= sam_weight <= 1:
        raise ValueError("sam_weight must be between 0 and 1")
    if kind not in {"rank", "probability"}:
        raise ValueError("kind must be rank or probability")
    knee_order = knee["StudyInstanceUID"].astype(str).tolist()
    if set(knee_order) != set(sam["StudyInstanceUID"].astype(str)):
        raise ValueError("knee and SAM prediction UID sets do not match")
    knee_aligned = validate_predictions(knee, knee_order, labels)
    sam_aligned = validate_predictions(sam, knee_order, labels)
    result = pd.DataFrame({"StudyInstanceUID": knee_order})
    for label in labels:
        knee_values = knee_aligned[label].astype(float)
        sam_values = sam_aligned[label].astype(float)
        if kind == "rank":
            knee_values = knee_values.rank(method="average", pct=True)
            sam_values = sam_values.rank(method="average", pct=True)
        result[label] = (1.0 - sam_weight) * knee_values.to_numpy() + sam_weight * sam_values.to_numpy()
    return validate_predictions(result, knee_order, labels)


__all__ = ["average_fold_predictions", "blend_predictions", "prediction_frame"]
