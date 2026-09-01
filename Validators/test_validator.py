"""Prediction and submission artifact validation.

The filename is retained for compatibility with the original project. This
module contains validators; it is not a pytest test module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_predictions(
    frame: pd.DataFrame,
    expected_uids: list[str],
    labels: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    required = ["StudyInstanceUID", *labels]
    missing = [column for column in required if column not in frame.columns]
    extra = [column for column in frame.columns if column not in required]
    if missing or extra:
        raise ValueError(f"prediction columns mismatch; missing={missing}, extra={extra}")
    result = frame.copy()
    result["StudyInstanceUID"] = result["StudyInstanceUID"].astype(str)
    if result["StudyInstanceUID"].duplicated().any():
        raise ValueError("prediction contains duplicate StudyInstanceUID values")
    expected = [str(uid) for uid in expected_uids]
    if len(expected) != len(set(expected)):
        raise ValueError("expected_uids contains duplicate values")
    actual = set(result["StudyInstanceUID"])
    if actual != set(expected):
        missing_uids = sorted(set(expected).difference(actual))
        extra_uids = sorted(actual.difference(expected))
        raise ValueError(f"prediction UID mismatch; missing={missing_uids}, extra={extra_uids}")
    numeric = result.loc[:, labels].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("prediction contains non-finite probabilities")
    if ((numeric < 0) | (numeric > 1)).any():
        raise ValueError("prediction probabilities must be between 0 and 1")
    indexed = result.set_index("StudyInstanceUID")
    return indexed.loc[expected].reset_index()


def validate_submission(
    frame: pd.DataFrame,
    expected_uids: list[str],
    labels: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    return validate_predictions(frame, expected_uids, labels)


__all__ = ["validate_predictions", "validate_submission"]
