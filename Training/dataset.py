"""Target construction, fold assignment, and cached study datasets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from Helpers.cache import load_cached_study
from .constants import LABELS


@dataclass(frozen=True)
class TargetTable:
    uids: list[str]
    targets: np.ndarray
    weights: np.ndarray
    is_gold: np.ndarray


def _validate_label_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = ["StudyInstanceUID", *LABELS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    result = frame.copy()
    result["StudyInstanceUID"] = result["StudyInstanceUID"].astype(str)
    if result["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"{name} contains duplicate StudyInstanceUID values")
    return result


def build_targets(
    gold: pd.DataFrame,
    weak: pd.DataFrame | None,
    *,
    gold_weight: float,
    silent_value: float,
    silent_weight: float,
) -> TargetTable:
    """Build aligned target and confidence matrices with gold overrides."""

    gold_frame = _validate_label_frame(gold, "gold labels")
    complete_gold = gold_frame.dropna(subset=list(LABELS), how="any").copy()

    if weak is None:
        base = complete_gold[["StudyInstanceUID", *LABELS]].copy()
        targets = base.loc[:, LABELS].to_numpy(dtype=np.float32, copy=True)
        weights = np.full_like(targets, float(gold_weight), dtype=np.float32)
        is_gold = np.ones(len(base), dtype=bool)
        return TargetTable(base["StudyInstanceUID"].tolist(), targets, weights, is_gold)

    weak_frame = _validate_label_frame(weak, "weak labels")
    weak_frame = weak_frame.dropna(subset=list(LABELS), how="any").copy()
    missing_gold = complete_gold.loc[
        ~complete_gold["StudyInstanceUID"].isin(weak_frame["StudyInstanceUID"]),
        ["StudyInstanceUID", *LABELS],
    ]
    base = pd.concat(
        [weak_frame[["StudyInstanceUID", *LABELS]], missing_gold],
        ignore_index=True,
    )
    base = base.set_index("StudyInstanceUID")
    target_values = base.loc[:, LABELS].to_numpy(dtype=np.float32, copy=True)
    weights = np.where(
        np.isclose(target_values, silent_value, atol=1e-6),
        float(silent_weight),
        1.0,
    ).astype(np.float32)
    is_gold = np.zeros(len(base), dtype=bool)

    gold_by_uid = complete_gold.set_index("StudyInstanceUID")
    gold_uids = [uid for uid in base.index if uid in gold_by_uid.index]
    if gold_uids:
        positions = base.index.get_indexer(gold_uids)
        target_values[positions] = gold_by_uid.loc[gold_uids, LABELS].to_numpy(dtype=np.float32)
        weights[positions] = float(gold_weight)
        is_gold[positions] = True

    return TargetTable(base.index.tolist(), target_values, weights, is_gold)


def _normalized_report(value: object) -> str:
    return " ".join(str(value).split()).lower()


def build_fold_map(train: pd.DataFrame, uids: list[str], n_folds: int) -> dict[str, int]:
    """Assign reports deterministically so duplicate reports stay together."""

    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if "StudyInstanceUID" not in train.columns:
        raise ValueError("train data is missing StudyInstanceUID")
    frame = train.copy()
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError("train data contains duplicate StudyInstanceUID values")
    requested = [str(uid) for uid in uids]
    missing = sorted(set(requested).difference(frame["StudyInstanceUID"]))
    if missing:
        raise ValueError(f"requested UIDs are missing from train data: {missing[:10]}")
    reports = (
        frame.set_index("StudyInstanceUID")["Report"].fillna("").to_dict()
        if "Report" in frame.columns
        else {}
    )
    result: dict[str, int] = {}
    for raw_uid in requested:
        uid = str(raw_uid)
        report = _normalized_report(reports.get(uid, "")) or uid.lower()
        digest = hashlib.md5(report.encode("utf-8"), usedforsecurity=False).hexdigest()
        result[uid] = int(digest[:8], 16) % n_folds
    return result


def fold_fingerprint(folds: dict[str, int]) -> str:
    payload = "\n".join(f"{uid}:{folds[uid]}" for uid in sorted(folds))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StudyRecord:
    uid: str
    targets: np.ndarray | None = None
    weights: np.ndarray | None = None
    fold: int | None = None


def _sample_indices(valid: np.ndarray, count: int, training: bool) -> np.ndarray:
    if valid.size == 0:
        return np.empty(0, dtype=np.int64)
    if training:
        return np.sort(np.random.choice(valid, size=count, replace=valid.size < count)).astype(np.int64)
    if valid.size >= count:
        positions = np.linspace(0, valid.size - 1, count).round().astype(np.int64)
        return valid[positions]
    return np.concatenate([valid, np.full(count - valid.size, valid[-1], dtype=np.int64)])


class StudyDataset(Dataset):
    """Read one shared cached study and sample it for a model branch."""

    def __init__(
        self,
        records: list[StudyRecord],
        cache_dir: str | Path,
        cache_fingerprint: str,
        *,
        branch: str,
        training: bool,
        knee_slices: int,
        sam_slices: int,
    ) -> None:
        if branch not in {"knee", "sam", "fusion"}:
            raise ValueError("branch must be knee, sam, or fusion")
        if knee_slices <= 0 or sam_slices <= 0:
            raise ValueError("slice counts must be positive")
        self.records = records
        self.cache_dir = Path(cache_dir)
        self.cache_fingerprint = cache_fingerprint
        self.branch = branch
        self.training = training
        self.knee_slices = knee_slices
        self.sam_slices = sam_slices

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        cached = load_cached_study(
            self.cache_dir / f"{record.uid}.npz",
            expected_fingerprint=self.cache_fingerprint,
        )
        if cached.uid != str(record.uid):
            raise ValueError(f"cache UID mismatch: expected {record.uid}, found {cached.uid}")
        if not cached.slot_mask.any():
            raise ValueError(f"study {record.uid} has no usable imaging in the shared cache")
        requested = self.knee_slices if self.branch in {"knee", "fusion"} else self.sam_slices
        output = np.zeros(
            (cached.images.shape[0], requested, *cached.images.shape[2:]),
            dtype=np.uint8,
        )
        sampled_mask = np.zeros((cached.images.shape[0], requested), dtype=bool)
        for slot in range(cached.images.shape[0]):
            valid = np.flatnonzero(cached.slice_mask[slot])
            selected = _sample_indices(valid, requested, self.training)
            if selected.size:
                output[slot] = cached.images[slot, selected]
                sampled_mask[slot] = True
        slot_mask = sampled_mask.any(axis=1)
        item: dict[str, object] = {
            "uid": str(record.uid),
            "images": torch.from_numpy(output),
            "slot_mask": torch.from_numpy(slot_mask),
            "slice_mask": torch.from_numpy(sampled_mask),
        }
        if self.branch == "fusion":
            sam_output = np.zeros(
                (cached.images.shape[0], self.sam_slices, *cached.images.shape[2:]),
                dtype=np.uint8,
            )
            sam_mask = np.zeros((cached.images.shape[0], self.sam_slices), dtype=bool)
            for slot in range(cached.images.shape[0]):
                valid = np.flatnonzero(cached.slice_mask[slot])
                selected = _sample_indices(valid, self.sam_slices, self.training)
                if selected.size:
                    sam_output[slot] = cached.images[slot, selected]
                    sam_mask[slot] = True
            item["sam_images"] = torch.from_numpy(sam_output)
            item["sam_slice_mask"] = torch.from_numpy(sam_mask)
        if record.targets is not None:
            item["targets"] = torch.as_tensor(record.targets, dtype=torch.float32)
        if record.weights is not None:
            item["weights"] = torch.as_tensor(record.weights, dtype=torch.float32)
        if record.fold is not None:
            item["fold"] = torch.tensor(record.fold, dtype=torch.int64)
        return item


__all__ = [
    "StudyDataset",
    "StudyRecord",
    "TargetTable",
    "build_fold_map",
    "build_targets",
    "fold_fingerprint",
]
