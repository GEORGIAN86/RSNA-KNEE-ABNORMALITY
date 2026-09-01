from pathlib import Path

import numpy as np
import pytest

from Helpers.cache import save_cached_study
from Training.dataset import StudyDataset, StudyRecord


def write_study(path: Path, fingerprint: str):
    images = np.arange(6 * 5 * 8 * 8, dtype=np.uint32).reshape(6, 5, 8, 8).astype(np.uint8)
    save_cached_study(
        path,
        "s",
        images,
        np.ones(6, bool),
        np.ones((6, 5), bool),
        fingerprint,
    )


def test_knee_dataset_returns_six_three_channel_slots(tmp_path: Path):
    write_study(tmp_path / "s.npz", "fp")
    dataset = StudyDataset(
        [StudyRecord("s")],
        tmp_path,
        "fp",
        branch="knee",
        training=False,
        knee_slices=3,
        sam_slices=4,
    )

    item = dataset[0]

    assert tuple(item["images"].shape) == (6, 3, 8, 8)
    assert tuple(item["slot_mask"].shape) == (6,)


def test_sam_dataset_returns_slice_mask(tmp_path: Path):
    write_study(tmp_path / "s.npz", "fp")
    dataset = StudyDataset(
        [StudyRecord("s")],
        tmp_path,
        "fp",
        branch="sam",
        training=False,
        knee_slices=3,
        sam_slices=4,
    )

    item = dataset[0]

    assert tuple(item["images"].shape) == (6, 4, 8, 8)
    assert tuple(item["slice_mask"].shape) == (6, 4)


def test_fusion_dataset_returns_independent_dino_and_sam_views(tmp_path: Path):
    write_study(tmp_path / "s.npz", "fp")
    dataset = StudyDataset(
        [StudyRecord("s")], tmp_path, "fp", branch="fusion", training=False,
        knee_slices=3, sam_slices=2,
    )

    item = dataset[0]

    assert tuple(item["images"].shape) == (6, 3, 8, 8)
    assert tuple(item["slice_mask"].shape) == (6, 3)
    assert tuple(item["sam_images"].shape) == (6, 2, 8, 8)
    assert tuple(item["sam_slice_mask"].shape) == (6, 2)


def test_dataset_rejects_study_without_usable_imaging(tmp_path: Path):
    save_cached_study(
        tmp_path / "empty.npz",
        "empty",
        np.zeros((6, 5, 8, 8), dtype=np.uint8),
        np.zeros(6, dtype=bool),
        np.zeros((6, 5), dtype=bool),
        "fp",
    )
    dataset = StudyDataset(
        [StudyRecord("empty")],
        tmp_path,
        "fp",
        branch="knee",
        training=False,
        knee_slices=3,
        sam_slices=4,
    )

    with pytest.raises(ValueError, match="empty.*no usable"):
        dataset[0]
