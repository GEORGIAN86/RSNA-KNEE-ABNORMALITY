from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Helpers.cache import (
    CacheError,
    build_study_cache,
    cache_fingerprint,
    load_cached_study,
    save_cached_study,
)
from Helpers.normalization import choose_slot_records


def test_choose_slot_records_prefers_longest_matching_series():
    rows = pd.DataFrame(
        [
            {
                "StudyInstanceUID": "s",
                "SeriesInstanceUID": "short",
                "Anatomical_Plane": "Sagittal",
                "Fat_Suppression": 1,
                "n_slices": 8,
            },
            {
                "StudyInstanceUID": "s",
                "SeriesInstanceUID": "long",
                "Anatomical_Plane": "Sagittal",
                "Fat_Suppression": 1,
                "n_slices": 16,
            },
        ]
    )

    slots = choose_slot_records(rows)

    assert slots[0]["SeriesInstanceUID"] == "long"


def test_cache_round_trip_and_fingerprint(tmp_path: Path):
    images = np.zeros((6, 4, 8, 8), dtype=np.uint8)
    slot_mask = np.array([1, 0, 0, 0, 0, 0], dtype=bool)
    slice_mask = np.zeros((6, 4), dtype=bool)
    slice_mask[0] = True
    fingerprint = cache_fingerprint({"image_size": 8, "crop_mm": 130.0})
    path = tmp_path / "s.npz"

    save_cached_study(path, "s", images, slot_mask, slice_mask, fingerprint)
    study = load_cached_study(path, expected_fingerprint=fingerprint)

    assert study.images.shape == (6, 4, 8, 8)
    with pytest.raises(CacheError, match="fingerprint"):
        load_cached_study(path, expected_fingerprint="wrong")


def test_cache_builder_rejects_study_without_usable_slots(tmp_path: Path):
    series = pd.DataFrame(
        [
            {
                "StudyInstanceUID": "empty",
                "SeriesInstanceUID": "unknown",
                "Anatomical_Plane": None,
                "Fat_Suppression": 0,
                "n_slices": 0,
            }
        ]
    )
    path = tmp_path / "empty.npz"

    with pytest.raises(CacheError, match="empty.*no usable"):
        build_study_cache(
            "empty",
            series,
            tmp_path / "series",
            path,
            {
                "image_size": 8,
                "crop_mm": 130.0,
                "max_slices_per_slot": 4,
                "slice_band": [0.12, 0.88],
            },
            "fp",
        )

    assert not path.exists()
