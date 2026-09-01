import numpy as np
import pandas as pd
import pytest

from Training.constants import LABELS
from Training.dataset import build_fold_map, build_targets


def gold_frame():
    rows = []
    for uid, report, value in [("a", "Same report", 1.0), ("b", " same  report ", 0.0)]:
        row = {"StudyInstanceUID": uid, "Report": report}
        row.update({label: value for label in LABELS})
        rows.append(row)
    return pd.DataFrame(rows)


def test_gold_overrides_weak_targets_and_weights():
    gold = gold_frame().iloc[:1]
    weak = pd.DataFrame([{"StudyInstanceUID": "a", **{label: 0.25 for label in LABELS}}])

    targets = build_targets(
        gold,
        weak,
        gold_weight=8.0,
        silent_value=0.25,
        silent_weight=0.05,
    )

    assert np.allclose(targets.targets[0], 1.0)
    assert np.allclose(targets.weights[0], 8.0)


def test_gold_only_mode_discards_incomplete_rows():
    gold = gold_frame()
    gold.loc[1, LABELS[0]] = np.nan

    targets = build_targets(
        gold,
        None,
        gold_weight=8.0,
        silent_value=0.25,
        silent_weight=0.05,
    )

    assert targets.uids == ["a"]
    assert targets.targets.flags.writeable


def test_equivalent_reports_share_a_fold():
    folds = build_fold_map(gold_frame(), ["a", "b"], n_folds=5)

    assert folds["a"] == folds["b"]


def test_fold_map_rejects_uids_missing_from_train_data():
    with pytest.raises(ValueError, match="missing.*train"):
        build_fold_map(gold_frame(), ["a", "missing"], n_folds=5)
