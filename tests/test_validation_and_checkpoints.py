from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from Helpers.cache import save_cached_study
from Training.checkpoints import (
    CheckpointError,
    architecture_fingerprint,
    load_checkpoint,
    read_checkpoint_metadata,
    save_checkpoint,
)
from Training.config import default_config
from Training.trainer import macro_auc
from Validators.test_validator import validate_predictions
from Validators.validator import (
    validate_cached_studies,
    validate_input_data,
    validate_prediction_artifacts,
)


def test_macro_auc_ignores_single_class_targets():
    target = np.array([[0, 1], [1, 1]])
    prediction = np.array([[0.1, 0.4], [0.9, 0.6]])

    metrics = macro_auc(target, prediction)

    assert metrics["macro_auc"] == pytest.approx(1.0)


def test_macro_auc_thresholds_soft_weak_targets():
    target = np.array([[0.25], [0.75]])
    prediction = np.array([[0.1], [0.9]])

    metrics = macro_auc(target, prediction)

    assert metrics["macro_auc"] == pytest.approx(1.0)


def test_checkpoint_rejects_wrong_branch(tmp_path: Path):
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "fold_0_best.pth"
    save_checkpoint(
        path,
        model,
        None,
        None,
        None,
        metadata={
            "branch": "knee",
            "fold": 0,
            "labels": ["x"],
            "cache_fingerprint": "fp",
        },
    )
    assert read_checkpoint_metadata(path)["branch"] == "knee"

    with pytest.raises(CheckpointError, match="branch"):
        load_checkpoint(path, model, expected={"branch": "sam"})


def test_prediction_validator_rejects_duplicate_uids():
    frame = pd.DataFrame({"StudyInstanceUID": ["a", "a"], "ACL": [0.1, 0.2]})

    with pytest.raises(ValueError, match="duplicate"):
        validate_predictions(frame, expected_uids=["a"], labels=["ACL"])


def test_predict_preflight_does_not_require_sam_base_checkpoint(tmp_path: Path):
    config = default_config()
    data = tmp_path / "data"
    data.mkdir()
    (data / "test_series").mkdir()
    pd.DataFrame({"StudyInstanceUID": ["t"]}).to_csv(data / "test.csv", index=False)
    pd.DataFrame(
        {"StudyInstanceUID": ["t"], "SeriesInstanceUID": ["series"]}
    ).to_csv(data / "test_series.csv", index=False)
    config["paths"] = {
        **config["paths"],
        "data_dir": data,
        "test_csv": data / "test.csv",
        "test_series_csv": data / "test_series.csv",
        "labels_csv": data / "labels.csv",
        "sam_base_checkpoint": tmp_path / "missing-sam.pth",
    }

    report = validate_input_data(config, "predict")

    assert not any("SAM base checkpoint" in error for error in report.errors)


def test_artifact_validator_checks_existing_prediction_files(tmp_path: Path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    pd.DataFrame(
        {"StudyInstanceUID": ["t", "t"], "ACL": [0.2, 0.3]}
    ).to_csv(outputs / "test_knee.csv", index=False)

    report = validate_prediction_artifacts(
        outputs,
        train_uids=["a"],
        test_uids=["t"],
        labels=["ACL"],
    )

    assert any("test_knee.csv" in error and "duplicate" in error for error in report.errors)


def test_architecture_fingerprint_ignores_optimizer_and_batch_settings():
    first = {
        "backbone": "dinov2",
        "pool": "cls_mean",
        "hidden": 256,
        "dropout": 0.2,
        "slices_per_slot": 3,
        "lr": 1e-4,
        "batch_size": 4,
        "epochs": 8,
    }
    second = {**first, "lr": 2e-5, "batch_size": 1, "epochs": 20}

    assert architecture_fingerprint("knee", first) == architecture_fingerprint(
        "knee", second
    )
    assert architecture_fingerprint("knee", first) != architecture_fingerprint(
        "knee", {**second, "backbone": "different"}
    )


def test_cached_study_preflight_rejects_study_without_usable_slots(tmp_path: Path):
    save_cached_study(
        tmp_path / "empty.npz",
        "empty",
        np.zeros((6, 3, 8, 8), dtype=np.uint8),
        np.zeros(6, dtype=bool),
        np.zeros((6, 3), dtype=bool),
        "fp",
    )

    report = validate_cached_studies(tmp_path, ["empty"], "fp")

    assert any("empty" in error and "no usable" in error for error in report.errors)
