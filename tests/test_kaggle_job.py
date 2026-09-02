import subprocess
import zipfile
from pathlib import Path

import pytest
import yaml


def _production_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "data_dir": "data",
                    "cache_dir": "data/cache",
                    "train_csv": "data/train.csv",
                    "test_csv": "data/test.csv",
                    "train_series_csv": "data/train_series.csv",
                    "test_series_csv": "data/test_series.csv",
                    "sample_submission_csv": "data/sample_submission.csv",
                    "labels_csv": "data/llm_labels_v4_blend.csv",
                    "sam_base_checkpoint": "weights/pretrained/sam.pth",
                    "external_knee_checkpoint": "weights/checkpoints/knee/m_f{fold}.pt",
                    "external_sam_checkpoint": "weights/checkpoints/sam/model",
                    "checkpoints_dir": "weights/checkpoints",
                    "outputs_dir": "outputs",
                },
                "runtime": {"device": "auto"},
            }
        )
    )


def _competition(root: Path) -> Path:
    root.mkdir()
    for name in (
        "train.csv",
        "test.csv",
        "train_series.csv",
        "test_series.csv",
        "sample_submission.csv",
    ):
        (root / name).write_text("StudyInstanceUID\na\n")
    for split in ("train", "test"):
        directory = root / f"{split}_images" / "study" / "series"
        directory.mkdir(parents=True)
        (directory / "image.dcm").write_bytes(b"dicom")
    return root


def test_prepare_data_root_maps_competition_image_directories(tmp_path: Path):
    from kaggle_job.run_training import prepare_data_root

    competition = _competition(tmp_path / "competition")
    data_root = prepare_data_root(competition, tmp_path / "scratch")

    assert (data_root / "train_series").resolve() == (competition / "train_images").resolve()
    assert (data_root / "test_series").resolve() == (competition / "test_images").resolve()


def test_build_kaggle_config_maps_inputs_scratch_and_artifacts(tmp_path: Path):
    from kaggle_job.run_training import build_kaggle_config

    source = tmp_path / "training.yaml"
    _production_config(source)
    project = tmp_path / "project"
    labels = project / "data/llm_labels_v4_blend.csv"
    labels.parent.mkdir(parents=True)
    labels.write_text("StudyInstanceUID\na\n")
    competition = _competition(tmp_path / "competition")
    scratch = tmp_path / "scratch"
    artifacts = tmp_path / "artifacts"
    data_root = scratch / "data"
    data_root.mkdir(parents=True)

    result = build_kaggle_config(
        source, project, competition, scratch, artifacts, data_root=data_root
    )

    paths = result["paths"]
    assert paths["data_dir"] == str(data_root)
    assert paths["train_csv"] == str(competition / "train.csv")
    assert paths["train_series_csv"] == str(competition / "train_series.csv")
    assert paths["labels_csv"] == str(labels)
    assert paths["cache_dir"] == str(scratch / "cache")
    assert paths["external_knee_checkpoint"] == str(
        scratch / "weights/checkpoints/knee/m_f{fold}.pt"
    )
    assert paths["external_sam_checkpoint"] == str(
        scratch / "weights/checkpoints/sam/submissions_epoch_8_step_11550"
    )
    assert paths["checkpoints_dir"] == str(artifacts / "checkpoints")
    assert paths["outputs_dir"] == str(artifacts / "outputs")
    assert result["runtime"]["device"] == "auto"
    assert yaml.safe_load(source.read_text())["paths"]["data_dir"] == "data"


def test_safe_extract_zip_rejects_parent_traversal(tmp_path: Path):
    from kaggle_job.run_training import safe_extract_zip

    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "bad")

    with pytest.raises(RuntimeError, match="unsafe archive member"):
        safe_extract_zip(archive, tmp_path / "destination")
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_zip_extracts_regular_files(tmp_path: Path):
    from kaggle_job.run_training import safe_extract_zip

    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("project/file.txt", "good")

    destination = tmp_path / "destination"
    safe_extract_zip(archive, destination)

    assert (destination / "project/file.txt").read_text() == "good"


def test_execute_pipeline_trains_before_predicting_and_forwards_folds(tmp_path: Path):
    from kaggle_job.run_training import execute_pipeline

    config = tmp_path / "kaggle_training.yaml"
    config.write_text("paths: {}\n")
    artifacts = tmp_path / "artifacts"
    resolved = artifacts / "outputs/auto_training.yaml"
    submission = artifacts / "outputs/submission.csv"
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        assert kwargs["cwd"] == tmp_path
        assert kwargs["check"] is True
        if "auto" in args:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text("runtime: {}\n")
        else:
            submission.write_text("StudyInstanceUID,ACL\na,0.5\n")

    execute_pipeline(
        python="python",
        project=tmp_path,
        config=config,
        artifacts=artifacts,
        folds="0,2",
        run=run,
    )

    assert calls == [
        ["python", "TrainEnsemble.py", "auto", "--config", str(config), "--folds", "0,2"],
        ["python", "TrainEnsemble.py", "predict", "--config", str(resolved), "--folds", "0,2"],
    ]


def test_execute_pipeline_stops_when_training_fails(tmp_path: Path):
    from kaggle_job.run_training import execute_pipeline

    calls = []

    def run(args, **kwargs):
        calls.append(args)
        raise subprocess.CalledProcessError(7, args)

    with pytest.raises(subprocess.CalledProcessError):
        execute_pipeline(
            python="python",
            project=tmp_path,
            config=tmp_path / "config.yaml",
            artifacts=tmp_path / "artifacts",
            folds=None,
            run=run,
        )
    assert len(calls) == 1


def test_execute_pipeline_requires_resolved_config_before_prediction(tmp_path: Path):
    from kaggle_job.run_training import execute_pipeline

    calls = []
    with pytest.raises(RuntimeError, match="adaptive configuration"):
        execute_pipeline(
            python="python",
            project=tmp_path,
            config=tmp_path / "config.yaml",
            artifacts=tmp_path / "artifacts",
            folds=None,
            run=lambda args, **kwargs: calls.append(args),
        )
    assert len(calls) == 1


def test_execute_pipeline_requires_submission_after_prediction(tmp_path: Path):
    from kaggle_job.run_training import execute_pipeline

    artifacts = tmp_path / "artifacts"

    def run(args, **kwargs):
        if "auto" in args:
            resolved = artifacts / "outputs/auto_training.yaml"
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text("runtime: {}\n")

    with pytest.raises(RuntimeError, match="submission"):
        execute_pipeline(
            python="python",
            project=tmp_path,
            config=tmp_path / "config.yaml",
            artifacts=artifacts,
            folds=None,
            run=run,
        )
