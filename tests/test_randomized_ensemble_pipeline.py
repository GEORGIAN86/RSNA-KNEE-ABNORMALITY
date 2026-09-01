import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Helpers.cache import cache_fingerprint, save_cached_study
from TrainEnsemble import run_predict, run_train
from Training.config import default_config, validate_config
from Training.constants import CACHE_SCHEMA_VERSION, LABELS
from Training.dataset import build_fold_map
from Validators.test_validator import validate_submission


class RandomTinyKnee(torch.nn.Module):
    def __init__(self, n_targets: int):
        super().__init__()
        self.head = torch.nn.Linear(1, n_targets)

    def forward(self, images, slot_mask):
        feature = images.float().mean((1, 2, 3, 4)).unsqueeze(1) / 255.0
        return self.head(feature)


class RandomTinySAM(torch.nn.Module):
    def __init__(self, n_targets: int):
        super().__init__()
        self.head = torch.nn.Linear(1, n_targets)

    def forward(self, images, slot_mask, slice_mask):
        feature = images.float().mean((1, 2, 3, 4)).unsqueeze(1) / 255.0
        return self.head(feature)


class RandomTinyFusion(torch.nn.Module):
    def __init__(self, n_targets: int):
        super().__init__()
        self.head = torch.nn.Linear(1, n_targets)

    def forward(self, images, slot_mask, slice_mask, sam_images, sam_slice_mask):
        feature = (images.float().mean((1, 2, 3, 4)) + sam_images.float().mean((1, 2, 3, 4))).unsqueeze(1) / 510.0
        return self.head(feature)


def _write_manifest(cache_dir: Path, split: str, fingerprint: str, studies: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "fingerprint": fingerprint,
                "split": split,
                "studies": studies,
                "built": studies,
                "reused": 0,
                "failures": [],
            }
        )
    )


def _random_project(project: Path, seed: int = 731):
    generator = np.random.default_rng(seed)
    data = project / "data"
    (data / "train_series").mkdir(parents=True)
    (data / "test_series").mkdir()

    config = default_config()
    config["project_root"] = project
    config["config_path"] = project / "config" / "training.yaml"
    config["paths"] = {
        **config["paths"],
        "data_dir": data,
        "cache_dir": data / "cache",
        "train_csv": data / "train.csv",
        "test_csv": data / "test.csv",
        "train_series_csv": data / "train_series.csv",
        "test_series_csv": data / "test_series.csv",
        "sample_submission_csv": data / "sample_submission.csv",
        "labels_csv": data / "labels.csv",
        "checkpoints_dir": project / "weights" / "checkpoints",
        "outputs_dir": project / "outputs",
    }
    config["data"].update(
        {"n_folds": 2, "folds": [0, 1], "image_size": 8, "max_slices_per_slot": 3, "workers": 0}
    )
    config["knee"].update(
        {"n_slice": 3, "slices_per_slot": 3, "epochs": 1, "batch_size": 2, "lr": 0.01, "scheduler": "none"}
    )
    config["sam"].update(
        {"slices_per_slot": 2, "epochs": 1, "batch_size": 2, "lr": 0.01, "scheduler": "none"}
    )
    config["fusion"].update(
        {"epochs": 1, "batch_size": 2, "lr": 0.01, "gradient_accumulation": 1, "scheduler": "none"}
    )
    for branch in ("knee", "sam"):
        config["checkpoint"][branch].update({"resume": "fresh", "initialize": "none"})
    config["checkpoint"]["fusion"].update({"resume": "fresh", "initialize": "none"})
    config["runtime"].update(
        {"device": "cpu", "seed": seed, "deterministic": True, "precision": "float32", "pin_memory": False}
    )
    validate_config(config)

    train_uids = [f"train-{index}" for index in range(8)]
    reports = [f"random report {index}" for index in range(len(train_uids))]
    train = pd.DataFrame({"StudyInstanceUID": train_uids, "Report": reports})
    folds = build_fold_map(train, train_uids, 2)
    if set(folds.values()) != {0, 1}:
        raise AssertionError("seeded reports did not populate both folds")
    for label_index, label in enumerate(LABELS):
        train[label] = [(index + label_index) % 2 for index in range(len(train))]
    train.to_csv(config["paths"]["train_csv"], index=False)

    test_uids = [f"test-{index}" for index in range(3)]
    pd.DataFrame({"StudyInstanceUID": test_uids}).to_csv(config["paths"]["test_csv"], index=False)
    for split, uids in (("train", train_uids), ("test", test_uids)):
        pd.DataFrame(
            {"StudyInstanceUID": uids, "SeriesInstanceUID": [f"series-{uid}" for uid in uids]}
        ).to_csv(config["paths"][f"{split}_series_csv"], index=False)

    sample = pd.DataFrame({"StudyInstanceUID": test_uids})
    for label in LABELS:
        sample[label] = 0.5
    sample.to_csv(config["paths"]["sample_submission_csv"], index=False)

    settings = {
        "image_size": 8,
        "crop_mm": float(config["data"]["crop_mm"]),
        "max_slices_per_slot": 3,
        "slice_band": list(config["data"]["slice_band"]),
    }
    fingerprint = cache_fingerprint(settings)
    for split, uids in (("train", train_uids), ("test", test_uids)):
        cache_dir = Path(config["paths"]["cache_dir"]) / split
        _write_manifest(cache_dir, split, fingerprint, len(uids))
        for uid in uids:
            images = generator.integers(1, 256, size=(6, 3, 8, 8), dtype=np.uint8)
            save_cached_study(
                cache_dir / f"{uid}.npz",
                uid,
                images,
                np.ones(6, dtype=bool),
                np.ones((6, 3), dtype=bool),
                fingerprint,
            )
    return config, train_uids, test_uids, folds


def test_randomized_training_ensemble_pipeline(tmp_path: Path):
    config, train_uids, test_uids, expected_folds = _random_project(tmp_path)
    factories = {
        "fusion": lambda branch_config, n_targets: RandomTinyFusion(n_targets),
    }

    run_train(config, model_factories=factories)
    predicted = run_predict(config, model_factories=factories)

    checkpoints = Path(config["paths"]["checkpoints_dir"])
    for fold in (0, 1):
        assert (checkpoints / "fusion" / f"fold_{fold}_best.pth").is_file()
        assert (checkpoints / "fusion" / f"fold_{fold}_last.pth").is_file()

    outputs = Path(config["paths"]["outputs_dir"])
    saved_folds = json.loads((outputs / "folds.json").read_text())["folds"]
    assert saved_folds == {uid: expected_folds[uid] for uid in train_uids}

    oof = pd.read_csv(outputs / "oof_fusion.csv", dtype={"StudyInstanceUID": str})
    assert sorted(oof["StudyInstanceUID"]) == sorted(train_uids)
    assert not oof["StudyInstanceUID"].duplicated().any()
    assert np.isfinite(oof.loc[:, LABELS].to_numpy(dtype=float)).all()
    metrics = json.loads((outputs / "metrics.json").read_text())
    assert "oof" in metrics["fusion"]
    assert not (outputs / "oof_sam.csv").exists()
    assert not (outputs / "oof_blended.csv").exists()

    branch = pd.read_csv(predicted.fusion_path, dtype={"StudyInstanceUID": str})
    assert branch["StudyInstanceUID"].tolist() == test_uids
    assert np.isfinite(branch.loc[:, LABELS].to_numpy(dtype=float)).all()
    submission = pd.read_csv(predicted.submission, dtype={"StudyInstanceUID": str})
    validate_submission(submission, test_uids, LABELS)
    values = submission.loc[:, LABELS].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert ((0 <= values) & (values <= 1)).all()
