import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from Helpers.cache import cache_fingerprint, save_cached_study
from TrainEnsemble import run_pipeline, run_validate
from Training.config import default_config, validate_config
from Training.constants import CACHE_SCHEMA_VERSION, LABELS
from Validators.test_validator import validate_submission
from Validators.validator import ValidationError


class TinyKnee(torch.nn.Module):
    def __init__(self, n_targets):
        super().__init__()
        self.head = torch.nn.Linear(1, n_targets)

    def forward(self, images, slot_mask):
        values = images.float().mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1) / 255.0
        return self.head(values)


class TinySAM(torch.nn.Module):
    def __init__(self, n_targets):
        super().__init__()
        self.head = torch.nn.Linear(1, n_targets)

    def forward(self, images, slot_mask, slice_mask):
        values = images.float().mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1) / 255.0
        return self.head(values)


class TinyFusion(torch.nn.Module):
    def __init__(self, n_targets):
        super().__init__()
        self.head = torch.nn.Linear(1, n_targets)

    def forward(self, images, slot_mask, slice_mask, sam_images, sam_slice_mask):
        values = (images.float().mean(dim=(1, 2, 3, 4)) + sam_images.float().mean(dim=(1, 2, 3, 4))).unsqueeze(1) / 510.0
        return self.head(values)


def _manifest(path: Path, split: str, fingerprint: str, studies: int):
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(
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


def _project_config(project: Path):
    config = default_config()
    data = project / "data"
    config["project_root"] = project
    config["config_path"] = project / "config" / "training.yaml"
    config["paths"] = {
        "data_dir": data,
        "cache_dir": data / "cache",
        "train_csv": data / "train.csv",
        "test_csv": data / "test.csv",
        "train_series_csv": data / "train_series.csv",
        "test_series_csv": data / "test_series.csv",
        "sample_submission_csv": data / "sample_submission.csv",
        "labels_csv": data / "labels.csv",
        "sam_base_checkpoint": project / "weights" / "pretrained" / "sam.pth",
        "checkpoints_dir": project / "weights" / "checkpoints",
        "outputs_dir": project / "outputs",
    }
    config["data"].update(
        {
            "n_folds": 2,
            "folds": [0, 1],
            "image_size": 8,
            "max_slices_per_slot": 3,
            "workers": 0,
        }
    )
    config["knee"].update(
        {
            "slices_per_slot": 3,
            "n_slice": 3,
            "epochs": 1,
            "batch_size": 2,
            "lr": 0.01,
            "scheduler": "none",
        }
    )
    config["sam"].update(
        {
            "slices_per_slot": 2,
            "epochs": 1,
            "batch_size": 2,
            "lr": 0.01,
            "scheduler": "none",
        }
    )
    config["fusion"].update(
        {"epochs": 1, "batch_size": 2, "lr": 0.01, "gradient_accumulation": 1, "scheduler": "none"}
    )
    config["checkpoint"]["knee"]["resume"] = "fresh"
    config["checkpoint"]["knee"]["initialize"] = "none"
    config["checkpoint"]["sam"]["resume"] = "fresh"
    config["checkpoint"]["sam"]["initialize"] = "none"
    config["checkpoint"]["fusion"].update({"resume": "fresh", "initialize": "none"})
    config["runtime"].update(
        {"device": "cpu", "precision": "float32", "pin_memory": False}
    )
    validate_config(config)
    return config


def test_synthetic_pipeline_writes_both_branches_and_submission(tmp_path: Path):
    config = _project_config(tmp_path)
    paths = config["paths"]
    Path(paths["data_dir"]).mkdir(parents=True)
    (Path(paths["data_dir"]) / "train_series").mkdir()
    (Path(paths["data_dir"]) / "test_series").mkdir()
    Path(paths["sam_base_checkpoint"]).parent.mkdir(parents=True)
    Path(paths["sam_base_checkpoint"]).write_bytes(b"test")

    train_uids = ["a", "b", "c", "d"]
    reports = ["r0", "r3", "beta", "gamma"]
    target_values = [0.0, 1.0, 1.0, 0.0]
    train_rows = []
    for uid, report, target in zip(train_uids, reports, target_values):
        row = {"StudyInstanceUID": uid, "Report": report}
        row.update({label: target for label in LABELS})
        train_rows.append(row)
    pd.DataFrame(train_rows).to_csv(paths["train_csv"], index=False)

    test_uids = ["t0", "t1"]
    pd.DataFrame({"StudyInstanceUID": test_uids}).to_csv(paths["test_csv"], index=False)
    pd.DataFrame(
        {
            "StudyInstanceUID": train_uids,
            "SeriesInstanceUID": [f"series-{uid}" for uid in train_uids],
        }
    ).to_csv(paths["train_series_csv"], index=False)
    pd.DataFrame(
        {
            "StudyInstanceUID": test_uids,
            "SeriesInstanceUID": [f"series-{uid}" for uid in test_uids],
        }
    ).to_csv(paths["test_series_csv"], index=False)
    sample = pd.DataFrame({"StudyInstanceUID": test_uids})
    for label in LABELS:
        sample[label] = 0.5
    sample.to_csv(paths["sample_submission_csv"], index=False)

    settings = {
        "image_size": 8,
        "crop_mm": float(config["data"]["crop_mm"]),
        "max_slices_per_slot": 3,
        "slice_band": list(config["data"]["slice_band"]),
    }
    fingerprint = cache_fingerprint(settings)
    for split, uids in (("train", train_uids), ("test", test_uids)):
        cache_dir = Path(paths["cache_dir"]) / split
        _manifest(
            cache_dir,
            split,
            "stale-manifest" if split == "train" else fingerprint,
            len(uids),
        )
        for index, uid in enumerate(uids):
            images = np.full((6, 3, 8, 8), 20 + index * 40, dtype=np.uint8)
            save_cached_study(
                cache_dir / f"{uid}.npz",
                uid,
                images,
                np.ones(6, dtype=bool),
                np.ones((6, 3), dtype=bool),
                fingerprint,
            )

    factories = {
        "fusion": lambda branch_config, n_targets: TinyFusion(n_targets),
    }
    result = run_pipeline(config, model_factories=factories)

    assert result.submission.is_file()
    assert sorted(result.fusion_checkpoints) == [0, 1]
    metrics = json.loads((Path(paths["outputs_dir"]) / "metrics.json").read_text())
    assert "oof" in metrics["fusion"]
    assert not (Path(paths["outputs_dir"]) / "oof_sam.csv").exists()
    assert not (Path(paths["outputs_dir"]) / "oof_blended.csv").exists()
    validate_submission(pd.read_csv(result.submission), test_uids, LABELS)
    run_validate(config, model_factories=factories)

    checkpoint = result.fusion_checkpoints[0]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["model_state"]["head.weight"] = torch.zeros(1, 1)
    torch.save(payload, checkpoint)
    with pytest.raises(ValidationError, match="checkpoint state is incompatible"):
        run_validate(config, model_factories=factories)
