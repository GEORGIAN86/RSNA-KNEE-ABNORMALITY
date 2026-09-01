import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from Training.checkpoints import CheckpointError
from Training.trainer import train_fold, train_global, weighted_bce


def test_weighted_bce_applies_cell_weights():
    logits = torch.zeros(1, 2)
    targets = torch.tensor([[1.0, 0.0]])
    weights = torch.tensor([[2.0, 0.0]])

    loss = weighted_bce(logits, targets, weights)

    assert loss.item() == pytest.approx(0.693147, abs=1e-5)


class TinyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {
            "uid": str(index),
            "images": torch.ones(2),
            "slot_mask": torch.ones(1),
            "targets": torch.tensor([float(index % 2)]),
            "weights": torch.ones(1),
        }


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def forward(self, images, slot_mask):
        return self.layer(images)


class OneClassDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "uid": str(index),
            "images": torch.ones(2),
            "slot_mask": torch.ones(1),
            "targets": torch.tensor([0.0]),
            "weights": torch.ones(1),
        }


def test_train_fold_writes_best_and_last_checkpoints(tmp_path: Path):
    loader = DataLoader(TinyDataset(), batch_size=2)

    result = train_fold(
        TinyModel(),
        "knee",
        0,
        loader,
        loader,
        tmp_path,
        {
            "epochs": 1,
            "lr": 1e-2,
            "weight_decay": 0.0,
            "gradient_accumulation": 1,
            "gradient_clip": 1.0,
            "scheduler": "none",
        },
        torch.device("cpu"),
        ["ACL"],
        "fp",
        "foldfp",
    )

    assert (tmp_path / "fold_0_best.pth").is_file()
    assert (tmp_path / "fold_0_last.pth").is_file()
    assert result.predictions.shape == (4, 1)


def test_resume_rejects_changed_model_configuration(tmp_path: Path):
    loader = DataLoader(TinyDataset(), batch_size=2)
    original = {
        "epochs": 1,
        "lr": 1e-2,
        "weight_decay": 0.0,
        "gradient_accumulation": 1,
        "gradient_clip": 1.0,
        "scheduler": "none",
        "backbone": "tiny-a",
    }
    train_fold(
        TinyModel(),
        "knee",
        0,
        loader,
        loader,
        tmp_path,
        original,
        torch.device("cpu"),
        ["ACL"],
        "fp",
        "foldfp",
    )
    changed = {**original, "epochs": 2, "backbone": "tiny-b"}

    with pytest.raises(CheckpointError, match="architecture_fingerprint"):
        train_fold(
            TinyModel(),
            "knee",
            0,
            loader,
            loader,
            tmp_path,
            changed,
            torch.device("cpu"),
            ["ACL"],
            "fp",
            "foldfp",
            resume_path=tmp_path / "fold_0_last.pth",
        )


def test_single_class_history_uses_standard_json_null(tmp_path: Path):
    train_loader = DataLoader(TinyDataset(), batch_size=2)
    validation_loader = DataLoader(OneClassDataset(), batch_size=2)

    train_fold(
        TinyModel(),
        "knee",
        0,
        train_loader,
        validation_loader,
        tmp_path,
        {
            "epochs": 1,
            "lr": 1e-2,
            "weight_decay": 0.0,
            "gradient_accumulation": 1,
            "gradient_clip": 1.0,
            "scheduler": "none",
        },
        torch.device("cpu"),
        ["ACL"],
        "fp",
        "foldfp",
    )

    text = (tmp_path / "fold_0_history.json").read_text()
    assert "NaN" not in text
    assert json.loads(text)[0]["macro_auc"] is None


def test_train_global_writes_last_and_final_checkpoints(tmp_path: Path):
    loader = DataLoader(TinyDataset(), batch_size=2)

    result = train_global(
        TinyModel(),
        "knee",
        loader,
        tmp_path,
        {
            "epochs": 1,
            "lr": 1e-2,
            "weight_decay": 0.0,
            "gradient_accumulation": 1,
            "gradient_clip": 1.0,
            "scheduler": "none",
            "backbone": "tiny-global",
        },
        torch.device("cpu"),
        ["ACL"],
        "fp",
    )

    assert result.last_path == tmp_path / "global_last.pth"
    assert result.final_path == tmp_path / "global_final.pth"
    assert result.last_path.is_file()
    assert result.final_path.is_file()
    assert result.history[0]["epoch"] == 0
