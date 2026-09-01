from pathlib import Path

import pytest

from Training.config import ConfigError, load_config
from Training.constants import LABELS, SLOTS


def test_load_config_resolves_paths_from_project_root(tmp_path: Path):
    project = tmp_path / "project"
    cfg_dir = project / "config"
    cfg_dir.mkdir(parents=True)
    path = cfg_dir / "training.yaml"
    path.write_text("paths:\n  data_dir: data\nruntime:\n  seed: 7\n")

    cfg = load_config(path, project_root=project)

    assert cfg["paths"]["data_dir"] == project / "data"
    assert cfg["runtime"]["seed"] == 7
    assert cfg["data"]["n_folds"] == 5


def test_load_config_rejects_invalid_blend_weight(tmp_path: Path):
    path = tmp_path / "training.yaml"
    path.write_text("ensemble:\n  sam_weight: 1.2\n")

    with pytest.raises(ConfigError, match="sam_weight"):
        load_config(path, project_root=tmp_path)


def test_constants_have_six_slots_and_twelve_labels():
    assert len(SLOTS) == 6
    assert len(LABELS) == 12
    assert LABELS[-1] == "Fracture"


def test_knee_model_requires_checkpoint_slice_count(tmp_path: Path):
    path = tmp_path / "training.yaml"
    path.write_text("knee:\n  slices_per_slot: 4\n")

    with pytest.raises(ConfigError, match="knee.slices_per_slot.*knee.n_slice"):
        load_config(path, project_root=tmp_path)
