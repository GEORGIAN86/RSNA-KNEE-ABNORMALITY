import subprocess
import sys
from pathlib import Path

import TrainEnsemble
from Training.config import default_config
from Training.resource_planner import Candidate


def test_import_has_no_training_side_effects():
    assert callable(TrainEnsemble.main)


def test_cli_help_lists_all_commands():
    result = subprocess.run(
        [sys.executable, "TrainEnsemble.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )

    for command in ("preprocess", "train", "validate", "predict", "all", "auto"):
        assert command in result.stdout


def test_auto_without_cuda_writes_report_and_does_not_train(tmp_path: Path):
    config = default_config()
    config["paths"]["outputs_dir"] = tmp_path / "outputs"
    trained = []

    result = TrainEnsemble.run_auto(
        config,
        hardware_fn=lambda: {
            "cuda_available": False,
            "cpu_count": 4,
            "ram_available": 8 * 1024**3,
            "vram_total": 0,
        },
        data_fn=lambda _: {"train": {"studies": 2}},
        train_fn=lambda _: trained.append(True),
    )

    assert result["selected"] is None
    assert result["status"] == "cuda_unavailable"
    assert (tmp_path / "outputs" / "resource_report.json").is_file()
    assert not trained


def test_auto_selects_probe_safe_candidate_and_dispatches_training(tmp_path: Path):
    config = default_config()
    config["paths"]["outputs_dir"] = tmp_path / "outputs"
    candidate = Candidate(1, 256, 1, 2, 0, True)
    trained = []

    result = TrainEnsemble.run_auto(
        config,
        plan_only=False,
        hardware_fn=lambda: {
            "cuda_available": True,
            "cpu_count": 8,
            "ram_available": 16 * 1024**3,
            "vram_total": 16 * 1024**3,
        },
        data_fn=lambda _: {"train": {"studies": 2}},
        candidates=[candidate],
        prepare_fn=lambda _: None,
        probe_fn=lambda resolved: {"success": True, "oom": False, "peak_reserved": 8 * 1024**3},
        train_fn=lambda resolved: trained.append(resolved),
    )

    assert result["selected"]["sam_input_size"] == 256
    assert trained[0]["sam"]["slices_per_slot"] == 1
    assert (tmp_path / "outputs" / "auto_training.yaml").is_file()


def test_legacy_preprocess_entrypoint_uses_unified_config_cli():
    result = subprocess.run(
        [sys.executable, "preprocess.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "--config" in result.stdout
    assert "--comp-dir" not in result.stdout
