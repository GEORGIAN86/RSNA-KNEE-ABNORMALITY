import os
import subprocess
import zipfile
from pathlib import Path


ENTRYPOINT = Path(__file__).parents[1] / "docker" / "entrypoint.sh"


def _fake_tools(tmp_path: Path) -> tuple[Path, Path]:
    tools = tmp_path / "bin"
    tools.mkdir()
    log = tmp_path / "commands.log"
    log.touch()
    for name, body in {
        "kaggle": '''#!/usr/bin/env bash
echo "kaggle $*" >> "$COMMAND_LOG"
[[ -n "${KAGGLE_FAKE_ARCHIVE:-}" ]] || exit 99
destination=""
while [[ $# -gt 0 ]]; do
  [[ "$1" == "-p" ]] && destination="$2" && break
  shift
done
cp "$KAGGLE_FAKE_ARCHIVE" "$destination/competition.zip"
''',
        "nvidia-smi": '#!/usr/bin/env bash\necho "nvidia-smi" >> "$COMMAND_LOG"\n',
        "python": '#!/usr/bin/env bash\necho "python $*" >> "$COMMAND_LOG"\n',
    }.items():
        path = tools / name
        path.write_text(body)
        path.chmod(0o755)
    return tools, log


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    for split in ("train", "test"):
        directory = project / "data" / f"{split}_series" / "study" / "series"
        directory.mkdir(parents=True)
        (directory / "image.dcm").write_bytes(b"dicom")
    for name in ("train.csv", "test.csv", "train_series.csv", "test_series.csv", "sample_submission.csv"):
        (project / "data" / name).write_text("StudyInstanceUID\na\n")
    for fold in range(5):
        path = project / "weights" / "checkpoints" / "knee" / f"m_f{fold}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weight")
    sam = project / "weights" / "checkpoints" / "sam" / "submissions_epoch_8_step_11550"
    sam.mkdir(parents=True)
    (sam / "checkpoint").write_bytes(b"weight")
    return project


def _run(tmp_path: Path, project: Path, **extra: str) -> subprocess.CompletedProcess[str]:
    tools, log = _fake_tools(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{tools}:{os.environ['PATH']}",
        "APP_ROOT": str(project),
        "DATA_DIR": str(project / "data"),
        "PYTHON_BIN": "python",
        "COMMAND_LOG": str(log),
        **extra,
    }
    return subprocess.run(["bash", str(ENTRYPOINT)], env=env, text=True, capture_output=True)


def test_existing_dicoms_skip_kaggle_and_start_auto_training(tmp_path: Path):
    project = _project(tmp_path)

    result = _run(tmp_path, project)

    assert result.returncode == 0, result.stderr
    commands = (tmp_path / "commands.log").read_text()
    assert "kaggle " not in commands
    assert "python TrainEnsemble.py auto --config config/training.yaml" in commands


def test_existing_dicoms_promote_known_duplicate_metadata_names(tmp_path: Path):
    project = _project(tmp_path)
    aliases = {
        "train.csv": "train (1).csv",
        "test.csv": "test (1).csv",
        "train_series.csv": "train_series (1).csv",
        "test_series.csv": "test_series (1).csv",
        "sample_submission.csv": "sample_submission (2).csv",
    }
    for canonical, alias in aliases.items():
        (project / "data" / canonical).rename(project / "data" / alias)

    result = _run(tmp_path, project)

    assert result.returncode == 0, result.stderr
    assert all((project / "data" / name).is_file() for name in aliases)
    assert "kaggle " not in (tmp_path / "commands.log").read_text()


def test_plan_mode_adds_plan_only(tmp_path: Path):
    project = _project(tmp_path)

    result = _run(tmp_path, project, TRAIN_MODE="plan")

    assert result.returncode == 0, result.stderr
    assert "--plan-only" in (tmp_path / "commands.log").read_text()


def test_missing_dicoms_requires_kaggle_credentials(tmp_path: Path):
    project = _project(tmp_path)
    (project / "data" / "test_series" / "study" / "series" / "image.dcm").unlink()

    result = _run(tmp_path, project)

    assert result.returncode != 0
    assert "KAGGLE_API_TOKEN" in result.stderr
    assert "kaggle " not in (tmp_path / "commands.log").read_text()


def test_missing_data_downloads_and_normalizes_competition_archive(tmp_path: Path):
    project = _project(tmp_path)
    for path in (project / "data").glob("*.csv"):
        path.unlink()
    for path in (project / "data").rglob("*.dcm"):
        path.unlink()
    archive = tmp_path / "competition.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name in ("train.csv", "test.csv", "train_series.csv", "test_series.csv", "sample_submission.csv"):
            bundle.writestr(f"release/{name}", "StudyInstanceUID\na\n")
        bundle.writestr("release/train_images/a/s/image.dcm", b"train")
        bundle.writestr("release/test_images/b/s/image.dcm", b"test")

    result = _run(
        tmp_path,
        project,
        KAGGLE_API_TOKEN="secret-token-not-for-output",
        KAGGLE_FAKE_ARCHIVE=str(archive),
    )

    assert result.returncode == 0, result.stderr
    assert (project / "data" / "train.csv").is_file()
    assert next((project / "data" / "train_series").rglob("*.dcm"))
    assert next((project / "data" / "test_series").rglob("*.dcm"))
    assert "rsna-knee-abnormality-detection" in (tmp_path / "commands.log").read_text()
    assert "secret-token-not-for-output" not in result.stdout + result.stderr
