import ast
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
import zipfile

import pytest


def completed(returncode: int = 0, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class FakeCli:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def run(self, args, *, check=True):
        self.calls.append(list(args))
        return self.responses.pop(0)


def test_load_dotenv_does_not_override_existing_environment(tmp_path: Path):
    from scripts.kaggle_train import load_dotenv

    env_file = tmp_path / ".env"
    env_file.write_text("KAGGLE_API_TOKEN=file-token\nKAGGLE_USERNAME=file-user\n")
    env = {"KAGGLE_API_TOKEN": "process-token"}

    load_dotenv(env_file, env)

    assert env == {
        "KAGGLE_API_TOKEN": "process-token",
        "KAGGLE_USERNAME": "file-user",
    }


def test_source_archive_is_deterministic_and_excludes_runtime_data(tmp_path: Path):
    from scripts.kaggle_train import build_source_archive, sha256_file

    root = tmp_path / "project"
    for relative, content in {
        "Helpers/cache.py": "CACHE = True\n",
        "Helpers/__pycache__/cache.pyc": "compiled",
        "Models/model.py": "MODEL = True\n",
        "Loss/weighted_bce.py": "LOSS = True\n",
        "Training/train.py": "TRAIN = True\n",
        "Validators/validator.py": "VALID = True\n",
        "config/training.yaml": "runtime: {}\n",
        "TrainEnsemble.py": "print('train')\n",
        "preprocess.py": "print('preprocess')\n",
        "train.py": "print('compat')\n",
        "requirements-kaggle.txt": "pydicom\n",
        "data/llm_labels_v4_blend.csv": "StudyInstanceUID\na\n",
        ".env": "KAGGLE_API_TOKEN=secret\n",
        "weights/checkpoint.pt": "weight",
        "outputs/result.csv": "result",
        "data/train.csv": "private raw data",
        "tests/test_unused.py": "assert True\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    build_source_archive(root, first)
    build_source_archive(root, second)

    assert sha256_file(first) == sha256_file(second)
    with zipfile.ZipFile(first) as archive:
        names = set(archive.namelist())
    assert "Helpers/cache.py" in names
    assert "Loss/weighted_bce.py" in names
    assert "data/llm_labels_v4_blend.csv" in names
    assert "Helpers/__pycache__/cache.pyc" not in names
    assert ".env" not in names
    assert "weights/checkpoint.pt" not in names
    assert "outputs/result.csv" not in names
    assert "data/train.csv" not in names
    assert "tests/test_unused.py" not in names


def test_weight_archive_requires_every_external_checkpoint(tmp_path: Path):
    from scripts.kaggle_train import LauncherError, validate_weights_archive

    archive_path = tmp_path / "weights.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for fold in range(4):
            archive.writestr(f"weights/checkpoints/knee/m_f{fold}.pt", b"weight")
        archive.writestr(
            "weights/checkpoints/sam/submissions_epoch_8_step_11550/data.pkl",
            b"sam",
        )

    with pytest.raises(LauncherError, match="m_f4.pt"):
        validate_weights_archive(archive_path)


def test_weight_archive_accepts_all_required_checkpoints(tmp_path: Path):
    from scripts.kaggle_train import validate_weights_archive

    archive_path = tmp_path / "weights.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for fold in range(5):
            archive.writestr(f"weights/checkpoints/knee/m_f{fold}.pt", b"weight")
        archive.writestr(
            "weights/checkpoints/sam/submissions_epoch_8_step_11550/data.pkl",
            b"sam",
        )

    validate_weights_archive(archive_path)


def test_stage_dataset_writes_private_metadata_and_content_manifest(tmp_path: Path):
    from scripts.kaggle_train import stage_dataset

    bundle = tmp_path / "source.zip"
    bundle.write_bytes(b"source")
    stage = tmp_path / "stage"

    manifest = stage_dataset(
        stage=stage,
        slug="owner/private-source",
        title="RSNA Knee Training Source",
        bundle=bundle,
        bundle_name="source.zip",
    )

    metadata = json.loads((stage / "dataset-metadata.json").read_text())
    assert metadata == {
        "id": "owner/private-source",
        "licenses": [{"name": "unknown"}],
        "title": "RSNA Knee Training Source",
    }
    assert json.loads((stage / "manifest.json").read_text()) == manifest
    assert manifest["bundle"] == "source.zip"
    assert manifest["schema_version"] == 1
    assert (stage / "source.zip").read_bytes() == b"source"


def test_sync_dataset_creates_missing_private_dataset(tmp_path: Path):
    from scripts.kaggle_train import sync_dataset

    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "manifest.json").write_text(json.dumps({"sha256": "abc"}))
    cli = FakeCli([completed(1, stderr="404 - Not Found"), completed()])

    action = sync_dataset("owner/private-source", stage, cli, tmp_path / "probe")

    assert action == "created"
    assert cli.calls[-1] == ["datasets", "create", "-p", str(stage), "--quiet"]


def test_sync_dataset_reuses_matching_remote_manifest(tmp_path: Path):
    from scripts.kaggle_train import sync_dataset

    stage = tmp_path / "stage"
    stage.mkdir()
    manifest = {"schema_version": 1, "sha256": "abc"}
    (stage / "manifest.json").write_text(json.dumps(manifest))
    probe = tmp_path / "probe"

    class ManifestCli:
        def run(self, args, *, check=True):
            if args[:2] == ["datasets", "download"]:
                probe.mkdir(parents=True, exist_ok=True)
                (probe / "manifest.json").write_text(json.dumps(manifest))
            return completed()

    assert sync_dataset("owner/private-source", stage, ManifestCli(), probe) == "reused"


def test_sync_dataset_versions_changed_content(tmp_path: Path):
    from scripts.kaggle_train import sync_dataset

    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "manifest.json").write_text(json.dumps({"sha256": "new"}))
    probe = tmp_path / "probe"
    calls = []

    class ManifestCli:
        def run(self, args, *, check=True):
            calls.append(args)
            if args[:2] == ["datasets", "download"]:
                probe.mkdir(parents=True, exist_ok=True)
                (probe / "manifest.json").write_text(json.dumps({"sha256": "old"}))
            return completed()

    action = sync_dataset("owner/private-source", stage, ManifestCli(), probe)

    assert action == "versioned"
    assert calls[-1][:2] == ["datasets", "version"]


def test_dataset_probe_does_not_hide_authentication_failure(tmp_path: Path):
    from scripts.kaggle_train import LauncherError, sync_dataset

    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "manifest.json").write_text("{}")
    cli = FakeCli([completed(1, stderr="401 Unauthorized")])

    with pytest.raises(LauncherError, match="probe private dataset"):
        sync_dataset("owner/private-source", stage, cli, tmp_path / "probe")


def test_kernel_metadata_is_private_t4_job_with_expected_sources():
    from scripts.kaggle_train import build_kernel_metadata

    metadata = build_kernel_metadata(
        kernel="owner/rsna-knee-training",
        source_dataset="owner/source",
        weights_dataset="owner/weights",
        competition="rsna-knee-abnormality-detection",
        accelerator="NvidiaTeslaT4",
    )

    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == ["owner/source", "owner/weights"]
    assert metadata["competition_sources"] == ["rsna-knee-abnormality-detection"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Kernel status: queued", "queued"),
        ("Kernel status: running", "running"),
        ("Kernel status: complete", "complete"),
        ("status: error", "error"),
    ],
)
def test_parse_kernel_status(text: str, expected: str):
    from scripts.kaggle_train import parse_kernel_status

    assert parse_kernel_status(text) == expected


def test_redact_replaces_every_nonempty_secret():
    from scripts.kaggle_train import redact

    assert redact("token=abc user=name", ["abc", ""]) == "token=*** user=name"


class StatusCli:
    def __init__(self, statuses):
        self.statuses = iter(statuses)
        self.calls = []

    def run(self, args, *, check=True):
        self.calls.append(args)
        status = next(self.statuses)
        return completed(stdout=f"Kernel status: {status}\n")


class Clock:
    def __init__(self, *values: float):
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


def test_wait_for_kernel_polls_until_complete():
    from scripts.kaggle_train import wait_for_kernel

    cli = StatusCli(["queued", "running", "complete"])

    assert wait_for_kernel(
        cli,
        "owner/job",
        poll_seconds=0,
        timeout_seconds=10,
        monotonic=Clock(0, 1, 2),
        sleep=lambda _: None,
    ) == "complete"
    assert len(cli.calls) == 3


def test_wait_for_kernel_rejects_failed_job():
    from scripts.kaggle_train import LauncherError, wait_for_kernel

    with pytest.raises(LauncherError, match="failed with status error"):
        wait_for_kernel(
            StatusCli(["error"]),
            "owner/job",
            poll_seconds=0,
            timeout_seconds=10,
            monotonic=Clock(0),
            sleep=lambda _: None,
        )


def test_wait_for_kernel_times_out_without_cancelling():
    from scripts.kaggle_train import LauncherError, wait_for_kernel

    cli = StatusCli(["running"])
    with pytest.raises(LauncherError, match="timed out"):
        wait_for_kernel(
            cli,
            "owner/job",
            poll_seconds=0,
            timeout_seconds=1,
            monotonic=Clock(0, 2),
            sleep=lambda _: None,
        )
    assert not any("delete" in call for call in cli.calls)


def test_wait_for_kernel_rejects_unknown_status():
    from scripts.kaggle_train import LauncherError, wait_for_kernel

    with pytest.raises(LauncherError, match="unknown status paused"):
        wait_for_kernel(
            StatusCli(["paused"]),
            "owner/job",
            poll_seconds=0,
            timeout_seconds=10,
            monotonic=Clock(0),
            sleep=lambda _: None,
        )


def test_kaggle_cli_redacts_secrets_from_failures():
    from scripts.kaggle_train import KaggleCli, LauncherError

    def execute(command, **kwargs):
        assert "private-token" not in command
        assert kwargs["env"]["KAGGLE_API_TOKEN"] == "private-token"
        return completed(1, stderr="request rejected for private-token")

    cli = KaggleCli(
        executable="kaggle",
        environment={"KAGGLE_API_TOKEN": "private-token"},
        secrets=["private-token"],
        execute=execute,
    )

    with pytest.raises(LauncherError) as error:
        cli.run(["kernels", "status", "owner/job"])
    assert "private-token" not in str(error.value)
    assert "***" in str(error.value)


def _launcher_project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    for relative, content in {
        "Helpers/cache.py": "CACHE = True\n",
        "Loss/weighted_bce.py": "LOSS = True\n",
        "Models/model.py": "MODEL = True\n",
        "Training/train.py": "TRAIN = True\n",
        "Validators/validator.py": "VALID = True\n",
        "config/training.yaml": "runtime: {}\n",
        "TrainEnsemble.py": "print('train')\n",
        "preprocess.py": "print('preprocess')\n",
        "train.py": "print('compat')\n",
        "requirements-kaggle.txt": "pydicom\n",
        "data/llm_labels_v4_blend.csv": "StudyInstanceUID\na\n",
        "kaggle_job/run_training.py": 'JOB_CONFIG_JSON = "{}"\n',
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    with zipfile.ZipFile(root / "weights.zip", "w") as archive:
        for fold in range(5):
            archive.writestr(f"weights/checkpoints/knee/m_f{fold}.pt", b"weight")
        archive.writestr(
            "weights/checkpoints/sam/submissions_epoch_8_step_11550/data.pkl",
            b"sam",
        )
    return root


class WorkflowCli:
    def __init__(self, statuses=("queued", "running", "complete")):
        self.statuses = iter(statuses)
        self.operations = []
        self.rendered_job = ""
        self.metadata = {}

    def run(self, args, *, check=True):
        operation = " ".join(args[:2])
        self.operations.append(operation)
        if args[:2] == ["datasets", "files"]:
            return completed(1, stderr="404 Not Found")
        if args[:2] == ["kernels", "push"]:
            stage = Path(args[args.index("-p") + 1])
            self.rendered_job = (stage / "run_training.py").read_text()
            self.metadata = json.loads((stage / "kernel-metadata.json").read_text())
        if args[:2] == ["kernels", "status"]:
            return completed(stdout=f"Kernel status: {next(self.statuses)}\n")
        if args[:2] == ["kernels", "output"]:
            output = Path(args[args.index("-p") + 1])
            submission = output / "artifacts/outputs/submission.csv"
            submission.parent.mkdir(parents=True)
            submission.write_text("StudyInstanceUID,ACL\na,0.5\n")
        return completed()


def test_run_launcher_submits_waits_and_downloads_outputs(tmp_path: Path):
    from scripts.kaggle_train import LaunchOptions, run_launcher

    root = _launcher_project(tmp_path)
    output = tmp_path / "download"
    cli = WorkflowCli()
    options = LaunchOptions(
        competition="rsna-knee-abnormality-detection",
        source_dataset="owner/rsna-source",
        weights_dataset="owner/rsna-weights",
        kernel="owner/rsna-training",
        accelerator="NvidiaTeslaT4",
        folds="0,2",
        poll_seconds=0,
        timeout_seconds=10,
        output_dir=output,
        no_wait=False,
    )

    result = run_launcher(
        root=root,
        options=options,
        cli=cli,
        monotonic=Clock(0, 1, 2),
        sleep=lambda _: None,
    )

    assert result == output
    assert cli.operations == [
        "datasets files",
        "datasets create",
        "datasets files",
        "datasets create",
        "kernels push",
        "kernels status",
        "kernels status",
        "kernels status",
        "kernels output",
    ]
    embedded = ast.literal_eval(cli.rendered_job.split("=", 1)[1].strip())
    assert json.loads(embedded)["folds"] == "0,2"
    assert cli.metadata["machine_shape"] == "NvidiaTeslaT4"
    assert (output / "artifacts/outputs/submission.csv").is_file()
    state = json.loads((root / ".kaggle/launcher-state.json").read_text())
    assert state["source_dataset"] == "owner/rsna-source"


def test_run_launcher_no_wait_stops_after_submission(tmp_path: Path):
    from scripts.kaggle_train import LaunchOptions, run_launcher

    root = _launcher_project(tmp_path)
    output = tmp_path / "download"
    cli = WorkflowCli()
    options = LaunchOptions(
        competition="competition",
        source_dataset="owner/source",
        weights_dataset="owner/weights",
        kernel="owner/training",
        accelerator="NvidiaTeslaT4",
        folds=None,
        poll_seconds=30,
        timeout_seconds=10,
        output_dir=output,
        no_wait=True,
    )

    assert run_launcher(root=root, options=options, cli=cli) is None
    assert cli.operations[-1] == "kernels push"
    assert not output.exists()


def test_run_launcher_refuses_to_overwrite_output_directory(tmp_path: Path):
    from scripts.kaggle_train import LaunchOptions, LauncherError, run_launcher

    root = _launcher_project(tmp_path)
    output = tmp_path / "download"
    output.mkdir()
    cli = WorkflowCli(statuses=("complete",))
    options = LaunchOptions(
        competition="competition",
        source_dataset="owner/source",
        weights_dataset="owner/weights",
        kernel="owner/training",
        accelerator="NvidiaTeslaT4",
        folds=None,
        poll_seconds=0,
        timeout_seconds=10,
        output_dir=output,
        no_wait=False,
    )

    with pytest.raises(LauncherError, match="already exists"):
        run_launcher(root=root, options=options, cli=cli, monotonic=Clock(0))
    assert cli.operations == []


def test_resolve_launch_options_builds_username_defaults_and_timestamped_output(tmp_path: Path):
    from scripts.kaggle_train import resolve_launch_options

    options = resolve_launch_options(
        ["--folds", "0,2"],
        root=tmp_path,
        environment={
            "KAGGLE_API_TOKEN": "token",
            "KAGGLE_USERNAME": "kaggle-user",
        },
        now=datetime(2026, 9, 2, 3, 4, 5, tzinfo=UTC),
    )

    assert options.source_dataset == "kaggle-user/rsna-knee-training-source"
    assert options.weights_dataset == "kaggle-user/rsna-knee-training-weights"
    assert options.kernel == "kaggle-user/rsna-knee-training"
    assert options.folds == "0,2"
    assert options.output_dir == tmp_path / "outputs/kaggle/20260902T030405Z"


def test_resolve_launch_options_requires_token_and_username(tmp_path: Path):
    from scripts.kaggle_train import LauncherError, resolve_launch_options

    with pytest.raises(LauncherError, match="KAGGLE_API_TOKEN"):
        resolve_launch_options([], root=tmp_path, environment={})
    with pytest.raises(LauncherError, match="KAGGLE_USERNAME"):
        resolve_launch_options(
            [], root=tmp_path, environment={"KAGGLE_API_TOKEN": "token"}
        )


@pytest.mark.parametrize("folds", ["", "0,5", "a", "0,,2", "-1"])
def test_resolve_launch_options_rejects_invalid_folds(tmp_path: Path, folds: str):
    from scripts.kaggle_train import LauncherError, resolve_launch_options

    with pytest.raises(LauncherError, match="folds"):
        resolve_launch_options(
            ["--folds", folds],
            root=tmp_path,
            environment={
                "KAGGLE_API_TOKEN": "token",
                "KAGGLE_USERNAME": "owner",
            },
        )


def test_launcher_help_runs_without_credentials():
    root = Path(__file__).parents[1]

    result = subprocess.run(
        [str(root / ".venv/bin/python"), "scripts/kaggle_train.py", "--help"],
        cwd=root,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "Submit" in result.stdout
    assert "--no-wait" in result.stdout


def test_readme_documents_one_command_kaggle_job():
    root = Path(__file__).parents[1]
    readme = (root / "README.md").read_text()
    env_example = (root / ".env.example").read_text()
    gitignore = (root / ".gitignore").read_text().splitlines()

    assert "python scripts/kaggle_train.py" in readme
    assert "KAGGLE_USERNAME=" in env_example
    assert ".kaggle/" in gitignore
    assert "outputs/kaggle/" in gitignore
