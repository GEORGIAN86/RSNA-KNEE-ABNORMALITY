#!/usr/bin/env python3
"""Submit the knee MRI training pipeline to a managed Kaggle GPU job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Mapping, MutableMapping, Sequence


SCHEMA_VERSION = 1
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
SOURCE_DIRECTORIES = ("Helpers", "Loss", "Models", "Training", "Validators", "config")
SOURCE_FILES = (
    "TrainEnsemble.py",
    "preprocess.py",
    "train.py",
    "requirements-kaggle.txt",
    "data/llm_labels_v4_blend.csv",
)
ACTIVE_STATES = {"queued", "pending", "running"}
TERMINAL_SUCCESS = {"complete", "completed"}
TERMINAL_FAILURE = {"error", "failed", "failure", "cancelled", "canceled"}


class LauncherError(RuntimeError):
    """A user-actionable launcher failure."""


@dataclass(frozen=True)
class LaunchOptions:
    competition: str
    source_dataset: str
    weights_dataset: str
    kernel: str
    accelerator: str
    folds: str | None
    poll_seconds: float
    timeout_seconds: float
    output_dir: Path
    no_wait: bool


class KaggleCli:
    """Credential-safe subprocess boundary for the public Kaggle CLI."""

    def __init__(
        self,
        *,
        executable: str,
        environment: Mapping[str, str],
        secrets: Sequence[str],
        execute: Callable | None = None,
    ) -> None:
        self.executable = executable
        self.environment = dict(environment)
        self.secrets = tuple(secrets)
        self.execute = execute or subprocess.run

    def run(self, args: Sequence[str], *, check: bool = True):
        result = self.execute(
            [self.executable, *args],
            env=self.environment,
            text=True,
            capture_output=True,
        )
        if check and result.returncode != 0:
            operation = " ".join(args[:2])
            detail = redact(result.stderr.strip() or result.stdout.strip(), self.secrets)
            raise LauncherError(f"Kaggle CLI operation '{operation}' failed: {detail}")
        return result


def load_dotenv(path: Path, environment: MutableMapping[str, str]) -> None:
    """Load simple dotenv values without replacing process environment values."""
    if not path.is_file():
        return
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise LauncherError(f"invalid .env entry on line {number}")
        name, value = line.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise LauncherError(f"invalid .env variable on line {number}")
        environment.setdefault(name, value.strip().strip("'\""))


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory in SOURCE_DIRECTORIES:
        base = root / directory
        if not base.is_dir():
            raise LauncherError(f"required source directory is missing: {base}")
        paths.extend(
            path
            for path in base.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise LauncherError(f"required source file is missing: {path}")
        paths.append(path)
    return sorted(set(paths), key=lambda path: path.relative_to(root).as_posix())


def build_source_archive(root: Path, destination: Path) -> None:
    """Create a reproducible archive containing only remote runtime source."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in _source_paths(root):
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def validate_weights_archive(path: Path) -> None:
    """Ensure the remote input archive has every required warm-start checkpoint."""
    if not path.is_file():
        raise LauncherError(f"weights archive is missing: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile as exc:
        raise LauncherError(f"invalid weights archive: {path}") from exc

    required = {f"weights/checkpoints/knee/m_f{fold}.pt" for fold in range(5)}
    missing = sorted(required - names)
    sam_prefix = "weights/checkpoints/sam/submissions_epoch_8_step_11550/"
    if not any(name.startswith(sam_prefix) and not name.endswith("/") for name in names):
        missing.append(sam_prefix)
    if missing:
        raise LauncherError("weights archive is missing: " + ", ".join(missing))


def _write_json(path: Path, value: Mapping) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LauncherError(f"invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise LauncherError(f"JSON object required: {path}")
    return value


def stage_dataset(
    *,
    stage: Path,
    slug: str,
    title: str,
    bundle: Path,
    bundle_name: str,
) -> dict:
    """Create a private Kaggle dataset upload directory for one bundle."""
    stage.mkdir(parents=True, exist_ok=False)
    destination = stage / bundle_name
    try:
        os.link(bundle, destination)
    except OSError:
        shutil.copy2(bundle, destination)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "bundle": bundle_name,
        "sha256": sha256_file(bundle),
    }
    _write_json(stage / "manifest.json", manifest)
    _write_json(
        stage / "dataset-metadata.json",
        {
            "title": title,
            "id": slug,
            "licenses": [{"name": "unknown"}],
        },
    )
    return manifest


def _is_not_found(result) -> bool:
    message = f"{result.stdout}\n{result.stderr}".lower()
    return result.returncode != 0 and any(
        marker in message for marker in ("404", "not found", "does not exist")
    )


def sync_dataset(slug: str, stage: Path, cli, probe: Path) -> str:
    """Create, update, or reuse a private dataset based on its manifest."""
    result = cli.run(["datasets", "files", slug, "--page-size", "1"], check=False)
    if _is_not_found(result):
        cli.run(["datasets", "create", "-p", str(stage), "--quiet"])
        return "created"
    if result.returncode != 0:
        raise LauncherError(f"failed to probe private dataset {slug}: {result.stderr.strip()}")

    probe.mkdir(parents=True, exist_ok=True)
    remote_manifest = probe / "manifest.json"
    remote_manifest.unlink(missing_ok=True)
    download = cli.run(
        [
            "datasets",
            "download",
            slug,
            "-f",
            "manifest.json",
            "-p",
            str(probe),
            "--force",
            "--quiet",
        ],
        check=False,
    )
    if download.returncode == 0 and remote_manifest.is_file():
        if _read_json(remote_manifest) == _read_json(stage / "manifest.json"):
            return "reused"
    elif download.returncode != 0 and not _is_not_found(download):
        raise LauncherError(f"failed to read manifest for {slug}: {download.stderr.strip()}")

    cli.run(
        [
            "datasets",
            "version",
            "-p",
            str(stage),
            "--message",
            "Update launcher input bundle",
            "--quiet",
        ]
    )
    return "versioned"


def build_kernel_metadata(
    *,
    kernel: str,
    source_dataset: str,
    weights_dataset: str,
    competition: str,
    accelerator: str,
) -> dict:
    """Build metadata for a private managed GPU script."""
    return {
        "id": kernel,
        "title": kernel.split("/", 1)[1].replace("-", " ").title(),
        "code_file": "run_training.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": accelerator,
        "dataset_sources": [source_dataset, weights_dataset],
        "competition_sources": [competition],
        "kernel_sources": [],
        "model_sources": [],
    }


def parse_kernel_status(text: str) -> str:
    """Extract the normalized state printed by ``kaggle kernels status``."""
    match = re.search(r"(?:kernel\s+)?status\s*:\s*([a-z_]+)", text, re.IGNORECASE)
    if not match:
        raise LauncherError(f"could not parse Kaggle kernel status: {text.strip()}")
    return match.group(1).lower()


def redact(text: str, secrets: Sequence[str]) -> str:
    """Remove credential values from diagnostics."""
    result = text
    for secret in secrets:
        if secret:
            result = result.replace(secret, "***")
    return result


def wait_for_kernel(
    cli,
    kernel: str,
    *,
    poll_seconds: float,
    timeout_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Poll a Kaggle kernel until it reaches a terminal state."""
    started = monotonic()
    while True:
        result = cli.run(["kernels", "status", kernel])
        status = parse_kernel_status(f"{result.stdout}\n{result.stderr}")
        if status in TERMINAL_SUCCESS:
            return status
        if status in TERMINAL_FAILURE:
            raise LauncherError(f"Kaggle kernel {kernel} failed with status {status}")
        if status not in ACTIVE_STATES:
            raise LauncherError(f"Kaggle kernel {kernel} returned unknown status {status}")
        if monotonic() - started >= timeout_seconds:
            raise LauncherError(f"waiting for Kaggle kernel {kernel} timed out")
        sleep(poll_seconds)


def render_job_script(template: Path, destination: Path, config: Mapping) -> None:
    """Embed immutable launch configuration into the single uploaded script."""
    marker = 'JOB_CONFIG_JSON = "{}"'
    source = template.read_text()
    if source.count(marker) != 1:
        raise LauncherError(f"remote runner must contain exactly one config marker: {template}")
    replacement = f"JOB_CONFIG_JSON = {json.dumps(json.dumps(config, sort_keys=True))}"
    destination.write_text(source.replace(marker, replacement))


def _require_nonempty(path: Path, description: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise LauncherError(f"{description} was not created: {path}")


def run_launcher(
    *,
    root: Path,
    options: LaunchOptions,
    cli,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Path | None:
    """Package inputs, submit the kernel, and optionally retrieve its outputs."""
    root = root.resolve()
    if not options.no_wait and options.output_dir.exists():
        raise LauncherError(f"output directory already exists: {options.output_dir}")
    weights_archive = root / "weights.zip"
    validate_weights_archive(weights_archive)

    with tempfile.TemporaryDirectory(prefix="rsna-kaggle-") as temporary:
        staging = Path(temporary)
        source_archive = staging / "source.zip"
        build_source_archive(root, source_archive)
        source_stage = staging / "source-dataset"
        weights_stage = staging / "weights-dataset"
        source_manifest = stage_dataset(
            stage=source_stage,
            slug=options.source_dataset,
            title="RSNA Knee Training Source",
            bundle=source_archive,
            bundle_name="source.zip",
        )
        weights_manifest = stage_dataset(
            stage=weights_stage,
            slug=options.weights_dataset,
            title="RSNA Knee Training Weights",
            bundle=weights_archive,
            bundle_name="weights.zip",
        )

        source_action = sync_dataset(
            options.source_dataset,
            source_stage,
            cli,
            staging / "source-probe",
        )
        weights_action = sync_dataset(
            options.weights_dataset,
            weights_stage,
            cli,
            staging / "weights-probe",
        )
        state_path = root / ".kaggle" / "launcher-state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(
            state_path,
            {
                "source_dataset": options.source_dataset,
                "source_sha256": source_manifest["sha256"],
                "source_action": source_action,
                "weights_dataset": options.weights_dataset,
                "weights_sha256": weights_manifest["sha256"],
                "weights_action": weights_action,
            },
        )

        kernel_stage = staging / "kernel"
        kernel_stage.mkdir()
        render_job_script(
            root / "kaggle_job" / "run_training.py",
            kernel_stage / "run_training.py",
            {
                "competition": options.competition,
                "folds": options.folds,
                "source_dataset": options.source_dataset,
                "weights_dataset": options.weights_dataset,
            },
        )
        _write_json(
            kernel_stage / "kernel-metadata.json",
            build_kernel_metadata(
                kernel=options.kernel,
                source_dataset=options.source_dataset,
                weights_dataset=options.weights_dataset,
                competition=options.competition,
                accelerator=options.accelerator,
            ),
        )
        cli.run(
            [
                "kernels",
                "push",
                "-p",
                str(kernel_stage),
                "--accelerator",
                options.accelerator,
            ]
        )

    if options.no_wait:
        return None

    wait_for_kernel(
        cli,
        options.kernel,
        poll_seconds=options.poll_seconds,
        timeout_seconds=options.timeout_seconds,
        monotonic=monotonic,
        sleep=sleep,
    )
    if options.output_dir.exists():
        raise LauncherError(f"output directory already exists: {options.output_dir}")
    options.output_dir.parent.mkdir(parents=True, exist_ok=True)
    cli.run(
        [
            "kernels",
            "output",
            options.kernel,
            "-p",
            str(options.output_dir),
            "--quiet",
        ]
    )
    _require_nonempty(
        options.output_dir / "artifacts" / "outputs" / "submission.csv",
        "Kaggle submission",
    )
    return options.output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit knee MRI training to a private Kaggle GPU job",
    )
    parser.add_argument(
        "--competition",
        default="rsna-knee-abnormality-detection",
        help="Kaggle competition slug attached to the job",
    )
    parser.add_argument("--source-dataset", help="private owner/dataset slug")
    parser.add_argument("--weights-dataset", help="private owner/dataset slug")
    parser.add_argument("--kernel", help="private owner/kernel slug")
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--folds", help="optional comma-separated folds from 0 to 4")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=86400.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="return after submission instead of polling and downloading output",
    )
    return parser


def _validate_remote_ref(value: str, description: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_-]+", value):
        raise LauncherError(f"invalid {description} slug: {value!r}")
    return value


def _validate_folds(value: str | None) -> str | None:
    if value is None:
        return None
    if not re.fullmatch(r"[0-4](?:,[0-4])*", value):
        raise LauncherError("--folds must be a comma-separated subset of 0,1,2,3,4")
    folds = value.split(",")
    if len(folds) != len(set(folds)):
        raise LauncherError("--folds cannot contain duplicates")
    return value


def resolve_launch_options(
    argv: Sequence[str],
    *,
    root: Path,
    environment: Mapping[str, str],
    now: datetime | None = None,
) -> LaunchOptions:
    """Parse CLI values after dotenv loading and derive private resource slugs."""
    args = build_parser().parse_args(argv)
    token = environment.get("KAGGLE_API_TOKEN", "").strip()
    if not token:
        raise LauncherError("KAGGLE_API_TOKEN is required in .env or the environment")
    username = environment.get("KAGGLE_USERNAME", "").strip()
    if not username:
        raise LauncherError("KAGGLE_USERNAME is required in .env or the environment")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", username):
        raise LauncherError("KAGGLE_USERNAME must be the account URL slug")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.competition):
        raise LauncherError(f"invalid competition slug: {args.competition!r}")
    if args.poll_seconds < 0:
        raise LauncherError("--poll-seconds cannot be negative")
    if args.timeout_seconds <= 0:
        raise LauncherError("--timeout-seconds must be positive")

    source_dataset = args.source_dataset or f"{username}/rsna-knee-training-source"
    weights_dataset = args.weights_dataset or f"{username}/rsna-knee-training-weights"
    kernel = args.kernel or f"{username}/rsna-knee-training"
    timestamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path("outputs") / "kaggle" / timestamp
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    return LaunchOptions(
        competition=args.competition,
        source_dataset=_validate_remote_ref(source_dataset, "source dataset"),
        weights_dataset=_validate_remote_ref(weights_dataset, "weights dataset"),
        kernel=_validate_remote_ref(kernel, "kernel"),
        accelerator=args.accelerator,
        folds=_validate_folds(args.folds),
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
        output_dir=output_dir,
        no_wait=args.no_wait,
    )


def find_kaggle_executable(environment: Mapping[str, str]) -> str:
    executable = shutil.which("kaggle", path=environment.get("PATH"))
    if executable:
        return executable
    sibling = Path(sys.executable).with_name("kaggle")
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    raise LauncherError(
        "Kaggle CLI is not installed; run: python -m pip install 'kaggle>=2,<3'"
    )


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    load_dotenv(root / ".env", environment)
    secrets = [environment.get("KAGGLE_API_TOKEN", ""), environment.get("KAGGLE_KEY", "")]
    try:
        options = resolve_launch_options(
            list(argv) if argv is not None else sys.argv[1:],
            root=root,
            environment=environment,
        )
        executable = find_kaggle_executable(environment)
        cli = KaggleCli(
            executable=executable,
            environment=environment,
            secrets=secrets,
        )
        print(f"Submitting private Kaggle GPU job: {options.kernel}")
        output = run_launcher(root=root, options=options, cli=cli)
        if output is None:
            print(f"Submitted: https://www.kaggle.com/code/{options.kernel}")
        else:
            print(f"Downloaded Kaggle artifacts to {output}")
        return 0
    except LauncherError as exc:
        message = redact(str(exc), secrets)
        print(f"ERROR: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
