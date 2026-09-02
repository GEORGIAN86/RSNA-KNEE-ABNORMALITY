# Kaggle API Training Launcher Design

## Goal

Provide one local command that securely packages the current project, publishes
or reuses private Kaggle datasets for the source and pretrained weights, starts
a private T4 GPU training job, waits for it to finish, and downloads the trained
checkpoints, reports, predictions, and submission.

The default user-facing command is:

```bash
python scripts/kaggle_train.py
```

## Scope

This change adds a Kaggle-hosted execution path around the existing training
pipeline. It does not change DICOM preprocessing, target construction, fold
assignment, model architecture, loss calculation, checkpoint compatibility, or
submission validation.

The launcher automates remote dataset and kernel operations through the public
`kaggle` CLI. It does not expose a Kaggle GPU to the local Docker container and
does not depend on Docker.

## Chosen Approach

A Python launcher will invoke stable public Kaggle CLI commands through
`subprocess`. Compared with a Bash wrapper, Python gives clearer validation,
structured metadata generation, testable status handling, and safer redaction.
Compared with calling undocumented Kaggle client internals, the CLI boundary is
less coupled to implementation details.

Project source and pretrained weights will be separate private datasets. Source
changes therefore upload only a small archive instead of uploading the 941 MB
weights archive again. A content manifest in each dataset lets unchanged
versions be reused.

## Components

### Local launcher

`scripts/kaggle_train.py` owns the local orchestration flow:

1. Resolve the repository root and load simple `KEY=VALUE` entries from `.env`
   without overriding variables already present in the process environment.
2. Require `KAGGLE_API_TOKEN` and `KAGGLE_USERNAME`, validate requested slugs,
   and verify that the `kaggle` executable is available.
3. Build deterministic source and weight dataset staging directories under a
   temporary directory.
4. Create missing private datasets, publish a new version when content changes,
   and reuse the existing version when its manifest matches.
5. Generate a private kernel staging directory containing the remote runner and
   `kernel-metadata.json`.
6. Push the kernel with an NVIDIA T4 accelerator.
7. Unless `--no-wait` is supplied, poll its status until completion, failure, or
   the local wait timeout.
8. Download successful output into a new timestamped directory under
   `outputs/kaggle/`.

The launcher will use these default remote identifiers:

```text
<username>/rsna-knee-training-source
<username>/rsna-knee-training-weights
<username>/rsna-knee-training
```

Command-line overrides will be available for the competition, dataset slugs,
kernel slug, accelerator, selected folds, poll interval, wait timeout, and local
output directory. `--no-wait` will return after Kaggle accepts the job.

### Source dataset

The source dataset will contain `source.zip` and `manifest.json`. The archive
will be assembled from an explicit allowlist:

- `Helpers/`
- `Models/`
- `Training/`
- `Validators/`
- `config/`
- `TrainEnsemble.py`, `preprocess.py`, and `train.py`
- `requirements-kaggle.txt`
- `data/llm_labels_v4_blend.csv`

Tests, Git data, `.env`, raw DICOMs, caches, local outputs, and all weights are
excluded. Zip entry names are repository-relative, sorted, and use stable
timestamps so identical source produces the same SHA-256 digest.

### Weights dataset

The weights dataset will contain the existing `weights.zip` and a manifest with
its SHA-256 digest. Before upload, the launcher will validate that the archive
contains all five `weights/checkpoints/knee/m_f<n>.pt` files and the extracted
SAM checkpoint rooted at
`weights/checkpoints/sam/submissions_epoch_8_step_11550/`.

`weights.zip` is the explicit pretrained-weight source of truth for remote jobs.
Fusion checkpoints produced by training are outputs and are not added back to
this input dataset automatically.

### Dataset reuse

Each staging directory will contain a private `dataset-metadata.json` and a
`manifest.json` recording the bundle name, digest, and launcher schema version.
The launcher will probe the remote dataset and retrieve its small manifest. The
behavior is deterministic:

- Remote dataset absent: create it privately.
- Remote dataset present with the same digest and schema: reuse it.
- Remote dataset present with a different digest or schema: publish a new
  private version.
- Authentication, permission, or network failure: stop without treating the
  failure as a missing dataset.

Successful digests and remote identifiers will also be recorded in the ignored
local `.kaggle/launcher-state.json` as an optimization, never as the sole source
of truth.

### Remote runner

`kaggle_job/run_training.py` will execute inside the Kaggle kernel. It will:

1. Locate the two attached private datasets and the attached competition under
   `/kaggle/input`.
2. Extract project source into `/kaggle/working/project`.
3. Extract pretrained weights and create the preprocessing cache under
   `/kaggle/temp/rsna-knee-training`, so neither is included in downloadable
   kernel output.
4. Install `requirements-kaggle.txt`, which deliberately excludes `torch` and
   `torchvision` so the CUDA-enabled versions supplied by Kaggle remain intact.
5. Verify `torch.cuda.is_available()` before preprocessing or training.
6. Generate `/kaggle/working/project/kaggle_training.yaml` from the production
   config with explicit Kaggle paths:
   - competition CSVs and DICOM trees from the read-only competition input;
   - weak labels from the extracted source;
   - cache and pretrained weights from `/kaggle/temp`;
   - trained checkpoints and reports under
     `/kaggle/working/artifacts`.
7. Run adaptive training with `TrainEnsemble.py auto`, forwarding an optional
   fold override.
8. Run `TrainEnsemble.py predict` with the resolved configuration produced by
   adaptive training.
9. Require non-empty `auto_training.yaml` and `submission.csv` before returning
   success.

Only `/kaggle/working/artifacts` is intended for download. It contains fusion
checkpoints, the resource report, resolved configuration, metrics, fold
predictions, and final submission. Scratch cache and pretrained weights are not
copied into it.

### Kaggle requirements

`requirements-kaggle.txt` will list only packages the project needs on top of
Kaggle's managed Python/CUDA image. It will not pin or install PyTorch or
torchvision. Kernel internet access will be enabled so missing packages can be
installed. The kernel metadata remains private and selects
`NvidiaTeslaT4` by default because the current Kaggle PyTorch image does not
support GPU computation on P100 without replacing PyTorch.

## Metadata and Data Flow

The generated kernel metadata will declare:

- a private Python script kernel;
- GPU enabled with `machine_shape` set to `NvidiaTeslaT4`;
- internet enabled for dependency installation;
- the configured competition source;
- the private source and weights dataset sources;
- no embedded credentials.

The complete flow is:

```text
.env + repository + weights.zip
              |
              v
validate credentials, archive contents, and CLI
              |
              v
create/version/reuse private source and weights datasets
              |
              v
push private T4 kernel
              |
              v
competition input + private datasets -> Kaggle temporary workspace
              |
              v
adaptive train -> predict -> /kaggle/working/artifacts
              |
              v
poll success -> outputs/kaggle/<UTC timestamp>/
```

## Credentials and Privacy

`.env.example` will document both required values:

```dotenv
KAGGLE_API_TOKEN=
KAGGLE_USERNAME=
```

The token is passed only through the local process environment used by the
Kaggle CLI. It is never written to a staging archive, manifest, kernel metadata,
state file, command line, captured diagnostic, or downloaded output. Error
messages will redact exact secret values before being displayed.

All created datasets and the kernel are private by default. The launcher will
not provide a flag that makes them public.

## Status, Failure Handling, and Idempotency

Queued and running kernel states will continue polling at the configured
interval. A completed state triggers output download. Error, cancellation, or
unknown terminal states return a nonzero exit code and print the kernel URL and
last status without downloading stale output. A local wait timeout stops
polling but does not cancel or delete the remote job; the user can rerun with
`--no-wait` or use the displayed kernel identifier to inspect it.

Dataset creation/versioning and kernel push failures stop immediately with the
failed operation named. Temporary staging directories are removed on both
success and failure. Existing remote resources are never deleted.

Every launcher run creates a distinct timestamped local output directory, so a
new download does not overwrite prior checkpoints or predictions. Re-running
the command reuses unchanged input datasets and pushes a new kernel version.

## Testing Strategy

Tests will not contact Kaggle, upload the 941 MB archive, or start a GPU. They
will use temporary repositories and a fake `kaggle` executable while exercising
the real launcher and remote-runner logic.

Required coverage is:

1. `.env` loading preserves existing environment variables and rejects missing
   credentials without printing a token.
2. Source packaging follows the allowlist and excludes `.env`, Git data, raw
   data, outputs, and weights.
3. Deterministic archives and manifests have stable digests.
4. Weight validation rejects missing fold or SAM entries.
5. A missing private dataset is created, an unchanged dataset is reused, and a
   changed dataset receives a new version.
6. Authentication and network errors are not misclassified as missing data.
7. Kernel metadata is private, attaches the expected sources, enables internet,
   and selects T4 GPU execution.
8. Optional folds are forwarded to the remote runner.
9. The launcher polls queued/running states, downloads only after completion,
   uses a distinct output directory, and returns nonzero on error or timeout.
10. `--no-wait` submits without polling or downloading.
11. The remote runner generates correct read-only input, scratch, checkpoint,
    and output paths without modifying the production YAML.
12. Remote command failure or missing required artifacts prevents a successful
    job exit.
13. Focused tests, the complete Pytest suite, Python compilation, and diff
    checks pass.

Live Kaggle submission is not part of automated verification because it changes
remote state, consumes quota, and requires the user's credentials. The final
handoff will provide the exact launch command; running it remains an explicit
user action.

## Documentation

The README will document:

- Kaggle CLI installation and accepted competition rules;
- `.env` configuration with token and username;
- the one-command launch workflow;
- private dataset reuse and the first-run 941 MB upload;
- T4 selection, internet use, GPU quota, and session-duration constraints;
- fold override and asynchronous submission examples;
- timestamped output location and rerun behavior.

## Success Criteria

The implementation is complete when:

- `python scripts/kaggle_train.py` securely creates or reuses the two private
  datasets and starts a private T4 job through the Kaggle CLI.
- An unchanged weights archive is not uploaded again.
- The Kaggle job uses competition data without copying it into a dataset.
- Adaptive training and prediction use writable scratch/checkpoint/output paths
  appropriate for Kaggle.
- The default command waits and downloads successful artifacts into a new local
  directory.
- Failures are actionable, nonzero, and do not disclose credentials.
- No automated test or verification command launches a real remote job.
