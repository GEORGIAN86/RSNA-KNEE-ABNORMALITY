# Dual-Model Knee MRI Training and Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a config-driven local/VM pipeline that preprocesses knee MRI data, trains and validates knee-slot and partially fine-tuned SAM classifiers, saves compatible fold checkpoints, and blends their predictions into a validated submission.

**Architecture:** `TrainEnsemble.py` will be a side-effect-free CLI orchestrator over focused helper, model, training, checkpoint, ensemble, and validation modules. Both branches will share a six-slot per-study cache and deterministic fold map while keeping model-specific sampling and optimizers independent. Checkpoint and cache metadata will be validated before reuse so training and prediction cannot silently mix incompatible artifacts.

**Tech Stack:** Python 3.10+, PyTorch, Transformers/DINOv2, Segment Anything, NumPy, pandas, pydicom, OpenCV, PyYAML, scikit-learn, pytest.

**Workspace rule:** Do not initialize Git and do not create commits. The user explicitly requested that this directory remain outside a Git repository; test checkpoints replace commit checkpoints in this plan.

---

## File Map

Create or replace these files with one responsibility each:

- `requirements.txt`: runtime and test dependencies.
- `config/training.yaml`: repository-relative pipeline defaults.
- `Training/config.py`: YAML loading, default merging, validation, and path resolution.
- `Training/constants.py`: target labels and six slot definitions.
- `Training/dataset.py`: targets, deterministic folds, cache sampling, and PyTorch datasets.
- `Training/checkpoints.py`: atomic `.pth` persistence, resume lookup, and compatibility checks.
- `Training/trainer.py`: loss, metrics, epoch loops, branch prediction, and fold training.
- `Training/ensemble.py`: UID alignment, fold averaging, and knee/SAM blending.
- `Helpers/preprocess.py`: DICOM metadata probing and series annotation.
- `Helpers/normalization.py`: anatomical ordering, slot selection, crop/window, and laterality.
- `Helpers/cache.py`: per-study cache creation and fingerprinted manifest.
- `Models/SlotHead.py`: importable label-specific six-slot attention head.
- `Models/Model.py`: importable DINO-style knee classifier and builder.
- `Models/SAMClassifier.py`: SAM study classifier, builder, and partial-freeze policy.
- `Validators/validator.py`: config, data, cache, fold, and checkpoint validation.
- `Validators/test_validator.py`: prediction and submission validation utilities.
- `TrainEnsemble.py`: CLI commands and stage orchestration.
- `train.py`: compatibility wrapper for `TrainEnsemble.py train`.
- `tests/`: unit and synthetic smoke tests for all contracts.

The malformed `Helpers/__init__.py/` directory will be replaced by a normal
`Helpers/__init__.py` package file after its three modules are moved.

### Task 1: Package scaffolding and configuration

**Files:**
- Create: `requirements.txt`
- Create: `config/training.yaml`
- Create: `Training/__init__.py`
- Create: `Training/config.py`
- Create: `Training/constants.py`
- Create: `Models/__init__.py`
- Create: `Validators/__init__.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing configuration tests**

```python
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
```

- [ ] **Step 2: Run the tests and confirm the missing-module failure**

Run: `pytest tests/test_config.py -q`

Expected: collection fails because `Training.config` does not exist.

- [ ] **Step 3: Implement constants and configuration loading**

`Training/constants.py` must export immutable canonical definitions:

```python
LABELS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
)

SLOTS = (
    ("Sagittal", True), ("Sagittal", False),
    ("Coronal", True), ("Coronal", False),
    ("Axial", True), ("Axial", False),
)

CACHE_SCHEMA_VERSION = 1
```

`Training/config.py` must define `ConfigError`, recursively merge the YAML over
defaults, resolve every `paths` value against the supplied project root, and
validate fold counts, image/slice sizes, weights, branch names, and resume modes.
Use this exact public entry point:

```python
def load_config(path: str | Path, project_root: str | Path | None = None) -> dict:
    config_path = Path(path).expanduser().resolve()
    root = Path(project_root).resolve() if project_root else config_path.parent.parent
    raw = yaml.safe_load(config_path.read_text()) or {}
    cfg = deep_merge(default_config(), raw)
    cfg["project_root"] = root
    cfg["config_path"] = config_path
    for key, value in cfg["paths"].items():
        candidate = Path(value).expanduser()
        cfg["paths"][key] = candidate if candidate.is_absolute() else root / candidate
    validate_config(cfg)
    return cfg
```

The default config must contain `paths`, `data`, `knee`, `sam`, `checkpoint`,
`ensemble`, and `runtime`. `config/training.yaml` must expose every supported
field, including `sam.trainable_blocks: 2`, `ensemble.kind: rank`, and
`ensemble.sam_weight: 0.2`.

- [ ] **Step 4: Add dependencies and package exports**

`requirements.txt` must list:

```text
numpy
pandas
torch
torchvision
transformers
segment-anything
pydicom
opencv-python-headless
PyYAML
scikit-learn
tqdm
pytest
```

Each package `__init__.py` must be side-effect free. Export only stable public
objects used by the CLI.

- [ ] **Step 5: Run the configuration tests**

Run: `pytest tests/test_config.py -q`

Expected: `3 passed`.

### Task 2: Label construction and deterministic folds

**Files:**
- Modify: `Training/dataset.py`
- Test: `tests/test_labels_and_folds.py`

- [ ] **Step 1: Write failing target and fold tests**

```python
import numpy as np
import pandas as pd

from Training.constants import LABELS
from Training.dataset import build_fold_map, build_targets


def gold_frame():
    rows = []
    for uid, report, value in [("a", "Same report", 1.0), ("b", " same  report ", 0.0)]:
        row = {"StudyInstanceUID": uid, "Report": report}
        row.update({label: value for label in LABELS})
        rows.append(row)
    return pd.DataFrame(rows)


def test_gold_overrides_weak_targets_and_weights():
    gold = gold_frame().iloc[:1]
    weak = pd.DataFrame([{"StudyInstanceUID": "a", **{label: 0.25 for label in LABELS}}])
    targets = build_targets(gold, weak, gold_weight=8.0, silent_value=0.25, silent_weight=0.05)
    assert np.allclose(targets.targets[0], 1.0)
    assert np.allclose(targets.weights[0], 8.0)


def test_gold_only_mode_discards_incomplete_rows():
    gold = gold_frame()
    gold.loc[1, LABELS[0]] = np.nan
    targets = build_targets(gold, None, gold_weight=8.0, silent_value=0.25, silent_weight=0.05)
    assert targets.uids == ["a"]


def test_equivalent_reports_share_a_fold():
    folds = build_fold_map(gold_frame(), ["a", "b"], n_folds=5)
    assert folds["a"] == folds["b"]
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `pytest tests/test_labels_and_folds.py -q`

Expected: import fails because the dataset functions are absent.

- [ ] **Step 3: Implement target construction**

Define a `TargetTable` dataclass with `uids`, `targets`, `weights`, and
`is_gold`. `build_targets` must validate all label columns, normalize UIDs to
strings, use optional weak rows as the base, assign low confidence to exact
silent-value cells, and overwrite matching rows with complete gold targets and
the configured gold weight. In gold-only mode, keep only complete rows.

```python
@dataclass(frozen=True)
class TargetTable:
    uids: list[str]
    targets: np.ndarray
    weights: np.ndarray
    is_gold: np.ndarray
```

- [ ] **Step 4: Implement stable report-group folds**

Normalize reports with `" ".join(str(report).split()).lower()`, hash them with
MD5, and assign `int(digest[:8], 16) % n_folds`. Missing reports fall back to
the UID. Return and persist a `dict[str, int]`; reject missing requested UIDs.

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_labels_and_folds.py -q`

Expected: `3 passed`.

### Task 3: DICOM normalization and fingerprinted cache

**Files:**
- Replace: `Helpers/preprocess.py`
- Replace: `Helpers/normalization.py`
- Replace: `Helpers/cache.py`
- Replace malformed directory: `Helpers/__init__.py/`
- Create: `Helpers/__init__.py`
- Test: `tests/test_cache.py`

- [ ] **Step 1: Write failing cache and normalization tests**

```python
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Helpers.cache import CacheError, cache_fingerprint, load_cached_study, save_cached_study
from Helpers.normalization import choose_slot_records


def test_choose_slot_records_prefers_longest_matching_series():
    rows = pd.DataFrame([
        {"StudyInstanceUID": "s", "SeriesInstanceUID": "short", "Anatomical_Plane": "Sagittal", "Fat_Suppression": 1, "n_slices": 8},
        {"StudyInstanceUID": "s", "SeriesInstanceUID": "long", "Anatomical_Plane": "Sagittal", "Fat_Suppression": 1, "n_slices": 16},
    ])
    slots = choose_slot_records(rows)
    assert slots[0]["SeriesInstanceUID"] == "long"


def test_cache_round_trip_and_fingerprint(tmp_path: Path):
    images = np.zeros((6, 4, 8, 8), dtype=np.uint8)
    slot_mask = np.array([1, 0, 0, 0, 0, 0], dtype=bool)
    slice_mask = np.zeros((6, 4), dtype=bool)
    slice_mask[0] = True
    fp = cache_fingerprint({"image_size": 8, "crop_mm": 130.0})
    path = tmp_path / "s.npz"
    save_cached_study(path, "s", images, slot_mask, slice_mask, fp)
    study = load_cached_study(path, expected_fingerprint=fp)
    assert study.images.shape == (6, 4, 8, 8)
    with pytest.raises(CacheError, match="fingerprint"):
        load_cached_study(path, expected_fingerprint="wrong")
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_cache.py -q`

Expected: helper imports fail because the current files are notebook fragments
inside a malformed directory.

- [ ] **Step 3: Repair the Helpers package**

Move the content responsibilities out of `Helpers/__init__.py/`, remove that
now-empty malformed directory, and create `Helpers/__init__.py`. Do not delete
any behavior until its replacement function and tests exist.

- [ ] **Step 4: Implement preprocessing utilities**

`Helpers/preprocess.py` must expose `probe_series`, `walk_series`, and
`annotate_series`. It must read only necessary DICOM tags, tolerate unreadable
headers by returning an error field, and normalize plane/fat-suppression columns
when the competition CSV already supplies them.

`Helpers/normalization.py` must expose:

```python
def choose_slot_records(series: pd.DataFrame) -> list[dict | None]:
    chosen: list[dict | None] = []
    for plane, fat_suppressed in SLOTS:
        candidates = series[
            (series["Anatomical_Plane"] == plane)
            & (series["Fat_Suppression"].astype(bool) == fat_suppressed)
        ]
        chosen.append(None if candidates.empty else candidates.sort_values("n_slices").iloc[-1].to_dict())
    return chosen
```

Also implement geometry-first slice ordering with `InstanceNumber` and natural
filename fallbacks, physical center cropping, percentile windowing,
MONOCHROME1 correction, and right-knee mirroring for coronal and axial planes.

- [ ] **Step 5: Implement atomic cache storage and building**

`Helpers/cache.py` must define `CachedStudy`, `CacheError`, SHA-256
`cache_fingerprint`, `save_cached_study`, `load_cached_study`,
`build_study_cache`, and `build_cache_split`. Save through a sibling temporary
file and `Path.replace`; validate shapes and dtypes before replacement. Write a
JSON manifest containing schema version, fingerprint, split, counts, and failed
studies.

- [ ] **Step 6: Run cache tests**

Run: `pytest tests/test_cache.py -q`

Expected: `2 passed`.

### Task 4: Shared branch datasets and sampling

**Files:**
- Modify: `Training/dataset.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Write failing dataset contract tests**

```python
from pathlib import Path

import numpy as np

from Helpers.cache import save_cached_study
from Training.dataset import StudyDataset, StudyRecord


def write_study(path: Path, fingerprint: str):
    images = np.arange(6 * 5 * 8 * 8, dtype=np.uint32).reshape(6, 5, 8, 8).astype(np.uint8)
    save_cached_study(path, "s", images, np.ones(6, bool), np.ones((6, 5), bool), fingerprint)


def test_knee_dataset_returns_six_three_channel_slots(tmp_path: Path):
    write_study(tmp_path / "s.npz", "fp")
    ds = StudyDataset([StudyRecord("s")], tmp_path, "fp", branch="knee", training=False, knee_slices=3, sam_slices=4)
    item = ds[0]
    assert tuple(item["images"].shape) == (6, 3, 8, 8)
    assert tuple(item["slot_mask"].shape) == (6,)


def test_sam_dataset_returns_slice_mask(tmp_path: Path):
    write_study(tmp_path / "s.npz", "fp")
    ds = StudyDataset([StudyRecord("s")], tmp_path, "fp", branch="sam", training=False, knee_slices=3, sam_slices=4)
    item = ds[0]
    assert tuple(item["images"].shape) == (6, 4, 8, 8)
    assert tuple(item["slice_mask"].shape) == (6, 4)
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `pytest tests/test_dataset.py -q`

Expected: imports or constructors fail.

- [ ] **Step 3: Implement records and deterministic sampling**

Define `StudyRecord(uid, targets=None, weights=None, fold=None)` and
`StudyDataset`. Knee evaluation indices are three evenly spaced valid indices;
SAM evaluation indices are `sam_slices` evenly spaced valid indices. Training
uses the worker-seeded NumPy generator to choose sorted indices, with replacement
only when a valid slot has fewer source slices than requested. Preserve masks
for padded slots and slices.

The returned mapping contains `uid`, `images`, `slot_mask`, `slice_mask`, and,
when supplied, `targets`, `weights`, and `fold` tensors.

- [ ] **Step 4: Run dataset tests**

Run: `pytest tests/test_dataset.py -q`

Expected: `2 passed`.

### Task 5: Canonical knee model and slot attention head

**Files:**
- Replace: `Models/SlotHead.py`
- Replace: `Models/Model.py`
- Test: `tests/test_knee_model.py`

- [ ] **Step 1: Write failing shape and masking tests**

```python
from types import SimpleNamespace

import torch
import torch.nn as nn

from Models.Model import Model
from Models.SlotHead import SlotHead


class FakeBackbone(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=dim)
        self.proj = nn.Linear(3, dim)

    def forward(self, pixel_values):
        pooled = pixel_values.mean((-2, -1))
        token = self.proj(pooled)
        return SimpleNamespace(last_hidden_state=torch.stack([token, token], dim=1))


def test_knee_model_returns_twelve_logits():
    model = Model(FakeBackbone(), feature_dim=8, n_slots=6, n_targets=12, pool="cls_mean")
    logits = model(torch.zeros(2, 6, 3, 8, 8), torch.ones(2, 6))
    assert tuple(logits.shape) == (2, 12)


def test_slot_head_handles_missing_slots_without_nan():
    head = SlotHead(16, n_slots=6, n_targets=12)
    mask = torch.tensor([[1, 0, 1, 0, 0, 0]], dtype=torch.float32)
    logits = head(torch.randn(1, 6, 16), mask)
    assert torch.isfinite(logits).all()
```

- [ ] **Step 2: Run tests and confirm current fragment failures**

Run: `pytest tests/test_knee_model.py -q`

Expected: import fails because current modules depend on undefined notebook
globals.

- [ ] **Step 3: Implement `SlotHead` with explicit dependencies**

Accept `feature_dim`, `n_slots`, `n_targets`, `hidden`, `dropout`, and optional
`slot_prior`. Validate input shape and require at least one valid slot per study.
Use a projected slot representation, learned slot embeddings, target queries,
masked target-to-slot attention, dropout, and target-specific linear weights.

- [ ] **Step 4: Implement the knee `Model` and builder**

`Model.forward(images, slot_mask, image_size=None)` must validate
`[B, 6, 3, H, W]`, normalize using ImageNet buffers, call the injected
backbone with `pixel_values`, pool CLS and mean-patch tokens, and call
`SlotHead`. Support `cls_mean` and `cls_mean_focal` only.

Expose a lazy builder so importing the module does not require downloading a
backbone:

```python
def build_knee_model(config: dict) -> Model:
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(
        config["backbone"],
        local_files_only=bool(config.get("local_files_only", False)),
    )
    return Model(
        backbone=backbone,
        feature_dim=backbone.config.hidden_size,
        n_slots=6,
        n_targets=12,
        pool=config["pool"],
        hidden=config["hidden"],
        dropout=config["dropout"],
    )
```

- [ ] **Step 5: Run knee model tests**

Run: `pytest tests/test_knee_model.py -q`

Expected: `2 passed`.

### Task 6: SAM classifier and partial fine-tuning

**Files:**
- Create: `Models/SAMClassifier.py`
- Test: `tests/test_sam_model.py`

- [ ] **Step 1: Write failing SAM aggregation and freezing tests**

```python
import torch
import torch.nn as nn

from Models.SAMClassifier import SAMClassifier, configure_sam_trainable


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block(), Block()])
        self.neck = nn.Conv2d(1, 4, 1)

    def forward(self, x):
        return self.neck(x[:, :1])


def test_partial_freeze_unfreezes_only_final_blocks_and_head():
    model = SAMClassifier(FakeEncoder(), feature_dim=4, n_targets=12, input_size=8)
    configure_sam_trainable(model, trainable_blocks=1)
    assert not model.image_encoder.blocks[0].weight.requires_grad
    assert model.image_encoder.blocks[-1].weight.requires_grad
    assert all(parameter.requires_grad for parameter in model.head.parameters())


def test_sam_forward_ignores_padded_slices():
    model = SAMClassifier(FakeEncoder(), feature_dim=4, n_targets=12, input_size=8)
    images = torch.zeros(2, 6, 2, 8, 8)
    slot_mask = torch.ones(2, 6, dtype=torch.bool)
    slice_mask = torch.ones(2, 6, 2, dtype=torch.bool)
    slice_mask[:, :, 1] = False
    assert tuple(model(images, slot_mask, slice_mask).shape) == (2, 12)
```

- [ ] **Step 2: Run tests and verify the missing module failure**

Run: `pytest tests/test_sam_model.py -q`

Expected: collection fails because `Models.SAMClassifier` does not exist.

- [ ] **Step 3: Implement SAM classification and aggregation**

The constructor accepts an injected `image_encoder`, its `feature_dim`, target
count, input size, and encode chunk. `forward` selects only slices allowed by
both masks, converts them to three channels, resizes and applies SAM pixel
normalization, encodes in chunks, globally averages feature maps, and uses
`index_add_` to compute one mean feature per study before applying the head.

Support both the real SAM encoder's explicit `patch_embed`/blocks/neck path and
a test encoder's ordinary `forward` path. Reject studies with no valid slices.

- [ ] **Step 4: Implement the freeze policy and lazy builder**

`configure_sam_trainable(model, trainable_blocks)` freezes the encoder first,
then unfreezes all blocks for `-1`, none for `0`, or only the final N blocks for
a positive value. The classifier head is always trainable; the encoder neck is
trainable whenever any encoder blocks are trainable.

`build_sam_model(config, checkpoint_path)` imports `sam_model_registry` lazily,
builds the configured SAM type with its base checkpoint, wraps its image
encoder, applies the freeze policy, and returns `SAMClassifier`.

- [ ] **Step 5: Run SAM tests**

Run: `pytest tests/test_sam_model.py -q`

Expected: `2 passed`.

### Task 7: Metrics, validators, and checkpoint lifecycle

**Files:**
- Replace: `Validators/validator.py`
- Replace: `Validators/test_validator.py`
- Create: `Training/checkpoints.py`
- Modify: `Training/trainer.py`
- Test: `tests/test_validation_and_checkpoints.py`

- [ ] **Step 1: Write failing validation and checkpoint tests**

```python
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from Training.checkpoints import CheckpointError, load_checkpoint, save_checkpoint
from Training.trainer import macro_auc
from Validators.test_validator import validate_predictions


def test_macro_auc_ignores_single_class_targets():
    target = np.array([[0, 1], [1, 1]])
    pred = np.array([[0.1, 0.4], [0.9, 0.6]])
    assert macro_auc(target, pred)["macro_auc"] == pytest.approx(1.0)


def test_checkpoint_rejects_wrong_branch(tmp_path: Path):
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "fold_0_best.pth"
    save_checkpoint(path, model, None, None, None, metadata={"branch": "knee", "fold": 0, "labels": ["x"], "cache_fingerprint": "fp"})
    with pytest.raises(CheckpointError, match="branch"):
        load_checkpoint(path, model, expected={"branch": "sam"})


def test_prediction_validator_rejects_duplicate_uids():
    frame = pd.DataFrame({"StudyInstanceUID": ["a", "a"], "ACL": [0.1, 0.2]})
    with pytest.raises(ValueError, match="duplicate"):
        validate_predictions(frame, expected_uids=["a"], labels=["ACL"])
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_validation_and_checkpoints.py -q`

Expected: imports fail because these interfaces do not exist.

- [ ] **Step 3: Implement macro AUC and validation reports**

`macro_auc` returns a mapping with per-label AUC values and their finite mean.
`Validators/validator.py` defines `ValidationError`, `ValidationReport`,
`validate_input_data`, `validate_folds`, `validate_cache_manifest`, and
`validate_checkpoint_coverage`. Aggregate all detected input problems into one
message where safe; stop before training on any error.

`Validators/test_validator.py` defines `validate_predictions` and
`validate_submission`. Enforce string UIDs, exact unique coverage, exact label
columns/order, numeric finite probabilities, and `[0, 1]` range.

- [ ] **Step 4: Implement atomic checkpoint save, resume, and compatibility**

Write checkpoint dictionaries to a temporary sibling and replace atomically.
`save_checkpoint` stores model and optional optimizer/scheduler/scaler states
plus metadata. `load_checkpoint` compares all supplied expected metadata before
loading the state dict and returns the saved training state. Implement
`resolve_resume_checkpoint(directory, fold, mode, explicit_path)` for `fresh`,
`auto`, and `explicit`.

- [ ] **Step 5: Run validation and checkpoint tests**

Run: `pytest tests/test_validation_and_checkpoints.py -q`

Expected: `3 passed`.

### Task 8: Generic branch trainer and out-of-fold predictions

**Files:**
- Create: `Training/trainer.py`
- Test: `tests/test_trainer.py`

- [ ] **Step 1: Write failing weighted-loss and training smoke tests**

```python
from pathlib import Path

import torch
import pytest
from torch.utils.data import DataLoader, Dataset

from Training.trainer import train_fold, weighted_bce


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


def test_train_fold_writes_best_and_last_checkpoints(tmp_path: Path):
    loader = DataLoader(TinyDataset(), batch_size=2)
    train_fold(TinyModel(), "knee", 0, loader, loader, tmp_path, {"epochs": 1, "lr": 1e-2, "weight_decay": 0.0}, torch.device("cpu"), ["ACL"], "fp", "foldfp")
    assert (tmp_path / "fold_0_best.pth").is_file()
    assert (tmp_path / "fold_0_last.pth").is_file()
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_trainer.py -q`

Expected: trainer functions are absent or incomplete.

- [ ] **Step 3: Implement weighted training and branch dispatch**

`weighted_bce` computes unreduced BCE, multiplies by weights, and divides by the
sum of weights rather than the number of cells. `forward_batch` dispatches knee
to `(images, slot_mask)` and SAM to `(images, slot_mask, slice_mask)`.

Implement `train_one_epoch`, `predict_loader`, and `train_fold` with AdamW,
configured scheduler, CUDA AMP/scaler when enabled, gradient accumulation,
gradient clipping, deterministic validation, best/last checkpoints, and a
history JSON file. On CPU, AMP must be disabled without warnings.

- [ ] **Step 4: Run trainer tests**

Run: `pytest tests/test_trainer.py -q`

Expected: `2 passed` and both synthetic checkpoints exist.

### Task 9: Fold averaging and two-branch ensemble

**Files:**
- Create: `Training/ensemble.py`
- Test: `tests/test_ensemble.py`

- [ ] **Step 1: Write failing alignment and blend tests**

```python
import pandas as pd
import pytest

from Training.ensemble import blend_predictions


def test_probability_blend_aligns_by_uid():
    knee = pd.DataFrame({"StudyInstanceUID": ["a", "b"], "ACL": [0.2, 0.8]})
    sam = pd.DataFrame({"StudyInstanceUID": ["b", "a"], "ACL": [0.4, 0.6]})
    result = blend_predictions(knee, sam, ["ACL"], sam_weight=0.25, kind="probability")
    assert result.loc[result.StudyInstanceUID == "a", "ACL"].item() == pytest.approx(0.3)


def test_blend_rejects_uid_mismatch():
    knee = pd.DataFrame({"StudyInstanceUID": ["a"], "ACL": [0.2]})
    sam = pd.DataFrame({"StudyInstanceUID": ["b"], "ACL": [0.6]})
    with pytest.raises(ValueError, match="UID"):
        blend_predictions(knee, sam, ["ACL"], sam_weight=0.2, kind="rank")
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `pytest tests/test_ensemble.py -q`

Expected: collection fails.

- [ ] **Step 3: Implement UID-safe averaging and blending**

Implement `prediction_frame`, `average_fold_predictions`, and
`blend_predictions`. Normalize UIDs to strings, require unique exact coverage,
align with one-to-one merges, and preserve the left frame order. Rank blending
uses `rank(method="average", pct=True)` independently for every label and
branch. Probability blending uses raw probabilities.

- [ ] **Step 4: Run ensemble tests**

Run: `pytest tests/test_ensemble.py -q`

Expected: `2 passed`.

### Task 10: CLI orchestration and compatibility entry point

**Files:**
- Replace: `TrainEnsemble.py`
- Replace: `train.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing side-effect and parser tests**

```python
import subprocess
import sys

import TrainEnsemble


def test_import_has_no_training_side_effects():
    assert callable(TrainEnsemble.main)


def test_cli_help_lists_all_commands():
    result = subprocess.run([sys.executable, "TrainEnsemble.py", "--help"], capture_output=True, text=True, check=True)
    for command in ("preprocess", "train", "validate", "predict", "all"):
        assert command in result.stdout
```

- [ ] **Step 2: Run CLI tests and verify current monolith failure**

Run: `pytest tests/test_cli.py -q`

Expected: importing the current `TrainEnsemble.py` attempts filesystem/model
work or fails on unavailable hard-coded Kaggle paths.

- [ ] **Step 3: Implement stage functions**

Implement `run_preprocess`, `run_validate`, `run_train`, and `run_predict`.
`run_train` builds the shared target table and fold map, creates branch-specific
datasets/loaders, builds each model lazily, applies branch resume settings,
trains every requested fold, and writes aligned OOF artifacts. `run_predict`
loads every best checkpoint, averages branch test predictions, blends them,
validates the result, and writes all configured CSV files.

Every stage receives the resolved config and logger; no stage reads global
paths. Model weights are never loaded while importing the module.

Also define `PipelineResult` and `run_pipeline(config, model_factories=None)`.
`run_pipeline` is the programmatic equivalent of `all` and returns the final
submission path plus the discovered knee and SAM best-checkpoint fold maps. The
optional factory mapping is the sole test seam and defaults to
`build_knee_model` and `build_sam_model`.

- [ ] **Step 4: Implement the parser and command dispatch**

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and ensemble knee MRI classifiers")
    parser.add_argument("command", choices=("preprocess", "train", "validate", "predict", "all"))
    parser.add_argument("--config", default="config/training.yaml")
    parser.add_argument("--folds", default=None, help="Optional comma-separated fold override")
    return parser
```

`all` executes validate, preprocess, validate, train, predict. `train.py` simply
prepends the `train` command and calls `TrainEnsemble.main`, preserving a familiar
entry point without maintaining a second trainer.

- [ ] **Step 5: Run CLI tests**

Run: `pytest tests/test_cli.py -q`

Expected: `2 passed` and no data/model access during import or `--help`.

### Task 11: Synthetic end-to-end smoke pipeline and documentation

**Files:**
- Create: `tests/test_smoke_pipeline.py`
- Create: `README.md`
- Modify: `config/training.yaml`

- [ ] **Step 1: Write a synthetic smoke test**

Create four tiny cached studies across two folds and inject lightweight knee and
SAM model factories into the orchestration layer. Run one CPU epoch for each
branch, write best and last checkpoints, generate test predictions, blend them,
and validate a two-study submission.

```python
def test_synthetic_pipeline_writes_both_branches_and_submission(synthetic_project):
    result = run_pipeline(
        synthetic_project.config,
        model_factories={"knee": synthetic_project.knee_factory, "sam": synthetic_project.sam_factory},
    )
    assert result.submission.is_file()
    assert sorted(result.knee_checkpoints) == [0, 1]
    assert sorted(result.sam_checkpoints) == [0, 1]
    validate_submission(pd.read_csv(result.submission), synthetic_project.test_uids, ["ACL"])
```

- [ ] **Step 2: Run the smoke test and verify failure**

Run: `pytest tests/test_smoke_pipeline.py -q`

Expected: failure identifies any remaining orchestration contract not yet wired.

- [ ] **Step 3: Complete the injectable orchestration seam**

Add optional `model_factories` arguments defaulting to the production builders.
Ensure runtime label selection permits the smoke fixture's one-label model while
production defaults remain the canonical 12 labels. Do not add a separate code
path for tests; dependency injection must use the same trainer, checkpoints,
prediction, blending, and validators as production.

- [ ] **Step 4: Write usage documentation**

`README.md` must document the exact data and weight directory layout, dependency
installation, YAML fields, expected SAM base checkpoint, each CLI command,
resume modes, output artifacts, and these commands:

```bash
python -m pip install -r requirements.txt
python TrainEnsemble.py validate --config config/training.yaml
python TrainEnsemble.py preprocess --config config/training.yaml
python TrainEnsemble.py train --config config/training.yaml
python TrainEnsemble.py predict --config config/training.yaml
python TrainEnsemble.py all --config config/training.yaml
```

- [ ] **Step 5: Run the complete test suite**

Run: `pytest -q`

Expected: all unit and smoke tests pass.

- [ ] **Step 6: Run static and CLI verification**

Run:

```bash
python -m compileall -q Helpers Models Training Validators TrainEnsemble.py train.py
python TrainEnsemble.py --help
python TrainEnsemble.py validate --config config/training.yaml
```

Expected: compilation and help succeed. Validation exits with an actionable
missing-data report until the user populates `data/`; it must not produce a
traceback for expected missing inputs.

## Final Self-Review Checklist

- Every design-spec component maps to a task above.
- All public function names used by later tasks are defined in earlier tasks.
- Production imports have no data loading, model downloading, training, or
  checkpoint side effects.
- The same cache fingerprint and fold fingerprint flow into both branch
  checkpoints and prediction validation.
- The test suite does not require real DICOM data, internet access, CUDA, a SAM
  checkpoint, or a DINO download.
- The repository remains outside Git.
