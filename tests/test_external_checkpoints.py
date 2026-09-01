from pathlib import Path
import zipfile

import pytest
import torch

from Training.external_checkpoints import (
    ExternalCheckpointError,
    load_external_knee_state,
    load_extracted_torch_checkpoint,
    resolve_external_knee_checkpoint,
    translate_external_sam_state,
)


def test_resolves_matching_external_knee_fold(tmp_path: Path):
    checkpoint = tmp_path / "m_f2.pt"
    torch.save({"state_dict": {"weight": torch.ones(1)}, "cfg": {"labels": ["ACL"]}, "fold": 2}, checkpoint)

    assert resolve_external_knee_checkpoint(tmp_path / "m_f{fold}.pt", 2) == checkpoint
    payload = load_external_knee_state(checkpoint, expected_fold=2, labels=["ACL"])
    assert payload["state_dict"]["weight"].item() == 1


def test_external_knee_checkpoint_rejects_wrong_fold(tmp_path: Path):
    checkpoint = tmp_path / "m_f0.pt"
    torch.save({"state_dict": {}, "cfg": {"labels": ["ACL"]}, "fold": 1}, checkpoint)

    with pytest.raises(ExternalCheckpointError, match="fold"):
        load_external_knee_state(checkpoint, expected_fold=0, labels=["ACL"])


def test_external_knee_checkpoint_rejects_wrong_labels(tmp_path: Path):
    checkpoint = tmp_path / "m_f0.pt"
    torch.save({"state_dict": {}, "cfg": {"labels": ["MCL"]}, "fold": 0}, checkpoint)

    with pytest.raises(ExternalCheckpointError, match="labels"):
        load_external_knee_state(checkpoint, expected_fold=0, labels=["ACL"])


def test_loads_extracted_torch_checkpoint_directory(tmp_path: Path):
    source = tmp_path / "source.pt"
    torch.save({"m": {"e.weight": torch.ones(2, 2)}}, source)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with zipfile.ZipFile(source) as archive:
        archive.extractall(extracted)
    extracted = next(path for path in extracted.iterdir() if path.is_dir())

    payload = load_extracted_torch_checkpoint(extracted)

    assert torch.equal(payload["m"]["e.weight"], torch.ones(2, 2))


def test_translates_external_sam_model_keys():
    state = {
        "m": torch.ones(1, 3, 1, 1),
        "s": torch.full((1, 3, 1, 1), 2.0),
        "e.layer.weight": torch.ones(2, 2),
        "h.0.weight": torch.ones(2),
    }

    translated = translate_external_sam_state(state)

    assert set(translated) == {
        "pixel_mean",
        "pixel_std",
        "image_encoder.layer.weight",
        "head.0.weight",
    }


def test_sam_translation_rejects_unknown_keys():
    with pytest.raises(ExternalCheckpointError, match="unknown SAM model key"):
        translate_external_sam_state({"unexpected": torch.ones(1)})
