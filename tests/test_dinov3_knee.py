from pathlib import Path

import pytest
import torch

from Models.DINOv3Knee import build_dinov3_knee_model


def test_supplied_dinov3_checkpoint_loads_strictly():
    path = Path("weights/checkpoints/knee/m_f0.pt")
    if not path.is_file():
        pytest.skip("production checkpoint is not available")
    payload = torch.load(path, map_location="cpu", weights_only=True)

    model = build_dinov3_knee_model(payload["cfg"], n_targets=12)
    result = model.load_state_dict(payload["state_dict"], strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_dinov3_builder_rejects_unsupported_checkpoint_recipe():
    with pytest.raises(ValueError, match="unsupported external knee"):
        build_dinov3_knee_model(
            {
                "backbone": "vit_small_patch16_dinov3.lvd1689m",
                "cond": "post",
                "pool": "mean_max",
                "stem": "compress",
                "n_slice": 3,
                "n_meta": 0,
                "img": 336,
            },
            n_targets=12,
        )
