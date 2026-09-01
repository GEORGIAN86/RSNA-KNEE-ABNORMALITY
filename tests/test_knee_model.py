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
    model = Model(
        FakeBackbone(),
        feature_dim=8,
        n_slots=6,
        n_targets=12,
        pool="cls_mean",
    )

    logits = model(torch.zeros(2, 6, 3, 8, 8), torch.ones(2, 6))

    assert tuple(logits.shape) == (2, 12)


def test_slot_head_handles_missing_slots_without_nan():
    head = SlotHead(16, n_slots=6, n_targets=12)
    mask = torch.tensor([[1, 0, 1, 0, 0, 0]], dtype=torch.float32)

    logits = head(torch.randn(1, 6, 16), mask)

    assert torch.isfinite(logits).all()
