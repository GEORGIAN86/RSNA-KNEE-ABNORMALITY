import pytest
import torch

from Loss import weighted_bce


def test_weighted_bce_applies_cell_weights():
    logits = torch.zeros(1, 2)
    targets = torch.tensor([[1.0, 0.0]])
    weights = torch.tensor([[2.0, 0.0]])

    loss = weighted_bce(logits, targets, weights)

    assert loss.item() == pytest.approx(0.693147, abs=1e-5)
