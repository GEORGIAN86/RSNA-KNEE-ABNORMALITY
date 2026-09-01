"""Label-specific attention over the six MRI acquisition slots."""

from __future__ import annotations

import torch
import torch.nn as nn


class SlotHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        *,
        n_slots: int,
        n_targets: int,
        hidden: int = 256,
        dropout: float = 0.2,
        slot_prior: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.n_slots = n_slots
        self.n_targets = n_targets
        self.hidden = hidden
        self.projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
        )
        self.slot_embedding = nn.Parameter(torch.randn(n_slots, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_targets, hidden) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden, n_targets)
        if slot_prior is not None:
            if tuple(slot_prior.shape) != (n_targets, n_slots):
                raise ValueError("slot_prior must have shape [n_targets, n_slots]")
            self.register_buffer("slot_prior", slot_prior.float().clone())
        else:
            self.slot_prior = None

    def forward(self, features: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, slots, feature_dim]")
        if features.shape[1] != self.n_slots:
            raise ValueError(f"expected {self.n_slots} slots, received {features.shape[1]}")
        if tuple(slot_mask.shape) != tuple(features.shape[:2]):
            raise ValueError("slot_mask must have shape [batch, slots]")
        valid = slot_mask.bool()
        if (~valid.any(dim=1)).any():
            raise ValueError("every study must contain at least one valid slot")
        hidden = self.projection(features) + self.slot_embedding.unsqueeze(0)
        attention = torch.einsum("bsh,th->bts", hidden, self.query) / self.hidden**0.5
        if self.slot_prior is not None:
            attention = attention + self.slot_prior.unsqueeze(0)
        attention = attention.masked_fill(~valid.unsqueeze(1), torch.finfo(attention.dtype).min)
        attention = attention.softmax(dim=-1)
        context = self.dropout(torch.einsum("bts,bsh->bth", attention, hidden))
        return (context * self.output.weight.unsqueeze(0)).sum(dim=-1) + self.output.bias


__all__ = ["SlotHead"]
