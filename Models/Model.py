"""Canonical six-slot knee MRI classifier."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .SlotHead import SlotHead


POOL_PARTS = {"cls_mean": 2, "cls_mean_focal": 3}


class Model(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        feature_dim: int,
        n_slots: int,
        n_targets: int,
        pool: str = "cls_mean",
        hidden: int = 256,
        dropout: float = 0.2,
        slot_prior: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if pool not in POOL_PARTS:
            raise ValueError(f"unsupported pooling mode: {pool}")
        self.backbone = backbone
        self.pool = pool
        self.n_slots = n_slots
        self.head = SlotHead(
            feature_dim * POOL_PARTS[pool],
            n_slots=n_slots,
            n_targets=n_targets,
            hidden=hidden,
            dropout=dropout,
            slot_prior=slot_prior,
        )
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(
        self,
        images: torch.Tensor,
        slot_mask: torch.Tensor,
        image_size: int | None = None,
    ) -> torch.Tensor:
        if images.ndim != 5 or images.shape[1] != self.n_slots or images.shape[2] != 3:
            raise ValueError(f"images must have shape [batch, {self.n_slots}, 3, height, width]")
        batch, slots, _, height, width = images.shape
        pixels = images.reshape(batch * slots, 3, height, width).float().div(255.0)
        if image_size is not None and tuple(pixels.shape[-2:]) != (image_size, image_size):
            pixels = F.interpolate(pixels, size=(image_size, image_size), mode="bilinear", align_corners=False)
        pixels = (pixels - self.mean) / self.std
        result = self.backbone(pixel_values=pixels)
        tokens = result.last_hidden_state if hasattr(result, "last_hidden_state") else result["last_hidden_state"]
        if tokens.ndim != 3 or tokens.shape[1] < 2:
            raise ValueError("backbone must return CLS plus at least one patch token")
        patches = tokens[:, 1:]
        parts = [tokens[:, 0], patches.mean(dim=1)]
        if self.pool == "cls_mean_focal":
            count = max(1, patches.shape[1] // 8)
            parts.append(patches.topk(count, dim=1).values.mean(dim=1))
        features = torch.cat(parts, dim=1).reshape(batch, slots, -1)
        return self.head(features, slot_mask)


def build_knee_model(config: dict[str, Any], *, n_targets: int = 12) -> Model:
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(
        config["backbone"],
        local_files_only=bool(config.get("local_files_only", False)),
    )
    return Model(
        backbone,
        feature_dim=int(backbone.config.hidden_size),
        n_slots=6,
        n_targets=n_targets,
        pool=str(config.get("pool", "cls_mean")),
        hidden=int(config.get("hidden", 256)),
        dropout=float(config.get("dropout", 0.2)),
    )


__all__ = ["Model", "POOL_PARTS", "build_knee_model"]
