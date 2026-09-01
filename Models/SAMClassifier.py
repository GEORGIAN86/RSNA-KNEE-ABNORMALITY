"""Study-level classifier built on a partially trainable SAM image encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SAMClassifier(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        *,
        feature_dim: int = 256,
        n_targets: int = 12,
        input_size: int = 512,
        encode_chunk: int = 6,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_size <= 0 or encode_chunk <= 0:
            raise ValueError("input_size and encode_chunk must be positive")
        self.image_encoder = image_encoder
        self.feature_dim = feature_dim
        self.input_size = input_size
        self.encode_chunk = encode_chunk
        hidden = max(64, feature_dim // 2)
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_targets),
        )
        self.register_buffer(
            "pixel_mean",
            torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1),
        )

    def _encode_real_sam(self, pixels: torch.Tensor) -> torch.Tensor:
        encoder = self.image_encoder
        values = (pixels - self.pixel_mean) / self.pixel_std
        embedded = encoder.patch_embed(values)
        positional = encoder.pos_embed
        if tuple(positional.shape[1:3]) != tuple(embedded.shape[1:3]):
            positional = F.interpolate(
                positional.permute(0, 3, 1, 2),
                size=embedded.shape[1:3],
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
        embedded = embedded + positional
        for block in encoder.blocks:
            embedded = block(embedded)
        return encoder.neck(embedded.permute(0, 3, 1, 2))

    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = F.interpolate(
            pixels,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        if all(hasattr(self.image_encoder, name) for name in ("patch_embed", "pos_embed", "blocks", "neck")):
            features = self._encode_real_sam(pixels)
        else:
            features = self.image_encoder((pixels - self.pixel_mean) / self.pixel_std)
        if features.ndim == 4:
            features = features.mean(dim=(-2, -1))
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"SAM encoder must produce [slices, {self.feature_dim}] features; received {tuple(features.shape)}"
            )
        return features

    def forward(
        self,
        images: torch.Tensor,
        slot_mask: torch.Tensor,
        slice_mask: torch.Tensor,
    ) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError("images must have shape [batch, slots, slices, height, width]")
        if tuple(slot_mask.shape) != tuple(images.shape[:2]):
            raise ValueError("slot_mask must have shape [batch, slots]")
        if tuple(slice_mask.shape) != tuple(images.shape[:3]):
            raise ValueError("slice_mask must have shape [batch, slots, slices]")
        valid = slice_mask.bool() & slot_mask.bool().unsqueeze(-1)
        if (~valid.flatten(1).any(dim=1)).any():
            raise ValueError("every study must contain at least one valid SAM slice")
        selected = images[valid].unsqueeze(1).expand(-1, 3, -1, -1).float()
        encoded = [
            self._encode(selected[start : start + self.encode_chunk])
            for start in range(0, len(selected), self.encode_chunk)
        ]
        slice_features = torch.cat(encoded, dim=0)
        batch = images.shape[0]
        study_indices = (
            torch.arange(batch, device=images.device)
            .view(batch, 1, 1)
            .expand_as(valid)[valid]
        )
        totals = torch.zeros(batch, self.feature_dim, device=images.device, dtype=slice_features.dtype)
        counts = torch.zeros(batch, 1, device=images.device, dtype=slice_features.dtype)
        totals.index_add_(0, study_indices, slice_features)
        counts.index_add_(0, study_indices, torch.ones(len(slice_features), 1, device=images.device, dtype=slice_features.dtype))
        return self.head(totals / counts.clamp_min(1))


class SAMStudyEncoder(nn.Module):
    """Return the averaged SAM image-encoder study embedding."""

    output_dim = 256

    def __init__(
        self,
        image_encoder: nn.Module,
        *,
        feature_dim: int = 256,
        input_size: int = 512,
        encode_chunk: int = 6,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.feature_dim = feature_dim
        self.output_dim = feature_dim
        self.input_size = input_size
        self.encode_chunk = encode_chunk
        self.gradient_checkpointing = gradient_checkpointing
        self.register_buffer("pixel_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))

    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = F.interpolate(pixels, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        encoder = self.image_encoder
        values = (pixels - self.pixel_mean) / self.pixel_std
        embedded = encoder.patch_embed(values)
        positional = encoder.pos_embed
        if tuple(positional.shape[1:3]) != tuple(embedded.shape[1:3]):
            positional = F.interpolate(
                positional.permute(0, 3, 1, 2),
                size=embedded.shape[1:3],
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
        embedded = embedded + positional
        for block in encoder.blocks:
            embedded = checkpoint(block, embedded, use_reentrant=False) if self.training and self.gradient_checkpointing else block(embedded)
        return encoder.neck(embedded.permute(0, 3, 1, 2)).mean((-2, -1))

    def forward(self, images: torch.Tensor, slot_mask: torch.Tensor, slice_mask: torch.Tensor) -> torch.Tensor:
        valid = slice_mask.bool() & slot_mask.bool().unsqueeze(-1)
        if (~valid.flatten(1).any(1)).any():
            raise ValueError("every study must contain at least one valid SAM slice")
        selected = images[valid].unsqueeze(1).expand(-1, 3, -1, -1).float()
        features = torch.cat(
            [self._encode(selected[start : start + self.encode_chunk]) for start in range(0, len(selected), self.encode_chunk)],
            dim=0,
        )
        batch = images.shape[0]
        study_indices = torch.arange(batch, device=images.device).view(batch, 1, 1).expand_as(valid)[valid]
        totals = torch.zeros(batch, self.feature_dim, device=images.device, dtype=features.dtype)
        counts = torch.zeros(batch, 1, device=images.device, dtype=features.dtype)
        totals.index_add_(0, study_indices, features)
        counts.index_add_(0, study_indices, torch.ones(len(features), 1, device=images.device, dtype=features.dtype))
        return totals / counts.clamp_min(1)


def configure_sam_trainable(model: SAMClassifier, trainable_blocks: int) -> None:
    if trainable_blocks < -1:
        raise ValueError("trainable_blocks must be -1, 0, or a positive integer")
    for parameter in model.image_encoder.parameters():
        parameter.requires_grad = False
    blocks = getattr(model.image_encoder, "blocks", None)
    if trainable_blocks == -1:
        for parameter in model.image_encoder.parameters():
            parameter.requires_grad = True
    elif trainable_blocks > 0:
        if blocks is None:
            raise ValueError("the SAM image encoder does not expose blocks")
        if trainable_blocks > len(blocks):
            raise ValueError(f"requested {trainable_blocks} trainable blocks, encoder has {len(blocks)}")
        for block in blocks[-trainable_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        neck = getattr(model.image_encoder, "neck", None)
        if neck is not None:
            for parameter in neck.parameters():
                parameter.requires_grad = True
    for parameter in model.head.parameters():
        parameter.requires_grad = True


def build_sam_model(
    config: dict[str, Any],
    checkpoint_path: str | Path | None,
    *,
    n_targets: int = 12,
) -> SAMClassifier:
    from segment_anything import sam_model_registry

    source = Path(checkpoint_path) if checkpoint_path is not None else None
    if source is not None and not source.is_file():
        raise FileNotFoundError(f"SAM base checkpoint not found: {source}")
    model_type = str(config.get("model_type", "vit_b"))
    if model_type not in sam_model_registry:
        raise ValueError(f"unknown SAM model type: {model_type}")
    sam = sam_model_registry[model_type](checkpoint=str(source) if source is not None else None)
    model = SAMClassifier(
        sam.image_encoder,
        feature_dim=int(config.get("feature_dim", 256)),
        n_targets=n_targets,
        input_size=int(config.get("input_size", 512)),
        encode_chunk=int(config.get("encode_chunk", 6)),
        dropout=float(config.get("dropout", 0.2)),
    )
    configure_sam_trainable(model, int(config.get("trainable_blocks", 2)))
    return model


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable, total


__all__ = [
    "SAMStudyEncoder",
    "SAMClassifier",
    "build_sam_model",
    "configure_sam_trainable",
    "parameter_counts",
]
