"""Checkpoint-compatible DINOv3 knee model from the reference inference recipe."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


N_SLOT_TYPES = 6


class ViTSlotToken(nn.Module):
    def __init__(
        self,
        vit: nn.Module,
        n_categories: int = N_SLOT_TYPES,
        *,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.vit = vit
        dimension = int(vit.embed_dim)
        self.tok = nn.Embedding(n_categories + 1, dimension, padding_idx=0)
        self.num_features = int(vit.num_features)
        self.gradient_checkpointing = gradient_checkpointing
        self._orig_prefix = int(getattr(vit, "num_prefix_tokens", 1))
        vit.num_prefix_tokens = self._orig_prefix + 1
        for block in vit.blocks:
            attention = getattr(block, "attn", None)
            if attention is not None and hasattr(attention, "num_prefix_tokens"):
                attention.num_prefix_tokens += 1

    @staticmethod
    def _maybe(module: nn.Module | None, value: torch.Tensor) -> torch.Tensor:
        return value if module is None else module(value)

    def forward_features(self, images: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        vit = self.vit
        tokens = vit.patch_embed(images)
        positioned = vit._pos_embed(tokens)
        rope = None
        if isinstance(positioned, tuple):
            tokens, rope = positioned
        else:
            tokens = positioned
        tokens = self._maybe(getattr(vit, "patch_drop", None), tokens)
        tokens = self._maybe(getattr(vit, "norm_pre", None), tokens)
        slot_token = self.tok(slots).unsqueeze(1)
        tokens = torch.cat(
            [tokens[:, : self._orig_prefix], slot_token, tokens[:, self._orig_prefix :]],
            dim=1,
        )
        if rope is None:
            for block in vit.blocks:
                tokens = checkpoint(block, tokens, use_reentrant=False) if self.training and self.gradient_checkpointing else block(tokens)
        elif getattr(vit, "rope_mixed", False):
            for index, block in enumerate(vit.blocks):
                if self.training and self.gradient_checkpointing:
                    tokens = checkpoint(lambda value, b=block, r=rope[index]: b(value, rope=r), tokens, use_reentrant=False)
                else:
                    tokens = block(tokens, rope=rope[index])
        else:
            for block in vit.blocks:
                if self.training and self.gradient_checkpointing:
                    tokens = checkpoint(lambda value, b=block: b(value, rope=rope), tokens, use_reentrant=False)
                else:
                    tokens = block(tokens, rope=rope)
        return vit.norm(tokens)


def _segment_mean_max(values: torch.Tensor, study_indices: torch.Tensor, batch: int) -> torch.Tensor:
    dimension = values.shape[1]
    ones = torch.ones(len(values), device=values.device, dtype=values.dtype)
    counts = torch.zeros(batch, device=values.device, dtype=values.dtype).index_add_(0, study_indices, ones)
    means = torch.zeros(batch, dimension, device=values.device, dtype=values.dtype).index_add_(
        0, study_indices, values
    )
    means = means / counts.clamp_min(1).unsqueeze(1)
    maxima = torch.full((batch, dimension), -10000.0, device=values.device, dtype=values.dtype)
    maxima = maxima.scatter_reduce(
        0,
        study_indices.unsqueeze(1).expand(-1, dimension),
        values,
        reduce="amax",
        include_self=True,
    )
    return torch.cat([means, maxima], dim=1)


def _pad_patch_tokens(
    tokens: torch.Tensor,
    study_indices: torch.Tensor,
    batch: int,
    normalization: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    total, patches, dimension = tokens.shape
    counts = torch.bincount(study_indices, minlength=batch)
    max_slots = int(counts.max().item())
    starts = torch.cumsum(counts, 0) - counts
    positions = torch.arange(total, device=tokens.device) - starts[study_indices]
    padded = tokens.new_zeros(batch, max_slots, patches, dimension)
    padded[study_indices, positions] = tokens
    keep = torch.zeros(batch, max_slots, dtype=torch.bool, device=tokens.device)
    keep[study_indices, positions] = True
    return normalization(padded.reshape(batch, max_slots * patches, dimension)), ~keep.repeat_interleave(
        patches, dim=1
    )


class CodexResidualPool(nn.Module):
    def __init__(self, dimension: int, n_targets: int, *, presence_dim: int = 64) -> None:
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_targets, dimension) * 0.02)
        self.kv_norm = nn.LayerNorm(dimension)
        self.attn = nn.MultiheadAttention(dimension, 6, dropout=0.2, batch_first=True)
        self.d_norm = nn.LayerNorm(dimension)
        self.dw = nn.Parameter(torch.randn(n_targets, dimension) * (1 / dimension**0.5))
        self.db = nn.Parameter(torch.zeros(n_targets))
        self.gate = nn.Parameter(torch.zeros(n_targets))
        self.base = nn.Sequential(
            nn.LayerNorm(2 * dimension + presence_dim),
            nn.Dropout(0.2),
            nn.Linear(2 * dimension + presence_dim, n_targets),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        study_indices: torch.Tensor,
        batch: int,
        presence: torch.Tensor,
    ) -> torch.Tensor:
        baseline = self.base(torch.cat([_segment_mean_max(tokens[:, 0], study_indices, batch), presence], 1))
        key_values, padding = _pad_patch_tokens(tokens[:, 1:], study_indices, batch, self.kv_norm)
        queries = self.q.unsqueeze(0).expand(batch, -1, -1)
        attended, _ = self.attn(queries, key_values, key_values, key_padding_mask=padding, need_weights=False)
        delta = (self.d_norm(attended) * self.dw).sum(-1) + self.db
        return baseline + self.gate * delta


class Readout(nn.Module):
    def __init__(self, dimension: int, n_targets: int) -> None:
        super().__init__()
        self.pres_emb = nn.Embedding(N_SLOT_TYPES + 1, 64, padding_idx=0)
        self.pool = CodexResidualPool(dimension, n_targets)

    def forward(
        self,
        tokens: torch.Tensor,
        slots: torch.Tensor,
        study_indices: torch.Tensor,
        batch: int,
    ) -> torch.Tensor:
        embedded = self.pres_emb(slots)
        presence = torch.zeros(batch, embedded.shape[1], device=tokens.device, dtype=tokens.dtype)
        presence.index_add_(0, study_indices, embedded)
        return self.pool(tokens, study_indices, batch, presence)


class DINOv3KneeModel(nn.Module):
    def __init__(self, encoder: ViTSlotToken, *, n_targets: int, normalization: str = "none") -> None:
        super().__init__()
        self.enc = encoder
        self.readout = Readout(encoder.num_features, n_targets)
        self.normalization = normalization

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float().div(255.0)
        mask = (images > 0).to(images.dtype)
        if self.normalization == "imagenet":
            return (images - 0.485) / 0.229 * mask
        if self.normalization == "zscore":
            count = mask.sum((1, 2, 3), keepdim=True).clamp_min(1)
            mean = (images * mask).sum((1, 2, 3), keepdim=True) / count
            variance = (((images - mean) * mask) ** 2).sum((1, 2, 3), keepdim=True) / count
            return (images - mean) / (variance.sqrt() + 1e-6) * mask
        return images

    def forward(self, images: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or images.shape[1] != N_SLOT_TYPES:
            raise ValueError("DINOv3 knee images must have shape [batch, 6, slices, height, width]")
        valid = slot_mask.bool()
        if (~valid.any(1)).any():
            raise ValueError("every study must contain at least one valid knee slot")
        batch = images.shape[0]
        packed = self._normalize(images[valid])
        slots = torch.arange(1, N_SLOT_TYPES + 1, device=images.device).unsqueeze(0).expand(batch, -1)[valid]
        study_indices = torch.arange(batch, device=images.device).unsqueeze(1).expand_as(valid)[valid]
        features = self.enc.forward_features(packed, slots)
        features = torch.cat([features[:, :1], features[:, self.enc._orig_prefix :]], dim=1)
        return self.readout(features, slots, study_indices, batch)


class DINOStudyEncoder(nn.Module):
    """Return a mean/max pooled CLS study embedding without legacy logits."""

    output_dim = 768

    def __init__(self, encoder: ViTSlotToken, *, normalization: str = "none") -> None:
        super().__init__()
        self.enc = encoder
        self.normalization = normalization
        self.output_dim = 2 * int(encoder.num_features)

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float().div(255.0)
        mask = (images > 0).to(images.dtype)
        if self.normalization == "imagenet":
            return (images - 0.485) / 0.229 * mask
        if self.normalization == "zscore":
            count = mask.sum((1, 2, 3), keepdim=True).clamp_min(1)
            mean = (images * mask).sum((1, 2, 3), keepdim=True) / count
            variance = (((images - mean) * mask) ** 2).sum((1, 2, 3), keepdim=True) / count
            return (images - mean) / (variance.sqrt() + 1e-6) * mask
        return images

    def forward(self, images: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
        valid = slot_mask.bool()
        if images.ndim != 5 or images.shape[1] != N_SLOT_TYPES:
            raise ValueError("DINO images must have shape [batch, 6, slices, height, width]")
        if (~valid.any(1)).any():
            raise ValueError("every study must contain at least one valid DINO slot")
        batch = images.shape[0]
        packed = self._normalize(images[valid])
        slots = torch.arange(1, N_SLOT_TYPES + 1, device=images.device).unsqueeze(0).expand(batch, -1)[valid]
        study_indices = torch.arange(batch, device=images.device).unsqueeze(1).expand_as(valid)[valid]
        features = self.enc.forward_features(packed, slots)
        return _segment_mean_max(features[:, 0], study_indices, batch)


def build_dinov3_knee_model(config: dict[str, Any], *, n_targets: int = 12) -> DINOv3KneeModel:
    expected = {
        "backbone": "vit_small_patch16_dinov3.lvd1689m",
        "cond": "token",
        "pool": "xcodex",
        "stem": "native",
        "n_slice": 16,
        "n_meta": 0,
    }
    mismatches = {key: config.get(key) for key, value in expected.items() if config.get(key) != value}
    if mismatches:
        raise ValueError(f"unsupported external knee checkpoint recipe: {mismatches}")
    import timm

    backbone = timm.create_model(
        str(config["backbone"]),
        pretrained=False,
        num_classes=0,
        in_chans=int(config["n_slice"]),
        img_size=int(config["img"]),
    )
    return DINOv3KneeModel(
        ViTSlotToken(backbone, gradient_checkpointing=bool(config.get("gradient_checkpointing", False))),
        n_targets=n_targets,
        normalization=str(config.get("norm", "none")),
    )


def build_dino_study_encoder(config: dict[str, Any]) -> DINOStudyEncoder:
    legacy = build_dinov3_knee_model(config, n_targets=12)
    return DINOStudyEncoder(legacy.enc, normalization=str(config.get("norm", "none")))


__all__ = [
    "DINOStudyEncoder",
    "DINOv3KneeModel",
    "ViTSlotToken",
    "build_dino_study_encoder",
    "build_dinov3_knee_model",
]
