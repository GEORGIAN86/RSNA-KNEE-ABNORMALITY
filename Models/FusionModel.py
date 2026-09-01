"""End-to-end transformer fusion of DINO and SAM study embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn

from .DINOv3Knee import build_dino_study_encoder
from .SAMClassifier import SAMStudyEncoder, build_sam_model


def configure_fusion_trainable(model: "FusionModel", dino_blocks: int, sam_blocks: int) -> None:
    if dino_blocks < -1 or sam_blocks < -1:
        raise ValueError("trainable block counts must be -1, 0, or positive")
    for parameter in model.dino.parameters():
        parameter.requires_grad = dino_blocks == -1
    dino_layers = model.dino.enc.vit.blocks
    if dino_blocks > len(dino_layers):
        raise ValueError(f"requested {dino_blocks} DINO blocks, encoder has {len(dino_layers)}")
    if dino_blocks > 0:
        for block in dino_layers[-dino_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in model.dino.enc.vit.norm.parameters():
            parameter.requires_grad = True

    for parameter in model.sam.parameters():
        parameter.requires_grad = sam_blocks == -1
    sam_layers = model.sam.image_encoder.blocks
    if sam_blocks > len(sam_layers):
        raise ValueError(f"requested {sam_blocks} SAM blocks, encoder has {len(sam_layers)}")
    if sam_blocks > 0:
        for block in sam_layers[-sam_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in model.sam.image_encoder.neck.parameters():
            parameter.requires_grad = True


class FusionModel(nn.Module):
    def __init__(
        self,
        dino: nn.Module,
        sam: nn.Module,
        *,
        n_targets: int = 12,
        fusion_dim: int = 256,
        decoder_layers: int = 2,
        attention_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if fusion_dim % attention_heads:
            raise ValueError("fusion_dim must be divisible by attention_heads")
        self.dino = dino
        self.sam = sam
        self.n_targets = n_targets
        self.dino_projection = nn.Sequential(
            nn.LayerNorm(int(dino.output_dim)), nn.Linear(int(dino.output_dim), fusion_dim)
        )
        self.sam_projection = nn.Sequential(
            nn.LayerNorm(int(sam.output_dim)), nn.Linear(int(sam.output_dim), fusion_dim)
        )
        self.modality_embedding = nn.Parameter(torch.randn(2, fusion_dim) * 0.02)
        self.label_query = nn.Parameter(torch.randn(n_targets, fusion_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=fusion_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=decoder_layers)
        self.output_norm = nn.LayerNorm(fusion_dim)
        self.output_weight = nn.Parameter(torch.randn(n_targets, fusion_dim) * (1 / fusion_dim**0.5))
        self.output_bias = nn.Parameter(torch.zeros(n_targets))

    def extract_memory(
        self,
        images: torch.Tensor,
        slot_mask: torch.Tensor,
        slice_mask: torch.Tensor,
        sam_images: torch.Tensor | None = None,
        sam_slice_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dino = self.dino(images, slot_mask)
        sam = self.sam(
            images if sam_images is None else sam_images,
            slot_mask,
            slice_mask if sam_slice_mask is None else sam_slice_mask,
        )
        tokens = torch.stack([self.dino_projection(dino), self.sam_projection(sam)], dim=1)
        return tokens + self.modality_embedding.unsqueeze(0)

    def forward(
        self,
        images: torch.Tensor,
        slot_mask: torch.Tensor,
        slice_mask: torch.Tensor,
        sam_images: torch.Tensor | None = None,
        sam_slice_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.extract_memory(images, slot_mask, slice_mask, sam_images, sam_slice_mask)
        queries = self.label_query.unsqueeze(0).expand(images.shape[0], -1, -1)
        decoded = self.output_norm(self.decoder(queries, memory))
        return (decoded * self.output_weight.unsqueeze(0)).sum(-1) + self.output_bias


def build_fusion_model(config: dict, *, n_targets: int = 12) -> FusionModel:
    dino = build_dino_study_encoder(config["knee"])
    sam_classifier = build_sam_model(config["sam"], None, n_targets=n_targets)
    for parameter in sam_classifier.image_encoder.parameters():
        parameter.requires_grad = True
    sam = SAMStudyEncoder(
        sam_classifier.image_encoder,
        feature_dim=int(config["sam"].get("feature_dim", 256)),
        input_size=int(config["sam"].get("input_size", 512)),
        encode_chunk=int(config["sam"].get("encode_chunk", 6)),
        gradient_checkpointing=bool(config["sam"].get("gradient_checkpointing", False)),
    )
    fusion = config["fusion"]
    model = FusionModel(
        dino,
        sam,
        n_targets=n_targets,
        fusion_dim=int(fusion["fusion_dim"]),
        decoder_layers=int(fusion["decoder_layers"]),
        attention_heads=int(fusion["attention_heads"]),
        feedforward_dim=int(fusion["feedforward_dim"]),
        dropout=float(fusion.get("dropout", 0.2)),
    )
    configure_fusion_trainable(
        model,
        int(config["knee"].get("trainable_blocks", 4)),
        int(config["sam"].get("trainable_blocks", 2)),
    )
    return model


__all__ = ["FusionModel", "build_fusion_model", "configure_fusion_trainable"]
