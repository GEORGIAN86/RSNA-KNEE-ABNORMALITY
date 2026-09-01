from types import SimpleNamespace

import pytest
import torch

from Models.FusionModel import FusionModel, configure_fusion_trainable


class TinyDINOEncoder(torch.nn.Module):
    output_dim = 4

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(1, self.output_dim)

    def forward(self, images, slot_mask):
        value = images.float().mean((1, 2, 3, 4)).unsqueeze(1)
        return self.projection(value)


class TinySAMEncoder(torch.nn.Module):
    output_dim = 3

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(1, self.output_dim)

    def forward(self, images, slot_mask, slice_mask):
        value = images.float().mean((1, 2, 3, 4)).unsqueeze(1)
        return self.projection(value)


def test_fusion_builds_two_memory_tokens_and_twelve_logits():
    model = FusionModel(
        TinyDINOEncoder(),
        TinySAMEncoder(),
        n_targets=12,
        fusion_dim=16,
        decoder_layers=2,
        attention_heads=4,
        feedforward_dim=32,
        dropout=0.0,
    )
    images = torch.rand(2, 6, 3, 8, 8)
    slot_mask = torch.ones(2, 6, dtype=torch.bool)
    slice_mask = torch.ones(2, 6, 3, dtype=torch.bool)

    memory = model.extract_memory(images, slot_mask, slice_mask)
    logits = model(images, slot_mask, slice_mask)

    assert memory.shape == (2, 2, 16)
    assert logits.shape == (2, 12)


def test_fusion_loss_backpropagates_into_both_encoders_and_decoder():
    model = FusionModel(
        TinyDINOEncoder(),
        TinySAMEncoder(),
        n_targets=12,
        fusion_dim=16,
        decoder_layers=1,
        attention_heads=4,
        feedforward_dim=32,
        dropout=0.0,
    )
    images = torch.rand(2, 6, 3, 8, 8)
    slot_mask = torch.ones(2, 6, dtype=torch.bool)
    slice_mask = torch.ones(2, 6, 3, dtype=torch.bool)

    model(images, slot_mask, slice_mask).sum().backward()

    assert model.dino.projection.weight.grad is not None
    assert model.sam.projection.weight.grad is not None
    assert model.decoder.layers[0].multihead_attn.in_proj_weight.grad is not None
    assert model.output_weight.grad is not None


def _block_model():
    dino_vit = torch.nn.Module()
    dino_vit.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(3)])
    dino_vit.norm = torch.nn.LayerNorm(2)
    dino = torch.nn.Module()
    dino.enc = torch.nn.Module()
    dino.enc.vit = dino_vit
    sam = torch.nn.Module()
    sam.image_encoder = torch.nn.Module()
    sam.image_encoder.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(3)])
    sam.image_encoder.neck = torch.nn.Linear(2, 2)
    return SimpleNamespace(dino=dino, sam=sam)


def test_fusion_trainable_block_policy_unfreezes_only_trailing_blocks():
    model = _block_model()

    configure_fusion_trainable(model, dino_blocks=1, sam_blocks=2)

    assert not any(parameter.requires_grad for parameter in model.dino.enc.vit.blocks[0].parameters())
    assert all(parameter.requires_grad for parameter in model.dino.enc.vit.blocks[-1].parameters())
    assert not any(parameter.requires_grad for parameter in model.sam.image_encoder.blocks[0].parameters())
    assert all(parameter.requires_grad for parameter in model.sam.image_encoder.blocks[-1].parameters())


def test_fusion_trainable_block_policy_rejects_excess_blocks():
    with pytest.raises(ValueError, match="encoder has 3"):
        configure_fusion_trainable(_block_model(), dino_blocks=4, sam_blocks=0)
