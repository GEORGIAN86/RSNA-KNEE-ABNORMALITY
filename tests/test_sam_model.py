import torch
import torch.nn as nn

from Models.SAMClassifier import SAMClassifier, configure_sam_trainable


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block(), Block()])
        self.neck = nn.Conv2d(1, 4, 1)

    def forward(self, images):
        return self.neck(images[:, :1])


def test_partial_freeze_unfreezes_only_final_blocks_and_head():
    model = SAMClassifier(FakeEncoder(), feature_dim=4, n_targets=12, input_size=8)

    configure_sam_trainable(model, trainable_blocks=1)

    assert not model.image_encoder.blocks[0].weight.requires_grad
    assert model.image_encoder.blocks[-1].weight.requires_grad
    assert all(parameter.requires_grad for parameter in model.head.parameters())


def test_sam_forward_ignores_padded_slices():
    model = SAMClassifier(FakeEncoder(), feature_dim=4, n_targets=12, input_size=8)
    images = torch.zeros(2, 6, 2, 8, 8)
    slot_mask = torch.ones(2, 6, dtype=torch.bool)
    slice_mask = torch.ones(2, 6, 2, dtype=torch.bool)
    slice_mask[:, :, 1] = False

    logits = model(images, slot_mask, slice_mask)

    assert tuple(logits.shape) == (2, 12)
