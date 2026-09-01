import torch

from Models.DINOv3Knee import ViTSlotToken
from Models.SAMClassifier import SAMStudyEncoder


class TinyViT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 4
        self.num_features = 4
        self.num_prefix_tokens = 1
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])


class TinySAM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])


def test_encoder_wrappers_store_gradient_checkpointing_policy():
    dino = ViTSlotToken(TinyViT(), gradient_checkpointing=True)
    sam = SAMStudyEncoder(TinySAM(), feature_dim=4, gradient_checkpointing=True)

    assert dino.gradient_checkpointing is True
    assert sam.gradient_checkpointing is True
