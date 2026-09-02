import torch
import torch.nn.functional as F


def weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    gamma_pos: float = 0.0,
    gamma_neg: float = 2.0,
    clip: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    if logits.shape != targets.shape or targets.shape != weights.shape:
        raise ValueError("logits, targets, and weights must have identical shapes")

    targets = targets.to(device=logits.device, dtype=logits.dtype)
    weights = weights.to(device=logits.device, dtype=logits.dtype)
    valid = torch.isfinite(targets) & (weights > 0)
    targets = torch.where(valid, targets, torch.zeros_like(targets))
    weights = torch.where(valid, weights, torch.zeros_like(weights))

    pos = torch.sigmoid(logits)
    neg = 1.0 - pos
    neg = (neg + clip).clamp(max=1.0) if clip > 0 else neg

    loss = targets * F.logsigmoid(logits) + (1.0 - targets) * torch.log(
        neg.clamp_min(eps)
    )
    focus = (1.0 - pos).pow(gamma_pos) * targets + (
        1.0 - neg
    ).pow(gamma_neg) * (1.0 - targets)

    return -(loss * focus * weights).sum() / weights.sum().clamp_min(eps)