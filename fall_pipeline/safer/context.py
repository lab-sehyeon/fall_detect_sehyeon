"""Causal S0-A posterior-dynamics and S0-G0 endpoint features."""

from __future__ import annotations

import torch
from torch import nn


def posterior_dynamics(logits: torch.Tensor, previous: torch.Tensor | None = None) -> torch.Tensor:
    """Build [p(16), delta-p(16), entropy, max, top1-top2] = 35-D."""
    if logits.shape[-1] != 16:
        raise ValueError("S0-A logits must have 16 classes")
    probability = logits.softmax(dim=-1)
    if previous is None:
        delta = torch.zeros_like(probability)
    else:
        if previous.shape != probability.shape:
            raise ValueError("previous posterior shape mismatch")
        delta = probability - previous
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1, keepdim=True)
    top2 = probability.topk(2, dim=-1).values
    maximum = top2[..., :1]
    margin = (top2[..., :1] - top2[..., 1:2])
    return torch.cat((probability, delta, entropy, maximum, margin), dim=-1)


class S0AStateHead(nn.Module):
    def __init__(self, feature_dim: int = 2048, classes: int = 16):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, classes)

    def forward(self, dense_feature: torch.Tensor) -> torch.Tensor:
        return self.classifier(dense_feature)


class S0G0EndpointHead(nn.Module):
    """Four-state endpoint head over DSTE 2048-D + posterior dynamics 35-D."""

    def __init__(self, dense_dim: int = 2048, dynamics_dim: int = 35, classes: int = 4):
        super().__init__()
        self.classifier = nn.Linear(dense_dim + dynamics_dim, classes)

    def forward(self, endpoint_dense: torch.Tensor, dynamics: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat((endpoint_dense, dynamics), dim=-1))
