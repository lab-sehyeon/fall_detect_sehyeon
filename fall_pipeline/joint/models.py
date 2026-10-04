"""Recovered J0/J1 and G0/G1/G2 model definitions."""

from __future__ import annotations

import torch
from torch import nn


class J1ResidualAdapter(nn.Module):
    """Zero-initialized residual bottleneck: 2048 -> 256 -> 2048."""

    def __init__(self, feature_dim: int = 2048, bottleneck_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.down = nn.Linear(feature_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, feature_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        correction = self.up(self.dropout(self.activation(self.down(self.norm(feature)))))
        return feature + correction


class JointFallModel(nn.Module):
    """Shared J1 adapter with label-unit-specific SAFER and FU heads."""

    def __init__(self, feature_dim: int = 2048, bottleneck_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.adapter = J1ResidualAdapter(feature_dim, bottleneck_dim, dropout)
        self.safer_head = nn.Linear(feature_dim, 4)
        self.fu_head = nn.Linear(feature_dim, 2)

    def forward(self, feature: torch.Tensor, dataset: str) -> torch.Tensor:
        adapted = self.adapter(feature)
        if dataset == "safer":
            return self.safer_head(adapted)
        if dataset == "fu":
            return self.fu_head(adapted)
        raise ValueError("dataset must be 'safer' or 'fu'")


class GlobalProbeHeads(nn.Module):
    """Matched four-class G0/G1/G2 linear probes."""

    def __init__(self, j1_dim: int = 2048, global_dim: int = 131, classes: int = 4):
        super().__init__()
        self.g0 = nn.Linear(j1_dim, classes)
        self.g1 = nn.Linear(global_dim, classes)
        self.g2 = nn.Linear(j1_dim + global_dim, classes)

    def forward(self, j1: torch.Tensor, global_feature: torch.Tensor, variant: str = "G2") -> torch.Tensor:
        variant = variant.upper()
        if variant == "G0":
            return self.g0(j1)
        if variant == "G1":
            return self.g1(global_feature)
        if variant == "G2":
            return self.g2(torch.cat((j1, global_feature), dim=-1))
        raise ValueError("variant must be G0, G1 or G2")
