"""Feature construction contracts shared by the fall branches."""

from __future__ import annotations

import torch


def pooled_dste_feature(temporal: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
    """Return the documented 2048-D max-pooled DSTE representation."""
    if temporal.ndim != 3 or spatial.ndim != 3:
        raise ValueError("temporal and spatial features must be [B,tokens,channels]")
    return torch.cat((temporal.amax(dim=1), spatial.amax(dim=1)), dim=-1)


def dense_dste_feature(
    temporal: torch.Tensor,
    spatial: torch.Tensor,
    spatial_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the project-specific 64x2048 dense representation.

    Spatial context is a valid-token masked mean broadcast over temporal
    tokens, then concatenated with each temporal token.
    """
    if temporal.ndim != 3 or spatial.ndim != 3:
        raise ValueError("temporal and spatial features must be rank three")
    if spatial_valid is None:
        context = spatial.mean(dim=1)
    else:
        if spatial_valid.shape != spatial.shape[:2]:
            raise ValueError("spatial_valid must have shape [B,spatial_tokens]")
        weight = spatial_valid.to(spatial.dtype).unsqueeze(-1)
        context = (spatial * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
    return torch.cat((temporal, context.unsqueeze(1).expand(-1, temporal.shape[1], -1)), dim=-1)


def fuse_overlapping_logits(
    starts: torch.Tensor,
    logits: torch.Tensor,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean raw overlapping logits before softmax, as required by the record."""
    if logits.ndim != 3:
        raise ValueError("logits must be [windows,window_size,classes]")
    totals = logits.new_zeros((sequence_length, logits.shape[-1]))
    counts = logits.new_zeros((sequence_length, 1))
    for start, window in zip(starts.tolist(), logits):
        stop = min(int(start) + window.shape[0], sequence_length)
        width = max(0, stop - int(start))
        if width:
            totals[int(start):stop] += window[:width]
            counts[int(start):stop] += 1
    valid = counts[:, 0] > 0
    totals[valid] /= counts[valid]
    return totals, valid
