"""Explicit reconstruction formulas, not the lost historical implementation."""
from __future__ import annotations

import numpy as np

EPS = 1e-8


def cosine(a, b):
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    value = np.sum(a * b, axis=-1) / np.maximum(denominator, EPS)
    return np.where(denominator > EPS, np.clip(value, -1, 1), 0)


def body_signals(pose, fps):
    """Input [T,25,3] -> float32 [T,12]; CPU eigensolver, no root motion."""
    p = np.asarray(pose, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (25, 3) or not len(p) or not np.isfinite(p).all() or fps <= 0:
        raise ValueError("finite nonempty [T,25,3] and positive fps required")
    p = p - p[:, 1:2]
    torso = p[:, 20] - p[:, 0]
    lengths = np.linalg.norm(torso, axis=1)
    valid = lengths > EPS
    output = np.zeros((len(p), 12), np.float64)
    if not valid.any():
        return output.astype(np.float32)
    first = np.flatnonzero(valid)[0]
    scale = np.median(lengths[valid])
    unit = torso / np.maximum(lengths[:, None], EPS)
    axis = unit[first]
    support = (p[:, 14] + p[:, 18]) / 2
    output[:, 0] = np.clip(unit @ axis, -1, 1)
    output[:, 1] = (p[:, 0] - support) @ axis / scale
    output[:, 2] = (p[:, 3] - support) @ axis / scale
    for column, (a, b) in enumerate(((14, 18), (6, 10), (3, 0)), 3):
        output[:, column] = np.linalg.norm(p[:, a] - p[:, b], axis=1) / scale
    output[:, 6] = (cosine(p[:, 12] - p[:, 13], p[:, 14] - p[:, 13]) +
                    cosine(p[:, 16] - p[:, 17], p[:, 18] - p[:, 17])) / 2
    output[:, 7] = (cosine(p[:, 4] - p[:, 5], p[:, 6] - p[:, 5]) +
                    cosine(p[:, 8] - p[:, 9], p[:, 10] - p[:, 9])) / 2
    shape = p[:, :20] / scale
    centered = shape - shape.mean(axis=1, keepdims=True)
    covariance = np.einsum("tji,tjk->tik", centered, centered) / 20
    eigen = np.maximum(np.linalg.eigvalsh(covariance), 0)
    output[:, 8] = eigen[:, 0] / np.maximum(eigen[:, -1], EPS)
    transitions = valid[1:] & valid[:-1]
    output[1:, 9] = np.linalg.norm(np.diff(shape, axis=0), axis=2).mean(axis=1) * fps * transitions
    angle = np.arccos(np.clip(np.sum(unit[1:] * unit[:-1], axis=1), -1, 1))
    output[1:, 10] = angle * fps * transitions
    output[:, 11] = np.sqrt(np.square(shape - shape[first]).sum(axis=2).mean(axis=1))
    output[~valid] = 0
    if not np.isfinite(output).all():
        raise ValueError("nonfinite primitive output")
    return output.astype(np.float32)


def descriptor(signals):
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 12 or not len(x) or not np.isfinite(x).all():
        raise ValueError("finite nonempty [T,12] required")
    stats = [x.mean(0), x.std(0), x.min(0), x.max(0),
             *np.quantile(x, [.1, .25, .5, .75, .9], axis=0, method="linear"),
             x[0], x[-1], x[-1] - x[0], np.abs(np.diff(x, axis=0)).mean(0) if len(x) > 1 else np.zeros(12)]
    return np.stack(stats, axis=1).reshape(156).astype(np.float32)


def fold_standardize(descriptors, train_indices):
    data = np.asarray(descriptors, dtype=np.float64)
    mean = data[train_indices].mean(0)
    scale = np.maximum(data[train_indices].std(0), 1e-6)
    return ((data - mean) / scale).astype(np.float32), mean, scale
