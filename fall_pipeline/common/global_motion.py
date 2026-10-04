"""Causal 131-D image-relative global-motion descriptor.

The recovered schema uses 12 input channels, five window statistics for their
causally smoothed levels and velocities (120 values), four vertical-
acceleration statistics and seven validity/quality values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LEVEL_STATS = ("mean", "std", "min", "max", "last")


def trailing_valid_mean(values: np.ndarray, valid: np.ndarray, width: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("values and valid must both be [frames,channels]")
    output = np.zeros_like(values)
    for frame in range(len(values)):
        start = max(0, frame - width + 1)
        chunk_valid = valid[start:frame + 1]
        count = chunk_valid.sum(axis=0)
        total = np.where(chunk_valid, values[start:frame + 1], 0).sum(axis=0)
        output[frame] = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    return output


def causal_derivative(values: np.ndarray, fps: float = 25.0) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float32)
    if len(values) > 1:
        output[1:] = np.diff(values, axis=0) * float(fps)
    return output


def _five_stats(values: np.ndarray) -> np.ndarray:
    return np.stack((
        values.mean(axis=0), values.std(axis=0), values.min(axis=0),
        values.max(axis=0), values[-1],
    ), axis=1).reshape(-1)


def build_global_131(
    channels: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    fps: float = 25.0,
    vertical_channel: int = 1,
) -> np.ndarray:
    channels = np.asarray(channels, dtype=np.float32)
    if channels.ndim != 2 or channels.shape[1] != 12:
        raise ValueError("global channels must be [frames,12]")
    if len(channels) == 0:
        raise ValueError("an empty window has no endpoint")
    if valid is None:
        valid = np.isfinite(channels)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(channels)
    clean = np.where(valid, channels, 0)
    smooth = trailing_valid_mean(clean, valid, width=3)
    velocity = causal_derivative(smooth, fps)
    acceleration = causal_derivative(velocity[:, vertical_channel:vertical_channel + 1], fps)[:, 0]
    vertical_stats = np.asarray((
        acceleration.mean(), acceleration.std(), acceleration.min(), acceleration.max()
    ), dtype=np.float32)
    frame_coverage = valid.all(axis=1)
    channel_coverage = valid.mean(axis=0)
    quality = np.asarray((
        valid.mean(), frame_coverage.mean(), valid[-1].mean(),
        channel_coverage.min(), channel_coverage.mean(), channel_coverage.max(),
        float(len(channels)),
    ), dtype=np.float32)
    feature = np.concatenate((_five_stats(smooth), _five_stats(velocity), vertical_stats, quality))
    if feature.shape != (131,) or not np.isfinite(feature).all():
        raise AssertionError("global feature contract violated")
    return feature.astype(np.float32, copy=False)


@dataclass
class TrainOnlyStandardizer:
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, training: np.ndarray) -> "TrainOnlyStandardizer":
        training = np.asarray(training, dtype=np.float64)
        self.mean = training.mean(axis=0)
        self.scale = training.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("standardizer must be fit on the training split first")
        return ((np.asarray(values) - self.mean) / self.scale).astype(np.float32)
