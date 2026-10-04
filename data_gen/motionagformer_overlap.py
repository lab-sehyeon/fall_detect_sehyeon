"""Label-blind V3 overlap-add policy for 243-frame MotionAGFormer windows."""

from __future__ import annotations

import numpy as np


WINDOW = 243
STRIDE = 121
EDGE_FLOOR = 0.05


def triangular_weights(length: int = WINDOW, edge_floor: float = EDGE_FLOOR) -> np.ndarray:
    if length < 1 or not 0 <= edge_floor <= 1:
        raise ValueError("invalid triangular window arguments")
    center = (length - 1) / 2.0
    distance = np.abs(np.arange(length, dtype=np.float64) - center)
    weight = 1.0 - distance / max(center, 1.0)
    return np.maximum(weight, edge_floor).astype(np.float32)


def window_starts(frame_count: int, window: int = WINDOW, stride: int = STRIDE) -> list[int]:
    if frame_count <= window:
        return [0]
    starts = list(range(0, frame_count - window + 1, stride))
    final = frame_count - window
    if starts[-1] != final:
        starts.append(final)
    return starts


def overlap_add(predictions: list[np.ndarray], starts: list[int], frame_count: int) -> np.ndarray:
    if len(predictions) != len(starts) or not predictions:
        raise ValueError("predictions and starts must be non-empty and aligned")
    tail_shape = predictions[0].shape[1:]
    total = np.zeros((frame_count,) + tail_shape, dtype=np.float64)
    divisor = np.zeros((frame_count,) + (1,) * len(tail_shape), dtype=np.float64)
    for prediction, start in zip(predictions, starts):
        prediction = np.asarray(prediction)
        width = min(prediction.shape[0], frame_count - start)
        weight = triangular_weights(prediction.shape[0])[:width]
        expand = weight.reshape((width,) + (1,) * len(tail_shape))
        total[start:start + width] += prediction[:width] * expand
        divisor[start:start + width] += expand
    if np.any(divisor == 0):
        raise ValueError("overlap-add left uncovered frames")
    return (total / divisor).astype(predictions[0].dtype, copy=False)
