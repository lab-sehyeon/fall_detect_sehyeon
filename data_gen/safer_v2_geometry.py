"""Fixed V2 reconstruction geometry; never reads labels or legacy 3D."""
from __future__ import annotations
import numpy as np

H36M_TO_NTU = np.array([0,7,9,10,11,12,13,13,14,15,16,16,4,5,6,6,1,2,3,3,8,13,13,16,16])
FLIP = np.array([0,4,5,6,1,2,3,7,8,9,10,14,15,16,11,12,13])


def starts_for(frames, window=243, stride=243):
    if frames <= 0 or window <= 0 or stride <= 0 or stride > window:
        raise ValueError("invalid timeline")
    if frames <= window:
        return np.array([0], np.int64)
    starts = list(range(0, frames-window+1, stride))
    if starts[-1] != frames-window:
        starts.append(frames-window)
    return np.array(starts, np.int64)


def coco_xy_to_h36m(keypoints):
    """SAFER author's coco_to_h36m assignments, explicitly cast to float32."""
    x = np.asarray(keypoints, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (17, 2) or not np.isfinite(x).all():
        raise ValueError("finite [T,17,2] required")
    y = np.zeros_like(x)
    y[:, [1,2,3,4,5,6,9,11,12,13,14,15,16]] = x[:, [12,14,16,11,13,15,0,5,7,9,6,8,10]]
    y[:, 0] = (x[:, 11] + x[:, 12]) * .5
    y[:, 8] = (x[:, 5] + x[:, 6]) * .5
    y[:, 7] = (y[:, 0] + y[:, 8]) * .5
    y[:, 10] = (x[:, 1] + x[:, 2]) * .5
    return y


def model_input(keypoints, score, width, height):
    xy = coco_xy_to_h36m(keypoints)
    score = np.array(score, dtype=np.float32, copy=True)
    if score.shape != xy.shape[:2] or not np.isfinite([width, height]).all() or min(width, height) <= 0:
        raise ValueError("input confidence/dimensions")
    invalid = ~np.isfinite(score)
    score[invalid] = 0
    # Preserve official SAFER's confidence index policy; XY alone is reordered.
    xy = xy / float(width) * 2 - np.array([1., float(height)/float(width)], np.float32)
    return np.concatenate((xy, score[..., None]), -1), {"values": int(invalid.sum()), "frames": int(invalid.any(1).sum())}


def windows_at(values, starts, width=243):
    if not len(values) or np.any(starts < 0) or np.any(starts >= len(values)):
        raise ValueError("window indices")
    indices = np.minimum(starts[:, None] + np.arange(width), len(values)-1)
    return np.ascontiguousarray(values[indices])


def flip_numpy(values):
    flipped = np.array(values[..., FLIP, :], copy=True)
    flipped[..., 0] *= -1
    return flipped


def fuse_windows(predictions, starts, frames, independent=False):
    p = np.asarray(predictions)
    if p.shape != (len(starts), 243, 17, 3) or not np.isfinite(p).all():
        raise ValueError("lifting output shape/finite")
    total = np.zeros((frames, 17, 3), np.float64)
    coverage = np.zeros(frames, np.int64)
    if independent:
        indices = starts[:, None] + np.arange(243)
        valid = indices < frames
        np.add.at(total, indices[valid], p[valid])
        coverage = np.bincount(indices[valid], minlength=frames)
    else:
        for start, window in zip(starts, p):
            width = min(243, frames-int(start))
            total[start:start+width] += window[:width]
            coverage[start:start+width] += 1
    if np.any(coverage == 0):
        raise ValueError("uncovered lifting frames")
    return (total / coverage[:, None, None]).astype(np.float32), coverage


def normalize_ntu(h36m, reference_torso=.5):
    h36m = np.asarray(h36m, dtype=np.float32)
    if h36m.ndim != 3 or h36m.shape[1:] != (17, 3) or not len(h36m) or not np.isfinite(h36m).all():
        raise ValueError("finite nonempty H36M sequence required")
    mapped = h36m[:, H36M_TO_NTU].copy()
    torso = np.linalg.norm(mapped[:, 20] - mapped[:, 0], axis=1)
    valid = torso > 1e-8
    scale = float(reference_torso / np.median(torso[valid])) if valid.any() else 1.
    scaled = mapped * scale
    shoulder = scaled[:, 8] - scaled[:, 4]
    usable = np.flatnonzero(np.linalg.norm(shoulder[:, [0, 2]], axis=1) > 1e-8)
    first = int(usable[0]) if len(usable) else None
    angle = float(np.arctan2(shoulder[first, 2], shoulder[first, 0])) if first is not None else 0.
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c,0,s], [0,1,0], [-s,0,c]], np.float32)
    root = scaled[:, 1].copy() @ rotation.T
    centered = scaled - scaled[:, 1:2]
    normalized = centered @ rotation.T
    return normalized, root, {"scale": scale, "reference_torso": reference_torso, "yaw": angle,
                              "reference_frame": first, "degenerate_torso_frames": int((~valid).sum())}


def lying_origin(labels):
    y = np.asarray(labels)
    result = np.full(y.shape, -1, np.int64)
    previous = -1
    origin = -1
    for i, label in enumerate(y):
        if label == 12:
            if previous != 12:
                origin = 0 if previous == 10 else 1 if previous == 11 else -1
            result[i] = origin
        previous = label
    return result
