#!/usr/bin/env python3
"""Deterministic H36M17 -> proxy NTU25 conversion recovered from the audit.

The result is a compatibility proxy, not native Kinect joints and not metric
3-D ground truth.  Joints unavailable in H36M17 are copied from the nearest
wrist/ankle and reported in the returned proxy mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


H36M17_NAMES = (
    "pelvis", "right_hip", "right_knee", "right_ankle", "left_hip",
    "left_knee", "left_ankle", "spine", "thorax", "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist", "right_shoulder",
    "right_elbow", "right_wrist",
)

NTU25_NAMES = (
    "spine_base", "spine_mid", "neck", "head", "left_shoulder",
    "left_elbow", "left_wrist", "left_hand", "right_shoulder",
    "right_elbow", "right_wrist", "right_hand", "left_hip", "left_knee",
    "left_ankle", "left_foot", "right_hip", "right_knee", "right_ankle",
    "right_foot", "spine_shoulder", "left_hand_tip", "left_thumb",
    "right_hand_tip", "right_thumb",
)

# NTU index -> H36M index.  The final hands/feet are explicit proxy copies.
H36M_TO_NTU = np.asarray((
    0, 7, 9, 10, 11, 12, 13, 13, 14, 15, 16, 16, 4, 5, 6, 6, 1, 2, 3, 3,
    8, 13, 13, 16, 16,
), dtype=np.int64)

PROXY_JOINTS = np.asarray((7, 11, 15, 19, 21, 22, 23, 24), dtype=np.int64)


def map_h36m17_to_ntu25(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose)
    if pose.shape[-2:] != (17, 3):
        raise ValueError(f"expected [...,17,3], got {pose.shape}")
    output = np.take(pose, H36M_TO_NTU, axis=-2).copy()
    proxy = np.zeros(25, dtype=bool)
    proxy[PROXY_JOINTS] = True
    return output, proxy


def normalize_proxy_ntu25(pose: np.ndarray, reference_torso: float = 1.0) -> np.ndarray:
    """Apply sequence torso scale, frame SpineMid center and shoulder yaw."""
    output = np.asarray(pose, dtype=np.float32).copy()
    if output.shape[-2:] != (25, 3):
        raise ValueError(f"expected [...,25,3], got {output.shape}")
    flat = output.reshape((-1, 25, 3))
    finite_frame = np.isfinite(flat).all(axis=(1, 2))
    nonzero_frame = np.linalg.norm(flat, axis=-1).max(axis=-1) > 0
    valid = finite_frame & nonzero_frame
    flat[~np.isfinite(flat)] = 0
    if not valid.any():
        return output

    torso = np.linalg.norm(flat[:, 20] - flat[:, 0], axis=-1)
    torso = torso[valid & (torso > 1e-8)]
    if torso.size:
        flat *= float(reference_torso) / float(np.median(torso))

    flat[valid] -= flat[valid, 1:2, :]

    first = int(np.flatnonzero(valid)[0])
    shoulder = flat[first, 8] - flat[first, 4]
    if np.linalg.norm(shoulder[[0, 2]]) > 1e-8:
        yaw = np.arctan2(shoulder[2], shoulder[0])
        c, s = float(np.cos(-yaw)), float(np.sin(-yaw))
        rotation = np.asarray(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=np.float32)
        flat[valid] = flat[valid] @ rotation.T
    return output


def convert(pose: np.ndarray, reference_torso: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    mapped, proxy = map_h36m17_to_ntu25(pose)
    return normalize_proxy_ntu25(mapped, reference_torso), proxy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help=".npy array shaped [T,17,3]")
    parser.add_argument("output", type=Path, help="destination .npy")
    parser.add_argument("--reference-torso", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    pose = np.load(args.input)
    converted, proxy = convert(pose, args.reference_torso)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, converted)
    manifest = {
        "source": str(args.input),
        "output": str(args.output),
        "shape": list(converted.shape),
        "reference_torso": args.reference_torso,
        "proxy_joint_indices_zero_based": np.flatnonzero(proxy).tolist(),
        "native_ntu25": False,
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
