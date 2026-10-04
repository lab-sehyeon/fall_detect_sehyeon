"""Auditable candidates for experimental recovery of SAFER legacy V1.

The lost project mapper incorrectly interpreted MotionAGFormer H36M17 output as
COCO17.  This module reproduces that *class* of error with an explicit COCO
body-to-NTU25 proxy and exposes normalization variants that can be compared
against preserved validation metrics.  None of these candidates is called the
original mapper until the validation experiment supports that conclusion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from data_gen.h36m17_to_ntu25 import H36M_TO_NTU
from data_gen.preprocess.preprocess import angle_between, rotation_matrix


COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)

NTU25_NAMES = (
    "spine_base", "spine_mid", "neck", "head", "left_shoulder",
    "left_elbow", "left_wrist", "left_hand", "right_shoulder",
    "right_elbow", "right_wrist", "right_hand", "left_hip", "left_knee",
    "left_ankle", "left_foot", "right_hip", "right_knee", "right_ankle",
    "right_foot", "spine_shoulder", "left_hand_tip", "left_thumb",
    "right_hand_tip", "right_thumb",
)


@dataclass(frozen=True)
class Candidate:
    name: str
    mapping: str
    alignment: str
    alignment_scope: str
    scale: str

    def manifest(self) -> dict:
        return asdict(self)


CANDIDATES = {
    value.name: value
    for value in (
        Candidate(
            "legacy_coco_umurl_window_noscale",
            "coco_midpoint_proxy",
            "umurl_full_3d",
            "window_first_frame",
            "none",
        ),
        Candidate(
            "legacy_coco_center_only_noscale",
            "coco_midpoint_proxy",
            "none",
            "window_first_frame",
            "none",
        ),
        Candidate(
            "legacy_coco_generic_z11_5_window_noscale",
            "coco_midpoint_proxy",
            "legacy_generic_z11_5",
            "window_first_frame",
            "none",
        ),
        Candidate(
            "legacy_coco_yaw_window_noscale",
            "coco_midpoint_proxy",
            "yaw_only",
            "window_first_frame",
            "none",
        ),
        Candidate(
            "legacy_coco_umurl_sequence_noscale",
            "coco_midpoint_proxy",
            "umurl_full_3d",
            "sequence_first_finite_frame",
            "none",
        ),
        Candidate(
            "legacy_coco_umurl_window_unit_torso",
            "coco_midpoint_proxy",
            "umurl_full_3d",
            "window_first_frame",
            "sequence_median_torso_to_one",
        ),
        Candidate(
            "legacy_coco_umurl_window_ntu_torso",
            "coco_midpoint_proxy",
            "umurl_full_3d",
            "window_first_frame",
            "sequence_median_torso_to_0p5",
        ),
        Candidate(
            "legacy_coco_yaw_window_ntu_torso",
            "coco_midpoint_proxy",
            "yaw_only",
            "window_first_frame",
            "sequence_median_torso_to_0p5",
        ),
        Candidate(
            "legacy_coco_umurl_sequence_ntu_torso",
            "coco_midpoint_proxy",
            "umurl_full_3d",
            "sequence_first_finite_frame",
            "sequence_median_torso_to_0p5",
        ),
        Candidate(
            "legacy_coco_yaw_sequence_ntu_torso",
            "coco_midpoint_proxy",
            "yaw_only",
            "sequence_first_finite_frame",
            "sequence_median_torso_to_0p5",
        ),
        Candidate(
            "correct_h36m_umurl_window_noscale_control",
            "h36m_direct_proxy",
            "umurl_full_3d",
            "window_first_frame",
            "none",
        ),
    )
}


def map_coco_midpoint_proxy(pose: np.ndarray) -> np.ndarray:
    """Map an array labelled COCO17 to an explicit NTU25 compatibility proxy."""
    pose = np.asarray(pose)
    if pose.shape[-2:] != (17, 3):
        raise ValueError(f"expected [...,17,3], got {pose.shape}")
    output = np.empty(pose.shape[:-2] + (25, 3), dtype=pose.dtype)
    hip_center = (pose[..., 11, :] + pose[..., 12, :]) * 0.5
    shoulder_center = (pose[..., 5, :] + pose[..., 6, :]) * 0.5

    output[..., 0, :] = hip_center
    output[..., 1, :] = (hip_center + shoulder_center) * 0.5
    output[..., 2, :] = shoulder_center
    output[..., 3, :] = pose[..., 0, :]
    output[..., 4, :] = pose[..., 5, :]
    output[..., 5, :] = pose[..., 7, :]
    output[..., 6, :] = pose[..., 9, :]
    output[..., 7, :] = pose[..., 9, :]
    output[..., 8, :] = pose[..., 6, :]
    output[..., 9, :] = pose[..., 8, :]
    output[..., 10, :] = pose[..., 10, :]
    output[..., 11, :] = pose[..., 10, :]
    output[..., 12, :] = pose[..., 11, :]
    output[..., 13, :] = pose[..., 13, :]
    output[..., 14, :] = pose[..., 15, :]
    output[..., 15, :] = pose[..., 15, :]
    output[..., 16, :] = pose[..., 12, :]
    output[..., 17, :] = pose[..., 14, :]
    output[..., 18, :] = pose[..., 16, :]
    output[..., 19, :] = pose[..., 16, :]
    output[..., 20, :] = shoulder_center
    output[..., 21, :] = pose[..., 9, :]
    output[..., 22, :] = pose[..., 9, :]
    output[..., 23, :] = pose[..., 10, :]
    output[..., 24, :] = pose[..., 10, :]
    return output


def map_pose(pose: np.ndarray, mapping: str) -> np.ndarray:
    if mapping == "coco_midpoint_proxy":
        return map_coco_midpoint_proxy(pose)
    if mapping == "h36m_direct_proxy":
        pose = np.asarray(pose)
        if pose.shape[-2:] != (17, 3):
            raise ValueError(f"expected [...,17,3], got {pose.shape}")
        return np.take(pose, H36M_TO_NTU, axis=-2).copy()
    raise ValueError(f"unknown mapping: {mapping}")


def alignment_matrix(shoulder: np.ndarray, mode: str) -> np.ndarray:
    shoulder = np.asarray(shoulder, dtype=np.float64)
    if shoulder.shape != (3,):
        raise ValueError(f"expected shoulder vector [3], got {shoulder.shape}")
    if mode == "none":
        return np.eye(3, dtype=np.float32)
    if mode in {"umurl_full_3d", "legacy_generic_z11_5"}:
        target = np.asarray(
            (1.0, 0.0, 0.0)
            if mode == "umurl_full_3d"
            else (0.0, 0.0, 1.0)
        )
        axis = np.cross(shoulder, target)
        return rotation_matrix(axis, angle_between(shoulder, target)).astype(np.float32)
    if mode == "yaw_only":
        if np.linalg.norm(shoulder[[0, 2]]) <= 1e-8:
            return np.eye(3, dtype=np.float32)
        yaw = np.arctan2(shoulder[2], shoulder[0])
        c, s = float(np.cos(-yaw)), float(np.sin(-yaw))
        return np.asarray(
            ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)),
            dtype=np.float32,
        )
    raise ValueError(f"unknown alignment: {mode}")


def alignment_vector(mapped_pose: np.ndarray, mode: str) -> np.ndarray:
    if mode == "legacy_generic_z11_5":
        # Upstream pre_normalization default: joint_idx1=11, joint_idx2=5.
        return mapped_pose[5] - mapped_pose[11]
    if mode in {"umurl_full_3d", "yaw_only"}:
        return mapped_pose[8] - mapped_pose[4]
    if mode == "none":
        return np.zeros(3, dtype=np.float32)
    raise ValueError(f"unknown alignment: {mode}")


def sequence_context(pose17: np.ndarray, candidate: Candidate) -> dict[str, np.ndarray | float]:
    pose17 = np.asarray(pose17)
    if pose17.ndim != 3 or pose17.shape[1:] != (17, 3):
        raise ValueError(f"expected [T,17,3], got {pose17.shape}")
    if (
        candidate.scale == "none"
        and candidate.alignment_scope == "window_first_frame"
    ):
        return {"scale": 1.0}

    finite = np.isfinite(pose17).all(axis=(1, 2))
    if not finite.any():
        raise ValueError("sequence contains no finite mapped frame")
    scale = 1.0
    scale_targets = {
        "sequence_median_torso_to_one": 1.0,
        "sequence_median_torso_to_0p5": 0.5,
    }
    if candidate.scale in scale_targets:
        mapped = map_pose(pose17, candidate.mapping)
        torso = np.linalg.norm(mapped[:, 20] - mapped[:, 0], axis=-1)
        values = torso[finite & (torso > 1e-8)]
        if not values.size:
            raise ValueError("sequence contains no valid torso length")
        scale = scale_targets[candidate.scale] / float(np.median(values))
    elif candidate.scale != "none":
        raise ValueError(f"unknown scale: {candidate.scale}")

    context: dict[str, np.ndarray | float] = {"scale": scale}
    if candidate.alignment_scope == "sequence_first_finite_frame":
        first = int(np.flatnonzero(finite)[0])
        mapped_first = map_pose(pose17[first], candidate.mapping)
        vector = alignment_vector(mapped_first, candidate.alignment)
        context["rotation"] = alignment_matrix(vector, candidate.alignment)
    elif candidate.alignment_scope != "window_first_frame":
        raise ValueError(f"unknown alignment scope: {candidate.alignment_scope}")
    return context


def normalize_window(
    mapped_window: np.ndarray,
    candidate: Candidate,
    context: dict[str, np.ndarray | float],
) -> np.ndarray:
    output = np.asarray(mapped_window, dtype=np.float32).copy()
    if output.shape != (64, 25, 3):
        raise ValueError(f"expected [64,25,3], got {output.shape}")
    if not np.isfinite(output).all():
        raise ValueError("candidate received a non-finite window")
    output *= float(context["scale"])
    output -= output[:, 1:2, :]
    if candidate.alignment_scope == "window_first_frame":
        vector = alignment_vector(output[0], candidate.alignment)
        rotation = alignment_matrix(vector, candidate.alignment)
    else:
        rotation = np.asarray(context["rotation"], dtype=np.float32)
    output = output @ rotation.T
    if not np.isfinite(output).all():
        raise ValueError("normalization produced non-finite coordinates")
    return output.astype(np.float32, copy=False)


def convert_window(
    pose17_window: np.ndarray,
    candidate: Candidate,
    context: dict[str, np.ndarray | float],
) -> np.ndarray:
    return normalize_window(map_pose(pose17_window, candidate.mapping), candidate, context)


def convert_windows(
    pose17_sequence: np.ndarray,
    starts: np.ndarray,
    candidate: Candidate,
    context: dict[str, np.ndarray | float],
) -> np.ndarray:
    """Convert a batch of 64-frame windows without changing candidate semantics.

    Mapping is performed once for the source sequence.  Centering and the
    window-first-frame rotation remain independent for every materialized
    window, matching repeated :func:`convert_window` calls.
    """
    pose17_sequence = np.asarray(pose17_sequence)
    starts = np.asarray(starts, dtype=np.int64)
    if pose17_sequence.ndim != 3 or pose17_sequence.shape[1:] != (17, 3):
        raise ValueError(f"expected [T,17,3], got {pose17_sequence.shape}")
    if starts.ndim != 1:
        raise ValueError(f"expected one-dimensional starts, got {starts.shape}")
    if starts.size == 0:
        return np.empty((0, 64, 25, 3), dtype=np.float32)
    if int(starts.min()) < 0 or int(starts.max()) + 64 > pose17_sequence.shape[0]:
        raise ValueError("window start falls outside the source sequence")

    mapped = map_pose(pose17_sequence, candidate.mapping)
    offsets = np.arange(64, dtype=np.int64)
    output = np.asarray(mapped[starts[:, None] + offsets[None, :]], dtype=np.float32).copy()
    if not np.isfinite(output).all():
        raise ValueError("candidate received a non-finite window batch")
    output *= float(context["scale"])
    output -= output[:, :, 1:2, :]

    if candidate.alignment_scope == "window_first_frame":
        rotations = np.stack(
            [
                alignment_matrix(
                    alignment_vector(window[0], candidate.alignment),
                    candidate.alignment,
                )
                for window in output
            ],
            axis=0,
        )
    elif candidate.alignment_scope == "sequence_first_finite_frame":
        rotation = np.asarray(context["rotation"], dtype=np.float32)
        rotations = np.broadcast_to(rotation, (output.shape[0], 3, 3))
    else:
        raise ValueError(f"unknown alignment scope: {candidate.alignment_scope}")

    output = np.einsum("btvc,bdc->btvd", output, rotations, optimize=True)
    if not np.isfinite(output).all():
        raise ValueError("normalization produced non-finite coordinates")
    return output.astype(np.float32, copy=False)
