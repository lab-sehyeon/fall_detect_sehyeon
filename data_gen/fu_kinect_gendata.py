#!/usr/bin/env python3
"""Build the documented FU-Kinect-Fall NTU25/UmURL-v1 input.

The original project file was lost.  This recovery is deliberately limited to
the contract preserved in the project record and facts that can be reproduced
from the official archive:

* MATLAB ``iskelet`` matrices are ``[T, 60]`` Kinect-v1 joint triplets.
* boundary all-zero frames are trimmed and internal all-zero frames interpolated;
* exact duplicates are handled without leaking one recording across subjects;
* Kinect-v1 20 joints are deterministically mapped to proxy NTU25;
* SpineMid is centered frame-wise and the first-frame shoulder vector is aligned
  with the positive x axis using the repository's UmURL rotation implementation;
* fold assignment is ``(subject_id - 1) % 5``.

The command refuses to write into a non-empty destination.  Raw files are read
only and every inclusion/exclusion is recorded in machine-readable manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat

from data_gen.preprocess.preprocess import rotation_matrix


ACTIONS = ("walking", "bending", "sitting", "squatting", "lying", "falling")
ACTION_TO_ID = {name: index for index, name in enumerate(ACTIONS)}
FALL_ACTION = "falling"

# Kinect for Windows v1 skeleton order:
# HipCenter, Spine, ShoulderCenter, Head, left arm (4), right arm (4),
# left leg (4), right leg (4).  NTU indices 0..19 have the same order.
# NTU SpineShoulder and the four distal hand joints have no separate Kinect-v1
# measurement here, so the nearest available source joint is copied.
KINECT20_TO_NTU25 = np.asarray(
    tuple(range(20)) + (2, 7, 7, 11, 11), dtype=np.int64
)
PROXY_NTU25_INDICES = np.asarray((20, 21, 22, 23, 24), dtype=np.int64)
EXPECTED_RAW_COUNT = 1006
EXPECTED_VALID_COUNT = 993
MAX_FRAMES = 300
FOLD_COUNT = 5


@dataclass
class RawClip:
    path: Path
    relpath: str
    action: str
    subject: int
    repeat: int
    raw: np.ndarray
    content_sha256: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def parse_clip(path: Path, root: Path) -> RawClip:
    rel = path.relative_to(root)
    if len(rel.parts) != 3:
        raise ValueError(f"unexpected skeleton path layout: {rel.as_posix()}")
    action, subject_text, filename = rel.parts
    if action not in ACTION_TO_ID:
        raise ValueError(f"unexpected action directory: {rel.as_posix()}")
    try:
        subject = int(subject_text)
    except ValueError as error:
        raise ValueError(f"invalid subject directory: {rel.as_posix()}") from error
    match = re.search(r"_(\d+)\.mat$", filename, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"cannot parse repetition number: {rel.as_posix()}")
    repeat = int(match.group(1))

    loaded = loadmat(path)
    if "iskelet" not in loaded:
        raise ValueError(f"missing MATLAB variable 'iskelet': {rel.as_posix()}")
    raw = np.asarray(loaded["iskelet"])
    if raw.ndim != 2 or raw.shape[1] != 60 or raw.shape[0] < 1:
        raise ValueError(f"expected iskelet [T,60], got {raw.shape}: {rel.as_posix()}")
    if not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all():
        raise ValueError(f"non-numeric or non-finite skeleton: {rel.as_posix()}")
    return RawClip(
        path=path,
        relpath=rel.as_posix(),
        action=action,
        subject=subject,
        repeat=repeat,
        raw=raw,
        content_sha256=matrix_sha256(raw),
    )


def discover_clips(root: Path) -> list[RawClip]:
    paths = list(root.glob("*/*/*.mat"))
    clips = [parse_clip(path, root) for path in paths]
    clips.sort(key=lambda clip: (
        ACTION_TO_ID[clip.action], clip.subject, clip.repeat, clip.relpath
    ))
    return clips


def exclusion_policy(clips: Iterable[RawClip]) -> dict[str, dict[str, object]]:
    """Return exclusions reproducing the documented 1006 -> 993 audit.

    Cross-subject exact duplicates are both removed because their subject
    provenance is ambiguous.  For a duplicate within one subject, the first
    repetition/path is retained.  Repetition numbers outside the official 1..8
    design are removed as unexpected records.
    """
    clips = list(clips)
    excluded: dict[str, dict[str, object]] = {}
    for clip in clips:
        if clip.repeat not in range(1, 9):
            excluded[clip.relpath] = {
                "reason": "unexpected_repetition_outside_1_to_8",
                "repeat": clip.repeat,
            }

    groups: dict[tuple[tuple[int, ...], str], list[RawClip]] = defaultdict(list)
    for clip in clips:
        groups[(tuple(clip.raw.shape), clip.content_sha256)].append(clip)
    for group in groups.values():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda clip: (clip.subject, clip.repeat, clip.relpath))
        subjects = sorted({clip.subject for clip in ordered})
        members = [clip.relpath for clip in ordered]
        if len(subjects) > 1:
            for clip in ordered:
                excluded[clip.relpath] = {
                    "reason": "exact_duplicate_across_subjects_exclude_all",
                    "subjects": subjects,
                    "duplicate_group": members,
                    "content_sha256": clip.content_sha256,
                }
        else:
            kept = ordered[0]
            for clip in ordered[1:]:
                excluded[clip.relpath] = {
                    "reason": "exact_duplicate_within_subject_keep_first",
                    "kept": kept.relpath,
                    "duplicate_group": members,
                    "content_sha256": clip.content_sha256,
                }
    return excluded


def repair_zero_frames(raw: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    pose = np.asarray(raw, dtype=np.float64).reshape((-1, 20, 3)).copy()
    zero = np.all(pose == 0, axis=(1, 2))
    valid_indices = np.flatnonzero(~zero)
    if valid_indices.size == 0:
        raise ValueError("clip contains only all-zero frames")
    start, stop = int(valid_indices[0]), int(valid_indices[-1]) + 1
    pose = pose[start:stop]
    zero = np.all(pose == 0, axis=(1, 2))
    internal_indices = np.flatnonzero(zero)
    if internal_indices.size:
        valid = np.flatnonzero(~zero)
        timeline = np.arange(pose.shape[0])
        flattened = pose.reshape((pose.shape[0], -1))
        for column in range(flattened.shape[1]):
            flattened[internal_indices, column] = np.interp(
                timeline[internal_indices], valid, flattened[valid, column]
            )
        pose = flattened.reshape((-1, 20, 3))
    if not np.isfinite(pose).all() or np.any(np.all(pose == 0, axis=(1, 2))):
        raise ValueError("zero-frame repair did not produce a finite valid sequence")
    return pose, {
        "original_frames": int(raw.shape[0]),
        "trimmed_leading_zero_frames": start,
        "trimmed_trailing_zero_frames": int(raw.shape[0] - stop),
        "interpolated_internal_zero_frames": internal_indices.astype(int).tolist(),
        "processed_frames": int(pose.shape[0]),
    }


def map_kinect20_to_ntu25(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose)
    if pose.ndim != 3 or pose.shape[1:] != (20, 3):
        raise ValueError(f"expected [T,20,3], got {pose.shape}")
    return np.take(pose, KINECT20_TO_NTU25, axis=1).copy()


def normalize_official_umurl(pose: np.ndarray) -> np.ndarray:
    """Apply the recorded UmURL-compatible center/first-frame alignment."""
    output = np.asarray(pose, dtype=np.float32).copy()
    if output.ndim != 3 or output.shape[1:] != (25, 3):
        raise ValueError(f"expected [T,25,3], got {output.shape}")
    output -= output[:, 1:2, :]
    shoulder = output[0, 8] - output[0, 4]
    target = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    norm = float(np.linalg.norm(shoulder))
    if norm > 1e-8:
        axis = np.cross(shoulder, target)
        cosine = float(np.clip(np.dot(shoulder / norm, target), -1.0, 1.0))
        matrix = rotation_matrix(axis, float(np.arccos(cosine)))
        output[:] = np.einsum("ij,tkj->tki", matrix, output)
    return output


def prepare_clip(clip: RawClip) -> tuple[np.ndarray, dict[str, object]]:
    repaired, repair = repair_zero_frames(clip.raw)
    mapped = map_kinect20_to_ntu25(repaired)
    normalized = normalize_official_umurl(mapped)
    center_error = float(np.max(np.abs(normalized[:, 1])))
    shoulder_error = float(np.max(np.abs((normalized[0, 8] - normalized[0, 4])[1:])))
    if center_error > 1e-5 or shoulder_error > 1e-5:
        raise ValueError(
            f"normalization check failed for {clip.relpath}: "
            f"center={center_error}, shoulder={shoulder_error}"
        )
    repair.update({
        "spine_mid_center_max_abs_error": center_error,
        "first_frame_shoulder_yz_max_abs_error": shoulder_error,
    })
    return normalized, repair


def ensure_empty_destination(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)


def json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    archive = args.archive.resolve() if args.archive else None
    if archive is not None and not archive.is_file():
        raise FileNotFoundError(archive)

    clips = discover_clips(input_root)
    if len(clips) != args.expected_raw_count:
        raise RuntimeError(
            f"raw skeleton count mismatch: {len(clips)} != {args.expected_raw_count}"
        )
    exclusions = exclusion_policy(clips)
    included = [clip for clip in clips if clip.relpath not in exclusions]
    if len(included) != args.expected_valid_count:
        raise RuntimeError(
            f"processed count mismatch: {len(included)} != {args.expected_valid_count}; "
            f"excluded={len(exclusions)}"
        )

    ensure_empty_destination(output_root)
    data_path = output_root / "data_joint.npy"
    data = np.lib.format.open_memmap(
        data_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(included), 3, args.max_frames, 25, 2),
    )
    data[:] = 0
    labels = np.empty(len(included), dtype=np.int64)
    action_ids = np.empty(len(included), dtype=np.int64)
    subjects = np.empty(len(included), dtype=np.int64)
    repeats = np.empty(len(included), dtype=np.int64)
    fold_ids = np.empty(len(included), dtype=np.int64)
    num_frames = np.empty(len(included), dtype=np.int64)
    sample_rows: list[dict[str, object]] = []

    for index, clip in enumerate(included):
        normalized, repair = prepare_clip(clip)
        frames = normalized.shape[0]
        if frames > args.max_frames:
            raise RuntimeError(
                f"clip longer than max_frames={args.max_frames}: {clip.relpath} ({frames})"
            )
        data[index, :, :frames, :, 0] = normalized.transpose((2, 0, 1))
        labels[index] = int(clip.action == FALL_ACTION)
        action_ids[index] = ACTION_TO_ID[clip.action]
        subjects[index] = clip.subject
        repeats[index] = clip.repeat
        fold_ids[index] = (clip.subject - 1) % args.fold_count
        num_frames[index] = frames
        sample_rows.append({
            "index": index,
            "sample_name": Path(clip.relpath).with_suffix("").as_posix(),
            "source_relpath": clip.relpath,
            "action": clip.action,
            "action_id": int(action_ids[index]),
            "binary_label": int(labels[index]),
            "subject": clip.subject,
            "repeat": clip.repeat,
            "fold": int(fold_ids[index]),
            "source_content_sha256": clip.content_sha256,
            **repair,
        })
    data.flush()
    del data

    np.save(output_root / "num_frame.npy", num_frames)
    np.save(output_root / "labels.npy", labels)
    np.save(output_root / "action_ids.npy", action_ids)
    np.save(output_root / "subjects.npy", subjects)
    np.save(output_root / "repeats.npy", repeats)
    np.save(output_root / "fold_ids.npy", fold_ids)
    proxy_mask = np.zeros(25, dtype=bool)
    proxy_mask[PROXY_NTU25_INDICES] = True
    np.save(output_root / "proxy_joint_mask.npy", proxy_mask)
    sample_names = [str(row["sample_name"]) for row in sample_rows]
    with (output_root / "label.pkl").open("wb") as stream:
        pickle.dump((sample_names, labels.tolist()), stream)

    fieldnames = list(sample_rows[0])
    with (output_root / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_rows)

    excluded_rows = []
    by_relpath = {clip.relpath: clip for clip in clips}
    for relpath in sorted(exclusions):
        clip = by_relpath[relpath]
        excluded_rows.append({
            "source_relpath": relpath,
            "action": clip.action,
            "subject": clip.subject,
            "repeat": clip.repeat,
            **exclusions[relpath],
        })
    json_dump(output_root / "exclusions.json", excluded_rows)

    fold_manifest = {
        str(fold): {
            "validation_subjects": sorted(np.unique(subjects[fold_ids == fold]).astype(int).tolist()),
            "training_subjects": sorted(np.unique(subjects[fold_ids != fold]).astype(int).tolist()),
            "validation_indices": np.flatnonzero(fold_ids == fold).astype(int).tolist(),
            "training_indices": np.flatnonzero(fold_ids != fold).astype(int).tolist(),
        }
        for fold in range(args.fold_count)
    }
    json_dump(output_root / "folds.json", fold_manifest)

    output_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_root.iterdir())
        if path.is_file() and path.name != "preprocess_manifest.json"
    }
    reason_counts = Counter(str(item["reason"]) for item in excluded_rows)
    manifest = {
        "dataset": "FU-Kinect-Fall",
        "revision": "ntu25_official_umurl_v1_recovered",
        "source_root": str(input_root),
        "source_archive": str(archive) if archive else None,
        "source_archive_sha256": sha256_file(archive) if archive else None,
        "official_dataset_url": "https://github.com/MuzafferAslan23/Fall-Detection-Dataset",
        "paper_doi": "10.17341/gazimmfd.369347",
        "official_design": {"subjects": 21, "actions": 6, "repetitions": 8, "reported_clips": 1008},
        "observed_raw_skeleton_clips": len(clips),
        "included_clips": len(included),
        "excluded_clips": len(excluded_rows),
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "actions": list(ACTIONS),
        "binary_label": {"positive": "falling", "negative": list(ACTIONS[:-1])},
        "included_by_action": dict(sorted(Counter(row["action"] for row in sample_rows).items())),
        "included_by_binary_label": {
            "0": int((labels == 0).sum()),
            "1": int((labels == 1).sum()),
        },
        "subjects": sorted(np.unique(subjects).astype(int).tolist()),
        "fold_count": args.fold_count,
        "fold_rule": "(subject_id - 1) % 5",
        "data_shape": [len(included), 3, args.max_frames, 25, 2],
        "data_dtype": "float32",
        "source_layout": "iskelet[T,60] interpreted as T x KinectV1-20 x XYZ",
        "kinect20_to_ntu25_source_indices_zero_based": KINECT20_TO_NTU25.astype(int).tolist(),
        "proxy_ntu25_indices_zero_based": PROXY_NTU25_INDICES.astype(int).tolist(),
        "zero_frame_policy": "trim boundary all-zero frames; linearly interpolate internal all-zero frames",
        "normalization": "frame-wise NTU SpineMid(index 1) center; first-frame left-to-right shoulder vector aligned to +x with UmURL rotation",
        "clip_labels_are_not_frame_labels": True,
        "output_hashes_sha256": output_hashes,
        "integrity": {
            "passed": True,
            "raw_count_matches": len(clips) == args.expected_raw_count,
            "valid_count_matches": len(included) == args.expected_valid_count,
            "all_21_subjects_present": np.unique(subjects).size == 21,
            "folds_subject_disjoint": all(
                not (set(value["validation_subjects"]) & set(value["training_subjects"]))
                for value in fold_manifest.values()
            ),
            "falling_count_matches_historical_record": int((labels == 1).sum()) == 165,
            "lying_count_matches_historical_record": int(
                sum(row["action"] == "lying" for row in sample_rows)
            ) == 168,
        },
    }
    if not all(manifest["integrity"].values()):
        raise RuntimeError(f"integrity gate failed: {manifest['integrity']}")
    json_dump(output_root / "preprocess_manifest.json", manifest)
    print(json.dumps({
        "output_root": str(output_root),
        "raw_clips": len(clips),
        "included_clips": len(included),
        "excluded_clips": len(excluded_rows),
        "data_shape": manifest["data_shape"],
        "included_by_action": manifest["included_by_action"],
        "fold_validation_sizes": {
            key: len(value["validation_indices"]) for key, value in fold_manifest.items()
        },
        "integrity": manifest["integrity"],
    }, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--expected-raw-count", type=int, default=EXPECTED_RAW_COUNT)
    parser.add_argument("--expected-valid-count", type=int, default=EXPECTED_VALID_COUNT)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--fold-count", type=int, default=FOLD_COUNT)
    args = parser.parse_args()
    if args.max_frames < 1 or args.fold_count != FOLD_COUNT:
        parser.error("max-frames must be positive and the recovered protocol requires 5 folds")
    return args


if __name__ == "__main__":
    build(parse_args())
