#!/usr/bin/env python3
"""Read-only F1 train/validation timeline audit; no model or test/OOD access."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from data_gen.safer_legacy_v1_gendata import derive_four_class
from fall_pipeline.common.integrity import sha256_file

DATA = ROOT / "data/fall_processed/SAFER-Activities/clean3d_v1_reconstructed"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_split(split, expected_count):
    base = DATA / split
    manifest = read(base / "split_manifest.json")
    sequences = read(base / "sequences.json")
    arrays = {}
    for name in ("dense_coarse_labels.npy", "dense_derived_labels.npy", "sequence_index.npy",
                 "window_start.npy", "center_coarse_labels.npy", "center_derived_labels.npy"):
        path = base / name
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        require(len(value) == expected_count, f"count: {split}/{name}")
        require(value.dtype == np.int64, f"dtype: {name}")
        digest = hashlib.sha256()
        for start in range(0, len(value), 8192):
            digest.update(np.asarray(value[start:start + 8192]).tobytes(order="C"))
        require(digest.hexdigest() == manifest["payload"][name]["array_payload_sha256"],
                f"source hash: {split}/{name}")
        arrays[name] = value
    require(sha256_file(base / "sequences.json") == manifest["small_file_sha256"]["sequences.json"],
            "sequence metadata hash")
    coarse = arrays["dense_coarse_labels.npy"]
    derived = arrays["dense_derived_labels.npy"]
    require(coarse.shape == derived.shape == (expected_count, 64), "dense shape")
    require(np.array_equal(coarse[:, 32], arrays["center_coarse_labels.npy"]), "coarse center")
    require(np.array_equal(derived[:, 32], arrays["center_derived_labels.npy"]), "derived center")
    window_counts = np.zeros(4, dtype=np.int64)
    timeline_counts = np.zeros(4, dtype=np.int64)
    timeline_coarse_counts = np.zeros(16, dtype=np.int64)
    covered_frames = total_frames = cursor = 0
    subjects = set()
    for sequence in sequences:
        begin, end = sequence["output_start"], sequence["output_stop"]
        frames = sequence["total_frames"]
        require(begin == cursor and end >= begin, "contiguous sequence offsets")
        cursor = end
        subjects.add(sequence["subject"])
        require(np.all(arrays["sequence_index.npy"][begin:end] == sequence["sequence_index"]),
                "sequence order")
        starts = arrays["window_start.npy"][begin:end]
        require(np.all(starts >= 0) and np.all(starts + 64 <= frames), "window bounds")
        require(np.all(np.diff(starts) > 0), "window order")
        labels = np.asarray(coarse[begin:end])
        labels4 = np.asarray(derived[begin:end])
        require(np.all((labels >= 0) & (labels < 16)), "coarse label range")
        require(np.array_equal(derive_four_class(labels), labels4), "derived label mapping")
        window_counts += np.bincount(labels4.ravel(), minlength=4)
        indices = (starts[:, None] + np.arange(64)[None, :]).ravel()
        minimum = np.full(frames, 16, dtype=np.int64)
        maximum = np.full(frames, -1, dtype=np.int64)
        np.minimum.at(minimum, indices, labels.ravel())
        np.maximum.at(maximum, indices, labels.ravel())
        covered = maximum >= 0
        require(np.array_equal(minimum[covered], maximum[covered]), "overlap label disagreement")
        unique_coarse = maximum[covered]
        timeline_coarse_counts += np.bincount(unique_coarse, minlength=16)
        timeline_counts += np.bincount(derive_four_class(unique_coarse), minlength=4)
        covered_frames += int(covered.sum())
        total_frames += frames
    require(cursor == expected_count, "full sequence/window coverage")
    require(int(window_counts.sum()) == expected_count * 64, "window label counts")
    require(int(timeline_counts.sum()) == covered_frames, "unique frame label counts")
    return {"passed": True, "windows": expected_count, "sequences": len(sequences),
            "subjects": sorted(subjects), "total_frames": total_frames,
            "covered_unique_frames": covered_frames, "uncovered_frames": total_frames - covered_frames,
            "dense_window_class_counts": window_counts.tolist(),
            "unique_covered_frame_class_counts": timeline_counts.tolist(),
            "unique_covered_frame_coarse_counts": timeline_coarse_counts.tolist(),
            "class_count_order": ["other", "fall", "lie_down", "lying_down"],
            "overlap_labels_consistent": True, "source_label_payload_hashes_match": True}


def main():
    splits = {split: audit_split(split, count) for split, count in
              (("train", 599986), ("val", 104589))}
    require(not set(splits["train"]["subjects"]) & set(splits["val"]["subjects"]), "subject leakage")
    print(json.dumps({"passed": True, "scope": "train/val labels and timeline only; no learning",
                      "test_ood_opened": False, "splits": splits,
                      "train_val_subject_disjoint": True,
                      "dense_fp32_train_val_temporal_plus_context_bytes": 187586048000},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
