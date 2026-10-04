#!/usr/bin/env python3
"""Read-only prerequisites audit, not an authorization to run downstream science.

No pickle/checkpoint deserialization, GPU inference, downloading, or deletion.
Optional JSON output is created exclusively; existing files are never replaced.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
F1 = "checkpoint/fall/F1_SAFER_DOCUMENT_RECONSTRUCTION_20260920_R2"
FU = "data/fall_processed/FU-Kinect-Fall/ntu25_official_umurl_v1"
SAFER = "data/source_archives/SAFER-Activities/pose_bboxes/extracted/3d_keypoints_pickle_ntu_format"
MOTION = "checkpoint/motionagformer/motionagformer-b-h36m.pth.tr"
MOTION_SHA = "eb15d088aadf193524a0d6c4a40680a324292db2781f3f45abe3400d5a4b343b"
UPSTREAM_COMMIT = "4756fd1eb7cc73f0e991f091ff2280e030ab85f3"
STAGES = [
    {
        "name": "fu_zs0_zs1", "record": "3.10",
        "required_modules": ["fall_pipeline/fu/eval_f1_fu_transfer.py", "fall_pipeline/fu/eval_f1_fu_transfer_windows.py"],
        "unresolved": ["clip aggregation and binary decision", "ZS0 interpolation contract", "aligned25 sampling indices and length rounding", "ZS1 final-window policy", "use of reconstructed instead of original F1 lineage"],
    },
    {
        "name": "fu_d0_d1_d2", "record": "3.10",
        "required_modules": ["fall_pipeline/fu/fu_d0_dste_probe_windows.py", "fall_pipeline/fu/fu_d1_safer_adapter_probe.py", "fall_pipeline/fu/fu_d2_concat_probe.py"],
        "unresolved": ["window-to-clip feature pooling", "probe optimizer and schedule", "train-only scaler and regularization", "epoch selection details"],
    },
    {
        "name": "primitive_p0_p1", "record": "3.11",
        "required_modules": ["fall_pipeline/primitives/audit_motion_primitives.py", "fall_pipeline/primitives/train_p1_safer_primitives.py"],
        "unresolved": ["exact 12-signal definitions and 156-D aggregation", "P1 optimizer and learning rate", "quarter-training subset selection"],
    },
    {
        "name": "safer_v2_v3", "record": "3.18-3.21,3.43-3.50",
        "required_modules": ["data_gen/lift_safer_pose3d_v2.py", "data_gen/audit_safer_v2.py", "data_gen/lift_safer_pose3d_v3.py", "data_gen/audit_safer_v3.py", "data_gen/safer_v3_sequence_cache.py"],
        "unresolved": ["complete historical lifting preprocessing/inference contract", "normalization details and reference torso", "lost pilot metric implementations and baseline lineage"],
    },
    {
        "name": "joint_j0_j1_locked_nested_final", "record": "3.12-3.14,3.22-3.29,3.51-3.57",
        "required_modules": ["fall_pipeline/joint/train_j0_joint_heads.py", "fall_pipeline/joint/train_j1_shared_adapter.py", "fall_pipeline/joint/eval_j0_j1_safer_locked.py", "fall_pipeline/joint/train_nested_j0_j1_fu.py", "fall_pipeline/joint/train_final_joint_fall.py"],
        "unresolved": ["optimizer type and schedule gamma not established by preserved CLI flags", "loss weighting and development selection", "recomputed nested median versus historical final epoch counts"],
    },
    {
        "name": "rgb_global_external", "record": "E06-E10,E13",
        "required_modules": [],
        "unresolved": ["detector/pose model asset identity and frontend completion", "external dataset inventory and access terms", "matching historical evaluation units and decoder contracts", "upstream final J1 dependency"],
    },
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify_file(path, expected_hash, expected_bytes=None):
    require(path.is_file(), f"missing file: {path}")
    size = path.stat().st_size
    require(expected_bytes is None or size == expected_bytes, f"size mismatch: {path}")
    actual = digest(path)
    require(actual == expected_hash, f"hash mismatch: {path}")
    return {"bytes": size, "sha256": actual}


def validate_fu_arrays(data, frames, labels, actions, subjects, folds):
    count = len(frames)
    require(data.shape == (count, 3, 300, 25, 2), "FU data shape")
    require(data.dtype == np.float32, "FU data dtype")
    require(all(x.shape == (count,) for x in (labels, actions, subjects, folds)), "FU aligned arrays")
    require(np.all((frames >= 1) & (frames <= 300)), "FU frame bounds")
    require(np.array_equal(folds, (subjects - 1) % 5), "FU fold rule")
    require(np.array_equal(labels, (actions == 5).astype(labels.dtype)), "FU fall label mapping")
    lengths = []
    for index, length in enumerate(frames):
        length = int(length)
        clip = data[index]
        require(np.isfinite(clip).all(), f"FU nonfinite clip {index}")
        require(not np.any(clip[:, :, :, 1]), f"FU second person {index}")
        require(not np.any(clip[:, length:]), f"FU trailing padding {index}")
        lengths.append(length)
    regular = sum(1 if t < 64 else (t - 64) // 8 + 1 for t in lengths)
    tail_needed = sum(t > 64 and (t - 64) % 8 != 0 for t in lengths)
    return {
        "clips": count, "frames": sum(lengths), "length_min": min(lengths), "length_max": max(lengths),
        "shorter_than_64": sum(t < 64 for t in lengths),
        "tail_ambiguous_clips": tail_needed,
        "native30_regular_stride8_windows": regular,
        "native30_with_end_anchor_windows": regular + tail_needed,
        "window_counts_are_policy_comparison_not_selected_protocol": True,
    }


def audit_fu(root):
    base = root / FU
    manifest = read(base / "preprocess_manifest.json")
    require(manifest["included_clips"] == 993 and manifest["integrity"]["passed"], "FU manifest gate")
    hashes = {}
    for name, expected in manifest["output_hashes_sha256"].items():
        require(Path(name).name == name, "unsafe FU manifest path")
        hashes[name] = verify_file(base / name, expected)
    names = ("data_joint", "num_frame", "labels", "action_ids", "subjects", "fold_ids")
    arrays = [np.load(base / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in names]
    require(len(arrays[1]) == 993 and int(arrays[2].sum()) == 165, "FU sample counts")
    result = validate_fu_arrays(*arrays)
    result["files_verified"] = hashes
    return result


def audit_f1(root):
    base = root / F1
    final, audit, lock = [read(base / name) for name in ("final_report.json", "independent_audit.json", "selection_lock.json")]
    require(final["status"] == "completed" and audit["passed"], "F1 final audit gate")
    require(lock == final["selection_lock"], "F1 selection mismatch")
    require(lock["candidate"] == "temporal_spatial__sqrt" and lock["epoch"] == 17, "F1 selected lineage")
    files = {
        "report": verify_file(base / "final_report.json", audit["report_sha256"]),
        "training_report": verify_file(base / "training/report.json", lock["training_report_sha256"]),
        "selected_head": verify_file(base / "training" / lock["candidate"] / "best.pt", lock["head_sha256"]),
        "config": verify_file(root / "configs/f1_safer_document_reconstruction_v2.json", lock["config_sha256"]),
    }
    return {"selection": lock, "files_verified": files, "historical_exact_reproduction": False}


def audit_safer(root):
    original = read(root / "data/fall_processed/SAFER-Activities/raw_audit_v1/audit_report_r2.json")
    names = {"normal": "aic_normal_dataset_with_3d.pkl", "ood": "aic_normal_test_set_with_split_3d.pkl"}
    result = {}
    for split, filename in names.items():
        spec = original["pinned_files"][split]
        result[split] = verify_file(root / SAFER / filename, spec["sha256"], spec["bytes"])
        result[split]["previously_audited_sequences"] = spec["sequences"]
        result[split]["previously_audited_frames"] = spec["frames"]
    result["pickle_deserialized"] = False
    result["labels_used_for_selection"] = False
    return result


def audit_motion(root):
    result = verify_file(root / MOTION, MOTION_SHA, 141930389)
    commit = subprocess.check_output(["git", "-C", str(root / "third_party/MotionAGFormer"), "rev-parse", "HEAD"], text=True).strip()
    require(commit == UPSTREAM_COMMIT, "MotionAGFormer upstream revision")
    provenance = read(root / "checkpoint/motionagformer/motionagformer-b-h36m.provenance.json")
    require(provenance["passed"] and provenance["strict_load"], "MotionAGFormer prior strict load")
    require(provenance["sha256"] == MOTION_SHA, "MotionAGFormer provenance")
    result.update({"upstream_commit": commit, "previous_cpu_strict_load": True, "new_inference": False})
    return result


def stage_inventory(root):
    return [{**stage, "missing_modules": [name for name in stage["required_modules"] if not (root / name).is_file()],
             "experiment_execution_ready": False, "reason": "unresolved historical contract; requires explicit reconstruction decision"}
            for stage in STAGES]


def audit_all(root):
    checks = {}
    for name, function in (("fu", audit_fu), ("f1", audit_f1), ("safer_raw", audit_safer), ("motionagformer", audit_motion)):
        try:
            checks[name] = {"passed": True, "details": function(root)}
        except Exception as error:
            checks[name] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    free = shutil.disk_usage(root).free
    materialized = 1007723 * 3 * 64 * 25 * 2 * 4
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(), "scope": "read-only asset audit; no downstream experiment",
        "assets_passed": all(row["passed"] for row in checks.values()), "checks": checks,
        "disk": {"free_bytes": free, "v2_joint_array_estimated_payload_bytes": materialized,
                 "v3_sequence_xyz_estimated_payload_bytes": 8091357 * 25 * 3 * 4,
                 "reserve_bytes": 64 * 1024**3, "estimates_exclude_headers_labels_caches_models": True},
        "stages": stage_inventory(root), "all_experiments_complete": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        output = args.output.resolve()
        require(output.parent == ROOT / "logs", "audit output must be a direct child of project logs")
        require(not output.exists(), "refusing to overwrite audit output")
    report = audit_all(ROOT)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(json.dumps({"assets_passed": report["assets_passed"], "checks": {k: v["passed"] for k, v in report["checks"].items()},
                      "disk": report["disk"], "output": str(args.output), "experiment_execution_ready": False}, indent=2))
    raise SystemExit(0 if report["assets_passed"] else 1)


if __name__ == "__main__":
    main()
