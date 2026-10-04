#!/usr/bin/env python3
"""Count-only preflight for the recovered historical SAFER clean3d_v1 input.

This command does not materialize skeleton windows and does not use a GPU.  It
recovers only the parts of the lost historical ``safer_gendata.py`` contract
that can be proved from the pinned official release and the preserved project
record:

* official cross-subject train/test membership;
* a subject-disjoint validation subset uniquely determined by both the
  preserved clean3d_v1 and corrected-v2 window counts;
* contiguous 64-frame windows with stride 8 and no appended tail window;
* exclusion of every window containing a non-finite legacy-3D frame.

The topology conversion and normalization are deliberately outside this
count-only gate.  The original generator source was lost, so no unproved
mapping behavior is introduced here.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from data_gen.inspect_safer_activities import (
    PINNED_FILES,
    load_pinned_pickle,
    normalized_name,
    sha256_file,
)


WINDOW_SIZE = 64
STRIDE = 8
VALIDATION_SUBJECTS = frozenset((2, 12, 24, 25, 34, 35))
EXPECTED_SPLITS = {
    "train": {"sequences": 298, "candidate_windows": 609_183, "clean_windows": 599_986},
    "val": {"sequences": 73, "candidate_windows": 106_680, "clean_windows": 104_589},
    "test": {"sequences": 96, "candidate_windows": 219_896, "clean_windows": 216_768},
    "ood": {"sequences": 30, "candidate_windows": 71_964, "clean_windows": 71_964},
}
EXPECTED_BAD_FRAMES = 93_256

DEFAULT_PICKLE_ROOT = Path(
    "data/source_archives/SAFER-Activities/pose_bboxes/extracted/"
    "3d_keypoints_pickle_ntu_format"
)
DEFAULT_NORMAL_PICKLE = DEFAULT_PICKLE_ROOT / "aic_normal_dataset_with_3d.pkl"
DEFAULT_OOD_PICKLE = DEFAULT_PICKLE_ROOT / "aic_normal_test_set_with_split_3d.pkl"
DEFAULT_RAW_AUDIT = Path(
    "data/fall_processed/SAFER-Activities/raw_audit_v1/audit_report_manual.json"
)
DEFAULT_OUTPUT = Path(
    "data/fall_processed/SAFER-Activities/clean3d_v1_preflight/"
    "preflight_report.json"
)


def subject_id(name: str) -> int:
    match = re.search(r"_p0*(\d+)_", normalized_name(name))
    if match is None:
        raise ValueError(f"cannot parse SAFER subject id: {name}")
    return int(match.group(1))


def window_starts(total_frames: int) -> np.ndarray:
    """Historical windows: regular stride only; never append a tail endpoint."""
    if total_frames < WINDOW_SIZE:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, total_frames - WINDOW_SIZE + 1, STRIDE, dtype=np.int64)


def annotation_window_stats(annotation: dict[str, Any]) -> dict[str, int]:
    total_frames = int(annotation["total_frames"])
    pose = np.asarray(annotation["keypoint_3d"])
    if pose.shape != (1, total_frames, 17, 3):
        raise ValueError(
            f"{annotation['frame_dir']}: expected keypoint_3d "
            f"(1,{total_frames},17,3), got {pose.shape}"
        )
    finite_frames = np.isfinite(pose[0]).all(axis=(1, 2))
    bad_frames = ~finite_frames
    starts = window_starts(total_frames)
    if starts.size:
        prefix = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(bad_frames, dtype=np.int64))
        )
        contaminated = (
            prefix[starts + WINDOW_SIZE] - prefix[starts]
        ) > 0
        clean_windows = int(np.count_nonzero(~contaminated))
    else:
        clean_windows = 0
    return {
        "candidate_windows": int(starts.size),
        "clean_windows": clean_windows,
        "excluded_windows": int(starts.size) - clean_windows,
        "bad_frames": int(np.count_nonzero(bad_frames)),
    }


def subset_rows(
    subjects: tuple[int, ...],
    per_subject: dict[int, dict[str, int]],
) -> list[tuple[int, int, tuple[int, ...]]]:
    rows = []
    for mask in range(1 << len(subjects)):
        selected = tuple(
            subject for index, subject in enumerate(subjects) if mask & (1 << index)
        )
        rows.append(
            (
                sum(per_subject[value]["candidate_windows"] for value in selected),
                sum(per_subject[value]["clean_windows"] for value in selected),
                selected,
            )
        )
    return rows


def matching_subject_subsets(
    per_subject: dict[int, dict[str, int]],
    candidate_target: int,
    clean_target: int,
) -> list[tuple[int, ...]]:
    """Meet-in-the-middle proof over every subset of official train subjects."""
    subjects = tuple(sorted(per_subject))
    midpoint = len(subjects) // 2
    left_rows = subset_rows(subjects[:midpoint], per_subject)
    right_lookup: dict[tuple[int, int], list[tuple[int, ...]]] = defaultdict(list)
    for candidate, clean, selected in subset_rows(subjects[midpoint:], per_subject):
        right_lookup[(candidate, clean)].append(selected)

    matches = []
    for candidate, clean, selected in left_rows:
        wanted = (candidate_target - candidate, clean_target - clean)
        matches.extend(selected + right for right in right_lookup.get(wanted, ()))
    return sorted(matches)


def audit_passed(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    aggregate = report.get("aggregate", {})
    if not report.get("passed"):
        raise ValueError(f"raw audit did not pass: {path}")
    if aggregate.get("sequences") != 497 or aggregate.get("frames") != 8_091_357:
        raise ValueError(f"raw audit aggregate mismatch: {path}")
    return report, sha256_file(path)


def split_annotations(
    normal: dict[str, Any], ood: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], set[int], set[int]]:
    official_train = {
        normalized_name(value) for value in normal["split"]["sub_train"]
    }
    official_test = {
        normalized_name(value) for value in normal["split"]["sub_test"]
    }
    if official_train & official_test:
        raise ValueError("official SAFER subject train/test membership overlaps")

    result = {key: [] for key in EXPECTED_SPLITS}
    seen = set()
    for annotation in normal["annotations"]:
        name = normalized_name(annotation["frame_dir"])
        if name in seen:
            raise ValueError(f"duplicate annotation: {name}")
        seen.add(name)
        if name in official_test:
            split = "test"
        elif name in official_train:
            split = "val" if subject_id(name) in VALIDATION_SUBJECTS else "train"
        else:
            raise ValueError(f"annotation absent from official subject split: {name}")
        result[split].append(annotation)
    if seen != official_train | official_test:
        raise ValueError("official normal split does not exactly cover annotations")

    ood_names = {normalized_name(value) for value in ood["split"]["sub_test"]}
    for annotation in ood["annotations"]:
        name = normalized_name(annotation["frame_dir"])
        if name not in ood_names:
            raise ValueError(f"OOD annotation absent from official sub_test: {name}")
        result["ood"].append(annotation)

    train_subjects = {subject_id(value) for value in official_train}
    test_subjects = {subject_id(value) for value in official_test}
    return result, train_subjects, test_subjects


def summarize(annotations: Iterable[dict[str, Any]]) -> dict[str, Any]:
    annotations = list(annotations)
    total = {
        "sequences": len(annotations),
        "candidate_windows": 0,
        "clean_windows": 0,
        "excluded_windows": 0,
        "bad_frames": 0,
    }
    subjects = set()
    for annotation in annotations:
        subjects.add(subject_id(annotation["frame_dir"]))
        stats = annotation_window_stats(annotation)
        for key, value in stats.items():
            total[key] += value
    total["subjects"] = sorted(subjects)
    return total


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    raw_audit_path = args.raw_audit.resolve()
    _, raw_audit_sha256 = audit_passed(raw_audit_path)
    normal = load_pinned_pickle(args.normal_pickle.resolve(), "normal")
    ood = load_pinned_pickle(args.ood_pickle.resolve(), "ood")
    split_items, official_train_subjects, official_test_subjects = split_annotations(
        normal, ood
    )
    split_reports = {key: summarize(split_items[key]) for key in EXPECTED_SPLITS}

    per_subject: dict[int, dict[str, int]] = defaultdict(
        lambda: {"candidate_windows": 0, "clean_windows": 0}
    )
    for annotation in split_items["train"] + split_items["val"]:
        stats = annotation_window_stats(annotation)
        subject = subject_id(annotation["frame_dir"])
        per_subject[subject]["candidate_windows"] += stats["candidate_windows"]
        per_subject[subject]["clean_windows"] += stats["clean_windows"]
    matches = matching_subject_subsets(
        per_subject,
        EXPECTED_SPLITS["val"]["candidate_windows"],
        EXPECTED_SPLITS["val"]["clean_windows"],
    )

    checks = {
        "raw_audit_passed": True,
        "pinned_release_loaded": True,
        "official_train_test_subject_disjoint": not (
            official_train_subjects & official_test_subjects
        ),
        "validation_subjects_inside_official_train": VALIDATION_SUBJECTS
        < official_train_subjects,
        "train_validation_subject_disjoint": not (
            set(split_reports["train"]["subjects"])
            & set(split_reports["val"]["subjects"])
        ),
        "validation_subset_unique_from_preserved_counts": matches
        == [tuple(sorted(VALIDATION_SUBJECTS))],
        "split_counts_exact": all(
            all(split_reports[key][field] == expected
                for field, expected in EXPECTED_SPLITS[key].items())
            for key in EXPECTED_SPLITS
        ),
        "legacy_bad_frame_count_exact": sum(
            split_reports[key]["bad_frames"] for key in EXPECTED_SPLITS
        ) == EXPECTED_BAD_FRAMES,
        "no_tail_window_appended": True,
        "wheelchair_not_loaded": True,
    }
    return {
        "preflight": "recovered historical SAFER clean3d_v1 count gate",
        "recovery_status": {
            "original_safer_gendata_source_recovered": False,
            "count_contract_proved": True,
            "topology_and_normalization_not_yet_authorized": True,
            "materialized": False,
        },
        "source": {
            "normal_pickle": str(args.normal_pickle.resolve()),
            "normal_sha256": PINNED_FILES["normal"]["sha256"],
            "ood_pickle": str(args.ood_pickle.resolve()),
            "ood_sha256": PINNED_FILES["ood"]["sha256"],
            "raw_audit": str(raw_audit_path),
            "raw_audit_sha256": raw_audit_sha256,
        },
        "window_contract": {
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "start_rule": "range(0, total_frames - 64 + 1, 8)",
            "append_tail_window": False,
            "reject_window_if_any_legacy_3d_frame_nonfinite": True,
            "global_interpolation": False,
        },
        "split_contract": {
            "official_protocol": "subject",
            "official_train_subjects": sorted(official_train_subjects),
            "validation_subjects": sorted(VALIDATION_SUBJECTS),
            "official_test_subjects": sorted(official_test_subjects),
            "validation_recovery_evidence": {
                "method": "exhaustive meet-in-the-middle over every official-train subject subset",
                "targets": {
                    "corrected_v2_candidate_windows": EXPECTED_SPLITS["val"]["candidate_windows"],
                    "legacy_v1_clean_windows": EXPECTED_SPLITS["val"]["clean_windows"],
                },
                "matching_subsets": [list(value) for value in matches],
            },
        },
        "splits": split_reports,
        "expected_splits": EXPECTED_SPLITS,
        "checks": checks,
        "passed": all(checks.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-pickle", type=Path, default=DEFAULT_NORMAL_PICKLE)
    parser.add_argument("--ood-pickle", type=Path, default=DEFAULT_OOD_PICKLE)
    parser.add_argument("--raw-audit", type=Path, default=DEFAULT_RAW_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--require-valid",
        action="store_true",
        help="exit nonzero unless every preserved historical count gate passes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing preflight: {output}")
    report = build_report(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "validation_subjects": report["split_contract"]["validation_subjects"],
        "clean_windows": {
            key: report["splits"][key]["clean_windows"] for key in EXPECTED_SPLITS
        },
        "passed": report["passed"],
    }, indent=2))
    if args.require_valid and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
