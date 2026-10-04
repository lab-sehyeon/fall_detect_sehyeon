#!/usr/bin/env python3
"""Audit the pinned SAFER-Activities pickle/CSV/split release.

The original project inspector was lost.  This recovery is limited to the raw
data contract preserved in the historical project record and the pinned
official archive:

* 467 in-lab plus 30 external/OOD sequences;
* 8,091,357 frames in total;
* official coarse ``labels`` remain the only learning labels;
* CSV timestamps are provenance/audit inputs and never replace pickle labels;
* wheelchair data is deliberately outside this historical audit;
* the known legacy-3D and 2D-confidence non-finite counts are reproduced.

Pickle is normally able to import arbitrary Python globals.  This reader first
checks the exact pinned file sizes and SHA-256 values, then permits only the
three NumPy globals used by this release.  It never loads the wheelchair file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PINNED_FILES = {
    "normal": {
        "bytes": 3_361_543_733,
        "sha256": "c65e1620f559ff39e0d67349b9f0c798a28d35df381f97dbc2cc65509787dcb7",
        "sequences": 467,
        "frames": 7_513_839,
    },
    "ood": {
        "bytes": 292_178_207,
        "sha256": "96660c4fafd6665df32e5e121cdd1e657b38814bb1ce7066a9eeb376409271f7",
        "sequences": 30,
        "frames": 577_518,
    },
}
EXPECTED_TOTAL_SEQUENCES = 497
EXPECTED_TOTAL_FRAMES = 8_091_357
EXPECTED_NORMAL_SPLITS = {
    "sub_train": 371,
    "sub_test": 96,
    "view_train": 350,
    "view_test": 117,
}
EXPECTED_OOD_VIEW_ONLY_NAMES = {
    f"day_normal_p06_cam{camera}" for camera in (2, 3, 5, 6, 7, 8)
}
EXPECTED_NONFINITE = {
    "legacy_3d_sequences": 193,
    "legacy_3d_frames": 93_256,
    "official_2d_coordinate_sequences": 0,
    "official_2d_coordinate_frames": 0,
    "official_2d_score_sequences": 193,
    "official_2d_score_frames": 525,
    "official_2d_score_values": 8_925,
}
REQUIRED_FIELDS = {
    "labels",
    "width",
    "height",
    "bboxes",
    "bbox_format",
    "frame_dir",
    "keypoint",
    "keypoint_score",
    "img_shape",
    "total_frames",
    "keypoint_3d",
    "keypoint_3d_score",
}
OPTIONAL_FIELDS = {"full_labels"}


class RestrictedNumpyUnpickler(pickle.Unpickler):
    """Allow only the NumPy constructors present in the pinned pickle."""

    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
    }

    def find_class(self, module: str, name: str) -> Any:
        key = (module, name)
        if key not in self.ALLOWED:
            raise pickle.UnpicklingError(f"blocked pickle global: {module}.{name}")
        return self.ALLOWED[key]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_name(value: Any) -> str:
    return Path(str(value)).stem.lower()


def sequence_base(name: str, domain: str) -> str:
    name = normalized_name(name)
    if domain == "normal":
        return re.sub(r"_d\d+$", "", name, flags=re.IGNORECASE)
    if domain == "ood":
        name = re.sub(r"_camera_\d+$", "", name, flags=re.IGNORECASE)
        return re.sub(r"_cam\d+$", "", name, flags=re.IGNORECASE)
    raise ValueError(f"unknown domain: {domain}")


def load_pinned_pickle(path: Path, domain: str) -> dict[str, Any]:
    expected = PINNED_FILES[domain]
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != expected["bytes"]:
        raise ValueError(
            f"{domain} pickle byte-size mismatch: expected {expected['bytes']}, got {size}"
        )
    digest = sha256_file(path)
    if digest != expected["sha256"]:
        raise ValueError(
            f"{domain} pickle SHA-256 mismatch: expected {expected['sha256']}, got {digest}"
        )
    with path.open("rb") as stream:
        value = RestrictedNumpyUnpickler(stream).load()
    if not isinstance(value, dict):
        raise TypeError(f"{domain} pickle root must be dict, got {type(value).__name__}")
    if set(value) != {"split", "annotations"}:
        raise ValueError(f"{domain} unexpected root keys: {sorted(value)}")
    return value


def nonfinite_stats(array: np.ndarray, frame_axis: int) -> tuple[int, int]:
    finite = np.isfinite(array)
    value_count = int(finite.size - np.count_nonzero(finite))
    moved = np.moveaxis(finite, frame_axis, 0)
    frame_count = int(np.count_nonzero(~moved.reshape(moved.shape[0], -1).all(axis=1)))
    return frame_count, value_count


def audit_annotations(
    annotations: Any,
    domain: str,
) -> tuple[dict[str, Any], set[str], set[str]]:
    if not isinstance(annotations, list):
        raise TypeError(f"{domain} annotations must be a list")

    expected = PINNED_FILES[domain]
    names: list[str] = []
    total_frames = 0
    label_counts: Counter[int] = Counter()
    schema_counts: Counter[tuple[str, ...]] = Counter()
    full_labels_present = 0
    length_mismatches: list[str] = []
    shape_mismatches: list[str] = []
    dtype_mismatches: list[str] = []
    dtype_counts: dict[str, Counter[str]] = {
        field: Counter()
        for field in (
            "keypoint",
            "keypoint_score",
            "labels",
            "keypoint_3d",
            "keypoint_3d_score",
            "bboxes",
        )
    }
    nonfinite = Counter()

    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            raise TypeError(f"{domain} annotation {index} is not a dict")
        keys = set(annotation)
        if not REQUIRED_FIELDS.issubset(keys) or not keys.issubset(
            REQUIRED_FIELDS | OPTIONAL_FIELDS
        ):
            raise ValueError(
                f"{domain} annotation {index} field mismatch: {sorted(keys)}"
            )
        schema_counts[tuple(sorted(keys))] += 1

        name = normalized_name(annotation["frame_dir"])
        names.append(name)
        frames = int(annotation["total_frames"])
        total_frames += frames
        if frames < 1:
            shape_mismatches.append(f"{name}: non-positive total_frames")
            continue

        keypoint = np.asarray(annotation["keypoint"])
        keypoint_score = np.asarray(annotation["keypoint_score"])
        labels = np.asarray(annotation["labels"])
        keypoint_3d = np.asarray(annotation["keypoint_3d"])
        keypoint_3d_score = np.asarray(annotation["keypoint_3d_score"])
        bboxes = np.asarray(annotation["bboxes"])
        arrays = {
            "keypoint": (keypoint, (1, frames, 17, 2), np.floating),
            "keypoint_score": (keypoint_score, (1, frames, 17), np.floating),
            "labels": (labels, (frames,), np.integer),
            "keypoint_3d": (keypoint_3d, (1, frames, 17, 3), np.floating),
            "keypoint_3d_score": (keypoint_3d_score, (1, frames), np.floating),
            "bboxes": (bboxes, (frames, 4), np.floating),
        }
        for field, (array, wanted_shape, wanted_kind) in arrays.items():
            dtype_counts[field][str(array.dtype)] += 1
            if array.shape != wanted_shape:
                shape_mismatches.append(
                    f"{name}:{field} expected {wanted_shape}, got {array.shape}"
                )
            if not np.issubdtype(array.dtype, wanted_kind):
                dtype_mismatches.append(
                    f"{name}:{field} expected {wanted_kind.__name__}, got {array.dtype}"
                )
        if "full_labels" in annotation:
            full_labels_present += 1
            full_labels = np.asarray(annotation["full_labels"])
            if full_labels.shape != (frames,):
                length_mismatches.append(
                    f"{name}:full_labels expected {frames}, got {full_labels.shape}"
                )

        labels_unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(labels_unique, counts):
            label_counts[int(label)] += int(count)

        kp_bad_frames, _ = nonfinite_stats(keypoint, 1)
        score_bad_frames, score_bad_values = nonfinite_stats(keypoint_score, 1)
        kp3_bad_frames, _ = nonfinite_stats(keypoint_3d, 1)
        kp3_score_bad_frames, kp3_score_bad_values = nonfinite_stats(
            keypoint_3d_score, 1
        )
        bbox_bad_frames, bbox_bad_values = nonfinite_stats(bboxes, 0)
        nonfinite["official_2d_coordinate_sequences"] += int(kp_bad_frames > 0)
        nonfinite["official_2d_coordinate_frames"] += kp_bad_frames
        nonfinite["official_2d_score_sequences"] += int(score_bad_frames > 0)
        nonfinite["official_2d_score_frames"] += score_bad_frames
        nonfinite["official_2d_score_values"] += score_bad_values
        nonfinite["legacy_3d_sequences"] += int(kp3_bad_frames > 0)
        nonfinite["legacy_3d_frames"] += kp3_bad_frames
        nonfinite["legacy_3d_score_sequences"] += int(kp3_score_bad_frames > 0)
        nonfinite["legacy_3d_score_frames"] += kp3_score_bad_frames
        nonfinite["legacy_3d_score_values"] += kp3_score_bad_values
        nonfinite["bbox_sequences"] += int(bbox_bad_frames > 0)
        nonfinite["bbox_frames"] += bbox_bad_frames
        nonfinite["bbox_values"] += bbox_bad_values

    name_set = set(names)
    bases = {sequence_base(name, domain) for name in names}
    checks = {
        "sequence_count": len(annotations) == expected["sequences"],
        "frame_count": total_frames == expected["frames"],
        "frame_dirs_unique": len(name_set) == len(names),
        "required_shapes": not shape_mismatches,
        "required_dtypes": not dtype_mismatches,
        "optional_lengths": not length_mismatches,
        "coarse_labels_0_to_15": set(label_counts) == set(range(16)),
    }
    return (
        {
            "sequence_count": len(annotations),
            "unique_frame_dirs": len(name_set),
            "total_frames": total_frames,
            "sequence_base_count": len(bases),
            "coarse_label_counts": dict(sorted(label_counts.items())),
            "full_labels_present": full_labels_present,
            "schema_counts": [
                {"count": count, "fields": list(fields)}
                for fields, count in sorted(schema_counts.items())
            ],
            "dtype_counts": {
                field: dict(sorted(counts.items()))
                for field, counts in dtype_counts.items()
            },
            "nonfinite": dict(nonfinite),
            "shape_mismatches": shape_mismatches[:20],
            "dtype_mismatches": dtype_mismatches[:20],
            "length_mismatches": length_mismatches[:20],
            "checks": checks,
        },
        name_set,
        bases,
    )


def read_split_file(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [normalized_name(line) for line in path.read_text().splitlines() if line.strip()]


def normalized_split(split: Any, key: str) -> list[str]:
    if not isinstance(split, dict) or key not in split:
        raise ValueError(f"pickle split is missing {key}")
    if not isinstance(split[key], list):
        raise TypeError(f"pickle split {key} must be a list")
    return [normalized_name(value) for value in split[key]]


def audit_normal_splits(
    split: dict[str, Any], names: set[str], splits_root: Path
) -> dict[str, Any]:
    files = {
        "sub_train": "non_wheelchair_subject_train.txt",
        "sub_test": "non_wheelchair_subject_test.txt",
        "view_train": "non_wheelchair_view_train.txt",
        "view_test": "non_wheelchair_view_test.txt",
    }
    details: dict[str, Any] = {}
    sets: dict[str, set[str]] = {}
    for key, filename in files.items():
        values = normalized_split(split, key)
        disk = read_split_file(splits_root / filename)
        value_set, disk_set = set(values), set(disk)
        sets[key] = value_set
        details[key] = {
            "count": len(values),
            "unique_count": len(value_set),
            "expected_count": EXPECTED_NORMAL_SPLITS[key],
            "text_set_exact_match": value_set == disk_set,
            "text_order_exact_match": values == disk,
            "missing_in_annotations": sorted(value_set - names),
            "annotations_not_in_split": len(names - value_set),
        }
    details["checks"] = {
        "counts": all(
            details[key]["count"] == EXPECTED_NORMAL_SPLITS[key] for key in files
        ),
        "unique": all(details[key]["count"] == details[key]["unique_count"] for key in files),
        "text_membership_exact": all(
            details[key]["text_set_exact_match"] for key in files
        ),
        "subject_disjoint_complete": (
            not (sets["sub_train"] & sets["sub_test"])
            and sets["sub_train"] | sets["sub_test"] == names
        ),
        "view_disjoint_complete": (
            not (sets["view_train"] & sets["view_test"])
            and sets["view_train"] | sets["view_test"] == names
        ),
    }
    return details


def audit_ood_split(split: dict[str, Any], names: set[str]) -> dict[str, Any]:
    sub_test = set(normalized_split(split, "sub_test"))
    view_test = set(normalized_split(split, "view_test"))
    sub_train = set(normalized_split(split, "sub_train"))
    view_train = set(normalized_split(split, "view_train"))
    view_only = view_test - names
    return {
        "sub_train_count": len(sub_train),
        "sub_test_count": len(sub_test),
        "view_train_count": len(view_train),
        "view_test_count": len(view_test),
        "view_test_only_names": sorted(view_only),
        "known_release_note": (
            "view_test contains six day_normal_p06 camera names absent from annotations; "
            "historical OOD uses the 30 annotation/sub_test sequences"
        ),
        "checks": {
            "empty_train_lists": not sub_train and not view_train,
            "sub_test_exact_annotations": sub_test == names,
            "known_view_only_names_exact": view_only == EXPECTED_OOD_VIEW_ONLY_NAMES,
            "all_annotations_in_view_test": names.issubset(view_test),
        },
    }


def audit_csv_root(csv_root: Path, bases: set[str], domain: str) -> dict[str, Any]:
    paths = sorted(csv_root.glob("*.csv"))
    expected_count = 60 if domain == "normal" else 5
    stems = {path.stem.lower() for path in paths}
    bad_headers: list[str] = []
    empty_annotation_files: list[str] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
        if not rows or len(rows[0]) != 4 or [x.strip() for x in rows[0][1:]] != [
            "Action",
            "Start Time",
            "End Time",
        ]:
            bad_headers.append(path.name)
        if len(rows) < 2:
            empty_annotation_files.append(path.name)
    return {
        "file_count": len(paths),
        "sequence_base_count": len(bases),
        "csv_without_sequence": sorted(stems - bases),
        "sequence_without_csv": sorted(bases - stems),
        "bad_headers": bad_headers,
        "empty_annotation_files": empty_annotation_files,
        "checks": {
            "expected_file_count": len(paths) == expected_count,
            "one_csv_per_sequence_base": stems == bases,
            "headers": not bad_headers,
            "nonempty": not empty_annotation_files,
        },
    }


def all_checks_true(value: Any) -> bool:
    if isinstance(value, dict):
        return all(all_checks_true(child) for child in value.values())
    if isinstance(value, list):
        return all(all_checks_true(child) for child in value)
    return bool(value)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    normal = load_pinned_pickle(args.normal_pickle.resolve(), "normal")
    normal_report, normal_names, normal_bases = audit_annotations(
        normal["annotations"], "normal"
    )
    normal_split = audit_normal_splits(
        normal["split"], normal_names, args.splits_root.resolve()
    )
    del normal

    ood = load_pinned_pickle(args.ood_pickle.resolve(), "ood")
    ood_report, ood_names, ood_bases = audit_annotations(ood["annotations"], "ood")
    ood_split = audit_ood_split(ood["split"], ood_names)
    del ood

    aggregate_nonfinite = Counter(normal_report["nonfinite"])
    aggregate_nonfinite.update(ood_report["nonfinite"])
    aggregate_checks = {
        "sequence_count": (
            normal_report["sequence_count"] + ood_report["sequence_count"]
            == EXPECTED_TOTAL_SEQUENCES
        ),
        "frame_count": (
            normal_report["total_frames"] + ood_report["total_frames"]
            == EXPECTED_TOTAL_FRAMES
        ),
        "legacy_3d_nonfinite": all(
            aggregate_nonfinite[key] == EXPECTED_NONFINITE[key]
            for key in ("legacy_3d_sequences", "legacy_3d_frames")
        ),
        "official_2d_coordinates_finite": all(
            aggregate_nonfinite[key] == EXPECTED_NONFINITE[key]
            for key in (
                "official_2d_coordinate_sequences",
                "official_2d_coordinate_frames",
            )
        ),
        "official_2d_score_nonfinite": all(
            aggregate_nonfinite[key] == EXPECTED_NONFINITE[key]
            for key in (
                "official_2d_score_sequences",
                "official_2d_score_frames",
                "official_2d_score_values",
            )
        ),
        "domains_disjoint": not (normal_names & ood_names),
    }
    checks = {
        "normal_annotations": normal_report["checks"],
        "ood_annotations": ood_report["checks"],
        "normal_splits": normal_split["checks"],
        "ood_split": ood_split["checks"],
        "normal_csv": audit_csv_root(
            args.normal_csv_root.resolve(), normal_bases, "normal"
        )["checks"],
        "ood_csv": audit_csv_root(args.ood_csv_root.resolve(), ood_bases, "ood")[
            "checks"
        ],
        "aggregate": aggregate_checks,
    }
    return {
        "audit": "SAFER-Activities pinned raw release preflight",
        "recovery_status": {
            "historical_structural_contract_recovered": True,
            "original_inspector_source_recovered": False,
            "does_not_load_wheelchair": True,
            "pickle_global_allowlist_enforced": True,
            "csv_does_not_replace_official_pickle_labels": True,
        },
        "pinned_files": PINNED_FILES,
        "normal": normal_report,
        "ood": ood_report,
        "normal_splits": normal_split,
        "ood_split": ood_split,
        "normal_csv": audit_csv_root(
            args.normal_csv_root.resolve(), normal_bases, "normal"
        ),
        "ood_csv": audit_csv_root(args.ood_csv_root.resolve(), ood_bases, "ood"),
        "aggregate": {
            "sequences": normal_report["sequence_count"] + ood_report["sequence_count"],
            "frames": normal_report["total_frames"] + ood_report["total_frames"],
            "nonfinite": dict(aggregate_nonfinite),
        },
        "checks": checks,
        "passed": all_checks_true(checks),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-pickle", type=Path, required=True)
    parser.add_argument("--ood-pickle", type=Path, required=True)
    parser.add_argument("--normal-csv-root", type=Path, required=True)
    parser.add_argument("--ood-csv-root", type=Path, required=True)
    parser.add_argument("--splits-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-valid",
        action="store_true",
        help="exit nonzero unless every historical raw-data gate passes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit report: {output}")
    report = build_report(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "sequences": report["aggregate"]["sequences"],
        "frames": report["aggregate"]["frames"],
        "legacy_3d_nonfinite_sequences": report["aggregate"]["nonfinite"]["legacy_3d_sequences"],
        "legacy_3d_nonfinite_frames": report["aggregate"]["nonfinite"]["legacy_3d_frames"],
        "official_2d_score_nonfinite_values": report["aggregate"]["nonfinite"]["official_2d_score_values"],
        "passed": report["passed"],
    }, indent=2))
    if args.require_valid and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
