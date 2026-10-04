#!/usr/bin/env python3
"""Materialize the frozen SAFER legacy clean3d_v1 reconstruction.

The original project-specific ``safer_gendata.py`` and COCO17-to-NTU25 mapper
were lost.  This command therefore does not claim byte-identical historical
reproduction.  It materializes only the structural reconstruction explicitly
authorized by ``configs/safer_legacy_v1_structural_reconstruction_v1.json``:

* the pinned official SAFER normal and non-lab/OOD pickle release;
* official subject train/test membership and the count-recovered validation set;
* 64-frame windows, stride 8, no appended tail;
* exclusion of any window containing a non-finite legacy-3D frame;
* the frozen ``legacy_coco_umurl_window_noscale`` compatibility candidate;
* official coarse frame labels plus the recorded four-class derivation;
* a zero-filled second person.

The writer is resumable at source-sequence boundaries, never overwrites a
completed destination, and writes a research-usable manifest only after all
four split counts and payload audits pass.  It is a CPU-only preprocessing
command and imports neither PyTorch nor CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from data_gen.inspect_safer_activities import (
    PINNED_FILES,
    load_pinned_pickle,
    normalized_name,
    sha256_file,
)
from data_gen.preflight_safer_legacy_v1 import (
    DEFAULT_NORMAL_PICKLE,
    DEFAULT_OOD_PICKLE,
    DEFAULT_OUTPUT as DEFAULT_PREFLIGHT,
    EXPECTED_SPLITS,
    STRIDE,
    VALIDATION_SUBJECTS,
    WINDOW_SIZE,
    split_annotations,
    subject_id,
    window_starts,
)
from data_gen.safer_legacy_v1_candidates import (
    CANDIDATES,
    Candidate,
    convert_windows,
    sequence_context,
)


FROZEN_CANDIDATE_NAME = "legacy_coco_umurl_window_noscale"
CENTER_OFFSET = 32
SPLIT_ORDER = ("train", "val", "test", "ood")
DEFAULT_FREEZE_MANIFEST = Path(
    "configs/safer_legacy_v1_structural_reconstruction_v1.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "data/fall_processed/SAFER-Activities/clean3d_v1_reconstructed"
)
RUN_STATE_NAME = "materialization_state.json"
FINAL_MANIFEST_NAME = "materialization_manifest.json"
DERIVED_CLASS_NAMES = ("other", "fall", "lie_down", "lying_down")
ARRAY_SPECS = {
    "data_joint.npy": (np.float32, lambda count: (count, 3, WINDOW_SIZE, 25, 2)),
    "dense_coarse_labels.npy": (np.int64, lambda count: (count, WINDOW_SIZE)),
    "dense_derived_labels.npy": (np.int64, lambda count: (count, WINDOW_SIZE)),
    "center_coarse_labels.npy": (np.int64, lambda count: (count,)),
    "center_derived_labels.npy": (np.int64, lambda count: (count,)),
    "sequence_index.npy": (np.int64, lambda count: (count,)),
    "window_start.npy": (np.int64, lambda count: (count,)),
    "num_frame.npy": (np.int64, lambda count: (count,)),
}


def atomic_json_dump(path: Path, value: object) -> None:
    """Replace a small JSON state file atomically."""
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def derive_four_class(coarse: np.ndarray) -> np.ndarray:
    coarse = np.asarray(coarse)
    if not np.issubdtype(coarse.dtype, np.integer):
        raise TypeError(f"coarse labels must be integral, got {coarse.dtype}")
    if coarse.size and (int(coarse.min()) < 0 or int(coarse.max()) > 15):
        raise ValueError("SAFER coarse labels must be inside 0..15")
    derived = np.zeros(coarse.shape, dtype=np.int64)
    derived[coarse == 10] = 1
    derived[coarse == 11] = 2
    derived[coarse == 12] = 3
    return derived


def load_freeze_manifest(path: Path) -> tuple[dict[str, Any], Candidate, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported structural freeze manifest schema")
    if manifest.get("status") != "approved_for_materialization":
        raise ValueError("structural reconstruction is not approved for materialization")
    if manifest.get("candidate_name") != FROZEN_CANDIDATE_NAME:
        raise ValueError("freeze manifest does not select the locked legacy candidate")
    candidate = CANDIDATES[FROZEN_CANDIDATE_NAME]
    if manifest.get("candidate_contract") != candidate.manifest():
        raise ValueError("freeze manifest candidate contract does not match code")
    limits = manifest.get("recovery_limits", {})
    required_limits = {
        "original_safer_gendata_source_recovered": False,
        "original_map_coco17_to_ntu25_source_recovered": False,
        "historical_weight_byte_identity_proved": False,
        "must_not_be_called_byte_identical_reproduction": True,
        "experimental_compatibility_reconstruction": True,
    }
    if any(limits.get(key) != value for key, value in required_limits.items()):
        raise ValueError("freeze manifest does not preserve reconstruction limits")
    contract = manifest.get("historical_contract", {})
    required_contract = {
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "append_tail_window": False,
        "reject_window_if_any_legacy_3d_frame_nonfinite": True,
        "global_interpolation": False,
        "second_person_zero_fill": True,
        "center_label_offset": CENTER_OFFSET,
    }
    if any(contract.get(key) != value for key, value in required_contract.items()):
        raise ValueError("freeze manifest historical contract mismatch")
    expected = {key: EXPECTED_SPLITS[key]["clean_windows"] for key in SPLIT_ORDER}
    if contract.get("expected_clean_windows") != expected:
        raise ValueError("freeze manifest clean-window counts mismatch")
    return manifest, candidate, sha256_file(path)


def load_preflight(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("passed"):
        raise ValueError(f"legacy clean3d_v1 preflight did not pass: {path}")
    if set(report.get("split_contract", {}).get("validation_subjects", ())) != set(
        VALIDATION_SUBJECTS
    ):
        raise ValueError("preflight validation subject contract mismatch")
    for split in SPLIT_ORDER:
        observed = report.get("splits", {}).get(split, {})
        expected = EXPECTED_SPLITS[split]
        if any(observed.get(key) != value for key, value in expected.items()):
            raise ValueError(f"preflight {split} count contract mismatch")
    source = report.get("source", {})
    if source.get("normal_sha256") != PINNED_FILES["normal"]["sha256"]:
        raise ValueError("preflight normal pickle hash mismatch")
    if source.get("ood_sha256") != PINNED_FILES["ood"]["sha256"]:
        raise ValueError("preflight OOD pickle hash mismatch")
    return report, sha256_file(path)


def annotation_arrays(annotation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    frames = int(annotation["total_frames"])
    pose = np.asarray(annotation["keypoint_3d"])
    labels = np.asarray(annotation["labels"])
    if pose.shape != (1, frames, 17, 3):
        raise ValueError(
            f"{annotation['frame_dir']}: expected keypoint_3d "
            f"(1,{frames},17,3), got {pose.shape}"
        )
    if labels.shape != (frames,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            f"{annotation['frame_dir']}: expected integral labels ({frames},), "
            f"got {labels.shape}/{labels.dtype}"
        )
    if labels.size and (int(labels.min()) < 0 or int(labels.max()) > 15):
        raise ValueError(f"{annotation['frame_dir']}: coarse label outside 0..15")
    return pose[0], labels


def clean_window_starts(annotation: dict[str, Any]) -> np.ndarray:
    pose, _ = annotation_arrays(annotation)
    finite = np.isfinite(pose).all(axis=(1, 2))
    starts = window_starts(pose.shape[0])
    if not starts.size:
        return starts
    prefix = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(~finite, dtype=np.int64))
    )
    return starts[(prefix[starts + WINDOW_SIZE] - prefix[starts]) == 0]


def build_split_plans(
    split_items: dict[str, list[dict[str, Any]]],
    smoke_windows_per_split: int,
) -> tuple[dict[str, list[np.ndarray]], dict[str, dict[str, int]]]:
    plans: dict[str, list[np.ndarray]] = {}
    inventory: dict[str, dict[str, int]] = {}
    for split in SPLIT_ORDER:
        annotations = split_items[split]
        full_starts = [clean_window_starts(annotation) for annotation in annotations]
        full_count = int(sum(starts.size for starts in full_starts))
        expected = EXPECTED_SPLITS[split]
        if len(annotations) != expected["sequences"]:
            raise RuntimeError(
                f"{split} sequence count mismatch: {len(annotations)} != "
                f"{expected['sequences']}"
            )
        if full_count != expected["clean_windows"]:
            raise RuntimeError(
                f"{split} clean-window count mismatch: {full_count} != "
                f"{expected['clean_windows']}"
            )

        remaining = smoke_windows_per_split
        selected: list[np.ndarray] = []
        for starts in full_starts:
            if smoke_windows_per_split == 0:
                chosen = starts
            elif remaining > 0:
                chosen = starts[:remaining]
                remaining -= int(chosen.size)
            else:
                chosen = starts[:0]
            selected.append(np.asarray(chosen, dtype=np.int64))
        written_count = int(sum(starts.size for starts in selected))
        wanted = full_count if smoke_windows_per_split == 0 else min(
            smoke_windows_per_split, full_count
        )
        if written_count != wanted:
            raise RuntimeError(f"{split} selected-window planning mismatch")
        plans[split] = selected
        inventory[split] = {
            "source_sequences": len(annotations),
            "full_clean_windows": full_count,
            "written_windows": written_count,
        }
    return plans, inventory


def estimated_output_bytes(inventory: dict[str, dict[str, int]]) -> int:
    bytes_per_window = (
        3 * WINDOW_SIZE * 25 * 2 * np.dtype(np.float32).itemsize
        + 2 * WINDOW_SIZE * np.dtype(np.int64).itemsize
        + 5 * np.dtype(np.int64).itemsize
        + 160  # conservative allowance for names and small metadata
    )
    return int(
        sum(value["written_windows"] for value in inventory.values())
        * bytes_per_window
    )


def check_disk_space(output_root: Path, required_bytes: int, reserve_gib: float) -> dict[str, int]:
    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(parent)
    reserve_bytes = int(reserve_gib * (1024 ** 3))
    if usage.free < required_bytes + reserve_bytes:
        raise RuntimeError(
            "insufficient free space: "
            f"need materialization {required_bytes} + reserve {reserve_bytes}, "
            f"available {usage.free} bytes"
        )
    return {
        "estimated_output_bytes": required_bytes,
        "required_reserve_bytes": reserve_bytes,
        "free_bytes_before": int(usage.free),
    }


def immutable_run_contract(
    args: argparse.Namespace,
    freeze_sha256: str,
    preflight_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": "full" if args.smoke_windows_per_split == 0 else "smoke",
        "normal_pickle": str(args.normal_pickle.resolve()),
        "normal_pickle_sha256": PINNED_FILES["normal"]["sha256"],
        "ood_pickle": str(args.ood_pickle.resolve()),
        "ood_pickle_sha256": PINNED_FILES["ood"]["sha256"],
        "preflight": str(args.preflight.resolve()),
        "preflight_sha256": preflight_sha256,
        "freeze_manifest": str(args.freeze_manifest.resolve()),
        "freeze_manifest_sha256": freeze_sha256,
        "candidate_name": FROZEN_CANDIDATE_NAME,
        "smoke_windows_per_split": args.smoke_windows_per_split,
    }


def initialize_run_root(
    output_root: Path,
    contract: dict[str, Any],
    disk: dict[str, int],
    resume: bool,
) -> dict[str, Any] | None:
    state_path = output_root / RUN_STATE_NAME
    final_path = output_root / FINAL_MANIFEST_NAME
    if output_root.exists() and any(output_root.iterdir()):
        if not resume:
            raise FileExistsError(
                f"refusing to overwrite non-empty output: {output_root}; "
                "use --resume only for this generator's matching incomplete run"
            )
        if final_path.is_file():
            completed = json.loads(final_path.read_text(encoding="utf-8"))
            if completed.get("run_contract") != contract:
                raise RuntimeError("completed output run contract differs from requested run")
            if not completed.get("integrity", {}).get("passed"):
                raise RuntimeError("completed output manifest did not pass integrity")
            return completed
        if not state_path.is_file():
            raise RuntimeError("cannot resume output without materialization state")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("run_contract") != contract:
            raise RuntimeError("incomplete output run contract differs from requested run")
        return None

    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(
        state_path,
        {
            "status": "in_progress",
            "research_usable": False,
            "run_contract": contract,
            "disk_preflight": disk,
            "completed_splits": [],
        },
    )
    return None


def open_split_arrays(
    split_root: Path,
    count: int,
    resume: bool,
) -> dict[str, np.memmap]:
    arrays: dict[str, np.memmap] = {}
    for name, (dtype, shape_fn) in ARRAY_SPECS.items():
        path = split_root / name
        shape = shape_fn(count)
        if resume:
            if not path.is_file():
                raise FileNotFoundError(f"resume array missing: {path}")
            value = np.load(path, mmap_mode="r+")
            if value.shape != shape or value.dtype != np.dtype(dtype):
                raise RuntimeError(
                    f"resume array mismatch for {path}: "
                    f"{value.shape}/{value.dtype} != {shape}/{np.dtype(dtype)}"
                )
        else:
            value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        arrays[name] = value
    return arrays


def flush_arrays(arrays: dict[str, np.memmap]) -> None:
    for value in arrays.values():
        value.flush()


def sequence_rows(
    annotations: Sequence[dict[str, Any]], plans: Sequence[np.ndarray]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    for index, (annotation, starts) in enumerate(zip(annotations, plans)):
        name = normalized_name(annotation["frame_dir"])
        count = int(starts.size)
        rows.append(
            {
                "sequence_index": index,
                "source_name": name,
                "subject": subject_id(name),
                "total_frames": int(annotation["total_frames"]),
                "written_window_count": count,
                "output_start": offset,
                "output_stop": offset + count,
                "first_window_start": int(starts[0]) if count else None,
                "last_window_start": int(starts[-1]) if count else None,
            }
        )
        offset += count
    return rows


def materialize_split(
    split: str,
    annotations: Sequence[dict[str, Any]],
    plans: Sequence[np.ndarray],
    split_root: Path,
    candidate: Candidate,
    batch_windows: int,
    full_clean_windows: int,
    resume: bool,
) -> dict[str, Any]:
    split_root.mkdir(parents=True, exist_ok=True)
    manifest_path = split_root / "split_manifest.json"
    progress_path = split_root / "progress.json"
    count = int(sum(starts.size for starts in plans))
    if manifest_path.is_file():
        if not resume:
            raise FileExistsError(f"split already materialized: {split_root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("integrity", {}).get("passed"):
            raise RuntimeError(f"existing split manifest failed: {manifest_path}")
        if manifest.get("written_windows") != count:
            raise RuntimeError(f"existing {split} written count mismatch")
        return manifest

    continuing = progress_path.is_file()
    if continuing and not resume:
        raise FileExistsError(f"incomplete split requires --resume: {split_root}")
    if continuing:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("split") != split or progress.get("expected_windows") != count:
            raise RuntimeError(f"{split} progress contract mismatch")
        next_sequence = int(progress["next_sequence_index"])
        written = int(progress["written_windows"])
        expected_written = int(sum(starts.size for starts in plans[:next_sequence]))
        if written != expected_written:
            raise RuntimeError(f"{split} progress offset mismatch")
    else:
        if any(split_root.iterdir()):
            raise FileExistsError(f"refusing to overwrite split output: {split_root}")
        next_sequence = 0
        written = 0

    arrays = open_split_arrays(split_root, count, resume=continuing)
    offsets = np.arange(WINDOW_SIZE, dtype=np.int64)
    for sequence_index in range(next_sequence, len(annotations)):
        annotation = annotations[sequence_index]
        pose, labels = annotation_arrays(annotation)
        starts = plans[sequence_index]
        context = sequence_context(pose, candidate)
        for begin in range(0, starts.size, batch_windows):
            batch_starts = starts[begin : begin + batch_windows]
            stop = written + int(batch_starts.size)
            converted = convert_windows(pose, batch_starts, candidate, context)
            packed = np.zeros(
                (batch_starts.size, 3, WINDOW_SIZE, 25, 2), dtype=np.float32
            )
            packed[..., 0] = converted.transpose(0, 3, 1, 2)
            dense_coarse = labels[
                batch_starts[:, None] + offsets[None, :]
            ].astype(np.int64, copy=False)
            dense_derived = derive_four_class(dense_coarse)

            target = slice(written, stop)
            arrays["data_joint.npy"][target] = packed
            arrays["dense_coarse_labels.npy"][target] = dense_coarse
            arrays["dense_derived_labels.npy"][target] = dense_derived
            arrays["center_coarse_labels.npy"][target] = dense_coarse[:, CENTER_OFFSET]
            arrays["center_derived_labels.npy"][target] = dense_derived[:, CENTER_OFFSET]
            arrays["sequence_index.npy"][target] = sequence_index
            arrays["window_start.npy"][target] = batch_starts
            arrays["num_frame.npy"][target] = WINDOW_SIZE
            written = stop

        flush_arrays(arrays)
        atomic_json_dump(
            progress_path,
            {
                "status": "in_progress",
                "split": split,
                "expected_windows": count,
                "next_sequence_index": sequence_index + 1,
                "written_windows": written,
            },
        )
        if (
            sequence_index == 0
            or (sequence_index + 1) % 10 == 0
            or sequence_index + 1 == len(annotations)
        ):
            print(
                f"split={split} sequence={sequence_index + 1}/{len(annotations)} "
                f"windows={written}/{count}",
                flush=True,
            )

    if written != count:
        raise RuntimeError(f"{split} wrote {written} windows, expected {count}")
    flush_arrays(arrays)
    del arrays

    rows = sequence_rows(annotations, plans)
    atomic_json_dump(split_root / "sequences.json", rows)
    sample_names: list[str] = []
    for row, starts in zip(rows, plans):
        sample_names.extend(
            f"{row['source_name']}__f{int(start):07d}" for start in starts
        )
    center_derived = np.load(split_root / "center_derived_labels.npy", mmap_mode="r")
    with (split_root / "label.pkl").open("wb") as stream:
        pickle.dump((sample_names, center_derived.astype(int).tolist()), stream)
    del center_derived

    audit = audit_split_payload(split_root, count, batch_windows)
    checks = {
        "source_sequence_count_exact": len(annotations) == EXPECTED_SPLITS[split]["sequences"],
        "full_clean_window_count_exact": full_clean_windows
        == EXPECTED_SPLITS[split]["clean_windows"],
        "written_window_count_exact": count
        == sum(int(row["written_window_count"]) for row in rows),
        "sample_names_unique": len(sample_names) == len(set(sample_names)),
        **audit["checks"],
    }
    manifest = {
        "split": split,
        "candidate": candidate.manifest(),
        "source_sequences": len(annotations),
        "full_clean_windows": full_clean_windows,
        "written_windows": count,
        "data_shape": [count, 3, WINDOW_SIZE, 25, 2],
        "data_dtype": "float32",
        "sample_order": "official pickle annotation order, then ascending regular window start",
        "sample_name_format": "{normalized_source_name}__f{zero_padded_start}",
        "center_label_offset": CENTER_OFFSET,
        "center_label": "derived four-class label; explicit coarse and derived arrays are both stored",
        "derived_class_names": list(DERIVED_CLASS_NAMES),
        "payload": audit["payload"],
        "small_file_sha256": {
            name: sha256_file(split_root / name)
            for name in ("label.pkl", "sequences.json")
        },
        "integrity": {"passed": all(checks.values()), **checks},
    }
    if not manifest["integrity"]["passed"]:
        raise RuntimeError(f"{split} integrity gate failed: {checks}")
    atomic_json_dump(manifest_path, manifest)
    atomic_json_dump(
        progress_path,
        {
            "status": "completed",
            "split": split,
            "expected_windows": count,
            "next_sequence_index": len(annotations),
            "written_windows": written,
            "split_manifest_sha256": sha256_file(manifest_path),
        },
    )
    return manifest


def array_payload_hash(array: np.ndarray, chunk_windows: int) -> str:
    digest = hashlib.sha256()
    for begin in range(0, array.shape[0], chunk_windows):
        block = np.ascontiguousarray(array[begin : begin + chunk_windows])
        digest.update(block.tobytes(order="C"))
    return digest.hexdigest()


def audit_split_payload(
    split_root: Path,
    count: int,
    chunk_windows: int,
) -> dict[str, Any]:
    arrays = {
        name: np.load(split_root / name, mmap_mode="r") for name in ARRAY_SPECS
    }
    shapes_and_dtypes = all(
        arrays[name].shape == shape_fn(count)
        and arrays[name].dtype == np.dtype(dtype)
        for name, (dtype, shape_fn) in ARRAY_SPECS.items()
    )
    data_finite = True
    second_person_zero = True
    center_joint_max_abs = 0.0
    shoulder_yz_max_abs = 0.0
    shoulder_x_min = float("inf")
    data_digest = hashlib.sha256()
    for begin in range(0, count, chunk_windows):
        block = np.ascontiguousarray(
            arrays["data_joint.npy"][begin : begin + chunk_windows]
        )
        data_digest.update(block.tobytes(order="C"))
        data_finite = data_finite and bool(np.isfinite(block).all())
        second_person_zero = second_person_zero and bool(np.count_nonzero(block[..., 1]) == 0)
        if block.size:
            center_joint_max_abs = max(
                center_joint_max_abs,
                float(np.max(np.abs(block[:, :, :, 1, 0]))),
            )
            shoulder = block[:, :, 0, 8, 0] - block[:, :, 0, 4, 0]
            shoulder_yz_max_abs = max(
                shoulder_yz_max_abs, float(np.max(np.abs(shoulder[:, 1:])))
            )
            shoulder_x_min = min(shoulder_x_min, float(np.min(shoulder[:, 0])))

    dense_coarse = arrays["dense_coarse_labels.npy"]
    dense_derived = arrays["dense_derived_labels.npy"]
    center_coarse = arrays["center_coarse_labels.npy"]
    center_derived = arrays["center_derived_labels.npy"]
    labels_consistent = True
    for begin in range(0, count, chunk_windows):
        stop = min(begin + chunk_windows, count)
        coarse = np.asarray(dense_coarse[begin:stop])
        derived = np.asarray(dense_derived[begin:stop])
        labels_consistent = labels_consistent and bool(
            np.array_equal(derived, derive_four_class(coarse))
            and np.array_equal(center_coarse[begin:stop], coarse[:, CENTER_OFFSET])
            and np.array_equal(center_derived[begin:stop], derived[:, CENTER_OFFSET])
        )

    with (split_root / "label.pkl").open("rb") as stream:
        names, pickle_labels = pickle.load(stream)
    label_pickle_consistent = (
        len(names) == count
        and len(pickle_labels) == count
        and np.array_equal(np.asarray(pickle_labels), np.asarray(center_derived))
    )
    index_ranges_valid = bool(
        np.all(np.asarray(arrays["sequence_index.npy"]) >= 0)
        and np.all(np.asarray(arrays["window_start.npy"]) >= 0)
        and np.all(np.asarray(arrays["window_start.npy"]) % STRIDE == 0)
        and np.all(np.asarray(arrays["num_frame.npy"]) == WINDOW_SIZE)
    )
    payload = {
        "data_joint.npy": {
            "bytes": (split_root / "data_joint.npy").stat().st_size,
            "array_payload_sha256": data_digest.hexdigest(),
        }
    }
    for name in ARRAY_SPECS:
        if name == "data_joint.npy":
            continue
        payload[name] = {
            "bytes": (split_root / name).stat().st_size,
            "array_payload_sha256": array_payload_hash(arrays[name], chunk_windows),
        }
    checks = {
        "array_shapes_and_dtypes_exact": shapes_and_dtypes,
        "all_coordinates_finite": data_finite,
        "second_person_exact_zero": second_person_zero,
        "spine_mid_centered": center_joint_max_abs <= 1e-5,
        "first_frame_shoulder_aligned_to_positive_x": (
            shoulder_yz_max_abs <= 1e-4 and shoulder_x_min >= -1e-5
        ),
        "coarse_and_derived_labels_consistent": labels_consistent,
        "label_pickle_matches_center_derived": label_pickle_consistent,
        "window_indices_and_num_frame_valid": index_ranges_valid,
    }
    return {
        "payload": payload,
        "diagnostics": {
            "spine_mid_center_max_abs": center_joint_max_abs,
            "first_frame_shoulder_yz_max_abs": shoulder_yz_max_abs,
            "first_frame_shoulder_x_min": shoulder_x_min,
            "center_derived_counts": dict(
                sorted(Counter(map(int, np.asarray(center_derived))).items())
            ),
        },
        "checks": checks,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_windows < 1 or args.smoke_windows_per_split < 0:
        raise ValueError("batch-windows must be positive and smoke count non-negative")
    freeze, candidate, freeze_sha256 = load_freeze_manifest(args.freeze_manifest)
    preflight, preflight_sha256 = load_preflight(args.preflight)
    normal = load_pinned_pickle(args.normal_pickle.resolve(), "normal")
    ood = load_pinned_pickle(args.ood_pickle.resolve(), "ood")
    split_items, official_train_subjects, official_test_subjects = split_annotations(
        normal, ood
    )
    plans, inventory = build_split_plans(
        split_items, args.smoke_windows_per_split
    )

    output_root = args.output_root.resolve()
    disk = check_disk_space(
        output_root,
        estimated_output_bytes(inventory),
        args.reserve_gib,
    )
    contract = immutable_run_contract(
        args, freeze_sha256=freeze_sha256, preflight_sha256=preflight_sha256
    )
    completed = initialize_run_root(output_root, contract, disk, args.resume)
    if completed is not None:
        print(json.dumps({
            "output_root": str(output_root),
            "status": "already_completed",
            "research_usable": completed["research_usable"],
            "integrity": completed["integrity"],
        }, indent=2))
        return completed

    split_manifests: dict[str, dict[str, Any]] = {}
    for split in SPLIT_ORDER:
        split_manifests[split] = materialize_split(
            split=split,
            annotations=split_items[split],
            plans=plans[split],
            split_root=output_root / split,
            candidate=candidate,
            batch_windows=args.batch_windows,
            full_clean_windows=inventory[split]["full_clean_windows"],
            resume=args.resume,
        )
        state_path = output_root / RUN_STATE_NAME
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["completed_splits"] = list(split_manifests)
        atomic_json_dump(state_path, state)

    full_mode = args.smoke_windows_per_split == 0
    checks = {
        "structural_freeze_manifest_valid": True,
        "preflight_valid": True,
        "pinned_normal_and_ood_loaded": True,
        "official_train_test_subject_disjoint": not (
            official_train_subjects & official_test_subjects
        ),
        "validation_subjects_exact": set(
            subject_id(annotation["frame_dir"])
            for annotation in split_items["val"]
        ) == set(VALIDATION_SUBJECTS),
        "full_eligible_counts_exact": all(
            inventory[split]["full_clean_windows"]
            == EXPECTED_SPLITS[split]["clean_windows"]
            for split in SPLIT_ORDER
        ),
        "all_written_split_payloads_passed": all(
            value["integrity"]["passed"] for value in split_manifests.values()
        ),
        "all_four_splits_materialized": set(split_manifests) == set(SPLIT_ORDER),
        "full_counts_written": full_mode and all(
            split_manifests[split]["written_windows"]
            == EXPECTED_SPLITS[split]["clean_windows"]
            for split in SPLIT_ORDER
        ),
    }
    research_usable = bool(full_mode and all(checks.values()))
    manifest = {
        "dataset": "SAFER-Activities",
        "revision": "clean3d_v1_record_grounded_structural_reconstruction_v1",
        "status": "completed" if research_usable else "smoke_completed",
        "research_usable": research_usable,
        "run_contract": contract,
        "recovery_status": freeze["recovery_limits"],
        "selection_basis": freeze["selection_basis"],
        "candidate": candidate.manifest(),
        "source": {
            "normal_pickle_sha256": PINNED_FILES["normal"]["sha256"],
            "ood_pickle_sha256": PINNED_FILES["ood"]["sha256"],
            "preflight_sha256": preflight_sha256,
            "freeze_manifest_sha256": freeze_sha256,
            "generator_sha256": sha256_file(Path(__file__).resolve()),
            "candidate_code_sha256": sha256_file(
                Path(__file__).with_name("safer_legacy_v1_candidates.py")
            ),
        },
        "protocol": {
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "append_tail_window": False,
            "reject_nonfinite_window": True,
            "global_interpolation": False,
            "second_person_zero_fill": True,
            "center_label_offset": CENTER_OFFSET,
            "validation_subjects": sorted(VALIDATION_SUBJECTS),
            "test_and_ood_used_for_candidate_selection": False,
            "test_and_ood_materialized_only_after_structural_freeze": True,
        },
        "inventory": inventory,
        "splits": {
            split: {
                "manifest": str((output_root / split / "split_manifest.json").resolve()),
                "manifest_sha256": sha256_file(
                    output_root / split / "split_manifest.json"
                ),
                "written_windows": split_manifests[split]["written_windows"],
                "integrity_passed": split_manifests[split]["integrity"]["passed"],
            }
            for split in SPLIT_ORDER
        },
        "disk_preflight": disk,
        "integrity": {"passed": research_usable if full_mode else all(
            value for key, value in checks.items() if key != "full_counts_written"
        ), **checks},
    }
    atomic_json_dump(output_root / FINAL_MANIFEST_NAME, manifest)
    atomic_json_dump(
        output_root / RUN_STATE_NAME,
        {
            "status": manifest["status"],
            "research_usable": research_usable,
            "run_contract": contract,
            "disk_preflight": disk,
            "completed_splits": list(SPLIT_ORDER),
            "final_manifest_sha256": sha256_file(output_root / FINAL_MANIFEST_NAME),
        },
    )
    print(json.dumps({
        "output_root": str(output_root),
        "status": manifest["status"],
        "research_usable": research_usable,
        "written_windows": {
            split: split_manifests[split]["written_windows"] for split in SPLIT_ORDER
        },
        "integrity": manifest["integrity"],
    }, ensure_ascii=False, indent=2))
    if args.require_valid and not manifest["integrity"]["passed"]:
        raise SystemExit(2)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-pickle", type=Path, default=DEFAULT_NORMAL_PICKLE)
    parser.add_argument("--ood-pickle", type=Path, default=DEFAULT_OOD_PICKLE)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument(
        "--freeze-manifest", type=Path, default=DEFAULT_FREEZE_MANIFEST
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-windows", type=int, default=256)
    parser.add_argument(
        "--smoke-windows-per-split",
        type=int,
        default=0,
        help="write only this many windows per split; output is never research-usable",
    )
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=10.0,
        help="free-space reserve required in addition to estimated output size",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only this generator's matching sequence-boundary checkpoint",
    )
    parser.add_argument(
        "--require-valid",
        action="store_true",
        help="exit nonzero unless the smoke or full integrity gates pass",
    )
    args = parser.parse_args()
    if args.reserve_gib < 0:
        parser.error("reserve-gib must be non-negative")
    return args


if __name__ == "__main__":
    build(parse_args())
