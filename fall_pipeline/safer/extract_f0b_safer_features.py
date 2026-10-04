#!/usr/bin/env python3
"""Extract the frozen DSTE features required by reconstructed SAFER F0B.

This command implements only the part of the historical F0B contract that is
recoverable from the preserved project record: max-pooled temporal 1024-D and
spatial 1024-D DSTE representations.  It does not create an optimizer, train a
head, select a checkpoint, or evaluate test/OOD.  The lost F0B optimizer
configuration is therefore outside this command's scope.

Only the reconstructed SAFER V1 train and validation splits are accepted.  The
physical GPU allocation is locked to GPU 0 through ``CUDA_VISIBLE_DEVICES=0``.
Outputs are resumable at flushed batch boundaries and become research-usable
only after both complete splits, source order, finite features, and frozen
DSTE/ADL state invariance have passed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from data_gen.preflight_safer_legacy_v1 import EXPECTED_SPLITS, WINDOW_SIZE
from data_gen.safer_legacy_v1_gendata import (
    FINAL_MANIFEST_NAME as MATERIALIZATION_MANIFEST_NAME,
    FROZEN_CANDIDATE_NAME,
    derive_four_class,
)
from fall_pipeline.common.eval_safer_legacy_v1_candidates import load_adl_model
from fall_pipeline.common.fall_safer_zeroshot_eval import (
    DEFAULT_ADL_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_ENCODER,
    validate_adl_run_config,
)
from fall_pipeline.common.integrity import hash_named_tensors, sha256_file
from fall_pipeline.fu.fall_fu_linear_eval import enforce_physical_gpu_zero


ALLOWED_SPLITS = ("train", "val")
EXPECTED_COUNTS = {
    split: int(EXPECTED_SPLITS[split]["clean_windows"]) for split in ALLOWED_SPLITS
}
FEATURE_DIM = 1024
FEATURE_FILENAMES = ("temporal_features.npy", "spatial_features.npy")
METADATA_FILENAMES = (
    "center_coarse_labels.npy",
    "center_derived_labels.npy",
    "sequence_index.npy",
    "window_start.npy",
)
RUN_STATE_NAME = "extraction_state.json"
FINAL_MANIFEST_NAME = "feature_cache_manifest.json"
SPLIT_PROGRESS_NAME = "progress.json"
SPLIT_MANIFEST_NAME = "feature_manifest.json"
DEFAULT_OUTPUT_DIR = Path(
    "checkpoint/fall/F0B_SAFER_V1_RECONSTRUCTED_20260904_R1/feature_cache"
)


def atomic_json_dump(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def normalize_requested_splits(splits: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(splits)
    if normalized != ALLOWED_SPLITS:
        raise ValueError(
            "F0B development extraction is locked to '--splits train val' in that "
            "order; test/OOD remain unopened until validation selection is frozen"
        )
    return normalized


def _validate_root_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("status") != "completed":
        raise RuntimeError("SAFER materialization status is not completed")
    if not manifest.get("research_usable"):
        raise RuntimeError("SAFER materialization is not research-usable")
    if not manifest.get("integrity", {}).get("passed"):
        raise RuntimeError("SAFER materialization integrity did not pass")
    if manifest.get("candidate", {}).get("name") != FROZEN_CANDIDATE_NAME:
        raise RuntimeError("unexpected SAFER structural reconstruction candidate")
    if manifest.get("selection_basis", {}).get("kind") != (
        "record_grounded_structural_reconstruction"
    ):
        raise RuntimeError("SAFER structural reconstruction basis is missing")
    limits = manifest.get("recovery_status", {})
    if not limits.get("experimental_compatibility_reconstruction"):
        raise RuntimeError("SAFER reconstruction limitation is missing")
    if not limits.get("must_not_be_called_byte_identical_reproduction"):
        raise RuntimeError("SAFER byte-identity limitation is missing")
    protocol = manifest.get("protocol", {})
    if protocol.get("window_size") != WINDOW_SIZE:
        raise RuntimeError("SAFER materialization window-size mismatch")
    if protocol.get("test_and_ood_used_for_candidate_selection") is not False:
        raise RuntimeError("test/OOD candidate-selection isolation was not proved")


def _validate_source_metadata(split_root: Path, count: int) -> dict[str, Any]:
    arrays = {
        name: np.load(split_root / name, mmap_mode="r")
        for name in METADATA_FILENAMES
    }
    for name, array in arrays.items():
        if array.shape != (count,) or array.dtype != np.int64:
            raise RuntimeError(
                f"invalid {split_root.name}/{name}: {array.shape}/{array.dtype}"
            )

    coarse = arrays["center_coarse_labels.npy"]
    derived = arrays["center_derived_labels.npy"]
    sequence = arrays["sequence_index.npy"]
    starts = arrays["window_start.npy"]
    chunk = 131072
    class_counts = np.zeros(4, dtype=np.int64)
    previous_sequence: int | None = None
    previous_start: int | None = None
    for begin in range(0, count, chunk):
        end = min(begin + chunk, count)
        coarse_chunk = np.asarray(coarse[begin:end])
        derived_chunk = np.asarray(derived[begin:end])
        if not np.array_equal(derive_four_class(coarse_chunk), derived_chunk):
            raise RuntimeError(f"{split_root.name} coarse/derived center labels disagree")
        if derived_chunk.size:
            class_counts += np.bincount(derived_chunk, minlength=4)

        sequence_chunk = np.asarray(sequence[begin:end])
        start_chunk = np.asarray(starts[begin:end])
        if np.any(sequence_chunk < 0) or np.any(start_chunk < 0):
            raise RuntimeError(f"{split_root.name} contains negative sample indices")
        if np.any(start_chunk % 8 != 0):
            raise RuntimeError(f"{split_root.name} window start violates stride 8")
        if sequence_chunk.size:
            joined_sequence = sequence_chunk
            joined_start = start_chunk
            if previous_sequence is not None:
                joined_sequence = np.concatenate(
                    (np.asarray([previous_sequence], dtype=np.int64), sequence_chunk)
                )
                joined_start = np.concatenate(
                    (np.asarray([previous_start], dtype=np.int64), start_chunk)
                )
            sequence_delta = np.diff(joined_sequence)
            if np.any(sequence_delta < 0):
                raise RuntimeError(f"{split_root.name} sequence order is not monotonic")
            same_sequence = sequence_delta == 0
            if np.any(np.diff(joined_start)[same_sequence] <= 0):
                raise RuntimeError(
                    f"{split_root.name} window order is not strictly increasing"
                )
            previous_sequence = int(sequence_chunk[-1])
            previous_start = int(start_chunk[-1])

    if int(class_counts.sum()) != count:
        raise RuntimeError(f"{split_root.name} class-count audit failed")
    return {
        "class_counts": class_counts.tolist(),
        "sample_order": "source sequence index, then ascending regular window start",
    }


def validate_materialization(
    data_root: Path, splits: Sequence[str] = ALLOWED_SPLITS
) -> dict[str, Any]:
    requested = normalize_requested_splits(splits)
    data_root = data_root.resolve()
    manifest_path = data_root / MATERIALIZATION_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_root_manifest(manifest)

    details: dict[str, dict[str, Any]] = {}
    for split in requested:
        expected_count = EXPECTED_COUNTS[split]
        split_root = data_root / split
        source_manifest_path = split_root / "split_manifest.json"
        if not source_manifest_path.is_file():
            raise FileNotFoundError(source_manifest_path)
        source_manifest_sha256 = sha256_file(source_manifest_path)
        root_entry = manifest.get("splits", {}).get(split, {})
        if root_entry.get("manifest_sha256") != source_manifest_sha256:
            raise RuntimeError(f"{split} source manifest hash mismatch")
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if not source_manifest.get("integrity", {}).get("passed"):
            raise RuntimeError(f"{split} source payload integrity did not pass")
        if source_manifest.get("written_windows") != expected_count:
            raise RuntimeError(f"{split} source window count mismatch")
        if source_manifest.get("candidate", {}).get("name") != FROZEN_CANDIDATE_NAME:
            raise RuntimeError(f"{split} source candidate mismatch")

        data_path = split_root / "data_joint.npy"
        data = np.load(data_path, mmap_mode="r")
        if data.shape != (expected_count, 3, WINDOW_SIZE, 25, 2):
            raise RuntimeError(f"invalid {split} data shape: {data.shape}")
        if data.dtype != np.float32:
            raise RuntimeError(f"invalid {split} data dtype: {data.dtype}")
        expected_data_bytes = source_manifest.get("payload", {}).get(
            "data_joint.npy", {}
        ).get("bytes")
        if data_path.stat().st_size != expected_data_bytes:
            raise RuntimeError(f"{split} source data byte-size mismatch")

        metadata = _validate_source_metadata(split_root, expected_count)
        details[split] = {
            "root": str(split_root.resolve()),
            "count": expected_count,
            "data_path": str(data_path.resolve()),
            "source_manifest": str(source_manifest_path.resolve()),
            "source_manifest_sha256": source_manifest_sha256,
            "source_data_payload_sha256": source_manifest["payload"][
                "data_joint.npy"
            ]["array_payload_sha256"],
            **metadata,
        }

    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "splits": details,
        "test_ood_arrays_opened": False,
    }


def pooled_backbone_features(
    model: torch.nn.Module, batch: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch.ndim != 5 or tuple(batch.shape[1:]) != (3, WINDOW_SIZE, 25, 2):
        raise RuntimeError(f"invalid DSTE input batch shape: {tuple(batch.shape)}")
    jt = batch.permute(0, 2, 4, 3, 1).reshape(batch.shape[0], WINDOW_SIZE, 150)
    js = batch.permute(0, 4, 3, 2, 1).reshape(batch.shape[0], 50, 192)
    temporal_tokens, spatial_tokens = model.backbone(jt, js)
    temporal = temporal_tokens.amax(dim=1)
    spatial = spatial_tokens.amax(dim=1)
    expected = (batch.shape[0], FEATURE_DIM)
    if tuple(temporal.shape) != expected or tuple(spatial.shape) != expected:
        raise RuntimeError(
            "unexpected DSTE pooled feature shapes: "
            f"temporal={tuple(temporal.shape)} spatial={tuple(spatial.shape)}"
        )
    if not torch.isfinite(temporal).all() or not torch.isfinite(spatial).all():
        raise RuntimeError("DSTE produced non-finite pooled features")
    return temporal, spatial


def immutable_run_contract(
    args: argparse.Namespace,
    materialization: dict[str, Any],
    encoder_sha256: str,
    head_sha256: str,
    adl_config_sha256: str,
    extractor_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "reconstructed SAFER V1 F0B frozen feature extraction",
        "data_root": str(args.data_root.resolve()),
        "materialization_manifest_sha256": materialization["manifest_sha256"],
        "encoder": str(args.encoder.resolve()),
        "encoder_sha256": encoder_sha256,
        "adl_head": str(args.adl_head.resolve()),
        "adl_head_sha256": head_sha256,
        "adl_run_config": str(args.adl_run_config.resolve()),
        "adl_run_config_sha256": adl_config_sha256,
        "extractor_sha256": extractor_sha256,
        "splits": list(args.splits),
        "batch_size": args.batch_size,
        "max_windows_per_split": args.max_windows_per_split,
        "pooling": {
            "temporal": "max over 64 DSTE temporal tokens -> 1024-D",
            "spatial": "max over 50 DSTE spatial tokens -> 1024-D",
            "temporal_spatial": "runtime concatenation -> 2048-D",
        },
        "training": False,
        "optimizer_created": False,
        "test_ood_access": False,
        "recovery_limit": "not a byte-identical reconstruction of the lost F0B cache",
    }


def _validate_memmap(path: Path, shape: tuple[int, ...], dtype: np.dtype) -> np.memmap:
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.load(path, mmap_mode="r+")
    if array.shape != shape or array.dtype != dtype:
        raise RuntimeError(f"invalid resumable array {path}: {array.shape}/{array.dtype}")
    return array


def initialize_output(
    output_dir: Path, contract: dict[str, Any], resume: bool
) -> dict[str, Any] | None:
    state_path = output_dir / RUN_STATE_NAME
    final_path = output_dir / FINAL_MANIFEST_NAME
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise FileExistsError(
                f"refusing to overwrite non-empty output: {output_dir}; "
                "use --resume only with the identical extraction contract"
            )
        if final_path.is_file():
            manifest = json.loads(final_path.read_text(encoding="utf-8"))
            if manifest.get("run_contract") != contract:
                raise RuntimeError("completed feature-cache contract mismatch")
            if not manifest.get("integrity", {}).get("passed"):
                raise RuntimeError("completed feature cache failed integrity")
            return manifest
        if not state_path.is_file():
            raise RuntimeError("cannot resume output without extraction state")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("run_contract") != contract:
            raise RuntimeError("incomplete feature-cache contract mismatch")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(
        state_path,
        {
            "status": "in_progress",
            "research_usable": False,
            "run_contract": contract,
            "completed_splits": [],
        },
    )
    return None


def _copy_source_metadata(
    source_root: Path, output_root: Path, count: int
) -> None:
    for name in METADATA_FILENAMES:
        source = np.load(source_root / name, mmap_mode="r")
        np.save(output_root / name, np.asarray(source[:count]))


def _validate_cached_metadata(
    source_root: Path, output_root: Path, count: int
) -> None:
    for name in METADATA_FILENAMES:
        source = np.load(source_root / name, mmap_mode="r")
        cached = np.load(output_root / name, mmap_mode="r")
        if cached.shape != (count,) or cached.dtype != np.int64:
            raise RuntimeError(f"invalid cached metadata {output_root / name}")
        if not np.array_equal(cached, source[:count]):
            raise RuntimeError(f"cached metadata differs from source: {name}")


def _audit_feature_array(array: np.ndarray, count: int) -> bool:
    if array.shape != (count, FEATURE_DIM) or array.dtype != np.float32:
        return False
    chunk = 32768
    return all(
        bool(np.isfinite(np.asarray(array[begin : begin + chunk])).all())
        for begin in range(0, count, chunk)
    )


def extract_split(
    model: torch.nn.Module,
    device: torch.device,
    split: str,
    source: dict[str, Any],
    output_dir: Path,
    batch_size: int,
    checkpoint_every_batches: int,
    max_windows: int,
    resume: bool,
) -> dict[str, Any]:
    full_count = int(source["count"])
    count = min(full_count, max_windows) if max_windows else full_count
    source_root = Path(source["root"])
    split_output = output_dir / split
    progress_path = split_output / SPLIT_PROGRESS_NAME
    manifest_path = split_output / SPLIT_MANIFEST_NAME
    temporal_path = split_output / FEATURE_FILENAMES[0]
    spatial_path = split_output / FEATURE_FILENAMES[1]

    if split_output.exists() and any(split_output.iterdir()):
        if not resume:
            raise FileExistsError(f"refusing to overwrite split output: {split_output}")
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("written_windows") != count:
                raise RuntimeError(f"completed {split} feature count mismatch")
            if not manifest.get("integrity", {}).get("passed"):
                raise RuntimeError(f"completed {split} feature manifest failed")
            return manifest
        if not progress_path.is_file():
            raise RuntimeError(f"cannot resume {split} without progress state")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("target_windows") != count:
            raise RuntimeError(f"{split} progress target mismatch")
        next_index = int(progress.get("next_index", -1))
        if not 0 <= next_index <= count:
            raise RuntimeError(f"invalid {split} resume index: {next_index}")
        temporal = _validate_memmap(
            temporal_path, (count, FEATURE_DIM), np.dtype(np.float32)
        )
        spatial = _validate_memmap(
            spatial_path, (count, FEATURE_DIM), np.dtype(np.float32)
        )
        _validate_cached_metadata(source_root, split_output, count)
    else:
        split_output.mkdir(parents=True, exist_ok=True)
        temporal = np.lib.format.open_memmap(
            temporal_path,
            mode="w+",
            dtype=np.float32,
            shape=(count, FEATURE_DIM),
        )
        spatial = np.lib.format.open_memmap(
            spatial_path,
            mode="w+",
            dtype=np.float32,
            shape=(count, FEATURE_DIM),
        )
        _copy_source_metadata(source_root, split_output, count)
        next_index = 0
        atomic_json_dump(
            progress_path,
            {
                "status": "in_progress",
                "split": split,
                "target_windows": count,
                "full_split_windows": full_count,
                "next_index": 0,
            },
        )

    source_data = np.load(Path(source["data_path"]), mmap_mode="r")
    total_batches = (count + batch_size - 1) // batch_size
    first_batch = next_index // batch_size
    with torch.inference_mode():
        for batch_number, begin in enumerate(
            range(next_index, count, batch_size), start=first_batch + 1
        ):
            end = min(begin + batch_size, count)
            batch = torch.from_numpy(
                np.array(source_data[begin:end], dtype=np.float32, copy=True)
            ).to(device, non_blocking=True)
            temporal_batch, spatial_batch = pooled_backbone_features(model, batch)
            temporal[begin:end] = temporal_batch.detach().float().cpu().numpy()
            spatial[begin:end] = spatial_batch.detach().float().cpu().numpy()
            should_checkpoint = (
                batch_number % checkpoint_every_batches == 0 or end == count
            )
            if should_checkpoint:
                temporal.flush()
                spatial.flush()
                atomic_json_dump(
                    progress_path,
                    {
                        "status": "in_progress" if end < count else "features_written",
                        "split": split,
                        "target_windows": count,
                        "full_split_windows": full_count,
                        "next_index": end,
                    },
                )
            if batch_number == first_batch + 1 or should_checkpoint:
                print(
                    f"split={split} batch={batch_number}/{total_batches} "
                    f"windows={end}/{count}",
                    flush=True,
                )

    temporal.flush()
    spatial.flush()
    del temporal, spatial, source_data
    temporal_read = np.load(temporal_path, mmap_mode="r")
    spatial_read = np.load(spatial_path, mmap_mode="r")
    temporal_valid = _audit_feature_array(temporal_read, count)
    spatial_valid = _audit_feature_array(spatial_read, count)
    _validate_cached_metadata(source_root, split_output, count)
    del temporal_read, spatial_read

    full_split = count == full_count and max_windows == 0
    checks = {
        "source_manifest_passed": True,
        "written_window_count_exact": count > 0,
        "temporal_shape_dtype_finite": temporal_valid,
        "spatial_shape_dtype_finite": spatial_valid,
        "metadata_exact_source_order": True,
        "full_split": full_split,
    }
    integrity_passed = all(value for key, value in checks.items() if key != "full_split")
    manifest = {
        "split": split,
        "status": "completed" if full_split else "smoke_completed",
        "research_usable": bool(full_split and integrity_passed),
        "written_windows": count,
        "full_split_windows": full_count,
        "feature_contract": {
            "temporal_shape": [count, FEATURE_DIM],
            "spatial_shape": [count, FEATURE_DIM],
            "dtype": "float32",
            "temporal_spatial_combination": "concatenate at head-training time",
        },
        "source": {
            "source_manifest": source["source_manifest"],
            "source_manifest_sha256": source["source_manifest_sha256"],
            "source_data_payload_sha256": source["source_data_payload_sha256"],
            "class_counts": source["class_counts"],
            "sample_order": source["sample_order"],
        },
        "payload": {
            name: {
                "bytes": (split_output / name).stat().st_size,
                "sha256": sha256_file(split_output / name),
            }
            for name in (*FEATURE_FILENAMES, *METADATA_FILENAMES)
        },
        "integrity": {"passed": integrity_passed, **checks},
    }
    atomic_json_dump(manifest_path, manifest)
    atomic_json_dump(
        progress_path,
        {
            "status": manifest["status"],
            "research_usable": manifest["research_usable"],
            "split": split,
            "target_windows": count,
            "full_split_windows": full_count,
            "next_index": count,
            "feature_manifest_sha256": sha256_file(manifest_path),
        },
    )
    return manifest


def estimated_output_bytes(materialization: dict[str, Any], max_windows: int) -> int:
    count = sum(
        min(item["count"], max_windows) if max_windows else item["count"]
        for item in materialization["splits"].values()
    )
    return int(count * (2 * FEATURE_DIM * 4 + len(METADATA_FILENAMES) * 8))


def disk_usage_for_target(path: Path) -> shutil._ntuple_diskusage:
    probe = path.resolve()
    while not probe.exists():
        if probe.parent == probe:
            raise FileNotFoundError(f"no existing parent for output target: {path}")
        probe = probe.parent
    return shutil.disk_usage(probe)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.splits = list(normalize_requested_splits(args.splits))
    if args.batch_size < 1 or args.checkpoint_every_batches < 1:
        raise ValueError("batch-size and checkpoint interval must be positive")
    if args.max_windows_per_split < 0 or args.reserve_gib < 0:
        raise ValueError("max windows and reserve must be non-negative")

    materialization = validate_materialization(args.data_root, args.splits)
    _, adl_config_sha256 = validate_adl_run_config(args.adl_run_config)
    encoder_sha256 = sha256_file(args.encoder)
    head_sha256 = sha256_file(args.adl_head)
    extractor_sha256 = sha256_file(Path(__file__))
    device = enforce_physical_gpu_zero()
    model = load_adl_model(args.encoder.resolve(), args.adl_head.resolve(), device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("DSTE/ADL model is not fully frozen")
    state_hash_before = hash_named_tensors(model.state_dict().items())
    contract = immutable_run_contract(
        args,
        materialization,
        encoder_sha256,
        head_sha256,
        adl_config_sha256,
        extractor_sha256,
    )
    estimate = estimated_output_bytes(materialization, args.max_windows_per_split)
    disk = disk_usage_for_target(args.output_dir)
    reserve = int(args.reserve_gib * 1024**3)
    disk_ok = disk.free >= estimate + reserve
    preflight = {
        "mode": "preflight_only" if args.preflight_only else "extract",
        "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": str(device),
        "device_name": torch.cuda.get_device_name(0),
        "materialization_manifest_sha256": materialization["manifest_sha256"],
        "encoder_sha256": encoder_sha256,
        "adl_head_sha256": head_sha256,
        "adl_run_config_sha256": adl_config_sha256,
        "extractor_sha256": extractor_sha256,
        "estimated_output_bytes": estimate,
        "free_bytes": disk.free,
        "reserve_bytes": reserve,
        "disk_gate_passed": disk_ok,
        "splits": args.splits,
        "test_ood_arrays_opened": materialization["test_ood_arrays_opened"],
    }
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if not disk_ok:
        raise RuntimeError("insufficient free space for feature cache plus reserve")
    if args.preflight_only:
        return preflight

    output_dir = args.output_dir.resolve()
    completed = initialize_output(output_dir, contract, args.resume)
    if completed is not None:
        print(json.dumps({
            "output_dir": str(output_dir),
            "status": "already_completed",
            "research_usable": completed["research_usable"],
            "integrity": completed["integrity"],
        }, ensure_ascii=False, indent=2), flush=True)
        return completed

    results: dict[str, dict[str, Any]] = {}
    for split in args.splits:
        results[split] = extract_split(
            model=model,
            device=device,
            split=split,
            source=materialization["splits"][split],
            output_dir=output_dir,
            batch_size=args.batch_size,
            checkpoint_every_batches=args.checkpoint_every_batches,
            max_windows=args.max_windows_per_split,
            resume=args.resume,
        )
        state = json.loads((output_dir / RUN_STATE_NAME).read_text(encoding="utf-8"))
        state["completed_splits"] = list(results)
        atomic_json_dump(output_dir / RUN_STATE_NAME, state)

    state_hash_after = hash_named_tensors(model.state_dict().items())
    full_extraction = (
        tuple(args.splits) == ALLOWED_SPLITS
        and args.max_windows_per_split == 0
        and all(results[split]["integrity"]["full_split"] for split in ALLOWED_SPLITS)
    )
    checks = {
        "materialization_gate_passed": True,
        "physical_gpu_zero_only": os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
        "epoch150_adl_contract_passed": True,
        "no_optimizer_or_training": True,
        "model_frozen_exact": state_hash_before == state_hash_after,
        "train_validation_only": tuple(args.splits) == ALLOWED_SPLITS,
        "test_ood_arrays_unopened": not materialization["test_ood_arrays_opened"],
        "all_split_payloads_passed": all(
            result["integrity"]["passed"] for result in results.values()
        ),
        "full_train_validation_extraction": full_extraction,
    }
    integrity_passed = all(
        value
        for key, value in checks.items()
        if key != "full_train_validation_extraction"
    )
    research_usable = bool(full_extraction and integrity_passed)
    manifest = {
        "experiment": "reconstructed SAFER V1 F0B frozen feature extraction",
        "status": "completed" if research_usable else "smoke_completed",
        "research_usable": research_usable,
        "run_contract": contract,
        "recovery_status": {
            "historical_representation_contract_recovered": True,
            "original_f0b_source_and_cache_recovered": False,
            "no_f0b_optimizer_configuration_selected_here": True,
            "must_not_be_called_byte_identical_reproduction": True,
        },
        "protocol": {
            "training": False,
            "optimizer": None,
            "splits": list(args.splits),
            "test_ood_access": False,
            "next_gate": (
                "freeze an explicitly sourced F0B head-training configuration; "
                "validation selects representation/checkpoint before test/OOD access"
            ),
        },
        "environment": {
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "physical_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_device": str(device),
            "device_name": torch.cuda.get_device_name(0),
        },
        "model_state_hash_before": state_hash_before,
        "model_state_hash_after": state_hash_after,
        "splits": {
            split: {
                "feature_manifest": str(
                    (output_dir / split / SPLIT_MANIFEST_NAME).resolve()
                ),
                "feature_manifest_sha256": sha256_file(
                    output_dir / split / SPLIT_MANIFEST_NAME
                ),
                "written_windows": result["written_windows"],
                "integrity_passed": result["integrity"]["passed"],
            }
            for split, result in results.items()
        },
        "integrity": {"passed": integrity_passed, **checks},
    }
    atomic_json_dump(output_dir / FINAL_MANIFEST_NAME, manifest)
    atomic_json_dump(
        output_dir / RUN_STATE_NAME,
        {
            "status": manifest["status"],
            "research_usable": research_usable,
            "run_contract": contract,
            "completed_splits": list(results),
            "final_manifest_sha256": sha256_file(
                output_dir / FINAL_MANIFEST_NAME
            ),
        },
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "status": manifest["status"],
        "research_usable": research_usable,
        "written_windows": {
            split: result["written_windows"] for split, result in results.items()
        },
        "feature_shapes": {
            split: {
                "temporal": result["feature_contract"]["temporal_shape"],
                "spatial": result["feature_contract"]["spatial_shape"],
            }
            for split, result in results.items()
        },
        "integrity": manifest["integrity"],
    }, ensure_ascii=False, indent=2), flush=True)
    if args.require_valid and not manifest["integrity"]["passed"]:
        raise SystemExit(2)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument(
        "--adl-head", type=Path, default=DEFAULT_ADL_ROOT / "best_adl_head.pth"
    )
    parser.add_argument(
        "--adl-run-config", type=Path, default=DEFAULT_ADL_ROOT / "run_config.json"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--splits", nargs="+", choices=ALLOWED_SPLITS, default=list(ALLOWED_SPLITS)
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-every-batches", type=int, default=25)
    parser.add_argument(
        "--max-windows-per-split",
        type=int,
        default=0,
        help="prefix cap for smoke only; zero extracts both complete splits",
    )
    parser.add_argument("--reserve-gib", type=float, default=10.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-valid", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
