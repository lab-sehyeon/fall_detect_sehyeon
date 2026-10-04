#!/usr/bin/env python3
"""Evaluate the frozen reconstructed SAFER F0B selection on test and OOD.

The validation-only F0B recovery run must already be complete.  This command
accepts only its frozen temporal+spatial epoch-46 head and evaluates the full
reconstructed V1 test and OOD splits in that order.  It has no training,
optimizer, threshold fitting, candidate comparison, or checkpoint selection
path.  The result remains a post-selection recovery-control result, not an
exact reproduction of the lost historical F0B optimizer or V1 mapper.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

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
from fall_pipeline.safer.extract_f0b_safer_features import (
    FEATURE_DIM,
    FEATURE_FILENAMES,
    METADATA_FILENAMES,
    pooled_backbone_features,
)
from fall_pipeline.safer.train_f0b_safer_heads import (
    CLASS_NAMES,
    FINAL_REPORT_NAME as TRAINING_REPORT_NAME,
    classification_metrics,
)


SPLIT_ORDER = ("test", "ood")
EXPECTED_COUNTS = {
    split: int(EXPECTED_SPLITS[split]["clean_windows"]) for split in SPLIT_ORDER
}
EXPECTED_CONFIG_SHA256 = (
    "b27aae0da7c5d7e8ed793a06009e97640db1d98d7ee9ac3f29c3940055250a34"
)
DEFAULT_CONFIG = Path("configs/f0b_safer_v1_recovery_postselection_v1.json")
DEFAULT_TRAINING_RUN_DIR = Path(
    "checkpoint/fall/F0B_SAFER_V1_RECOVERY_SGD_20260904_R1"
)
DEFAULT_OUTPUT_DIR = Path(
    "checkpoint/fall/F0B_SAFER_V1_RECOVERY_SGD_POSTSELECTION_20260904_R1"
)
RUN_STATE_NAME = "evaluation_state.json"
FINAL_REPORT_NAME = "f0b_postselection_report.json"
SPLIT_PROGRESS_NAME = "progress.json"
SPLIT_RESULT_NAME = "result.json"
LOGITS_NAME = "logits.npy"


def strict_json_load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def atomic_json_dump(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_eval_config(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    actual_sha256 = sha256_file(path)
    if actual_sha256 != EXPECTED_CONFIG_SHA256:
        raise RuntimeError(
            "post-selection configuration hash differs from the preregistered "
            f"contract: {actual_sha256}"
        )
    config = strict_json_load(path)
    expected = {
        "schema_version": 1,
        "status": "preregistered_post_selection_evaluation",
        "experiment_id": "F0B_SAFER_V1_RECOVERY_SGD_POSTSELECTION_R1",
        "candidate": "temporal_spatial",
        "epoch": 46,
        "splits": {"test": 216768, "ood": 71964},
        "split_order": ["test", "ood"],
        "selected_input_dim": 2048,
        "class_names": list(CLASS_NAMES),
        "batch_size": 256,
        "training": False,
        "optimizer": None,
        "threshold_fitting": False,
        "candidate_comparison": False,
        "checkpoint_reselection": False,
        "physical_cuda_visible_devices": "0",
    }
    actual = {
        "schema_version": config.get("schema_version"),
        "status": config.get("status"),
        "experiment_id": config.get("experiment_id"),
        "candidate": config.get("selection_lock", {}).get("candidate"),
        "epoch": config.get("selection_lock", {}).get("epoch"),
        "splits": config.get("input", {}).get("splits"),
        "split_order": config.get("evaluation", {}).get("split_order"),
        "selected_input_dim": config.get("representation", {}).get(
            "selected_input_dim"
        ),
        "class_names": config.get("representation", {}).get("class_names"),
        "batch_size": config.get("evaluation", {}).get("batch_size"),
        "training": config.get("evaluation", {}).get("training"),
        "optimizer": config.get("evaluation", {}).get("optimizer"),
        "threshold_fitting": config.get("evaluation", {}).get(
            "threshold_fitting"
        ),
        "candidate_comparison": config.get("evaluation", {}).get(
            "candidate_comparison"
        ),
        "checkpoint_reselection": config.get("evaluation", {}).get(
            "checkpoint_reselection"
        ),
        "physical_cuda_visible_devices": config.get("execution", {}).get(
            "physical_cuda_visible_devices"
        ),
    }
    if actual != expected:
        raise RuntimeError(
            "post-selection configuration content differs from the locked contract"
        )
    lock = config["selection_lock"]
    if not lock.get("selection_frozen_before_test_ood"):
        raise RuntimeError("configuration does not prove frozen selection")
    if not lock.get("test_ood_unopened_before_selection"):
        raise RuntimeError("configuration does not prove prior test/OOD isolation")
    if not config["evaluation"].get("test_ood_used_once_after_validation_selection"):
        raise RuntimeError("configuration does not lock one post-selection evaluation")
    if not config["claim_boundary"].get(
        "must_not_be_called_historical_exact_reproduction"
    ):
        raise RuntimeError("configuration is missing the historical claim boundary")
    return config, actual_sha256


def validate_selection_run(
    training_run_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    training_run_dir = training_run_dir.resolve()
    report_path = training_run_dir / TRAINING_REPORT_NAME
    selected_head_path = training_run_dir / "selected_head.pth"
    lock = config["selection_lock"]
    report_sha256 = sha256_file(report_path)
    if report_sha256 != lock["training_report_sha256"]:
        raise RuntimeError("F0B training report hash mismatch")
    report = strict_json_load(report_path)
    integrity = report.get("integrity", {})
    required_integrity = (
        "passed",
        "feature_cache_gate_passed",
        "physical_gpu_zero_only",
        "explicit_recovery_config_frozen",
        "only_two_linear_heads_trainable",
        "same_minibatch_order_for_candidates",
        "unweighted_cross_entropy",
        "full_validation_each_epoch",
        "selection_uses_validation_only",
        "test_ood_unopened",
        "all_50_epochs_completed",
    )
    if report.get("status") != "completed" or not report.get("research_usable"):
        raise RuntimeError("F0B validation-selection run is not complete")
    if not all(integrity.get(key) is True for key in required_integrity):
        raise RuntimeError("F0B validation-selection integrity did not pass")
    selection = report.get("selection", {})
    expected_selection = {
        "candidate": lock["candidate"],
        "epoch": lock["epoch"],
        "primary": lock["validation_primary_value"],
        "secondary": lock["validation_secondary_value"],
        "frozen": True,
    }
    actual_selection = {
        "candidate": selection.get("candidate"),
        "epoch": selection.get("epoch"),
        "primary": selection.get("metrics", {}).get("macro_f1"),
        "secondary": selection.get("metrics", {}).get(
            "conditional_fall_vs_lie_auprc"
        ),
        "frozen": selection.get("frozen_before_test_ood"),
    }
    if actual_selection != expected_selection:
        raise RuntimeError("F0B selected candidate/epoch/metrics changed")
    contract = report.get("run_contract", {})
    expected_input = config["input"]
    if contract.get("mode") != "full" or contract.get("test_ood_access") is not False:
        raise RuntimeError("F0B training report does not prove development isolation")
    if contract.get("config_sha256") != expected_input[
        "f0b_training_config_sha256"
    ]:
        raise RuntimeError("F0B training configuration hash mismatch")
    if contract.get("trainer_sha256") != expected_input["f0b_trainer_sha256"]:
        raise RuntimeError("F0B trainer hash mismatch")
    if sha256_file(selected_head_path) != lock["selected_head_sha256"]:
        raise RuntimeError("selected F0B head file hash mismatch")
    if report.get("selected_head_sha256") != lock["selected_head_sha256"]:
        raise RuntimeError("selected F0B head provenance mismatch")
    selected_summary = report.get("candidates", {}).get(lock["candidate"], {})
    if selected_summary.get("head_state_hash") != lock["selected_head_state_hash"]:
        raise RuntimeError("selected F0B head tensor-state provenance mismatch")
    if selected_summary.get("best_epoch") != lock["epoch"]:
        raise RuntimeError("selected F0B summary epoch mismatch")
    return {
        "report": report,
        "report_path": str(report_path),
        "report_sha256": report_sha256,
        "selected_head_path": str(selected_head_path),
        "selected_head_sha256": lock["selected_head_sha256"],
        "selected_head_state_hash": lock["selected_head_state_hash"],
    }


def validate_materialization_manifests(
    data_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    data_root = data_root.resolve()
    manifest_path = data_root / MATERIALIZATION_MANIFEST_NAME
    expected_input = config["input"]
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != expected_input["materialization_manifest_sha256"]:
        raise RuntimeError("SAFER materialization manifest hash mismatch")
    manifest = strict_json_load(manifest_path)
    if manifest.get("status") != "completed" or not manifest.get(
        "research_usable"
    ):
        raise RuntimeError("SAFER materialization is not research-usable")
    if not manifest.get("integrity", {}).get("passed"):
        raise RuntimeError("SAFER materialization integrity did not pass")
    if manifest.get("candidate", {}).get("name") != FROZEN_CANDIDATE_NAME:
        raise RuntimeError("unexpected SAFER reconstruction candidate")
    recovery = manifest.get("recovery_status", {})
    if not recovery.get("experimental_compatibility_reconstruction"):
        raise RuntimeError("SAFER reconstruction limitation is missing")
    if not recovery.get("must_not_be_called_byte_identical_reproduction"):
        raise RuntimeError("SAFER byte-identity limitation is missing")
    protocol = manifest.get("protocol", {})
    if protocol.get("window_size") != WINDOW_SIZE:
        raise RuntimeError("SAFER materialization window size mismatch")
    if protocol.get("test_and_ood_used_for_candidate_selection") is not False:
        raise RuntimeError("test/OOD selection isolation was not proved")
    if protocol.get("test_and_ood_materialized_only_after_structural_freeze") is not True:
        raise RuntimeError("post-freeze test/OOD materialization was not proved")

    split_details: dict[str, dict[str, Any]] = {}
    for split in SPLIT_ORDER:
        count = int(expected_input["splits"][split])
        if count != EXPECTED_COUNTS[split]:
            raise RuntimeError(f"configured {split} count differs from history")
        split_root = data_root / split
        split_manifest_path = split_root / "split_manifest.json"
        split_manifest_sha256 = sha256_file(split_manifest_path)
        root_entry = manifest.get("splits", {}).get(split, {})
        if root_entry.get("manifest_sha256") != split_manifest_sha256:
            raise RuntimeError(f"{split} materialization manifest hash mismatch")
        split_manifest = strict_json_load(split_manifest_path)
        if split_manifest.get("written_windows") != count:
            raise RuntimeError(f"{split} materialized count mismatch")
        if split_manifest.get("data_shape") != [count, 3, WINDOW_SIZE, 25, 2]:
            raise RuntimeError(f"{split} materialized shape contract mismatch")
        if split_manifest.get("data_dtype") != "float32":
            raise RuntimeError(f"{split} materialized dtype contract mismatch")
        if split_manifest.get("candidate", {}).get("name") != FROZEN_CANDIDATE_NAME:
            raise RuntimeError(f"{split} reconstruction candidate mismatch")
        if not split_manifest.get("integrity", {}).get("passed"):
            raise RuntimeError(f"{split} materialization integrity did not pass")
        paths = {
            name: split_root / name
            for name in ("data_joint.npy", *METADATA_FILENAMES)
        }
        payload = split_manifest.get("payload", {})
        for name, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != payload.get(name, {}).get("bytes"):
                raise RuntimeError(f"{split}/{name} byte-size mismatch")
        split_details[split] = {
            "root": str(split_root),
            "count": count,
            "data_path": str(paths["data_joint.npy"]),
            "source_manifest": str(split_manifest_path),
            "source_manifest_sha256": split_manifest_sha256,
            "source_data_payload_sha256": payload["data_joint.npy"][
                "array_payload_sha256"
            ],
        }
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "splits": split_details,
        "payload_arrays_opened": False,
    }


def load_selected_head(
    path: Path, device: torch.device, expected_state_hash: str
) -> nn.Linear:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or set(state) != {"weight", "bias"}:
        raise RuntimeError("selected F0B head is not a plain Linear state dict")
    if tuple(state["weight"].shape) != (4, 2 * FEATURE_DIM):
        raise RuntimeError("selected F0B head weight shape mismatch")
    if tuple(state["bias"].shape) != (4,):
        raise RuntimeError("selected F0B head bias shape mismatch")
    if hash_named_tensors(state.items()) != expected_state_hash:
        raise RuntimeError("selected F0B head tensor hash mismatch")
    head = nn.Linear(2 * FEATURE_DIM, 4).to(device)
    head.load_state_dict(state, strict=True)
    head.requires_grad_(False)
    head.eval()
    return head


def validate_source_arrays(source: dict[str, Any]) -> dict[str, Any]:
    split_root = Path(source["root"])
    count = int(source["count"])
    data = np.load(source["data_path"], mmap_mode="r")
    if data.shape != (count, 3, WINDOW_SIZE, 25, 2) or data.dtype != np.float32:
        raise RuntimeError(f"invalid {split_root.name} source data array")
    arrays = {
        name: np.load(split_root / name, mmap_mode="r")
        for name in METADATA_FILENAMES
    }
    for name, array in arrays.items():
        if array.shape != (count,) or array.dtype != np.int64:
            raise RuntimeError(f"invalid {split_root.name}/{name}")
    coarse = np.asarray(arrays["center_coarse_labels.npy"])
    derived = np.asarray(arrays["center_derived_labels.npy"])
    if not np.array_equal(derive_four_class(coarse), derived):
        raise RuntimeError(f"{split_root.name} center labels disagree")
    sequence = np.asarray(arrays["sequence_index.npy"])
    starts = np.asarray(arrays["window_start.npy"])
    if np.any(sequence < 0) or np.any(starts < 0) or np.any(starts % 8 != 0):
        raise RuntimeError(f"{split_root.name} sample indices are invalid")
    sequence_delta = np.diff(sequence)
    if np.any(sequence_delta < 0):
        raise RuntimeError(f"{split_root.name} sequence order is not monotonic")
    if np.any(np.diff(starts)[sequence_delta == 0] <= 0):
        raise RuntimeError(f"{split_root.name} window order is not increasing")
    return {
        "data": data,
        "metadata": arrays,
        "class_counts": np.bincount(derived, minlength=4).tolist(),
    }


def forward_selected(
    model: nn.Module, head: nn.Linear, batch: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    temporal, spatial = pooled_backbone_features(model, batch)
    logits = head(torch.cat((temporal, spatial), dim=1))
    if tuple(logits.shape) != (batch.shape[0], 4):
        raise RuntimeError(f"invalid F0B logits shape: {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise RuntimeError("selected F0B head produced non-finite logits")
    return temporal, spatial, logits


def historical_difference(
    split: str, metrics: dict[str, Any], config: dict[str, Any]
) -> dict[str, float | None]:
    reference = config["evaluation"]["historical_v1_reference_percent"][split]
    return {
        key: None if metrics[key] is None else float(metrics[key]) * 100.0 - value
        for key, value in reference.items()
    }


def immutable_run_contract(
    args: argparse.Namespace,
    config_sha256: str,
    materialization: dict[str, Any],
    selection: dict[str, Any],
    evaluator_sha256: str,
    metric_code_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "F0B SAFER V1 recovery post-selection test/OOD evaluation",
        "config": str(args.config.resolve()),
        "config_sha256": config_sha256,
        "data_root": str(args.data_root.resolve()),
        "materialization_manifest_sha256": materialization["manifest_sha256"],
        "encoder": str(args.encoder.resolve()),
        "encoder_sha256": sha256_file(args.encoder),
        "adl_head": str(args.adl_head.resolve()),
        "adl_head_sha256": sha256_file(args.adl_head),
        "adl_run_config": str(args.adl_run_config.resolve()),
        "adl_run_config_sha256": sha256_file(args.adl_run_config),
        "training_run_dir": str(args.training_run_dir.resolve()),
        "training_report_sha256": selection["report_sha256"],
        "selected_head_sha256": selection["selected_head_sha256"],
        "selected_head_state_hash": selection["selected_head_state_hash"],
        "selected_candidate": "temporal_spatial",
        "selected_epoch": 46,
        "splits": list(SPLIT_ORDER),
        "evaluator_sha256": evaluator_sha256,
        "metric_code_sha256": metric_code_sha256,
        "training": False,
        "optimizer": None,
        "threshold_fitting": False,
        "candidate_comparison": False,
        "checkpoint_reselection": False,
    }


def initialize_output(
    output_dir: Path, contract: dict[str, Any], resume: bool
) -> dict[str, Any] | None:
    state_path = output_dir / RUN_STATE_NAME
    report_path = output_dir / FINAL_REPORT_NAME
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise FileExistsError(
                f"refusing to overwrite non-empty output: {output_dir}; "
                "use --resume only for the identical post-selection contract"
            )
        if report_path.is_file():
            report = strict_json_load(report_path)
            if report.get("run_contract") != contract:
                raise RuntimeError("completed F0B post-selection contract mismatch")
            if not report.get("integrity", {}).get("passed"):
                raise RuntimeError("completed F0B post-selection integrity failed")
            return report
        if not state_path.is_file():
            raise RuntimeError("cannot resume post-selection output without state")
        state = strict_json_load(state_path)
        if state.get("run_contract") != contract:
            raise RuntimeError("incomplete F0B post-selection contract mismatch")
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


def _validate_memmap(
    path: Path, shape: tuple[int, ...], dtype: np.dtype
) -> np.memmap:
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.load(path, mmap_mode="r+")
    if array.shape != shape or array.dtype != dtype:
        raise RuntimeError(f"invalid resumable array {path}: {array.shape}/{array.dtype}")
    return array


def _copy_metadata(source_root: Path, output_root: Path, count: int) -> None:
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
            raise RuntimeError(f"invalid cached metadata: {output_root / name}")
        if not np.array_equal(cached, source):
            raise RuntimeError(f"cached metadata differs from source: {name}")


def _all_finite(array: np.ndarray, chunk: int = 32768) -> bool:
    return all(
        bool(np.isfinite(np.asarray(array[begin : begin + chunk])).all())
        for begin in range(0, len(array), chunk)
    )


def evaluate_split(
    model: nn.Module,
    head: nn.Linear,
    device: torch.device,
    split: str,
    source: dict[str, Any],
    output_dir: Path,
    config: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    count = int(source["count"])
    batch_size = int(config["evaluation"]["batch_size"])
    checkpoint_every = int(config["execution"]["checkpoint_every_batches"])
    source_root = Path(source["root"])
    validated_source = validate_source_arrays(source)
    source_data = validated_source["data"]
    split_root = output_dir / split
    progress_path = split_root / SPLIT_PROGRESS_NAME
    result_path = split_root / SPLIT_RESULT_NAME
    temporal_path = split_root / FEATURE_FILENAMES[0]
    spatial_path = split_root / FEATURE_FILENAMES[1]
    logits_path = split_root / LOGITS_NAME

    if result_path.is_file():
        if not resume:
            raise FileExistsError(f"F0B post-selection split exists: {split_root}")
        result = strict_json_load(result_path)
        if result.get("evaluated_windows") != count:
            raise RuntimeError(f"completed {split} count mismatch")
        if not result.get("integrity", {}).get("passed"):
            raise RuntimeError(f"completed {split} integrity failed")
        return result

    continuing = progress_path.is_file()
    if continuing and not resume:
        raise FileExistsError(f"incomplete {split} output requires --resume")
    if continuing:
        progress = strict_json_load(progress_path)
        if progress.get("split") != split or progress.get("target_windows") != count:
            raise RuntimeError(f"{split} progress contract mismatch")
        next_index = int(progress.get("next_index", -1))
        if not 0 <= next_index <= count:
            raise RuntimeError(f"invalid {split} resume index")
        temporal = _validate_memmap(
            temporal_path, (count, FEATURE_DIM), np.dtype(np.float32)
        )
        spatial = _validate_memmap(
            spatial_path, (count, FEATURE_DIM), np.dtype(np.float32)
        )
        logits = _validate_memmap(logits_path, (count, 4), np.dtype(np.float32))
        _validate_cached_metadata(source_root, split_root, count)
    else:
        if split_root.exists() and any(split_root.iterdir()):
            raise FileExistsError(f"refusing to overwrite split output: {split_root}")
        split_root.mkdir(parents=True, exist_ok=True)
        temporal = np.lib.format.open_memmap(
            temporal_path, mode="w+", dtype=np.float32, shape=(count, FEATURE_DIM)
        )
        spatial = np.lib.format.open_memmap(
            spatial_path, mode="w+", dtype=np.float32, shape=(count, FEATURE_DIM)
        )
        logits = np.lib.format.open_memmap(
            logits_path, mode="w+", dtype=np.float32, shape=(count, 4)
        )
        _copy_metadata(source_root, split_root, count)
        next_index = 0
        atomic_json_dump(
            progress_path,
            {
                "status": "in_progress",
                "split": split,
                "target_windows": count,
                "next_index": 0,
            },
        )

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
            temporal_batch, spatial_batch, logits_batch = forward_selected(
                model, head, batch
            )
            temporal[begin:end] = temporal_batch.float().cpu().numpy()
            spatial[begin:end] = spatial_batch.float().cpu().numpy()
            logits[begin:end] = logits_batch.float().cpu().numpy()
            should_checkpoint = batch_number % checkpoint_every == 0 or end == count
            if should_checkpoint:
                temporal.flush()
                spatial.flush()
                logits.flush()
                atomic_json_dump(
                    progress_path,
                    {
                        "status": "in_progress" if end < count else "features_written",
                        "split": split,
                        "target_windows": count,
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
    logits.flush()
    del temporal, spatial, logits, source_data
    temporal_read = np.load(temporal_path, mmap_mode="r")
    spatial_read = np.load(spatial_path, mmap_mode="r")
    logits_read = np.load(logits_path, mmap_mode="r")
    labels = np.load(split_root / "center_derived_labels.npy", mmap_mode="r")
    checks = {
        "full_window_count": len(labels) == count,
        "temporal_shape_dtype_finite": (
            temporal_read.shape == (count, FEATURE_DIM)
            and temporal_read.dtype == np.float32
            and _all_finite(temporal_read)
        ),
        "spatial_shape_dtype_finite": (
            spatial_read.shape == (count, FEATURE_DIM)
            and spatial_read.dtype == np.float32
            and _all_finite(spatial_read)
        ),
        "logits_shape_dtype_finite": (
            logits_read.shape == (count, 4)
            and logits_read.dtype == np.float32
            and _all_finite(logits_read)
        ),
        "metadata_exact_source_order": True,
        "fixed_selected_head_only": True,
        "no_training_or_threshold_fitting": True,
    }
    _validate_cached_metadata(source_root, split_root, count)
    metrics = classification_metrics(
        np.array(labels, copy=True), np.array(logits_read, copy=True)
    )
    del temporal_read, spatial_read, logits_read, labels
    integrity_passed = all(checks.values())
    payload_names = (*FEATURE_FILENAMES, LOGITS_NAME, *METADATA_FILENAMES)
    result = {
        "split": split,
        "status": "completed" if integrity_passed else "failed",
        "research_usable": integrity_passed,
        "evaluated_windows": count,
        "class_counts": validated_source["class_counts"],
        "metrics": metrics,
        "historical_v1_reference_percent": config["evaluation"][
            "historical_v1_reference_percent"
        ][split],
        "current_minus_historical_percentage_points": historical_difference(
            split, metrics, config
        ),
        "source": {
            "source_manifest": source["source_manifest"],
            "source_manifest_sha256": source["source_manifest_sha256"],
            "source_data_payload_sha256": source["source_data_payload_sha256"],
        },
        "payload": {
            name: {
                "bytes": (split_root / name).stat().st_size,
                "sha256": sha256_file(split_root / name),
            }
            for name in payload_names
        },
        "integrity": {"passed": integrity_passed, **checks},
    }
    atomic_json_dump(result_path, result)
    atomic_json_dump(
        progress_path,
        {
            "status": result["status"],
            "research_usable": result["research_usable"],
            "split": split,
            "target_windows": count,
            "next_index": count,
            "result_sha256": sha256_file(result_path),
        },
    )
    return result


def estimated_output_bytes(config: dict[str, Any]) -> int:
    count = sum(int(config["input"]["splits"][split]) for split in SPLIT_ORDER)
    feature_bytes = 2 * FEATURE_DIM * np.dtype(np.float32).itemsize
    logits_bytes = 4 * np.dtype(np.float32).itemsize
    metadata_bytes = len(METADATA_FILENAMES) * np.dtype(np.int64).itemsize
    return int(count * (feature_bytes + logits_bytes + metadata_bytes))


def disk_usage_for_target(path: Path) -> shutil._ntuple_diskusage:
    probe = path.resolve()
    while not probe.exists():
        if probe.parent == probe:
            raise FileNotFoundError(f"no existing parent for output target: {path}")
        probe = probe.parent
    return shutil.disk_usage(probe)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, config_sha256 = load_eval_config(args.config)
    materialization = validate_materialization_manifests(args.data_root, config)
    selection = validate_selection_run(args.training_run_dir, config)
    expected_input = config["input"]
    pinned_files = {
        "encoder": (args.encoder, expected_input["encoder_sha256"]),
        "adl_head": (args.adl_head, expected_input["adl_head_sha256"]),
        "adl_run_config": (
            args.adl_run_config,
            expected_input["adl_run_config_sha256"],
        ),
    }
    for name, (path, expected_sha256) in pinned_files.items():
        if sha256_file(path) != expected_sha256:
            raise RuntimeError(f"pinned {name} hash mismatch")
    _, adl_config_sha256 = validate_adl_run_config(args.adl_run_config)
    if adl_config_sha256 != expected_input["adl_run_config_sha256"]:
        raise RuntimeError("epoch-150 ADL run config mismatch")

    device = enforce_physical_gpu_zero()
    model = load_adl_model(args.encoder.resolve(), args.adl_head.resolve(), device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("DSTE/ADL model is not fully frozen")
    head = load_selected_head(
        Path(selection["selected_head_path"]),
        device,
        selection["selected_head_state_hash"],
    )
    if any(parameter.requires_grad for parameter in head.parameters()):
        raise RuntimeError("selected F0B head is not frozen")
    model_hash_before = hash_named_tensors(model.state_dict().items())
    head_hash_before = hash_named_tensors(head.state_dict().items())
    evaluator_sha256 = sha256_file(Path(__file__).resolve())
    metric_code_sha256 = sha256_file(
        Path(__file__).with_name("train_f0b_safer_heads.py")
    )
    contract = immutable_run_contract(
        args,
        config_sha256,
        materialization,
        selection,
        evaluator_sha256,
        metric_code_sha256,
    )
    estimate = estimated_output_bytes(config)
    disk = disk_usage_for_target(args.output_dir)
    reserve = int(config["execution"]["reserve_gib"] * 1024**3)
    disk_ok = disk.free >= estimate + reserve
    preflight = {
        "mode": "preflight_only" if args.preflight_only else "evaluate",
        "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": str(device),
        "device_name": torch.cuda.get_device_name(0),
        "config_sha256": config_sha256,
        "materialization_manifest_sha256": materialization["manifest_sha256"],
        "encoder_sha256": expected_input["encoder_sha256"],
        "adl_head_sha256": expected_input["adl_head_sha256"],
        "adl_run_config_sha256": expected_input["adl_run_config_sha256"],
        "training_report_sha256": selection["report_sha256"],
        "selected_head_sha256": selection["selected_head_sha256"],
        "selected_head_state_hash": head_hash_before,
        "evaluator_sha256": evaluator_sha256,
        "metric_code_sha256": metric_code_sha256,
        "selected_candidate": config["selection_lock"]["candidate"],
        "selected_epoch": config["selection_lock"]["epoch"],
        "splits": list(SPLIT_ORDER),
        "expected_windows": config["input"]["splits"],
        "estimated_output_bytes": estimate,
        "free_bytes": disk.free,
        "reserve_bytes": reserve,
        "disk_gate_passed": disk_ok,
        "test_ood_payload_arrays_opened": materialization[
            "payload_arrays_opened"
        ],
        "training_or_tuning_available": False,
    }
    print(
        json.dumps(preflight, ensure_ascii=False, indent=2, allow_nan=False),
        flush=True,
    )
    if not disk_ok:
        raise RuntimeError("insufficient free space for post-selection artifacts")
    if args.preflight_only:
        return preflight

    output_dir = args.output_dir.resolve()
    completed = initialize_output(output_dir, contract, args.resume)
    if completed is not None:
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "status": "already_completed",
                    "research_usable": completed["research_usable"],
                    "integrity": completed["integrity"],
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            flush=True,
        )
        return completed

    results: dict[str, dict[str, Any]] = {}
    for split in SPLIT_ORDER:
        results[split] = evaluate_split(
            model=model,
            head=head,
            device=device,
            split=split,
            source=materialization["splits"][split],
            output_dir=output_dir,
            config=config,
            resume=args.resume,
        )
        state = strict_json_load(output_dir / RUN_STATE_NAME)
        state["completed_splits"] = list(results)
        atomic_json_dump(output_dir / RUN_STATE_NAME, state)

    model_hash_after = hash_named_tensors(model.state_dict().items())
    head_hash_after = hash_named_tensors(head.state_dict().items())
    checks = {
        "materialization_gate_passed": True,
        "validation_selection_gate_passed": True,
        "physical_gpu_zero_only": os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
        "epoch150_adl_contract_passed": True,
        "selected_temporal_spatial_epoch46_only": True,
        "selection_frozen_before_test_ood": True,
        "no_training_optimizer_or_threshold_fitting": True,
        "no_candidate_comparison_or_checkpoint_reselection": True,
        "dste_adl_model_frozen_exact": model_hash_before == model_hash_after,
        "selected_head_frozen_exact": head_hash_before == head_hash_after,
        "full_test_ood_evaluation": tuple(results) == SPLIT_ORDER
        and all(
            results[split]["evaluated_windows"] == EXPECTED_COUNTS[split]
            for split in SPLIT_ORDER
        ),
        "all_split_payloads_passed": all(
            results[split]["integrity"]["passed"] for split in SPLIT_ORDER
        ),
    }
    integrity_passed = all(checks.values())
    report = {
        "experiment": "F0B SAFER V1 recovery post-selection test/OOD evaluation",
        "status": "completed" if integrity_passed else "failed",
        "research_usable": integrity_passed,
        "run_contract": contract,
        "claim_boundary": config["claim_boundary"],
        "selection_lock": config["selection_lock"],
        "protocol": config["evaluation"],
        "input": {
            "materialization_manifest": materialization["manifest_path"],
            "materialization_manifest_sha256": materialization["manifest_sha256"],
            "training_report": selection["report_path"],
            "training_report_sha256": selection["report_sha256"],
            "selected_head": selection["selected_head_path"],
            "selected_head_sha256": selection["selected_head_sha256"],
            "encoder_sha256": expected_input["encoder_sha256"],
            "adl_head_sha256": expected_input["adl_head_sha256"],
            "adl_run_config_sha256": expected_input["adl_run_config_sha256"],
        },
        "environment": {
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "physical_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_device": str(device),
            "device_name": torch.cuda.get_device_name(0),
        },
        "model_state_hash_before": model_hash_before,
        "model_state_hash_after": model_hash_after,
        "selected_head_state_hash_before": head_hash_before,
        "selected_head_state_hash_after": head_hash_after,
        "results": results,
        "integrity": {"passed": integrity_passed, **checks},
    }
    atomic_json_dump(output_dir / FINAL_REPORT_NAME, report)
    atomic_json_dump(
        output_dir / RUN_STATE_NAME,
        {
            "status": report["status"],
            "research_usable": report["research_usable"],
            "run_contract": contract,
            "completed_splits": list(results),
            "final_report_sha256": sha256_file(output_dir / FINAL_REPORT_NAME),
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "status": report["status"],
                "research_usable": report["research_usable"],
                "selection": {
                    "candidate": config["selection_lock"]["candidate"],
                    "epoch": config["selection_lock"]["epoch"],
                    "changed_after_validation": False,
                },
                "metrics_percent": {
                    split: {
                        key: None if value is None else value * 100.0
                        for key, value in result["metrics"].items()
                        if key
                        in (
                            "accuracy",
                            "macro_f1",
                            "fall_f1",
                            "all_action_fall_auprc",
                            "conditional_fall_vs_lie_auprc",
                        )
                    }
                    for split, result in results.items()
                },
                "current_minus_historical_percentage_points": {
                    split: result[
                        "current_minus_historical_percentage_points"
                    ]
                    for split, result in results.items()
                },
                "integrity": report["integrity"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )
    if args.require_valid and not integrity_passed:
        raise SystemExit(2)
    return report


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
    parser.add_argument(
        "--training-run-dir", type=Path, default=DEFAULT_TRAINING_RUN_DIR
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-valid", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
