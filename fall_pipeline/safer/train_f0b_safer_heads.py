#!/usr/bin/env python3
"""Train the preregistered reconstructed SAFER F0B linear-head control.

The historical F0B optimizer details were lost.  This trainer therefore
requires an explicit recovery configuration that combines the preserved F0B
representation/selection contract with the official FoundSkelModel linear
evaluation optimizer convention.  It must not be reported as a byte-identical
or optimizer-identical reproduction of the lost F0B experiment.

Only the completed frozen train/validation cache is accepted.  Temporal-only
and temporal+spatial heads receive the same shuffled minibatches.  Test/OOD
paths are neither accepted nor opened; they remain locked until the validation
selection manifest has been frozen.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from fall_pipeline.common.integrity import hash_named_tensors, sha256_file
from fall_pipeline.fu.fall_fu_linear_eval import enforce_physical_gpu_zero
from fall_pipeline.safer.extract_f0b_safer_features import (
    FEATURE_DIM,
    FEATURE_FILENAMES,
    FINAL_MANIFEST_NAME as FEATURE_CACHE_MANIFEST_NAME,
    METADATA_FILENAMES,
    SPLIT_MANIFEST_NAME as FEATURE_SPLIT_MANIFEST_NAME,
)


CLASS_NAMES = ("other", "fall", "lie_down", "lying_down")
CANDIDATE_ORDER = ("temporal_only", "temporal_spatial")
DEFAULT_CACHE_DIR = Path(
    "checkpoint/fall/F0B_SAFER_V1_RECONSTRUCTED_20260904_R1/feature_cache"
)
DEFAULT_CONFIG = Path("configs/f0b_safer_v1_recovery_sgd_v1.json")
DEFAULT_OUTPUT_DIR = Path(
    "checkpoint/fall/F0B_SAFER_V1_RECOVERY_SGD_20260904_R1"
)
RUN_STATE_NAME = "training_state.json"
LATEST_CHECKPOINT_NAME = "latest_training_state.pth"
FINAL_REPORT_NAME = "f0b_head_report.json"


EXPECTED_CONFIG = {
    "schema_version": 1,
    "status": "approved_explicit_recovery_configuration",
    "experiment_id": "F0B_SAFER_V1_RECOVERY_SGD_R1",
    "original_f0b_optimizer_config_recovered": False,
    "configuration_kind": "preregistered_recovery_control",
    "must_not_be_called_historical_exact_reproduction": True,
    "candidates": [
        {
            "name": "temporal_only",
            "feature_names": ["temporal"],
            "input_dim": 1024,
            "num_classes": 4,
        },
        {
            "name": "temporal_spatial",
            "feature_names": ["temporal", "spatial"],
            "input_dim": 2048,
            "num_classes": 4,
        },
    ],
    "optimization": {
        "epochs": 50,
        "batch_size": 512,
        "optimizer": "SGD",
        "learning_rate": 0.006,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "lr_schedule_milestones": [120, 140],
        "lr_schedule_gamma": 0.1,
        "loss": "CrossEntropyLoss",
        "class_weight": None,
        "weight_initialization": {
            "distribution": "normal",
            "mean": 0.0,
            "std": 0.01,
        },
        "bias_initialization": 0.0,
        "seed": 0,
        "train_shuffle": True,
        "same_minibatch_order_for_candidates": True,
        "drop_last": False,
    },
    "evaluation": {
        "every_epochs": 1,
        "batch_size": 512,
        "primary_selection_metric": "validation_macro_f1_4class",
        "secondary_selection_metric": (
            "validation_conditional_fall_vs_lie_auprc"
        ),
        "conditional_labels": [1, 2, 3],
        "conditional_positive_label": 1,
        "conditional_score": (
            "softmax_over_logits_classes_1_2_3_probability_of_class_1"
        ),
        "epoch_exact_tie": "earliest_epoch",
        "candidate_exact_tie": "first_preregistered_candidate",
        "early_stopping": False,
        "test_ood_evaluation": (
            "forbidden_until_selection_manifest_is_frozen"
        ),
    },
    "execution": {
        "physical_cuda_visible_devices": "0",
        "logical_device": "cuda:0",
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
    },
}


def atomic_json_dump(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def load_recovery_config(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    limits = config.get("claim_boundary", {})
    checks = {
        "schema_version": config.get("schema_version"),
        "status": config.get("status"),
        "experiment_id": config.get("experiment_id"),
        "original_f0b_optimizer_config_recovered": limits.get(
            "original_f0b_optimizer_config_recovered"
        ),
        "configuration_kind": limits.get("configuration_kind"),
        "must_not_be_called_historical_exact_reproduction": limits.get(
            "must_not_be_called_historical_exact_reproduction"
        ),
        "candidates": config.get("candidates"),
        "optimization": config.get("optimization"),
        "evaluation": config.get("evaluation"),
        "execution": config.get("execution"),
    }
    if checks != EXPECTED_CONFIG:
        mismatches = {
            key: {"expected": EXPECTED_CONFIG[key], "actual": checks.get(key)}
            for key in EXPECTED_CONFIG
            if checks.get(key) != EXPECTED_CONFIG[key]
        }
        raise RuntimeError(
            "F0B recovery configuration differs from the preregistered contract: "
            f"{mismatches}"
        )
    source = config.get("input", {})
    if source.get("allowed_splits") != ["train", "val"]:
        raise RuntimeError("F0B config must allow train/validation only")
    if source.get("test_ood_access") is not False:
        raise RuntimeError("F0B config does not lock test/OOD")
    if source.get("class_names") != list(CLASS_NAMES):
        raise RuntimeError("F0B class-name contract mismatch")
    return config, sha256_file(path)


def _verify_file_payloads(
    split_root: Path, manifest: dict[str, Any], expected_count: int
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    expected_shapes = {
        "temporal_features.npy": (expected_count, FEATURE_DIM),
        "spatial_features.npy": (expected_count, FEATURE_DIM),
        **{name: (expected_count,) for name in METADATA_FILENAMES},
    }
    expected_dtypes = {
        "temporal_features.npy": np.dtype(np.float32),
        "spatial_features.npy": np.dtype(np.float32),
        **{name: np.dtype(np.int64) for name in METADATA_FILENAMES},
    }
    payload = manifest.get("payload", {})
    for name, shape in expected_shapes.items():
        path = split_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        array = np.load(path, mmap_mode="r")
        if array.shape != shape or array.dtype != expected_dtypes[name]:
            raise RuntimeError(f"invalid cached {name}: {array.shape}/{array.dtype}")
        entry = payload.get(name, {})
        if path.stat().st_size != entry.get("bytes"):
            raise RuntimeError(f"cached {name} byte-size mismatch")
        if sha256_file(path) != entry.get("sha256"):
            raise RuntimeError(f"cached {name} hash mismatch")
        arrays[name] = array
    return arrays


def _all_finite(array: np.ndarray, chunk: int = 32768) -> bool:
    return all(
        bool(np.isfinite(np.asarray(array[begin : begin + chunk])).all())
        for begin in range(0, len(array), chunk)
    )


def validate_feature_cache(
    cache_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    cache_dir = cache_dir.resolve()
    manifest_path = cache_dir / FEATURE_CACHE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    expected_input = config["input"]
    if manifest_sha256 != expected_input["feature_cache_manifest_sha256"]:
        raise RuntimeError("feature-cache root manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    integrity = manifest.get("integrity", {})
    if manifest.get("status") != "completed" or not manifest.get("research_usable"):
        raise RuntimeError("feature cache is not a completed research-usable cache")
    if not integrity.get("passed") or not integrity.get(
        "full_train_validation_extraction"
    ):
        raise RuntimeError("feature-cache integrity did not pass")
    if not integrity.get("model_frozen_exact"):
        raise RuntimeError("feature cache did not preserve the frozen model")
    if not integrity.get("train_validation_only") or not integrity.get(
        "test_ood_arrays_unopened"
    ):
        raise RuntimeError("feature-cache development split isolation failed")
    contract = manifest.get("run_contract", {})
    for key in (
        "materialization_manifest_sha256",
        "encoder_sha256",
        "adl_head_sha256",
        "extractor_sha256",
    ):
        if contract.get(key) != expected_input[key]:
            raise RuntimeError(f"feature-cache provenance mismatch: {key}")
    if contract.get("splits") != ["train", "val"]:
        raise RuntimeError("feature-cache split contract mismatch")
    if contract.get("test_ood_access") is not False:
        raise RuntimeError("feature cache reports test/OOD access")

    splits: dict[str, dict[str, Any]] = {}
    for split in ("train", "val"):
        expected_count = int(expected_input["split_counts"][split])
        split_root = cache_dir / split
        split_manifest_path = split_root / FEATURE_SPLIT_MANIFEST_NAME
        split_manifest_sha256 = sha256_file(split_manifest_path)
        if split_manifest_sha256 != expected_input[
            "split_feature_manifest_sha256"
        ][split]:
            raise RuntimeError(f"{split} feature manifest hash mismatch")
        root_entry = manifest.get("splits", {}).get(split, {})
        if root_entry.get("feature_manifest_sha256") != split_manifest_sha256:
            raise RuntimeError(f"{split} root/split manifest mismatch")
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        if not split_manifest.get("integrity", {}).get("passed"):
            raise RuntimeError(f"{split} feature payload integrity did not pass")
        if split_manifest.get("written_windows") != expected_count:
            raise RuntimeError(f"{split} feature count mismatch")
        arrays = _verify_file_payloads(split_root, split_manifest, expected_count)
        labels = np.asarray(arrays["center_derived_labels.npy"])
        counts = np.bincount(labels, minlength=4).tolist()
        if counts != expected_input["class_counts"][split]:
            raise RuntimeError(f"{split} cached class-count mismatch")
        if not _all_finite(arrays["temporal_features.npy"]) or not _all_finite(
            arrays["spatial_features.npy"]
        ):
            raise RuntimeError(f"{split} cached features contain non-finite values")
        splits[split] = {
            "root": str(split_root),
            "count": expected_count,
            "manifest_sha256": split_manifest_sha256,
            "arrays": arrays,
        }
    return {
        "root": str(cache_dir),
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "splits": splits,
        "test_ood_access": False,
    }


def initialize_heads(
    config: dict[str, Any], device: torch.device
) -> dict[str, nn.Linear]:
    initialization = config["optimization"]["weight_initialization"]
    heads: dict[str, nn.Linear] = {}
    for candidate in config["candidates"]:
        head = nn.Linear(candidate["input_dim"], candidate["num_classes"])
        nn.init.normal_(
            head.weight,
            mean=initialization["mean"],
            std=initialization["std"],
        )
        nn.init.constant_(
            head.bias, config["optimization"]["bias_initialization"]
        )
        heads[candidate["name"]] = head.to(device)
    return heads


def create_optimizers(
    heads: dict[str, nn.Linear], config: dict[str, Any]
) -> dict[str, torch.optim.Optimizer]:
    optimization = config["optimization"]
    optimizers = {
        name: torch.optim.SGD(
            head.parameters(),
            lr=optimization["learning_rate"],
            momentum=optimization["momentum"],
            weight_decay=optimization["weight_decay"],
        )
        for name, head in heads.items()
    }
    optimized = {
        id(parameter)
        for optimizer in optimizers.values()
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trainable = {
        id(parameter)
        for head in heads.values()
        for parameter in head.parameters()
        if parameter.requires_grad
    }
    if optimized != trainable or len(trainable) != 4:
        raise RuntimeError("optimizer is not restricted to the two linear heads")
    return optimizers


def learning_rate_for_epoch(config: dict[str, Any], epoch_index: int) -> float:
    optimization = config["optimization"]
    decays = sum(
        epoch_index >= milestone
        for milestone in optimization["lr_schedule_milestones"]
    )
    return optimization["learning_rate"] * (
        optimization["lr_schedule_gamma"] ** decays
    )


def set_learning_rate(
    optimizers: dict[str, torch.optim.Optimizer], learning_rate: float
) -> None:
    for optimizer in optimizers.values():
        for group in optimizer.param_groups:
            group["lr"] = learning_rate


def features_for_candidate(
    name: str, temporal: torch.Tensor, spatial: torch.Tensor
) -> torch.Tensor:
    if name == "temporal_only":
        return temporal
    if name == "temporal_spatial":
        return torch.cat((temporal, spatial), dim=1)
    raise ValueError(f"unknown F0B candidate: {name}")


def _safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        return None
    return float(average_precision_score(labels, scores))


def classification_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float32)
    if logits.shape != (len(labels), 4) or not np.isfinite(logits).all():
        raise RuntimeError("invalid validation logits")
    if labels.size and (labels.min() < 0 or labels.max() > 3):
        raise RuntimeError("validation labels must be in 0..3")
    predictions = logits.argmax(axis=1)
    confusion = np.zeros((4, 4), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for index, name in enumerate(CLASS_NAMES):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        support = int(confusion[index, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        per_class[name] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    logits_tensor = torch.from_numpy(logits)
    probabilities = torch.softmax(logits_tensor, dim=1).numpy()
    fall_labels = (labels == 1).astype(np.int64)
    conditional_mask = np.isin(labels, [1, 2, 3])
    conditional_labels = (labels[conditional_mask] == 1).astype(np.int64)
    conditional_scores = torch.softmax(
        logits_tensor[torch.from_numpy(conditional_mask), 1:4], dim=1
    )[:, 0].numpy()
    return {
        "samples": len(labels),
        "accuracy": float((predictions == labels).mean()) if len(labels) else 0.0,
        "macro_f1": float(np.mean(f1_values)),
        "fall_f1": per_class["fall"]["f1"],
        "all_action_fall_auprc": _safe_auprc(fall_labels, probabilities[:, 1]),
        "conditional_fall_vs_lie_auprc": _safe_auprc(
            conditional_labels, conditional_scores
        ),
        "per_class": per_class,
        "confusion": confusion.tolist(),
    }


def selection_key(metrics: dict[str, Any]) -> tuple[float, float]:
    secondary = metrics["conditional_fall_vs_lie_auprc"]
    return (
        float(metrics["macro_f1"]),
        float(secondary) if secondary is not None else float("-inf"),
    )


def selection_key_for_json(metrics: dict[str, Any]) -> list[float | None]:
    """Return the selection evidence without non-standard Infinity literals."""
    secondary = metrics["conditional_fall_vs_lie_auprc"]
    return [
        float(metrics["macro_f1"]),
        float(secondary) if secondary is not None else None,
    ]


def candidate_is_better(
    metrics: dict[str, Any], best_metrics: dict[str, Any] | None
) -> bool:
    return best_metrics is None or selection_key(metrics) > selection_key(best_metrics)


def _batch_from_memmaps(
    split: dict[str, Any], indices: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    arrays = split["arrays"]
    temporal = torch.from_numpy(
        np.array(arrays["temporal_features.npy"][indices], copy=True)
    ).to(device, non_blocking=True)
    spatial = torch.from_numpy(
        np.array(arrays["spatial_features.npy"][indices], copy=True)
    ).to(device, non_blocking=True)
    labels = torch.from_numpy(
        np.array(arrays["center_derived_labels.npy"][indices], copy=True)
    ).long().to(device, non_blocking=True)
    return temporal, spatial, labels


def train_epoch(
    heads: dict[str, nn.Linear],
    optimizers: dict[str, torch.optim.Optimizer],
    split: dict[str, Any],
    permutation: torch.Tensor,
    batch_size: int,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    for head in heads.values():
        head.train()
    totals = {name: 0.0 for name in CANDIDATE_ORDER}
    seen = 0
    limit = len(permutation)
    if max_batches:
        limit = min(limit, max_batches * batch_size)
    total_batches = (limit + batch_size - 1) // batch_size
    for begin in range(0, limit, batch_size):
        index_tensor = permutation[begin : begin + batch_size]
        indices = index_tensor.numpy()
        temporal, spatial, labels = _batch_from_memmaps(split, indices, device)
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        losses: dict[str, torch.Tensor] = {}
        for name in CANDIDATE_ORDER:
            logits = heads[name](features_for_candidate(name, temporal, spatial))
            losses[name] = nn.functional.cross_entropy(logits, labels)
        sum(losses.values()).backward()
        for optimizer in optimizers.values():
            optimizer.step()
        batch_count = len(indices)
        for name, loss in losses.items():
            totals[name] += float(loss.detach()) * batch_count
        seen += batch_count
        batch_number = begin // batch_size + 1
        if batch_number == 1 or batch_number % 100 == 0 or batch_number == total_batches:
            print(
                f"train batch={batch_number}/{total_batches} samples={seen}/{limit}",
                flush=True,
            )
    if seen == 0:
        raise RuntimeError("no F0B training samples were processed")
    return {name: total / seen for name, total in totals.items()}


def evaluate_heads(
    heads: dict[str, nn.Linear],
    split: dict[str, Any],
    batch_size: int,
    device: torch.device,
    max_batches: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], np.ndarray]:
    for head in heads.values():
        head.eval()
    count = split["count"]
    if max_batches:
        count = min(count, max_batches * batch_size)
    logits_parts: dict[str, list[np.ndarray]] = {
        name: [] for name in CANDIDATE_ORDER
    }
    labels_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for begin in range(0, count, batch_size):
            indices = np.arange(begin, min(begin + batch_size, count), dtype=np.int64)
            temporal, spatial, labels = _batch_from_memmaps(split, indices, device)
            labels_parts.append(labels.cpu().numpy())
            for name in CANDIDATE_ORDER:
                logits = heads[name](features_for_candidate(name, temporal, spatial))
                logits_parts[name].append(logits.float().cpu().numpy())
    labels_array = np.concatenate(labels_parts)
    logits_arrays = {
        name: np.concatenate(parts) for name, parts in logits_parts.items()
    }
    metrics = {
        name: classification_metrics(labels_array, logits)
        for name, logits in logits_arrays.items()
    }
    return metrics, logits_arrays, labels_array


def immutable_run_contract(
    args: argparse.Namespace,
    config_sha256: str,
    cache_manifest_sha256: str,
    trainer_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config": str(args.config.resolve()),
        "config_sha256": config_sha256,
        "cache_dir": str(args.cache_dir.resolve()),
        "cache_manifest_sha256": cache_manifest_sha256,
        "trainer_sha256": trainer_sha256,
        "mode": "smoke" if args.smoke else "full",
        "smoke_epochs": 1 if args.smoke else 0,
        "smoke_train_batches": 2 if args.smoke else 0,
        "smoke_val_batches": 2 if args.smoke else 0,
        "test_ood_access": False,
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
                "use --resume only for an identical run contract"
            )
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("run_contract") != contract:
                raise RuntimeError("completed F0B head run contract mismatch")
            if not report.get("integrity", {}).get("passed"):
                raise RuntimeError("completed F0B head report failed integrity")
            return report
        if not state_path.is_file():
            raise RuntimeError("cannot resume F0B head run without state")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("run_contract") != contract:
            raise RuntimeError("incomplete F0B head run contract mismatch")
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(
        state_path,
        {
            "status": "in_progress",
            "research_usable": False,
            "run_contract": contract,
            "completed_epochs": 0,
        },
    )
    return None


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, config_sha256 = load_recovery_config(args.config)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != config["execution"][
        "cublas_workspace_config"
    ]:
        raise RuntimeError("CUBLAS deterministic workspace contract mismatch")
    cache = validate_feature_cache(args.cache_dir, config)
    device = enforce_physical_gpu_zero()
    seed = int(config["optimization"]["seed"])
    seed_everything(seed)
    trainer_sha256 = sha256_file(Path(__file__))
    contract = immutable_run_contract(
        args, config_sha256, cache["manifest_sha256"], trainer_sha256
    )
    preflight = {
        "mode": "preflight_only" if args.preflight_only else contract["mode"],
        "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": str(device),
        "device_name": torch.cuda.get_device_name(0),
        "config_sha256": config_sha256,
        "cache_manifest_sha256": cache["manifest_sha256"],
        "trainer_sha256": trainer_sha256,
        "train_windows": cache["splits"]["train"]["count"],
        "validation_windows": cache["splits"]["val"]["count"],
        "candidates": list(CANDIDATE_ORDER),
        "optimizer": config["optimization"],
        "test_ood_access": False,
    }
    print(
        json.dumps(preflight, ensure_ascii=False, indent=2, allow_nan=False),
        flush=True,
    )
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
                    "selected_candidate": completed.get("selection", {}).get(
                        "candidate"
                    ),
                    "integrity": completed["integrity"],
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            flush=True,
        )
        return completed

    heads = initialize_heads(config, device)
    optimizers = create_optimizers(heads, config)
    initial_head_hashes = {
        name: hash_named_tensors(head.state_dict().items())
        for name, head in heads.items()
    }
    histories: dict[str, list[dict[str, Any]]] = {
        name: [] for name in CANDIDATE_ORDER
    }
    best: dict[str, dict[str, Any] | None] = {
        name: None for name in CANDIDATE_ORDER
    }
    shuffle_generator = torch.Generator().manual_seed(seed)
    start_epoch = 1
    latest_path = output_dir / LATEST_CHECKPOINT_NAME
    if args.resume:
        if not latest_path.is_file():
            raise RuntimeError("cannot resume without latest training checkpoint")
        latest = torch.load(latest_path, map_location=device, weights_only=False)
        if latest.get("run_contract") != contract:
            raise RuntimeError("latest F0B checkpoint contract mismatch")
        for name in CANDIDATE_ORDER:
            heads[name].load_state_dict(latest["heads"][name], strict=True)
            optimizers[name].load_state_dict(latest["optimizers"][name])
        histories = latest["histories"]
        best = latest["best"]
        shuffle_generator.set_state(latest["shuffle_generator_state"].cpu())
        initial_head_hashes = latest["initial_head_hashes"]
        start_epoch = int(latest["completed_epoch"]) + 1

    full_epochs = int(config["optimization"]["epochs"])
    epochs = 1 if args.smoke else full_epochs
    train_max_batches = 2 if args.smoke else 0
    val_max_batches = 2 if args.smoke else 0
    batch_size = int(config["optimization"]["batch_size"])
    eval_batch_size = int(config["evaluation"]["batch_size"])
    train_count = int(cache["splits"]["train"]["count"])
    for epoch in range(start_epoch, epochs + 1):
        epoch_index = epoch - 1
        learning_rate = learning_rate_for_epoch(config, epoch_index)
        set_learning_rate(optimizers, learning_rate)
        permutation = torch.randperm(train_count, generator=shuffle_generator)
        losses = train_epoch(
            heads,
            optimizers,
            cache["splits"]["train"],
            permutation,
            batch_size,
            device,
            train_max_batches,
        )
        metrics, logits, labels = evaluate_heads(
            heads,
            cache["splits"]["val"],
            eval_batch_size,
            device,
            val_max_batches,
        )
        for name in CANDIDATE_ORDER:
            item = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": losses[name],
                **metrics[name],
            }
            histories[name].append(item)
            previous = best[name]
            if candidate_is_better(
                metrics[name], None if previous is None else previous["metrics"]
            ):
                best[name] = {
                    "epoch": epoch,
                    "metrics": metrics[name],
                    "state_dict": _cpu_state_dict(heads[name]),
                    "val_logits": logits[name].copy(),
                    "val_labels": labels.copy(),
                }
            conditional = metrics[name]["conditional_fall_vs_lie_auprc"]
            conditional_text = "NA" if conditional is None else f"{conditional * 100:.3f}"
            print(
                f"candidate={name} epoch={epoch:03d}/{epochs} "
                f"loss={losses[name]:.6f} "
                f"val_macro_f1={metrics[name]['macro_f1'] * 100:.3f} "
                f"val_fall_f1={metrics[name]['fall_f1'] * 100:.3f} "
                f"val_cond_auprc={conditional_text}",
                flush=True,
            )

        atomic_torch_save(
            latest_path,
            {
                "run_contract": contract,
                "completed_epoch": epoch,
                "heads": {
                    name: _cpu_state_dict(heads[name]) for name in CANDIDATE_ORDER
                },
                "optimizers": {
                    name: optimizer.state_dict()
                    for name, optimizer in optimizers.items()
                },
                "histories": histories,
                "best": best,
                "shuffle_generator_state": shuffle_generator.get_state(),
                "initial_head_hashes": initial_head_hashes,
            },
        )
        atomic_json_dump(
            output_dir / RUN_STATE_NAME,
            {
                "status": "in_progress",
                "research_usable": False,
                "run_contract": contract,
                "completed_epochs": epoch,
                "latest_checkpoint_sha256": sha256_file(latest_path),
            },
        )

    if any(value is None for value in best.values()):
        raise RuntimeError("F0B candidate selection state is incomplete")
    selected_name = CANDIDATE_ORDER[0]
    for name in CANDIDATE_ORDER[1:]:
        assert best[name] is not None and best[selected_name] is not None
        if candidate_is_better(best[name]["metrics"], best[selected_name]["metrics"]):
            selected_name = name

    np.save(output_dir / "validation_labels.npy", best[selected_name]["val_labels"])
    candidate_summaries: dict[str, dict[str, Any]] = {}
    for name in CANDIDATE_ORDER:
        assert best[name] is not None
        candidate_dir = output_dir / name
        candidate_dir.mkdir(exist_ok=True)
        atomic_torch_save(candidate_dir / "best_head.pth", best[name]["state_dict"])
        np.save(candidate_dir / "best_val_logits.npy", best[name]["val_logits"])
        atomic_json_dump(candidate_dir / "history.json", histories[name])
        summary = {
            "candidate": name,
            "best_epoch": best[name]["epoch"],
            "best_metrics": best[name]["metrics"],
            "head_state_hash": hash_named_tensors(best[name]["state_dict"].items()),
        }
        atomic_json_dump(candidate_dir / "summary.json", summary)
        candidate_summaries[name] = summary

    assert best[selected_name] is not None
    atomic_torch_save(output_dir / "selected_head.pth", best[selected_name]["state_dict"])
    full_run = not args.smoke and epochs == full_epochs
    checks = {
        "feature_cache_gate_passed": True,
        "physical_gpu_zero_only": os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
        "explicit_recovery_config_frozen": True,
        "only_two_linear_heads_trainable": True,
        "same_minibatch_order_for_candidates": True,
        "unweighted_cross_entropy": True,
        "full_validation_each_epoch": not args.smoke,
        "selection_uses_validation_only": True,
        "test_ood_unopened": True,
        "all_50_epochs_completed": full_run,
    }
    integrity_passed = all(
        value
        for key, value in checks.items()
        if key not in {"full_validation_each_epoch", "all_50_epochs_completed"}
    )
    research_usable = bool(full_run and integrity_passed)
    report = {
        "experiment": "SAFER V1 F0B explicit SGD recovery control",
        "status": "completed" if research_usable else "smoke_completed",
        "research_usable": research_usable,
        "run_contract": contract,
        "claim_boundary": config["claim_boundary"],
        "configuration": config,
        "configuration_sha256": config_sha256,
        "input": {
            "feature_cache_manifest_sha256": cache["manifest_sha256"],
            "train_windows": cache["splits"]["train"]["count"],
            "validation_windows": cache["splits"]["val"]["count"],
            "test_ood_access": False,
        },
        "selection": {
            "candidate": selected_name,
            "epoch": best[selected_name]["epoch"],
            "metrics": best[selected_name]["metrics"],
            "selection_key": selection_key_for_json(
                best[selected_name]["metrics"]
            ),
            "frozen_before_test_ood": research_usable,
        },
        "candidates": candidate_summaries,
        "initial_head_hashes": initial_head_hashes,
        "selected_head_sha256": sha256_file(output_dir / "selected_head.pth"),
        "trainer_sha256": trainer_sha256,
        "environment": {
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "physical_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_device": str(device),
            "device_name": torch.cuda.get_device_name(0),
        },
        "integrity": {"passed": integrity_passed, **checks},
    }
    atomic_json_dump(output_dir / FINAL_REPORT_NAME, report)
    atomic_json_dump(
        output_dir / RUN_STATE_NAME,
        {
            "status": report["status"],
            "research_usable": research_usable,
            "run_contract": contract,
            "completed_epochs": epochs,
            "final_report_sha256": sha256_file(output_dir / FINAL_REPORT_NAME),
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "status": report["status"],
                "research_usable": research_usable,
                "selection": report["selection"],
                "candidate_best": {
                    name: {
                        "epoch": summary["best_epoch"],
                        "macro_f1_percent": (
                            summary["best_metrics"]["macro_f1"] * 100.0
                        ),
                        "fall_f1_percent": (
                            summary["best_metrics"]["fall_f1"] * 100.0
                        ),
                        "conditional_auprc_percent": (
                            None
                            if summary["best_metrics"][
                                "conditional_fall_vs_lie_auprc"
                            ]
                            is None
                            else summary["best_metrics"][
                                "conditional_fall_vs_lie_auprc"
                            ]
                            * 100.0
                        ),
                    }
                    for name, summary in candidate_summaries.items()
                },
                "integrity": report["integrity"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )
    if args.require_valid and not report["integrity"]["passed"]:
        raise SystemExit(2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-valid", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
