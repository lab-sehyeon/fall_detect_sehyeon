#!/usr/bin/env python3
"""F0A: frozen NTU60 A043 zero-shot evaluation on SAFER clean3d_v1.

This is the next historical control after SAFER materialization.  It performs
no training and fits no threshold: a window is predicted as fall only when the
frozen 60-way NTU classifier's argmax is A043 (zero-based index 42).  The
official SAFER coarse center-frame label 10 is the positive class.

The original evaluator and mapper were lost.  The input is therefore required
to declare itself a record-grounded structural compatibility reconstruction,
not a byte-identical historical reproduction.  Test and OOD evaluation are
allowed only after the materialization manifest proves that preprocessing was
frozen without using those splits for candidate selection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_gen.preflight_safer_legacy_v1 import EXPECTED_SPLITS, WINDOW_SIZE
from data_gen.safer_legacy_v1_gendata import (
    FINAL_MANIFEST_NAME,
    FROZEN_CANDIDATE_NAME,
)
from fall_pipeline.common.eval_safer_legacy_v1_candidates import (
    f0a_metrics,
    load_adl_model,
)
from fall_pipeline.common.integrity import hash_named_tensors, sha256_file
from fall_pipeline.fu.fall_fu_linear_eval import enforce_physical_gpu_zero


SPLIT_ORDER = ("val", "test", "ood")
EXPECTED_COUNTS = {
    split: EXPECTED_SPLITS[split]["clean_windows"] for split in SPLIT_ORDER
}
HISTORICAL_F0A_PERCENT = {
    "val": {"f1": 7.989, "auprc": 8.249, "fall_vs_lie_auprc": 61.446},
    "test": {"f1": 14.640, "auprc": 10.871, "fall_vs_lie_auprc": 64.964},
    "ood": {"f1": 4.834, "auprc": 2.573, "fall_vs_lie_auprc": 59.363},
}
DEFAULT_DATA_ROOT = Path(
    "data/fall_processed/SAFER-Activities/clean3d_v1_reconstructed"
)
DEFAULT_ENCODER = Path("checkpoint/ntu60/xsub/ntu60_xs_joint_dste.pth.tar")
DEFAULT_ADL_ROOT = Path(
    "checkpoint/adl_baseline/"
    "ntu60_xsub_joint_dste_official_umurl_seed0_gpu0_epoch150"
)
RUN_STATE_NAME = "evaluation_state.json"
FINAL_REPORT_NAME = "f0a_report.json"


def atomic_json_dump(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def validate_materialization(data_root: Path) -> dict[str, Any]:
    data_root = data_root.resolve()
    manifest_path = data_root / FINAL_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
        raise RuntimeError("SAFER structural selection basis is missing")
    limits = manifest.get("recovery_status", {})
    if not limits.get("experimental_compatibility_reconstruction"):
        raise RuntimeError("SAFER reconstruction limit is missing")
    if not limits.get("must_not_be_called_byte_identical_reproduction"):
        raise RuntimeError("SAFER byte-identity limitation is missing")
    protocol = manifest.get("protocol", {})
    if protocol.get("window_size") != WINDOW_SIZE:
        raise RuntimeError("SAFER materialization window size mismatch")
    if protocol.get("test_and_ood_used_for_candidate_selection") is not False:
        raise RuntimeError("test/OOD candidate-selection isolation was not proved")
    if protocol.get("test_and_ood_materialized_only_after_structural_freeze") is not True:
        raise RuntimeError("test/OOD post-freeze materialization was not proved")

    split_details: dict[str, dict[str, Any]] = {}
    for split, expected_count in EXPECTED_COUNTS.items():
        root_entry = manifest.get("splits", {}).get(split, {})
        split_root = data_root / split
        split_manifest_path = split_root / "split_manifest.json"
        if not split_manifest_path.is_file():
            raise FileNotFoundError(split_manifest_path)
        actual_manifest_sha = sha256_file(split_manifest_path)
        if root_entry.get("manifest_sha256") != actual_manifest_sha:
            raise RuntimeError(f"{split} manifest hash mismatch")
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        if not split_manifest.get("integrity", {}).get("passed"):
            raise RuntimeError(f"{split} materialized payload did not pass")
        if split_manifest.get("written_windows") != expected_count:
            raise RuntimeError(f"{split} materialized count mismatch")
        if split_manifest.get("candidate", {}).get("name") != FROZEN_CANDIDATE_NAME:
            raise RuntimeError(f"{split} candidate mismatch")

        data_path = split_root / "data_joint.npy"
        labels_path = split_root / "center_coarse_labels.npy"
        data = np.load(data_path, mmap_mode="r")
        labels = np.load(labels_path, mmap_mode="r")
        if data.shape != (expected_count, 3, WINDOW_SIZE, 25, 2):
            raise RuntimeError(f"{split} data shape mismatch: {data.shape}")
        if data.dtype != np.float32:
            raise RuntimeError(f"{split} data dtype mismatch: {data.dtype}")
        if labels.shape != (expected_count,) or labels.dtype != np.int64:
            raise RuntimeError(
                f"{split} center coarse label mismatch: {labels.shape}/{labels.dtype}"
            )
        expected_bytes = split_manifest.get("payload", {}).get(
            "data_joint.npy", {}
        ).get("bytes")
        if data_path.stat().st_size != expected_bytes:
            raise RuntimeError(f"{split} data file byte size mismatch")
        split_details[split] = {
            "count": expected_count,
            "data_path": str(data_path.resolve()),
            "labels_path": str(labels_path.resolve()),
            "split_manifest": str(split_manifest_path.resolve()),
            "split_manifest_sha256": actual_manifest_sha,
            "data_payload_sha256": split_manifest["payload"]["data_joint.npy"][
                "array_payload_sha256"
            ],
        }

    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "splits": split_details,
    }


def validate_adl_run_config(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    args = config.get("args", {})
    expected = {
        "epochs": 150,
        "extra_epochs": 0,
        "finetune_dataset": "ntu60",
        "data_profile": "official_umurl",
        "protocol": "cross_subject",
        "moda": "joint",
        "backbone": "DSTE",
    }
    if any(args.get(key) != value for key, value in expected.items()):
        raise RuntimeError("ADL run config does not match the epoch-150 contract")
    if config.get("best_epoch") != 150:
        raise RuntimeError("F0A requires the separately preserved epoch-150 ADL head")
    top1 = float(config.get("best_top1", -1.0))
    if abs(top1 - 85.2853775024414) > 1e-6:
        raise RuntimeError(f"unexpected epoch-150 ADL Top-1: {top1}")
    return config, sha256_file(path)


def historical_difference(split: str, metrics: dict[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key, historical in HISTORICAL_F0A_PERCENT[split].items():
        current = metrics.get(key)
        result[key] = None if current is None else float(current) * 100.0 - historical
    return result


def forward_adl_logits(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    jt = batch.permute(0, 2, 4, 3, 1).reshape(batch.shape[0], 64, 150)
    js = batch.permute(0, 4, 3, 2, 1).reshape(batch.shape[0], 50, 192)
    temporal, spatial = model.backbone(jt, js)
    features = torch.cat((temporal.amax(1), spatial.amax(1)), dim=1)
    logits = model.fc(features)
    if logits.shape != (batch.shape[0], 60) or not torch.isfinite(logits).all():
        raise RuntimeError(f"invalid ADL logits: {tuple(logits.shape)}")
    return logits


def immutable_run_contract(
    args: argparse.Namespace,
    materialization: dict[str, Any],
    encoder_sha256: str,
    head_sha256: str,
    adl_config_sha256: str,
    evaluator_sha256: str,
    metric_code_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "data_root": str(args.data_root.resolve()),
        "materialization_manifest_sha256": materialization["manifest_sha256"],
        "encoder": str(args.encoder.resolve()),
        "encoder_sha256": encoder_sha256,
        "adl_head": str(args.adl_head.resolve()),
        "adl_head_sha256": head_sha256,
        "adl_run_config": str(args.adl_run_config.resolve()),
        "adl_run_config_sha256": adl_config_sha256,
        "evaluator_sha256": evaluator_sha256,
        "metric_code_sha256": metric_code_sha256,
        "splits": list(args.splits),
        "batch_size": args.batch_size,
        "max_windows_per_split": args.max_windows_per_split,
        "decision": "60-way argmax equals A043/index42; no fitted threshold",
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
                "use --resume only for a matching F0A run"
            )
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("run_contract") != contract:
                raise RuntimeError("completed F0A run contract mismatch")
            if not report.get("integrity", {}).get("passed"):
                raise RuntimeError("completed F0A report failed integrity")
            return report
        if not state_path.is_file():
            raise RuntimeError("cannot resume F0A output without evaluation state")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("run_contract") != contract:
            raise RuntimeError("incomplete F0A run contract mismatch")
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


def evaluate_split(
    model: torch.nn.Module,
    device: torch.device,
    data_root: Path,
    output_dir: Path,
    split: str,
    batch_size: int,
    max_windows: int,
    resume: bool,
) -> dict[str, Any]:
    source_root = data_root / split
    data = np.load(source_root / "data_joint.npy", mmap_mode="r")
    labels_source = np.load(
        source_root / "center_coarse_labels.npy", mmap_mode="r"
    )
    full_count = EXPECTED_COUNTS[split]
    count = full_count if max_windows == 0 else min(max_windows, full_count)
    split_root = output_dir / split
    split_root.mkdir(parents=True, exist_ok=True)
    logits_path = split_root / "logits.npy"
    labels_path = split_root / "coarse_labels.npy"
    progress_path = split_root / "progress.json"
    result_path = split_root / "result.json"

    if result_path.is_file():
        if not resume:
            raise FileExistsError(f"F0A split already exists: {split_root}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("evaluated_windows") != count:
            raise RuntimeError(f"existing {split} F0A count mismatch")
        return result

    continuing = progress_path.is_file()
    if continuing and not resume:
        raise FileExistsError(f"incomplete F0A split requires --resume: {split_root}")
    if continuing:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("split") != split or progress.get("expected_windows") != count:
            raise RuntimeError(f"{split} F0A progress contract mismatch")
        next_index = int(progress["next_index"])
        logits = np.load(logits_path, mmap_mode="r+")
        saved_labels = np.load(labels_path, mmap_mode="r")
        if logits.shape != (count, 60) or logits.dtype != np.float32:
            raise RuntimeError(f"{split} resume logits mismatch")
        if not np.array_equal(saved_labels, labels_source[:count]):
            raise RuntimeError(f"{split} resume labels mismatch")
    else:
        if any(split_root.iterdir()):
            raise FileExistsError(f"refusing to overwrite F0A split: {split_root}")
        logits = np.lib.format.open_memmap(
            logits_path, mode="w+", dtype=np.float32, shape=(count, 60)
        )
        np.save(labels_path, np.asarray(labels_source[:count], dtype=np.int64))
        next_index = 0
        atomic_json_dump(
            progress_path,
            {
                "status": "in_progress",
                "split": split,
                "expected_windows": count,
                "next_index": 0,
            },
        )

    with torch.inference_mode():
        batch_number = next_index // batch_size
        total_batches = (count + batch_size - 1) // batch_size
        for begin in range(next_index, count, batch_size):
            stop = min(begin + batch_size, count)
            batch = torch.from_numpy(np.array(data[begin:stop], copy=True)).to(
                device, non_blocking=True
            )
            logits[begin:stop] = forward_adl_logits(model, batch).cpu().numpy()
            batch_number += 1
            if batch_number == 1 or batch_number % 25 == 0 or stop == count:
                logits.flush()
                atomic_json_dump(
                    progress_path,
                    {
                        "status": "in_progress",
                        "split": split,
                        "expected_windows": count,
                        "next_index": stop,
                    },
                )
                print(
                    f"split={split} batch={batch_number}/{total_batches} "
                    f"windows={stop}/{count}",
                    flush=True,
                )
    logits.flush()
    labels = np.load(labels_path, mmap_mode="r")
    metrics = f0a_metrics(labels, logits)
    result = {
        "split": split,
        "full_split_windows": full_count,
        "evaluated_windows": count,
        "full_split": count == full_count,
        "metrics": metrics,
        "historical_reference_percent": HISTORICAL_F0A_PERCENT[split],
        "current_minus_historical_percentage_points": historical_difference(
            split, metrics
        ),
        "artifacts": {
            "logits": str(logits_path.resolve()),
            "logits_sha256": sha256_file(logits_path),
            "coarse_labels": str(labels_path.resolve()),
            "coarse_labels_sha256": sha256_file(labels_path),
        },
    }
    atomic_json_dump(result_path, result)
    atomic_json_dump(
        progress_path,
        {
            "status": "completed",
            "split": split,
            "expected_windows": count,
            "next_index": count,
            "result_sha256": sha256_file(result_path),
        },
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1 or args.max_windows_per_split < 0:
        raise ValueError("batch size must be positive and max windows non-negative")
    if len(set(args.splits)) != len(args.splits):
        raise ValueError("F0A split names must be unique")

    materialization = validate_materialization(args.data_root)
    _, adl_config_sha256 = validate_adl_run_config(args.adl_run_config)
    if not args.encoder.is_file():
        raise FileNotFoundError(args.encoder)
    if not args.adl_head.is_file():
        raise FileNotFoundError(args.adl_head)
    encoder_sha256 = sha256_file(args.encoder)
    head_sha256 = sha256_file(args.adl_head)
    evaluator_sha256 = sha256_file(Path(__file__).resolve())
    metric_code_sha256 = sha256_file(
        Path(__file__).with_name("eval_safer_legacy_v1_candidates.py")
    )
    contract = immutable_run_contract(
        args,
        materialization,
        encoder_sha256,
        head_sha256,
        adl_config_sha256,
        evaluator_sha256,
        metric_code_sha256,
    )
    output_dir = args.output_dir.resolve()
    completed = initialize_output(output_dir, contract, args.resume)
    if completed is not None:
        print(json.dumps({
            "output_dir": str(output_dir),
            "status": "already_completed",
            "research_usable": completed["research_usable"],
            "integrity": completed["integrity"],
        }, indent=2))
        return completed

    device = enforce_physical_gpu_zero()
    model = load_adl_model(args.encoder.resolve(), args.adl_head.resolve(), device)
    state_hash_before = hash_named_tensors(model.state_dict().items())
    results: dict[str, dict[str, Any]] = {}
    for split in args.splits:
        results[split] = evaluate_split(
            model=model,
            device=device,
            data_root=args.data_root.resolve(),
            output_dir=output_dir,
            split=split,
            batch_size=args.batch_size,
            max_windows=args.max_windows_per_split,
            resume=args.resume,
        )
        state = json.loads((output_dir / RUN_STATE_NAME).read_text(encoding="utf-8"))
        state["completed_splits"] = list(results)
        atomic_json_dump(output_dir / RUN_STATE_NAME, state)
    state_hash_after = hash_named_tensors(model.state_dict().items())

    full_evaluation = (
        tuple(args.splits) == SPLIT_ORDER
        and args.max_windows_per_split == 0
        and all(results[split]["full_split"] for split in SPLIT_ORDER)
    )
    checks = {
        "materialization_gate_passed": True,
        "physical_gpu_zero_only": os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
        "epoch150_adl_contract_passed": True,
        "no_training_or_threshold_fitting": True,
        "model_frozen_exact": state_hash_before == state_hash_after,
        "requested_splits_completed": set(results) == set(args.splits),
        "full_val_test_ood_evaluation": full_evaluation,
    }
    research_usable = bool(full_evaluation and all(checks.values()))
    integrity_passed = all(
        value for key, value in checks.items() if key != "full_val_test_ood_evaluation"
    )
    report = {
        "experiment": "F0A SAFER legacy V1 reconstructed A043 zero-shot",
        "status": "completed" if research_usable else "smoke_completed",
        "research_usable": research_usable,
        "run_contract": contract,
        "recovery_status": materialization["manifest"]["recovery_status"],
        "protocol": {
            "training": False,
            "threshold_fitting": False,
            "prediction": "NTU60 argmax equals A043/index42",
            "positive_label": "official SAFER coarse label 10 at center offset 32",
            "conditional_auprc_labels": [10, 11, 12],
            "test_ood_evaluated_only_after_structural_freeze": True,
            "results_must_not_be_called_historical_exact_reproduction": True,
        },
        "input": {
            "materialization_manifest": materialization["manifest_path"],
            "materialization_manifest_sha256": materialization["manifest_sha256"],
            "split_provenance": materialization["splits"],
            "encoder_sha256": encoder_sha256,
            "adl_head_sha256": head_sha256,
            "adl_run_config_sha256": adl_config_sha256,
            "evaluator_sha256": evaluator_sha256,
            "metric_code_sha256": metric_code_sha256,
        },
        "cuda": {
            "physical_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_device": str(device),
            "device_name": torch.cuda.get_device_name(0),
        },
        "model_state_hash_before": state_hash_before,
        "model_state_hash_after": state_hash_after,
        "results": results,
        "integrity": {"passed": integrity_passed, **checks},
    }
    atomic_json_dump(output_dir / FINAL_REPORT_NAME, report)
    atomic_json_dump(
        output_dir / RUN_STATE_NAME,
        {
            "status": report["status"],
            "research_usable": research_usable,
            "run_contract": contract,
            "completed_splits": list(results),
            "final_report_sha256": sha256_file(output_dir / FINAL_REPORT_NAME),
        },
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "status": report["status"],
        "research_usable": research_usable,
        "metrics_percent": {
            split: {
                key: None if value is None else value * 100.0
                for key, value in result["metrics"].items()
                if key in ("f1", "auprc", "fall_vs_lie_auprc")
            }
            for split, result in results.items()
        },
        "integrity": report["integrity"],
    }, ensure_ascii=False, indent=2), flush=True)
    if args.require_valid and not report["integrity"]["passed"]:
        raise SystemExit(2)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--adl-head", type=Path, default=DEFAULT_ADL_ROOT / "best_adl_head.pth")
    parser.add_argument("--adl-run-config", type=Path, default=DEFAULT_ADL_ROOT / "run_config.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=SPLIT_ORDER, default=list(SPLIT_ORDER)
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--max-windows-per-split",
        type=int,
        default=0,
        help="prefix cap for smoke only; zero evaluates complete splits",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-valid", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
