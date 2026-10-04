#!/usr/bin/env python3
"""Validation-only F0A fingerprinting for experimental SAFER V1 recovery.

The original project-specific H36M17-as-COCO17 mapper was lost.  This command
compares explicit compatibility candidates against the preserved historical
F0A validation metrics.  Candidate selection is permitted only on the complete
validation split.  It neither accepts nor opens the SAFER test/OOD pickle.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

from data_gen.inspect_safer_activities import load_pinned_pickle, normalized_name
from data_gen.preflight_safer_legacy_v1 import (
    DEFAULT_NORMAL_PICKLE,
    DEFAULT_OUTPUT as DEFAULT_PREFLIGHT,
    EXPECTED_SPLITS,
    VALIDATION_SUBJECTS,
    WINDOW_SIZE,
    subject_id,
    window_starts,
)
from data_gen.safer_legacy_v1_candidates import (
    CANDIDATES,
    Candidate,
    convert_window,
    sequence_context,
)
from fall_pipeline.common.integrity import hash_named_tensors, sha256_file
from fall_pipeline.fu.fall_fu_linear_eval import build_encoder, enforce_physical_gpu_zero


FALL_COARSE_LABEL = 10
LIE_COARSE_LABELS = (11, 12)
NTU_FALL_INDEX = 42  # A043, zero-based class index.
CENTER_OFFSET = 32
EXPECTED_VALIDATION_WINDOWS = EXPECTED_SPLITS["val"]["clean_windows"]
HISTORICAL_VALIDATION_PERCENT = {
    "f1": 7.989,
    "auprc": 8.249,
    "fall_vs_lie_auprc": 61.446,
}
HISTORICAL_ADL_HEAD_EPOCH = 150
MAX_ACCEPTABLE_METRIC_ERROR_PP = 1.0
DEFAULT_ADL_HEAD = Path(
    "checkpoint/adl_baseline/"
    "ntu60_xsub_joint_dste_official_umurl_seed0_gpu0/best_adl_head.pth"
)
DEFAULT_ENCODER = Path("checkpoint/ntu60/xsub/ntu60_xs_joint_dste.pth.tar")


@dataclass(frozen=True)
class WindowRef:
    annotation_index: int
    start: int


def json_dump(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def validate_preflight(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("passed"):
        raise RuntimeError(f"legacy V1 count preflight did not pass: {path}")
    val = report.get("splits", {}).get("val", {})
    if val.get("clean_windows") != EXPECTED_VALIDATION_WINDOWS:
        raise RuntimeError("preflight validation window count mismatch")
    if set(val.get("subjects", ())) != set(VALIDATION_SUBJECTS):
        raise RuntimeError("preflight validation subject set mismatch")
    return report


def build_validation_index(normal: dict[str, Any]) -> list[WindowRef]:
    """Build only the subject-disjoint validation index in annotation order."""
    official_train = {
        normalized_name(value) for value in normal["split"]["sub_train"]
    }
    official_test = {
        normalized_name(value) for value in normal["split"]["sub_test"]
    }
    if official_train & official_test:
        raise RuntimeError("official SAFER subject train/test membership overlaps")

    refs: list[WindowRef] = []
    validation_sequences = 0
    for annotation_index, annotation in enumerate(normal["annotations"]):
        name = normalized_name(annotation["frame_dir"])
        if name not in official_train and name not in official_test:
            raise RuntimeError(f"annotation absent from official subject split: {name}")
        if name not in official_train or subject_id(name) not in VALIDATION_SUBJECTS:
            continue
        validation_sequences += 1
        frames = int(annotation["total_frames"])
        pose = np.asarray(annotation["keypoint_3d"])
        labels = np.asarray(annotation["labels"])
        if pose.shape != (1, frames, 17, 3) or labels.shape != (frames,):
            raise RuntimeError(f"invalid validation annotation shape: {name}")
        finite = np.isfinite(pose[0]).all(axis=(1, 2))
        prefix = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(~finite, dtype=np.int64))
        )
        starts = window_starts(frames)
        clean = starts[(prefix[starts + WINDOW_SIZE] - prefix[starts]) == 0]
        refs.extend(WindowRef(annotation_index, int(start)) for start in clean)

    if validation_sequences != EXPECTED_SPLITS["val"]["sequences"]:
        raise RuntimeError(
            f"expected 73 validation sequences, got {validation_sequences}"
        )
    if len(refs) != EXPECTED_VALIDATION_WINDOWS:
        raise RuntimeError(
            f"expected {EXPECTED_VALIDATION_WINDOWS} clean validation windows, "
            f"got {len(refs)}"
        )
    return refs


class CandidateDataset(Dataset):
    def __init__(
        self,
        annotations: Sequence[dict[str, Any]],
        refs: Sequence[WindowRef],
        candidate: Candidate,
    ) -> None:
        self.annotations = annotations
        self.refs = refs
        self.candidate = candidate
        self._contexts: dict[int, dict[str, np.ndarray | float]] = {}

    def __len__(self) -> int:
        return len(self.refs)

    def _context(self, annotation_index: int) -> dict[str, np.ndarray | float]:
        if annotation_index not in self._contexts:
            pose = np.asarray(
                self.annotations[annotation_index]["keypoint_3d"]
            )[0]
            self._contexts[annotation_index] = sequence_context(pose, self.candidate)
        return self._contexts[annotation_index]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        ref = self.refs[index]
        annotation = self.annotations[ref.annotation_index]
        pose = np.asarray(annotation["keypoint_3d"])[
            0, ref.start : ref.start + WINDOW_SIZE
        ]
        converted = convert_window(
            pose, self.candidate, self._context(ref.annotation_index)
        )
        data = np.zeros((3, WINDOW_SIZE, 25, 2), dtype=np.float32)
        data[:, :, :, 0] = converted.transpose(2, 0, 1)
        label = int(np.asarray(annotation["labels"])[ref.start + CENTER_OFFSET])
        return torch.from_numpy(data), label


def _safe_average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    return float(average_precision_score(labels, scores))


def f0a_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float32)
    if logits.shape != (labels.size, 60):
        raise ValueError(f"expected logits [N,60], got {logits.shape}")
    if not np.isfinite(logits).all():
        raise ValueError("logits contain non-finite values")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp[:, NTU_FALL_INDEX] / exp.sum(axis=1)
    truth = labels == FALL_COARSE_LABEL
    prediction = logits.argmax(axis=1) == NTU_FALL_INDEX
    tp = int(np.count_nonzero(truth & prediction))
    fp = int(np.count_nonzero(~truth & prediction))
    fn = int(np.count_nonzero(truth & ~prediction))
    tn = int(np.count_nonzero(~truth & ~prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    conditional = np.isin(labels, (FALL_COARSE_LABEL, *LIE_COARSE_LABELS))
    return {
        "samples": int(labels.size),
        "fall_samples": int(np.count_nonzero(truth)),
        "predicted_fall_samples": int(np.count_nonzero(prediction)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auprc": _safe_average_precision(truth.astype(np.int64), probabilities),
        "fall_vs_lie_samples": int(np.count_nonzero(conditional)),
        "fall_vs_lie_auprc": _safe_average_precision(
            truth[conditional].astype(np.int64), probabilities[conditional]
        ),
    }


def historical_distance(metrics: dict[str, Any]) -> dict[str, Any]:
    deltas: dict[str, float | None] = {}
    for name, target in HISTORICAL_VALIDATION_PERCENT.items():
        value = metrics.get(name)
        deltas[name] = None if value is None else abs(float(value) * 100.0 - target)
    available = [value for value in deltas.values() if value is not None]
    return {
        "absolute_percentage_point_error": deltas,
        "sum_absolute_percentage_point_error": (
            float(sum(available)) if len(available) == len(deltas) else None
        ),
    }


def candidate_decision(
    results: Sequence[dict[str, Any]],
    full_validation: bool,
    adl_head_epoch: int | None,
) -> dict[str, Any]:
    ranked = sorted(
        results,
        key=lambda item: (
            float("inf")
            if item["historical_distance"]["sum_absolute_percentage_point_error"] is None
            else item["historical_distance"]["sum_absolute_percentage_point_error"],
            item["candidate"]["name"],
        ),
    )
    closest = ranked[0] if full_validation and ranked else None
    closest_errors = (
        closest["historical_distance"]["absolute_percentage_point_error"]
        if closest is not None
        else {}
    )
    errors_available = bool(closest_errors) and all(
        value is not None for value in closest_errors.values()
    )
    max_error = (
        max(float(value) for value in closest_errors.values())
        if errors_available
        else None
    )
    epoch_matches = adl_head_epoch == HISTORICAL_ADL_HEAD_EPOCH
    passed = bool(
        full_validation
        and epoch_matches
        and max_error is not None
        and max_error <= MAX_ACCEPTABLE_METRIC_ERROR_PP
    )
    return {
        "closest_candidate": closest["candidate"]["name"] if closest else None,
        "selected_candidate": closest["candidate"]["name"] if passed else None,
        "selection_gate": {
            "passed": passed,
            "full_validation": full_validation,
            "historical_adl_head_epoch_required": HISTORICAL_ADL_HEAD_EPOCH,
            "evaluated_adl_head_epoch": adl_head_epoch,
            "adl_head_epoch_matches": epoch_matches,
            "maximum_metric_error_pp_allowed": MAX_ACCEPTABLE_METRIC_ERROR_PP,
            "closest_candidate_maximum_metric_error_pp": max_error,
            "metric_error_gate_passed": bool(
                max_error is not None and max_error <= MAX_ACCEPTABLE_METRIC_ERROR_PP
            ),
        },
    }


def load_adl_model(
    encoder_path: Path, head_path: Path, device: torch.device
) -> torch.nn.Module:
    if not encoder_path.is_file():
        raise FileNotFoundError(encoder_path)
    if not head_path.is_file():
        raise FileNotFoundError(head_path)
    model = build_encoder(encoder_path, device)
    head = torch.load(head_path, map_location="cpu", weights_only=True)
    if set(head) != {"weight", "bias"}:
        raise RuntimeError(f"unexpected ADL head keys: {sorted(head)}")
    if tuple(head["weight"].shape) != (60, 2048) or tuple(head["bias"].shape) != (60,):
        raise RuntimeError("unexpected ADL head tensor shape")
    model.fc.load_state_dict(head, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


def infer_candidate(
    model: torch.nn.Module,
    annotations: Sequence[dict[str, Any]],
    refs: Sequence[WindowRef],
    candidate: Candidate,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = CandidateDataset(annotations, refs, candidate)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch_index, (batch, labels) in enumerate(loader, start=1):
            batch = batch.to(device, non_blocking=True)
            jt = batch.permute(0, 2, 4, 3, 1).reshape(batch.shape[0], 64, 150)
            js = batch.permute(0, 4, 3, 2, 1).reshape(batch.shape[0], 50, 192)
            temporal, spatial = model.backbone(jt, js)
            features = torch.cat((temporal.amax(1), spatial.amax(1)), dim=1)
            all_logits.append(model.fc(features).cpu())
            all_labels.append(labels.cpu())
            if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(loader):
                print(
                    f"candidate={candidate.name} batch={batch_index}/{len(loader)}",
                    flush=True,
                )
    return torch.cat(all_labels).numpy(), torch.cat(all_logits).numpy()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1 or args.sample_stride < 1 or args.max_windows < 0:
        raise ValueError("batch-size/sample-stride must be positive; max-windows >= 0")
    unknown = sorted(set(args.candidates) - set(CANDIDATES))
    if unknown:
        raise ValueError(f"unknown candidates: {unknown}")
    if len(set(args.candidates)) != len(args.candidates):
        raise ValueError("candidate names must be unique")

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    preflight_path = args.preflight.resolve()
    normal_path = args.normal_pickle.resolve()
    encoder_path = args.encoder.resolve()
    head_path = args.adl_head.resolve()
    adl_run_config_path = args.adl_run_config.resolve()
    validate_preflight(preflight_path)
    if not adl_run_config_path.is_file():
        raise FileNotFoundError(adl_run_config_path)
    adl_run_config = json.loads(adl_run_config_path.read_text(encoding="utf-8"))
    adl_head_epoch = adl_run_config.get("best_epoch")
    if adl_head_epoch is not None:
        adl_head_epoch = int(adl_head_epoch)
    normal = load_pinned_pickle(normal_path, "normal")
    full_refs = build_validation_index(normal)
    refs = full_refs[:: args.sample_stride]
    if args.max_windows:
        refs = refs[: args.max_windows]
    full_validation = (
        args.sample_stride == 1
        and args.max_windows == 0
        and len(refs) == EXPECTED_VALIDATION_WINDOWS
    )

    device = enforce_physical_gpu_zero()
    model = load_adl_model(encoder_path, head_path, device)
    state_hash_before = hash_named_tensors(model.state_dict().items())
    results = []
    reference_labels: np.ndarray | None = None
    for name in args.candidates:
        candidate = CANDIDATES[name]
        print(json.dumps(candidate.manifest(), indent=2), flush=True)
        labels, logits = infer_candidate(
            model, normal["annotations"], refs, candidate, device, args.batch_size
        )
        if reference_labels is None:
            reference_labels = labels
            np.save(output / "validation_coarse_labels.npy", labels)
        elif not np.array_equal(labels, reference_labels):
            raise RuntimeError("candidate label/order mismatch")
        np.save(output / f"{name}_validation_logits.npy", logits)
        metrics = f0a_metrics(labels, logits)
        result = {
            "candidate": candidate.manifest(),
            "metrics": metrics,
            "historical_validation_percent": HISTORICAL_VALIDATION_PERCENT,
            "historical_distance": historical_distance(metrics),
        }
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    state_hash_after = hash_named_tensors(model.state_dict().items())
    if state_hash_before != state_hash_after:
        raise RuntimeError("frozen ADL model changed during candidate evaluation")

    decision = candidate_decision(results, full_validation, adl_head_epoch)

    report = {
        "experiment": "SAFER legacy V1 experimental compatibility recovery",
        "output_dir": str(output),
        "recovery_status": {
            "original_mapper_source_recovered": False,
            "experimental_reconstruction": True,
            "must_not_be_called_byte_identical_reproduction": True,
        },
        "selection_protocol": {
            "selection_data": "subject-disjoint validation only",
            "test_or_ood_pickle_accepted_or_opened": False,
            "full_validation_required_for_selection": True,
            "historical_fingerprint_percent": HISTORICAL_VALIDATION_PERCENT,
            "ranking": "minimum sum absolute percentage-point error; name tie-break",
        },
        "input": {
            "normal_pickle": str(normal_path),
            "normal_pickle_sha256": sha256_file(normal_path),
            "preflight": str(preflight_path),
            "preflight_sha256": sha256_file(preflight_path),
            "encoder": str(encoder_path),
            "encoder_sha256": sha256_file(encoder_path),
            "adl_head": str(head_path),
            "adl_head_sha256": sha256_file(head_path),
            "adl_run_config": str(adl_run_config_path),
            "adl_run_config_sha256": sha256_file(adl_run_config_path),
            "adl_head_best_epoch": adl_head_epoch,
        },
        "protocol": {
            "window_size": WINDOW_SIZE,
            "stride": 8,
            "center_offset": CENTER_OFFSET,
            "fall_coarse_label": FALL_COARSE_LABEL,
            "lie_coarse_labels": list(LIE_COARSE_LABELS),
            "ntu_fall_class": "A043",
            "ntu_fall_zero_based_index": NTU_FALL_INDEX,
            "sample_stride": args.sample_stride,
            "max_windows": args.max_windows,
            "full_validation_windows": len(full_refs),
            "evaluated_windows": len(refs),
            "full_validation": full_validation,
        },
        "cuda": {
            "physical_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_device": str(device),
            "device_name": torch.cuda.get_device_name(0),
        },
        "model_state_hash_before": state_hash_before,
        "model_state_hash_after": state_hash_after,
        "model_frozen_exact": state_hash_before == state_hash_after,
        "results": results,
        **decision,
        "integrity": {
            "passed": state_hash_before == state_hash_after,
            "physical_gpu_zero_only": os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
            "validation_only": True,
            "test_ood_unread": True,
            "labels_and_order_equal_across_candidates": True,
        },
    }
    json_dump(output / "candidate_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-pickle", type=Path, default=DEFAULT_NORMAL_PICKLE)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--adl-head", type=Path, default=DEFAULT_ADL_HEAD)
    parser.add_argument(
        "--adl-run-config",
        type=Path,
        default=DEFAULT_ADL_HEAD.parent / "run_config.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidates", nargs="+", default=list(CANDIDATES), choices=sorted(CANDIDATES)
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--sample-stride", type=int, default=1,
        help="validation index subsampling for smoke/screening; 1 is required to select",
    )
    parser.add_argument(
        "--max-windows", type=int, default=0,
        help="optional smoke cap; 0 means no cap and is required to select",
    )
    return parser.parse_args()


if __name__ == "__main__":
    final_report = run(parse_args())
    print(json.dumps({
        "output_dir": final_report["output_dir"],
        "full_validation": final_report["protocol"]["full_validation"],
        "selected_candidate": final_report["selected_candidate"],
        "integrity": final_report["integrity"],
    }, indent=2), flush=True)
