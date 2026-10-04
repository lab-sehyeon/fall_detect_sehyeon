#!/usr/bin/env python3
"""Read-only checks of the saved recovery baselines; never train or infer.

Recomputes metrics from existing NTU/FU/SAFER-validation predictions. SAFER
test/OOD input payloads are not opened. Reports JSON to stdout for an audit log.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from fall_pipeline.common.integrity import sha256_file
from fall_pipeline.fu.fall_fu_linear_eval import confusion_metrics
from fall_pipeline.safer.eval_f0b_safer_postselection import (
    DEFAULT_CONFIG,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRAINING_RUN_DIR,
    load_eval_config,
    validate_materialization_manifests,
    validate_selection_run,
)
from fall_pipeline.safer.train_f0b_safer_heads import classification_metrics


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
        raise RuntimeError(f"invalid saved tensor: {path}")
    return value


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def ntu() -> dict:
    base = ROOT / "checkpoint/adl_baseline/ntu60_xsub_joint_dste_official_umurl_seed0_gpu0_epoch150"
    config = read_json(base / "run_config.json")
    logits = tensor(base / "baseline_val_logits.pt")
    labels = tensor(base / "baseline_val_labels.pt").long()
    predictions = tensor(base / "baseline_predictions.pt")
    require(tuple(logits.shape) == (16487, 60), "NTU logits shape")
    require(tuple(labels.shape) == (16487,), "NTU labels shape")
    require(torch.equal(logits.argmax(1), predictions), "NTU prediction mismatch")
    top1 = float((logits.argmax(1) == labels).double().mean()) * 100
    top5 = float((logits.topk(5, dim=1).indices == labels[:, None]).any(1).double().mean()) * 100
    require(abs(top1 - config["best_top1"]) < 1e-4, "NTU Top-1 mismatch")
    require(config["best_epoch"] == 150, "NTU ADL epoch mismatch")
    return {"samples": len(labels), "epoch": 150, "top1_percent": top1,
            "top5_percent": top5, "logits_sha256": sha256_file(base / "baseline_val_logits.pt"),
            "scope": "saved predictions; no new model inference"}


def fu() -> dict:
    base = ROOT / "checkpoint/fall/F0_FU_RECOVERED_20260903"
    processed = ROOT / "data/fall_processed/FU-Kinect-Fall/ntu25_official_umurl_v1"
    config = read_json(base / "run_config.json")
    labels = tensor(base / "oof_labels.pt")
    logits = tensor(base / "oof_logits.pt")
    require(tuple(logits.shape) == (993, 2), "FU logits shape")
    metrics = confusion_metrics(labels, logits)
    require(metrics == config["oof_metrics"], "FU OOF metric mismatch")
    require(sha256_file(processed / "preprocess_manifest.json") ==
            config["preprocess_manifest_sha256"], "FU manifest hash mismatch")
    subjects = np.load(processed / "subjects.npy", allow_pickle=False)
    folds = np.load(processed / "fold_ids.npy", allow_pickle=False)
    require(np.array_equal(folds, (subjects - 1) % 5), "FU fold rule mismatch")
    seen = torch.zeros(993, dtype=torch.int64)
    for fold in range(5):
        fold_dir = base / f"fold{fold}"
        indices = tensor(fold_dir / "val_indices.pt").long()
        require(np.array_equal(indices.numpy(), np.flatnonzero(folds == fold)), "FU fold indices")
        require(torch.equal(tensor(fold_dir / "val_labels.pt"), labels[indices]), "FU fold labels")
        require(torch.equal(tensor(fold_dir / "val_logits.pt"), logits[indices]), "FU fold logits")
        require(not (set(subjects[folds == fold]) & set(subjects[folds != fold])), "FU subject overlap")
        seen[indices] += 1
    require(bool(torch.all(seen == 1)), "FU OOF coverage")
    return {"samples": 993, "metrics": metrics, "folds_verified": 5,
            "claim_boundary": config["recovery_status"]}


def safer() -> dict:
    config, config_hash = load_eval_config(ROOT / DEFAULT_CONFIG)
    manifests = validate_materialization_manifests(ROOT / DEFAULT_DATA_ROOT, config)
    selection = validate_selection_run(ROOT / DEFAULT_TRAINING_RUN_DIR, config)
    base = ROOT / DEFAULT_TRAINING_RUN_DIR
    require(sha256_file(ROOT / "fall_pipeline/safer/train_f0b_safer_heads.py") ==
            config["input"]["f0b_trainer_sha256"], "SAFER trainer hash mismatch")
    require(sha256_file(ROOT / "configs/f0b_safer_v1_recovery_sgd_v1.json") ==
            config["input"]["f0b_training_config_sha256"], "SAFER training config hash mismatch")
    labels = np.load(base / "validation_labels.npy", allow_pickle=False)
    candidates = {}
    for candidate in ("temporal_only", "temporal_spatial"):
        summary = read_json(base / candidate / "summary.json")
        logits = np.load(base / candidate / "best_val_logits.npy", allow_pickle=False)
        metrics = classification_metrics(labels, logits)
        require(metrics == summary["best_metrics"], f"SAFER {candidate} validation metrics")
        candidates[candidate] = {"epoch": summary["best_epoch"], "metrics": metrics}
    f0a = read_json(ROOT / "checkpoint/fall/F0A_SAFER_V1_RECONSTRUCTED_20260904_R1/f0a_report.json")
    require(f0a["status"] == "completed" and f0a["integrity"]["passed"], "SAFER F0A stored report")
    return {"config_sha256": config_hash, "selected": selection["report"]["selection"],
            "validation_candidates": candidates, "f0a_saved_report_status": f0a["status"],
            "test_ood_payload_arrays_opened": manifests["payload_arrays_opened"],
            "postselection_output_exists": (ROOT / DEFAULT_OUTPUT_DIR).exists(),
            "claim_boundary": config["claim_boundary"]}


def main() -> None:
    os.chdir(ROOT)
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "python": sys.version, "interpreter": sys.executable,
              "packages": {name: importlib.metadata.version(name) for name in
                           ("torch", "numpy", "scipy", "scikit-learn", "torchvision")},
              "scope": "saved baseline audit only; no training, downloads or test/OOD inference",
              "checks": {}}
    for name, check in (("ntu_adl", ntu), ("fu_oof", fu), ("safer_f0b", safer)):
        try:
            report["checks"][name] = {"passed": True, "details": check()}
        except Exception as exc:
            report["checks"][name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    report["passed"] = all(item["passed"] for item in report["checks"].values())
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
