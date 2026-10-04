#!/usr/bin/env python3
"""Read-only, CPU-only full audit of completed F0B evaluation artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from fall_pipeline.common.integrity import sha256_file
from fall_pipeline.safer.eval_f0b_safer_postselection import (
    DEFAULT_CONFIG, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR, DEFAULT_TRAINING_RUN_DIR,
    METADATA_FILENAMES, load_eval_config, validate_materialization_manifests,
    validate_selection_run,
)
from fall_pipeline.safer.train_f0b_safer_heads import classification_metrics


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(value, message):
    if not value:
        raise RuntimeError(message)


def payload_hash(path):
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    digest = hashlib.sha256()
    for begin in range(0, len(array), 256):
        digest.update(np.asarray(array[begin:begin + 256]).tobytes(order="C"))
    return digest.hexdigest()


def audit():
    torch.set_num_threads(2)
    os.chdir(ROOT)
    config, config_hash = load_eval_config(ROOT / DEFAULT_CONFIG)
    validate_materialization_manifests(ROOT / DEFAULT_DATA_ROOT, config)
    validate_selection_run(ROOT / DEFAULT_TRAINING_RUN_DIR, config)
    output = ROOT / DEFAULT_OUTPUT_DIR
    report_path = output / "f0b_postselection_report.json"
    report = read(report_path)
    state = read(output / "evaluation_state.json")
    require(report["status"] == state["status"] == "completed", "run incomplete")
    require(report["research_usable"] and all(report["integrity"].values()), "run integrity")
    require(state["final_report_sha256"] == sha256_file(report_path), "report hash")
    require(state["run_contract"] == report["run_contract"], "state/report contract")
    contract = report["run_contract"]
    require(contract["config_sha256"] == config_hash, "config changed")
    for name in ("encoder", "adl_head", "adl_run_config"):
        require(sha256_file(Path(contract[name])) == contract[name + "_sha256"], name)
    for name, filename in (("evaluator", "eval_f0b_safer_postselection.py"),
                           ("metric_code", "train_f0b_safer_heads.py")):
        require(sha256_file(ROOT / "fall_pipeline/safer" / filename) ==
                contract[name + "_sha256"], name)
    require(report["model_state_hash_before"] == report["model_state_hash_after"], "model state")
    require(report["selected_head_state_hash_before"] ==
            report["selected_head_state_hash_after"] ==
            config["selection_lock"]["selected_head_state_hash"], "head state")
    require(state["completed_splits"] == ["test", "ood"], "split completion")
    splits = {}
    for split, count in config["input"]["splits"].items():
        source = ROOT / DEFAULT_DATA_ROOT / split
        target = output / split
        result = read(target / "result.json")
        progress = read(target / "progress.json")
        require(result == report["results"][split], "result/report mismatch")
        require(result["status"] == progress["status"] == "completed", "split incomplete")
        require(result["evaluated_windows"] == progress["next_index"] == count, "count")
        require(progress["result_sha256"] == sha256_file(target / "result.json"), "split hash")
        require(all(result["integrity"].values()), "split integrity")
        for name, spec in result["payload"].items():
            path = target / name
            require(path.stat().st_size == spec["bytes"], f"size: {path}")
            require(sha256_file(path) == spec["sha256"], f"hash: {path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            require(len(array) == count, f"count: {path}")
            require(all(np.isfinite(array[i:i + 256]).all() for i in range(0, count, 256)),
                    f"finite: {path}")
        source_manifest = read(source / "split_manifest.json")
        source_hashes = {}
        # Hash every source array, including dense labels needed by the next F1 stage.
        for name, spec in source_manifest["payload"].items():
            require((source / name).stat().st_size == spec["bytes"], f"source size: {name}")
            digest = payload_hash(source / name)
            require(digest == spec["array_payload_sha256"], f"source payload: {split}/{name}")
            source_hashes[name] = digest
        for name in METADATA_FILENAMES:
            require(np.array_equal(np.load(source / name, allow_pickle=False),
                                   np.load(target / name, allow_pickle=False)), f"order: {name}")
        labels = np.load(target / "center_derived_labels.npy", allow_pickle=False)
        logits = np.load(target / "logits.npy", allow_pickle=False)
        metrics = classification_metrics(labels, logits)
        require(metrics == result["metrics"], "metric recomputation")
        predicted = logits.argmax(axis=1)
        cm = confusion_matrix(labels, predicted, labels=[0, 1, 2, 3])
        require(cm.tolist() == metrics["confusion"], "independent confusion")
        macro = f1_score(labels, predicted, labels=[0, 1, 2, 3], average="macro", zero_division=0)
        fall = f1_score(labels == 1, predicted == 1, zero_division=0)
        require(abs(macro - metrics["macro_f1"]) < 1e-12, "independent macro F1")
        require(abs(fall - metrics["fall_f1"]) < 1e-12, "independent fall F1")
        splits[split] = {"passed": True, "count": count, "metrics": metrics,
                         "source_payload_sha256": source_hashes,
                         "output_payload_files_verified": len(result["payload"])}
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
            "scope": "all output files and source arrays; saved logits only, no inference/tuning",
            "report_sha256": sha256_file(report_path), "config_sha256": config_hash,
            "splits": splits, "model_state_unchanged": True, "head_state_unchanged": True}


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2, allow_nan=False))
