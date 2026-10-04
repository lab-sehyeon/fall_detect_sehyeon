"""CPU/read-only audit of final F1 artifacts; never chooses new parameters."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from fall_pipeline.common.integrity import sha256_file
from fall_pipeline.safer.f1_core import require, selection_key


def independently_check_f1(timeline, logits, metrics):
    predictions = logits.argmax(1)
    require(confusion_matrix(timeline.labels, predictions, labels=[0, 1, 2, 3]).tolist() == metrics["confusion"], "independent confusion")
    macro = f1_score(timeline.labels, predictions, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    fall = f1_score(timeline.labels == 1, predictions == 1, zero_division=0)
    require(abs(macro - metrics["macro_f1"]) < 1e-12, "independent macro F1")
    require(abs(fall - metrics["fall_f1"]) < 1e-12, "independent fall F1")


def audit(root):
    from fall_pipeline.safer.run_f1_reconstruction import (
        config_load, read, source, timeline_from, assert_contract, CONFIG_HASH,
    )
    config = config_load()
    torch.set_num_threads(2)
    assert_contract(root, config)
    final = read(root / "final_report.json")
    training = read(root / "training/report.json")
    lock = read(root / "selection_lock.json")
    require(final["status"] == training["status"] == "completed", "incomplete experiment")
    require(final["frozen_backbone_adl_exact"] and final["frozen_selected_head_exact"], "invariance gate")
    require(final["selection_lock"] == lock and lock["config_sha256"] == CONFIG_HASH, "lock mismatch")
    require(sha256_file(root / "training/report.json") == lock["training_report_sha256"], "training report hash")
    require(sha256_file(root / "training" / lock["candidate"] / "best.pt") == lock["head_sha256"], "head hash")
    val_timeline = timeline_from(source(config, "val", include_data=False))
    history = [read(root / "training" / f"epoch_{epoch:03d}.json")
               for epoch in range(1, config["training"]["epochs"] + 1)]
    for epoch, row in enumerate(history, 1):
        require(row["epoch"] == epoch and row["windows"] == config["input"]["windows"]["train"], "incomplete epoch")
    candidate_results = {}
    for name in config["candidates"]:
        best_row = max(history, key=lambda row: selection_key(row["validation"][name]))
        recorded = training["candidates"][name]
        require(best_row["epoch"] == recorded["epoch"], "not earliest validation best epoch")
        require(best_row["validation"][name] == recorded["metrics"], "selected validation metrics mismatch")
        logits = np.load(root / "training" / name / "best_val_timeline_logits.npy", allow_pickle=False)
        recomputed = val_timeline.metrics(logits)
        require(recomputed == recorded["metrics"], "saved validation logits mismatch")
        independently_check_f1(val_timeline, logits, recomputed)
        checkpoint = torch.load(root / "training" / name / "best.pt", map_location="cpu", weights_only=True)
        require(checkpoint["epoch"] == recorded["epoch"] and checkpoint["metrics"] == recomputed, "saved checkpoint identity")
        require(checkpoint["candidate"] == name and checkpoint["config_sha256"] == CONFIG_HASH, "saved checkpoint config")
        require(all(bool(torch.isfinite(v).all()) for v in checkpoint["state_dict"].values()), "nonfinite trained state")
        candidate_results[name] = {"epoch": recorded["epoch"], "metrics_recomputed": True,
                                   "checkpoint_sha256": sha256_file(root / "training" / name / "best.pt")}
    best = training["candidates"]
    winner = max(config["candidates"], key=lambda name: (*selection_key(best[name]["metrics"]), -best[name]["epoch"], -config["candidates"].index(name)))
    require(winner == lock["candidate"] and best[winner]["epoch"] == lock["epoch"], "global candidate selection")
    results = {}
    for split in ("test", "ood"):
        base = root / "evaluation" / split
        result = read(base / "result.json")
        require(result == final["results"][split] and result["selection_lock"] == lock, "holdout report mismatch")
        require(result["windows"] == config["input"]["windows"][split], "holdout count")
        for name, spec in result["files"].items():
            require((base / name).stat().st_size == spec["bytes"] and sha256_file(base / name) == spec["sha256"], "holdout file hash/size")
        timeline = timeline_from(source(config, split, include_data=False))
        require(np.array_equal(np.load(base / "frame_indices.npy", allow_pickle=False), np.flatnonzero(timeline.covered)), "covered frame order")
        require(np.array_equal(np.load(base / "coarse_labels.npy", allow_pickle=False), timeline.coarse[timeline.covered]), "timeline label order")
        window_logits = np.load(base / "window_logits.npy", mmap_mode="r", allow_pickle=False)
        require(window_logits.shape == (result["windows"], 64, 4), "window logit shape")
        # Independent overlap sum via bincount, not the evaluator's np.add.at.
        totals = timeline.empty()
        for begin in range(0, len(window_logits), 1024):
            chunk = np.asarray(window_logits[begin:begin + 1024])
            require(np.isfinite(chunk).all(), "nonfinite stored window logits")
            indices = (timeline.window_offsets[begin:begin + len(chunk), None] + np.arange(64)).ravel()
            lo, hi = int(indices.min()), int(indices.max()) + 1
            for label in range(4):
                totals[lo:hi, label] += np.bincount(indices - lo, weights=chunk[:, :, label].ravel(), minlength=hi - lo)
        mean = (totals[timeline.covered] / timeline.counts[timeline.covered, None]).astype(np.float32)
        saved = np.load(base / "timeline_logits.npy", allow_pickle=False)
        require(np.array_equal(mean, saved), "independent raw-logit overlap mean")
        recomputed = timeline.metrics(saved)
        require(recomputed == result["metrics"], "holdout metric recomputation")
        independently_check_f1(timeline, saved, recomputed)
        results[split] = {"passed": True, "windows": result["windows"], "covered_frames": int(timeline.covered.sum()),
                          "overlap_mean_exact": True, "metrics_exact": True, "artifact_hashes_verified": True}
    return {"passed": True, "scope": "read-only CPU audit; no inference/learning/selection changes",
            "config_sha256": CONFIG_HASH, "report_sha256": sha256_file(root / "final_report.json"),
            "epochs_verified": len(history), "candidates": candidate_results, "selection_verified": True, "results": results}


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    print(json.dumps(audit(parser.parse_args().run_dir.resolve()), indent=2, ensure_ascii=False, allow_nan=False))
