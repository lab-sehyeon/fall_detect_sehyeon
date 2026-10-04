"""Preregistered FU ZS0/ZS1 reconstruction, never the missing original code."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from fall_pipeline.common.integrity import sha256_file, hash_named_tensors
from fall_pipeline.common.metrics import precision_recall_f1
from fall_pipeline.fu.fall_fu_linear_eval import load_arrays, resize_whole_clips
from fall_pipeline.safer.f1_core import ResidualTemporalAdapter, dense_backbone_features, candidate_input, require
from fall_pipeline.safer.run_f1_reconstruction import (
    ROOT, CODE_FILES, config_load, assert_contract, backbone, setup_device,
    read, save_json, log, initialize,
)

CONFIG = ROOT / "configs/fu_zs_document_reconstruction_v1.json"
# Set after registration, before any inference.
CONFIG_HASH = "a8416070bf4a4ae76267148db7a22aee89074901e986bf6151df694127fd48be"
PROFILES = ("zs0_resize64", "zs1_native30", "zs1_aligned25")


def starts_for(length):
    require(length >= 1, "empty clip")
    if length <= 64:
        return [0]
    starts = list(range(0, length - 63, 8))
    if starts[-1] != length - 64:
        starts.append(length - 64)
    return starts


def aligned_indices(length):
    require(length >= 1, "empty clip")
    count = (length - 1) * 5 // 6 + 1
    return (12 * np.arange(count, dtype=np.int64) + 5) // 10


def prepare(data, frames, profile):
    require(profile in PROFILES, "unknown profile")
    windows, owners, starts, lengths = [], [], [], []
    for i, length in enumerate(frames):
        clip = np.array(data[i, :, :int(length)], copy=True)
        if profile == "zs0_resize64":
            clip = resize_whole_clips(data[i:i + 1], frames[i:i + 1])[0].numpy()
        elif profile == "zs1_aligned25":
            clip = clip[:, aligned_indices(int(length))]
        length = clip.shape[1]
        lengths.append(length)
        for start in starts_for(length):
            window = clip[:, start:start + 64]
            if window.shape[1] < 64:
                window = np.concatenate((window, np.repeat(window[:, -1:], 64 - window.shape[1], axis=1)), axis=1)
            windows.append(window)
            owners.append(i)
            starts.append(start)
    return np.stack(windows).astype(np.float32), {
        "owners": np.asarray(owners, dtype=np.int64), "starts": np.asarray(starts, dtype=np.int64),
        "lengths": np.asarray(lengths, dtype=np.int64),
    }


def aggregate(logits, plan, independent=False):
    require(logits.shape == (len(plan["starts"]), 64, 4) and np.isfinite(logits).all(), "invalid window logits")
    offsets = np.r_[0, np.cumsum(plan["lengths"])].astype(np.int64)
    sums = np.zeros((offsets[-1], 4), dtype=np.float64)
    coverage = np.zeros(offsets[-1], dtype=np.int64)
    if independent:
        indices = offsets[plan["owners"]][:, None] + plan["starts"][:, None] + np.arange(64)
        valid = plan["starts"][:, None] + np.arange(64) < plan["lengths"][plan["owners"]][:, None]
        indices = indices[valid]
        coverage[:] = np.bincount(indices, minlength=len(coverage))
        for k in range(4):
            sums[:, k] = np.bincount(indices, weights=logits[:, :, k][valid], minlength=len(coverage))
    else:
        for values, owner, start in zip(logits, plan["owners"], plan["starts"]):
            width = min(64, plan["lengths"][owner] - start)
            lo = offsets[owner] + start
            sums[lo:lo + width] += values[:width]
            coverage[lo:lo + width] += 1
    require(np.all(coverage > 0), "uncovered real frames")
    mean = (sums / coverage[:, None]).astype(np.float32)
    return mean, coverage, offsets


def clip_outputs(timeline, offsets, independent=False):
    if independent:
        probability = torch.softmax(torch.from_numpy(timeline).double(), dim=1).numpy()
    else:
        stable = timeline.astype(np.float64) - timeline.max(1, keepdims=True)
        exp = np.exp(stable)
        probability = exp / exp.sum(1, keepdims=True)
    decisions, scores = [], []
    for begin, end in zip(offsets[:-1], offsets[1:]):
        decisions.append(bool(np.any(timeline[begin:end].argmax(1) == 1)))
        scores.append(float(probability[begin:end, 1].max()))
    return np.asarray(decisions, dtype=bool), np.asarray(scores, dtype=np.float64)


def metrics(labels, actions, decisions, scores):
    truth = labels.astype(bool)
    subset = np.isin(actions, [4, 5])
    result = precision_recall_f1(int(np.sum(truth & decisions)), int(np.sum(~truth & decisions)), int(np.sum(truth & ~decisions)))
    result.update({"clips": len(labels), "accuracy": float(np.mean(truth == decisions)),
                   "auprc": float(average_precision_score(truth, scores)) if truth.any() else None,
                   "auroc": float(roc_auc_score(truth, scores)) if len(np.unique(truth)) == 2 else None,
                   "fall_vs_lying_auprc": float(average_precision_score(truth[subset], scores[subset])) if truth[subset].any() else None,
                   "lying_fp": int(np.sum(decisions & (actions == 4))), "lying_clips": int(np.sum(actions == 4)),
                   "positive_predictions": int(decisions.sum())})
    return result


def load_contract(config, smoke):
    require(sha256_file(CONFIG) == CONFIG_HASH, "preregistered ZS config changed")
    f1_config = config_load()
    f1_root = ROOT / config["f1_run"]
    assert_contract(f1_root, f1_config)
    final = read(f1_root / "final_report.json")
    audit = read(f1_root / "independent_audit.json")
    require(final["status"] == "completed" and audit["passed"], "F1 incomplete")
    require(sha256_file(f1_root / "final_report.json") == audit["report_sha256"], "F1 report changed")
    lock = final["selection_lock"]
    require(lock == read(f1_root / "selection_lock.json"), "F1 lock changed")
    require(lock["candidate"] == config["f1_candidate"] and lock["epoch"] == config["f1_epoch"], "F1 lineage mismatch")
    head_path = f1_root / "training" / config["f1_candidate"] / "best.pt"
    require(sha256_file(head_path) == config["f1_head_sha256"] == lock["head_sha256"], "F1 selected head changed")
    fu = ROOT / config["fu_root"]
    require(sha256_file(fu / "preprocess_manifest.json") == config["fu_manifest_sha256"], "FU manifest changed")
    manifest = read(fu / "preprocess_manifest.json")
    for name, expected in manifest["output_hashes_sha256"].items():
        require(Path(name).name == name and sha256_file(fu / name) == expected, "FU payload changed")
    files = (*CODE_FILES, "fall_pipeline/fu/zs_reconstruction.py")
    contract = {"config_sha256": CONFIG_HASH, "code_sha256": {f: sha256_file(ROOT / f) for f in files},
                "smoke_only": smoke, "input": config, "historical_exact_reproduction": False}
    return contract, f1_config, head_path


def evaluate(config, root, contract, f1_config, head_path):
    device = setup_device(f1_config)
    data, frames, labels, actions, _, _ = load_arrays(ROOT / config["fu_root"])
    if contract["smoke_only"]:
        chosen = np.unique([0, int(frames.argmin()), int(frames.argmax()), int(np.flatnonzero(actions == 4)[0]), int(np.flatnonzero(actions == 5)[0])])
        data, frames, labels, actions = data[chosen], frames[chosen], labels[chosen], actions[chosen]
    np.save(root / "labels.npy", labels)
    np.save(root / "actions.npy", actions)
    model = backbone(f1_config, device)
    checkpoint = torch.load(head_path, map_location="cpu", weights_only=True)
    head = ResidualTemporalAdapter(2048).eval().to(device)
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    head.requires_grad_(False)
    before = {"backbone_adl": hash_named_tensors(model.state_dict().items()), "f1_head": hash_named_tensors(head.state_dict().items())}
    for profile in PROFILES:
        dest = root / profile
        dest.mkdir(exist_ok=True)
        windows, plan = prepare(data, frames, profile)
        np.savez(dest / "plan.npz", **plan)
        progress = dest / "progress.json"
        position = read(progress)["next_index"] if progress.exists() else 0
        require(0 <= position <= len(windows), "resume index outside input")
        path = dest / "window_logits.npy"
        require(progress.exists() or not path.exists(), "orphan logits; preserve and inspect")
        logits = np.lib.format.open_memmap(path, mode="r+" if progress.exists() else "w+", dtype=np.float32, shape=(len(windows), 64, 4))
        require(logits.shape == (len(windows), 64, 4) and logits.dtype == np.float32, "cached logits shape")
        save_json(progress, {"next_index": position, "target": len(windows)})
        save_json(root / "status.json", {"status": "in_progress", "profile": profile})
        with torch.inference_mode():
            for begin in range(position, len(windows), config["execution"]["batch_size"]):
                end = min(begin + config["execution"]["batch_size"], len(windows))
                batch = torch.from_numpy(windows[begin:end]).to(device)
                temporal, context = dense_backbone_features(model, batch)
                values = head(candidate_input(config["f1_candidate"], temporal, context))
                require(bool(torch.isfinite(values).all()), "nonfinite ZS logits")
                logits[begin:end] = values.cpu().numpy()
                if (end // config["execution"]["batch_size"]) % config["execution"]["flush_batches"] == 0 or end == len(windows):
                    logits.flush()
                    save_json(progress, {"next_index": end, "target": len(windows)})
                    log(stage="fu_zs", profile=profile, windows=end, target=len(windows), smoke=contract["smoke_only"])
        mean, coverage, offsets = aggregate(logits, plan)
        decisions, scores = clip_outputs(mean, offsets)
        for name, value in {"timeline_logits": mean, "coverage": coverage, "offsets": offsets, "decisions": decisions, "scores": scores}.items():
            np.save(dest / f"{name}.npy", value)
        result = {"profile": profile, "windows": len(windows), "frames": len(mean), "metrics": metrics(labels, actions, decisions, scores),
                  "files": {p.name: sha256_file(p) for p in dest.iterdir() if p.suffix in (".npy", ".npz")}}
        save_json(dest / "result.json", result)
        log(stage="fu_zs_profile_complete", profile=profile, metrics=result["metrics"], smoke=contract["smoke_only"])
        del windows, logits
    after = {"backbone_adl": hash_named_tensors(model.state_dict().items()), "f1_head": hash_named_tensors(head.state_dict().items())}
    require(before == after, "frozen model mutation")
    require(load_contract(config, contract["smoke_only"])[0] == contract, "inputs/code changed while running")
    save_json(root / "invariance.json", {"passed": True, "before": before, "after": after})


def audit_run(root):
    contract = read(root / "run_contract.json")
    config = read(CONFIG)
    require(load_contract(config, contract["smoke_only"])[0] == contract, "audit contract mismatch")
    require(read(root / "invariance.json")["passed"], "frozen gate missing")
    data, frames, expected_labels, expected_actions, _, _ = load_arrays(ROOT / config["fu_root"])
    if contract["smoke_only"]:
        chosen = np.unique([0, int(frames.argmin()), int(frames.argmax()), int(np.flatnonzero(expected_actions == 4)[0]), int(np.flatnonzero(expected_actions == 5)[0])])
        data, frames, expected_labels, expected_actions = data[chosen], frames[chosen], expected_labels[chosen], expected_actions[chosen]
    labels, actions = (np.load(root / f"{name}.npy", allow_pickle=False) for name in ("labels", "actions"))
    require(np.array_equal(labels, expected_labels) and np.array_equal(actions, expected_actions), "saved labels/order mismatch")
    results = {}
    for profile in PROFILES:
        dest = root / profile
        result = read(dest / "result.json")
        for name, expected in result["files"].items():
            require(Path(name).name == name and sha256_file(dest / name) == expected, "ZS output hash changed")
        regenerated, expected_plan = prepare(data, frames, profile)
        del regenerated
        with np.load(dest / "plan.npz", allow_pickle=False) as saved:
            plan = {key: saved[key] for key in ("owners", "starts", "lengths")}
        require(all(np.array_equal(plan[k], expected_plan[k]) for k in plan), "sampling plan mismatch")
        logits = np.load(dest / "window_logits.npy", allow_pickle=False)
        require(read(dest / "progress.json")["next_index"] == len(logits) == result["windows"], "incomplete profile")
        mean, coverage, offsets = aggregate(logits, plan, independent=True)
        require(len(mean) == result["frames"], "frame count")
        for name, value in {"timeline_logits": mean, "coverage": coverage, "offsets": offsets}.items():
            require(np.array_equal(value, np.load(dest / f"{name}.npy", allow_pickle=False)), "independent overlap mismatch")
        decisions, scores = clip_outputs(mean, offsets, independent=True)
        require(np.array_equal(decisions, np.load(dest / "decisions.npy", allow_pickle=False)), "independent decision mismatch")
        require(np.allclose(scores, np.load(dest / "scores.npy", allow_pickle=False), rtol=0, atol=1e-12), "independent score mismatch")
        recomputed = metrics(labels, actions, decisions, scores)
        require(all(value == result["metrics"][k] or (value is not None and abs(value - result["metrics"][k]) < 1e-12) for k, value in recomputed.items()), "independent metrics mismatch")
        require(abs(f1_score(labels, decisions, zero_division=0) - recomputed["f1"]) < 1e-12, "sklearn F1 mismatch")
        results[profile] = result
    return {"passed": True, "smoke_only": contract["smoke_only"], "results": results, "independent_overlap_exact": True,
            "independent_scores_atol": 1e-12, "frozen_models_unchanged": True, "historical_exact_reproduction": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "run", "audit"), default="run")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = read(CONFIG)
    smoke = args.stage == "smoke"
    root = ROOT / (config["output_dir"] + ("_SMOKE" if smoke else ""))
    require(root.parent == ROOT / "checkpoint/fall", "output scope")
    torch.set_num_threads(2)
    with (root.parent / (root.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "audit":
            report = audit_run(root)
            print({"passed": report["passed"], "profiles": list(report["results"])}, flush=True)
            return
        contract, f1_config, head_path = load_contract(config, smoke)
        require(shutil.disk_usage(ROOT).free > (config["execution"]["reserve_gib"] + 1) * 1024**3, "disk reserve")
        initialize(root, contract, args.resume)
        if (root / "final_report.json").exists():
            audit = audit_run(root)
            log(stage="fu_zs_existing_completed_audit", passed=audit["passed"])
            return
        try:
            evaluate(config, root, contract, f1_config, head_path)
            audit = audit_run(root)
            save_json(root / "independent_audit.json", audit)
            save_json(root / "final_report.json", {**audit, "status": "completed", "research_usable": not smoke, "config_sha256": CONFIG_HASH})
            save_json(root / "status.json", {"status": "completed", "smoke_only": smoke})
            log(stage="fu_zs_complete", smoke=smoke, passed=True)
        except BaseException as error:
            save_json(root / "status.json", {"status": "paused", "error": f"{type(error).__name__}: {error}"})
            raise


if __name__ == "__main__":
    main()
