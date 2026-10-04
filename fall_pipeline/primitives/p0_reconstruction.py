"""Preregistered P0 matched probes with subject-cluster uncertainty."""
from __future__ import annotations
import argparse
import fcntl
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score

from fall_pipeline.primitives.body_signals import body_signals, descriptor, fold_standardize
from fall_pipeline.fu import probe_reconstruction as probe
from fall_pipeline.fu import zs_reconstruction as zs
from fall_pipeline.fu.fall_fu_linear_eval import load_arrays
from fall_pipeline.safer.run_f1_reconstruction import ROOT, read, save_json, log, initialize, setup_device, disk_gate
from fall_pipeline.safer.f1_core import require
from fall_pipeline.common.integrity import sha256_file

CONFIG = ROOT / "configs/primitive_document_reconstruction_v1.json"
FILES = ["fall_pipeline/primitives/body_signals.py", "fall_pipeline/primitives/p0_reconstruction.py"]


def contract(smoke):
    config = read(CONFIG)
    require(sha256_file(probe.CONFIG) == config["probe_config_sha256"], "probe config lineage")
    previous, f1_config, _ = probe.get_contract(read(probe.CONFIG), False)
    require(read(ROOT / read(probe.CONFIG)["output_dir"] / "run_contract.json") == previous, "previous probe lineage")
    require(sha256_file(ROOT / config["p0"]["baseline_cache"]) == config["p0"]["baseline_cache_sha256"], "TS cache hash")
    return {"config_sha256": sha256_file(CONFIG), "source": previous, "code_sha256": {p: sha256_file(ROOT / p) for p in FILES}, "smoke": smoke}, config, f1_config


def build_features(config):
    data, frames, labels, actions, subjects, folds = load_arrays(ROOT / read(zs.CONFIG)["fu_root"])
    primitive = np.stack([descriptor(body_signals(np.asarray(data[i, :, :int(n), :, 0]).transpose(1, 2, 0), 30)) for i, n in enumerate(frames)])
    with np.load(ROOT / config["p0"]["baseline_cache"], allow_pickle=False) as cache:
        ts = np.concatenate((cache["t"], cache["s"]), 1)
    return ts, primitive, labels, actions, subjects, folds


def features_for(ts, primitives, folds, fold):
    scaled, mean, scale = fold_standardize(primitives, np.flatnonzero(folds != fold))
    return {"d0_ts": ts, "primitive": scaled, "d0_ts_primitive": np.concatenate((ts, scaled), 1)}, mean, scale


def bootstrap(labels, subjects, folds, baseline, hybrid, repeats=10000):
    # Paired cluster draws within each fold retain all five fold strata.
    unique = np.unique(subjects)
    counts = np.zeros((len(unique), 2, 3), np.int64)
    subject_fold = []
    for i, subject in enumerate(unique):
        mask = subjects == subject
        subject_fold.append(int(folds[mask][0]))
        for j, prediction in enumerate((baseline, hybrid)):
            y, p = labels[mask].astype(bool), prediction[mask].astype(bool)
            counts[i, j] = [(y & p).sum(), (~y & p).sum(), (y & ~p).sum()]
    rng = np.random.default_rng(0)
    delta = np.zeros(repeats)
    for fold in range(5):
        group = np.flatnonzero(np.asarray(subject_fold) == fold)
        draws = rng.choice(group, size=(repeats, len(group)), replace=True)
        totals = counts[draws].sum(1)
        tp, fp, fn = totals[..., 0], totals[..., 1], totals[..., 2]
        denominator = 2 * tp + fp + fn
        f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)
        delta += (f1[:, 1] - f1[:, 0]) / 5
    interval = np.quantile(delta, [.025, .975], method="linear").tolist()
    return {"repeats": repeats, "seed": 0, "mean_fold_f1_delta_ci95": interval, "excludes_zero": interval[0] > 0 or interval[1] < 0}, delta


def audit(root, config, initial, smoke):
    current, _, _ = contract(smoke)
    require(current == initial == read(root / "run_contract.json"), "P0 source/config changed")
    ts, primitive, labels, actions, subjects, folds = build_features(config)
    np.testing.assert_array_equal(np.load(root / "descriptors.npy", allow_pickle=False), primitive)
    reports, predictions = {}, {}
    for name in config["p0"]["representations"]:
        logits_oof = np.zeros((len(labels), 2), np.float32)
        seen = np.zeros(len(labels), np.int64)
        fold_metrics = []
        for fold in range(1 if smoke else 5):
            features, mean, scale = features_for(ts, primitive, folds, fold)
            dest = root / name / f"fold{fold}"
            with np.load(dest / "scaler.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["mean"], mean)
                np.testing.assert_array_equal(saved["scale"], scale)
            result, history = read(dest / "result.json"), read(dest / "history.json")
            for path, digest in result["files"].items():
                require(sha256_file(dest / path) == digest, "P0 artifact changed")
            require([r["epoch"] for r in history] == list(range(1, config["training"]["epochs"] + 1)), "missing epoch")
            best = max(history, key=lambda r: probe.selection_key(r["metrics"]))
            require(best["epoch"] == result["epoch"] and best["metrics"] == result["metrics"], "best selection")
            idx = np.load(dest / "val_indices.npy", allow_pickle=False)
            require(np.array_equal(idx, np.flatnonzero(folds == fold)), "fold selection")
            require(not set(subjects[idx]) & set(subjects[folds != fold]), "subject leakage")
            logits = np.load(dest / "val_logits.npy", allow_pickle=False)
            state = torch.load(dest / "best.pt", map_location="cpu", weights_only=True)["state_dict"]
            independent = features[name][idx].astype(np.float64) @ state["weight"].numpy().astype(np.float64).T + state["bias"].numpy()
            require(np.allclose(logits, independent, rtol=2e-5, atol=2e-5), "independent linear output")
            metrics = probe.binary_metrics(labels[idx], actions[idx], logits)
            require(metrics == result["metrics"], "metric mismatch")
            require(abs(f1_score(labels[idx], logits.argmax(1)) - metrics["f1"]) < 1e-12, "independent F1")
            if name == "d0_ts" and not smoke:
                previous = ROOT / read(probe.CONFIG)["output_dir"] / "zs1_native30/d0_temporal_spatial" / f"fold{fold}/val_logits.npy"
                require(np.array_equal(logits, np.load(previous, allow_pickle=False)), "matched baseline not exact")
            logits_oof[idx] = logits
            seen[idx] += 1
            fold_metrics.append(metrics)
        selected = seen == 1
        require((seen <= 1).all() and (smoke or selected.all()), "OOF coverage")
        np.save(root / name / "oof_logits.npy", logits_oof)
        reports[name] = {"metrics": probe.binary_metrics(labels[selected], actions[selected], logits_oof[selected]), "mean_fold_f1": float(np.mean([m["f1"] for m in fold_metrics])), "fold_metrics": fold_metrics}
        predictions[name] = logits_oof.argmax(1)
    uncertainty, gate = None, None
    if not smoke:
        uncertainty, draws = bootstrap(labels, subjects, folds, predictions["d0_ts"], predictions["d0_ts_primitive"])
        np.save(root / "bootstrap_deltas.npy", draws)
        a, b = reports["d0_ts"], reports["d0_ts_primitive"]
        # Metric key names inherited from ZS/probe.
        am, bm = a["metrics"], b["metrics"]
        gate = {"mean_fold_f1_improves": b["mean_fold_f1"] > a["mean_fold_f1"],
                "lying_fp_decreases": bm["lying_fp"] < am["lying_fp"],
                "fall_tp_not_lost": bm["tp"] >= am["tp"]}
        gate["passed"] = all(gate.values())
    return {"passed": True, "smoke_only": smoke, "historical_exact_reproduction": False, "results": reports, "point_gate": gate, "bootstrap": uncertainty, "matched_baseline_exact": not smoke, "claim": config["p0"]["claim"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "run", "audit"), default="run")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    smoke = args.stage == "smoke"
    initial, config, f1_config = contract(smoke)
    root = ROOT / config["output_dir"] / ("p0_smoke" if smoke else "p0")
    disk_gate(f1_config)
    initialize(root, initial, args.resume or args.stage == "audit")
    if smoke:
        config["training"]["epochs"] = 2
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage != "audit" and not (root / "final_report.json").exists():
            device = setup_device(f1_config)
            torch.cuda.set_per_process_memory_fraction(config["execution"]["gpu_memory_fraction"], device)
            ts, primitive, labels, actions, subjects, folds = build_features(config)
            np.save(root / "descriptors.npy", primitive)
            for fold in range(1 if smoke else 5):
                features, mean, scale = features_for(ts, primitive, folds, fold)
                for name, x in features.items():
                    dest = root / name / f"fold{fold}"
                    dest.mkdir(parents=True, exist_ok=True)
                    np.savez(dest / "scaler.npz", mean=mean, scale=scale)
                    save_json(root / "status.json", {"stage": "training", "fold": fold, "representation": name})
                    result = probe.fit_fold(x, labels, actions, folds, subjects, fold, config, dest, device)
                    log(stage="p0_fold", fold=fold, representation=name, epoch=result["epoch"], metrics=result["metrics"])
        report = audit(root, config, initial, smoke)
        save_json(root / "final_report.json", report)
        save_json(root / "status.json", {"stage": "completed", "smoke_only": smoke})
        log(stage="p0_complete", **report)


if __name__ == "__main__":
    main()
