"""Matched D0/D1/D2 development probes on frozen FU window features."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import numpy as np
import torch
from torch import nn

from fall_pipeline.common.integrity import sha256_file, hash_named_tensors
from fall_pipeline.fu.fall_fu_linear_eval import load_arrays, class_weights
from fall_pipeline.fu import zs_reconstruction as zs
from fall_pipeline.safer.f1_core import ResidualTemporalAdapter, require
from fall_pipeline.safer.run_f1_reconstruction import ROOT, read, save_json, save_torch, log, initialize, setup_device, backbone

CONFIG = ROOT / "configs/fu_probe_document_reconstruction_v1.json"
CONFIG_HASH = "946d48f10c31dbcb514a320c66dcbee131071c767e9b53acabc9fd4a917b115c"


def pooled_parts(temporal, spatial, hidden, widths):
    ts, ss, hs = [], [], []
    for i, width in enumerate(widths):
        width = int(width)
        require(1 <= width <= 64, "valid frame count")
        ts.append(temporal[i, :width].amax(0))
        ss.append(spatial[i].amax(0))
        hs.append(hidden[i, :width].mean(0))
    return {"t": torch.stack(ts), "s": torch.stack(ss), "h": torch.stack(hs)}


def clip_pool(values, owners, count):
    require(np.isfinite(values).all() and len(values) == len(owners), "invalid features")
    require(np.all((owners >= 0) & (owners < count)), "owner bounds")
    output = np.zeros((count, values.shape[1]), dtype=np.float64)
    np.add.at(output, owners, values)
    coverage = np.bincount(owners, minlength=count)
    require(np.all(coverage > 0), "missing clip features")
    return (output / coverage[:, None]).astype(np.float32)


def representations(parts):
    return {"d0_temporal": parts["t"], "d0_temporal_spatial": np.concatenate((parts["t"], parts["s"]), 1),
            "d1_hidden": parts["h"], "d2_concat": np.concatenate((parts["t"], parts["s"], parts["h"]), 1)}


def get_contract(config, smoke):
    require(sha256_file(CONFIG) == CONFIG_HASH, "probe config changed")
    require(sha256_file(zs.CONFIG) == config["zs_config_sha256"], "ZS protocol changed")
    source, f1_config, head_path = zs.load_contract(read(zs.CONFIG), False)
    return {"config_sha256": CONFIG_HASH, "source": source, "smoke_only": smoke,
            "code_sha256": {"fall_pipeline/fu/probe_reconstruction.py": sha256_file(Path(__file__))}}, f1_config, head_path


def extract(config, root, f1_config, head_path, device):
    data, frames, _, _, _, _ = load_arrays(ROOT / read(zs.CONFIG)["fu_root"])
    model = backbone(f1_config, device)
    head = ResidualTemporalAdapter(2048).to(device).eval()
    head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True)["state_dict"], strict=True)
    head.requires_grad_(False)
    before = [hash_named_tensors(m.state_dict().items()) for m in (model, head)]
    reports = {}
    for profile in config["profiles"]:
        base = root / profile
        base.mkdir(exist_ok=True)
        cache = base / "features.npz"
        if (base / "cache_report.json").exists():
            reports[profile] = read(base / "cache_report.json")
            require(sha256_file(cache) == reports[profile]["sha256"], "feature cache changed")
            continue
        windows, plan = zs.prepare(data, frames, profile)
        parts = {"t": [], "s": [], "h": []}
        with torch.inference_mode():
            for start in range(0, len(windows), config["execution"]["feature_batch_size"]):
                end = min(start + config["execution"]["feature_batch_size"], len(windows))
                batch = torch.from_numpy(windows[start:end]).to(device)
                n = len(batch)
                jt = batch.permute(0, 2, 4, 3, 1).reshape(n, 64, 150)
                js = batch.permute(0, 4, 3, 2, 1).reshape(n, 50, 192)
                temporal, spatial = model.backbone(jt, js)
                mask = js.ne(0).any(-1).to(spatial.dtype).unsqueeze(-1)
                context = (spatial * mask).sum(1) / mask.sum(1).clamp_min(1)
                hidden = head.hidden(torch.cat((temporal, context[:, None].expand(-1, 64, -1)), -1))
                widths = np.minimum(64, plan["lengths"][plan["owners"][start:end]] - plan["starts"][start:end])
                for key, value in pooled_parts(temporal, spatial, hidden, widths).items():
                    require(bool(torch.isfinite(value).all()), "nonfinite feature")
                    parts[key].append(value.cpu().numpy())
        arrays = {"window_" + k: np.concatenate(v) for k, v in parts.items()}
        for k in parts:
            arrays[k] = clip_pool(arrays["window_" + k], plan["owners"], len(frames))
        temporary = cache.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays, **plan)
        os.replace(temporary, cache)
        reports[profile] = {"sha256": sha256_file(cache), "windows": len(windows), "clips": len(frames)}
        save_json(base / "cache_report.json", reports[profile])
        log(stage="fu_probe_features", profile=profile, **reports[profile])
    after = [hash_named_tensors(m.state_dict().items()) for m in (model, head)]
    require(before == after, "feature extraction changed frozen model")
    save_json(root / "invariance.json", {"passed": True, "before": before, "after": after})
    del model, head
    torch.cuda.empty_cache()


def binary_metrics(labels, actions, logits):
    scores = torch.softmax(torch.from_numpy(logits).double(), 1).numpy()[:, 1]
    return zs.metrics(labels, actions, logits.argmax(1).astype(bool), scores)


def selection_key(row):
    return tuple(float(row[k]) for k in ("f1", "auprc", "accuracy"))


def fit_fold(features, labels, actions, folds, subjects, fold, config, dest, device):
    settings = config["training"]
    train, val = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    require(not set(subjects[train]) & set(subjects[val]), "subject leakage")
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "result.json").exists():
        return read(dest / "result.json")
    torch.manual_seed(settings["seed"])
    head = nn.Linear(features.shape[1], 2).to(device)
    nn.init.normal_(head.weight, mean=0, std=.01)
    nn.init.zeros_(head.bias)
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings["lr"], betas=tuple(settings["betas"]), eps=settings["eps"], weight_decay=settings["weight_decay"])
    x = torch.from_numpy(features).to(device)
    y = torch.from_numpy(labels).long().to(device)
    weight = class_weights(y[train], "sqrt_inverse_frequency", device)
    begin_epoch, history, best = 1, [], None
    if (dest / "last.pt").exists():
        saved = torch.load(dest / "last.pt", map_location=device, weights_only=True)
        head.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        begin_epoch, history, best = saved["epoch"] + 1, saved["history"], saved["best"]
    for epoch in range(begin_epoch, settings["epochs"] + 1):
        head.train()
        order = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(settings["seed"] + epoch)).numpy()]
        for start in range(0, len(order), settings["batch_size"]):
            index = order[start:start + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(x[index]), y[index], weight=weight)
            require(bool(torch.isfinite(loss)), "nonfinite probe loss")
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.inference_mode():
            logits = head(x[val]).cpu().numpy()
        measured = binary_metrics(labels[val], actions[val], logits)
        history.append({"epoch": epoch, "metrics": measured})
        if best is None or selection_key(measured) > selection_key(best["metrics"]):
            best = {"epoch": epoch, "metrics": measured, "state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
        save_torch(dest / "last.pt", {"epoch": epoch, "state_dict": head.state_dict(), "optimizer": optimizer.state_dict(), "history": history, "best": best})
    head.load_state_dict(best["state_dict"])
    with torch.inference_mode():
        logits = head(x[val]).cpu().numpy()
    save_torch(dest / "best.pt", best)
    np.save(dest / "val_logits.npy", logits)
    np.save(dest / "val_indices.npy", val)
    save_json(dest / "history.json", history)
    result = {"epoch": best["epoch"], "metrics": best["metrics"], "train_subjects": sorted(set(subjects[train].tolist())), "val_subjects": sorted(set(subjects[val].tolist())),
              "files": {name: sha256_file(dest / name) for name in ("best.pt", "val_logits.npy", "val_indices.npy", "history.json")}}
    save_json(dest / "result.json", result)
    return result


def train(config, root, device, smoke):
    _, _, labels, actions, subjects, folds = load_arrays(ROOT / read(zs.CONFIG)["fu_root"])
    for profile in config["profiles"]:
        with np.load(root / profile / "features.npz", allow_pickle=False) as values:
            features = representations({k: values[k] for k in ("t", "s", "h")})
        for representation, data in features.items():
            require(data.shape == (993, config["representations"][representation]) and np.isfinite(data).all(), "feature shape")
            for fold in (range(1) if smoke else range(5)):
                save_json(root / "status.json", {"status": "in_progress", "profile": profile, "representation": representation, "fold": fold})
                result = fit_fold(data, labels, actions, folds, subjects, fold, config, root / profile / representation / f"fold{fold}", device)
                log(stage="fu_probe_fold", profile=profile, representation=representation, fold=fold, epoch=result["epoch"], metrics=result["metrics"], smoke=smoke)


def audit(config, root, smoke):
    contract, _, _ = get_contract(read(CONFIG), smoke)
    require(read(root / "run_contract.json") == contract, "probe audit contract changed")
    invariant = read(root / "invariance.json")
    require(invariant["passed"] and invariant["before"] == invariant["after"], "frozen invariant")
    source_data, source_frames, labels, actions, subjects, folds = load_arrays(ROOT / read(zs.CONFIG)["fu_root"])
    reports = {}
    for profile in config["profiles"]:
        base = root / profile
        require(sha256_file(base / "features.npz") == read(base / "cache_report.json")["sha256"], "feature hash")
        with np.load(base / "features.npz", allow_pickle=False) as cache:
            parts = {k: cache[k] for k in ("t", "s", "h")}
            owners = cache["owners"]
            regenerated, expected_plan = zs.prepare(source_data, source_frames, profile)
            del regenerated
            require(all(np.array_equal(cache[k], expected_plan[k]) for k in expected_plan), "probe window plan mismatch")
            for k in parts:
                independent = np.stack([cache["window_" + k][owners == i].astype(np.float64).mean(0).astype(np.float32) for i in range(993)])
                require(np.array_equal(parts[k], independent), "independent clip pooling mismatch")
        reports[profile] = {}
        for name, data in representations(parts).items():
            oof = np.zeros((993, 2), np.float32)
            coverage = np.zeros(993, np.int64)
            for fold in (range(1) if smoke else range(5)):
                dest = base / name / f"fold{fold}"
                result = read(dest / "result.json")
                for path, expected in result["files"].items():
                    require(sha256_file(dest / path) == expected, "probe artifact hash")
                history = read(dest / "history.json")
                require([r["epoch"] for r in history] == list(range(1, config["training"]["epochs"] + 1)), "missing epoch")
                best = max(history, key=lambda row: selection_key(row["metrics"]))
                require(best["epoch"] == result["epoch"] and best["metrics"] == result["metrics"], "probe best epoch selection")
                checkpoint = torch.load(dest / "best.pt", map_location="cpu", weights_only=True)
                require(checkpoint["epoch"] == result["epoch"] and checkpoint["metrics"] == result["metrics"], "head identity")
                indices = np.load(dest / "val_indices.npy", allow_pickle=False)
                require(np.array_equal(indices, np.flatnonzero(folds == fold)), "fold indices")
                require(not set(subjects[indices]) & set(subjects[folds != fold]), "subject overlap")
                logits = np.load(dest / "val_logits.npy", allow_pickle=False)
                # Float64 NumPy matmul independently verifies saved GPU linear output.
                state = checkpoint["state_dict"]
                predicted = data[indices].astype(np.float64) @ state["weight"].numpy().astype(np.float64).T + state["bias"].numpy()
                require(np.allclose(logits, predicted, rtol=2e-5, atol=2e-5), "linear checkpoint output mismatch")
                measured = binary_metrics(labels[indices], actions[indices], logits)
                require(measured == result["metrics"], "saved probe metric mismatch")
                require(abs(float(zs.f1_score(labels[indices], logits.argmax(1), zero_division=0)) - measured["f1"]) < 1e-12, "independent F1")
                oof[indices], coverage[indices] = logits, coverage[indices] + 1
            selected = coverage == 1
            require(np.all(coverage <= 1) and (smoke or selected.all()), "OOF coverage")
            measured = binary_metrics(labels[selected], actions[selected], oof[selected])
            np.save(base / name / "oof_logits.npy", oof)
            reports[profile][name] = {"metrics": measured, "clips": int(selected.sum()), "folds": 1 if smoke else 5}
    return {"passed": True, "results": reports, "smoke_only": smoke, "claim": "development validation-OOF, not nested", "historical_exact_reproduction": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "run", "audit"), default="run")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    original = read(CONFIG)
    smoke = args.stage == "smoke"
    contract, f1_config, head_path = get_contract(original, smoke)
    config = read(CONFIG)
    if smoke:
        config["training"]["epochs"] = 2
    root = ROOT / (config["output_dir"] + ("_SMOKE" if smoke else ""))
    require(root.parent == ROOT / "checkpoint/fall", "probe output scope")
    torch.set_num_threads(2)
    with (root.parent / (root.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "audit":
            print(audit(config, root, False))
            return
        require(shutil.disk_usage(ROOT).free > (config["execution"]["reserve_gib"] + 1) * 1024**3, "probe disk gate")
        initialize(root, contract, args.resume)
        if (root / "final_report.json").exists():
            log(stage="probe_existing_complete", passed=audit(config, root, smoke)["passed"])
            return
        try:
            device = setup_device(f1_config)
            extract(config, root, f1_config, head_path, device)
            train(config, root, device, smoke)
            report = audit(config, root, smoke)
            save_json(root / "independent_audit.json", report)
            save_json(root / "final_report.json", {**report, "status": "completed", "research_usable": not smoke})
            save_json(root / "status.json", {"status": "completed", "smoke_only": smoke})
            log(stage="fu_probe_completed", smoke=smoke, passed=True)
        except BaseException as error:
            save_json(root / "status.json", {"status": "paused", "error": f"{type(error).__name__}: {error}"})
            raise


if __name__ == "__main__":
    main()
