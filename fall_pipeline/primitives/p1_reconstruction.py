"""P1 frozen F1 plus zero-init body-relative correction, with epoch0 gate."""
from __future__ import annotations
import argparse
import copy
import fcntl
from pathlib import Path
import numpy as np
import torch
from torch import nn

from fall_pipeline.primitives.body_signals import body_signals
from fall_pipeline.primitives import p0_reconstruction as p0
from fall_pipeline.fu import zs_reconstruction as zs
from fall_pipeline.safer import run_f1_reconstruction as f1
from fall_pipeline.safer.f1_core import ResidualTemporalAdapter, require, selection_key, sqrt_weights
from fall_pipeline.safer.audit_f1_reconstruction import independently_check_f1
from fall_pipeline.common.integrity import sha256_file, hash_named_tensors

ROOT = f1.ROOT
FILES = ["fall_pipeline/primitives/body_signals.py", "fall_pipeline/primitives/p0_reconstruction.py", "fall_pipeline/primitives/p1_reconstruction.py"]


class PrimitiveCorrection(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.projection = nn.Linear(12, 512)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.classifier = copy.deepcopy(classifier).requires_grad_(True)

    def forward(self, hidden, signals):
        return self.classifier(hidden + self.projection(signals))


def subset_indices(count, seed=0):
    return np.sort(torch.randperm(count, generator=torch.Generator().manual_seed(seed)).numpy()[:count // 4])


def signals_batch(windows):
    return np.stack([body_signals(np.asarray(p[:, :, :, 0]).transpose(1, 2, 0), 25) for p in windows])


def contract(smoke):
    inherited, config, f1_config = p0.contract(False)
    old = ROOT / f1_config["execution"]["output_dir"]
    _, _, head_path = zs.load_contract(f1.read(zs.CONFIG), False)
    manifests = {split: sha256_file(old / "cache" / split / "manifest.json") for split in ("train", "val")}
    value = {"config_sha256": sha256_file(p0.CONFIG), "source": inherited, "f1_cache_manifests": manifests,
             "code_sha256": {p: sha256_file(ROOT / p) for p in FILES}, "smoke": smoke}
    return value, config, f1_config, old, head_path


def load_base(path, device):
    base = ResidualTemporalAdapter(2048).to(device).eval().requires_grad_(False)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require(checkpoint["candidate"] == "temporal_spatial__sqrt" and checkpoint["epoch"] == 17, "F1 selection identity")
    base.load_state_dict(checkpoint["state_dict"], strict=True)
    return base


def progress(root, **values):
    f1.save_json(root / "status.json", {"time": f1.now(), **values})
    f1.log(**values)


def cache_split(root, split, indices, source_tuple, base, correction, old, f1_config, device):
    dest = root / "cache" / split
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "manifest.json").exists():
        return load_cache(root, split, len(indices))
    progress(root, stage="verifying_f1_cache", split=split)
    arrays = f1.validate_cache_split(old, split, f1_config["input"]["windows"][split])
    joints = source_tuple[2]["data_joint.npy"]
    require(f1.payload_digest(joints) == source_tuple[1]["payload"]["data_joint.npy"]["array_payload_sha256"], "source skeleton payload changed")
    shapes = {"hidden.npy": (len(indices), 64, 512), "signals.npy": (len(indices), 64, 12)}
    count = f1.read(dest / "progress.json")["next_index"] if (dest / "progress.json").exists() else 0
    require(0 <= count <= len(indices), "cache progress bounds")
    needed = sum(int(np.prod(shape)) * 4 for name, shape in shapes.items() if not (dest / name).exists())
    f1.disk_gate(f1_config, additional=needed)
    saved = {}
    for name, shape in shapes.items():
        path = dest / name
        require(not path.exists() or (dest / "progress.json").exists(), "partial cache without progress")
        saved[name] = np.load(path, mmap_mode="r+", allow_pickle=False) if path.exists() else np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
        require(saved[name].shape == shape and saved[name].dtype == np.float32, "cache shape")
    f1.save_json(dest / "progress.json", {"next_index": count})
    with torch.inference_mode():
        for begin in range(count, len(indices), 128):
            idx = indices[begin:begin + 128]
            temporal, context = f1.feature_batch(arrays, idx, device)
            features = torch.cat((temporal, context[:, None].expand(-1, 64, -1)), -1)
            hidden = base.hidden(features)
            signals = signals_batch(joints[idx])
            if split == "val":
                corrected = correction(hidden, torch.from_numpy(signals).to(device))
                # Independent hidden forward via unmodified F1, before any training.
                original = base(features)
                require(torch.equal(corrected, original), "epoch0 not bit-exact with F1")
            end = begin + len(idx)
            saved["hidden.npy"][begin:end] = hidden.cpu().numpy()
            saved["signals.npy"][begin:end] = signals
            if (begin // 128 + 1) % 25 == 0 or end == len(indices):
                for array in saved.values():
                    array.flush()
                f1.save_json(dest / "progress.json", {"next_index": end})
                progress(root, stage="p1_cache", split=split, windows=end, target=len(indices))
                f1.disk_gate(f1_config)
    np.save(dest / "indices.npy", indices)
    manifest = {"windows": len(indices), "source_joint_payload_verified": True, "f1_cache_sha256_verified": True,
                "epoch0_window_logits_exact": split == "val", "files": {name: sha256_file(dest / name) for name in (*shapes, "indices.npy")}}
    f1.save_json(dest / "manifest.json", manifest)
    return load_cache(root, split, len(indices))


def load_cache(root, split, count):
    dest = root / "cache" / split
    manifest = f1.read(dest / "manifest.json")
    require(manifest["windows"] == count and manifest["source_joint_payload_verified"] and manifest["f1_cache_sha256_verified"], "cache provenance")
    values = {}
    for name, digest in manifest["files"].items():
        require(sha256_file(dest / name) == digest, "P1 cache hash")
        values[name] = np.load(dest / name, mmap_mode="r", allow_pickle=False)
    require(values["hidden.npy"].shape == (count, 64, 512) and values["signals.npy"].shape == (count, 64, 12), "P1 cache dimensions")
    return values


def batch(values, indices, device):
    return tuple(torch.from_numpy(np.array(values[name][indices], copy=True)).to(device) for name in ("hidden.npy", "signals.npy"))


def evaluate(model, cache, timeline, root, epoch, device):
    model.eval()
    totals = timeline.empty()
    dest = root / "validation" / f"epoch{epoch:02d}"
    dest.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(dest / "window_logits.npy", mode="w+", dtype=np.float32, shape=(len(timeline.starts), 64, 4))
    with torch.inference_mode():
        for begin in range(0, len(timeline.starts), 128):
            values = model(*batch(cache, slice(begin, begin + 128), device)).cpu().numpy()
            require(np.isfinite(values).all(), "nonfinite P1 logits")
            output[begin:begin + len(values)] = values
            timeline.add(totals, begin, values)
    output.flush()
    logits = timeline.mean(totals)
    np.save(dest / "timeline_logits.npy", logits)
    metrics = timeline.metrics(logits)
    f1.save_json(dest / "result.json", {"epoch": epoch, "metrics": metrics, "files": {name: sha256_file(dest / name) for name in ("window_logits.npy", "timeline_logits.npy")}})
    return metrics, logits


def train(root, config, f1_config, caches, inputs, base, model, device, initial):
    settings = config["p1"]
    train_indices = subset_indices(f1_config["input"]["windows"]["train"])
    require(np.array_equal(caches["train"]["indices.npy"], train_indices), "quarter subset changed")
    timeline = f1.timeline_from(inputs["val"])
    train_timeline = f1.timeline_from(inputs["train"])
    counts = np.bincount(train_timeline.labels, minlength=4)
    require(counts.tolist() == [4570626, 57983, 182231, 18056], "F1 train weights mismatch")
    weights = sqrt_weights(counts).to(device)
    require(not {s["subject"] for s in inputs["train"][3]} & {s["subject"] for s in inputs["val"][3]}, "subject leakage")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], betas=tuple(settings["betas"]), eps=settings["eps"], weight_decay=settings["weight_decay"])
    last = root / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location=device, weights_only=True)
        require(saved["contract"] == initial, "P1 resume contract")
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        history, best, begin_epoch = saved["history"], saved["best"], saved["epoch"] + 1
    else:
        progress(root, stage="p1_epoch0_validation")
        require(f1.read(root / "cache/val/manifest.json")["epoch0_window_logits_exact"], "epoch0 full-window gate missing")
        metrics, logits = evaluate(model, caches["val"], timeline, root, 0, device)
        original = np.load(ROOT / f1_config["execution"]["output_dir"] / "training/temporal_spatial__sqrt/best_val_timeline_logits.npy", allow_pickle=False)
        require(np.array_equal(logits, original), "epoch0 timeline not exact with recorded F1")
        f1.save_json(root / "epoch0_gate.json", {"passed": True, "windows": len(timeline.starts), "window_logits_exact": True, "timeline_exact_with_saved_f1": True, "metrics": metrics})
        history, best, begin_epoch = [{"epoch": 0, "metrics": metrics}], {"epoch": 0, "metrics": metrics}, 1
        f1.save_torch(root / "best.pt", {**best, "state_dict": model.state_dict(), "contract": initial})
        f1.save_torch(last, {"epoch": 0, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history, "best": best, "contract": initial})
    for epoch in range(begin_epoch, settings["epochs"] + 1):
        model.train()
        order = torch.randperm(len(train_indices), generator=torch.Generator().manual_seed(epoch)).numpy()
        sum_loss, seen = 0., 0
        for begin in range(0, len(order), settings["batch_size"]):
            idx = order[begin:begin + settings["batch_size"]]
            y = torch.from_numpy(np.array(inputs["train"][2]["dense_derived_labels.npy"][train_indices[idx]], copy=True)).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*batch(caches["train"], idx, device))
            loss = nn.functional.cross_entropy(logits.reshape(-1, 4), y.reshape(-1), weight=weights)
            require(bool(torch.isfinite(loss)), "nonfinite P1 loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            sum_loss += float(loss.detach()) * len(idx)
            seen += len(idx)
            if (begin // settings["batch_size"] + 1) % 100 == 0 or seen == len(order):
                progress(root, stage="p1_training", epoch=epoch, windows=seen, target=len(order))
                f1.disk_gate(f1_config)
        row = {"epoch": epoch, "windows": seen, "mean_batch_loss": sum_loss / seen}
        if epoch in settings["validation_epochs"]:
            progress(root, stage="p1_validation", epoch=epoch)
            metrics, _ = evaluate(model, caches["val"], timeline, root, epoch, device)
            row["metrics"] = metrics
            if selection_key(metrics) > selection_key(best["metrics"]):
                best = {"epoch": epoch, "metrics": metrics}
                f1.save_torch(root / "best.pt", {**best, "state_dict": model.state_dict(), "contract": initial})
        history.append(row)
        f1.save_torch(last, {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history, "best": best, "contract": initial})
        f1.save_json(root / "history.json", history)
        progress(root, stage="p1_epoch_completed", epoch=epoch, best_epoch=best["epoch"], validation=row.get("metrics"))
        require(contract(False)[0] == initial, "P1 lineage changed")
    f1.save_json(root / "training_report.json", {"epochs": settings["epochs"], "windows_per_epoch": len(train_indices), "best": best, "best_sha256": sha256_file(root / "best.pt"), "history_sha256": sha256_file(root / "history.json")})


def audit(root, config, f1_config, initial, base):
    require(contract(False)[0] == initial == f1.read(root / "run_contract.json"), "P1 audit lineage")
    require(f1.read(root / "epoch0_gate.json")["passed"], "epoch0 gate")
    invariance = f1.read(root / "invariance.json")
    require(invariance["before"] == invariance["after"] == hash_named_tensors(base.state_dict().items()), "frozen F1 changed")
    report = f1.read(root / "training_report.json")
    require(sha256_file(root / "best.pt") == report["best_sha256"] and sha256_file(root / "history.json") == report["history_sha256"], "training artifact hash")
    history = f1.read(root / "history.json")
    require([r["epoch"] for r in history] == list(range(11)), "training epochs missing")
    require(all(r["windows"] == 149996 for r in history[1:]), "quarter train count")
    validation = [r for r in history if "metrics" in r]
    require([r["epoch"] for r in validation] == [0, 2, 4, 6, 8, 10], "validation schedule")
    best = max(validation, key=lambda r: selection_key(r["metrics"]))
    require(report["best"] == {k: best[k] for k in ("epoch", "metrics")}, "best epoch selection")
    timeline = f1.timeline_from(f1.source(f1_config, "val", include_data=False))
    cache = load_cache(root, "val", len(timeline.starts))
    require(np.array_equal(cache["indices.npy"], np.arange(len(timeline.starts))), "val cache indices")
    train_count = len(subset_indices(f1_config["input"]["windows"]["train"]))
    train_cache = load_cache(root, "train", train_count)
    require(np.array_equal(train_cache["indices.npy"], subset_indices(f1_config["input"]["windows"]["train"])), "train cache indices")
    state = torch.load(root / "best.pt", map_location="cpu", weights_only=True)
    require(state["epoch"] == best["epoch"] and state["metrics"] == best["metrics"] and state["contract"] == initial, "selected checkpoint identity")
    parameters = {k: v.numpy().astype(np.float64) for k, v in state["state_dict"].items()}
    for row in validation:
        dest = root / "validation" / f'epoch{row["epoch"]:02d}'
        result = f1.read(dest / "result.json")
        for name, digest in result["files"].items():
            require(sha256_file(dest / name) == digest, "validation artifact changed")
        windows = np.load(dest / "window_logits.npy", mmap_mode="r", allow_pickle=False)
        totals = timeline.empty()
        for begin in range(0, len(windows), 128):
            x = np.asarray(windows[begin:begin + 128])
            idx = (timeline.window_offsets[begin:begin + len(x), None] + np.arange(64)).ravel()
            lo, hi = int(idx.min()), int(idx.max()) + 1
            for c in range(4):
                totals[lo:hi, c] += np.bincount(idx - lo, weights=x[:, :, c].ravel(), minlength=hi-lo)
            if row["epoch"] == best["epoch"]:
                h = np.asarray(cache["hidden.npy"][begin:begin + len(x)], dtype=np.float64)
                p = np.asarray(cache["signals.npy"][begin:begin + len(x)], dtype=np.float64)
                corrected = h + p @ parameters["projection.weight"].T + parameters["projection.bias"]
                independent = corrected @ parameters["classifier.weight"].T + parameters["classifier.bias"]
                require(np.allclose(x, independent, rtol=5e-5, atol=5e-5), "independent selected P1 outputs")
        logits = np.load(dest / "timeline_logits.npy", allow_pickle=False)
        require(np.array_equal(logits, timeline.mean(totals)), "independent overlap mean")
        metrics = timeline.metrics(logits)
        require(metrics == row["metrics"] == result["metrics"], "validation metrics mismatch")
        independently_check_f1(timeline, logits, metrics)
    return {"passed": True, "historical_exact_reproduction": False, "best": report["best"], "epoch0": history[0]["metrics"],
            "learned_correction_selected": best["epoch"] > 0, "frozen_f1_exact": True, "epoch0_exact": True,
            "training_windows": train_count, "validation_windows": len(timeline.starts), "test_ood_evaluated": False, "completed_at": f1.now()}


def smoke_run(root, f1_config, old, base, model, device):
    # Eight real windows per split, training count/metrics never reported as research.
    before = hash_named_tensors(base.state_dict().items())
    exact = []
    for split in ("train", "val"):
        src = f1.source(f1_config, split)
        idx = np.linspace(0, f1_config["input"]["windows"][split] - 1, 8, dtype=int)
        arrays = f1.validate_cache_split(old, split, f1_config["input"]["windows"][split], hash_files=False)
        with torch.no_grad():
            t, s = f1.feature_batch(arrays, idx, device)
            feature = torch.cat((t, s[:, None].expand(-1, 64, -1)), -1)
            hidden = base.hidden(feature)
            signals = torch.from_numpy(signals_batch(src[2]["data_joint.npy"][idx])).to(device)
            exact.append(torch.equal(model(hidden, signals), base(feature)))
            if split == "train":
                train_hidden, train_signals = hidden.clone(), signals.clone()
                train_labels = torch.from_numpy(np.array(src[2]["dense_derived_labels.npy"][idx], copy=True)).to(device)
    require(all(exact), "smoke epoch0 exact")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    logits = model(train_hidden, train_signals)
    nn.functional.cross_entropy(logits.reshape(-1, 4), train_labels.reshape(-1)).backward()
    require(model.projection.weight.grad is not None and model.projection.weight.grad.abs().sum() > 0, "no primitive gradient")
    optimizer.step()
    require(hash_named_tensors(base.state_dict().items()) == before and not any(p.requires_grad for p in base.parameters()), "frozen base mutated")
    f1.save_json(root / "smoke_report.json", {"passed": True, "research_usable": False, "epoch0_exact": True, "base_unchanged": True, "real_windows_each_split": 8})
    progress(root, stage="p1_smoke_passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "run", "audit"), default="run")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    smoke = args.stage == "smoke"
    initial, config, f1_config, old, head_path = contract(smoke)
    root = ROOT / config["output_dir"] / ("p1_smoke" if smoke else "p1")
    f1.initialize(root, initial, args.resume or args.stage == "audit")
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = torch.device("cpu") if args.stage == "audit" else f1.setup_device(f1_config)
        if device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(config["execution"]["gpu_memory_fraction"], device)
        base = load_base(head_path, device)
        before = hash_named_tensors(base.state_dict().items())
        model = PrimitiveCorrection(base.classifier).to(device)
        if smoke:
            smoke_run(root, f1_config, old, base, model, device)
            return
        if args.stage != "audit" and not (root / "final_report.json").exists():
            inputs = {s: f1.source(f1_config, s) for s in ("train", "val")}
            indices = {"train": subset_indices(f1_config["input"]["windows"]["train"]), "val": np.arange(f1_config["input"]["windows"]["val"])}
            caches = {s: cache_split(root, s, indices[s], inputs[s], base, model, old, f1_config, device) for s in ("train", "val")}
            train(root, config, f1_config, caches, inputs, base, model, device, initial)
            after = hash_named_tensors(base.state_dict().items())
            require(before == after, "frozen F1 invariant")
            f1.save_json(root / "invariance.json", {"before": before, "after": after})
        progress(root, stage="p1_independent_audit")
        report = audit(root, config, f1_config, initial, base)
        f1.save_json(root / "final_report.json", report)
        progress(root, stage="completed", best_epoch=report["best"]["epoch"], audit_passed=True)


if __name__ == "__main__":
    main()
