#!/usr/bin/env python3
"""Explicit F1 recovery run: guarded cache -> four candidates -> locked holdouts.

Only physical GPU 0 is accepted. This is a new document-grounded recovery
configuration, not a claim to restore the lost original F1 implementation.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data_gen.safer_legacy_v1_gendata import derive_four_class
from fall_pipeline.common.eval_safer_legacy_v1_candidates import load_adl_model
from fall_pipeline.common.fall_safer_zeroshot_eval import validate_adl_run_config
from fall_pipeline.common.integrity import hash_named_tensors, sha256_file
from fall_pipeline.fu.fall_fu_linear_eval import enforce_physical_gpu_zero
from fall_pipeline.safer.f1_core import (
    ResidualTemporalAdapter, Timeline, candidate_input, dense_backbone_features,
    make_candidates, require, selection_key, sqrt_weights,
)
from fall_pipeline.safer.f1_progress_docs import publish

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/f1_safer_document_reconstruction_v2.json"
CONFIG_HASH = "4e27a0d8537713e1495b4ba2ec007ce18c6a8ca567410813420533de35ea6cc8"
CODE_FILES = (
    "fall_pipeline/safer/run_f1_reconstruction.py", "fall_pipeline/safer/f1_core.py",
    "fall_pipeline/safer/train_f0b_safer_heads.py", "fall_pipeline/common/integrity.py",
    "fall_pipeline/common/metrics.py", "fall_pipeline/common/eval_safer_legacy_v1_candidates.py",
    "fall_pipeline/common/fall_safer_zeroshot_eval.py", "fall_pipeline/fu/fall_fu_linear_eval.py",
    "data_gen/safer_legacy_v1_gendata.py", "model/DSTE.py",
    "fall_pipeline/safer/audit_f1_reconstruction.py", "fall_pipeline/safer/f1_progress_docs.py",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def log(**values):
    print(json.dumps({"time": now(), **values}, ensure_ascii=False, allow_nan=False), flush=True)


def config_load(path=CONFIG):
    require(sha256_file(path) == CONFIG_HASH, "F1 preregistered config changed; new revision required")
    return read(path)


def input_gate(config):
    inputs = config["input"]
    for key in ("encoder", "adl_head", "adl_run_config"):
        require(sha256_file(ROOT / inputs[key]) == inputs[key + "_sha256"], f"pinned {key} changed")
    validate_adl_run_config(ROOT / inputs["adl_run_config"])
    manifest_path = ROOT / inputs["data_root"] / "materialization_manifest.json"
    require(sha256_file(manifest_path) == inputs["materialization_manifest_sha256"], "materialization changed")
    manifest = read(manifest_path)
    require(manifest["status"] == "completed" and manifest["research_usable"] and manifest["integrity"]["passed"],
            "invalid materialization")
    return manifest


def payload_digest(array):
    digest = hashlib.sha256()
    for begin in range(0, len(array), 256):
        digest.update(np.asarray(array[begin:begin + 256]).tobytes(order="C"))
    return digest.hexdigest()


def source(config, split, include_data=True):
    require(split in config["input"]["windows"], "unknown source split")
    base = ROOT / config["input"]["data_root"] / split
    root_manifest = read(base.parent / "materialization_manifest.json")
    require(sha256_file(base / "split_manifest.json") == root_manifest["splits"][split]["manifest_sha256"],
            "split manifest hash")
    manifest = read(base / "split_manifest.json")
    count = config["input"]["windows"][split]
    require(manifest["written_windows"] == count and manifest["integrity"]["passed"], "source count/integrity")
    arrays = {}
    names = ["dense_coarse_labels.npy", "dense_derived_labels.npy", "sequence_index.npy", "window_start.npy"]
    if include_data:
        names.append("data_joint.npy")
    for name in names:
        path = base / name
        require(path.stat().st_size == manifest["payload"][name]["bytes"], f"source size {split}/{name}")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        expected = ((count, 3, 64, 25, 2) if name == "data_joint.npy" else
                    (count, 64) if name.startswith("dense_") else (count,))
        require(value.shape == expected, f"source shape {split}/{name}")
        require(value.dtype == (np.float32 if name == "data_joint.npy" else np.int64), "source dtype")
        if name != "data_joint.npy":
            require(payload_digest(value) == manifest["payload"][name]["array_payload_sha256"], f"source hash {name}")
        arrays[name] = value
    for begin in range(0, count, 8192):
        require(np.array_equal(derive_four_class(arrays["dense_coarse_labels.npy"][begin:begin + 8192]),
                               arrays["dense_derived_labels.npy"][begin:begin + 8192]), "derived label mapping")
    require(sha256_file(base / "sequences.json") == manifest["small_file_sha256"]["sequences.json"], "sequence metadata hash")
    return base, manifest, arrays, read(base / "sequences.json")


def timeline_from(source_tuple):
    _, _, arrays, sequences = source_tuple
    return Timeline(sequences, arrays["sequence_index.npy"], arrays["window_start.npy"], arrays["dense_coarse_labels.npy"])


def disk_gate(config, additional=0):
    free = shutil.disk_usage(ROOT).free
    require(free >= additional + config["execution"]["reserve_gib"] * 1024**3, "disk reserve gate failed")
    return free


def immutable_contract(config, smoke):
    return {"experiment_id": config["experiment_id"], "config_sha256": CONFIG_HASH,
            "code_sha256": {name: sha256_file(ROOT / name) for name in CODE_FILES},
            "input": config["input"], "smoke_only": smoke,
            "physical_gpu": "0", "historical_exact_reproduction": False}


def initialize(root, contract, resume):
    if root.exists() and any(root.iterdir()):
        require(resume, "nonempty output; identical --resume required")
        require((root / "run_contract.json").is_file(), "output lacks contract")
        require(read(root / "run_contract.json") == contract, "resume contract mismatch")
    else:
        root.mkdir(parents=True, exist_ok=True)
        save_json(root / "run_contract.json", contract)


def assert_contract(root, config, smoke=False):
    require(read(root / "run_contract.json") == immutable_contract(config, smoke), "code/config changed during run")
    input_gate(config)


def setup_device(config):
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == config["execution"]["cublas_workspace_config"], "CUBLAS workspace contract")
    torch.set_num_threads(config["execution"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = enforce_physical_gpu_zero()
    free, total = torch.cuda.mem_get_info(device)
    require(free >= config["execution"]["minimum_gpu_free_gib"] * 1024**3, "insufficient GPU 0 free memory")
    torch.manual_seed(config["training"]["seed"])
    return device


def backbone(config, device):
    model = load_adl_model(ROOT / config["input"]["encoder"], ROOT / config["input"]["adl_head"], device)
    require(not model.training and not any(p.requires_grad for p in model.parameters()), "backbone/ADL not frozen")
    return model


def validate_cache_split(root, split, count, hash_files=True):
    base = root / "cache" / split
    manifest = read(base / "manifest.json")
    require(manifest["status"] == "completed" and manifest["windows"] == count, "incomplete cache")
    arrays = {}
    for name, shape in (("temporal.npy", (count, 64, 1024)), ("context.npy", (count, 1024))):
        path = base / name
        require(path.stat().st_size == manifest["files"][name]["bytes"], "cache size")
        if hash_files:
            require(sha256_file(path) == manifest["files"][name]["sha256"], "cache file hash")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        require(value.shape == shape and value.dtype == np.float32, "cache shape/dtype")
        arrays[name] = value
    return arrays


def extract(config, root, device):
    cache = root / "cache"
    if (cache / "report.json").exists():
        report = read(cache / "report.json")
        require(report["status"] == "completed" and report["frozen_exact"], "cache report failed")
        return
    needed = 0
    for split in ("train", "val"):
        count = config["input"]["windows"][split]
        progress = cache / split / "progress.json"
        written = read(progress)["next_index"] if progress.exists() else 0
        needed += (count - written) * 65 * 1024 * 4
    disk_gate(config, needed)
    model = backbone(config, device)
    before = hash_named_tensors(model.state_dict().items())
    batch_size = config["execution"]["extraction_batch_size"]
    results = {}
    for split in ("train", "val"):
        dest = cache / split
        count = config["input"]["windows"][split]
        if (dest / "manifest.json").exists():
            validate_cache_split(root, split, count)
            results[split] = read(dest / "manifest.json")
            continue
        _, manifest, arrays, _ = source(config, split)
        progress_path = dest / "progress.json"
        next_index = read(progress_path)["next_index"] if progress_path.exists() else 0
        require(0 <= next_index <= count, "cache progress index")
        if dest.exists() and any(dest.iterdir()) and not progress_path.exists():
            raise RuntimeError("partial cache without progress; preserve and inspect")
        dest.mkdir(parents=True, exist_ok=True)
        outputs = {}
        for name, shape in (("temporal.npy", (count, 64, 1024)), ("context.npy", (count, 1024))):
            path = dest / name
            if progress_path.exists():
                value = np.load(path, mmap_mode="r+", allow_pickle=False)
                require(value.shape == shape and value.dtype == np.float32, "resumed cache shape")
            else:
                value = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
            outputs[name] = value
        save_json(progress_path, {"next_index": next_index, "target": count})
        with torch.inference_mode():
            for begin in range(next_index, count, batch_size):
                end = min(begin + batch_size, count)
                batch = torch.from_numpy(np.array(arrays["data_joint.npy"][begin:end], copy=True)).to(device)
                temporal, context = dense_backbone_features(model, batch)
                outputs["temporal.npy"][begin:end] = temporal.cpu().numpy()
                outputs["context.npy"][begin:end] = context.cpu().numpy()
                if (begin // batch_size + 1) % config["execution"]["checkpoint_every_batches"] == 0 or end == count:
                    for value in outputs.values():
                        value.flush()
                    save_json(progress_path, {"next_index": end, "target": count})
                    disk_gate(config)
                    log(stage="extract", split=split, windows=end, target=count)
        for value in outputs.values():
            value.flush()
            require(all(np.isfinite(value[i:i + batch_size]).all() for i in range(0, count, batch_size)), "cache finite")
        outputs.clear()
        require(payload_digest(arrays["data_joint.npy"]) == manifest["payload"]["data_joint.npy"]["array_payload_sha256"], "source joint hash")
        result = {"status": "completed", "windows": count, "source_joint_payload_verified": True,
                  "files": {name: {"bytes": (dest / name).stat().st_size, "sha256": sha256_file(dest / name)}
                            for name in ("temporal.npy", "context.npy")}}
        save_json(dest / "manifest.json", result)
        results[split] = result
        log(stage="cache_split_verified", split=split)
    after = hash_named_tensors(model.state_dict().items())
    require(before == after, "DSTE/ADL changed during extraction")
    assert_contract(root, config)
    save_json(cache / "report.json", {"status": "completed", "splits": results, "frozen_exact": True,
                                      "model_hash_before": before, "model_hash_after": after,
                                      "test_ood_opened": False, "completed_at": now()})
    del model
    torch.cuda.empty_cache()


def feature_batch(arrays, indices, device):
    return tuple(torch.from_numpy(np.array(arrays[name][indices], copy=True)).to(device)
                 for name in ("temporal.npy", "context.npy"))


def optimizer_for(model, config):
    train = config["training"]
    return torch.optim.AdamW(model.parameters(), lr=train["learning_rate"], betas=tuple(train["betas"]),
                             eps=train["epsilon"], weight_decay=train["weight_decay"])


def train_step(models, optimizers, temporal, context, labels, weights, config):
    losses = {}
    for name, model in models.items():
        model.train()
        optimizers[name].zero_grad(set_to_none=True)
        logits = model(candidate_input(name, temporal, context))
        loss = F.cross_entropy(logits.reshape(-1, 4), labels.reshape(-1),
                               weight=weights if name.endswith("__sqrt") else None)
        require(bool(torch.isfinite(loss)), f"nonfinite loss {name}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"], error_if_nonfinite=True)
        optimizers[name].step()
        losses[name] = float(loss.detach())
    return losses


def validation(models, arrays, timeline, device, config):
    totals = {name: timeline.empty() for name in models}
    for model in models.values():
        model.eval()
    batch_size = config["evaluation"]["batch_size"]
    count = len(timeline.starts)
    with torch.inference_mode():
        for begin in range(0, count, batch_size):
            temporal, context = feature_batch(arrays, slice(begin, begin + batch_size), device)
            for name, model in models.items():
                output = model(candidate_input(name, temporal, context)).cpu().numpy()
                timeline.add(totals[name], begin, output)
    predictions = {name: timeline.mean(value) for name, value in totals.items()}
    return {name: timeline.metrics(value) for name, value in predictions.items()}, predictions


def train(config, root, device):
    train_root = root / "training"
    report_path = train_root / "report.json"
    if report_path.exists():
        require(read(report_path)["status"] == "completed", "training report failed")
        return
    require(read(root / "cache/report.json")["frozen_exact"], "frozen cache gate")
    train_root.mkdir(parents=True, exist_ok=True)
    cache_arrays = {split: validate_cache_split(root, split, config["input"]["windows"][split]) for split in ("train", "val")}
    inputs = {split: source(config, split, include_data=False) for split in ("train", "val")}
    timelines = {split: timeline_from(inputs[split]) for split in ("train", "val")}
    train_subjects = {s["subject"] for s in inputs["train"][3]}
    require(not train_subjects.intersection(s["subject"] for s in inputs["val"][3]), "subject leakage")
    counts = np.bincount(timelines["train"].labels, minlength=4)
    weights = sqrt_weights(counts).to(device)
    models = make_candidates(config, device)
    optimizers = {name: optimizer_for(model, config) for name, model in models.items()}
    best, start_epoch = {}, 1
    last_path = train_root / "last.pt"
    if last_path.exists():
        last = torch.load(last_path, map_location="cpu", weights_only=True)
        require(last["contract"] == read(root / "run_contract.json"), "training resume contract")
        for name in models:
            models[name].load_state_dict(last["models"][name], strict=True)
            optimizers[name].load_state_dict(last["optimizers"][name])
        best, start_epoch = last["best"], last["epoch"] + 1
        torch.set_rng_state(last["cpu_rng"])
        torch.cuda.set_rng_state(last["cuda_rng"], device)
    save_json(train_root / "loss_weights.json", {"unit": "unique covered train frames", "counts": counts.tolist(), "weights": weights.cpu().tolist()})
    batch_size = config["training"]["batch_size"]
    n = config["input"]["windows"]["train"]
    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        started = time.monotonic()
        generator = torch.Generator().manual_seed(config["training"]["seed"] + epoch)
        order = torch.randperm(n, generator=generator).numpy()
        seen = 0
        loss_sums = {name: 0.0 for name in models}
        for begin in range(0, n, batch_size):
            indices = order[begin:begin + batch_size]
            temporal, context = feature_batch(cache_arrays["train"], indices, device)
            labels = torch.from_numpy(np.array(inputs["train"][2]["dense_derived_labels.npy"][indices], copy=True)).to(device)
            losses = train_step(models, optimizers, temporal, context, labels, weights, config)
            for name, loss in losses.items():
                loss_sums[name] += loss * len(indices)
            seen += len(indices)
            if begin == 0 or (begin // batch_size + 1) % 100 == 0 or seen == n:
                log(stage="train", epoch=epoch, windows=seen, target=n, losses=losses)
                disk_gate(config)
                save_json(root / "status.json", {"stage": "training", "epoch": epoch, "windows": seen, "target": n, "time": now()})
        require(seen == n, "incomplete training epoch")
        log(stage="validation_started", epoch=epoch)
        metrics, predictions = validation(models, cache_arrays["val"], timelines["val"], device, config)
        for name in models:
            if name not in best or selection_key(metrics[name]) > selection_key(best[name]["metrics"]):
                dest = train_root / name
                best[name] = {"epoch": epoch, "metrics": metrics[name]}
                save_torch(dest / "best.pt", {"candidate": name, "epoch": epoch, "config_sha256": CONFIG_HASH,
                                               "state_dict": models[name].state_dict(), "metrics": metrics[name]})
                np.save(dest / "best_val_timeline_logits.npy", predictions[name], allow_pickle=False)
                save_json(dest / "best.json", best[name])
        epoch_report = {"epoch": epoch, "windows": seen, "validation": metrics,
                        "window_weighted_mean_batch_loss": {k: v / n for k, v in loss_sums.items()},
                        "elapsed_seconds": time.monotonic() - started}
        save_json(train_root / f"epoch_{epoch:03d}.json", epoch_report)
        save_torch(last_path, {"contract": read(root / "run_contract.json"), "epoch": epoch,
                               "models": {k: v.state_dict() for k, v in models.items()},
                               "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
                               "best": best, "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device)})
        log(stage="epoch_completed", epoch=epoch, elapsed_seconds=epoch_report["elapsed_seconds"],
            validation={k: {m: v[m] for m in config["evaluation"]["selection"]} for k, v in metrics.items()})
        assert_contract(root, config)
        publish(root, config, "epoch", epoch=epoch, metrics=metrics)
    # Candidate ties prefer the earliest epoch, then fixed candidate order.
    selected = max(config["candidates"], key=lambda k: (*selection_key(best[k]["metrics"]), -best[k]["epoch"], -config["candidates"].index(k)))
    np.save(train_root / "validation_coarse.npy", timelines["val"].coarse[timelines["val"].covered], allow_pickle=False)
    np.save(train_root / "validation_frame_indices.npy", np.flatnonzero(timelines["val"].covered), allow_pickle=False)
    report = {"status": "completed", "config_sha256": CONFIG_HASH, "epochs": config["training"]["epochs"],
              "candidates": best, "selected_candidate": selected, "selected_epoch": best[selected]["epoch"],
              "selected_head_sha256": sha256_file(train_root / selected / "best.pt"),
              "test_ood_used_for_selection": False, "historical_exact_reproduction": False,
              "completed_at": now()}
    assert_contract(root, config)
    save_json(report_path, report)
    save_json(root / "selection_lock.json", {"candidate": selected, "epoch": best[selected]["epoch"],
                                             "head_sha256": report["selected_head_sha256"],
                                             "training_report_sha256": sha256_file(report_path), "config_sha256": CONFIG_HASH})
    log(stage="selection_locked", candidate=selected, epoch=best[selected]["epoch"])
    del models, optimizers, cache_arrays
    torch.cuda.empty_cache()


def evaluate(config, root, device):
    lock = read(root / "selection_lock.json")
    report = read(root / "training/report.json")
    require(lock["config_sha256"] == CONFIG_HASH and not report["test_ood_used_for_selection"], "selection isolation")
    require(sha256_file(root / "training/report.json") == lock["training_report_sha256"], "training report changed")
    name = lock["candidate"]
    checkpoint_path = root / "training" / name / "best.pt"
    require(sha256_file(checkpoint_path) == lock["head_sha256"], "selected head changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(checkpoint["candidate"] == name and checkpoint["epoch"] == lock["epoch"], "checkpoint selection mismatch")
    head = ResidualTemporalAdapter(1024 if name.startswith("temporal_only") else 2048).to(device).eval()
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    head.requires_grad_(False)
    model = backbone(config, device)
    before = hash_named_tensors(model.state_dict().items())
    head_before = hash_named_tensors(head.state_dict().items())
    results = {}
    for split in ("test", "ood"):
        dest = root / "evaluation" / split
        if (dest / "result.json").exists():
            existing = read(dest / "result.json")
            require(existing["selection_lock"] == lock and existing["status"] == "completed", "completed holdout mismatch")
            for filename, spec in existing["files"].items():
                require(sha256_file(dest / filename) == spec["sha256"], "holdout artifact changed")
            results[split] = existing
            continue
        src = source(config, split)
        _, source_manifest, arrays, _ = src
        timeline = timeline_from(src)
        n = len(timeline.starts)
        path = dest / "window_logits.npy"
        progress_path = dest / "progress.json"
        next_index = read(progress_path)["next_index"] if progress_path.exists() else 0
        require(0 <= next_index <= n, "holdout resume index")
        disk_gate(config, (n - next_index) * 64 * 4 * 4)
        if dest.exists() and any(dest.iterdir()) and not progress_path.exists():
            raise RuntimeError("incomplete holdout without progress; preserve and inspect")
        dest.mkdir(parents=True, exist_ok=True)
        logits = np.lib.format.open_memmap(path, mode="r+" if progress_path.exists() else "w+", dtype=np.float32, shape=(n, 64, 4))
        require(logits.shape == (n, 64, 4) and logits.dtype == np.float32, "holdout shape/dtype")
        save_json(progress_path, {"next_index": next_index, "target": n})
        batch_size = config["evaluation"]["batch_size"]
        with torch.inference_mode():
            for begin in range(next_index, n, batch_size):
                end = min(begin + batch_size, n)
                batch = torch.from_numpy(np.array(arrays["data_joint.npy"][begin:end], copy=True)).to(device)
                temporal, context = dense_backbone_features(model, batch)
                output = head(candidate_input(name, temporal, context))
                require(bool(torch.isfinite(output).all()), "holdout nonfinite logits")
                logits[begin:end] = output.cpu().numpy()
                if (begin // batch_size + 1) % 25 == 0 or end == n:
                    logits.flush()
                    save_json(progress_path, {"next_index": end, "target": n})
                    log(stage="holdout", split=split, windows=end, target=n)
                    disk_gate(config)
        logits.flush()
        totals = timeline.empty()
        for begin in range(0, n, batch_size):
            timeline.add(totals, begin, np.asarray(logits[begin:begin + batch_size]))
        mean = timeline.mean(totals)
        metrics = timeline.metrics(mean)
        require(payload_digest(arrays["data_joint.npy"]) == source_manifest["payload"]["data_joint.npy"]["array_payload_sha256"], "holdout source payload hash")
        np.save(dest / "timeline_logits.npy", mean, allow_pickle=False)
        np.save(dest / "coarse_labels.npy", timeline.coarse[timeline.covered], allow_pickle=False)
        np.save(dest / "frame_indices.npy", np.flatnonzero(timeline.covered), allow_pickle=False)
        files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(dest.glob("*.npy"))}
        results[split] = {"status": "completed", "selection_lock": lock, "windows": n, "metrics": metrics,
                          "files": files, "input_payload_verified": True, "no_tuning": True}
        save_json(dest / "result.json", results[split])
        log(stage="holdout_complete", split=split, metrics=metrics)
    require(hash_named_tensors(model.state_dict().items()) == before, "backbone state changed")
    require(hash_named_tensors(head.state_dict().items()) == head_before, "selected head changed")
    assert_contract(root, config)
    save_json(root / "final_report.json", {"status": "completed", "research_usable": True,
                                           "historical_exact_reproduction": False, "selection_lock": lock,
                                           "frozen_backbone_adl_exact": True, "frozen_selected_head_exact": True,
                                           "model_state_hash": before, "selected_head_state_hash": head_before,
                                           "results": results, "config_sha256": CONFIG_HASH, "completed_at": now()})


def smoke(config, root, device):
    src = source(config, "train")
    timeline = timeline_from(src)
    batch_size = config["training"]["batch_size"]
    model = backbone(config, device)
    before = hash_named_tensors(model.state_dict().items())
    batch = torch.from_numpy(np.array(src[2]["data_joint.npy"][:batch_size], copy=True)).to(device)
    with torch.inference_mode():
        t, s = dense_backbone_features(model, batch)
        zero_t, zero_s = dense_backbone_features(model, torch.zeros_like(batch[:1]))
        require(bool(torch.isfinite(zero_t).all() and torch.count_nonzero(zero_s) == 0), "empty context convention")
    # Clone outside inference_mode so downstream trainable layers can save inputs.
    temporal, context = t.clone(), s.clone()
    models = make_candidates(config, device)
    optimizers = {name: optimizer_for(value, config) for name, value in models.items()}
    weights = sqrt_weights(np.bincount(timeline.labels, minlength=4)).to(device)
    labels = torch.from_numpy(np.array(src[2]["dense_derived_labels.npy"][:batch_size], copy=True)).to(device)
    started = time.monotonic()
    losses = train_step(models, optimizers, temporal, context, labels, weights, config)
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    for name, head in models.items():
        head.eval()
        checkpoint_path = root / f"{name}.pt"
        save_torch(checkpoint_path, head.state_dict())
        with torch.inference_mode():
            expected = head(candidate_input(name, temporal, context))
            state = torch.load(checkpoint_path, map_location=device, weights_only=True)
            head.load_state_dict(state, strict=True)
            actual = head(candidate_input(name, temporal, context))
        require(torch.equal(expected, actual) and bool(torch.isfinite(actual).all()), "smoke checkpoint roundtrip")
    require(before == hash_named_tensors(model.state_dict().items()), "smoke backbone changed")
    assert_contract(root, config, smoke=True)
    result = {"status": "passed", "research_usable": False, "training_windows": batch_size,
              "candidate_losses": losses, "four_candidate_step_seconds": seconds,
              "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "frozen_dste_adl_exact": True, "checkpoint_roundtrip_exact": True,
              "test_ood_opened": False, "scope": "execution smoke; not performance or selection"}
    result["all_zero_input_context_zero"] = True
    save_json(root / "smoke_report.json", result)
    log(stage="smoke_complete", **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "smoke", "extract", "train", "evaluate", "all"), default="preflight")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    config = config_load()
    root = (args.output_dir or ROOT / config["execution"]["output_dir"]).resolve()
    require(root.is_relative_to(ROOT / "checkpoint/fall"), "output must remain under project checkpoint/fall")
    require(root != ROOT / "checkpoint/fall", "output must be a dedicated child directory")
    input_gate(config)
    device = setup_device(config)
    expected_bytes = sum(config["input"]["windows"][s] for s in ("train", "val")) * 65 * 1024 * 4
    log(stage="preflight", config_sha256=CONFIG_HASH, gpu=torch.cuda.get_device_name(0),
        dense_cache_bytes=expected_bytes, free_disk_bytes=disk_gate(config), output=str(root))
    if args.stage == "preflight":
        disk_gate(config, expected_bytes)
        return
    require(args.stage != "smoke" or root != ROOT / config["execution"]["output_dir"], "smoke requires separate output namespace")
    # Keep the descriptor alive for the entire run; a second --resume must not
    # mutate progress/checkpoints while this process owns the namespace.
    lock_handle = (root.parent / ("." + root.name + ".lock")).open("a+")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    initialize(root, immutable_contract(config, args.stage == "smoke"), args.resume)
    try:
        if args.stage == "smoke":
            smoke(config, root, device)
        else:
            for stage, function in (("extract", extract), ("train", train), ("evaluate", evaluate)):
                if args.stage in (stage, "all"):
                    save_json(root / "status.json", {"stage": stage, "time": now()})
                    publish(root, config, stage)
                    function(config, root, device)
                    if stage == "evaluate":
                        from fall_pipeline.safer.audit_f1_reconstruction import audit
                        save_json(root / "independent_audit.json", audit(root))
                        final = read(root / "final_report.json")
                        publish(root, config, "complete", metrics={k: v["metrics"] for k, v in final["results"].items()}, audited=True)
            save_json(root / "status.json", {"stage": args.stage, "status": "completed", "time": now()})
    except Exception as exc:
        save_json(root / "failure.json", {"stage": args.stage, "type": type(exc).__name__, "message": str(exc),
                                         "traceback": traceback.format_exc(), "time": now()})
        save_json(root / "failures" / f"{time.time_ns()}.json", read(root / "failure.json"))
        if args.stage != "smoke":
            save_json(root / "status.json", {"stage": args.stage, "status": "failed", "time": now()})
            publish(root, config, "failed", failed=True)
        raise


if __name__ == "__main__":
    main()
