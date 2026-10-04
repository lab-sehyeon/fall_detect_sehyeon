#!/usr/bin/env python3
"""Five-fold frozen-DSTE linear evaluation on FU-Kinect-Fall.

This is a recovery of the structural F0-FU contract documented by the project:
whole clips are resized to 64 frames, the NTU60 DSTE encoder stays frozen, and
only a binary ``Linear(2048, 2)`` head is optimized.  Validation folds are
subject-disjoint according to the preprocessing manifest.

The original source file and its complete optimizer configuration were not
recoverable.  Consequently the output records both the historical contract and
the explicitly supplied recovery hyperparameters; it must not be described as
byte-identical to the lost implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from fall_pipeline.common.integrity import hash_named_tensors, sha256_file
from model.DSTE import Downstream
from tools import remove_prefix


HISTORICAL_FOLD_RESULTS = {
    "0": {"f1_percent": 95.122, "auprc_percent": 99.106, "best_epoch": 50},
    "1": {"f1_percent": 91.176, "auprc_percent": 99.342, "best_epoch": 48},
    "2": {"f1_percent": 90.909, "auprc_percent": 98.249, "best_epoch": 44},
    "3": {"f1_percent": 92.537, "auprc_percent": 97.279, "best_epoch": 50},
    "4": {"f1_percent": 94.118, "auprc_percent": 99.210, "best_epoch": 28},
}


def json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def enforce_physical_gpu_zero() -> torch.device:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError(
            "this recovery is locked to physical GPU 0; launch with CUDA_VISIBLE_DEVICES=0"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected one visible CUDA device after masking, got {torch.cuda.device_count()}"
        )
    return torch.device("cuda:0")


def validate_inputs(data_root: Path, pretrained: Path) -> dict:
    required = (
        "data_joint.npy", "num_frame.npy", "labels.npy", "action_ids.npy",
        "subjects.npy", "fold_ids.npy", "samples.csv", "preprocess_manifest.json",
    )
    missing = [name for name in required if not (data_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing FU preprocessing outputs: {missing}")
    if not pretrained.is_file():
        raise FileNotFoundError(pretrained)
    manifest = json.loads((data_root / "preprocess_manifest.json").read_text(encoding="utf-8"))
    integrity = manifest.get("integrity", {})
    if manifest.get("included_clips") != 993 or not integrity.get("passed"):
        raise RuntimeError("FU preprocessing manifest did not pass the 993-clip integrity gate")
    if manifest.get("fold_rule") != "(subject_id - 1) % 5":
        raise RuntimeError("unexpected FU fold rule")
    return manifest


def load_arrays(data_root: Path) -> tuple[np.ndarray, ...]:
    data = np.load(data_root / "data_joint.npy", mmap_mode="r")
    num_frames = np.load(data_root / "num_frame.npy")
    labels = np.load(data_root / "labels.npy")
    actions = np.load(data_root / "action_ids.npy")
    subjects = np.load(data_root / "subjects.npy")
    folds = np.load(data_root / "fold_ids.npy")
    expected = (993, 3, 300, 25, 2)
    if data.shape != expected or data.dtype != np.float32:
        raise RuntimeError(f"expected FU data {expected} float32, got {data.shape} {data.dtype}")
    if not all(array.shape == (993,) for array in (num_frames, labels, actions, subjects, folds)):
        raise RuntimeError("FU metadata arrays are not aligned to 993 samples")
    if set(np.unique(labels).tolist()) != {0, 1} or int(labels.sum()) != 165:
        raise RuntimeError("FU binary label contract failed")
    if set(np.unique(folds).tolist()) != set(range(5)):
        raise RuntimeError("FU five-fold contract failed")
    return data, num_frames, labels, actions, subjects, folds


def resize_whole_clips(data: np.ndarray, num_frames: np.ndarray) -> torch.Tensor:
    resized = torch.empty((len(num_frames), 3, 64, 25, 2), dtype=torch.float32)
    for index, frames_value in enumerate(num_frames):
        frames = int(frames_value)
        clip = torch.from_numpy(np.array(data[index, :, :frames], copy=True))
        flattened = clip.permute(0, 2, 3, 1).reshape(3 * 25 * 2, frames)
        output = F.interpolate(
            flattened[None, :, :, None], size=(64, 1), mode="bilinear", align_corners=False
        )
        resized[index] = output[0, :, :, 0].reshape(3, 25, 2, 64).permute(0, 3, 1, 2)
    return resized


def build_encoder(pretrained: Path, device: torch.device) -> Downstream:
    model = Downstream(
        t_input_size=150,
        s_input_size=192,
        hidden_size=1024,
        num_head=1,
        num_layer=2,
        num_class=60,
        modality="joint",
        alpha=0.5,
        gap=4,
        kernel_size=1,
    )
    checkpoint = torch.load(pretrained, map_location="cpu", weights_only=True)
    state = remove_prefix(checkpoint["state_dict"])
    message = model.load_state_dict(state, strict=False)
    if set(message.missing_keys) != {"fc.weight", "fc.bias"}:
        raise RuntimeError(f"unexpected missing checkpoint keys: {message.missing_keys}")
    unexpected_roots = {name.split(".", 1)[0] for name in message.unexpected_keys}
    if unexpected_roots != {"j_proj", "s_proj", "t_proj"}:
        raise RuntimeError(f"unexpected checkpoint key groups: {sorted(unexpected_roots)}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval().to(device)
    return model


def extract_features(
    encoder: Downstream,
    clips: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    features = []
    loader = DataLoader(TensorDataset(clips), batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            jt = batch.permute(0, 2, 4, 3, 1).reshape(batch.shape[0], 64, 150)
            js = batch.permute(0, 4, 3, 2, 1).reshape(batch.shape[0], 50, 192)
            temporal, spatial = encoder.backbone(jt, js)
            features.append(torch.cat((temporal.amax(1), spatial.amax(1)), dim=1).cpu())
    output = torch.cat(features)
    if output.shape != (len(clips), 2048) or not torch.isfinite(output).all():
        raise RuntimeError(f"invalid DSTE feature cache: {tuple(output.shape)}")
    return output


def confusion_metrics(labels: torch.Tensor, scores: torch.Tensor) -> dict:
    predictions = scores.argmax(dim=1)
    labels = labels.long()
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    probabilities = scores.softmax(dim=1)[:, 1].numpy()
    auprc = float(average_precision_score(labels.numpy(), probabilities))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "auprc": auprc,
        "accuracy": (tp + tn) / len(labels),
    }


def class_weights(labels: torch.Tensor, mode: str, device: torch.device) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = torch.bincount(labels.long(), minlength=2).float()
    inverse = counts.sum() / counts.clamp_min(1)
    weights = inverse.sqrt() if mode == "sqrt_inverse_frequency" else inverse
    return (weights / weights.mean()).to(device)


def train_fold(
    fold: int,
    features: torch.Tensor,
    labels: torch.Tensor,
    actions: torch.Tensor,
    subjects: torch.Tensor,
    folds: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    output_root: Path,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    seed_everything(args.seed)
    train_indices = torch.nonzero(folds != fold, as_tuple=True)[0]
    val_indices = torch.nonzero(folds == fold, as_tuple=True)[0]
    if set(subjects[train_indices].tolist()) & set(subjects[val_indices].tolist()):
        raise RuntimeError(f"subject leakage in fold {fold}")

    head = nn.Linear(2048, 2).to(device)
    nn.init.normal_(head.weight, mean=0.0, std=0.01)
    nn.init.zeros_(head.bias)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weight = class_weights(labels[train_indices], args.class_weight, device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(features[train_indices], labels[train_indices]),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    best_key = (-1.0, -1.0, -1.0)
    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        seen = 0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            logits = head(batch_features)
            loss = criterion(logits, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(batch_labels)
            seen += len(batch_labels)
        head.eval()
        with torch.inference_mode():
            val_logits = head(features[val_indices].to(device)).cpu()
        metrics = confusion_metrics(labels[val_indices], val_logits)
        metrics.update({"epoch": epoch, "train_loss": total_loss / seen})
        history.append(metrics)
        key = (metrics["f1"], metrics["auprc"], metrics["accuracy"])
        if key > best_key:
            best_key = key
            best = {
                "epoch": epoch,
                "state_dict": {name: value.detach().cpu().clone() for name, value in head.state_dict().items()},
                "logits": val_logits.clone(),
                "metrics": dict(metrics),
            }
        print(
            f"fold={fold} epoch={epoch:03d}/{args.epochs} loss={total_loss / seen:.6f} "
            f"val_f1={metrics['f1'] * 100:.3f} val_auprc={metrics['auprc'] * 100:.3f}",
            flush=True,
        )
    assert best is not None

    fold_root = output_root / f"fold{fold}"
    fold_root.mkdir(parents=True, exist_ok=False)
    torch.save(best["state_dict"], fold_root / "best_head.pth")
    torch.save(best["logits"], fold_root / "val_logits.pt")
    torch.save(labels[val_indices], fold_root / "val_labels.pt")
    torch.save(val_indices, fold_root / "val_indices.pt")
    json_dump(fold_root / "history.json", history)

    predictions = best["logits"].argmax(dim=1)
    action_fpr = {}
    for action_id, action_name in enumerate(("walking", "bending", "sitting", "squatting", "lying", "falling")):
        mask = actions[val_indices] == action_id
        if action_name == "falling":
            continue
        action_fpr[action_name] = float((predictions[mask] == 1).float().mean()) if mask.any() else None
    summary = {
        "fold": fold,
        "training_subjects": sorted(set(subjects[train_indices].tolist())),
        "validation_subjects": sorted(set(subjects[val_indices].tolist())),
        "training_samples": len(train_indices),
        "validation_samples": len(val_indices),
        "best_epoch": best["epoch"],
        "metrics": best["metrics"],
        "negative_action_fpr": action_fpr,
        "historical_reference": HISTORICAL_FOLD_RESULTS[str(fold)],
    }
    json_dump(fold_root / "summary.json", summary)
    return summary, val_indices, best["logits"]


def main(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    pretrained = args.pretrained.resolve()
    output_root = args.output_dir.resolve()
    manifest = validate_inputs(data_root, pretrained)
    device = enforce_physical_gpu_zero()
    print(json.dumps({
        "physical_cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "logical_device": str(device),
        "device_name": torch.cuda.get_device_name(0),
        "data_root": str(data_root),
        "pretrained": str(pretrained),
        "pretrained_sha256": sha256_file(pretrained),
        "preprocess_manifest_sha256": sha256_file(data_root / "preprocess_manifest.json"),
    }, indent=2), flush=True)
    if args.preflight_only:
        return
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    data, num_frames, labels_np, actions_np, subjects_np, folds_np = load_arrays(data_root)
    clips = resize_whole_clips(data, num_frames)
    labels = torch.from_numpy(labels_np).long()
    actions = torch.from_numpy(actions_np).long()
    subjects = torch.from_numpy(subjects_np).long()
    folds = torch.from_numpy(folds_np).long()
    encoder = build_encoder(pretrained, device)
    encoder_hash_before = hash_named_tensors(encoder.backbone.state_dict().items())
    features = extract_features(encoder, clips, device, args.feature_batch_size)
    encoder_hash_after = hash_named_tensors(encoder.backbone.state_dict().items())
    if encoder_hash_before != encoder_hash_after or any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("frozen DSTE integrity check failed")
    torch.save(features, output_root / "features.pt")

    fold_summaries = []
    oof_logits = torch.full((993, 2), float("nan"))
    oof_seen = torch.zeros(993, dtype=torch.int64)
    for fold in args.folds:
        summary, indices, logits = train_fold(
            fold, features, labels, actions, subjects, folds, args, device, output_root
        )
        fold_summaries.append(summary)
        oof_logits[indices] = logits
        oof_seen[indices] += 1
    all_folds = sorted(args.folds) == list(range(5))
    aggregate = None
    if all_folds:
        if not torch.equal(oof_seen, torch.ones_like(oof_seen)) or not torch.isfinite(oof_logits).all():
            raise RuntimeError("OOF coverage/integrity failed")
        aggregate = confusion_metrics(labels, oof_logits)
        torch.save(oof_logits, output_root / "oof_logits.pt")
        torch.save(labels, output_root / "oof_labels.pt")

    run_config = {
        "experiment": "F0-FU frozen DSTE binary linear evaluation",
        "recovery_status": {
            "historical_structural_contract_recovered": True,
            "original_source_text_recovered": False,
            "optimizer_hyperparameters_are_explicit_recovery_configuration": True,
            "do_not_claim_byte_identical_reproduction": True,
        },
        "args": {
            **vars(args),
            "data_root": str(data_root),
            "pretrained": str(pretrained),
            "output_dir": str(output_root),
        },
        "data_manifest_integrity": manifest["integrity"],
        "pretrained_sha256": sha256_file(pretrained),
        "preprocess_manifest_sha256": sha256_file(data_root / "preprocess_manifest.json"),
        "encoder_state_hash_before": encoder_hash_before,
        "encoder_state_hash_after": encoder_hash_after,
        "encoder_frozen_exact": encoder_hash_before == encoder_hash_after,
        "feature_shape": list(features.shape),
        "folds": fold_summaries,
        "oof_metrics": aggregate,
        "historical_reference": HISTORICAL_FOLD_RESULTS,
        "integrity": {
            "passed": encoder_hash_before == encoder_hash_after and (not all_folds or bool(torch.all(oof_seen == 1))),
            "physical_gpu_zero_only": os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
            "only_linear_heads_trainable": True,
            "subject_disjoint": all(
                not (set(item["training_subjects"]) & set(item["validation_subjects"]))
                for item in fold_summaries
            ),
            "oof_each_sample_once": bool(torch.all(oof_seen == 1)) if all_folds else None,
        },
    }
    json_dump(output_root / "run_config.json", run_config)
    print(json.dumps({
        "output_dir": str(output_root),
        "fold_best": [
            {"fold": item["fold"], "best_epoch": item["best_epoch"], **item["metrics"]}
            for item in fold_summaries
        ],
        "oof_metrics": aggregate,
        "integrity": run_config["integrity"],
    }, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--class-weight",
        choices=("none", "inverse_frequency", "sqrt_inverse_frequency"),
        default="sqrt_inverse_frequency",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not args.folds or len(set(args.folds)) != len(args.folds) or any(fold not in range(5) for fold in args.folds):
        parser.error("--folds must contain unique values from 0 through 4")
    if min(args.epochs, args.batch_size, args.feature_batch_size) < 1:
        parser.error("epochs and batch sizes must be positive")
    return args


if __name__ == "__main__":
    main(parse_args())
