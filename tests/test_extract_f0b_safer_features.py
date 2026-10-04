import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from fall_pipeline.common.integrity import sha256_file
from fall_pipeline.safer.extract_f0b_safer_features import (
    ALLOWED_SPLITS,
    EXPECTED_COUNTS,
    extract_split,
    immutable_run_contract,
    normalize_requested_splits,
    pooled_backbone_features,
    validate_materialization,
)


class DummyDSTE(torch.nn.Module):
    def backbone(self, temporal_input, spatial_input):
        batch = temporal_input.shape[0]
        temporal = torch.arange(64, dtype=torch.float32).view(1, 64, 1)
        temporal = temporal.expand(batch, 64, 1024)
        spatial = torch.arange(50, dtype=torch.float32).view(1, 50, 1)
        spatial = spatial.expand(batch, 50, 1024)
        return temporal, spatial


def write_source_split(root: Path, split: str, count: int) -> str:
    split_root = root / split
    split_root.mkdir()
    data = np.lib.format.open_memmap(
        split_root / "data_joint.npy",
        mode="w+",
        dtype=np.float32,
        shape=(count, 3, 64, 25, 2),
    )
    data[:] = 0
    data.flush()
    del data
    coarse = np.asarray(([0, 10, 11, 12] * ((count + 3) // 4))[:count], dtype=np.int64)
    derived = np.zeros(count, dtype=np.int64)
    derived[coarse == 10] = 1
    derived[coarse == 11] = 2
    derived[coarse == 12] = 3
    np.save(split_root / "center_coarse_labels.npy", coarse)
    np.save(split_root / "center_derived_labels.npy", derived)
    np.save(split_root / "sequence_index.npy", np.zeros(count, dtype=np.int64))
    np.save(split_root / "window_start.npy", np.arange(count, dtype=np.int64) * 8)
    manifest = {
        "written_windows": count,
        "candidate": {"name": "legacy_coco_umurl_window_noscale"},
        "payload": {
            "data_joint.npy": {
                "bytes": (split_root / "data_joint.npy").stat().st_size,
                "array_payload_sha256": f"test-{split}",
            }
        },
        "integrity": {"passed": True},
    }
    path = split_root / "split_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return sha256_file(path)


class F0BSaferFeatureExtractionTest(unittest.TestCase):
    def test_development_splits_are_hard_locked(self):
        self.assertEqual(normalize_requested_splits(["train", "val"]), ALLOWED_SPLITS)
        for invalid in (["val", "train"], ["train"], ["train", "test"]):
            with self.assertRaisesRegex(ValueError, "locked"):
                normalize_requested_splits(invalid)

    def test_pooling_keeps_temporal_and_spatial_features_separate(self):
        batch = torch.zeros((2, 3, 64, 25, 2), dtype=torch.float32)
        temporal, spatial = pooled_backbone_features(DummyDSTE(), batch)
        self.assertEqual(tuple(temporal.shape), (2, 1024))
        self.assertEqual(tuple(spatial.shape), (2, 1024))
        self.assertTrue(torch.equal(temporal, torch.full_like(temporal, 63.0)))
        self.assertTrue(torch.equal(spatial, torch.full_like(spatial, 49.0)))

    def test_materialization_gate_reads_only_train_and_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            counts = {"train": 4, "val": 3}
            split_entries = {}
            with patch.dict(EXPECTED_COUNTS, counts, clear=True):
                for split, count in counts.items():
                    split_entries[split] = {
                        "manifest_sha256": write_source_split(root, split, count)
                    }
                manifest = {
                    "status": "completed",
                    "research_usable": True,
                    "candidate": {"name": "legacy_coco_umurl_window_noscale"},
                    "selection_basis": {
                        "kind": "record_grounded_structural_reconstruction"
                    },
                    "recovery_status": {
                        "experimental_compatibility_reconstruction": True,
                        "must_not_be_called_byte_identical_reproduction": True,
                    },
                    "protocol": {
                        "window_size": 64,
                        "test_and_ood_used_for_candidate_selection": False,
                    },
                    "splits": split_entries,
                    "integrity": {"passed": True},
                }
                (root / "materialization_manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                result = validate_materialization(root, ["train", "val"])
            self.assertFalse(result["test_ood_arrays_opened"])
            self.assertEqual(result["splits"]["train"]["class_counts"], [1, 1, 1, 1])
            self.assertNotIn("test", result["splits"])
            self.assertNotIn("ood", result["splits"])

    def test_run_contract_declares_no_training_or_test_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = argparse.Namespace(
                data_root=root / "data",
                encoder=root / "encoder.pth",
                adl_head=root / "head.pth",
                adl_run_config=root / "config.json",
                splits=["train", "val"],
                batch_size=256,
                max_windows_per_split=0,
            )
            contract = immutable_run_contract(
                args,
                {"manifest_sha256": "materialization"},
                "encoder",
                "head",
                "config",
                "extractor",
            )
        self.assertFalse(contract["training"])
        self.assertFalse(contract["optimizer_created"])
        self.assertFalse(contract["test_ood_access"])
        self.assertEqual(contract["pooling"]["temporal_spatial"], "runtime concatenation -> 2048-D")

    def test_split_writer_is_resumable_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_source_split(root, "train", 4)
            output = root / "cache"
            source = {
                "root": str(root / "train"),
                "count": 4,
                "data_path": str(root / "train" / "data_joint.npy"),
                "source_manifest": str(root / "train" / "split_manifest.json"),
                "source_manifest_sha256": sha256_file(
                    root / "train" / "split_manifest.json"
                ),
                "source_data_payload_sha256": "test-train",
                "class_counts": [1, 1, 1, 1],
                "sample_order": "test order",
            }
            first = extract_split(
                model=DummyDSTE(),
                device=torch.device("cpu"),
                split="train",
                source=source,
                output_dir=output,
                batch_size=2,
                checkpoint_every_batches=1,
                max_windows=0,
                resume=False,
            )
            second = extract_split(
                model=DummyDSTE(),
                device=torch.device("cpu"),
                split="train",
                source=source,
                output_dir=output,
                batch_size=2,
                checkpoint_every_batches=1,
                max_windows=0,
                resume=True,
            )
            temporal = np.load(output / "train" / "temporal_features.npy")
            spatial = np.load(output / "train" / "spatial_features.npy")
            cached_starts = np.load(output / "train" / "window_start.npy")
        self.assertTrue(first["integrity"]["passed"])
        self.assertEqual(first, second)
        self.assertTrue(np.all(temporal == 63.0))
        self.assertTrue(np.all(spatial == 49.0))
        np.testing.assert_array_equal(cached_starts, np.arange(4) * 8)


if __name__ == "__main__":
    unittest.main()
