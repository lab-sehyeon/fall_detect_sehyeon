import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from fall_pipeline.common.integrity import hash_named_tensors, sha256_file
from fall_pipeline.safer.eval_f0b_safer_postselection import (
    EXPECTED_COUNTS,
    SPLIT_ORDER,
    atomic_json_dump,
    evaluate_split,
    forward_selected,
    historical_difference,
    load_eval_config,
    validate_materialization_manifests,
    validate_selection_run,
)


class DummyDSTE(torch.nn.Module):
    def backbone(self, temporal_input, spatial_input):
        batch = temporal_input.shape[0]
        temporal = torch.arange(64, dtype=torch.float32).view(1, 64, 1)
        temporal = temporal.expand(batch, 64, 1024)
        spatial = torch.arange(50, dtype=torch.float32).view(1, 50, 1)
        spatial = spatial.expand(batch, 50, 1024)
        return temporal, spatial


def write_source_split(root: Path, split: str, count: int) -> dict:
    split_root = root / split
    split_root.mkdir(parents=True)
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
    payload = {
        name: {
            "bytes": (split_root / name).stat().st_size,
            "array_payload_sha256": f"test-{split}-{name}",
        }
        for name in (
            "data_joint.npy",
            "center_coarse_labels.npy",
            "center_derived_labels.npy",
            "sequence_index.npy",
            "window_start.npy",
        )
    }
    manifest = {
        "written_windows": count,
        "data_shape": [count, 3, 64, 25, 2],
        "data_dtype": "float32",
        "candidate": {"name": "legacy_coco_umurl_window_noscale"},
        "payload": payload,
        "integrity": {"passed": True},
    }
    manifest_path = split_root / "split_manifest.json"
    atomic_json_dump(manifest_path, manifest)
    return {
        "root": str(split_root),
        "count": count,
        "data_path": str(split_root / "data_joint.npy"),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_data_payload_sha256": payload["data_joint.npy"][
            "array_payload_sha256"
        ],
    }


class F0BSaferPostselectionTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.config_path = (
            root / "configs/f0b_safer_v1_recovery_postselection_v1.json"
        )
        self.config, _ = load_eval_config(self.config_path)

    def test_preregistered_config_locks_postselection_only(self):
        self.assertEqual(tuple(self.config["evaluation"]["split_order"]), SPLIT_ORDER)
        self.assertFalse(self.config["evaluation"]["training"])
        self.assertIsNone(self.config["evaluation"]["optimizer"])
        self.assertFalse(self.config["evaluation"]["threshold_fitting"])
        self.assertFalse(self.config["evaluation"]["candidate_comparison"])
        self.assertFalse(self.config["evaluation"]["checkpoint_reselection"])
        self.assertEqual(self.config["selection_lock"]["candidate"], "temporal_spatial")
        self.assertEqual(self.config["selection_lock"]["epoch"], 46)

    def test_changed_config_is_rejected(self):
        changed = copy.deepcopy(self.config)
        changed["selection_lock"]["epoch"] = 50
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "changed.json"
            atomic_json_dump(path, changed)
            with self.assertRaisesRegex(RuntimeError, "hash differs"):
                load_eval_config(path)

    def test_strict_json_writer_rejects_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            with self.assertRaises(ValueError):
                atomic_json_dump(path, {"metric": float("inf")})
            self.assertFalse(path.exists())

    def test_selection_gate_accepts_only_frozen_validation_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = {
                "weight": torch.zeros((4, 2048), dtype=torch.float32),
                "bias": torch.zeros(4, dtype=torch.float32),
            }
            head_path = root / "selected_head.pth"
            torch.save(state, head_path)
            state_hash = hash_named_tensors(state.items())
            head_sha256 = sha256_file(head_path)
            config = {
                "selection_lock": {
                    "candidate": "temporal_spatial",
                    "epoch": 46,
                    "validation_primary_value": 0.7,
                    "validation_secondary_value": 0.96,
                    "selected_head_sha256": head_sha256,
                    "selected_head_state_hash": state_hash,
                },
                "input": {
                    "f0b_training_config_sha256": "training-config",
                    "f0b_trainer_sha256": "trainer",
                },
            }
            required = {
                "passed": True,
                "feature_cache_gate_passed": True,
                "physical_gpu_zero_only": True,
                "explicit_recovery_config_frozen": True,
                "only_two_linear_heads_trainable": True,
                "same_minibatch_order_for_candidates": True,
                "unweighted_cross_entropy": True,
                "full_validation_each_epoch": True,
                "selection_uses_validation_only": True,
                "test_ood_unopened": True,
                "all_50_epochs_completed": True,
            }
            report = {
                "status": "completed",
                "research_usable": True,
                "run_contract": {
                    "mode": "full",
                    "test_ood_access": False,
                    "config_sha256": "training-config",
                    "trainer_sha256": "trainer",
                },
                "selection": {
                    "candidate": "temporal_spatial",
                    "epoch": 46,
                    "metrics": {
                        "macro_f1": 0.7,
                        "conditional_fall_vs_lie_auprc": 0.96,
                    },
                    "frozen_before_test_ood": True,
                },
                "selected_head_sha256": head_sha256,
                "candidates": {
                    "temporal_spatial": {
                        "best_epoch": 46,
                        "head_state_hash": state_hash,
                    }
                },
                "integrity": required,
            }
            report_path = root / "f0b_head_report.json"
            atomic_json_dump(report_path, report)
            config["selection_lock"]["training_report_sha256"] = sha256_file(
                report_path
            )
            validated = validate_selection_run(root, config)
            self.assertEqual(validated["selected_head_state_hash"], state_hash)
            report["integrity"]["test_ood_unopened"] = False
            atomic_json_dump(report_path, report)
            config["selection_lock"]["training_report_sha256"] = sha256_file(
                report_path
            )
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                validate_selection_run(root, config)

    def test_materialization_preflight_does_not_open_payload_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            counts = {"test": 4, "ood": 4}
            entries = {}
            with patch.dict(EXPECTED_COUNTS, counts, clear=True):
                for split, count in counts.items():
                    source = write_source_split(root, split, count)
                    entries[split] = {
                        "manifest_sha256": source["source_manifest_sha256"]
                    }
                manifest = {
                    "status": "completed",
                    "research_usable": True,
                    "candidate": {"name": "legacy_coco_umurl_window_noscale"},
                    "recovery_status": {
                        "experimental_compatibility_reconstruction": True,
                        "must_not_be_called_byte_identical_reproduction": True,
                    },
                    "protocol": {
                        "window_size": 64,
                        "test_and_ood_used_for_candidate_selection": False,
                        "test_and_ood_materialized_only_after_structural_freeze": True,
                    },
                    "splits": entries,
                    "integrity": {"passed": True},
                }
                root_manifest = root / "materialization_manifest.json"
                atomic_json_dump(root_manifest, manifest)
                config = copy.deepcopy(self.config)
                config["input"]["splits"] = counts
                config["input"]["materialization_manifest_sha256"] = sha256_file(
                    root_manifest
                )
                validated = validate_materialization_manifests(root, config)
            self.assertFalse(validated["payload_arrays_opened"])
            self.assertEqual(set(validated["splits"]), set(SPLIT_ORDER))

    def test_selected_forward_and_resumable_full_split_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_source_split(root, "test", 4)
            output = root / "output"
            head = torch.nn.Linear(2048, 4)
            head.requires_grad_(False)
            batch = torch.zeros((2, 3, 64, 25, 2), dtype=torch.float32)
            temporal, spatial, logits = forward_selected(DummyDSTE(), head, batch)
            self.assertEqual(tuple(temporal.shape), (2, 1024))
            self.assertEqual(tuple(spatial.shape), (2, 1024))
            self.assertEqual(tuple(logits.shape), (2, 4))
            config = copy.deepcopy(self.config)
            config["evaluation"]["batch_size"] = 2
            config["execution"]["checkpoint_every_batches"] = 1
            first = evaluate_split(
                model=DummyDSTE(),
                head=head,
                device=torch.device("cpu"),
                split="test",
                source=source,
                output_dir=output,
                config=config,
                resume=False,
            )
            second = evaluate_split(
                model=DummyDSTE(),
                head=head,
                device=torch.device("cpu"),
                split="test",
                source=source,
                output_dir=output,
                config=config,
                resume=True,
            )
            saved_logits = np.load(output / "test" / "logits.npy")
            saved_starts = np.load(output / "test" / "window_start.npy")
            self.assertEqual(first, second)
            self.assertTrue(first["integrity"]["passed"])
            self.assertEqual(saved_logits.shape, (4, 4))
            np.testing.assert_array_equal(saved_starts, np.arange(4) * 8)

    def test_historical_difference_is_percentage_points(self):
        metrics = {
            "macro_f1": 0.55584,
            "fall_f1": 0.55242,
            "conditional_fall_vs_lie_auprc": 0.96807,
        }
        difference = historical_difference("test", metrics, self.config)
        for value in difference.values():
            self.assertAlmostEqual(value, 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
