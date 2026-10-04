import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fall_pipeline.common.fall_safer_zeroshot_eval import (
    EXPECTED_COUNTS,
    historical_difference,
    validate_materialization,
)
from fall_pipeline.common.integrity import sha256_file


class SaferF0AZeroShotTest(unittest.TestCase):
    def test_historical_difference_uses_percentage_points(self):
        metrics = {
            "f1": 0.08989,
            "auprc": 0.08249,
            "fall_vs_lie_auprc": 0.61446,
        }
        difference = historical_difference("val", metrics)
        self.assertAlmostEqual(difference["f1"], 1.0)
        self.assertAlmostEqual(difference["auprc"], 0.0)
        self.assertAlmostEqual(difference["fall_vs_lie_auprc"], 0.0)

    def test_materialization_gate_accepts_only_completed_frozen_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_entries = {}
            counts = {"val": 2, "test": 3, "ood": 1}
            with patch.dict(EXPECTED_COUNTS, counts, clear=True):
                for split, count in counts.items():
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
                    np.save(
                        split_root / "center_coarse_labels.npy",
                        np.zeros(count, dtype=np.int64),
                    )
                    data_path = split_root / "data_joint.npy"
                    split_manifest = {
                        "written_windows": count,
                        "candidate": {"name": "legacy_coco_umurl_window_noscale"},
                        "payload": {
                            "data_joint.npy": {
                                "bytes": data_path.stat().st_size,
                                "array_payload_sha256": "test-payload",
                            }
                        },
                        "integrity": {"passed": True},
                    }
                    split_manifest_path = split_root / "split_manifest.json"
                    split_manifest_path.write_text(json.dumps(split_manifest))
                    split_entries[split] = {
                        "manifest_sha256": sha256_file(split_manifest_path)
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
                        "test_and_ood_materialized_only_after_structural_freeze": True,
                    },
                    "splits": split_entries,
                    "integrity": {"passed": True},
                }
                (root / "materialization_manifest.json").write_text(json.dumps(manifest))
                validated = validate_materialization(root)
            self.assertEqual(validated["splits"]["test"]["count"], 3)

    def test_materialization_gate_rejects_research_unusable_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "materialization_manifest.json").write_text(json.dumps({
                "status": "smoke_completed",
                "research_usable": False,
                "integrity": {"passed": True},
            }))
            with self.assertRaisesRegex(RuntimeError, "status is not completed"):
                validate_materialization(root)


if __name__ == "__main__":
    unittest.main()
