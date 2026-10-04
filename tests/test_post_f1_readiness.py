import tempfile
from pathlib import Path
import unittest

import numpy as np

from scripts.audit_post_f1_readiness import digest, stage_inventory, validate_fu_arrays, verify_file


class PostF1ReadinessTest(unittest.TestCase):
    def fixture(self):
        frames = np.asarray([35, 64, 65, 72])
        data = np.zeros((4, 3, 300, 25, 2), dtype=np.float32)
        for index, length in enumerate(frames):
            data[index, :, :length, :, 0] = 1
        actions = np.asarray([0, 5, 4, 1])
        labels = (actions == 5).astype(np.int64)
        subjects = np.asarray([1, 2, 3, 4])
        return data, frames, labels, actions, subjects, (subjects - 1) % 5

    def test_tail_policy_is_reported_not_chosen(self):
        result = validate_fu_arrays(*self.fixture())
        self.assertEqual(result["native30_regular_stride8_windows"], 5)
        self.assertEqual(result["native30_with_end_anchor_windows"], 6)
        self.assertEqual(result["shorter_than_64"], 1)
        self.assertTrue(result["window_counts_are_policy_comparison_not_selected_protocol"])

    def test_invalid_padding_is_rejected(self):
        values = self.fixture()
        values[0][0, 0, 35, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "padding"):
            validate_fu_arrays(*values)

    def test_wrong_fold_is_rejected(self):
        values = self.fixture()
        values[-1][0] = 4
        with self.assertRaisesRegex(ValueError, "fold"):
            validate_fu_arrays(*values)

    def test_nonfinite_is_rejected(self):
        values = self.fixture()
        values[0][0, 0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            validate_fu_arrays(*values)

    def test_file_hash_and_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload"
            path.write_bytes(b"frozen")
            expected = digest(path)
            self.assertEqual(verify_file(path, expected, 6)["bytes"], 6)
            path.write_bytes(b"edited")
            with self.assertRaisesRegex(ValueError, "hash"):
                verify_file(path, expected, 6)

    def test_modules_do_not_authorize_experiments(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = stage_inventory(Path(directory))
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(not row["experiment_execution_ready"] for row in rows))
            self.assertTrue(all(row["unresolved"] for row in rows))


if __name__ == "__main__":
    unittest.main()
