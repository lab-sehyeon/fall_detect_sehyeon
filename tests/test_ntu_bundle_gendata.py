import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_gen.ntu_bundle_gendata import convert_split


class NtuBundleConversionTest(unittest.TestCase):
    def test_flattened_mvc_frames_become_ctvm(self):
        frames = np.arange(2 * 2 * 25 * 3, dtype=np.float32).reshape(2, 150)
        descs = np.array([[1], [1], [1], [1], [1], [1], [2]], dtype=np.int64)
        lengths = np.array([2], dtype=np.int64)

        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            convert_split(
                frames,
                descs,
                lengths,
                np.array([0]),
                out_dir,
                "train",
                overwrite=False,
            )
            converted = np.load(out_dir / "train_data_joint.npy")
            expected = frames.reshape(2, 2, 25, 3).transpose(3, 0, 2, 1)
            np.testing.assert_array_equal(converted[0, :, :2], expected)
            self.assertEqual(converted.shape, (1, 3, 300, 25, 2))
            np.testing.assert_array_equal(
                np.load(out_dir / "train_num_frame.npy"), np.array([2])
            )


if __name__ == "__main__":
    unittest.main()
