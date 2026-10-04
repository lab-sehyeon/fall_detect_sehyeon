import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data_gen.ntu_gendata import gendata
from options.options_downstream import opts_ntu_60_cross_subject


def write_skeleton(path: Path, frame_count: int = 2) -> None:
    lines = [str(frame_count)]
    for frame in range(frame_count):
        lines.extend(["1", "1 0 0 0 0 0 0 0 0 2", "25"])
        for joint in range(25):
            x = frame + joint / 100.0
            y = joint / 50.0
            z = 1.0 + frame / 10.0
            lines.append("{} {} {} 0 0 0 0 1 0 0 0 2".format(x, y, z))
    path.write_text("\n".join(lines) + "\n")


class NtuGendataTest(unittest.TestCase):
    def test_downstream_official_umurl_profile(self) -> None:
        options = opts_ntu_60_cross_subject(data_profile="official_umurl")
        self.assertEqual(options.data_profile, "official_umurl")
        self.assertTrue(options.train_feeder_args["data_path"].endswith(
            "/NTU-RGB-D-60-AGCN-UMURL/xsub/train_data_joint.npy"))
        self.assertTrue(options.test_feeder_args["data_path"].endswith(
            "/NTU-RGB-D-60-AGCN-UMURL/xsub/val_data_joint.npy"))

    def test_downstream_profile_rejects_unknown_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown NTU60 XSub data profile"):
            opts_ntu_60_cross_subject(data_profile="unknown")

    def test_joint_memmap_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw"
            output = root / "output"
            raw.mkdir()
            output.mkdir()
            sample_name = "S001C001P001R001A001.skeleton"
            write_skeleton(raw / sample_name)

            count = gendata(
                str(raw),
                str(output),
                benchmark="xsub",
                part="train",
                dataset="ntu60",
                normalization_profile="official_umurl",
                expected_count=1,
                modalities=("joint",),
            )

            self.assertEqual(count, 1)
            joint = np.load(output / "train_data_joint.npy", mmap_mode="r")
            frames = np.load(output / "train_num_frame.npy")
            with (output / "train_label.pkl").open("rb") as stream:
                names, labels = pickle.load(stream)
            self.assertEqual(joint.shape, (1, 3, 300, 25, 2))
            self.assertEqual(frames.tolist(), [2])
            self.assertEqual(names, [sample_name])
            self.assertEqual(labels, [0])
            self.assertTrue(np.isfinite(joint).all())
            self.assertFalse((output / "train_data_motion.npy").exists())
            self.assertFalse((output / "train_data_bone.npy").exists())

            np.testing.assert_allclose(joint[0, :, :2, 1, 0], 0.0, atol=1e-6)
            shoulder = joint[0, :, 0, 8, 0] - joint[0, :, 0, 4, 0]
            self.assertGreater(float(shoulder[0]), 0.0)
            np.testing.assert_allclose(shoulder[1:], 0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
