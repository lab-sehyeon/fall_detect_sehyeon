import os
import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data_gen.inspect_safer_activities import (
    RestrictedNumpyUnpickler,
    audit_annotations,
    sequence_base,
)


class _UnsafePayload:
    def __reduce__(self):
        return os.system, ("true",)


def annotation(frames: int = 3):
    return {
        "labels": np.arange(frames, dtype=np.int64),
        "width": 1920,
        "height": 1080,
        "bboxes": np.zeros((frames, 4), dtype=np.float32),
        "bbox_format": "xyxy",
        "full_labels": np.arange(frames, dtype=np.int64),
        "frame_dir": "example_d01.mp4",
        "keypoint": np.zeros((1, frames, 17, 2), dtype=np.float32),
        "keypoint_score": np.ones((1, frames, 17), dtype=np.float32),
        "img_shape": (1080, 1920),
        "total_frames": frames,
        "keypoint_3d": np.zeros((1, frames, 17, 3), dtype=np.float32),
        "keypoint_3d_score": np.ones((1, frames), dtype=np.float32),
    }


class SaferInspectorTest(unittest.TestCase):
    def test_sequence_base_rules(self):
        self.assertEqual(sequence_base("abc_d08.mp4", "normal"), "abc")
        self.assertEqual(sequence_base("Day_02_P01_camera_06", "ood"), "day_02_p01")
        self.assertEqual(sequence_base("day_normal_p05_cam8", "ood"), "day_normal_p05")

    def test_restricted_unpickler_blocks_other_globals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.pkl"
            path.write_bytes(pickle.dumps(_UnsafePayload(), protocol=4))
            with path.open("rb") as stream:
                with self.assertRaises(pickle.UnpicklingError):
                    RestrictedNumpyUnpickler(stream).load()

    def test_annotation_contract_and_nonfinite_count(self):
        item = annotation()
        item["keypoint_3d"][0, 1] = np.nan
        report, names, bases = audit_annotations([item], "normal")
        self.assertEqual(names, {"example_d01"})
        self.assertEqual(bases, {"example"})
        self.assertEqual(report["nonfinite"]["legacy_3d_sequences"], 1)
        self.assertEqual(report["nonfinite"]["legacy_3d_frames"], 1)
        self.assertTrue(report["checks"]["required_shapes"])
        self.assertTrue(report["checks"]["required_dtypes"])

    def test_official_mixed_float_dtypes_are_allowed(self):
        first = annotation()
        second = annotation()
        second["frame_dir"] = "second_d02.mp4"
        for field in ("keypoint", "keypoint_score", "keypoint_3d", "keypoint_3d_score", "bboxes"):
            second[field] = second[field].astype(np.float64)
        report, _, _ = audit_annotations([first, second], "normal")
        self.assertTrue(report["checks"]["required_dtypes"])
        self.assertEqual(report["dtype_counts"]["keypoint"], {"float32": 1, "float64": 1})


if __name__ == "__main__":
    unittest.main()
