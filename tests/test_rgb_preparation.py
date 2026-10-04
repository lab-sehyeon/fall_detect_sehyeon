from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_rgb_recovery_data_r2 as run


class PreparationTests(unittest.TestCase):
    labels = {1: 'fall', 8: 'standing', 9: 'other'}

    def row(self, start, end, label, index):
        return dict(start=start, end=end, label=label, source_row=index)

    def test_specific_subtracts_other_and_preserves_sources(self):
        rows = [self.row('0', '10', 9, 2), self.row('2', '3', 1, 3),
                self.row('5', '8', 8, 4)]
        original = [r.copy() for r in rows]
        normalized, audit = run.normalize_intervals(rows, self.labels)
        self.assertEqual([(r['start'], r['end']) for r in normalized],
                         [('0', '2'), ('2', '3'), ('3', '5'), ('5', '8'), ('8', '10')])
        self.assertEqual(audit['overlap_pairs'], 2)
        self.assertEqual(audit['removed_other_seconds'], '4')
        self.assertEqual(rows, original)

    def test_decimal_tiny_overlap_not_dropped_or_rounded(self):
        normalized, audit = run.normalize_intervals([
            self.row('0', '1.015', 9, 2), self.row('1.000', '2', 1, 3)], self.labels)
        self.assertEqual(audit['removed_other_seconds'], '0.015')
        self.assertEqual(normalized[0]['end'], '1.000')

    def test_invalid_or_specific_overlaps_fail(self):
        for labels in ((1, 8), (1, 1), (9, 9)):
            with self.assertRaisesRegex(RuntimeError, 'overlap'):
                run.normalize_intervals([self.row(0, 2, labels[0], 2),
                                         self.row(1, 3, labels[1], 3)], self.labels)
        for start, end, label in [('NaN', 2, 1), (0, 'Infinity', 1),
                                  (-1, 2, 1), (2, 2, 1), (3, 2, 1), (0, 1, 99)]:
            with self.assertRaises(RuntimeError):
                run.normalize_intervals([self.row(start, end, label, 2)], self.labels)

    def test_touching_and_same_class_boundaries_preserved(self):
        normalized, audit = run.normalize_intervals([
            self.row(0, 1, 1, 2), self.row(1, 2, 1, 3)], self.labels)
        self.assertEqual(len(normalized), 2)
        self.assertEqual(audit['overlap_pairs'], 0)

    def test_fallback_only_for_last_seek(self):
        with patch.object(run.source, 'video_probe', side_effect=RuntimeError('video last frame decode failed')), \
                patch.object(run, 'sequential_probe', return_value={'full_frame_decode': True}) as probe:
            result = run.robust_probe(Path('unused'))
            self.assertTrue(result['full_frame_decode'])
            self.assertFalse(result['random_seek_last_frame'])
            probe.assert_called_once()
        with patch.object(run.source, 'video_probe', side_effect=RuntimeError('video open failed')), \
                patch.object(run, 'sequential_probe') as probe:
            with self.assertRaisesRegex(RuntimeError, 'open failed'):
                run.robust_probe(Path('unused'))
            probe.assert_not_called()

    def test_sequential_count_and_shape_gate(self):
        import cv2
        import numpy as np
        class Capture:
            def __init__(self, actual, shape=(2, 3, 3)):
                self.actual, self.shape, self.i = actual, shape, 0
            def isOpened(self): return True
            def get(self, key):
                return {cv2.CAP_PROP_FRAME_COUNT: 4, cv2.CAP_PROP_FPS: 25,
                        cv2.CAP_PROP_FRAME_WIDTH: 3, cv2.CAP_PROP_FRAME_HEIGHT: 2}[key]
            def read(self):
                self.i += 1
                return (True, np.zeros(self.shape, dtype=np.uint8)) if self.i <= self.actual else (False, None)
            def release(self): pass
        with patch.object(cv2, 'VideoCapture', return_value=Capture(4)):
            self.assertEqual(run.sequential_probe(Path('unused'), check=lambda: None)['decoded_frames'], 4)
        with patch.object(cv2, 'VideoCapture', return_value=Capture(3)):
            with self.assertRaisesRegex(RuntimeError, 'frame count'):
                run.sequential_probe(Path('unused'), check=lambda: None)
        with patch.object(cv2, 'VideoCapture', return_value=Capture(4, (3, 3, 3))):
            with self.assertRaisesRegex(RuntimeError, 'shape'):
                run.sequential_probe(Path('unused'), check=lambda: None)

    def test_old_and_new_pause(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(run, 'OUTPUT', Path(folder)), \
                patch.object(run.source, 'pause_space') as check:
            run.safety(12)
            check.assert_called_with(12)
            (Path(folder) / 'PAUSE_REQUESTED').touch()
            with self.assertRaisesRegex(RuntimeError, 'pause'):
                run.safety()


if __name__ == '__main__':
    unittest.main()
