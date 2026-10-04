import unittest
import numpy as np
from fall_pipeline.external.le2i_current_evaluation import rising_edges, match_events, metrics
from fall_pipeline.external.rgb_recovery_common import decode_recovery_endpoints, D1_DECOUPLED
from data_gen.rgb_alignment_reconstruction import past_frame_indices


class Le2iEvaluationTests(unittest.TestCase):
    def test_rising_edges_equal_existing_d1(self):
        frames = np.arange(63, 63+8*10, 8)
        classes = np.array([1, 1, 0, 1, 2, 3, 1, 1, 0, 1])
        logits = np.eye(4)[classes]
        events = rising_edges(frames, logits)
        reference = decode_recovery_endpoints(frames, logits, np.zeros(len(frames)), fall_alert_mode=D1_DECOUPLED)
        reference = [{k: e[k] for k in ('frame', 'time')} for e in reference['events'] if e['type'] == 'fall_trigger']
        self.assertEqual(events, reference)
        self.assertEqual(len(events), 4)

    def test_no_events(self):
        self.assertEqual(rising_edges([], np.empty((0, 4))), [])
        self.assertEqual(match_events([], [])['f1'], 0)

    def test_quality_failure_stays_fn(self):
        result = match_events([], [dict(fall_start=2., fall_end=3.)])
        self.assertEqual((result['tp'], result['fp'], result['fn']), (0, 0, 1))

    def test_boundary_and_duplicate_fp(self):
        episode = [dict(fall_start=3., fall_end=4.)]
        for t in (2.5, 7.):
            self.assertEqual(match_events([dict(time=t)], episode)['tp'], 1)
        for t in (2.49, 7.01):
            result = match_events([dict(time=t)], episode)
            self.assertEqual((result['tp'], result['fp'], result['fn']), (0, 1, 1))
        result = match_events([dict(time=2.5), dict(time=3.), dict(time=7.)], episode)
        self.assertEqual((result['tp'], result['fp'], result['fn']), (1, 2, 0))

    def test_alert_before_window_cannot_hide_later_match(self):
        result = match_events([dict(time=1.), dict(time=3.)], [dict(fall_start=3., fall_end=4.)])
        self.assertEqual((result['tp'], result['fp'], result['fn']), (1, 1, 0))

    def test_published_metrics_from_counts(self):
        self.assertAlmostEqual(metrics(79, 14, 17)['f1']*100, 83.5978835978836)
        self.assertAlmostEqual(metrics(80, 14, 16)['f1']*100, 84.21052631578947)

    def test_current_no_tail_and_past_only(self):
        self.assertEqual(np.arange(0, 70-63, 8).tolist(), [0])
        self.assertEqual(np.arange(0, 195-63, 8).tolist()[-1], 128)
        pts = np.round(np.arange(240)/24, 3)
        times, indices = past_frame_indices(pts, 10.)
        self.assertEqual(len(times), 250)
        self.assertTrue(np.all(pts[indices] <= times))
        self.assertTrue(np.all(indices < 240))

    def test_independent_single_event_audit(self):
        from scripts.audit_le2i_current_evaluation import independent_counts
        generator = np.random.default_rng(9)
        for fps in (25., 500000/20833):
            for n in range(12):
                times = sorted(generator.uniform(0, 10, n))
                for header in ([0, 0], [48, 80]):
                    episodes = [] if header == [0, 0] else [dict(fall_start=header[0]/fps, fall_end=header[1]/fps)]
                    result = match_events([dict(time=float(t)) for t in times], episodes)
                    self.assertEqual(independent_counts(times, header, fps), tuple(result[k] for k in ('tp', 'fp', 'fn')))

    def test_independent_metric_audit_rejects_changed_f1(self):
        from scripts.audit_le2i_current_evaluation import validate_metrics
        original = metrics(79, 14, 17)
        self.assertEqual(validate_metrics((79, 14, 17), original), original)
        with self.assertRaises(RuntimeError):
            validate_metrics((79, 14, 17), {**original, 'f1': 1.})


if __name__ == '__main__': unittest.main()
