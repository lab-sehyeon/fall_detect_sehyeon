import unittest
import numpy as np

from scripts.evaluate_usdrl_external_20261004 import edges, summary
from scripts.audit_usdrl_external_20261004 import counts, timestamps
from fall_pipeline.external.le2i_current_evaluation import match_events
from fall_pipeline.external.rgb_document_io import windows


class USDRLExternalTests(unittest.TestCase):
    def logits(self, labels):
        z = np.zeros((len(labels), 60))
        for i, label in enumerate(labels):
            z[i, label] = 1.
        return z

    def test_ntu_fall_is_zero_based_42_not_43_or_g0_index(self):
        z = self.logits([1, 43, 42, 42, 0, 42])
        self.assertEqual(edges(np.arange(63, 104, 8), z), [dict(frame=79, time=3.16), dict(frame=103, time=4.12)])

    def test_first_window_fall_is_an_alarm(self):
        self.assertEqual(edges([63,71], self.logits([42,42])), [dict(frame=63,time=2.52)])

    def test_zero_windows(self):
        self.assertEqual(edges([],np.empty((0,60))), [])

    def test_tied_logits_use_first_argmax(self):
        self.assertEqual(edges([63],np.zeros((1,60))), [])

    def test_wrong_logits_shape_and_nonfinite_rejected(self):
        with self.assertRaises(RuntimeError): edges([63], np.zeros((1,4)))
        with self.assertRaises(RuntimeError): edges([63], np.full((1,60),np.nan))
        with self.assertRaises(RuntimeError): edges([71,63], self.logits([42,42]))

    def test_independent_edges(self):
        rng=np.random.default_rng(42)
        labels=rng.integers(0,3,100)*21
        endpoints=np.arange(100)*8+63
        self.assertEqual([p['time'] for p in edges(endpoints,self.logits(labels))],timestamps(labels,endpoints,42))

    def test_event_boundaries_and_duplicate_alarms(self):
        ep=[dict(fall_start=4.,fall_end=5.)]
        self.assertEqual(counts([3.5,8.],ep),(1,1,0))
        self.assertEqual(counts([3.499,8.001],ep),(0,2,1))

    def test_independent_matching_random_cases(self):
        rng=np.random.default_rng(0)
        for _ in range(50):
            ep=[dict(fall_start=float(i),fall_end=float(i+1)) for i in [2,4,10]]
            times=rng.uniform(0,15,10).tolist()
            ref=match_events([dict(time=t) for t in times],ep)
            self.assertEqual(counts(times,ep),tuple(ref[k] for k in ('tp','fp','fn')))

    def test_rejected_positive_stays_in_denominator(self):
        ep=[dict(fall_start=2.,fall_end=4.)]
        rows=[dict(episodes=ep,processed=False,video_prediction=0,event=dict(tp=0,fp=0,fn=1)),
              dict(episodes=[],processed=True,video_prediction=0,event=dict(tp=0,fp=0,fn=0))]
        s=summary(rows)
        self.assertEqual((s['videos'],s['event']['fn'],s['video']['tn'],s['video']['accuracy']),(2,1,1,.5))

    def test_window_layout_and_second_person_zero(self):
        ntu=np.arange(72*25*3,dtype=np.float32).reshape(72,25,3)
        out=windows(ntu,[0,8])
        self.assertEqual(out.shape,(2,3,64,25,2))
        np.testing.assert_array_equal(out[1,...,0],ntu[8:72].transpose(2,0,1))
        self.assertFalse(out[...,1].any())


if __name__ == '__main__':
    unittest.main()
