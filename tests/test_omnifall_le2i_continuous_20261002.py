import unittest
import numpy as np
from fall_pipeline.external.le2i_current_evaluation import rising_edges,match_events
from scripts.audit_omnifall_le2i_continuous_20261002 import independent_counts
from scripts.prepare_omnifall_le2i_continuous_20261002 import identity


class ContinuousTest(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(identity('Coffee_room_01/video_26'),('Coffee_room_01_026','Coffee_room_01',26))

    def test_rising_not_every_positive(self):
        z=np.eye(4)[[0,1,1,0,1]]
        self.assertEqual([r['frame'] for r in rising_edges([63,71,79,87,95],z)],[71,95])

    def test_duplicates_and_boundaries(self):
        e=[dict(fall_start=3.,fall_end=4.)]
        p=[dict(time=t) for t in [2.5,3.,7.,7.01]]
        m=match_events(p,e)
        self.assertEqual((m['tp'],m['fp'],m['fn']),(1,3,0))
        self.assertEqual(independent_counts([2.5,3.,7.,7.01],[(3.,4.)]),(1,3,0))

    def test_quality_rejection_retains_positive(self):
        self.assertEqual(independent_counts([],[(3.,4.)]),(0,0,1))

    def test_negative_video_counts_all_events(self):
        self.assertEqual(independent_counts([1.,2.],[]),(0,2,0))

    def test_random_scorer_agreement(self):
        rng=np.random.default_rng(0)
        for _ in range(100):
            start=float(rng.uniform(1,10));end=start+float(rng.uniform(.1,5))
            times=sorted(rng.uniform(0,20,size=int(rng.integers(0,10))))
            m=match_events([dict(time=float(t)) for t in times],[dict(fall_start=start,fall_end=end)])
            self.assertEqual(independent_counts(times,[(start,end)]),(m['tp'],m['fp'],m['fn']))

    def test_multiple_gt_not_silently_accepted_by_audit(self):
        with self.assertRaises(RuntimeError):independent_counts([],[(1.,2.),(3.,4.)])


if __name__=='__main__':unittest.main()
