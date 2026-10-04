import copy
import json
from pathlib import Path
import unittest
import numpy as np
from fall_pipeline.safer import s0e_core as e

CONFIG = json.loads((Path(__file__).resolve().parents[1]/'configs/s0e_document_reconstruction_v1.json').read_text())


class HardStateTests(unittest.TestCase):
    def setUp(self):
        self.spec = {'fall_settle': 3, 'recovery_confirmation': 2, 'recovery_hold': 3, 'sit_confirmation': 3}

    def decode(self, values):
        return e.decode(values, self.spec, CONFIG['rules']).tolist()

    def test_grid(self):
        rows = e.grid(CONFIG)
        self.assertEqual(len(rows), 72)
        self.assertEqual(len({x['name'] for x in rows}), 72)
        self.assertEqual(rows[0]['name'], 'fall08__rec03__hold05__sit00')

    def test_settle_and_recovery_hold(self):
        self.assertEqual(self.decode([10,11,11,11,11,3,3,3,3,3]), [10,11,11,12,12,3,7,7,7,3])

    def test_no_fall_no_lie_conversion(self):
        self.assertEqual(self.decode([11]*10), [11]*10)

    def test_stable_lying_without_fall(self):
        self.assertEqual(self.decode([12,12,12,3,3,3,3,3]), [12,12,12,3,7,7,7,3])

    def test_new_fall_interrupts_hold(self):
        self.assertEqual(self.decode([10,11,11,11,3,3,10,11]), [10,11,11,12,3,7,10,11])

    def test_ignore_is_not_confirmation(self):
        self.assertEqual(self.decode([10,11,11,11,3,0,3,3]), [10,11,11,12,3,0,3,7])

    def test_sit_confirmation(self):
        self.assertEqual(self.decode([6,4,4,4,4,3,4]), [6,6,6,4,4,3,4])
        self.spec['sit_confirmation'] = 0
        self.assertEqual(self.decode([6,4,4]), [6,4,4])

    def test_prefix_fall_mask_sequence_reset(self):
        rng = np.random.default_rng(0)
        values = rng.integers(0,16,1000)
        expected = e.decode(values,self.spec,CONFIG['rules'])
        for cut in range(0,1000,17):
            np.testing.assert_array_equal(expected[:cut],e.decode(values[:cut],self.spec,CONFIG['rules']))
        np.testing.assert_array_equal(expected==10,values==10)
        self.assertEqual(self.decode([11,11,11]), [11,11,11])

    def test_bad_input(self):
        for values in ([16],[-1],[1.5],[[1,2]]):
            with self.assertRaises(ValueError):self.decode(values)

    def test_selection_eligible_then_fallback_ties(self):
        def row(name, eligible, count, macro):
            return {'name':name,'gate':{'eligible':eligible,'passed_count':count},'metrics':{'macro_f1':macro,'segment_f1_50':0.4,'edit':0.5}}
        rows=[row('high',False,7,0.9),row('eligible',True,8,0.6)]
        self.assertEqual(e.choose(rows)['name'],'eligible')
        self.assertTrue(e.choose(rows)['adoption'])
        self.assertEqual(e.choose([row('first',False,5,0.6),row('second',False,5,0.6)])['name'],'first')
        self.assertFalse(e.choose(rows[:1])['adoption'])

    def test_eight_gates_and_no_events(self):
        base={'macro_f1':.6,'switches_per_minute':10.,'per_class':[{'f1':.5} for _ in range(16)]}
        lat={p:{'events':100,'detected':40,'premature':20} for p in ('10->12','6->4','12->7')}
        improved=copy.deepcopy(lat)
        for v in improved.values():v.update(detected=55,premature=10)
        result=e.gates(base,base,improved,lat,True,CONFIG['evaluation'])
        self.assertEqual(result['passed_count'],8);self.assertTrue(result['eligible'])
        improved['10->12']['events']=0
        self.assertFalse(e.gates(base,base,improved,lat,True,CONFIG['evaluation'])['eligible'])


if __name__ == '__main__':unittest.main()
