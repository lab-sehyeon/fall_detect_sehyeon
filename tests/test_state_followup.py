import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from fall_pipeline.safer import state_followup_core as c
from fall_pipeline.safer import run_s0b_reconstruction as b


class StateFollowupTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.bc=b.io.read(b.CONFIG)

    def head(self):
        base=nn.Linear(2048,16)
        return c.ResidualStateHead(base.state_dict(),{**self.bc['model'],'hidden':8})

    def test_zero_correction_exact_and_frozen_base(self):
        h=self.head();x=torch.randn(3,64,2048)
        for training in (True,False):
            h.train(training)
            self.assertTrue(torch.equal(h(x),h.base(x)))
        h.train();h(x).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in h.base.parameters()))
        self.assertTrue(all(p.grad is not None for p in h.parameters() if p.requires_grad))

    def test_weighted_microbatch_matches_effective_batch(self):
        torch.manual_seed(3)
        first=nn.Linear(2048,16);second=copy.deepcopy(first)
        opt=torch.optim.SGD(first.parameters(),lr=.01)
        ref=torch.optim.SGD(second.parameters(),lr=.01)
        x=np.random.default_rng(2).normal(size=(7,64,2048)).astype(np.float32)
        y=np.random.default_rng(1).integers(0,16,(7,64));weights=torch.linspace(.1,2,16)
        with patch.object(b.s.core,'dense_features',side_effect=lambda model,z:z):
            num,den,_=b.micro_step(first,opt,None,x,y,weights,torch.device('cpu'),{'microbatch_size':3,'gradient_clip':1.})
        loss=nn.functional.cross_entropy(second(torch.from_numpy(x)).flatten(0,1),torch.from_numpy(y).flatten(),weight=weights)
        loss.backward();nn.utils.clip_grad_norm_(second.parameters(),1.);ref.step()
        self.assertAlmostEqual(num/den,loss.item(),places=5)
        for p,q in zip(first.parameters(),second.parameters()):torch.testing.assert_close(p,q,atol=1e-7,rtol=1e-6)

    def test_dropout_rng_restore_replays_training(self):
        h=self.head().train();x=torch.randn(2,64,2048);y=torch.randint(16,(2,64));opt=torch.optim.AdamW((p for p in h.parameters() if p.requires_grad),lr=.001)
        weights=copy.deepcopy(h.state_dict());saved_opt=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state()
        def step():
            opt.zero_grad();nn.functional.cross_entropy(h(x).flatten(0,1),y.flatten()).backward();opt.step()
        step();result=copy.deepcopy(h.state_dict())
        h.load_state_dict(weights);opt.load_state_dict(saved_opt);torch.set_rng_state(rng);step()
        for key,value in h.state_dict().items():self.assertTrue(torch.equal(result[key],value),key)

    def test_b_gate_requires_all_five(self):
        base={'macro_f1':.6,'segment_f1_50':.1,'edit':.2,'switches_per_minute':100.,'per_class':[{'recall':.7}]*16}
        candidate={**base,'segment_f1_50':.13,'edit':.23,'switches_per_minute':70.}
        self.assertTrue(c.adoption_b(candidate,base,self.bc['evaluation'])['adoption'])
        self.assertFalse(c.adoption_b({**candidate,'switches_per_minute':76.},base,self.bc['evaluation'])['adoption'])
        self.assertFalse(c.adoption_b({**candidate,'macro_f1':.59},base,self.bc['evaluation'])['adoption'])

    def test_grid_22_per_source(self):
        cfg=b.io.read(b.io.ROOT/'configs/s0c_document_reconstruction_v1.json');grid=c.decoder_grid(cfg)
        self.assertEqual(len(grid),22);self.assertEqual(len({s['name'] for s in grid}),22)

    def test_raw_and_decoder_prefix_invariance(self):
        x=np.random.default_rng(2).normal(size=(100,16)).astype(np.float32)
        cfg=b.io.read(b.io.ROOT/'configs/s0c_document_reconstruction_v1.json')
        for spec in c.decoder_grid(cfg):np.testing.assert_array_equal(c.decode(x,spec)[:51],c.decode(x[:51],spec))
        np.testing.assert_array_equal(c.decode(x,c.decoder_grid(cfg)[0]),x.argmax(1))

    def test_hysteresis_confirmation_is_not_retroactive(self):
        x=np.full((10,16),-100.);x[:4,0]=100.;x[4:,1]=100.
        pred=c.decode(x,{'alpha':0.,'duration':3,'margin':.05})
        np.testing.assert_array_equal(pred,[0,0,0,0,0,0,1,1,1,1])

    def test_ema_alpha_is_past_weight(self):
        x=np.full((5,16),-100.);x[0,0]=100.;x[1:,1]=100.
        pred=c.decode(x,{'alpha':.85,'duration':1,'margin':0.})
        np.testing.assert_array_equal(pred,[0,0,0,0,0])

    def test_decoder_selection_eligible_before_score(self):
        weak={'source':'a','gate':{'eligible':True},'metrics':{'segment_f1_50':.2,'edit':.3,'macro_f1':.6}}
        high={'source':'b','gate':{'eligible':False},'metrics':{'segment_f1_50':.4,'edit':.5,'macro_f1':.7}}
        self.assertEqual(c.choose_decoder([weak,high])['source'],'a')
        weak['gate']['eligible']=False
        selected=c.choose_decoder([weak,high]);self.assertEqual(selected['source'],'b');self.assertFalse(selected['adoption'])

    def test_contextual_targets_hard_negative_and_unresolved(self):
        labels=[7,7,0,10,10,11,12,7,7,0,7,10,11,0]
        expected=[0,0,0,1,1,2,2,3,3,0,0,1,-1,-1]
        target,episodes=c.contextual_targets(labels)
        np.testing.assert_array_equal(target,expected);np.testing.assert_array_equal(target,c.reference_contextual(labels))
        self.assertEqual(len(episodes),2);self.assertIsNone(episodes[-1]['recovery_start'])

    def test_contextual_reference_random_and_empty(self):
        rng=np.random.default_rng(4)
        for n in [0,1,2,7,64,1000]:
            for _ in range(8):
                y=rng.choice([0,7,10,11,12],n)
                np.testing.assert_array_equal(c.contextual_targets(y)[0],c.reference_contextual(y))

    def test_latency_delayed_premature_and_short(self):
        cfg=b.io.read(b.io.ROOT/'configs/s0c_document_reconstruction_v1.json')['latency_diagnostic']
        y=np.array([10]*10+[12]*70);p=np.array([12]*3+[10]*12+[12]*65)
        events=c.latency_events(p,y,[(10,12)],cfg)
        self.assertEqual(events[0]['delay_frames'],5);self.assertTrue(events[0]['premature'])
        self.assertEqual(c.latency_events([],[],[(10,12)],cfg),[])


if __name__=='__main__':unittest.main()
