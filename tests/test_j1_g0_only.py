import unittest
import torch
from torch import nn
import numpy as np
from fall_pipeline.external.j1_g0_only import J1G0Only,adapter_state
from fall_pipeline.joint.models import JointFallModel
from scripts.audit_j1_g0_evaluation_20261002 import cpu_replay


class FakeBackbone(nn.Module):
    def forward(self,jt,js):
        return jt.mean(-1,keepdim=True).expand(-1,-1,1024),js.mean(-1,keepdim=True).expand(-1,-1,1024)


class J1G0OnlyTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(0)

    def test_only_active_modules_and_outputs(self):
        model=J1G0Only(FakeBackbone()).eval()
        self.assertEqual(set(dict(model.named_children())),{'backbone','adapter','head'})
        result=model(torch.zeros(2,3,64,25,2))
        self.assertEqual(set(result),{'pooled','adapted','G0'})
        self.assertEqual(result['G0'].shape,(2,4))

    def test_adapter_only_strict_loading(self):
        joint=JointFallModel();model=J1G0Only(FakeBackbone()).eval()
        state=adapter_state(joint.state_dict())
        self.assertFalse(any('head' in key for key in state))
        model.adapter.load_state_dict(state,strict=True)
        self.assertEqual(sum(p.numel() for p in model.adapter.parameters()),1054976)
        self.assertEqual(sum(p.numel() for p in model.head.parameters()),8196)

    def test_missing_adapter_rejected(self):
        model=J1G0Only(FakeBackbone())
        with self.assertRaises(RuntimeError):model.adapter.load_state_dict(adapter_state({'safer_head.weight':torch.ones(4,2048)}),strict=True)

    def test_cpu_formula_matches_nonzero_adapter(self):
        joint=JointFallModel().eval();nn.init.normal_(joint.adapter.up.weight,std=.01)
        model=J1G0Only(FakeBackbone(),joint.adapter).eval()
        with torch.inference_mode():
            out=model(torch.randn(3,3,64,25,2))
            a,g=cpu_replay(out['pooled'].numpy(),joint.state_dict(),model.head.state_dict())
        np.testing.assert_array_equal(a,out['adapted'].numpy());np.testing.assert_array_equal(g,out['G0'].numpy())

    def test_layout_validation(self):
        with self.assertRaises(ValueError):J1G0Only(FakeBackbone())(torch.zeros(1,3,63,25,2))

    def test_eval_no_dropout_randomness(self):
        m=J1G0Only(FakeBackbone()).eval().requires_grad_(False);x=torch.randn(1,3,64,25,2)
        self.assertTrue(torch.equal(m(x)['G0'],m(x)['G0']))
        self.assertFalse(any(p.requires_grad for p in m.parameters()))


if __name__=='__main__':unittest.main()
