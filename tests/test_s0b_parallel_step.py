import copy
import unittest
import torch
from torch import nn
from fall_pipeline.safer.s0b_parallel_step import merge_gradients


class ParallelGradientTests(unittest.TestCase):
    def test_global_denominator_sum_matches_full_weighted_batch(self):
        torch.manual_seed(7)
        first=nn.Linear(5,3);second=copy.deepcopy(first);reference=copy.deepcopy(first)
        x=torch.randn(7,5);y=torch.tensor([0,1,2,2,1,0,2]);w=torch.tensor([.2,.9,2.])
        denominator=w[y].sum()
        for model,xx,yy in ((first,x[:4],y[:4]),(second,x[4:],y[4:])):
            (nn.functional.cross_entropy(model(xx),yy,weight=w,reduction='sum')/denominator).backward()
        merge_gradients(first,second)
        nn.functional.cross_entropy(reference(x),y,weight=w).backward()
        for a,b in zip(first.parameters(),reference.parameters()):
            torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=1e-6)
        oa=torch.optim.AdamW(first.parameters(),lr=.001);ob=torch.optim.AdamW(reference.parameters(),lr=.001)
        nn.utils.clip_grad_norm_(first.parameters(),.1);nn.utils.clip_grad_norm_(reference.parameters(),.1)
        oa.step();ob.step()
        for a,b in zip(first.parameters(),reference.parameters()):
            torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
        self.assertTrue(all(s['step'].item()==1 for s in oa.state.values()))

    def test_missing_gradient_fails(self):
        first=nn.Linear(2,2);second=copy.deepcopy(first)
        first(torch.ones(1,2)).sum().backward()
        with self.assertRaises(RuntimeError):merge_gradients(first,second)

    def test_frozen_parameters_untouched(self):
        first=nn.Sequential(nn.Linear(2,2),nn.Linear(2,2));first[0].requires_grad_(False)
        second=copy.deepcopy(first)
        for h in (first,second):h(torch.ones(1,2)).sum().backward()
        merge_gradients(first,second)
        self.assertIsNone(first[0].weight.grad)


if __name__=='__main__':unittest.main()
