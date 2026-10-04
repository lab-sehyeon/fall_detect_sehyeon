import copy
import unittest
import numpy as np
import torch
from torch import nn
from fall_pipeline.primitives.p1_reconstruction import PrimitiveCorrection, subset_indices, signals_batch
from fall_pipeline.common.integrity import hash_named_tensors


class P1Tests(unittest.TestCase):
    def test_epoch0_exact_and_copied_not_shared(self):
        base = nn.Linear(512, 4).eval().requires_grad_(False)
        model = PrimitiveCorrection(base)
        h, p = torch.randn(3, 64, 512), torch.randn(3, 64, 12)
        self.assertTrue(torch.equal(model(h, p), base(h)))
        self.assertNotEqual(model.classifier.weight.data_ptr(), base.weight.data_ptr())
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 8708)

    def test_gradients_frozen_original(self):
        base = nn.Linear(512, 4).requires_grad_(False)
        before = hash_named_tensors(base.state_dict().items())
        model = PrimitiveCorrection(base)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model(torch.randn(2, 64, 512), torch.randn(2, 64, 12)).square().mean().backward()
        self.assertGreater(float(model.projection.weight.grad.abs().sum()), 0)
        opt.step()
        self.assertEqual(before, hash_named_tensors(base.state_dict().items()))

    def test_subset_deterministic_sorted_quarter(self):
        a = subset_indices(599986)
        self.assertEqual(len(a), 149996)
        self.assertTrue((np.diff(a) > 0).all())
        np.testing.assert_array_equal(a, subset_indices(599986))
        self.assertFalse(np.array_equal(a, subset_indices(599986, 1)))

    def test_real_shape_zero_signals(self):
        x = signals_batch(np.zeros((3, 3, 64, 25, 2), np.float32))
        self.assertEqual(x.shape, (3, 64, 12))
        self.assertTrue((x == 0).all())

    def test_resume_optimizer_exact(self):
        torch.manual_seed(0)
        model = PrimitiveCorrection(nn.Linear(512, 4))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        h, p = torch.randn(2, 64, 512), torch.randn(2, 64, 12)
        def step(m, o):
            o.zero_grad(set_to_none=True)
            m(h, p).square().mean().backward()
            o.step()
        step(model, optimizer)
        saved = copy.deepcopy((model.state_dict(), optimizer.state_dict()))
        step(model, optimizer)
        resumed = PrimitiveCorrection(nn.Linear(512, 4))
        resumed.load_state_dict(saved[0])
        opt2 = torch.optim.AdamW(resumed.parameters(), lr=1e-4)
        opt2.load_state_dict(saved[1])
        step(resumed, opt2)
        self.assertEqual(hash_named_tensors(model.state_dict().items()), hash_named_tensors(resumed.state_dict().items()))
