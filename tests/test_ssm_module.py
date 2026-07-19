import unittest

import torch
import torch.nn as nn

from model import SSMBlock2D, SimpleSSM2D, VSSBlock2D


class SimpleSSM2DTest(unittest.TestCase):
    def test_simple_ssm_2d_preserves_shape_and_backpropagates(self):
        torch.manual_seed(0)
        module = SimpleSSM2D(channels=8, hidden_channels=8)
        x = torch.randn(2, 8, 5, 6, requires_grad=True)

        y = module(x)
        loss = y.square().mean()
        loss.backward()

        self.assertEqual(y.shape, x.shape)
        self.assertIsNotNone(x.grad)
        self.assertGreater(float(x.grad.abs().sum()), 0.0)

    def test_vss_block_preserves_shape_and_backpropagates(self):
        torch.manual_seed(0)
        module = VSSBlock2D(channels=8, hidden_channels=8)
        x = torch.randn(2, 8, 5, 6, requires_grad=True)

        y = module(x)
        loss = y.square().mean()
        loss.backward()

        self.assertEqual(y.shape, x.shape)
        self.assertIsNotNone(x.grad)
        self.assertGreater(float(x.grad.abs().sum()), 0.0)

    def test_ssm_block_uses_vss_block_by_default(self):
        block = SSMBlock2D(channels=8, dropout=0.0, normalization=True)

        self.assertIsInstance(block.ssm_module, VSSBlock2D)
        self.assertNotIsInstance(block.ssm_module, nn.Identity)


if __name__ == "__main__":
    unittest.main()
