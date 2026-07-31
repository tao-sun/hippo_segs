import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import SNNBraTS, SS2D, SSMBlock2D


def selective_scan_test_double(u, delta, A, B, C, D=None, z=None,
                               delta_bias=None, delta_softplus=False,
                               return_last_state=False):
    """Differentiable CPU reference with the VM-UNet kernel contract."""
    batch, packed_channels, length = u.shape
    directions = B.shape[1]
    channels = packed_channels // directions
    state_size = A.shape[-1]
    u = u.view(batch, directions, channels, length)
    delta = delta.view(batch, directions, channels, length)
    A = A.view(directions, channels, state_size)
    D = D.view(directions, channels) if D is not None else None
    if delta_bias is not None:
        delta = delta + delta_bias.view(1, directions, channels, 1)
    if delta_softplus:
        delta = F.softplus(delta)

    state = u.new_zeros(batch, directions, channels, state_size)
    outputs = []
    for index in range(length):
        delta_t = delta[..., index]
        u_t = u[..., index]
        delta_a = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))
        delta_b_u = (
            delta_t.unsqueeze(-1)
            * B[:, :, :, index].unsqueeze(2)
            * u_t.unsqueeze(-1)
        )
        state = delta_a * state + delta_b_u
        y_t = (state * C[:, :, :, index].unsqueeze(2)).sum(dim=-1)
        if D is not None:
            y_t = y_t + D.unsqueeze(0) * u_t
        outputs.append(y_t)
    output = torch.stack(outputs, dim=-1).reshape(batch, packed_channels, length)
    return (output, state) if return_last_state else output


class RecordingScan:
    def __init__(self):
        self.call = None

    def __call__(self, u, delta, A, B, C, D=None, z=None,
                 delta_bias=None, delta_softplus=False,
                 return_last_state=False):
        self.call = {
            "u": u.shape,
            "delta": delta.shape,
            "A": A.shape,
            "B": B.shape,
            "C": C.shape,
            "D": None if D is None else D.shape,
            "z": z,
            "delta_bias": None if delta_bias is None else delta_bias.shape,
            "delta_softplus": delta_softplus,
            "return_last_state": return_last_state,
        }
        return torch.zeros_like(u)


class SS2DTest(unittest.TestCase):
    def test_initialization_matches_selective_scan_dimensions(self):
        module = SS2D(channels=8, d_state=4,
                      selective_scan=selective_scan_test_double)

        self.assertEqual(module.channels, 8)
        self.assertEqual(module.dt_rank, 1)
        self.assertEqual(module.A_logs.shape, (32, 4))
        self.assertEqual(module.Ds.shape, (32,))
        self.assertTrue(torch.all(-torch.exp(module.A_logs) < 0))
        self.assertTrue(torch.allclose(module.Ds, torch.ones_like(module.Ds)))
        initialized_dt = F.softplus(module.dt_projs_bias)
        self.assertGreaterEqual(float(initialized_dt.min()), 0.001 - 1e-6)
        self.assertLessEqual(float(initialized_dt.max()), 0.1 + 1e-6)
        self.assertFalse(hasattr(module, "in_proj"))
        self.assertFalse(hasattr(module, "conv2d"))
        self.assertFalse(hasattr(module, "out_norm"))
        self.assertFalse(hasattr(module, "out_proj"))

    def test_cross_scan_uses_vmunet_direction_order(self):
        module = SS2D(channels=1, d_state=1,
                      selective_scan=selective_scan_test_double)
        x = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]])

        scans = module._cross_scan(x)

        expected = torch.tensor([[[[1, 2, 3, 4, 5, 6]],
                                  [[1, 4, 2, 5, 3, 6]],
                                  [[6, 5, 4, 3, 2, 1]],
                                  [[6, 3, 5, 2, 4, 1]]]], dtype=torch.float32)
        self.assertTrue(torch.equal(scans, expected))

    def test_forward_core_calls_official_kernel_contract(self):
        recorder = RecordingScan()
        module = SS2D(channels=4, d_state=3,
                      selective_scan=recorder)

        output = module(torch.randn(2, 4, 2, 3))

        self.assertEqual(output.shape, torch.Size((2, 4, 2, 3)))
        self.assertEqual(recorder.call["u"], torch.Size((2, 16, 6)))
        self.assertEqual(recorder.call["delta"], torch.Size((2, 16, 6)))
        self.assertEqual(recorder.call["A"], torch.Size((16, 3)))
        self.assertEqual(recorder.call["B"], torch.Size((2, 4, 3, 6)))
        self.assertEqual(recorder.call["C"], torch.Size((2, 4, 3, 6)))
        self.assertEqual(recorder.call["D"], torch.Size((16,)))
        self.assertEqual(recorder.call["delta_bias"], torch.Size((16,)))
        self.assertIsNone(recorder.call["z"])
        self.assertTrue(recorder.call["delta_softplus"])
        self.assertFalse(recorder.call["return_last_state"])

    def test_ss2d_forward_preserves_bchw_shape_and_backpropagates(self):
        torch.manual_seed(0)
        module = SS2D(channels=8, d_state=2,
                      selective_scan=selective_scan_test_double)
        x = torch.randn(2, 8, 5, 6, requires_grad=True)

        y = module(x)
        y.square().mean().backward()

        self.assertEqual(y.shape, x.shape)
        self.assertIsNotNone(x.grad)
        self.assertGreater(float(x.grad.abs().sum()), 0.0)


class SS2DIntegrationTest(unittest.TestCase):
    def test_ssm_block_honors_injected_module(self):
        injected = nn.Identity()
        block = SSMBlock2D(channels=8, ssm_module=injected,
                           dropout=0.0, normalization=True)

        self.assertIs(block.ssm_module, injected)

    def test_ssm_block_applies_residual_around_injected_ss2d(self):
        class Double(nn.Module):
            def forward(self, x):
                return 2 * x

        block = SSMBlock2D(channels=8, ssm_module=Double(),
                           dropout=0.0, normalization=True)
        x = torch.randn(2, 8, 5, 6)

        self.assertTrue(torch.equal(block._apply_ssm(x), 3 * x))

    def test_snn_brats_activates_direct_ss2d(self):
        torch.manual_seed(0)
        model = SNNBraTS(
            out_channels=4,
            selective_scan=selective_scan_test_double,
            ssm_d_state=2,
        )
        x = torch.randn(1, 1, 4, 16, 16)

        y = model(x, t0=0)

        self.assertEqual(y.shape, (1, 4, 1, 16, 16))
        self.assertIsInstance(model.ssm_block3, SSMBlock2D)
        self.assertIsInstance(model.ssm_block3.ssm_module, SS2D)
        self.assertIs(
            model.ssm_block3.ssm_module.selective_scan,
            selective_scan_test_double,
        )


if __name__ == "__main__":
    unittest.main()
