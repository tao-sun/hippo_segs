import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ConvBlock, SNNBraTS, SpikMamba2D
from spike_neurons import PLIFNode


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


class PassthroughPLIF(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_steps = []

    def forward(self, x, time_step):
        self.time_steps.append(time_step)
        return x, x


class SpikMamba2DConstructionTest(unittest.TestCase):
    def test_constructor_builds_independent_spiking_scan_stages(self):
        module = SpikMamba2D(
            channels=8,
            d_state=4,
            selective_scan=selective_scan_test_double,
        )

        self.assertEqual(module.channels, 8)
        self.assertEqual(module.dt_rank, 1)
        self.assertEqual(module.linear_m.in_features, 8)
        self.assertEqual(module.linear_m.out_features, 8)
        self.assertIsInstance(module.lif_1, PLIFNode)
        self.assertIsInstance(module.lif_2, PLIFNode)
        self.assertIsInstance(module.lif_ssm, PLIFNode)
        self.assertIsNot(module.lif_1, module.lif_2)
        self.assertIsNot(module.lif_1, module.lif_ssm)
        self.assertIsNot(module.lif_2, module.lif_ssm)
        self.assertEqual(module.scan_conv1d.in_channels, 32)
        self.assertEqual(module.scan_conv1d.out_channels, 32)
        self.assertEqual(module.scan_conv1d.groups, 32)
        self.assertEqual(module.scan_conv1d.kernel_size, (3,))
        self.assertEqual(module.scan_conv1d.padding, (1,))
        self.assertEqual(module.A_logs.shape, (32, 4))
        self.assertEqual(module.Ds.shape, (32,))
        self.assertIs(module.selective_scan, selective_scan_test_double)

    def test_constructor_rejects_kernel_without_symmetric_same_padding(self):
        for kernel_size in (0, 2, -1):
            with self.subTest(kernel_size=kernel_size):
                with self.assertRaisesRegex(
                    ValueError, "positive odd integer"
                ):
                    SpikMamba2D(
                        channels=4,
                        conv_kernel_size=kernel_size,
                        selective_scan=selective_scan_test_double,
                    )

    def test_selective_parameters_keep_stable_initialization(self):
        module = SpikMamba2D(
            channels=8,
            d_state=4,
            selective_scan=selective_scan_test_double,
        )

        initialized_dt = F.softplus(module.dt_projs_bias)
        self.assertGreaterEqual(float(initialized_dt.min()), 0.001 - 1e-6)
        self.assertLessEqual(float(initialized_dt.max()), 0.1 + 1e-6)
        self.assertTrue(torch.all(-torch.exp(module.A_logs) < 0))
        self.assertTrue(torch.equal(module.Ds, torch.ones_like(module.Ds)))

class SpikMamba2DScanTest(unittest.TestCase):
    def test_prepare_scans_preserves_direction_order_and_filters_independently(self):
        module = SpikMamba2D(
            channels=1,
            d_state=1,
            selective_scan=selective_scan_test_double,
        )
        module.linear_m = nn.Identity()
        module.lif_1 = PassthroughPLIF()
        module.lif_2 = PassthroughPLIF()
        with torch.no_grad():
            module.scan_conv1d.weight.zero_()
            module.scan_conv1d.bias.zero_()
            module.scan_conv1d.weight[:, 0, 1] = torch.tensor(
                [1.0, 2.0, 3.0, 4.0]
            )
        F_in = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]])

        scans = module._prepare_scans(F_in, time_step=7)

        expected = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
              [[2.0, 8.0, 4.0, 10.0, 6.0, 12.0]],
              [[18.0, 15.0, 12.0, 9.0, 6.0, 3.0]],
              [[24.0, 12.0, 20.0, 8.0, 16.0, 4.0]]]]
        )
        self.assertTrue(torch.equal(scans, expected))
        self.assertEqual(module.lif_1.time_steps, [7])
        self.assertEqual(module.lif_2.time_steps, [7])

if __name__ == "__main__":
    unittest.main()
