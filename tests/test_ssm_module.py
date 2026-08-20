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


def ones_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None,
              delta_softplus=False, return_last_state=False):
    return torch.ones_like(u)

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


class AddOneSpikMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, F, time_step):
        self.calls.append((F.detach().clone(), time_step))
        return F + 1


class TimesTwo(nn.Module):
    def forward(self, x):
        return 2 * x


class AddThreePLIF(nn.Module):
    def forward(self, x, time_step):
        return x + 3, x

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

    def test_selective_scan_uses_packed_four_direction_contract(self):
        recorder = RecordingScan()
        module = SpikMamba2D(
            channels=4,
            d_state=3,
            selective_scan=recorder,
        )
        scans = torch.randn(2, 4, 4, 6)

        raw = module._run_selective_scan(scans)

        self.assertEqual(raw.shape, (2, 4, 4, 6))
        self.assertEqual(recorder.call["u"], torch.Size((2, 16, 6)))
        self.assertEqual(recorder.call["delta"], torch.Size((2, 16, 6)))
        self.assertEqual(recorder.call["A"], torch.Size((16, 3)))
        self.assertEqual(recorder.call["B"], torch.Size((2, 4, 3, 6)))
        self.assertEqual(recorder.call["C"], torch.Size((2, 4, 3, 6)))
        self.assertEqual(recorder.call["D"], torch.Size((16,)))
        self.assertEqual(
            recorder.call["delta_bias"], torch.Size((16,))
        )
        self.assertIsNone(recorder.call["z"])
        self.assertTrue(recorder.call["delta_softplus"])
        self.assertFalse(recorder.call["return_last_state"])

    def test_merge_restores_directions_and_sums_them(self):
        module = SpikMamba2D(
            channels=1,
            d_state=1,
            selective_scan=selective_scan_test_double,
        )
        raw = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
              [[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]],
              [[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]],
              [[6.0, 3.0, 5.0, 2.0, 4.0, 1.0]]]]
        )

        merged = module._merge_scan_outputs(raw, height=2, width=3)

        expected = torch.tensor(
            [[[[4.0, 8.0, 12.0], [16.0, 20.0, 24.0]]]]
        )
        self.assertTrue(torch.equal(merged, expected))

class ConvBlockSpikMambaIntegrationTest(unittest.TestCase):
    def test_spikmamba_is_opt_in(self):
        block = ConvBlock(1, 1, kernel_size=1, dropout=0.0)

        self.assertFalse(block.spikMamba)
        self.assertFalse(hasattr(block, "spik_mamba"))

    def test_spikmamba_runs_after_conv_and_before_norm_and_output_plif(self):
        block = ConvBlock(
            1,
            1,
            kernel_size=1,
            dropout=0.0,
            spikMamba=True,
            selective_scan=ones_scan,
        )
        with torch.no_grad():
            block.conv.weight.fill_(1.0)
        recorder = AddOneSpikMamba()
        block.spik_mamba = recorder
        block.norm = TimesTwo()
        block.spike_neurons = AddThreePLIF()

        output = block(torch.ones(1, 1, 1, 1), time_step=9)

        self.assertTrue(
            torch.equal(output, torch.tensor([[[[7.0]]]]))
        )
        self.assertTrue(
            torch.equal(
                recorder.calls[0][0],
                torch.ones(1, 1, 1, 1),
            )
        )
        self.assertEqual(recorder.calls[0][1], 9)

class SNNBraTSSpikMambaIntegrationTest(unittest.TestCase):
    def test_all_three_encoder_blocks_use_injected_spikmamba(self):
        recorder = RecordingScan()
        model = SNNBraTS(
            out_channels=4,
            selective_scan=recorder,
            ssm_d_state=2,
        )

        for block in (
            model.conv_block1,
            model.conv_block2,
            model.conv_block3,
        ):
            self.assertTrue(block.spikMamba)
            self.assertIsInstance(block.spik_mamba, SpikMamba2D)
            self.assertIs(block.spik_mamba.selective_scan, recorder)
            self.assertEqual(block.spik_mamba.d_state, 2)

        for block in (
            model.deconv1_conv,
            model.concat1_conv,
            model.deconv2_conv,
            model.concat2_conv,
            model.deconv3_conv,
            model.class_conv,
        ):
            self.assertFalse(block.spikMamba)
            self.assertFalse(hasattr(block, "spik_mamba"))

    def test_small_network_forward_preserves_segmentation_shape(self):
        model = SNNBraTS(
            out_channels=4,
            selective_scan=ones_scan,
            ssm_d_state=2,
        ).eval()
        x = torch.randn(1, 1, 4, 16, 16)

        with torch.no_grad():
            output = model(x, t0=0)

        self.assertEqual(output.shape, (1, 4, 1, 16, 16))

class SpikMamba2DForwardTest(unittest.TestCase):
    def test_forward_uses_original_F_for_hadamard_and_residual(self):
        module = SpikMamba2D(
            channels=2,
            d_state=1,
            selective_scan=ones_scan,
        )
        module.lif_1 = PassthroughPLIF()
        module.lif_2 = PassthroughPLIF()
        module.lif_ssm = PassthroughPLIF()
        F_in = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])

        output = module(F_in, time_step=5)

        self.assertTrue(torch.equal(output, 5 * F_in))
        self.assertEqual(module.lif_ssm.time_steps, [5])

    def test_forward_passes_same_time_step_to_all_three_plifs(self):
        module = SpikMamba2D(
            channels=2,
            d_state=1,
            selective_scan=ones_scan,
        )
        module.lif_1 = PassthroughPLIF()
        module.lif_2 = PassthroughPLIF()
        module.lif_ssm = PassthroughPLIF()

        module(torch.randn(1, 2, 2, 3), time_step=11)

        self.assertEqual(module.lif_1.time_steps, [11])
        self.assertEqual(module.lif_2.time_steps, [11])
        self.assertEqual(module.lif_ssm.time_steps, [11])

    def test_forward_validates_layout_channels_and_scan_dependency(self):
        module = SpikMamba2D(
            channels=2,
            d_state=1,
            selective_scan=ones_scan,
        )
        with self.assertRaisesRegex(ValueError, "BCHW"):
            module(torch.randn(1, 2, 4), time_step=0)
        with self.assertRaisesRegex(ValueError, "expects 2 channels"):
            module(torch.randn(1, 3, 2, 2), time_step=0)
        module.selective_scan = None
        with self.assertRaisesRegex(ImportError, "mamba_ssm"):
            module(torch.randn(1, 2, 2, 2), time_step=0)

    def test_forward_preserves_shape_and_backpropagates(self):
        torch.manual_seed(0)
        module = SpikMamba2D(
            channels=4,
            d_state=2,
            selective_scan=selective_scan_test_double,
        )
        F_in = torch.randn(2, 4, 3, 4, requires_grad=True)

        output = module(F_in, time_step=0)
        output.square().mean().backward()

        self.assertEqual(output.shape, F_in.shape)
        self.assertIsNotNone(F_in.grad)
        self.assertGreater(float(F_in.grad.abs().sum()), 0.0)
        self.assertIsNotNone(module.linear_m.weight.grad)
        self.assertIsNotNone(module.scan_conv1d.weight.grad)

if __name__ == "__main__":
    unittest.main()
