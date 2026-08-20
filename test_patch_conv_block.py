import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
import yaml

from model import ConvBlock, SNNBraTS


def selective_scan_reference(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    """Small differentiable CPU implementation of the kernel contract."""
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

    output = torch.stack(outputs, dim=-1).reshape(
        batch, packed_channels, length
    )
    return (output, state) if return_last_state else output


def identity_selective_scan(u, *args, **kwargs):
    return u


class ZeroSS2D(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class PatchConvBlockTest(unittest.TestCase):
    def make_patch_block(self, channels=32, patch_size=4):
        block = ConvBlock(
            channels,
            channels,
            padding=1,
            dropout=0.0,
            normalization=False,
            spiking=False,
            ss2d=True,
            patch_size=patch_size,
        )
        block.ssm_module.selective_scan = identity_selective_scan
        return block

    def test_patch_embedding_reduces_grid_and_block_restores_shape(self):
        block = self.make_patch_block()
        x = torch.randn(2, 32, 20, 24)

        embedded = block.patch_embed(x)
        output = block(x, time_step=0)

        self.assertEqual(embedded.shape, (2, 32, 5, 6))
        self.assertEqual(output.shape, x.shape)

    def test_non_divisible_dimensions_are_cropped_to_input_shape(self):
        block = self.make_patch_block()
        x = torch.randn(2, 32, 17, 19)

        output = block(x, time_step=0)

        self.assertEqual(output.shape, x.shape)

    def test_patch_path_does_not_add_the_convolution_residual(self):
        block = self.make_patch_block(channels=4)
        block.ssm_module = ZeroSS2D()
        x = torch.randn(1, 4, 8, 8)

        output = block(x, time_step=0)

        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_gradients_reach_embedding_ss2d_and_unembedding(self):
        torch.manual_seed(0)
        block = ConvBlock(
            4,
            4,
            padding=1,
            dropout=0.0,
            normalization=False,
            spiking=False,
            ss2d=True,
            patch_size=2,
        )
        block.ssm_module.selective_scan = selective_scan_reference
        x = torch.randn(1, 4, 6, 8, requires_grad=True)

        block(x, time_step=0).square().mean().backward()

        self.assertIsNotNone(block.patch_embed[0].weight.grad)
        self.assertGreater(
            float(block.patch_embed[0].weight.grad.abs().sum()), 0.0
        )
        self.assertIsNotNone(block.ssm_module.x_proj_weight.grad)
        self.assertGreater(
            float(block.ssm_module.x_proj_weight.grad.abs().sum()), 0.0
        )
        self.assertIsNotNone(block.patch_unembed.weight.grad)
        self.assertGreater(
            float(block.patch_unembed.weight.grad.abs().sum()), 0.0
        )

    def test_snn_brats_patchifies_only_first_encoder_block(self):
        model = SNNBraTS(out_channels=3, patch_size=4)

        self.assertEqual(model.conv_block1.patch_size, 4)
        self.assertIsNone(model.conv_block2.patch_size)
        self.assertIsNone(model.conv_block3.patch_size)

    def test_default_conv_block_keeps_pixel_level_ss2d(self):
        block = ConvBlock(4, 4, dropout=0.0, ss2d=True)

        self.assertIsNone(block.patch_size)
        self.assertFalse(hasattr(block, "patch_embed"))
        self.assertFalse(hasattr(block, "patch_unembed"))

    def test_patch_size_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "patch_size"):
            ConvBlock(4, 4, ss2d=True, patch_size=0)

    def test_yaml_selects_four_by_four_patches(self):
        config_path = Path(__file__).parent / "experiments_snn_fptt.yaml"

        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        self.assertEqual(config["patch_size"], 4)

    def test_model_factory_passes_patch_size_to_first_encoder(self):
        accelerate = types.ModuleType("accelerate")
        accelerate.Accelerator = type("Accelerator", (), {})
        accelerate_utils = types.ModuleType("accelerate.utils")
        accelerate_utils.broadcast_object_list = lambda *args, **kwargs: None

        with mock.patch.dict(
            sys.modules,
            {
                "accelerate": accelerate,
                "accelerate.utils": accelerate_utils,
            },
        ):
            sys.modules.pop("snn_fptt", None)
            from snn_fptt import build_model

        model = build_model("orig", out_channels=3, patch_size=2)

        self.assertEqual(model.conv_block1.patch_size, 2)
        self.assertIsNone(model.conv_block2.patch_size)
        self.assertIsNone(model.conv_block3.patch_size)


if __name__ == "__main__":
    unittest.main()
