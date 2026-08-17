import torch
import torch.nn as nn
import pytest

import surrogate
from spike_neurons import PLIFNode
from spikmamba import Spiking2DPatchEmbedding


class RecordingPLIF(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_steps = []

    def forward(self, x, time_step):
        self.time_steps.append(time_step)
        return x, x


def test_patch_embedding_construction_and_shape():
    module = Spiking2DPatchEmbedding(
        4, 8, image_size=(16, 20)
    )
    assert module.proj.kernel_size == (4, 4)
    assert module.proj.stride == (4, 4)
    assert module.proj.bias is not None
    assert isinstance(module.norm, nn.BatchNorm2d)
    assert isinstance(module.sl_patch, PLIFNode)
    assert module.sl_patch.detach_reset is True
    assert isinstance(module.sl_patch.surrogate_function, surrogate.ATan)
    assert module.spatial_pos_embed.shape == (1, 8, 4, 5)
    assert module.temporal_pos_embed.shape == (256, 8)

    recorder = RecordingPLIF()
    module.sl_patch = recorder
    module.eval()
    output = module(torch.randn(2, 4, 16, 20), time_step=7)
    assert output.shape == (2, 8, 4, 5)
    assert recorder.time_steps == [7]


def test_patch_embedding_interpolates_space_and_selects_time():
    module = Spiking2DPatchEmbedding(
        1, 2, image_size=(8, 8), max_time_steps=4
    )
    module.sl_patch = RecordingPLIF()
    module.eval()
    with torch.no_grad():
        module.spatial_pos_embed.zero_()
        module.temporal_pos_embed.zero_()
        module.temporal_pos_embed[2].fill_(1.5)
    x = torch.randn(1, 1, 8, 12)
    at_two = module(x, 2)
    at_three = module(x, 3)
    assert at_two.shape == (1, 2, 2, 3)
    torch.testing.assert_close(
        at_two - at_three, torch.full_like(at_two, 1.5)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"in_channels": 0, "embed_dim": 4},
        {"in_channels": 1, "embed_dim": 0},
        {"in_channels": 1, "embed_dim": 4, "patch_size": 0},
        {"in_channels": 1, "embed_dim": 4, "max_time_steps": 0},
        {"in_channels": 1, "embed_dim": 4, "image_size": (9, 8)},
    ],
)
def test_patch_embedding_rejects_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        Spiking2DPatchEmbedding(**kwargs)


def test_patch_embedding_rejects_bad_input():
    module = Spiking2DPatchEmbedding(
        4, 8, image_size=(16, 20), max_time_steps=3
    )
    with pytest.raises(ValueError, match="BCHW"):
        module(torch.randn(2, 4, 16), 0)
    with pytest.raises(ValueError, match="4 channels"):
        module(torch.randn(2, 3, 16, 20), 0)
    with pytest.raises(ValueError, match="divisible"):
        module(torch.randn(2, 4, 15, 20), 0)
    with pytest.raises(ValueError, match="time_step"):
        module(torch.randn(2, 4, 16, 20), 3)


def test_patch_plif_continues_then_restarts_at_time_zero():
    module = Spiking2DPatchEmbedding(
        1, 1, image_size=(8, 8), max_time_steps=4
    ).eval()
    with torch.no_grad():
        module.proj.weight.zero_()
        module.proj.bias.fill_(0.5)
        module.spatial_pos_embed.zero_()
        module.temporal_pos_embed.zero_()
    x = torch.zeros(1, 1, 8, 8)
    module(x, 0)
    first = module.sl_patch.v.detach().clone()
    module(x, 1)
    continued = module.sl_patch.v.detach().clone()
    module(x, 0)
    restarted = module.sl_patch.v.detach().clone()
    assert not torch.allclose(first, continued)
    torch.testing.assert_close(first, restarted)
