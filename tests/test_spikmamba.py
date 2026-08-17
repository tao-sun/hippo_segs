import torch
import torch.nn as nn
import pytest

import surrogate
from spike_neurons import PLIFNode
from spikmamba import SpikeMambaLayer, Spiking2DPatchEmbedding


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


class IdentityPLIF(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_steps = []

    def forward(self, x, time_step):
        self.time_steps.append(time_step)
        return x, x


class OnesPLIF(nn.Module):
    def forward(self, x, time_step):
        return torch.ones_like(x), x


class RecordingScan:
    def __init__(self):
        self.call = None
        self.calls = 0
        self.u_values = None

    def __call__(
        self, u, delta, A, B, C, D=None, z=None,
        delta_bias=None, delta_softplus=False,
        return_last_state=False,
    ):
        self.calls += 1
        self.u_values = u.detach().clone()
        self.call = {
            "u": u.shape,
            "delta": delta.shape,
            "A": A.shape,
            "B": B.shape,
            "C": C.shape,
            "D": D.shape,
            "z": z,
            "delta_bias": delta_bias.shape,
            "delta_softplus": delta_softplus,
            "return_last_state": return_last_state,
        }
        return u


def test_mamba_construction_and_independent_plifs():
    module = SpikeMambaLayer(
        4, d_state=3, selective_scan=RecordingScan()
    )
    assert module.d_inner == 8
    assert module.dt_rank == 1
    assert (module.linear_m.in_features,
            module.linear_m.out_features) == (4, 8)
    assert module.conv1d_m.in_channels == 8
    assert module.conv1d_m.out_channels == 8
    assert module.conv1d_m.groups == 8
    assert module.conv1d_m.kernel_size == (4,)
    assert module.conv1d_m.padding == (3,)
    assert all(
        isinstance(stage, PLIFNode)
        for stage in (module.sl_m1, module.sl_m2, module.sl_ssm)
    )
    assert all(
        stage.detach_reset is True
        and isinstance(stage.surrogate_function, surrogate.ATan)
        for stage in (module.sl_m1, module.sl_m2, module.sl_ssm)
    )
    assert len({
        id(module.sl_m1), id(module.sl_m2), id(module.sl_ssm)
    }) == 3


def test_patch_grid_uses_exact_row_major_order():
    patches = torch.tensor(
        [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]]]
    )
    tokens = SpikeMambaLayer._to_tokens(patches)
    assert tokens.shape == (1, 6, 1)
    torch.testing.assert_close(
        tokens[0, :, 0], torch.arange(6, dtype=torch.float32)
    )


def test_forward_scans_one_row_major_sequence_without_flip_or_reverse():
    recorder = RecordingScan()
    module = SpikeMambaLayer(
        1,
        d_state=1,
        d_conv=1,
        expand=1,
        selective_scan=recorder,
    )
    module.sl_m1 = IdentityPLIF()
    module.sl_m2 = IdentityPLIF()
    module.sl_ssm = IdentityPLIF()
    with torch.no_grad():
        module.linear_m.weight.fill_(1.0)
        module.conv1d_m.weight.fill_(1.0)
        module.conv1d_m.bias.zero_()

    patches = torch.arange(6, dtype=torch.float32).view(1, 1, 2, 3)
    module(patches, time_step=2)

    assert recorder.calls == 1
    torch.testing.assert_close(
        recorder.u_values[0, 0],
        torch.arange(6, dtype=torch.float32),
    )


def test_scan_length_is_patch_count_with_no_direction_axis():
    recorder = RecordingScan()
    module = SpikeMambaLayer(
        4, d_state=3, selective_scan=recorder
    )
    module.sl_m1 = IdentityPLIF()
    module.sl_m2 = IdentityPLIF()
    module.sl_ssm = IdentityPLIF()
    output = module(torch.randn(2, 4, 2, 3), time_step=7)
    assert output.shape == (2, 4, 2, 3)
    assert recorder.call == {
        "u": torch.Size([2, 8, 6]),
        "delta": torch.Size([2, 8, 6]),
        "A": torch.Size([8, 3]),
        "B": torch.Size([2, 3, 6]),
        "C": torch.Size([2, 3, 6]),
        "D": torch.Size([8]),
        "z": None,
        "delta_bias": torch.Size([8]),
        "delta_softplus": True,
        "return_last_state": False,
    }
    assert module.sl_m1.time_steps == [7]
    assert module.sl_m2.time_steps == [7]
    assert module.sl_ssm.time_steps == [7]


def test_mamba_plifs_continue_then_restart_at_time_zero():
    module = SpikeMambaLayer(
        2,
        d_state=2,
        expand=1,
        selective_scan=RecordingScan(),
    )
    stimulus = torch.full((1, 2, 3), 0.5)

    for stage in (module.sl_m1, module.sl_m2, module.sl_ssm):
        stage(stimulus, 0)
        first = stage.v.detach().clone()
        stage(stimulus, 1)
        continued = stage.v.detach().clone()
        stage(stimulus, 0)
        restarted = stage.v.detach().clone()

        assert not torch.allclose(first, continued)
        torch.testing.assert_close(first, restarted)


def test_conv1d_is_causal_and_preserves_length():
    module = SpikeMambaLayer(
        2, d_state=2, d_conv=3, expand=1,
        selective_scan=RecordingScan(),
    )
    with torch.no_grad():
        module.conv1d_m.weight.fill_(1)
        module.conv1d_m.bias.zero_()
    sequence = torch.zeros(1, 2, 5)
    sequence[:, :, -1] = 1
    output = module._causal_conv(sequence)
    assert output.shape == sequence.shape
    torch.testing.assert_close(output[:, :, :4], torch.zeros(1, 2, 4))
    torch.testing.assert_close(output[:, :, 4], torch.ones(1, 2))


def test_ssm_spikes_gate_original_tokens():
    module = SpikeMambaLayer(
        3, d_state=2, expand=1,
        selective_scan=RecordingScan(),
    )
    module.sl_m1 = IdentityPLIF()
    module.sl_m2 = IdentityPLIF()
    module.sl_ssm = OnesPLIF()
    patches = torch.randn(2, 3, 2, 2)
    torch.testing.assert_close(module(patches, 4), patches)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dim": 0},
        {"dim": 4, "d_state": 0},
        {"dim": 4, "d_conv": 0},
        {"dim": 4, "expand": 0},
        {"dim": 4, "dt_rank": 0},
        {"dim": 4, "dt_init": "bad"},
    ],
)
def test_mamba_rejects_bad_configuration(kwargs):
    with pytest.raises((ValueError, NotImplementedError)):
        SpikeMambaLayer(
            selective_scan=RecordingScan(), **kwargs
        )


def test_mamba_rejects_bad_grid_and_missing_scan():
    module = SpikeMambaLayer(
        4, d_state=2, selective_scan=RecordingScan()
    )
    with pytest.raises(ValueError, match="BDHW"):
        module(torch.randn(1, 4, 8), 0)
    with pytest.raises(ValueError, match="4 channels"):
        module(torch.randn(1, 3, 2, 2), 0)
    module.selective_scan = None
    with pytest.raises(ImportError, match="mamba_ssm"):
        module(torch.randn(1, 4, 2, 2), 0)
