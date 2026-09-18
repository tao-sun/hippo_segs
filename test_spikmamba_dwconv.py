import torch
import torch.nn as nn

from model import SNNBraTSVSS
from spike_neurons import PLIFNode
from spikmamba import SpikeMambaLayer


class SelectiveScanIdentity:
    def __init__(self):
        self.scan_input = None

    def __call__(self, scan_input, *args, **kwargs):
        self.scan_input = scan_input.detach().clone()
        return scan_input


def test_depthwise_conv2d_runs_before_the_four_direction_scan():
    selective_scan = SelectiveScanIdentity()
    layer = SpikeMambaLayer(
        dim=1,
        d_state=1,
        linear_projection=False,
        patch_embedding_spiking=False,
        dwconv2d_spiking=False,
        conv_bias=False,
        selective_scan=selective_scan,
    )

    assert isinstance(layer.linear_m, nn.Identity)
    assert isinstance(layer.out_proj, nn.Identity)
    assert isinstance(layer.dwconv2d, nn.Conv2d)
    assert layer.dwconv2d.kernel_size == (3, 3)
    assert layer.dwconv2d.padding == (1, 1)
    assert layer.dwconv2d.groups == layer.d_inner == 1
    assert not any(isinstance(module, nn.Conv1d) for module in layer.modules())

    with torch.no_grad():
        layer.dwconv2d.weight.zero_()
        layer.dwconv2d.weight[0, 0, 1, 1] = 1.0
        layer.dwconv2d.weight[0, 0, 1, 2] = 1.0

    tokens = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]])
    output = layer(tokens, time_step=0, spatial_shape=(2, 3))

    expected_routes = torch.tensor(
        [[
            [3.0, 5.0, 3.0, 9.0, 11.0, 6.0],
            [6.0, 11.0, 9.0, 3.0, 5.0, 3.0],
            [3.0, 9.0, 5.0, 11.0, 3.0, 6.0],
            [6.0, 3.0, 11.0, 5.0, 9.0, 3.0],
        ]]
    )
    torch.testing.assert_close(selective_scan.scan_input, expected_routes)
    torch.testing.assert_close(output, expected_routes[:, 0].unsqueeze(-1))


def test_spiking_order_is_patch_plif_then_dwconv_then_optional_plif():
    events = []
    layer = SpikeMambaLayer(
        dim=2,
        d_state=1,
        expand=2,
        linear_projection=True,
        patch_embedding_spiking=True,
        dwconv2d_spiking=True,
        selective_scan=SelectiveScanIdentity(),
    )

    assert isinstance(layer.linear_m, nn.Linear)
    assert isinstance(layer.out_proj, nn.Linear)
    assert layer.d_inner == 4
    assert layer.dwconv2d.groups == 4
    assert isinstance(layer.patch_sl, PLIFNode)
    assert isinstance(layer.dwconv2d_sl, PLIFNode)

    hooks = [
        layer.patch_sl.register_forward_hook(
            lambda module, inputs, output: events.append("patch_plif")
        ),
        layer.dwconv2d.register_forward_hook(
            lambda module, inputs, output: events.append("dwconv2d")
        ),
        layer.dwconv2d_sl.register_forward_hook(
            lambda module, inputs, output: events.append("dwconv2d_plif")
        ),
    ]
    try:
        output = layer(
            torch.ones(1, 6, 2),
            time_step=0,
            spatial_shape=(2, 3),
        )
    finally:
        for hook in hooks:
            hook.remove()

    assert output.shape == (1, 6, 2)
    assert events == ["patch_plif", "dwconv2d", "dwconv2d_plif"]


def test_dwconv2d_spiking_can_be_disabled_independently():
    layer = SpikeMambaLayer(
        dim=2,
        d_state=1,
        linear_projection=True,
        patch_embedding_spiking=True,
        dwconv2d_spiking=False,
        selective_scan=SelectiveScanIdentity(),
    )

    assert isinstance(layer.patch_sl, PLIFNode)
    assert layer.dwconv2d_sl is None


def test_model_propagates_dwconv_flag_and_optional_linear_projection():
    model = SNNBraTSVSS(
        out_channels=3,
        selective_scan=SelectiveScanIdentity(),
        patch_size=2,
        linear_projection=False,
        dwconv2d_spiking=False,
    )

    layers = [
        model.conv_block1.spik_mamba.spik_mamba.mamba_layer,
        model.conv_block2.spik_mamba.spik_mamba.mamba_layer,
        model.conv_block3.spik_mamba.spik_mamba.mamba_layer,
    ]
    assert all(layer.dwconv2d_sl is None for layer in layers)
    assert all(isinstance(layer.linear_m, nn.Identity) for layer in layers)
    assert all(layer.d_inner == layer.dim for layer in layers)
