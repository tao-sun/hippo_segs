import pytest
import torch
from torch import nn

import snn_nnunet.network_adapter as adapter_module
from test_network_adapter import APPROVED_CONFIG


class EchoCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.windows = []

    def forward(self, x_win, t0):
        self.windows.append((tuple(x_win.shape), t0, x_win[:, 0].clone()))
        return x_win[:, :, :3].movedim(1, 2)

    def detach_states(self):
        pass


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_volume_shape_and_values_are_restored_on_every_temporal_axis(axis):
    config = adapter_module.SNNConfig.from_plans(
        {"snn_config": {**APPROVED_CONFIG, "temporal_axis": axis, "k": 8}}
    )
    core = EchoCore()
    adapter = adapter_module.SNNnnUNetAdapter(config, core=core)
    x = torch.arange(2 * 4 * 16 * 20 * 24, dtype=torch.float32).reshape(2, 4, 16, 20, 24)

    logits = adapter(x)

    assert logits.shape == (2, 3, 16, 20, 24)
    torch.testing.assert_close(logits, x[:, :3])
    assert [t0 for _, t0, _ in core.windows] == list(range(0, x.shape[2 + axis], 8))
    assert all(shape[0] == 2 and shape[2] == 4 for shape, _, _ in core.windows)
    assert all(shape[1] <= 8 for shape, _, _ in core.windows)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_only_requested_native_window_is_moved_to_core_layout(axis):
    x = torch.arange(1 * 4 * 3 * 5 * 7).reshape(1, 4, 3, 5, 7)
    native_index = [slice(None)] * 5
    native_index[2 + axis] = slice(1, 3)
    first_frame_index = [slice(None)] * 5
    first_frame_index[2 + axis] = 1

    window = adapter_module.slice_temporal_window(x, 1, 3, axis)
    core_window = adapter_module.to_core_layout(window, axis)
    restored = adapter_module.from_core_layout(core_window.movedim(1, 2), axis)

    assert window.shape[2 + axis] == 2
    torch.testing.assert_close(window, x[tuple(native_index)])
    torch.testing.assert_close(core_window[:, 0], x[tuple(first_frame_index)])
    torch.testing.assert_close(restored, x[tuple(native_index)])
