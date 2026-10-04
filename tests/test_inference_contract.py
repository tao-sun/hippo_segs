import pytest
import torch
from torch import nn

import snn_nnunet.network_adapter as adapter_module
from test_network_adapter import APPROVED_CONFIG


class StatefulCore(nn.Module):
    """Deterministic membrane surrogate with the core's t0=0 reset behavior."""

    def __init__(self):
        super().__init__()
        self.state = None
        self.calls = []
        self.detach_calls = 0

    def forward(self, x_win, t0):
        self.calls.append((t0, x_win.shape[1]))
        if t0 == 0:
            self.state = torch.zeros_like(x_win[:, 0, :3])
        frames = []
        for index in range(x_win.shape[1]):
            self.state = self.state + x_win[:, index, :3]
            frames.append(self.state + t0 + index)
        return torch.stack(frames, dim=2)

    def detach_states(self):
        self.detach_calls += 1
        self.state = self.state.detach()


def make_adapter(axis=0, k=8):
    config = adapter_module.SNNConfig.from_plans(
        {"snn_config": {**APPROVED_CONFIG, "temporal_axis": axis, "k": k}}
    )
    core = StatefulCore()
    return adapter_module.SNNnnUNetAdapter(config, core=core), core


@pytest.mark.parametrize("chunk_size", [1, 8, 16])
def test_chunk_size_preserves_stateful_predictions_and_absolute_time(chunk_size):
    x = torch.ones(2, 4, 17, 3, 5)
    adapter, core = make_adapter()

    logits = adapter(x, chunk_size=chunk_size)

    assert logits.shape == (2, 3, 17, 3, 5)
    torch.testing.assert_close(logits[0, 0, :, 0, 0], torch.arange(1, 18, dtype=x.dtype) * 2 - 1)
    assert core.calls == [(start, min(chunk_size, 17 - start)) for start in range(0, 17, chunk_size)]
    assert core.detach_calls == 0


def test_default_k_handles_final_partial_window_and_k_larger_than_length():
    x = torch.ones(1, 4, 17, 2, 3)
    adapter, core = make_adapter(k=8)
    adapter(x)
    assert core.calls == [(0, 8), (8, 8), (16, 1)]

    short_adapter, short_core = make_adapter(k=32)
    logits = short_adapter(x)
    assert short_core.calls == [(0, 17)]
    assert logits.shape == (1, 3, 17, 2, 3)


def test_full_forward_resets_state_for_each_independent_tile():
    adapter, core = make_adapter(axis=2)
    tile_a = torch.ones(1, 4, 2, 3, 17)
    tile_b = torch.full_like(tile_a, 5)

    first_a = adapter(tile_a)
    adapter(tile_b)
    second_a = adapter(tile_a)

    torch.testing.assert_close(second_a, first_a)
    assert [call for call in core.calls if call[0] == 0] == [(0, 8)] * 3


def test_explicit_t0_continues_across_forward_calls_and_detach_delegates():
    x = torch.ones(1, 4, 2, 17, 3, requires_grad=True)
    adapter, core = make_adapter(axis=1)

    first = adapter(x[:, :, :, :8], t0=0)
    second = adapter(x[:, :, :, 8:], t0=8)

    torch.testing.assert_close(first[0, 0, 0, :, 0], torch.arange(1, 9, dtype=x.dtype) * 2 - 1)
    torch.testing.assert_close(second[0, 0, 0, :, 0], torch.arange(9, 18, dtype=x.dtype) * 2 - 1)
    assert core.calls == [(0, 8), (8, 8), (16, 1)]
    assert core.detach_calls == 0
    second.sum().backward()
    assert torch.count_nonzero(x.grad[:, :, :, :8]) > 0
    adapter.detach_states()
    assert core.detach_calls == 1


def test_forward_window_calls_core_once_with_absolute_t0():
    adapter, core = make_adapter(axis=2)
    x = torch.ones(1, 4, 2, 3, 4)
    adapter.forward_window(x[..., :1], t0=0)
    core.calls.clear()

    logits = adapter.forward_window(x, t0=11)

    assert core.calls == [(11, 4)]
    assert logits.shape == (1, 3, 2, 3, 4)


@pytest.mark.parametrize("chunk_size", [0, -1, True])
def test_invalid_chunk_sizes_are_rejected(chunk_size):
    adapter, _ = make_adapter()
    with pytest.raises(ValueError, match="chunk_size"):
        adapter(torch.ones(1, 4, 2, 3, 4), chunk_size=chunk_size)
