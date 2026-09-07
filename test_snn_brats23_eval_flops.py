from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from snn_brats23_eval_flops import (
    HybridOperationProfiler,
    aggregate_volume_profiles,
    evaluate,
    load_testing_config,
    model_spec_from_checkpoint,
)
from spike_neurons import PLIFNode


def test_plif_detach_leaves_scalar_reset_state_unchanged():
    node = PLIFNode(init_tau=2.0, v_threshold=1.0, v_reset=0.0)
    node.reset()

    node.detach()

    assert node.v == 0.0


def test_load_testing_config_resolves_paths_and_accepts_requested_fold_view(tmp_path):
    config_path = tmp_path / "testing.yaml"
    config_path.write_text(
        "\n".join(
            [
                "checkpoint: runs/example/checkpoint_best.pt",
                "data_root: datasets/brats23",
                "view: coronal",
                "fold: 3",
                "device: cpu",
                "tbptt_k: null",
                "threshold: 0.4",
                "num_workers: 0",
                "max_subjects: 2",
            ]
        ),
        encoding="utf-8",
    )

    config = load_testing_config(config_path, require_paths_exist=False)

    assert config["checkpoint"] == (tmp_path / "runs/example/checkpoint_best.pt").resolve()
    assert config["data_root"] == (tmp_path / "datasets/brats23").resolve()
    assert config["view"] == "coronal"
    assert config["fold"] == 3
    assert config["tbptt_k"] is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("view", "transverse", "view"), ("fold", 6, "fold")],
)
def test_load_testing_config_rejects_invalid_fold_or_view(tmp_path, field, value, message):
    values = {
        "checkpoint": "checkpoint.pt",
        "data_root": "dataset",
        "view": "axial",
        "fold": 1,
    }
    values[field] = value
    config_path = tmp_path / "testing.yaml"
    config_path.write_text(
        "\n".join(f"{key}: {item}" for key, item in values.items()),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_testing_config(config_path, require_paths_exist=False)


class TinySpikingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_input = nn.Conv2d(1, 2, kernel_size=1, bias=False)
        self.plif = PLIFNode(init_tau=2.0, v_threshold=1.0, v_reset=0.0)
        self.spike_driven = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.dense_input.weight.copy_(torch.tensor([[[[2.0]]], [[[0.0]]]]))
            self.spike_driven.weight.fill_(1.0)

    def forward(self, inputs):
        currents = self.dense_input(inputs)
        spikes, _ = self.plif(currents, time_step=0)
        return self.spike_driven(spikes)


def test_profiler_counts_dense_layer_multiplications_but_spike_driven_layer_only_additions():
    model = TinySpikingNetwork().eval()
    profiler = HybridOperationProfiler(model)
    profiler.start_volume()
    with torch.no_grad():
        model(torch.ones(1, 1, 2, 2))
    volume = profiler.finish_volume()
    profiler.close()

    plif = volume.plif_layers["plif"]
    assert plif.firing_rate == pytest.approx(0.5)

    dense = volume.weighted_layers["dense_input"]
    assert dense.is_spike_driven is False
    assert dense.dense_multiplications == 8
    assert dense.effective_multiplications == 8

    spike_driven = volume.weighted_layers["spike_driven"]
    assert spike_driven.is_spike_driven is True
    assert spike_driven.input_firing_rate == pytest.approx(0.5)
    assert spike_driven.dense_additions == 4
    assert spike_driven.effective_additions == pytest.approx(2.0)
    assert spike_driven.effective_multiplications == 0


def test_aggregate_volume_profiles_returns_mean_per_volume():
    model = TinySpikingNetwork().eval()
    profiler = HybridOperationProfiler(model)
    profiles = []
    for inputs in (torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)):
        profiler.start_volume()
        with torch.no_grad():
            model(inputs)
        profiles.append(profiler.finish_volume())
    profiler.close()

    summary = aggregate_volume_profiles(profiles)

    assert summary["n_volumes"] == 2
    assert summary["plif_layers"]["plif"]["mean_firing_rate"] == pytest.approx(0.25)
    assert summary["weighted_layers"]["spike_driven"]["mean_effective_additions"] == pytest.approx(1.0)
    assert summary["mean_total_effective_additions"] == pytest.approx(1.0)
    assert summary["mean_total_effective_multiplications"] == pytest.approx(8.0)
    assert summary["mean_total_effective_operations"] == pytest.approx(9.0)


def test_model_spec_uses_architecture_saved_by_snn_fptt_checkpoint():
    checkpoint = {
        "model": {"some.weight": torch.ones(1)},
        "config": {
            "model": "orig",
            "patch_size": 2,
            "linear_projection": True,
            "residual_connections": False,
            "dwconv2d_spiking": True,
            "patch_embedding_spiking": False,
            "tbptt_k": 4,
            "prob_threshold": 0.35,
        },
    }

    spec = model_spec_from_checkpoint(checkpoint)

    assert spec == {
        "model_name": "orig",
        "patch_size": 2,
        "linear_projection": True,
        "residual_connections": False,
        "dwconv2d_spiking": True,
        "patch_embedding_spiking": False,
        "num_encoder_stages": 3,
        "tbptt_k": 4,
        "prob_threshold": 0.35,
    }


def test_model_spec_detects_four_stage_fptt_checkpoint_from_state_dict():
    checkpoint = {
        "model": {"conv_block4.spik_mamba.patch_embed.proj.weight": torch.ones(1)},
        "config": {"model": "orig"},
    }

    spec = model_spec_from_checkpoint(checkpoint)

    assert spec["num_encoder_stages"] == 4


class FunctionalConvSpikingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.plif = PLIFNode(init_tau=2.0, v_threshold=1.0, v_reset=0.0)
        self.weight = nn.Parameter(torch.ones(1, 2, 1))

    def forward(self, currents):
        spikes, _ = self.plif(currents, time_step=0)
        return F.conv1d(spikes, self.weight)


def test_profiler_includes_functional_conv1d_used_by_spikmamba():
    model = FunctionalConvSpikingNetwork().eval()
    profiler = HybridOperationProfiler(model)
    profiler.start_volume()
    currents = torch.tensor([[[2.0, 2.0], [0.0, 0.0]]])
    with torch.no_grad():
        model(currents)
    volume = profiler.finish_volume()
    profiler.close()

    functional = volume.weighted_layers["functional_conv1d_1"]
    assert functional.is_spike_driven is True
    assert functional.input_firing_rate == pytest.approx(0.5)
    assert functional.dense_additions == 2
    assert functional.effective_additions == pytest.approx(1.0)
    assert functional.effective_multiplications == 0


class ReusedWeightedLayerNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.plif = PLIFNode(init_tau=2.0, v_threshold=1.0, v_reset=0.0)
        self.shared = nn.Conv2d(2, 1, kernel_size=1, bias=False)

    def forward(self, currents):
        spikes, _ = self.plif(currents, time_step=0)
        spike_result = self.shared(spikes)
        dense_result = self.shared(torch.full_like(currents, 0.25))
        return spike_result + dense_result


class DenseThenSpikeReusedLayerNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.plif = PLIFNode(init_tau=2.0, v_threshold=1.0, v_reset=0.0)

    def forward(self, currents):
        dense_result = self.shared(currents)
        spikes, _ = self.plif(currents, time_step=0)
        spike_result = self.shared(spikes)
        return dense_result + spike_result


def test_profiler_recomputes_provenance_when_one_layer_has_spike_and_dense_calls():
    model = ReusedWeightedLayerNetwork().eval()
    profiler = HybridOperationProfiler(model)
    profiler.start_volume()
    currents = torch.tensor([[[[2.0, 2.0], [2.0, 2.0]], [[0.0, 0.0], [0.0, 0.0]]]])
    with torch.no_grad():
        model(currents)
    volume = profiler.finish_volume()
    profiler.close()

    shared = volume.weighted_layers["shared"]
    assert shared.calls == 2
    assert shared.spike_driven_calls == 1
    assert shared.dense_calls == 1
    assert shared.effective_additions == pytest.approx(6.0)
    assert shared.effective_multiplications == pytest.approx(8.0)


def test_profiler_recomputes_provenance_when_dense_call_precedes_spike_call():
    model = DenseThenSpikeReusedLayerNetwork().eval()
    profiler = HybridOperationProfiler(model)
    profiler.start_volume()
    currents = torch.tensor([[[[2.0, 2.0], [2.0, 2.0]], [[0.0, 0.0], [0.0, 0.0]]]])
    with torch.no_grad():
        model(currents)
    volume = profiler.finish_volume()
    profiler.close()

    shared = volume.weighted_layers["shared"]
    assert shared.calls == 2
    assert shared.spike_driven_calls == 1
    assert shared.dense_calls == 1
    assert shared.effective_additions == pytest.approx(6.0)
    assert shared.effective_multiplications == pytest.approx(8.0)


def test_profiler_counts_only_valid_padded_conv_transpose_contributions():
    model = nn.ConvTranspose2d(
        1, 1, kernel_size=3, stride=1, padding=1, bias=False
    ).eval()
    profiler = HybridOperationProfiler(model)
    profiler.start_volume()
    with torch.no_grad():
        model(torch.full((1, 1, 2, 2), 0.25))
    volume = profiler.finish_volume()
    profiler.close()

    layer = volume.weighted_layers[""]
    assert layer.dense_multiplications == 16
    assert layer.dense_additions == 12
    assert layer.effective_multiplications == 16
    assert layer.effective_additions == 12


def test_profiler_counts_only_valid_zero_padded_conv_contributions():
    model = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).eval()
    profiler = HybridOperationProfiler(model)
    profiler.start_volume()
    with torch.no_grad():
        model(torch.full((1, 1, 2, 2), 0.25))
    volume = profiler.finish_volume()
    profiler.close()

    layer = volume.weighted_layers[""]
    assert layer.dense_multiplications == 16
    assert layer.dense_additions == 12


class PaddedDenseSlicesNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.plif = PLIFNode(init_tau=2.0, v_threshold=1.0, v_reset=0.0)

    def forward(self, volume):
        outputs = []
        for time_step in range(volume.shape[1]):
            padded = F.pad(volume[:, time_step], (0, 1, 0, 1))
            current = self.shared(padded)
            outputs.append(self.plif(current, time_step=time_step)[0])
        return torch.stack(outputs, dim=1)


def test_profiler_keeps_zero_padded_raw_slice_dense_after_a_plif_has_run():
    model = PaddedDenseSlicesNetwork().eval()
    volume_input = torch.tensor([0.25, 0.0]).reshape(1, 2, 1, 1, 1)
    profiler = HybridOperationProfiler(model)
    profiler.start_volume()
    with torch.no_grad():
        model(volume_input)
    volume = profiler.finish_volume()
    profiler.close()

    shared = volume.weighted_layers["shared"]
    assert shared.calls == 2
    assert shared.dense_calls == 2
    assert shared.spike_driven_calls == 0
    assert shared.effective_multiplications == 8


class ConcatenatedSpikesNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.left_plif = PLIFNode(init_tau=2.0, v_threshold=1.0, v_reset=0.0)
        self.right_plif = PLIFNode(init_tau=2.0, v_threshold=1.0, v_reset=0.0)
        self.synapse = nn.Conv2d(2, 1, kernel_size=1, bias=False)

    def forward(self, currents):
        left = self.left_plif(currents[:, :1], time_step=0)[0]
        right = self.right_plif(currents[:, 1:], time_step=0)[0]
        return self.synapse(torch.cat([left, right], dim=1))


def test_profiler_propagates_spike_provenance_through_concatenation():
    model = ConcatenatedSpikesNetwork().eval()
    profiler = HybridOperationProfiler(model)
    profiler.start_volume()
    currents = torch.tensor([[[[2.0]], [[0.0]]]])
    with torch.no_grad():
        model(currents)
    volume = profiler.finish_volume()
    profiler.close()

    synapse = volume.weighted_layers["synapse"]
    assert synapse.spike_driven_calls == 1
    assert synapse.input_firing_rate == pytest.approx(0.5)
    assert synapse.effective_multiplications == 0


class PooledSpikesNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.plif = PLIFNode(init_tau=2.0, v_threshold=1.0, v_reset=0.0)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.synapse = nn.Conv2d(1, 1, kernel_size=1, bias=False)

    def forward(self, currents):
        spikes = self.plif(currents, time_step=0)[0]
        return self.synapse(self.pool(spikes))


def test_profiler_propagates_spike_provenance_through_max_pool():
    model = PooledSpikesNetwork().eval()
    profiler = HybridOperationProfiler(model)
    profiler.start_volume()
    with torch.no_grad():
        model(torch.full((1, 1, 4, 4), 2.0))
    volume = profiler.finish_volume()
    profiler.close()

    synapse = volume.weighted_layers["synapse"]
    assert synapse.spike_driven_calls == 1
    assert synapse.input_firing_rate == pytest.approx(1.0)
    assert synapse.effective_multiplications == 0


def _fake_selective_scan(
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
    del delta, A, B, C, D, z, delta_bias, delta_softplus
    output = torch.zeros_like(u)
    if return_last_state:
        return output, torch.empty(0, device=u.device)
    return output


class SpikeMambaLayer(nn.Module):
    """Minimal real-call boundary for profiling a selective scan."""

    def __init__(self, binary_input: bool):
        super().__init__()
        self.patch_embedding_spiking = False
        self.dwconv2d_spiking = binary_input
        self.selective_scan = _fake_selective_scan

    def forward(self, u):
        batch, channels, length = u.shape
        state_size = 2
        delta = torch.ones_like(u)
        A = torch.ones(channels, state_size, device=u.device)
        state_B = torch.ones(batch, 1, state_size, length, device=u.device)
        state_C = torch.ones_like(state_B)
        D = torch.ones(channels, device=u.device)
        delta_bias = torch.ones(channels, device=u.device)
        return self.selective_scan(
            u,
            delta,
            A,
            state_B,
            state_C,
            D,
            z=None,
            delta_bias=delta_bias,
            delta_softplus=True,
            return_last_state=False,
        )


class TinySelectiveScanNetwork(nn.Module):
    def __init__(self, binary_input: bool):
        super().__init__()
        self.scan = SpikeMambaLayer(binary_input)

    def forward(self, inputs):
        return self.scan(inputs)


def test_profiler_counts_selective_scan_with_binary_input_integration():
    model = TinySelectiveScanNetwork(binary_input=True).eval()
    original_scan = model.scan.selective_scan
    profiler = HybridOperationProfiler(model)
    profiler.start_volume()
    scan_input = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])

    with torch.no_grad():
        model(scan_input)

    volume = profiler.finish_volume()
    profiler.close()

    scan = volume.selective_scan_layers["scan.selective_scan"]
    assert scan.is_binary_input is True
    assert scan.input_firing_rate == pytest.approx(0.5)
    assert scan.dense_multiplications == 36
    assert scan.input_gated_multiplications == 6
    assert scan.dense_additions == 12
    assert scan.spike_driven_additions == 9
    assert scan.effective_multiplications == 42
    assert scan.effective_additions == 21
    assert scan.exp_operations == 16
    assert scan.softplus_operations == 6
    assert volume.total_effective_operations == 85
    assert model.scan.selective_scan is original_scan


def test_profiler_counts_dense_selective_scan_without_binary_shortcuts():
    model = TinySelectiveScanNetwork(binary_input=False).eval()
    profiler = HybridOperationProfiler(model)
    profiler.start_volume()

    with torch.no_grad():
        model(torch.full((1, 2, 3), 0.25))

    volume = profiler.finish_volume()
    profiler.close()

    scan = volume.selective_scan_layers["scan.selective_scan"]
    assert scan.is_binary_input is False
    assert scan.dense_multiplications == 66
    assert scan.input_gated_multiplications == 0
    assert scan.dense_additions == 30
    assert scan.spike_driven_additions == 0
    assert scan.effective_multiplications == 66
    assert scan.effective_additions == 30


def test_aggregate_includes_selective_scan_in_effective_operation_totals():
    model = TinySelectiveScanNetwork(binary_input=True).eval()
    profiler = HybridOperationProfiler(model)
    profiles = []
    inputs = (
        torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]),
        torch.zeros(1, 2, 3),
    )
    for scan_input in inputs:
        profiler.start_volume()
        with torch.no_grad():
            model(scan_input)
        profiles.append(profiler.finish_volume())
    profiler.close()

    summary = aggregate_volume_profiles(profiles)

    scan = summary["selective_scan_layers"]["scan.selective_scan"]
    assert scan["is_binary_input"] is True
    assert scan["mean_input_firing_rate"] == pytest.approx(0.25)
    assert scan["mean_dense_additions"] == 12
    assert scan["mean_spike_driven_additions"] == pytest.approx(4.5)
    assert scan["mean_dense_multiplications"] == 36
    assert scan["mean_input_gated_multiplications"] == pytest.approx(3.0)
    assert scan["mean_exp_operations"] == 16
    assert scan["mean_softplus_operations"] == 6
    assert summary["mean_total_nonlinear_operations"] == pytest.approx(22.0)
    assert summary["mean_total_effective_additions"] == pytest.approx(16.5)
    assert summary["mean_total_effective_multiplications"] == pytest.approx(39.0)
    assert summary["mean_total_effective_operations"] == pytest.approx(77.5)


class FourDirectionSpikeMambaLayer(nn.Module):
    """Small real forward boundary with the same four-route merge as SpikeMamba."""

    def __init__(self):
        super().__init__()
        self.patch_embedding_spiking = False
        self.dwconv2d_spiking = False
        self.num_directions = 4
        self.d_inner = 2
        self.selective_scan = _fake_selective_scan

    def forward(self, tokens):
        batch, length, _ = tokens.shape
        channels = self.num_directions * self.d_inner
        u = torch.zeros(batch, channels, length, device=tokens.device)
        delta = torch.ones_like(u)
        A = torch.ones(channels, 2, device=tokens.device)
        state = torch.ones(batch, 1, 2, length, device=tokens.device)
        self.selective_scan(u, delta, A, state, state)
        merged = torch.zeros(batch, length, self.d_inner, device=tokens.device)
        return 0.25 * (merged + merged + merged + merged)


# The profiler recognizes the production module boundary by its class name.
FourDirectionSpikeMambaLayer.__name__ = "SpikeMambaLayer"


def test_profiler_includes_four_direction_merge_additions_and_scaling():
    model = FourDirectionSpikeMambaLayer().eval()
    profiler = HybridOperationProfiler(model)
    profiler.start_volume()

    with torch.no_grad():
        model(torch.zeros(1, 3, 2))

    volume = profiler.finish_volume()
    profiler.close()

    merge = volume.elementwise_layers["direction_merge"]
    assert merge.additions == 18
    assert merge.multiplications == 6


class SpikMambaBlock(nn.Module):
    def __init__(self, residual_connections):
        super().__init__()
        self.residual_connections = residual_connections

    def forward(self, tokens):
        transformed = torch.ones_like(tokens)
        if self.residual_connections:
            return tokens + transformed
        return transformed


@pytest.mark.parametrize(
    ("residual_connections", "expected_additions"),
    [(True, 6), (False, 0)],
)
def test_profiler_counts_residual_additions_only_when_enabled(
    residual_connections, expected_additions
):
    model = SpikMambaBlock(residual_connections).eval()
    profiler = HybridOperationProfiler(model)
    profiler.start_volume()

    with torch.no_grad():
        model(torch.zeros(1, 3, 2))

    volume = profiler.finish_volume()
    profiler.close()

    residual = volume.elementwise_layers.get("residual")
    assert (0 if residual is None else residual.additions) == expected_additions


def test_profiled_sigmoid_is_included_in_nonlinear_and_operation_totals():
    model = nn.Identity().eval()
    profiler = HybridOperationProfiler(model)
    profiler.start_volume()
    logits = torch.zeros(1, 3, 2, 2)

    probabilities = profiler.profiled_sigmoid(logits)

    volume = profiler.finish_volume()
    profiler.close()

    assert torch.equal(probabilities, torch.full_like(logits, 0.5))
    assert volume.nonlinear_layers["output_sigmoid"].operations == 12
    assert volume.total_effective_operations == 12


class OneTinyVolume(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, _index):
        slices = torch.zeros(1, 4, 2, 2)
        targets = torch.zeros(1, 3, 2, 2)
        return slices, targets, {"sid": "tiny", "xyz": (1, 2, 2)}


class ZeroLogitNetwork(nn.Module):
    def forward(self, x_window, t0=0):
        del t0
        batch, time_steps, _, height, width = x_window.shape
        return torch.zeros(
            batch, 3, time_steps, height, width, device=x_window.device
        )


def test_evaluate_includes_output_sigmoid_in_each_volume_total():
    _, profiles = evaluate(
        ZeroLogitNetwork().eval(),
        DataLoader(OneTinyVolume()),
        view="sagittal",
        device=torch.device("cpu"),
        tbptt_k=1,
        threshold=0.5,
    )

    sigmoid = profiles[0].nonlinear_layers["output_sigmoid"]
    assert sigmoid.operations == 12
    assert profiles[0].total_effective_operations == 12
