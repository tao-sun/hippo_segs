"""Parameter usage contracts for the real plans-driven SNN core."""

import pytest
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from snn_nnunet.network_adapter import SNNConfig, build_core


def _config(*, model_name="orig", use_fptt=False, output_spiking=True):
    return SNNConfig.from_plans({"snn_config": {
        "model_name": model_name,
        "model_kwargs": {
            "patch_size": 4,
            "linear_projection": True,
            "residual_connections": True,
            "dwconv2d_spiking": True,
            "patch_embedding_spiking": False,
            "input_skip": False,
            "output_spiking": output_spiking,
        },
        "temporal_axis": 0,
        "k": 1,
        "use_fptt": use_fptt,
        "fptt_alpha": 0.5,
        "fptt_beta": 0.5,
        "fptt_rho": 0.0,
        "fptt_lambda": 2.0,
        "num_input_channels": 4,
        "num_output_channels": 3,
    }})


@pytest.mark.parametrize("alpha", [0, -0.1])
def test_fptt_rejects_nonpositive_alpha_before_core_construction(alpha):
    data = _config(use_fptt=True).to_dict()
    data["fptt_alpha"] = alpha

    with pytest.raises(ValueError, match="fptt_alpha"):
        SNNConfig.from_plans({"snn_config": data})


def test_non_fptt_orig_freezes_disconnected_convblock_parameters():
    core = build_core(_config())
    parameters = dict(core.named_parameters())

    for name in ("conv_block1", "conv_block2", "conv_block3"):
        assert not parameters[f"{name}.norm.weight"].requires_grad
        assert not parameters[f"{name}.norm.bias"].requires_grad
        assert not parameters[f"{name}.spike_neurons.w"].requires_grad
        assert parameters[f"{name}.spik_mamba.post_norm.weight"].requires_grad

    assert not parameters["class_conv.norm.weight"].requires_grad
    assert parameters["class_conv.conv.weight"].requires_grad
    assert set(core.state_dict()) == set(build_core(_config(use_fptt=True)).state_dict())


@pytest.mark.parametrize("model_name", ["shallow", "medium", "deep"])
def test_non_fptt_unet_freezes_unused_classifier_norm(model_name):
    parameters = dict(build_core(_config(model_name=model_name)).named_parameters())

    assert not parameters["class_conv.norm.weight"].requires_grad
    assert not parameters["class_conv.norm.bias"].requires_grad
    assert parameters["class_conv.conv.weight"].requires_grad
    assert parameters["enc1a.norm.weight"].requires_grad


def test_fptt_retains_disconnected_parameters_for_regularizer():
    parameters = dict(build_core(_config(use_fptt=True)).named_parameters())

    assert parameters["conv_block1.norm.weight"].requires_grad
    assert parameters["conv_block1.spike_neurons.w"].requires_grad
    assert parameters["class_conv.norm.weight"].requires_grad
    assert parameters["conv_block1.spik_mamba.post_norm.weight"].requires_grad


def test_non_fptt_output_spiking_off_has_no_trainable_encoder_post_plif():
    parameters = dict(build_core(_config(output_spiking=False)).named_parameters())

    assert not any(
        parameter.requires_grad
        for name, parameter in parameters.items()
        if name.startswith("conv_block1.spik_mamba.post_plif.")
    )
    assert parameters["conv_block1.spik_mamba.post_norm.weight"].requires_grad


def _ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        core = build_core(_config(model_name="shallow"))
        network = DistributedDataParallel(core)
        optimizer = torch.optim.SGD(network.parameters(), lr=1e-3)
        for t0 in range(2):
            optimizer.zero_grad(set_to_none=True)
            inputs = torch.full((1, 1, 4, 16, 16), 0.1 * (rank + 1))
            loss = network(inputs, t0=t0).square().mean()
            loss.backward()
            optimizer.step()
            core.detach_states()
    finally:
        dist.destroy_process_group()


def test_non_fptt_core_trains_across_two_ddp_ranks(tmp_path):
    torch.multiprocessing.spawn(
        _ddp_worker, args=(str(tmp_path / "ddp-init"),), nprocs=2, join=True
    )
