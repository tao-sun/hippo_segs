"""Parameter usage contracts for the real plans-driven SNN core."""

import inspect

import pytest
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from model import build_model as current_build_model
import snn_nnunet.network_adapter as adapter_module
from snn_nnunet.network_adapter import SNNConfig, build_core


def _output_spiking(module):
    return getattr(module, "output_spiking", getattr(module, "vss_output_spiking", None))


def _config(*, model_name="orig", use_fptt=False, output_spiking=True,
            output_key="output_spiking"):
    return SNNConfig.from_plans({"snn_config": {
        "model_name": model_name,
        "model_kwargs": {
            "patch_size": 4,
            "linear_projection": True,
            "residual_connections": True,
            "dwconv2d_spiking": True,
            "patch_embedding_spiking": False,
            "input_skip": False,
            output_key: output_spiking,
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


@pytest.mark.parametrize("output_key", ["output_spiking", "vss_output_spiking"])
def test_plans_accept_exactly_one_output_spiking_name(output_key):
    config = _config(output_key=output_key, output_spiking=False)

    assert config.to_dict()["model_kwargs"][output_key] is False
    core = build_core(config)
    assert _output_spiking(core) is False
    assert _output_spiking(core.conv_block1.spik_mamba) is False


@pytest.mark.parametrize("case", ["missing", "duplicate", "unknown"])
def test_plans_reject_missing_duplicate_or_unknown_output_settings(case):
    data = _config().to_dict()
    kwargs = data["model_kwargs"]
    if case == "missing":
        del kwargs["output_spiking"]
    elif case == "duplicate":
        kwargs["vss_output_spiking"] = True
    else:
        kwargs["unknown_flag"] = True

    with pytest.raises(ValueError, match="model_kwargs"):
        SNNConfig.from_plans({"snn_config": data})


@pytest.mark.parametrize("output_key", ["output_spiking", "vss_output_spiking"])
def test_build_core_maps_output_name_for_clean_factory_and_freezes_post_plif(
    monkeypatch, output_key
):
    def clean_factory(model_name, *, out_channels, patch_size, linear_projection,
                      residual_connections, dwconv2d_spiking, patch_embedding_spiking,
                      vss_output_spiking, input_skip):
        output_name = next(
            name for name in ("output_spiking", "vss_output_spiking")
            if name in inspect.signature(current_build_model).parameters
        )
        core = current_build_model(
            model_name, out_channels=out_channels, patch_size=patch_size,
            linear_projection=linear_projection,
            residual_connections=residual_connections,
            dwconv2d_spiking=dwconv2d_spiking,
            patch_embedding_spiking=patch_embedding_spiking,
            input_skip=input_skip, **{output_name: vss_output_spiking},
        )
        return core

    monkeypatch.setattr(adapter_module, "build_model", clean_factory)
    config = _config(output_key=output_key, output_spiking=False)
    core = build_core(config)
    parameters = dict(core.named_parameters())

    encoder = core.conv_block1.spik_mamba
    assert _output_spiking(encoder) is False
    assert all(not parameter.requires_grad for parameter in encoder.post_plif.parameters())
    assert parameters["conv_block1.spik_mamba.post_norm.weight"].requires_grad
    fptt_core = build_core(
        _config(output_key=output_key, output_spiking=False, use_fptt=True)
    )
    assert set(core.state_dict()) == set(fptt_core.state_dict())
    core.load_state_dict(fptt_core.state_dict(), strict=True)


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
    core = build_core(_config(output_spiking=False))
    parameters = dict(core.named_parameters())

    assert all(
        not parameter.requires_grad
        for parameter in core.conv_block1.spik_mamba.post_plif.parameters()
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
