from dataclasses import FrozenInstanceError

import pytest

from model import (
    SNNBraTS,
    SNNBraTSUNetDeep,
    SNNBraTSUNetMedium,
    SNNBraTSUNetShallow,
    build_model,
)
from snn_nnunet.network_adapter import SNNConfig, build_core
from snn_fptt import build_model as build_legacy_model


APPROVED_CONFIG = {
    "model_name": "orig",
    "model_kwargs": {
        "patch_size": 4,
        "linear_projection": True,
        "residual_connections": True,
        "dwconv2d_spiking": True,
        "patch_embedding_spiking": True,
        "vss_output_spiking": True,
        "input_skip": False,
    },
    "temporal_axis": 0,
    "k": 16,
    "use_fptt": True,
    "fptt_alpha": 0.5,
    "fptt_beta": 0.5,
    "fptt_rho": 0.0,
    "fptt_lambda": 2.0,
    "num_input_channels": 4,
    "num_output_channels": 3,
}


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("orig", SNNBraTS),
        ("shallow", SNNBraTSUNetShallow),
        ("medium", SNNBraTSUNetMedium),
        ("deep", SNNBraTSUNetDeep),
    ],
)
def test_shared_factory_dispatches_existing_models(name, expected_type):
    model = build_model(name)
    assert type(model) is expected_type
    assert model.class_conv.conv.out_channels == 3


def test_shared_factory_passes_all_orig_architecture_settings():
    model = build_model(
        "orig",
        patch_size=4,
        linear_projection=False,
        residual_connections=False,
        dwconv2d_spiking=False,
        patch_embedding_spiking=True,
        vss_output_spiking=False,
        input_skip=True,
    )
    assert model.patch_size == 4
    assert model.linear_projection is False
    assert model.residual_connections is False
    assert model.dwconv2d_spiking is False
    assert model.patch_embedding_spiking is True
    assert model.vss_output_spiking is False
    assert model.input_skip is True


def test_legacy_factory_retains_shared_factory_defaults():
    model = build_legacy_model("orig")
    assert model.patch_size == 4
    assert model.patch_embedding_spiking is False
    assert model.vss_output_spiking is True
    assert model.input_skip is False


@pytest.mark.parametrize("factory", [build_model, build_legacy_model])
def test_factories_reject_unknown_names(factory):
    with pytest.raises(ValueError, match="Unknown model"):
        factory("unknown")


def test_non_orig_model_rejects_input_skip():
    with pytest.raises(ValueError, match="input_skip"):
        build_model("shallow", input_skip=True)


def test_plans_config_round_trips_exact_approved_schema():
    config = SNNConfig.from_plans({"snn_config": APPROVED_CONFIG})
    assert config.to_dict() == APPROVED_CONFIG
    assert config.model_name == "orig"
    assert config.model_kwargs["patch_embedding_spiking"] is True
    assert type(build_core(config)) is SNNBraTS


def test_plans_config_is_immutable():
    config = SNNConfig.from_plans({"snn_config": APPROVED_CONFIG})
    with pytest.raises(FrozenInstanceError):
        config.k = 2
    with pytest.raises(TypeError):
        config.model_kwargs["patch_size"] = 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_name", "missing"),
        ("num_input_channels", 3),
        ("num_output_channels", 4),
        ("temporal_axis", -1),
        ("temporal_axis", 3),
        ("temporal_axis", True),
        ("k", 0),
        ("k", -1),
        ("k", True),
        ("use_fptt", "true"),
    ],
)
def test_plans_config_rejects_invalid_top_level_values(field, value):
    data = {**APPROVED_CONFIG, field: value}
    with pytest.raises(ValueError, match=field):
        SNNConfig.from_plans({"snn_config": data})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("patch_size", 0),
        ("patch_size", True),
        ("linear_projection", 1),
        ("residual_connections", "false"),
        ("dwconv2d_spiking", 0),
        ("patch_embedding_spiking", "true"),
        ("vss_output_spiking", None),
        ("input_skip", 1),
    ],
)
def test_plans_config_rejects_invalid_model_kwargs(field, value):
    data = {**APPROVED_CONFIG, "model_kwargs": {**APPROVED_CONFIG["model_kwargs"], field: value}}
    with pytest.raises(ValueError, match=field):
        SNNConfig.from_plans({"snn_config": data})


@pytest.mark.parametrize("field", ["model_name", "k", "fptt_alpha", "num_input_channels"])
def test_plans_config_requires_every_top_level_field(field):
    data = {key: value for key, value in APPROVED_CONFIG.items() if key != field}
    with pytest.raises(ValueError, match=field):
        SNNConfig.from_plans({"snn_config": data})


def test_plans_config_requires_all_model_kwargs_and_rejects_extras():
    data = {**APPROVED_CONFIG, "model_kwargs": {"patch_size": 4}}
    with pytest.raises(ValueError, match="linear_projection"):
        SNNConfig.from_plans({"snn_config": data})
    data["model_kwargs"] = {**APPROVED_CONFIG["model_kwargs"], "extra": True}
    with pytest.raises(ValueError, match="extra"):
        SNNConfig.from_plans({"snn_config": data})


def test_plans_config_rejects_missing_or_extra_schema():
    with pytest.raises(ValueError, match="snn_config"):
        SNNConfig.from_plans({})
    data = {**APPROVED_CONFIG, "extra": 1}
    with pytest.raises(ValueError, match="extra"):
        SNNConfig.from_plans({"snn_config": data})
