"""Construct the existing SNN core from settings stored in nnU-Net plans."""

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

from torch import nn

from model import build_model


_MODEL_KWARGS = (
    "patch_size",
    "linear_projection",
    "residual_connections",
    "dwconv2d_spiking",
    "patch_embedding_spiking",
    "vss_output_spiking",
    "input_skip",
)
_FIELDS = (
    "model_name",
    "model_kwargs",
    "temporal_axis",
    "k",
    "use_fptt",
    "fptt_alpha",
    "fptt_beta",
    "fptt_rho",
    "fptt_lambda",
    "num_input_channels",
    "num_output_channels",
)


def _require_keys(value: Mapping[str, Any], names: tuple[str, ...], label: str) -> None:
    missing = set(names) - value.keys()
    extra = value.keys() - set(names)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if extra:
            details.append(f"unexpected {', '.join(sorted(extra))}")
        raise ValueError(f"{label}: {'; '.join(details)}")


@dataclass(frozen=True)
class SNNConfig:
    model_name: str
    model_kwargs: Mapping[str, Any]
    temporal_axis: int
    k: int
    use_fptt: bool
    fptt_alpha: float
    fptt_beta: float
    fptt_rho: float
    fptt_lambda: float
    num_input_channels: int
    num_output_channels: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_kwargs", MappingProxyType(dict(self.model_kwargs)))

    @classmethod
    def from_plans(cls, plans: Mapping[str, Any]) -> "SNNConfig":
        if not isinstance(plans, Mapping) or "snn_config" not in plans:
            raise ValueError("plans must contain snn_config")
        data = plans["snn_config"]
        if not isinstance(data, Mapping):
            raise ValueError("snn_config must be a mapping")
        _require_keys(data, _FIELDS, "snn_config")

        if data["model_name"] not in {"orig", "shallow", "medium", "deep"}:
            raise ValueError("model_name must be orig, shallow, medium, or deep")
        kwargs = data["model_kwargs"]
        if not isinstance(kwargs, Mapping):
            raise ValueError("model_kwargs must be a mapping")
        _require_keys(kwargs, _MODEL_KWARGS, "model_kwargs")
        patch_size = kwargs["patch_size"]
        if type(patch_size) is not int or patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        for name in _MODEL_KWARGS[1:]:
            if type(kwargs[name]) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if kwargs["input_skip"] and data["model_name"] != "orig":
            raise ValueError("input_skip is supported only for model_name=orig")

        axis = data["temporal_axis"]
        if type(axis) is not int or axis not in (0, 1, 2):
            raise ValueError("temporal_axis must be 0, 1, or 2")
        k = data["k"]
        if type(k) is not int or k <= 0:
            raise ValueError("k must be a positive integer")
        if type(data["use_fptt"]) is not bool:
            raise ValueError("use_fptt must be a boolean")
        for name in ("fptt_alpha", "fptt_beta", "fptt_rho", "fptt_lambda"):
            value = data[name]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if type(data["num_input_channels"]) is not int or data["num_input_channels"] != 4:
            raise ValueError("num_input_channels must be 4")
        if type(data["num_output_channels"]) is not int or data["num_output_channels"] != 3:
            raise ValueError("num_output_channels must be 3")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_kwargs": dict(self.model_kwargs),
            "temporal_axis": self.temporal_axis,
            "k": self.k,
            "use_fptt": self.use_fptt,
            "fptt_alpha": self.fptt_alpha,
            "fptt_beta": self.fptt_beta,
            "fptt_rho": self.fptt_rho,
            "fptt_lambda": self.fptt_lambda,
            "num_input_channels": self.num_input_channels,
            "num_output_channels": self.num_output_channels,
        }


def build_core(config: SNNConfig) -> nn.Module:
    return build_model(
        config.model_name,
        out_channels=config.num_output_channels,
        **config.model_kwargs,
    )
