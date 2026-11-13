"""Construct the existing SNN core from settings stored in nnU-Net plans."""

from dataclasses import dataclass
import inspect
import math
from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from snn_nnunet.model import ConvBlock, build_model, ARCHITECTURE_DEFAULTS, architecture_options
from snn_nnunet.config_aliases import normalize_spiking_options
from snn_nnunet.paper_config import normalize_bidirectional_fusion
from snn_nnunet.precision import FP32Module, fp32_forward


_MODEL_KWARGS = (
    "patch_size",
    "linear_projection",
    "residual_connections",
    "dwconv2d_spiking",
    "patch_embedding_spiking",
    "input_skip",
)
_OUTPUT_SPIKING_KEYS = ("output_spiking", "vss_output_spiking")
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
    "num_epochs",
    "overfit_num_volumes",
    "train_batches_per_epoch",
    "val_batches_per_epoch",
    "initial_lr",
    "validation_every",
    "validation_patch_size",
    "seed",
    "deterministic",
    "augmentation_workers",
    "bidirectional",
    "bidirectional_fusion",
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


_TEMPORAL_MAMBA_DEFAULTS = {
    "enabled": False,
    "kernel_size": 3,
    "d_state": 16,
    "dt_rank": "auto",
    "debug_shapes": False,
    "norm_spiking": True,
    "dwconv1D_spiking": True,
}


def _validate_temporal_mamba(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("temporal_mamba must be a mapping")
    allowed = _TEMPORAL_MAMBA_DEFAULTS.keys() | {"plif_enabled"}
    if value.keys() - allowed:
        raise ValueError("temporal_mamba contains unexpected options")
    result = dict(_TEMPORAL_MAMBA_DEFAULTS)
    legacy_plif = value.get("plif_enabled")
    if legacy_plif is not None:
        if type(legacy_plif) is not bool:
            raise ValueError("temporal_mamba.plif_enabled must be a boolean")
        result["norm_spiking"] = legacy_plif
        result["dwconv1D_spiking"] = legacy_plif
    result.update(value)
    result.pop("plif_enabled", None)
    if type(result["enabled"]) is not bool:
        raise ValueError("temporal_mamba.enabled must be a boolean")
    for name in ("debug_shapes", "norm_spiking", "dwconv1D_spiking"):
        if type(result[name]) is not bool:
            raise ValueError(f"temporal_mamba.{name} must be a boolean")
    for name in ("kernel_size", "d_state"):
        if type(result[name]) is not int or result[name] <= 0:
            raise ValueError(f"temporal_mamba.{name} must be a positive integer")
    dt_rank = result["dt_rank"]
    if dt_rank != "auto" and (type(dt_rank) is not int or dt_rank <= 0):
        raise ValueError("temporal_mamba.dt_rank must be 'auto' or a positive integer")
    return result


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
    num_epochs: int = 300
    overfit_num_volumes: int = 0
    train_batches_per_epoch: int = 250
    val_batches_per_epoch: int = 50
    initial_lr: float = 0.01
    validation_every: int = 0
    validation_patch_size: tuple[int, int, int] = (128, 128, 128)
    seed: int = 42
    deterministic: bool = False
    augmentation_workers: int = 8
    bidirectional: bool = False
    bidirectional_fusion: str = 'conv_plif'

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_kwargs", MappingProxyType(dict(self.model_kwargs)))

    @property
    def requires_full_sequence(self) -> bool:
        return self.bidirectional or (
            self.model_name == 'paper_unet' and self.model_kwargs.get('temporal_mamba', False))

    @classmethod
    def from_plans(cls, plans: Mapping[str, Any]) -> "SNNConfig":
        if not isinstance(plans, Mapping) or "snn_config" not in plans:
            raise ValueError("plans must contain snn_config")
        data = plans["snn_config"]
        if not isinstance(data, Mapping):
            raise ValueError("snn_config must be a mapping")
        data = dict(data)
        data.setdefault("num_epochs", 300)
        data.setdefault("overfit_num_volumes", 0)
        if type(data["overfit_num_volumes"]) is not int or data["overfit_num_volumes"] < 0:
            raise ValueError("overfit_num_volumes must be a nonnegative integer")
        data.setdefault("train_batches_per_epoch", 250)
        data.setdefault("val_batches_per_epoch", 50)
        data.setdefault("initial_lr", 0.01)
        data.setdefault("validation_every", 0)
        data.setdefault("validation_patch_size", (128, 128, 128))
        data.setdefault("seed", 42)
        data.setdefault("deterministic", False)
        data.setdefault("augmentation_workers", 8)
        data.setdefault("bidirectional", False)
        if type(data['bidirectional']) is not bool:
            raise ValueError('bidirectional must be boolean')
        if data['bidirectional'] and data.get('model_name') != 'paper_unet':
            raise ValueError('bidirectional requires model_name=paper_unet')
        data['bidirectional_fusion'] = normalize_bidirectional_fusion(
            data.get('bidirectional_fusion', 'conv_plif'), enabled=data['bidirectional'])
        _require_keys(data, _FIELDS, "snn_config")
        from snn_nnunet.reproducibility import validate_policy
        validate_policy(data['seed'], data['deterministic'], data['augmentation_workers'])

        if data["model_name"] not in {"orig", "shallow", "medium", "deep", "paper_unet", "arch2"}:
            raise ValueError("model_name must be orig, shallow, medium, deep, paper_unet, or arch2")
        kwargs = data["model_kwargs"]
        if not isinstance(kwargs, Mapping):
            raise ValueError("model_kwargs must be a mapping")
        if data["model_name"] == "arch2":
            from snn_nnunet.arch2_config import arch2_options
            kwargs = arch2_options(kwargs)
        elif data["model_name"] == "paper_unet":
            from snn_nnunet.paper_config import paper_options
            kwargs = paper_options(kwargs)
            if kwargs['previous_probability_feedback']:
                if data['bidirectional']:
                    raise ValueError('previous_probability_feedback requires bidirectional=false')
                if (data['num_input_channels'], data['num_output_channels']) != (4, 3):
                    raise ValueError('previous_probability_feedback requires four MRI inputs and three [WT, TC, ET] outputs')
            from snn_nnunet.paper_config import validate_shared_soma_options
            validate_shared_soma_options(
                kwargs, enabled=data['bidirectional'], fusion=data['bidirectional_fusion'])
        else:
            kwargs = normalize_spiking_options(kwargs)
            output_keys = tuple(name for name in _OUTPUT_SPIKING_KEYS if name in kwargs)
            output_key = output_keys[0] if len(output_keys) == 1 else "output_spiking"
            required_kwargs = set(_MODEL_KWARGS + (output_key,))
            allowed_kwargs = required_kwargs | {"temporal_mamba", "spatial_output_spiking"} | ARCHITECTURE_DEFAULTS.keys()
            missing = required_kwargs - kwargs.keys()
            extra = kwargs.keys() - allowed_kwargs
            if missing or extra:
                details = []
                if missing:
                    details.append(f"missing {', '.join(sorted(missing))}")
                if extra:
                    details.append(f"unexpected {', '.join(sorted(extra))}")
                raise ValueError(f"model_kwargs: {'; '.join(details)}")
            if len(output_keys) != 1:
                raise ValueError("model_kwargs must contain exactly one output_spiking or vss_output_spiking")
            kwargs = dict(kwargs)
            validated_architecture = architecture_options(kwargs)
            for key in ARCHITECTURE_DEFAULTS:
                if key in kwargs:
                    kwargs[key] = validated_architecture[key]
            # Keep legacy serialized plans identical when the optional branch is absent.
            if kwargs.get("recurrent_neuron") is None:
                kwargs.pop("recurrent_neuron", None)
            if kwargs.get("recurrent_neuron") is not None and data["model_name"] != "orig":
                raise ValueError("recurrent_neuron requires model_name=orig")
            if kwargs.get("decoder_block", "standard") != "standard" and data["model_name"] != "orig":
                raise ValueError("decoder_block=resdecoder is supported only for model_name=orig")
            if output_key == "vss_output_spiking":
                # Normalize old plans to the general output switch.
                kwargs["output_spiking"] = kwargs.pop("vss_output_spiking")
            if "spatial_output_spiking" in kwargs and type(kwargs["spatial_output_spiking"]) is not bool:
                raise ValueError("spatial_output_spiking must be a boolean")
            if "temporal_mamba" in kwargs:
                kwargs["temporal_mamba"] = _validate_temporal_mamba(kwargs["temporal_mamba"])
                if kwargs["temporal_mamba"]["enabled"] and data["model_name"] != "orig":
                    raise ValueError("temporal_mamba.enabled=true is supported only for model_name=orig")
            if any(kwargs.get(key, False) for key in ("fue_skip1", "fue_skip2")) and data["model_name"] != "orig":
                raise ValueError("FUE skips are supported only for model_name=orig")
            patch_size = kwargs["patch_size"]
            if type(patch_size) is not int or patch_size <= 0:
                raise ValueError("patch_size must be a positive integer")
            for name in _MODEL_KWARGS[1:] + ("output_spiking",):
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
        temporal_mamba = data['model_name'] == 'paper_unet' and kwargs['temporal_mamba']
        feedback = data['model_name'] == 'paper_unet' and kwargs['previous_probability_feedback']
        if feedback and plans.get('image_reader_writer') == 'SimpleITKIO':
            z_axis = plans.get('transpose_forward', [0, 1, 2]).index(0)
            if axis != z_axis:
                raise ValueError('previous_probability_feedback temporal_axis must match anatomical Z in the preprocessed SimpleITK plans')
        if temporal_mamba and data['use_fptt']:
            raise ValueError('temporal_mamba requires use_fptt=false and full-patch BPTT')
        if data['bidirectional'] or temporal_mamba:
            if data['bidirectional']:
                data['use_fptt'] = False
            mode = 'bidirectional' if data['bidirectional'] else 'temporal Mamba'
            patch = plans.get('configurations', {}).get('3d_fullres', {}).get('patch_size')
            if patch is not None:
                if len(patch) != 3 or any(type(n) is not int or n <= 0 for n in patch):
                    raise ValueError(f'{mode} training patch_size must contain three positive integers')
                data['k'] = patch[axis]
            if plans.get('image_reader_writer') == 'SimpleITKIO':
                z_axis = plans.get('transpose_forward', [0, 1, 2]).index(0)
                if axis != z_axis:
                    raise ValueError(f'{mode} temporal_axis must match anatomical Z in the preprocessed SimpleITK plans')
        for name in ("train_batches_per_epoch", "val_batches_per_epoch"):
            if type(data[name]) is not int or data[name] <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(data["num_epochs"]) is not int or data["num_epochs"] <= 0:
            raise ValueError("num_epochs must be a positive integer")
        if type(data["validation_every"]) is not int or data["validation_every"] < 0:
            raise ValueError("validation_every must be a nonnegative integer")
        patch = data["validation_patch_size"]
        if (not isinstance(patch, (list, tuple)) or len(patch) != 3
                or any(type(value) is not int or value <= 0 for value in patch)):
            raise ValueError("validation_patch_size must contain three positive integers")
        data["validation_patch_size"] = tuple(patch)
        learning_rate = data["initial_lr"]
        if type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("initial_lr must be a positive finite number")
        data["initial_lr"] = float(learning_rate)
        if type(data["use_fptt"]) is not bool:
            raise ValueError("use_fptt must be a boolean")
        for name in ("fptt_alpha", "fptt_beta", "fptt_rho", "fptt_lambda"):
            value = data[name]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if data["use_fptt"] and data["fptt_alpha"] <= 0:
            raise ValueError("fptt_alpha must be positive when use_fptt is true")
        if type(data["num_input_channels"]) is not int or data["num_input_channels"] != 4:
            raise ValueError("num_input_channels must be 4")
        if type(data["num_output_channels"]) is not int or data["num_output_channels"] != 3:
            raise ValueError("num_output_channels must be 3")
        normalized = dict(data)
        normalized["model_kwargs"] = kwargs
        return cls(**normalized)

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
            "num_epochs": self.num_epochs,
            "overfit_num_volumes": self.overfit_num_volumes,
            "train_batches_per_epoch": self.train_batches_per_epoch,
            "val_batches_per_epoch": self.val_batches_per_epoch,
            "initial_lr": self.initial_lr,
            "validation_every": self.validation_every,
            "validation_patch_size": list(self.validation_patch_size),
            "seed": self.seed,
            "deterministic": self.deterministic,
            "augmentation_workers": self.augmentation_workers,
            "bidirectional": self.bidirectional,
            "bidirectional_fusion": self.bidirectional_fusion,
        }


def build_core(config: SNNConfig) -> nn.Module:
    kwargs = dict(config.model_kwargs)
    if config.model_name == 'arch2':
        from snn_nnunet.arch2 import Arch2
        return Arch2(out_channels=config.num_output_channels, **kwargs)
    if config.model_name == "paper_unet":
        from snn_nnunet.paper_unet import PaperUNet
        return PaperUNet(out_channels=config.num_output_channels, bidirectional=config.bidirectional,
                         bidirectional_fusion=config.bidirectional_fusion, **kwargs)
    # Plans written before the hierarchical encoder existed describe the
    # original patch-embedding encoder; preserve that checkpoint architecture.
    kwargs.setdefault("hierarchical_encoder", False)
    kwargs.setdefault("temporal_mamba", dict(_TEMPORAL_MAMBA_DEFAULTS))
    kwargs.setdefault("spatial_output_spiking", False)
    output_key = next(name for name in _OUTPUT_SPIKING_KEYS if name in kwargs)
    accepted_kwargs = inspect.signature(build_model).parameters
    if output_key not in accepted_kwargs:
        target_key = next(name for name in _OUTPUT_SPIKING_KEYS if name != output_key)
        if target_key not in accepted_kwargs:
            raise TypeError("build_model has no output spiking parameter")
        kwargs[target_key] = kwargs.pop(output_key)
    core = build_model(
        config.model_name,
        out_channels=config.num_output_channels,
        **kwargs,
    )
    if not config.use_fptt:
        _freeze_disconnected_parameters(core)
    return core


def _freeze_disconnected_parameters(core: nn.Module) -> None:
    """Keep unused core weights in checkpoints but omit them from plain DDP."""
    for module in core.modules():
        if not isinstance(module, ConvBlock):
            continue
        if module.spikMamba or not module.normalization:
            for parameter in module.norm.parameters():
                parameter.requires_grad_(False)
        if module.spikMamba:
            for parameter in module.spike_neurons.parameters():
                parameter.requires_grad_(False)
            encoder = module.spik_mamba
            output_spiking = (
                encoder.output_spiking
                if hasattr(encoder, "output_spiking") else encoder.vss_output_spiking
            )
            if not output_spiking:
                for parameter in encoder.post_plif.parameters():
                    parameter.requires_grad_(False)


def slice_temporal_window(x: Tensor, start: int, end: int, axis: int) -> Tensor:
    """Select a window on a native nnU-Net spatial axis."""
    return x.narrow(2 + axis, start, end - start)


def to_core_layout(window: Tensor, axis: int) -> Tensor:
    """Map one native window to [B, k, C, H, W]."""
    return window.movedim(2 + axis, 1)


def from_core_layout(logits: Tensor, axis: int) -> Tensor:
    """Restore [B, classes, k, H, W] logits to native spatial order."""
    return logits.movedim(2, 2 + axis)


class SNNnnUNetAdapter(FP32Module):
    """Run a 2D stateful SNN over one spatial axis of a 3D volume."""

    def __init__(self, config: SNNConfig, core: nn.Module | None = None):
        super().__init__()
        self.config = config
        self.k = config.k
        self.temporal_axis = config.temporal_axis
        self.core = build_core(config) if core is None else core

    def _check_input(self, x: Tensor) -> None:
        if x.ndim != 5:
            raise ValueError("expected input shape [B, C, X, Y, Z]")
        if x.shape[1] != self.config.num_input_channels:
            raise ValueError(f"expected {self.config.num_input_channels} input channels")

    @fp32_forward
    def forward_window(self, window: Tensor, *, t0: int, return_auxiliary: bool = False):
        self._check_input(window)
        # fp32_forward rejects rounded inputs and disables any outer AMP context.
        kwargs = {'return_auxiliary': True} if return_auxiliary else {}
        logits = self.core(to_core_layout(window, self.temporal_axis), t0=t0, **kwargs)
        outputs = logits if isinstance(logits, tuple) else (logits,)
        for output in outputs:
            if output.ndim != 5 or output.shape[1] != self.config.num_output_channels:
                raise ValueError(f"expected {self.config.num_output_channels} output channels")
        restored = tuple(from_core_layout(output, self.temporal_axis) for output in outputs)
        return restored if isinstance(logits, tuple) else restored[0]

    def forward(self, x: Tensor, *, t0: int = 0, chunk_size: int | None = None,
                return_auxiliary: bool = False):
        self._check_input(x)
        if type(return_auxiliary) is not bool:
            raise ValueError('return_auxiliary must be boolean')
        if return_auxiliary and self.config.model_name not in {'paper_unet', 'arch2'}:
            raise ValueError('return_auxiliary is supported only for paper_unet or arch2')
        size = self.k if chunk_size is None else chunk_size
        if type(size) is not int or size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        length = x.shape[2 + self.temporal_axis]
        if length == 0:
            raise ValueError("temporal dimension must be nonempty")
        if self.config.requires_full_sequence:
            if t0 != 0 or (chunk_size is not None and size < length):
                raise ValueError('bidirectional/temporal Mamba mode requires a whole patch, without continuation or truncated chunks')
            return self.forward_window(x, t0=0, return_auxiliary=return_auxiliary)

        outputs = []
        for start in range(0, length, size):
            end = min(start + size, length)
            window = slice_temporal_window(x, start, end, self.temporal_axis)
            outputs.append(self.forward_window(window, t0=t0 + start,
                                               return_auxiliary=return_auxiliary))
        if return_auxiliary:
            return tuple(torch.cat([output[head] for output in outputs], dim=2 + self.temporal_axis)
                         for head in range(3))
        return torch.cat(outputs, dim=2 + self.temporal_axis)

    def detach_states(self) -> None:
        self.core.detach_states()
