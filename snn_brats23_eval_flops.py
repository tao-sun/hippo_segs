#!/usr/bin/env python3
"""Evaluate an ``snn_fptt.py`` BraTS 2023 checkpoint and profile SNN operations.

The operation report follows this convention:

* a weighted layer with dense input counts dense multiplications and additions;
* a weighted layer with binary spike input counts only
  ``dense additions * input firing rate``;
* the selective scan counts dense state dynamics plus binary-input-gated
  integration, including ``exp`` and ``softplus`` calls in the total;
* four-direction merges, Mamba residual additions, and the output sigmoid are
  counted separately and included in the total;
* PLIF membrane updates, normalisation, and other activations are outside the
  operation estimate.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from spike_neurons import PLIFNode


VALID_VIEWS = {"sagittal", "coronal", "axial"}
WEIGHTED_LAYER_TYPES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
)
SPIKE_PRESERVING_LAYER_TYPES = (
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
)


@dataclass
class PlifLayerStats:
    spikes: int = 0
    elements: int = 0

    @property
    def firing_rate(self) -> float:
        return self.spikes / self.elements if self.elements else 0.0


@dataclass
class WeightedLayerStats:
    calls: int = 0
    dense_additions: int = 0
    dense_multiplications: int = 0
    effective_additions: float = 0.0
    effective_multiplications: float = 0.0
    spike_input_nonzero: int = 0
    spike_input_elements: int = 0
    spike_driven_calls: int = 0
    dense_calls: int = 0

    @property
    def is_spike_driven(self) -> bool:
        return self.spike_driven_calls > 0 and self.dense_calls == 0

    @property
    def input_firing_rate(self) -> float:
        if not self.spike_input_elements:
            return 0.0
        return self.spike_input_nonzero / self.spike_input_elements


@dataclass
class SelectiveScanLayerStats:
    calls: int = 0
    dense_additions: int = 0
    dense_multiplications: int = 0
    spike_driven_additions: int = 0
    input_gated_multiplications: int = 0
    exp_operations: int = 0
    softplus_operations: int = 0
    spike_input_nonzero: int = 0
    spike_input_elements: int = 0
    binary_input_calls: int = 0
    dense_input_calls: int = 0

    @property
    def is_binary_input(self) -> bool:
        return self.binary_input_calls > 0 and self.dense_input_calls == 0

    @property
    def input_firing_rate(self) -> float:
        if not self.spike_input_elements:
            return 0.0
        return self.spike_input_nonzero / self.spike_input_elements

    @property
    def effective_additions(self) -> int:
        return self.dense_additions + self.spike_driven_additions

    @property
    def effective_multiplications(self) -> int:
        return self.dense_multiplications + self.input_gated_multiplications


@dataclass
class ElementwiseLayerStats:
    calls: int = 0
    additions: int = 0
    multiplications: int = 0


@dataclass
class NonlinearLayerStats:
    calls: int = 0
    operations: int = 0


@dataclass
class VolumeProfile:
    plif_layers: Dict[str, PlifLayerStats] = field(default_factory=dict)
    weighted_layers: Dict[str, WeightedLayerStats] = field(default_factory=dict)
    selective_scan_layers: Dict[str, SelectiveScanLayerStats] = field(
        default_factory=dict
    )
    elementwise_layers: Dict[str, ElementwiseLayerStats] = field(
        default_factory=dict
    )
    nonlinear_layers: Dict[str, NonlinearLayerStats] = field(
        default_factory=dict
    )

    @property
    def total_effective_additions(self) -> float:
        weighted = sum(
            layer.effective_additions for layer in self.weighted_layers.values()
        )
        scans = sum(
            layer.effective_additions
            for layer in self.selective_scan_layers.values()
        )
        elementwise = sum(
            layer.additions for layer in self.elementwise_layers.values()
        )
        return weighted + scans + elementwise

    @property
    def total_effective_multiplications(self) -> float:
        weighted = sum(
            layer.effective_multiplications
            for layer in self.weighted_layers.values()
        )
        scans = sum(
            layer.effective_multiplications
            for layer in self.selective_scan_layers.values()
        )
        elementwise = sum(
            layer.multiplications for layer in self.elementwise_layers.values()
        )
        return weighted + scans + elementwise

    @property
    def total_nonlinear_operations(self) -> int:
        scans = sum(
            layer.exp_operations + layer.softplus_operations
            for layer in self.selective_scan_layers.values()
        )
        nonlinear = sum(
            layer.operations for layer in self.nonlinear_layers.values()
        )
        return scans + nonlinear

    @property
    def total_effective_operations(self) -> float:
        return (
            self.total_effective_additions
            + self.total_effective_multiplications
            + self.total_nonlinear_operations
        )


def _resolved_config_path(config_dir: Path, raw_value: Any, key: str) -> Path:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"{key} must be a non-empty path")
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def load_testing_config(
    config_path: Path | str,
    *,
    require_paths_exist: bool = True,
) -> Dict[str, Any]:
    """Load and validate the YAML used by the evaluation script."""
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("testing YAML must contain a mapping")

    missing = sorted({"checkpoint", "data_root", "view", "fold"} - set(raw))
    if missing:
        raise ValueError(f"testing YAML is missing keys: {', '.join(missing)}")

    config: Dict[str, Any] = dict(raw)
    config["checkpoint"] = _resolved_config_path(
        config_path.parent, raw["checkpoint"], "checkpoint"
    )
    config["data_root"] = _resolved_config_path(
        config_path.parent, raw["data_root"], "data_root"
    )
    config["view"] = str(raw["view"]).lower()
    if config["view"] not in VALID_VIEWS:
        raise ValueError(f"view must be one of {sorted(VALID_VIEWS)}")

    config["fold"] = int(raw["fold"])
    if config["fold"] not in {1, 2, 3, 4, 5}:
        raise ValueError("fold must be in {1,2,3,4,5}")

    config.setdefault("device", "auto")
    config.setdefault("tbptt_k", None)
    config.setdefault("threshold", None)
    config.setdefault("num_workers", 2)
    config.setdefault("max_subjects", None)

    if config["tbptt_k"] is not None:
        config["tbptt_k"] = int(config["tbptt_k"])
        if config["tbptt_k"] <= 0:
            raise ValueError("tbptt_k must be positive or null")
    if config["threshold"] is not None:
        config["threshold"] = float(config["threshold"])
        if not 0.0 <= config["threshold"] <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
    config["num_workers"] = int(config["num_workers"])
    if config["num_workers"] < 0:
        raise ValueError("num_workers must be non-negative")
    if config["max_subjects"] is not None:
        config["max_subjects"] = int(config["max_subjects"])
        if config["max_subjects"] <= 0:
            raise ValueError("max_subjects must be positive or null")

    if require_paths_exist:
        if not config["checkpoint"].is_file():
            raise FileNotFoundError(f"checkpoint not found: {config['checkpoint']}")
        if not config["data_root"].is_dir():
            raise FileNotFoundError(f"data_root not found: {config['data_root']}")
    return config


def model_spec_from_checkpoint(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract the model/inference options persisted by ``snn_fptt.py``."""
    if not isinstance(checkpoint.get("model"), Mapping):
        raise ValueError("checkpoint does not contain a model state_dict")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint does not contain the snn_fptt config")

    state_names = checkpoint["model"].keys()
    return {
        "model_name": str(config.get("model", "orig")),
        "patch_size": int(config.get("patch_size", 4)),
        "linear_projection": bool(config.get("linear_projection", True)),
        "residual_connections": bool(config.get("residual_connections", True)),
        "dwconv2d_spiking": bool(config.get("dwconv2d_spiking", False)),
        "patch_embedding_spiking": bool(
            config.get("patch_embedding_spiking", False)
        ),
        "num_encoder_stages": (
            4 if any(name.startswith("conv_block4.") for name in state_names) else 3
        ),
        "tbptt_k": int(config.get("tbptt_k", 1)),
        "prob_threshold": float(config.get("prob_threshold", 0.5)),
    }


def _extract_tensor(output: Any) -> Optional[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    return None


def _dense_weighted_operations(
    module: nn.Module,
    inputs: torch.Tensor,
    output: torch.Tensor,
) -> Tuple[int, int]:
    """Return ``(additions, multiplications)`` for one dense layer call."""
    if isinstance(module, nn.Linear):
        output_elements = output.numel()
        terms = module.in_features
        multiplications = output_elements * terms
        additions = output_elements * max(terms - 1, 0)
        if module.bias is not None:
            additions += output_elements
        return int(additions), int(multiplications)

    if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        spatial_contributions, nonempty_outputs = _transpose_spatial_counts(
            tuple(inputs.shape[2:]),
            tuple(output.shape[2:]),
            tuple(module.kernel_size),
            tuple(module.stride),
            tuple(module.padding),
            tuple(module.dilation),
        )
        batch_size = inputs.shape[0]
        contributions = (
            batch_size
            * module.out_channels
            * (module.in_channels // module.groups)
            * spatial_contributions
        )
        additions = contributions - (
            batch_size * module.out_channels * nonempty_outputs
        )
        if module.bias is not None:
            additions += output.numel()
        return int(additions), int(contributions)

    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        if module.padding_mode == "zeros" and isinstance(module.padding, tuple):
            spatial_contributions, nonempty_outputs = _forward_spatial_counts(
                tuple(inputs.shape[2:]),
                tuple(output.shape[2:]),
                tuple(module.kernel_size),
                tuple(module.stride),
                tuple(module.padding),
                tuple(module.dilation),
            )
            multiplications = (
                inputs.shape[0]
                * module.out_channels
                * (module.in_channels // module.groups)
                * spatial_contributions
            )
            additions = multiplications - (
                inputs.shape[0] * module.out_channels * nonempty_outputs
            )
        else:
            kernel_elements = int(np.prod(module.kernel_size))
            terms = (module.in_channels // module.groups) * kernel_elements
            multiplications = output.numel() * terms
            additions = output.numel() * max(terms - 1, 0)
        if module.bias is not None:
            additions += output.numel()
        return int(additions), int(multiplications)

    raise TypeError(f"unsupported weighted module: {type(module).__name__}")


@lru_cache(maxsize=128)
def _forward_spatial_counts(
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    kernel_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    dilation: Tuple[int, ...],
) -> Tuple[int, int]:
    valid_pairs_per_dimension = []
    nonempty_positions_per_dimension = []
    for input_size, output_size, kernel, step, pad, dilate in zip(
        input_shape, output_shape, kernel_size, stride, padding, dilation
    ):
        output_counts = []
        for output_index in range(output_size):
            valid_terms = 0
            for kernel_index in range(kernel):
                input_index = output_index * step - pad + kernel_index * dilate
                valid_terms += int(0 <= input_index < input_size)
            output_counts.append(valid_terms)
        valid_pairs_per_dimension.append(sum(output_counts))
        nonempty_positions_per_dimension.append(
            sum(count > 0 for count in output_counts)
        )
    return (
        int(np.prod(valid_pairs_per_dimension)),
        int(np.prod(nonempty_positions_per_dimension)),
    )


@lru_cache(maxsize=128)
def _transpose_spatial_counts(
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    kernel_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    dilation: Tuple[int, ...],
) -> Tuple[int, int]:
    """Count valid kernel contributions and non-empty transposed-conv outputs."""
    valid_pairs_per_dimension = []
    nonempty_positions_per_dimension = []
    for input_size, output_size, kernel, step, pad, dilate in zip(
        input_shape, output_shape, kernel_size, stride, padding, dilation
    ):
        output_counts = [0] * output_size
        for input_index in range(input_size):
            for kernel_index in range(kernel):
                output_index = input_index * step - pad + kernel_index * dilate
                if 0 <= output_index < output_size:
                    output_counts[output_index] += 1
        valid_pairs_per_dimension.append(sum(output_counts))
        nonempty_positions_per_dimension.append(
            sum(count > 0 for count in output_counts)
        )
    return (
        int(np.prod(valid_pairs_per_dimension)),
        int(np.prod(nonempty_positions_per_dimension)),
    )


class HybridOperationProfiler:
    """Collect PLIF firing rates and hybrid dense/spike-driven operation counts."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._handles: List[Any] = []
        self._current: Optional[VolumeProfile] = None
        self._tensor_provenance: Dict[Tuple[str, int], str] = {}
        self._functional_conv1d_calls = 0
        self._functional_context: Optional[Tuple[str, int, str]] = None
        self._forced_layer_provenance: Dict[str, str] = {}
        self._selective_scan_modules: List[Tuple[str, nn.Module]] = []
        self._original_selective_scans: Dict[nn.Module, Any] = {}
        self._original_conv1d = None
        self._original_pad = None
        self._original_cat = None
        for name, module in model.named_modules():
            if isinstance(module, PLIFNode) and not getattr(module, "no_spiking", False):
                self._handles.append(module.register_forward_hook(self._plif_hook(name)))
            elif isinstance(module, WEIGHTED_LAYER_TYPES):
                self._handles.append(
                    module.register_forward_hook(self._weighted_hook(name))
                )
            elif isinstance(module, SPIKE_PRESERVING_LAYER_TYPES):
                self._handles.append(
                    module.register_forward_hook(self._provenance_hook())
                )
            elif module.__class__.__name__ == "SpikeMambaLayer":
                self._selective_scan_modules.append((name, module))
                self._forced_layer_provenance[f"{name}.linear_m"] = "dense"
                self._forced_layer_provenance[f"{name}.dwconv2d"] = (
                    "spike"
                    if getattr(module, "patch_embedding_spiking", False)
                    else "dense"
                )
                self._forced_layer_provenance[f"{name}.out_proj"] = "dense"
                self._handles.append(
                    module.register_forward_pre_hook(
                        self._mamba_pre_hook(name, module)
                    )
                )
                self._handles.append(
                    module.register_forward_hook(
                        self._mamba_post_hook(name, module)
                    )
                )
            elif module.__class__.__name__ == "SpikMambaBlock":
                self._handles.append(
                    module.register_forward_hook(
                        self._residual_hook(name, module)
                    )
                )
        self._handles.append(model.register_forward_pre_hook(self._model_pre_hook()))

    def start_volume(self) -> None:
        if self._current is not None:
            raise RuntimeError("finish the current volume before starting another")
        self._current = VolumeProfile()
        self._tensor_provenance.clear()
        self._functional_conv1d_calls = 0
        if self._original_conv1d is None:
            self._original_conv1d = F.conv1d
            F.conv1d = self._profiled_conv1d
        if self._original_pad is None:
            self._original_pad = F.pad
            F.pad = self._profiled_pad
        if self._original_cat is None:
            self._original_cat = torch.cat
            torch.cat = self._profiled_cat
        self._patch_selective_scans()

    def finish_volume(self) -> VolumeProfile:
        if self._current is None:
            raise RuntimeError("start_volume must be called first")
        profile = self._current
        self._current = None
        self._restore_patched_operations()
        return profile

    def close(self) -> None:
        self._restore_patched_operations()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _plif_hook(self, name: str):
        def hook(_module, _inputs, output):
            if self._current is None:
                return
            spikes = _extract_tensor(output)
            if spikes is None:
                return
            stats = self._current.plif_layers.setdefault(name, PlifLayerStats())
            stats.spikes += int(torch.count_nonzero(spikes).item())
            stats.elements += spikes.numel()
            self._tensor_provenance[self._tensor_key(spikes)] = "spike"

        return hook

    def _weighted_hook(self, name: str):
        def hook(module, module_inputs, output):
            if self._current is None or not module_inputs:
                return
            inputs = _extract_tensor(module_inputs[0])
            outputs = _extract_tensor(output)
            if inputs is None or outputs is None:
                return

            additions, multiplications = _dense_weighted_operations(
                module, inputs, outputs
            )
            stats = self._current.weighted_layers.setdefault(
                name, WeightedLayerStats()
            )
            stats.calls += 1
            stats.dense_additions += additions
            stats.dense_multiplications += multiplications

            self._record_effective_operations(
                name,
                inputs,
                additions,
                multiplications,
                stats,
                forced_provenance=self._forced_layer_provenance.get(name),
            )
            self._tensor_provenance[self._tensor_key(outputs)] = "dense"

        return hook

    def _provenance_hook(self):
        def hook(_module, module_inputs, output):
            if self._current is None or not module_inputs:
                return
            inputs = _extract_tensor(module_inputs[0])
            outputs = _extract_tensor(output)
            if inputs is None or outputs is None:
                return
            provenance = self._tensor_provenance.get(
                self._tensor_key(inputs), "dense"
            )
            self._tensor_provenance[self._tensor_key(outputs)] = provenance

        return hook

    def _record_effective_operations(
        self,
        name: str,
        inputs: torch.Tensor,
        additions: int,
        multiplications: int,
        stats: WeightedLayerStats,
        forced_provenance: Optional[str] = None,
    ) -> None:
        provenance = forced_provenance or self._tensor_provenance.get(
            self._tensor_key(inputs), "dense"
        )
        spike_driven = provenance == "spike"

        if spike_driven:
            nonzero = int(torch.count_nonzero(inputs).item())
            elements = inputs.numel()
            firing_rate = nonzero / elements if elements else 0.0
            stats.spike_driven_calls += 1
            stats.spike_input_nonzero += nonzero
            stats.spike_input_elements += elements
            stats.effective_additions += additions * firing_rate
        else:
            stats.dense_calls += 1
            stats.effective_additions += additions
            stats.effective_multiplications += multiplications

    def _profiled_conv1d(
        self,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
    ) -> torch.Tensor:
        if self._original_conv1d is None:
            raise RuntimeError("functional conv1d profiler is not active")
        output = self._original_conv1d(
            inputs, weight, bias, stride, padding, dilation, groups
        )
        if self._current is None:
            return output

        self._functional_conv1d_calls += 1
        if self._functional_context is None:
            name = f"functional_conv1d_{self._functional_conv1d_calls}"
            forced_provenance = None
        else:
            context_name, context_call, x_projection_provenance = (
                self._functional_context
            )
            suffix = "x_proj" if context_call == 0 else "dt_proj"
            name = f"{context_name}.functional_{suffix}"
            forced_provenance = (
                x_projection_provenance if context_call == 0 else "dense"
            )
            self._functional_context = (
                context_name,
                context_call + 1,
                x_projection_provenance,
            )

        terms = weight.shape[1] * weight.shape[2]
        multiplications = output.numel() * terms
        additions = output.numel() * max(terms - 1, 0)
        if bias is not None:
            additions += output.numel()
        stats = self._current.weighted_layers.setdefault(name, WeightedLayerStats())
        stats.calls += 1
        stats.dense_additions += int(additions)
        stats.dense_multiplications += int(multiplications)
        self._record_effective_operations(
            name,
            inputs,
            int(additions),
            int(multiplications),
            stats,
            forced_provenance=forced_provenance,
        )
        self._tensor_provenance[self._tensor_key(output)] = "dense"
        return output

    def _model_pre_hook(self):
        def hook(_module, module_inputs):
            for value in module_inputs:
                if isinstance(value, torch.Tensor):
                    self._tensor_provenance[self._tensor_key(value)] = "dense"

        return hook

    def _profiled_pad(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self._original_pad is None:
            raise RuntimeError("functional pad profiler is not active")
        output = self._original_pad(inputs, *args, **kwargs)
        provenance = self._tensor_provenance.get(
            self._tensor_key(inputs), "dense"
        )
        self._tensor_provenance[self._tensor_key(output)] = provenance
        return output

    def _profiled_cat(self, tensors, *args, **kwargs) -> torch.Tensor:
        if self._original_cat is None:
            raise RuntimeError("cat profiler is not active")
        tensors = tuple(tensors)
        output = self._original_cat(tensors, *args, **kwargs)
        provenances = [
            self._tensor_provenance.get(self._tensor_key(tensor))
            for tensor in tensors
            if isinstance(tensor, torch.Tensor)
        ]
        provenance = (
            "spike"
            if provenances and all(value == "spike" for value in provenances)
            else "dense"
        )
        self._tensor_provenance[self._tensor_key(output)] = provenance
        return output

    def _mamba_pre_hook(self, name: str, module: nn.Module):
        def hook(_module, _inputs):
            x_projection_provenance = (
                "spike"
                if getattr(module, "dwconv2d_spiking", False)
                else "dense"
            )
            self._functional_context = (name, 0, x_projection_provenance)

        return hook

    def _mamba_post_hook(self, name: str, module: nn.Module):
        def hook(_module, module_inputs, _output):
            if (
                self._current is not None
                and module_inputs
                and isinstance(module_inputs[0], torch.Tensor)
                and getattr(module, "num_directions", None) == 4
                and isinstance(getattr(module, "d_inner", None), int)
            ):
                tokens = module_inputs[0]
                merged_elements = (
                    tokens.shape[0] * tokens.shape[1] * module.d_inner
                )
                stats_name = (
                    f"{name}.direction_merge" if name else "direction_merge"
                )
                stats = self._current.elementwise_layers.setdefault(
                    stats_name, ElementwiseLayerStats()
                )
                stats.calls += 1
                stats.additions += 3 * merged_elements
                stats.multiplications += merged_elements
            self._functional_context = None

        return hook

    def _residual_hook(self, name: str, module: nn.Module):
        def hook(_module, _inputs, output):
            if (
                self._current is None
                or not getattr(module, "residual_connections", False)
            ):
                return
            outputs = _extract_tensor(output)
            if outputs is None:
                return
            stats_name = f"{name}.residual" if name else "residual"
            stats = self._current.elementwise_layers.setdefault(
                stats_name, ElementwiseLayerStats()
            )
            stats.calls += 1
            stats.additions += outputs.numel()

        return hook

    def profiled_sigmoid(
        self,
        inputs: torch.Tensor,
        name: str = "output_sigmoid",
    ) -> torch.Tensor:
        output = torch.sigmoid(inputs)
        if self._current is not None:
            stats = self._current.nonlinear_layers.setdefault(
                name, NonlinearLayerStats()
            )
            stats.calls += 1
            stats.operations += inputs.numel()
        return output

    def _patch_selective_scans(self) -> None:
        for name, module in self._selective_scan_modules:
            if module in self._original_selective_scans:
                continue
            original = module.selective_scan
            self._original_selective_scans[module] = original
            module.selective_scan = self._profiled_selective_scan(
                name,
                original,
                binary_input=bool(getattr(module, "dwconv2d_spiking", False)),
            )

    def _profiled_selective_scan(
        self,
        name: str,
        original,
        *,
        binary_input: bool,
    ):
        def profiled(
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
            output = original(
                u,
                delta,
                A,
                B,
                C,
                D,
                z,
                delta_bias,
                delta_softplus,
                return_last_state,
            )
            if self._current is None:
                return output
            if u.ndim != 3 or A.ndim != 2:
                raise ValueError("selective_scan profiler expects u=BDL and A=DN")
            if A.is_complex():
                raise ValueError("complex selective_scan states are not supported")

            elements = u.numel()
            state_size = A.shape[1]
            state_elements = elements * state_size
            stats_name = f"{name}.selective_scan" if name else "selective_scan"
            stats = self._current.selective_scan_layers.setdefault(
                stats_name, SelectiveScanLayerStats()
            )
            stats.calls += 1
            stats.exp_operations += state_elements + A.numel()
            if delta_softplus:
                stats.softplus_operations += elements

            bias_additions = elements if delta_bias is not None else 0
            readout_additions = elements * max(state_size - 1, 0)
            skip_elements = elements if D is not None else 0

            if binary_input:
                nonzero = int(torch.count_nonzero(u).item())
                active_state_elements = nonzero * state_size
                stats.binary_input_calls += 1
                stats.spike_input_nonzero += nonzero
                stats.spike_input_elements += elements
                stats.dense_multiplications += 3 * state_elements
                stats.input_gated_multiplications += active_state_elements
                stats.dense_additions += bias_additions + readout_additions
                stats.spike_driven_additions += active_state_elements
                if D is not None:
                    stats.spike_driven_additions += nonzero
            else:
                stats.dense_input_calls += 1
                stats.dense_multiplications += 5 * state_elements + skip_elements
                stats.dense_additions += (
                    bias_additions
                    + state_elements
                    + readout_additions
                    + skip_elements
                )

            if z is not None:
                stats.dense_multiplications += 2 * elements
            return output

        return profiled

    def _restore_patched_operations(self) -> None:
        if self._original_conv1d is not None:
            F.conv1d = self._original_conv1d
            self._original_conv1d = None
        if self._original_pad is not None:
            F.pad = self._original_pad
            self._original_pad = None
        if self._original_cat is not None:
            torch.cat = self._original_cat
            self._original_cat = None
        for module, original in self._original_selective_scans.items():
            module.selective_scan = original
        self._original_selective_scans.clear()

    @staticmethod
    def _tensor_key(tensor: torch.Tensor) -> Tuple[str, int]:
        return str(tensor.device), tensor.untyped_storage().data_ptr()


def aggregate_volume_profiles(profiles: Iterable[VolumeProfile]) -> Dict[str, Any]:
    profiles = list(profiles)
    if not profiles:
        raise ValueError("at least one volume profile is required")

    plif_names = sorted({name for profile in profiles for name in profile.plif_layers})
    weighted_names = sorted(
        {name for profile in profiles for name in profile.weighted_layers}
    )
    scan_names = sorted(
        {name for profile in profiles for name in profile.selective_scan_layers}
    )
    elementwise_names = sorted(
        {name for profile in profiles for name in profile.elementwise_layers}
    )
    nonlinear_names = sorted(
        {name for profile in profiles for name in profile.nonlinear_layers}
    )
    plif_summary: Dict[str, Dict[str, float]] = {}
    weighted_summary: Dict[str, Dict[str, float | bool]] = {}
    scan_summary: Dict[str, Dict[str, float | bool]] = {}
    elementwise_summary: Dict[str, Dict[str, float]] = {}
    nonlinear_summary: Dict[str, Dict[str, float]] = {}

    for name in plif_names:
        rates = np.asarray(
            [profile.plif_layers.get(name, PlifLayerStats()).firing_rate for profile in profiles],
            dtype=np.float64,
        )
        plif_summary[name] = {
            "mean_firing_rate": float(rates.mean()),
            "std_firing_rate": float(rates.std()),
        }

    for name in weighted_names:
        layers = [profile.weighted_layers.get(name, WeightedLayerStats()) for profile in profiles]
        weighted_summary[name] = {
            "is_spike_driven": all(layer.is_spike_driven for layer in layers),
            "mean_input_firing_rate": float(
                np.mean([layer.input_firing_rate for layer in layers])
            ),
            "mean_dense_additions": float(
                np.mean([layer.dense_additions for layer in layers])
            ),
            "mean_dense_multiplications": float(
                np.mean([layer.dense_multiplications for layer in layers])
            ),
            "mean_effective_additions": float(
                np.mean([layer.effective_additions for layer in layers])
            ),
            "mean_effective_multiplications": float(
                np.mean([layer.effective_multiplications for layer in layers])
            ),
        }

    for name in scan_names:
        layers = [
            profile.selective_scan_layers.get(name, SelectiveScanLayerStats())
            for profile in profiles
        ]
        scan_summary[name] = {
            "is_binary_input": all(layer.is_binary_input for layer in layers),
            "mean_input_firing_rate": float(
                np.mean([layer.input_firing_rate for layer in layers])
            ),
            "mean_dense_additions": float(
                np.mean([layer.dense_additions for layer in layers])
            ),
            "mean_spike_driven_additions": float(
                np.mean([layer.spike_driven_additions for layer in layers])
            ),
            "mean_dense_multiplications": float(
                np.mean([layer.dense_multiplications for layer in layers])
            ),
            "mean_input_gated_multiplications": float(
                np.mean([layer.input_gated_multiplications for layer in layers])
            ),
            "mean_effective_additions": float(
                np.mean([layer.effective_additions for layer in layers])
            ),
            "mean_effective_multiplications": float(
                np.mean([layer.effective_multiplications for layer in layers])
            ),
            "mean_exp_operations": float(
                np.mean([layer.exp_operations for layer in layers])
            ),
            "mean_softplus_operations": float(
                np.mean([layer.softplus_operations for layer in layers])
            ),
        }

    for name in elementwise_names:
        layers = [
            profile.elementwise_layers.get(name, ElementwiseLayerStats())
            for profile in profiles
        ]
        elementwise_summary[name] = {
            "mean_additions": float(np.mean([layer.additions for layer in layers])),
            "mean_multiplications": float(
                np.mean([layer.multiplications for layer in layers])
            ),
        }

    for name in nonlinear_names:
        layers = [
            profile.nonlinear_layers.get(name, NonlinearLayerStats())
            for profile in profiles
        ]
        nonlinear_summary[name] = {
            "mean_operations": float(
                np.mean([layer.operations for layer in layers])
            ),
        }

    addition_totals = np.asarray(
        [profile.total_effective_additions for profile in profiles],
        dtype=np.float64,
    )
    multiplication_totals = np.asarray(
        [profile.total_effective_multiplications for profile in profiles],
        dtype=np.float64,
    )
    nonlinear_totals = np.asarray(
        [profile.total_nonlinear_operations for profile in profiles],
        dtype=np.float64,
    )
    totals = addition_totals + multiplication_totals + nonlinear_totals
    return {
        "n_volumes": len(profiles),
        "plif_layers": plif_summary,
        "weighted_layers": weighted_summary,
        "selective_scan_layers": scan_summary,
        "elementwise_layers": elementwise_summary,
        "nonlinear_layers": nonlinear_summary,
        "mean_total_effective_additions": float(addition_totals.mean()),
        "std_total_effective_additions": float(addition_totals.std()),
        "mean_total_effective_multiplications": float(
            multiplication_totals.mean()
        ),
        "std_total_effective_multiplications": float(
            multiplication_totals.std()
        ),
        "mean_total_nonlinear_operations": float(nonlinear_totals.mean()),
        "std_total_nonlinear_operations": float(nonlinear_totals.std()),
        "mean_total_effective_operations": float(totals.mean()),
        "std_total_effective_operations": float(totals.std()),
    }


def _install_numpy_checkpoint_compatibility() -> None:
    """Allow NumPy 1.x to unpickle RNG metadata written by NumPy 2.x."""
    if not hasattr(np, "core"):
        return
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def load_fptt_checkpoint(path: Path) -> Dict[str, Any]:
    _install_numpy_checkpoint_compatibility()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary produced by snn_fptt.py")
    model_spec_from_checkpoint(checkpoint)
    return checkpoint


class FourStageSNNBraTS(nn.Module):
    """Four-stage ``orig`` architecture used by deeper FPTT checkpoints."""

    def __init__(
        self,
        out_channels: int,
        patch_size: int,
        linear_projection: bool,
        residual_connections: bool,
        dwconv2d_spiking: bool,
        patch_embedding_spiking: bool,
    ) -> None:
        super().__init__()
        from model import ConvBlock, DeconvBlock

        self.patch_size = patch_size
        self.encoder_scale = patch_size**4
        encoder_options = {
            "spikMamba": True,
            "patch_size": patch_size,
            "linear_projection": linear_projection,
            "residual_connections": residual_connections,
            "dwconv2d_spiking": dwconv2d_spiking,
            "patch_embedding_spiking": patch_embedding_spiking,
        }
        self.conv_block1 = ConvBlock(4, 32, padding=1, dropout=0.1, **encoder_options)
        self.conv_block2 = ConvBlock(32, 64, padding=1, dropout=0.1, **encoder_options)
        self.conv_block3 = ConvBlock(64, 128, padding=1, dropout=0.1, **encoder_options)
        self.conv_block4 = ConvBlock(128, 256, padding=1, dropout=0.1, **encoder_options)

        self.deconv_block0 = DeconvBlock(256, 128, patch_size, patch_size, dropout=0.1)
        self.deconv0_conv = ConvBlock(128, 128, padding=1, dropout=0.1)
        self.concat0_conv = ConvBlock(256, 128, padding=1, dropout=0.1)

        self.deconv_block1 = DeconvBlock(128, 128, patch_size, patch_size, dropout=0.1)
        self.deconv1_conv = ConvBlock(128, 128, padding=1, dropout=0.1)
        self.concat1_conv = ConvBlock(192, 128, padding=1, dropout=0.1)

        self.deconv_block2 = DeconvBlock(128, 128, patch_size, patch_size, dropout=0.1)
        self.deconv2_conv = ConvBlock(128, 128, padding=1, dropout=0.1)
        self.concat2_conv = ConvBlock(160, 128, padding=1, dropout=0.1)

        self.deconv_block3 = DeconvBlock(128, 128, patch_size, patch_size, dropout=0.1)
        self.deconv3_conv = ConvBlock(128, 128, padding=1, dropout=0.1)
        self.class_conv = ConvBlock(
            128,
            out_channels,
            padding=1,
            dropout=0.0,
            normalization=False,
            spiking=False,
        )

    def forward(self, x_window: torch.Tensor, t0: int = 0) -> torch.Tensor:
        _, time_steps, _, height, width = x_window.shape
        pad_height = (-height) % self.encoder_scale
        pad_width = (-width) % self.encoder_scale
        logits = []
        for index in range(time_steps):
            time_step = t0 + index
            x = x_window[:, index]
            if pad_height or pad_width:
                x = F.pad(x, (0, pad_width, 0, pad_height))

            skip1 = self.conv_block1(x, time_step)
            skip2 = self.conv_block2(skip1, time_step)
            skip3 = self.conv_block3(skip2, time_step)
            x = self.conv_block4(skip3, time_step)

            x = self.deconv_block0(x, time_step)
            x = self.deconv0_conv(x, time_step)
            x = self.concat0_conv(torch.cat([skip3, x], dim=1), time_step)
            x = self.deconv_block1(x, time_step)
            x = self.deconv1_conv(x, time_step)
            x = self.concat1_conv(torch.cat([skip2, x], dim=1), time_step)
            x = self.deconv_block2(x, time_step)
            x = self.deconv2_conv(x, time_step)
            x = self.concat2_conv(torch.cat([skip1, x], dim=1), time_step)
            x = self.deconv_block3(x, time_step)
            x = self.deconv3_conv(x, time_step)
            logits.append(self.class_conv(x, time_step)[..., :height, :width])
        return torch.stack(logits, dim=2)

    def detach_states(self) -> None:
        for module in self.modules():
            if hasattr(module, "detach") and callable(module.detach):
                module.detach()


def build_checkpoint_model(checkpoint: Mapping[str, Any], device: torch.device) -> nn.Module:
    from model import (
        SNNBraTS,
        SNNBraTSUNetDeep,
        SNNBraTSUNetMedium,
        SNNBraTSUNetShallow,
    )

    state_dict = checkpoint["model"]
    if any(".conv1d_m." in name for name in state_dict):
        raise ValueError(
            "This checkpoint uses the legacy conv1d_m architecture, while the current "
            "model.py uses dwconv2d. Evaluate it with the matching historical model code."
        )
    spec = model_spec_from_checkpoint(checkpoint)
    model_options = {
        "out_channels": 3,
        "patch_size": spec["patch_size"],
        "linear_projection": spec["linear_projection"],
        "residual_connections": spec["residual_connections"],
        "dwconv2d_spiking": spec["dwconv2d_spiking"],
        "patch_embedding_spiking": spec["patch_embedding_spiking"],
    }
    if spec["model_name"] == "orig" and spec["num_encoder_stages"] == 4:
        model = FourStageSNNBraTS(**model_options)
    elif spec["model_name"] == "orig":
        model = SNNBraTS(**model_options)
    elif spec["model_name"] == "shallow":
        model = SNNBraTSUNetShallow(out_channels=3)
    elif spec["model_name"] == "medium":
        model = SNNBraTSUNetMedium(out_channels=3)
    elif spec["model_name"] == "deep":
        model = SNNBraTSUNetDeep(out_channels=3)
    else:
        raise ValueError(f"unknown checkpoint model: {spec['model_name']}")
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def choose_device(value: str) -> torch.device:
    value = str(value).lower()
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def reset_spiking_state(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, PLIFNode):
            module.reset()
            module.neuro_states_init = False


def _metadata_shape(meta_xyz: Any) -> Tuple[int, int, int]:
    values = []
    for value in meta_xyz:
        if isinstance(value, torch.Tensor):
            value = value.reshape(-1)[0].item()
        elif isinstance(value, (list, tuple)):
            value = value[0]
        values.append(int(value))
    return tuple(values)  # type: ignore[return-value]


def _dice_per_channel(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    result = []
    for channel in range(3):
        pred = prediction[channel].reshape(-1).astype(np.uint8)
        truth = target[channel].reshape(-1).astype(np.uint8)
        intersection = (pred & truth).sum()
        denominator = pred.sum() + truth.sum()
        result.append((2 * intersection + 1e-6) / (denominator + 1e-6))
    return np.asarray(result, dtype=np.float64)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    view: str,
    device: torch.device,
    tbptt_k: int,
    threshold: float,
) -> Tuple[List[np.ndarray], List[VolumeProfile]]:
    from snn_fptt import stack_back

    dice_values: List[np.ndarray] = []
    volume_profiles: List[VolumeProfile] = []
    profiler = HybridOperationProfiler(model)
    try:
        for xs, ys, meta in tqdm(loader, desc=f"BraTS23 view={view}"):
            reset_spiking_state(model)
            xs = xs.to(device, non_blocking=True)
            slice_count = xs.shape[1]
            predicted_slices = []
            profiler.start_volume()
            for t0 in range(0, slice_count, tbptt_k):
                x_window = xs[:, t0 : t0 + tbptt_k]
                logits = model(x_window, t0=t0)
                probabilities = profiler.profiled_sigmoid(logits).cpu().numpy()
                predicted_slices.append(np.transpose(probabilities[0], (1, 0, 2, 3)))
                if hasattr(model, "detach_states"):
                    model.detach_states()
            volume_profiles.append(profiler.finish_volume())

            predicted = np.concatenate(predicted_slices, axis=0)
            target_slices = ys.numpy()[0]
            xyz = _metadata_shape(meta["xyz"])
            prediction_volume = stack_back(predicted, view, xyz)
            target_volume = stack_back(target_slices, view, xyz).astype(np.uint8)
            binary_volume = (prediction_volume >= threshold).astype(np.uint8)
            dice = _dice_per_channel(binary_volume, target_volume)
            dice_values.append(dice)
            subject_id = meta["sid"][0]
            print(
                f"{subject_id}: Dice ET={dice[0]:.4f} TC={dice[1]:.4f} "
                f"WT={dice[2]:.4f} | effective ops="
                f"{volume_profiles[-1].total_effective_operations:.3e}"
            )
    finally:
        profiler.close()
    return dice_values, volume_profiles


def _format_operations(value: float) -> str:
    for divisor, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= divisor:
            return f"{value / divisor:.3f} {suffix}"
    return f"{value:.0f}"


def print_report(dices: List[np.ndarray], summary: Mapping[str, Any]) -> None:
    dice_array = np.stack(dices)
    dice_mean = dice_array.mean(axis=0)
    dice_std = dice_array.std(axis=0)
    print("\n=== Dice 3D medio per volume ===")
    for name, mean, std in zip(("ET", "TC", "WT"), dice_mean, dice_std):
        print(f"{name}: {mean:.4f} +/- {std:.4f}")
    print(f"Mean: {dice_mean.mean():.4f}")

    print("\n=== Firing rate medio per PLIF ===")
    for name, values in summary["plif_layers"].items():
        print(
            f"{name:70s} {values['mean_firing_rate']:.6f} "
            f"+/- {values['std_firing_rate']:.6f}"
        )

    print("\n=== Operazioni pesate medie per volume ===")
    print("layer | mode | input FR | dense adds | dense muls | effective adds | effective muls")
    for name, values in summary["weighted_layers"].items():
        mode = "AC/spike" if values["is_spike_driven"] else "dense"
        print(
            f"{name} | {mode} | {values['mean_input_firing_rate']:.6f} | "
            f"{_format_operations(values['mean_dense_additions'])} | "
            f"{_format_operations(values['mean_dense_multiplications'])} | "
            f"{_format_operations(values['mean_effective_additions'])} | "
            f"{_format_operations(values['mean_effective_multiplications'])}"
        )

    print("\n=== Selective scan medio per volume ===")
    print(
        "layer | mode | input FR | dense adds | spike ACs | dense muls | "
        "input-gated muls | effective adds | effective muls | exp | softplus"
    )
    for name, values in summary["selective_scan_layers"].items():
        mode = "binary input" if values["is_binary_input"] else "dense input"
        print(
            f"{name} | {mode} | {values['mean_input_firing_rate']:.6f} | "
            f"{_format_operations(values['mean_dense_additions'])} | "
            f"{_format_operations(values['mean_spike_driven_additions'])} | "
            f"{_format_operations(values['mean_dense_multiplications'])} | "
            f"{_format_operations(values['mean_input_gated_multiplications'])} | "
            f"{_format_operations(values['mean_effective_additions'])} | "
            f"{_format_operations(values['mean_effective_multiplications'])} | "
            f"{_format_operations(values['mean_exp_operations'])} | "
            f"{_format_operations(values['mean_softplus_operations'])}"
        )

    print("\n=== Operazioni element-wise medie per volume ===")
    print("layer | additions | multiplications")
    for name, values in summary["elementwise_layers"].items():
        print(
            f"{name} | {_format_operations(values['mean_additions'])} | "
            f"{_format_operations(values['mean_multiplications'])}"
        )

    print("\n=== Altre non-linearita medie per volume ===")
    print("layer | operations")
    for name, values in summary["nonlinear_layers"].items():
        print(f"{name} | {_format_operations(values['mean_operations'])}")

    print("\n=== Totale ibrido medio per volume ===")
    print(
        "Addizioni/AC effettive: "
        f"{_format_operations(summary['mean_total_effective_additions'])} "
        f"+/- {_format_operations(summary['std_total_effective_additions'])}"
    )
    print(
        "Moltiplicazioni effettive: "
        f"{_format_operations(summary['mean_total_effective_multiplications'])} "
        f"+/- {_format_operations(summary['std_total_effective_multiplications'])}"
    )
    print(
        "Operazioni non lineari (exp/softplus/sigmoid): "
        f"{_format_operations(summary['mean_total_nonlinear_operations'])} "
        f"+/- {_format_operations(summary['std_total_nonlinear_operations'])}"
    )
    print(
        "Totale: "
        f"{_format_operations(summary['mean_total_effective_operations'])} "
        f"+/- {_format_operations(summary['std_total_effective_operations'])} operations"
    )
    print(
        "Nota: il totale include Conv/ConvTranspose/Linear e l'aritmetica del "
        "selective_scan, exp/softplus, merge direzionali, residuali Mamba e "
        "sigmoid sui logits. PLIF, normalizzazioni e altre attivazioni non sono "
        "incluse nel totale. Ogni exp/softplus/sigmoid vale una operazione "
        "non lineare."
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a BraTS23 FPTT checkpoint and profile SNN operations."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("testing.yaml"),
        help="Testing YAML path (default: testing.yaml)",
    )
    args = parser.parse_args(argv)
    config = load_testing_config(args.config)
    checkpoint = load_fptt_checkpoint(config["checkpoint"])
    spec = model_spec_from_checkpoint(checkpoint)
    device = choose_device(config["device"])
    model = build_checkpoint_model(checkpoint, device)

    from snn_fptt import BratsVolumeDataset

    dataset = BratsVolumeDataset(
        root=str(config["data_root"]),
        val_fold=config["fold"],
        view=config["view"],
    )
    if config["max_subjects"] is not None:
        count = min(config["max_subjects"], len(dataset))
        dataset = Subset(dataset, range(count))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )

    tbptt_k = config["tbptt_k"] or spec["tbptt_k"]
    threshold = (
        spec["prob_threshold"]
        if config["threshold"] is None
        else config["threshold"]
    )
    print(
        f"checkpoint={config['checkpoint']}\ndata_root={config['data_root']}\n"
        f"view={config['view']} fold={config['fold']} device={device} "
        f"k={tbptt_k} threshold={threshold}"
    )
    checkpoint_view = checkpoint["config"].get("view")
    if checkpoint_view and checkpoint_view != config["view"]:
        print(
            f"[WARN] checkpoint trained on view={checkpoint_view}, "
            f"evaluated on view={config['view']}"
        )

    dices, profiles = evaluate(
        model,
        loader,
        view=config["view"],
        device=device,
        tbptt_k=tbptt_k,
        threshold=threshold,
    )
    if not dices:
        raise RuntimeError("no BraTS23 volumes were evaluated")
    print_report(dices, aggregate_volume_profiles(profiles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
