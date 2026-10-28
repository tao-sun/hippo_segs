"""Patch-based spiking Mamba components for per-frame BCHW features."""

import math

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from snn_nnunet import surrogate
from snn_nnunet.spike_neurons import SPIKING_NODE_TYPES, make_spiking_node

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    _MAMBA_IMPORT_ERROR = None
except ImportError as exc:
    selective_scan_fn = None
    _MAMBA_IMPORT_ERROR = exc

__all__ = [
    "Spiking2DPatchEmbedding",
    "SpikeMambaLayer",
    "SpikMambaBlock",
    "TemporalSpikMamba",
]


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _make_neuron(init_tau, channels, channel_dim, neuron_model="plif", device=None, dtype=None, silif=None):
    node = make_spiking_node(
        neuron_model,
        channels=channels,
        channel_dim=channel_dim,
        init_tau=init_tau,
        surrogate_function=surrogate.ATan(),
        detach_reset=True, silif=silif,
    )
    if device is not None or dtype is not None:
        node = node.to(device=device, dtype=dtype)
    return node


class Spiking2DPatchEmbedding(nn.Module):
    """Project one BCHW frame into a spiking patch grid without temporal embedding."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        image_size: Tuple[int, int] = (160, 192),
        patch_size: int = 4,
        max_time_steps: int = 256,
        device=None,
        dtype=None,
        kernel_size=None,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int("in_channels", in_channels)
        self.embed_dim = _positive_int("embed_dim", embed_dim)
        self.patch_size = _positive_int("patch_size", patch_size)
        self.overlapping = kernel_size is not None
        kernel = self.patch_size if kernel_size is None else _positive_int("kernel_size", kernel_size)
        if self.overlapping and kernel % 2 == 0:
            raise ValueError("kernel_size must be odd for overlapping downsampling")
        self.max_time_steps = _positive_int(
            "max_time_steps", max_time_steps
        )
        if not isinstance(image_size, tuple) or len(image_size) != 2:
            raise ValueError("image_size must be a (height, width) tuple")
        height = _positive_int("image_size[0]", image_size[0])
        width = _positive_int("image_size[1]", image_size[1])
        if not self.overlapping and (height % self.patch_size or width % self.patch_size):
            raise ValueError("image_size must be divisible by patch_size")

        kwargs = {"device": device, "dtype": dtype}
        self.proj = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel,
            stride=self.patch_size,
            padding=kernel // 2 if self.overlapping else 0,
            bias=True,
            **kwargs,
        )
        self.norm = nn.GroupNorm(1, self.embed_dim, **kwargs)

    def forward(self, x: torch.Tensor, time_step: int) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"expected BCHW input, got shape={tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels, got {x.shape[1]}"
            )
        height, width = x.shape[-2:]
        if not self.overlapping and (height % self.patch_size or width % self.patch_size):
            raise ValueError("input size must be divisible by patch_size")
        if (
            isinstance(time_step, bool)
            or not isinstance(time_step, int)
            or not 0 <= time_step < self.max_time_steps
        ):
            raise ValueError("time_step is outside the configured range")

        return self.norm(self.proj(x))


class SpikeMambaLayer(nn.Module):
    """Spatial depthwise mixing followed by four directional SSM scans."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        init_tau: float = 2.0,
        selective_scan=None,
        conv_bias: bool = True,
        bias: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        device=None,
        dtype=None,
        max_time_steps: int = 256,
        linear_projection: bool = True,
        patch_embedding_spiking: bool = False,
        dwconv2d_spiking: bool = True,
        spatial_output_spiking: bool = False,
        dwconv2d_kernel_size: int = 3,
        neuron_model: str = "plif",
        silif=None,
    ) -> None:
        super().__init__()
        self.dim = _positive_int("dim", dim)
        self.d_state = _positive_int("d_state", d_state)
        self.expand = _positive_int("expand", expand)
        self.max_time_steps = _positive_int("max_time_steps", max_time_steps)
        if not isinstance(linear_projection, bool):
            raise TypeError("linear_projection must be a boolean")
        if not isinstance(patch_embedding_spiking, bool):
            raise TypeError("patch_embedding_spiking must be a boolean")
        if not isinstance(dwconv2d_spiking, bool):
            raise TypeError("dwconv2d_spiking must be a boolean")
        if not isinstance(spatial_output_spiking, bool):
            raise TypeError("spatial_output_spiking must be a boolean")
        self.linear_projection = linear_projection
        self.patch_embedding_spiking = patch_embedding_spiking
        self.dwconv2d_spiking = dwconv2d_spiking
        self.spatial_output_spiking = spatial_output_spiking
        if neuron_model not in ("plif", "silif"):
            raise ValueError("neuron_model must be 'plif' or 'silif'")
        self.neuron_model = neuron_model
        self.dwconv2d_kernel_size = _positive_int("dwconv2d_kernel_size", dwconv2d_kernel_size)
        if self.dwconv2d_kernel_size % 2 == 0:
            raise ValueError("dwconv2d_kernel_size must be odd")
        self.num_directions = 4
        self.d_inner = self.expand * self.dim if linear_projection else self.dim
        self.dt_rank = (
            math.ceil(self.dim / 16)
            if dt_rank == "auto"
            else _positive_int("dt_rank", dt_rank)
        )
        if not all(
            math.isfinite(value)
            for value in (dt_min, dt_max, dt_scale, dt_init_floor)
        ):
            raise ValueError("Delta configuration values must be finite")
        if dt_min <= 0 or dt_max <= 0 or dt_min > dt_max:
            raise ValueError("require 0 < dt_min <= dt_max")
        if dt_init_floor <= 0 or dt_init_floor > dt_max:
            raise ValueError("require 0 < dt_init_floor <= dt_max")
        if dt_init not in {"constant", "random"}:
            raise NotImplementedError(f"unsupported dt_init={dt_init!r}")

        kwargs = {"device": device, "dtype": dtype}
        if self.linear_projection:
            self.linear_m = nn.Linear(
                self.dim, self.d_inner, bias=bias, **kwargs
            )
        else:
            self.linear_m = nn.Identity()
        self.patch_sl = (
            _make_neuron(init_tau, self.d_inner, -1, neuron_model, device=device, dtype=dtype, silif=silif)
            if self.patch_embedding_spiking
            else nn.GELU()
        )
        self.dwconv2d = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.dwconv2d_kernel_size,
            groups=self.d_inner,
            padding=self.dwconv2d_kernel_size // 2,
            bias=conv_bias,
            **kwargs,
        )
        self.dwconv2d_sl = (
            _make_neuron(init_tau, self.d_inner, 1, neuron_model, device=device, dtype=dtype, silif=silif)
            if self.dwconv2d_spiking
            else nn.GELU()
        )
        self.ss2d_output_plif = (
            _make_neuron(init_tau, self.d_inner, -1, neuron_model, device=device, dtype=dtype, silif=silif)
            if self.spatial_output_spiking
            else nn.GELU()
        )
        x_proj_layers = [
            nn.Linear(
                self.d_inner,
                self.dt_rank + 2 * self.d_state,
                bias=False,
                **kwargs,
            )
            for _ in range(self.num_directions)
        ]
        self.x_proj_weight = nn.Parameter(
            torch.stack([layer.weight for layer in x_proj_layers], dim=0)
        )
        self.dt_proj_weight = nn.Parameter(
            torch.empty(
                self.num_directions, self.d_inner, self.dt_rank, **kwargs
            )
        )
        std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj_weight, std)
        else:
            nn.init.uniform_(self.dt_proj_weight, -std, std)
        dt = torch.exp(
            torch.rand(
                self.num_directions, self.d_inner, **kwargs
            )
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inverse_softplus = dt + torch.log(-torch.expm1(-dt))
        self.dt_proj_bias = nn.Parameter(inverse_softplus)
        self.dt_proj_bias._no_reinit = True

        base = torch.arange(
            1, self.d_state + 1,
            dtype=torch.float32, device=device,
        )
        self.A_log = nn.Parameter(
            torch.log(base).view(1, 1, self.d_state).repeat(
                self.num_directions, self.d_inner, 1
            )
        )
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(
            torch.ones(
                self.num_directions, self.d_inner,
                dtype=torch.float32, device=device,
            )
        )
        self.D._no_weight_decay = True
        if self.linear_projection:
            self.out_proj = nn.Linear(
                self.d_inner, self.dim, bias=bias, **kwargs
            )
        else:
            self.out_proj = nn.Identity()
        self.selective_scan = (
            selective_scan
            if selective_scan is not None
            else selective_scan_fn
        )

    def _cross_scan_routes(self, x: torch.Tensor, height: int, width: int):
        """Build four patch routes from a grid tensor of shape (B, H, W, D)."""
        batch = x.shape[0]
        x_grid = x.reshape(batch, height, width, self.d_inner)

        row_major = x_grid.reshape(batch, height * width, self.d_inner)
        row_major_rev = torch.flip(row_major, dims=[1])

        col_major = x_grid.permute(0, 2, 1, 3).reshape(
            batch, height * width, self.d_inner
        )
        col_major_rev = torch.flip(col_major, dims=[1])

        return [row_major, row_major_rev, col_major, col_major_rev]

    def forward(
        self,
        tokens: torch.Tensor,
        time_step: int,
        spatial_shape: Tuple[int, int],
        return_inner: bool = False,
        sequence_shape: Tuple[int, int] = None,
        debug_shapes: bool = False,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                f"expected BLD patch tokens, got {tuple(tokens.shape)}"
            )
        if tokens.shape[-1] != self.dim:
            raise ValueError(
                f"expected token dimension {self.dim}, got {tokens.shape[-1]}"
            )
        if not isinstance(spatial_shape, tuple) or len(spatial_shape) != 2:
            raise ValueError("spatial_shape must be a (height, width) tuple")
        height = _positive_int("spatial_shape[0]", spatial_shape[0])
        width = _positive_int("spatial_shape[1]", spatial_shape[1])
        if tokens.shape[1] != height * width:
            raise ValueError(
                "token count must equal spatial_shape height * width"
            )
        if self.selective_scan is None:
            raise ImportError(
                "mamba_ssm or an injected selective_scan is required"
            ) from _MAMBA_IMPORT_ERROR

        p_global = self.linear_m(tokens)
        if debug_shapes:
            print(f"after input projection: {tuple(p_global.shape)}")
        batch = tokens.shape[0]
        if sequence_shape is not None:
            sequence_batch, steps = sequence_shape
            if sequence_batch * steps != batch:
                raise ValueError("sequence_shape does not match flattened slice batch")
        else:
            sequence_batch, steps = batch, 1
        if self.patch_embedding_spiking:
            if sequence_shape is None:
                p_global, _ = self.patch_sl(p_global, time_step)
            else:
                # [B*T,N,C] -> [B,T,N,C], PLIF evolves over T without folding T into its batch.
                sequence = p_global.reshape(sequence_batch, steps, *p_global.shape[1:])
                values = [self.patch_sl(sequence[:, t], time_step + t)[0] for t in range(steps)]
                p_global = torch.stack(values, dim=1).reshape_as(p_global)
        else:
            p_global = self.patch_sl(p_global)

        spatial_features = (
            p_global.transpose(1, 2)
            .contiguous()
            .reshape(batch, self.d_inner, height, width)
        )
        spatial_features = self.dwconv2d(spatial_features)
        if self.dwconv2d_spiking:
            if sequence_shape is None:
                spatial_features, _ = self.dwconv2d_sl(
                    spatial_features, time_step
                )
            else:
                # [B*T,C,Hp,Wp] -> [B,T,C,Hp,Wp] for sequential membrane updates.
                sequence = spatial_features.reshape(
                    sequence_batch, steps, self.d_inner, height, width
                )
                values = [
                    self.dwconv2d_sl(sequence[:, t], time_step + t)[0]
                    for t in range(steps)
                ]
                spatial_features = torch.stack(values, dim=1).reshape_as(spatial_features)
        else:
            spatial_features = self.dwconv2d_sl(spatial_features)
        p_global = (
            spatial_features.flatten(2)
            .transpose(1, 2)
            .contiguous()
        )

        routes = torch.stack(
            self._cross_scan_routes(p_global, height, width),
            dim=1,
        )
        route_convs = routes.permute(0, 1, 3, 2).contiguous()

        sequence_length = height * width
        projection_size = self.dt_rank + 2 * self.d_state
        projected = F.conv1d(
            route_convs.reshape(
                batch,
                self.num_directions * self.d_inner,
                sequence_length,
            ),
            self.x_proj_weight.reshape(
                self.num_directions * projection_size,
                self.d_inner,
                1,
            ),
            groups=self.num_directions,
        ).view(
            batch,
            self.num_directions,
            projection_size,
            sequence_length,
        )
        delta_low, state_B, state_C = torch.split(
            projected,
            [self.dt_rank, self.d_state, self.d_state],
            dim=2,
        )
        delta = F.conv1d(
            delta_low.reshape(
                batch,
                self.num_directions * self.dt_rank,
                sequence_length,
            ),
            self.dt_proj_weight.reshape(
                self.num_directions * self.d_inner,
                self.dt_rank,
                1,
            ),
            groups=self.num_directions,
        )
        scan_input = route_convs.reshape(
            batch,
            self.num_directions * self.d_inner,
            sequence_length,
        )
        y = self.selective_scan(
            scan_input,
            delta,
            -torch.exp(self.A_log.float()).reshape(
                self.num_directions * self.d_inner,
                self.d_state,
            ),
            state_B,
            state_C,
            self.D.float().reshape(self.num_directions * self.d_inner),
            z=None,
            delta_bias=self.dt_proj_bias.float().reshape(
                self.num_directions * self.d_inner
            ),
            delta_softplus=True,
            return_last_state=False,
        )
        route_outputs = (
            y.reshape(
                batch,
                self.num_directions,
                self.d_inner,
                sequence_length,
            )
            .permute(0, 1, 3, 2)
            .contiguous()
        )

        row_forward = route_outputs[:, 0]
        row_backward = torch.flip(route_outputs[:, 1], dims=[1])
        col_forward = (
            route_outputs[:, 2]
            .reshape(batch, width, height, self.d_inner)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch, sequence_length, self.d_inner)
        )
        col_backward = (
            torch.flip(route_outputs[:, 3], dims=[1])
            .reshape(batch, width, height, self.d_inner)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch, sequence_length, self.d_inner)
        )
        y = 0.25 * (
            row_forward + row_backward + col_forward + col_backward
        )
        if isinstance(self.ss2d_output_plif, SPIKING_NODE_TYPES):
            if sequence_shape is None:
                y, _ = self.ss2d_output_plif(y, time_step)
            else:
                # [B*T,N,C] -> [B,T,N,C] so this PLIF advances in slice order.
                sequence = y.reshape(sequence_batch, steps, *y.shape[1:])
                values = [
                    self.ss2d_output_plif(sequence[:, t], time_step + t)[0]
                    for t in range(steps)
                ]
                y = torch.stack(values, dim=1).reshape_as(y)
        else:
            y = self.ss2d_output_plif(y)
        if return_inner:
            return y
        if self.linear_projection:
            y = y.to(self.out_proj.weight.dtype)
        else:
            y = y.to(tokens.dtype)
        return self.out_proj(y)


class TemporalSpikMamba(nn.Module):
    """Depthwise spiking mixer and one selective scan along MRI depth T."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        dt_rank: Union[int, str] = "auto",
        kernel_size: int = 3,
        init_tau: float = 2.0,
        selective_scan=None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        debug_shapes: bool = False,
        norm_spiking: bool = True,
        dwconv1D_spiking: bool = True,
        plif_enabled: Optional[bool] = None,
        neuron_model: str = "plif",
        silif=None,
    ) -> None:
        super().__init__()
        self.dim = _positive_int("dim", dim)
        self.d_state = _positive_int("d_state", d_state)
        self.kernel_size = _positive_int("kernel_size", kernel_size)
        if not isinstance(debug_shapes, bool):
            raise TypeError("debug_shapes must be a boolean")
        if plif_enabled is not None:
            if not isinstance(plif_enabled, bool):
                raise TypeError("plif_enabled must be a boolean")
            # Backward compatibility: the old switch controlled both PLIFs.
            norm_spiking = dwconv1D_spiking = plif_enabled
        if not isinstance(norm_spiking, bool):
            raise TypeError("norm_spiking must be a boolean")
        if not isinstance(dwconv1D_spiking, bool):
            raise TypeError("dwconv1D_spiking must be a boolean")
        self.debug_shapes = debug_shapes
        self.norm_spiking = norm_spiking
        self.dwconv1D_spiking = dwconv1D_spiking
        self.dt_rank = math.ceil(self.dim / 16) if dt_rank == "auto" else _positive_int("dt_rank", dt_rank)
        if not all(math.isfinite(v) for v in (dt_min, dt_max, dt_scale, dt_init_floor)):
            raise ValueError("Delta configuration values must be finite")
        if dt_min <= 0 or dt_max < dt_min or dt_init_floor <= 0 or dt_init_floor > dt_max:
            raise ValueError("require 0 < dt_min <= dt_max and 0 < dt_init_floor <= dt_max")
        if dt_init not in {"constant", "random"}:
            raise NotImplementedError(f"unsupported dt_init={dt_init!r}")

        self.norm = nn.GroupNorm(1, self.dim)
        self.norm_plif = _make_neuron(init_tau, self.dim, -1, neuron_model, silif=silif) if self.norm_spiking else nn.GELU()
        self.dwconv1d = nn.Conv1d(
            self.dim, self.dim, self.kernel_size, groups=self.dim,
            padding=0, bias=True,
        )
        self.conv_plif = _make_neuron(init_tau, self.dim, -1, neuron_model, silif=silif) if self.dwconv1D_spiking else nn.GELU()
        self.x_proj = nn.Linear(self.dim, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj_weight = nn.Parameter(torch.empty(self.dim, self.dt_rank))
        std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj_weight, std)
        else:
            nn.init.uniform_(self.dt_proj_weight, -std, std)
        dt = torch.exp(
            torch.rand(self.dim) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        self.dt_proj_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_proj_bias._no_reinit = True
        base = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(base).view(1, -1).repeat(self.dim, 1))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.dim, dtype=torch.float32))
        self.D._no_weight_decay = True
        self.selective_scan = selective_scan if selective_scan is not None else selective_scan_fn

    @staticmethod
    def _step_plif(node: nn.Module, x: torch.Tensor, time_step: int) -> torch.Tensor:
        """Advance membrane state one temporal slice at a time."""
        outputs = []
        for t in range(x.shape[1]):
            # [B*N,T,C] -> [B*N,C] for one slice; T stays outside PLIF's batch.
            value, _ = node(x[:, t, :], time_step + t)
            outputs.append(value)
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor, time_step: int = 0) -> torch.Tensor:
        if x.ndim != 4 or x.shape[-1] != self.dim:
            raise ValueError(f"expected [B,T,N,{self.dim}], got {tuple(x.shape)}")
        if self.selective_scan is None:
            raise ImportError("mamba_ssm or an injected selective_scan is required") from _MAMBA_IMPORT_ERROR
        batch, steps, tokens, channels = x.shape
        if steps <= 0:
            raise ValueError("temporal sequence must contain at least one slice")
        f_spatial = x
        # [B,T,N,C] -> [B,N,T,C], keeping samples and token sequences separate.
        temporal = x.permute(0, 2, 1, 3).contiguous()
        # [B,N,T,C] -> [B*N,T,C]; the SSM sequence axis is explicitly T.
        temporal = temporal.reshape(batch * tokens, steps, channels)
        if self.debug_shapes:
            print(f"after temporal reshape [B*N,T,C]: {tuple(temporal.shape)}")
        # GroupNorm expects channels at dim 1: [B*N,T,C] -> [B*N,C,T].
        temporal = self.norm(temporal.transpose(1, 2).contiguous())
        # [B*N,C,T] -> [B*N,T,C]; PLIF advances once for each T position.
        temporal = temporal.transpose(1, 2).contiguous()
        if self.norm_spiking:
            temporal = self._step_plif(self.norm_plif, temporal, time_step)
        else:
            temporal = self.norm_plif(temporal)
        # [B*N,C,T] receives asymmetric end padding so any kernel size preserves T.
        conv_input = temporal.transpose(1, 2).contiguous()
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        conv_input = F.pad(conv_input, (pad_left, pad_right))
        temporal_conv = self.dwconv1d(conv_input)
        if self.debug_shapes:
            print(f"after temporal DWConv1D [B*N,C,T]: {tuple(temporal_conv.shape)}")
        temporal = temporal_conv.transpose(1, 2).contiguous()
        if self.dwconv1D_spiking:
            temporal = self._step_plif(self.conv_plif, temporal, time_step)
        else:
            temporal = self.conv_plif(temporal)
        if temporal.shape[1] != steps:
            raise RuntimeError("temporal depthwise convolution changed sequence length")

        projected = self.x_proj(temporal)
        delta_low, state_b, state_c = torch.split(
            projected, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.linear(delta_low, self.dt_proj_weight).transpose(1, 2).contiguous()
        scan_input = temporal.transpose(1, 2).contiguous()  # [B*N,C,T], scan axis T.
        if self.debug_shapes:
            print(f"before temporal selective SSM [B*N,C,T]: {tuple(scan_input.shape)}")
        y = self.selective_scan(
            scan_input,
            delta,
            -torch.exp(self.A_log.float()),
            state_b.transpose(1, 2).contiguous(),
            state_c.transpose(1, 2).contiguous(),
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj_bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )
        if self.debug_shapes:
            print(f"after temporal selective SSM [B*N,C,T]: {tuple(y.shape)}")
        # [B*N,C,T] -> [B*N,T,C] -> [B,T,N,C].
        f_temporal = y.transpose(1, 2).contiguous().reshape(batch, tokens, steps, channels)
        f_temporal = f_temporal.permute(0, 2, 1, 3).contiguous()
        assert f_temporal.shape == f_spatial.shape
        fused = f_spatial + f_temporal
        if self.debug_shapes:
            print(f"after spatial-temporal residual [B,T,N,C]: {tuple(fused.shape)}")
        return fused


class SpikMambaBlock(nn.Module):
    """Minimal residual patch mixer: Mamba + residual only."""

    def __init__(
        self,
        dim: int,
        residual_connections: bool = True,
        temporal_mamba=None,
        **mamba_kwargs,
    ) -> None:
        super().__init__()
        self.dim = _positive_int("dim", dim)
        if not isinstance(residual_connections, bool):
            raise TypeError("residual_connections must be a boolean")
        self.residual_connections = residual_connections
        self.mamba_layer = SpikeMambaLayer(
            dim=self.dim,
            **mamba_kwargs,
        )
        temporal_config = temporal_mamba or {"enabled": False}
        if not isinstance(temporal_config, dict):
            raise TypeError("temporal_mamba must be a mapping")
        if not isinstance(temporal_config.get("enabled", False), bool):
            raise TypeError("temporal_mamba.enabled must be a boolean")
        if not isinstance(temporal_config.get("debug_shapes", False), bool):
            raise TypeError("temporal_mamba.debug_shapes must be a boolean")
        legacy_plif = temporal_config.get("plif_enabled")
        if legacy_plif is not None and not isinstance(legacy_plif, bool):
            raise TypeError("temporal_mamba.plif_enabled must be a boolean")
        norm_spiking = temporal_config.get("norm_spiking", legacy_plif if legacy_plif is not None else True)
        dwconv1D_spiking = temporal_config.get("dwconv1D_spiking", legacy_plif if legacy_plif is not None else True)
        if not isinstance(norm_spiking, bool):
            raise TypeError("temporal_mamba.norm_spiking must be a boolean")
        if not isinstance(dwconv1D_spiking, bool):
            raise TypeError("temporal_mamba.dwconv1D_spiking must be a boolean")
        if temporal_config.get("enabled", False):
            self.temporal_mamba = TemporalSpikMamba(
                dim=self.mamba_layer.d_inner,
                d_state=temporal_config.get("d_state", self.mamba_layer.d_state),
                dt_rank=temporal_config.get("dt_rank", self.mamba_layer.dt_rank),
                kernel_size=temporal_config.get("kernel_size", 3),
                init_tau=mamba_kwargs.get("init_tau", 2.0),
                selective_scan=self.mamba_layer.selective_scan,
                debug_shapes=temporal_config.get("debug_shapes", False),
                norm_spiking=norm_spiking,
                dwconv1D_spiking=dwconv1D_spiking,
                neuron_model=mamba_kwargs.get("neuron_model", "plif"),
                silif=mamba_kwargs.get("silif"),
            )
        else:
            self.temporal_mamba = None
        self.debug_shapes = bool(temporal_config.get("debug_shapes", False))

    def forward(
        self,
        tokens: torch.Tensor,
        time_step: int,
        spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [B, L, D]")
        if tokens.shape[-1] != self.dim:
            raise ValueError(
                f"expected token dimension {self.dim}, got {tokens.shape[-1]}"
            )

        mamba_features = self.mamba_layer(
            tokens,
            time_step,
            spatial_shape=spatial_shape,
        )
        if self.residual_connections:
            return tokens + mamba_features
        return mamba_features

    def forward_sequence(
        self,
        tokens: torch.Tensor,
        time_step: int,
        spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Process [B,T,N,D] with per-slice SS2D and optional temporal SSM."""
        if tokens.ndim != 4 or tokens.shape[-1] != self.dim:
            raise ValueError(f"expected [B,T,N,{self.dim}] tokens, got {tuple(tokens.shape)}")
        batch, steps, token_count, dim = tokens.shape
        # [B,T,N,D] -> [B*T,N,D]; SS2D sequence axis remains N per slice.
        flat_tokens = tokens.reshape(batch * steps, token_count, dim)
        flat_inner = self.mamba_layer(
            flat_tokens, time_step, spatial_shape, return_inner=True,
            sequence_shape=(batch, steps), debug_shapes=self.debug_shapes,
        )
        # [B*T,N,D_inner] -> [B,T,N,D_inner] before temporal modeling.
        f_spatial = flat_inner.reshape(batch, steps, token_count, -1)
        if self.debug_shapes:
            print(f"after Spatial SS2D fusion [B,T,N,C]: {tuple(f_spatial.shape)}")
        if self.temporal_mamba is not None:
            if self.debug_shapes:
                print(f"before Temporal Mamba [B,T,N,C]: {tuple(f_spatial.shape)}")
            f_spatiotemporal = self.temporal_mamba(f_spatial, time_step=time_step)
        else:
            f_spatiotemporal = f_spatial
        # [B,T,N,D_inner] -> [B,T,N,D] through the existing output projection.
        if self.mamba_layer.linear_projection:
            f_spatiotemporal = f_spatiotemporal.to(self.mamba_layer.out_proj.weight.dtype)
        else:
            f_spatiotemporal = f_spatiotemporal.to(tokens.dtype)
        output = self.mamba_layer.out_proj(f_spatiotemporal)
        if self.debug_shapes:
            print(f"after output projection [B,T,N,D]: {tuple(output.shape)}")
        if self.residual_connections:
            output = tokens + output
        return output
