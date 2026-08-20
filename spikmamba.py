"""Patch-based spiking Mamba components for per-frame BCHW features."""

import math

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import surrogate
from spike_neurons import PLIFNode

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
]


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _make_plif(init_tau: float, device=None, dtype=None) -> PLIFNode:
    node = PLIFNode(
        init_tau=init_tau,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
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
        init_tau: float = 2.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int("in_channels", in_channels)
        self.embed_dim = _positive_int("embed_dim", embed_dim)
        self.patch_size = _positive_int("patch_size", patch_size)
        self.max_time_steps = _positive_int(
            "max_time_steps", max_time_steps
        )
        if not isinstance(image_size, tuple) or len(image_size) != 2:
            raise ValueError("image_size must be a (height, width) tuple")
        height = _positive_int("image_size[0]", image_size[0])
        width = _positive_int("image_size[1]", image_size[1])
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")

        kwargs = {"device": device, "dtype": dtype}
        self.proj = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            self.patch_size,
            stride=self.patch_size,
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
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("input size must be divisible by patch_size")
        if (
            isinstance(time_step, bool)
            or not isinstance(time_step, int)
            or not 0 <= time_step < self.max_time_steps
        ):
            raise ValueError("time_step is outside the configured range")

        return self.norm(self.proj(x))


class SpikeMambaLayer(nn.Module):
    """Continuous causal SSM over row-major patch tokens."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
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
    ) -> None:
        super().__init__()
        self.dim = _positive_int("dim", dim)
        self.d_state = _positive_int("d_state", d_state)
        self.d_conv = _positive_int("d_conv", d_conv)
        self.expand = _positive_int("expand", expand)
        self.max_time_steps = _positive_int("max_time_steps", max_time_steps)
        self.d_inner = self.expand * self.dim
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
        self.linear_m = nn.Linear(
            self.dim, self.d_inner, bias=bias, **kwargs
        )
        self.sl_m1 = _make_plif(init_tau, device=device, dtype=dtype)
        self.conv1d_m = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=conv_bias,
            **kwargs,
        )
        self.sl_m2 = _make_plif(init_tau, device=device, dtype=dtype)
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * self.d_state,
            bias=False,
            **kwargs,
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **kwargs
        )

        std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, std)
        else:
            nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(
            torch.rand(self.d_inner, **kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inverse_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inverse_softplus)
        self.dt_proj.bias._no_reinit = True

        base = torch.arange(
            1, self.d_state + 1,
            dtype=torch.float32, device=device,
        )
        self.A_log = nn.Parameter(
            torch.log(base.unsqueeze(0).repeat(self.d_inner, 1))
        )
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(
            torch.ones(self.d_inner, dtype=torch.float32, device=device)
        )
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(
            self.d_inner, self.dim, bias=bias, **kwargs
        )
        self.sl_ssm = _make_plif(init_tau, device=device, dtype=dtype)
        self.selective_scan = (
            selective_scan
            if selective_scan is not None
            else selective_scan_fn
        )

    @staticmethod
    def _to_tokens(patches: torch.Tensor) -> torch.Tensor:
        # B D H W -> B (H W) D, row-major.
        return patches.flatten(2).transpose(1, 2).contiguous()

    def _causal_conv(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.conv1d_m(sequence)[..., :sequence.shape[-1]]

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

    def forward(self, patches: torch.Tensor, time_step: int) -> torch.Tensor:
        if patches.ndim != 4:
            raise ValueError(
                f"expected BDHW patches, got {tuple(patches.shape)}"
            )
        if patches.shape[1] != self.dim:
            raise ValueError(
                f"expected {self.dim} channels, got {patches.shape[1]}"
            )
        if self.selective_scan is None:
            raise ImportError(
                "mamba_ssm or an injected selective_scan is required"
            ) from _MAMBA_IMPORT_ERROR

        batch, _, height, width = patches.shape
        p_local = self._to_tokens(patches)

        p_global = self.linear_m(p_local)
        p_global, _ = self.sl_m1(p_global, time_step)

        route_outputs = []
        for route in self._cross_scan_routes(p_global, height, width):
            route_conv = self._causal_conv(route.transpose(1, 2).contiguous())
            route_conv, _ = self.sl_m2(route_conv, time_step)

            projected = self.x_proj(route_conv.transpose(1, 2))
            delta_low, B, C = torch.split(
                projected,
                [self.dt_rank, self.d_state, self.d_state],
                dim=-1,
            )
            delta = F.linear(
                delta_low, self.dt_proj.weight, bias=None
            ).transpose(1, 2).contiguous()
            B = B.transpose(1, 2).contiguous()
            C = C.transpose(1, 2).contiguous()
            A = -torch.exp(self.A_log.float())

            y = self.selective_scan(
                route_conv,
                delta,
                A,
                B,
                C,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=False,
            )
            y = self.out_proj(
                y.transpose(1, 2).to(self.out_proj.weight.dtype)
            )
            route_outputs.append(y)

        y = torch.stack(route_outputs, dim=0).mean(dim=0)
        y, _ = self.sl_ssm(y, time_step)
        gated = y * p_local
        return (
            gated.transpose(1, 2)
            .contiguous()
            .view(batch, self.dim, height, width)
        )


class SpikMambaBlock(nn.Module):
    """Minimal residual patch mixer: Mamba + residual only."""

    def __init__(
        self,
        dim: int,
        **mamba_kwargs,
    ) -> None:
        super().__init__()
        self.dim = _positive_int("dim", dim)
        self.mamba_layer = SpikeMambaLayer(dim=self.dim, **mamba_kwargs)

    def forward(self, patches: torch.Tensor, time_step: int) -> torch.Tensor:
        if patches.ndim != 4:
            raise ValueError("patches must have shape [B, D, Hp, Wp]")
        if patches.shape[1] != self.dim:
            raise ValueError(
                f"expected {self.dim} patch channels, got {patches.shape[1]}"
            )

        return patches + self.mamba_layer(patches, time_step)
