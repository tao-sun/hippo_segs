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
    """Project one BCHW frame into a spiking patch grid."""

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
        self.norm = nn.BatchNorm2d(self.embed_dim, **kwargs)
        self.sl_patch = _make_plif(
            init_tau, device=device, dtype=dtype
        )
        self.spatial_pos_embed = nn.Parameter(
            torch.empty(
                1,
                self.embed_dim,
                height // self.patch_size,
                width // self.patch_size,
                **kwargs,
            )
        )
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(self.max_time_steps, self.embed_dim, **kwargs)
        )
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)

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

        patches = self.norm(self.proj(x))
        patches, _ = self.sl_patch(patches, time_step)
        spatial = self.spatial_pos_embed
        if spatial.shape[-2:] != patches.shape[-2:]:
            spatial = F.interpolate(
                spatial,
                size=patches.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        temporal = self.temporal_pos_embed[time_step].view(
            1, self.embed_dim, 1, 1
        )
        return patches + spatial + temporal


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
    ) -> None:
        super().__init__()
        self.dim = _positive_int("dim", dim)
        self.d_state = _positive_int("d_state", d_state)
        self.d_conv = _positive_int("d_conv", d_conv)
        self.expand = _positive_int("expand", expand)
        self.d_inner = self.expand * self.dim
        self.dt_rank = (
            math.ceil(self.dim / 16)
            if dt_rank == "auto"
            else _positive_int("dt_rank", dt_rank)
        )
        if dt_min <= 0 or dt_max <= 0 or dt_min > dt_max:
            raise ValueError("require 0 < dt_min <= dt_max")
        if dt_init_floor <= 0:
            raise ValueError("dt_init_floor must be positive")
        if dt_init not in {"constant", "random"}:
            raise NotImplementedError(f"unsupported dt_init={dt_init!r}")

        kwargs = {"device": device, "dtype": dtype}
        self.linear_m = nn.Linear(
            self.dim, self.d_inner, bias=bias, **kwargs
        )
        self.sl_m1 = _make_plif(
            init_tau, device=device, dtype=dtype
        )
        self.conv1d_m = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=conv_bias,
            **kwargs,
        )
        self.sl_m2 = _make_plif(
            init_tau, device=device, dtype=dtype
        )
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
        self.sl_ssm = _make_plif(
            init_tau, device=device, dtype=dtype
        )
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
        original = self._to_tokens(patches)
        x = self.linear_m(original)
        x, _ = self.sl_m1(x, time_step)
        x = self._causal_conv(x.transpose(1, 2).contiguous())
        x, _ = self.sl_m2(x, time_step)

        projected = self.x_proj(x.transpose(1, 2))
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

        # Continuous recurrence: no spike operation occurs in this call.
        y = self.selective_scan(
            x,
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
        ssm_spikes, _ = self.sl_ssm(y, time_step)
        gated = ssm_spikes * original
        return (
            gated.transpose(1, 2)
            .contiguous()
            .view(batch, self.dim, height, width)
        )


class SpikMambaBlock(nn.Module):
    """Mamba patch mixer followed by a token-wise FFN, both residual."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        **mamba_kwargs,
    ) -> None:
        super().__init__()
        self.dim = _positive_int("dim", dim)
        if not isinstance(mlp_ratio, (int, float)) or mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if not isinstance(dropout, (int, float)) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")

        hidden_dim = int(self.dim * float(mlp_ratio))
        if hidden_dim < 1:
            raise ValueError("mlp_ratio produces an empty hidden dimension")

        self.mamba_layer = SpikeMambaLayer(dim=self.dim, **mamba_kwargs)
        self.ffn_norm = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, patches: torch.Tensor, time_step: int) -> torch.Tensor:
        if patches.ndim != 4:
            raise ValueError("patches must have shape [B, D, Hp, Wp]")
        if patches.shape[1] != self.dim:
            raise ValueError(
                f"expected {self.dim} patch channels, got {patches.shape[1]}"
            )

        global_features = patches + self.mamba_layer(patches, time_step)
        batch, channels, height, width = global_features.shape
        tokens = global_features.flatten(2).transpose(1, 2)
        ffn_tokens = self.ffn(self.ffn_norm(tokens))
        ffn_features = ffn_tokens.transpose(1, 2).reshape(
            batch,
            channels,
            height,
            width,
        )
        return global_features + ffn_features
