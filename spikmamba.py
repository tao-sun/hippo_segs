"""Patch-based spiking Mamba components for per-frame BCHW features."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import surrogate
from spike_neurons import PLIFNode

__all__ = ["Spiking2DPatchEmbedding"]


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
