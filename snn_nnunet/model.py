import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from snn_nnunet import surrogate
from snn_nnunet.config_aliases import normalize_spiking_options
from snn_nnunet.spike_neurons import PLIFNode, SiLIFNode, SPIKING_NODE_TYPES, make_spiking_node
from snn_nnunet.silif_config import SILIF_DEFAULTS, silif_options
from snn_nnunet.spikmamba import Spiking2DPatchEmbedding, SpikMambaBlock

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    _MAMBA_IMPORT_ERROR = None
except ImportError as exc:
    selective_scan_fn = None
    _MAMBA_IMPORT_ERROR = exc


def print_model_info(model: nn.Module):
    """
    Print a concise summary of the model, similar to PyTorch Lightning,
    but without input/output shapes. Shows parameter counts with K/M units.
    """
    def human_readable(num):
        if num >= 1e6:
            return f"{num/1e6:.2f} M"
        elif num >= 1e3:
            return f"{num/1e3:.2f} K"
        return str(num)

    print("=" * 60)
    print(f"{'Layer (type)':35s} {'Param #':>12s}")
    print("=" * 60)
    total_params = 0
    trainable_params = 0

    for name, module in model.named_modules():
        # Skip container modules
        if len(list(module.children())) > 0 and not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            continue

        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if params > 0:
            print(f"{name:35s} {human_readable(params):>12s}")
            total_params += params
            trainable_params += params

    print("=" * 60)
    print(f"{'Total trainable params:':35s} {human_readable(trainable_params):>12s}")
    print("=" * 60)


ARCHITECTURE_DEFAULTS = {
    "neuron_model": "plif",
    "silif": SILIF_DEFAULTS,
    "hierarchical_encoder": True,
    "stem_depth": 1,
    "stem_spiking": True,
    "downsampling_spiking": True,
    "stem_channels": 32,
    "encoder_channels": (64, 128, 256),
    "downsample_kernel_size": 3,
    "downsample_strides": (2, 2, 2),
    "dwconv2d_kernel_size": 3,
    "vss_depths": (1, 1, 2),
    "full_resolution_skip": True,
    "final_skip_fusion": "concat",
    "decoder_block": "standard",
    "resdecoder_spiking1": True,
    "resdecoder_spiking2": True,
    "fue_skip1": False,
    "fue_skip2": False,
}


def output_spiking_option(values):
    """Read the global output switch, accepting its former VSS-specific name."""
    names = [name for name in ("output_spiking", "vss_output_spiking") if name in values]
    if len(names) > 1:
        raise ValueError("Specify only output_spiking; vss_output_spiking is a legacy alias")
    name = names[0] if names else "output_spiking"
    value = values.get(name, True)
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def architecture_options(values):
    """Validate optional architecture keys shared by the existing entry points."""
    values = normalize_spiking_options(values)
    result = {key: values.get(key, default) for key, default in ARCHITECTURE_DEFAULTS.items()}
    result["silif"] = silif_options(result["silif"])
    if result["neuron_model"] not in ("plif", "silif"):
        raise ValueError("neuron_model must be 'plif' or 'silif'")
    for key in ("hierarchical_encoder", "full_resolution_skip", "stem_spiking", "downsampling_spiking",
                "resdecoder_spiking1", "resdecoder_spiking2", "fue_skip1", "fue_skip2"):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    for key in ("stem_depth", "stem_channels", "downsample_kernel_size", "dwconv2d_kernel_size"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("downsample_kernel_size", "dwconv2d_kernel_size"):
        if result[key] % 2 == 0:
            raise ValueError(f"{key} must be odd")
    for key in ("encoder_channels", "vss_depths", "downsample_strides"):
        value = result[key]
        if (not isinstance(value, (list, tuple)) or len(value) != 3
                or any(type(item) is not int or item < 1 for item in value)):
            raise ValueError(f"{key} must contain three positive integers")
        result[key] = list(value)
    if not result["hierarchical_encoder"] and result["downsample_strides"] != [2, 2, 2]:
        raise ValueError("Custom downsample_strides requires hierarchical_encoder=true")
    if result["final_skip_fusion"] != "concat":
        raise ValueError("final_skip_fusion must be 'concat'")
    if result["decoder_block"] not in ("standard", "resdecoder"):
        raise ValueError("decoder_block must be standard or resdecoder")
    return result


class SpikingStem(nn.Module):
    """Full-resolution Conv2D + GroupNorm blocks with a neuron or GELU."""

    def __init__(self, in_channels=4, channels=32, depth=1, init_tau=2.0, stem_spiking=True,
                 neuron_model="plif", silif=None):
        super().__init__()
        architecture_options({"stem_depth": depth, "stem_channels": channels, "stem_spiking": stem_spiking})
        self.blocks = nn.ModuleList([
            ConvBlock(in_channels if i == 0 else channels, channels,
                      padding=1, dropout=0.0, init_tau=init_tau, spiking=stem_spiking,
                      neuron_model=neuron_model, silif=silif)
            for i in range(depth)
        ])

    def forward(self, x, time_step):
        for block in self.blocks:
            x = block(x, time_step)
        return x

    def forward_sequence(self, x, time_step):
        # PLIF state uses B as its batch; advance T in the original slice order.
        return torch.stack([self(x[:, t], time_step + t)
                            for t in range(x.shape[1])], dim=1)


class _SpikMambaEncoderAdapter(nn.Module):
    """Downsample one feature map with patch embedding, then apply SS2D."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 4,
        d_state: int = 16,
        dt_rank: Union[int, str] = "auto",
        init_tau: float = 2.0,
        selective_scan=None,
        max_time_steps: int = 256,
        linear_projection: bool = True,
        residual_connections: bool = True,
        dwconv2d_spiking: bool = True,
        patch_embedding_spiking: bool = False,
        output_spiking: bool = True,
        temporal_mamba=None,
        spatial_output_spiking: bool = False,
        downsample_kernel_size=None,
        downsample=True,
        downsampling_spiking=True,
        dwconv2d_kernel_size=3,
        neuron_model="plif",
        silif=None,
    ) -> None:
        super().__init__()
        if not isinstance(output_spiking, bool):
            raise TypeError("output_spiking must be a boolean")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.channels = self.out_channels
        self.patch_size = int(patch_size)
        self.output_spiking = output_spiking
        self.max_time_steps = max_time_steps
        self.patch_embed = Spiking2DPatchEmbedding(
            in_channels=self.in_channels,
            embed_dim=self.out_channels,
            image_size=(self.patch_size, self.patch_size),
            patch_size=self.patch_size,
            max_time_steps=max_time_steps,
            kernel_size=downsample_kernel_size,
        ) if downsample else None
        self.downsample_plif = (
            (make_spiking_node(neuron_model, channels=self.out_channels, init_tau=init_tau,
                               surrogate_function=surrogate.ATan(), detach_reset=True, silif=silif)
             if downsampling_spiking else nn.GELU())
            if downsample and downsample_kernel_size is not None else None
        )
        # Suppressing duplicate spikes is distinct from disabling a neuron.
        skip_patch_activation = (patch_embedding_spiking and
                                 isinstance(self.downsample_plif, SPIKING_NODE_TYPES) and
                                 not linear_projection)
        self.spik_mamba = SpikMambaBlock(
            dim=self.out_channels,
            d_state=d_state,
            dt_rank=dt_rank,
            init_tau=init_tau,
            selective_scan=selective_scan,
            max_time_steps=max_time_steps,
            linear_projection=linear_projection,
            patch_embedding_spiking=patch_embedding_spiking and not skip_patch_activation,
            spatial_output_spiking=spatial_output_spiking,
            residual_connections=residual_connections,
            dwconv2d_spiking=dwconv2d_spiking,
            dwconv2d_kernel_size=dwconv2d_kernel_size,
            neuron_model=neuron_model, silif=silif,
            temporal_mamba=temporal_mamba,
        )
        if skip_patch_activation:
            self.spik_mamba.mamba_layer.patch_sl = nn.Identity()
        self.post_norm = nn.GroupNorm(1, self.out_channels)
        self.post_plif = (
            make_spiking_node(
                neuron_model, channels=self.out_channels,
                init_tau=init_tau,
                surrogate_function=surrogate.ATan(),
                detach_reset=True, silif=silif,
            )
            if output_spiking else nn.GELU()
        )

    def forward(self, x: torch.Tensor, time_step: int, *, return_pre_spike=False):
        if x.ndim != 4:
            raise ValueError(f"expected BCHW input, got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels, got {x.shape[1]}"
            )

        if self.patch_embed is not None:
            x = self.patch_embed(x, time_step)
        if isinstance(self.downsample_plif, SPIKING_NODE_TYPES):
            x, _ = self.downsample_plif(x, time_step)
        elif self.downsample_plif is not None:
            x = self.downsample_plif(x)
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        tokens = self.spik_mamba(
            tokens,
            time_step,
            spatial_shape=(height, width),
        )
        x = tokens.transpose(1, 2).contiguous().view(
            batch, channels, height, width
        )

        pre_spike = self.post_norm(x)
        x = self.post_plif(pre_spike, time_step)[0] if self.output_spiking else self.post_plif(pre_spike)
        return (x, pre_spike) if return_pre_spike else x

    def forward_sequence(self, x: torch.Tensor, time_step: int, *, return_pre_spike=False):
        """Encode a complete [B,T,C,H,W] window for temporal Mamba."""
        if x.ndim != 5 or x.shape[2] != self.in_channels:
            raise ValueError(f"expected [B,T,{self.in_channels},H,W], got {tuple(x.shape)}")
        batch, steps, channels, height, width = x.shape
        if steps <= 0:
            raise ValueError("sequence window must contain at least one slice")
        if time_step < 0 or time_step + steps > self.max_time_steps:
            raise ValueError("sequence time range is outside the configured range")
        # [B,T,C,H,W] -> [B*T,C,H,W] for per-slice patch embedding.
        flat = x.reshape(batch * steps, channels, height, width)
        embedded = self.patch_embed(flat, time_step) if self.patch_embed is not None else flat
        if isinstance(self.downsample_plif, SPIKING_NODE_TYPES):
            sequence = embedded.reshape(batch, steps, *embedded.shape[1:])
            embedded = torch.stack([self.downsample_plif(sequence[:, t], time_step + t)[0]
                                    for t in range(steps)], dim=1).reshape_as(embedded)
        elif self.downsample_plif is not None:
            embedded = self.downsample_plif(embedded)
        if self.spik_mamba.debug_shapes:
            print(f"after patch embedding [B*T,D,Hp,Wp]: {tuple(embedded.shape)}")
        _, embedded_channels, grid_h, grid_w = embedded.shape
        # [B*T,C,Hp,Wp] -> [B,T,N,D], with N=Hp*Wp.
        tokens = embedded.flatten(2).transpose(1, 2).contiguous()
        tokens = tokens.reshape(batch, steps, grid_h * grid_w, embedded_channels)
        tokens = self.spik_mamba.forward_sequence(
            tokens, time_step, spatial_shape=(grid_h, grid_w)
        )
        # [B,T,N,D] -> [B*T,D,Hp,Wp] for existing output normalization.
        features = tokens.reshape(batch * steps, grid_h * grid_w, embedded_channels)
        features = features.transpose(1, 2).contiguous().reshape(
            batch * steps, embedded_channels, grid_h, grid_w
        )
        features = self.post_norm(features)
        pre_spike = features.reshape(batch, steps, embedded_channels, grid_h, grid_w)
        if self.output_spiking:
            # Keep each MRI slice's PLIF update ordered and state-independent by sample.
            # [B*T,D,Hp,Wp] -> [B,T,D,Hp,Wp] before ordered PLIF updates.
            sequence = features.reshape(batch, steps, embedded_channels, grid_h, grid_w)
            outputs = []
            for t in range(steps):
                value, _ = self.post_plif(sequence[:, t], time_step + t)
                outputs.append(value)
            features = torch.stack(outputs, dim=1).reshape(
                batch * steps, embedded_channels, grid_h, grid_w
            )
        else:
            features = self.post_plif(features)
        # [B*T,D,Hp,Wp] -> [B,T,D,Hp,Wp].
        out = features.reshape(batch, steps, embedded_channels, grid_h, grid_w)
        return (out, pre_spike) if return_pre_spike else out


class FeatureUncertaintyEnhancement(nn.Module):
    """SegMamba FUE: one continuous spatial weight shared across channels."""

    def forward(self, x):
        logits = x.mean(dim=1, keepdim=True)
        uncertainty = -torch.sigmoid(logits) * F.logsigmoid(logits)
        return x * (2.0 - uncertainty)


class FUESkipConnection(nn.Module):
    """Reweight pre-PLIF encoder features with an independent optional PLIF."""

    def __init__(self, output_spiking=True, init_tau=2.0, channels=None, neuron_model="plif", silif=None):
        super().__init__()
        self.fue = FeatureUncertaintyEnhancement()
        self.plif = (
            make_spiking_node(neuron_model, channels=channels, init_tau=init_tau,
                              surrogate_function=surrogate.ATan(), detach_reset=True, silif=silif)
            if output_spiking else nn.GELU()
        )

    def forward(self, x, time_step):
        x = self.fue(x)
        return self.plif(x, time_step)[0] if isinstance(self.plif, SPIKING_NODE_TYPES) else self.plif(x)

    def forward_sequence(self, x, time_step):
        # Each MRI slice updates its own sample's membrane in temporal order.
        return torch.stack([self(x[:, t], time_step + t) for t in range(x.shape[1])], dim=1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=0,
                 dropout=0.3, init_tau=2.0,
                 normalization=True, spiking=True,
                 spikMamba=False, selective_scan=None,
                 ssm_d_state=16, ssm_dt_rank="auto",
                 patch_size=4, linear_projection=True,
                 residual_connections=True,
                 dwconv2d_spiking=True,
                 patch_embedding_spiking=False,
                 output_spiking=True,
                 temporal_mamba=None,
                 spatial_output_spiking=False,
                 downsample_kernel_size=None, vss_depth=1, downsampling_spiking=True,
                 fue_skip=False, dwconv2d_kernel_size=3, neuron_model="plif", silif=None,
                 activation=True):
        super().__init__()
        self.dropout = float(dropout)
        self.normalization = normalization
        self.spiking = spiking
        self.spikMamba = bool(spikMamba)
        if type(fue_skip) is not bool:
            raise ValueError("fue_skip must be a boolean")
        if fue_skip and not self.spikMamba:
            raise ValueError("FUE skips require a SpikMamba encoder block")
        self.fue_skip = FUESkipConnection(output_spiking, init_tau, channels=out_channels,
                                          neuron_model=neuron_model, silif=silif) if fue_skip else None

        if self.spikMamba:
            self.spik_mamba = _SpikMambaEncoderAdapter(
                in_channels=in_channels,
                out_channels=out_channels,
                patch_size=patch_size,
                d_state=ssm_d_state,
                dt_rank=ssm_dt_rank,
                init_tau=init_tau,
                selective_scan=selective_scan,
                linear_projection=linear_projection,
                residual_connections=residual_connections,
                dwconv2d_spiking=dwconv2d_spiking,
                patch_embedding_spiking=patch_embedding_spiking,
                spatial_output_spiking=spatial_output_spiking,
                output_spiking=output_spiking,
                temporal_mamba=temporal_mamba,
                downsample_kernel_size=downsample_kernel_size,
                downsampling_spiking=downsampling_spiking,
                dwconv2d_kernel_size=dwconv2d_kernel_size,
                neuron_model=neuron_model, silif=silif,
            )
            self.extra_vss = nn.ModuleList([
                _SpikMambaEncoderAdapter(
                    in_channels=out_channels, out_channels=out_channels,
                    d_state=ssm_d_state, dt_rank=ssm_dt_rank, init_tau=init_tau,
                    selective_scan=selective_scan, linear_projection=linear_projection,
                    residual_connections=residual_connections,
                    dwconv2d_spiking=dwconv2d_spiking,
                    dwconv2d_kernel_size=dwconv2d_kernel_size,
                    neuron_model=neuron_model, silif=silif,
                    patch_embedding_spiking=patch_embedding_spiking,
                    spatial_output_spiking=spatial_output_spiking,
                    output_spiking=output_spiking, temporal_mamba=None,
                    downsample=False,
                ) for _ in range(vss_depth - 1)
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )

        self.norm = (nn.Identity() if self.spikMamba and downsample_kernel_size is not None
                     else nn.GroupNorm(1, out_channels))
        self.spike_neurons = (
            make_spiking_node(
                neuron_model, channels=out_channels,
                init_tau=init_tau,
                surrogate_function=surrogate.ATan(),
                detach_reset=True, silif=silif,
            )
            if activation and spiking and output_spiking and not (self.spikMamba and downsample_kernel_size is not None)
            else (nn.GELU() if activation and not self.spikMamba else nn.Identity())
        )

    def _forward_encoder(self, x, time_step, *, sequence=False, return_skip=False):
        blocks = (self.spik_mamba, *self.extra_vss)
        pre_spike = None
        out = x
        for index, block in enumerate(blocks):
            forward = block.forward_sequence if sequence else block
            if return_skip and self.fue_skip is not None and index == len(blocks) - 1:
                out, pre_spike = forward(out, time_step, return_pre_spike=True)
            else:
                out = forward(out, time_step)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        if not return_skip:
            return out
        if self.fue_skip is None:
            # Preserve the original shared encoder/skip tensor when FUE is off.
            return out, out
        skip_forward = self.fue_skip.forward_sequence if sequence else self.fue_skip
        skip = skip_forward(pre_spike, time_step)
        if self.dropout > 0:
            skip = F.dropout(skip, p=self.dropout, training=self.training)
        return out, skip

    def forward_sequence(self, x: torch.Tensor, time_step: int, *, return_skip=False):
        if not self.spikMamba:
            raise ValueError("sequence forward is supported only for SpikMamba encoder blocks")
        return self._forward_encoder(x, time_step, sequence=True, return_skip=return_skip)

    def forward(self, x, time_step: int, *, return_skip=False):
        if self.spikMamba:
            return self._forward_encoder(x, time_step, return_skip=return_skip)
        if return_skip:
            raise ValueError("skip feature return is supported only for SpikMamba encoder blocks")

        out = self.conv(x)

        if self.normalization:
            out = self.norm(out)

        if isinstance(self.spike_neurons, SPIKING_NODE_TYPES):
            out, _ = self.spike_neurons(out, time_step)
        else:
            out = self.spike_neurons(out)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class ResDecoderBlock(nn.Module):
    """Fuse concatenated skips with a projected residual and two optional neurons.

    Conv3x3 -> GN -> spike -> Conv3x3, then add Conv1x1(input),
    followed by GN -> spike -> dropout. Disabled neurons use GELU,
    matching the existing decoder's output_spiking switch.
    """

    def __init__(self, in_channels, out_channels, dropout=0.1,
                 init_tau=2.0, output_spiking=True,
                 spiking1_enabled=True, spiking2_enabled=True, neuron_model="plif", silif=None):
        super().__init__()
        for name, enabled in (("spiking1_enabled", spiking1_enabled), ("spiking2_enabled", spiking2_enabled)):
            if type(enabled) is not bool:
                raise TypeError(f"{name} must be a boolean")
        if type(output_spiking) is not bool:
            raise TypeError("output_spiking must be a boolean")
        self.dropout = float(dropout)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.plif1 = (
            make_spiking_node(neuron_model, channels=out_channels, init_tau=init_tau,
                     surrogate_function=surrogate.ATan(),
                     detach_reset=True, silif=silif)
            if output_spiking and spiking1_enabled else nn.GELU()
        )
        self.plif2 = (
            make_spiking_node(neuron_model, channels=out_channels, init_tau=init_tau,
                     surrogate_function=surrogate.ATan(),
                     detach_reset=True, silif=silif)
            if output_spiking and spiking2_enabled else nn.GELU()
        )

    def forward(self, x, time_step: int):
        out = self.norm1(self.conv1(x))
        if isinstance(self.plif1, SPIKING_NODE_TYPES):
            out, _ = self.plif1(out, time_step)
        else:
            out = self.plif1(out)
        out = self.norm2(self.conv2(out) + self.shortcut(x))
        if isinstance(self.plif2, SPIKING_NODE_TYPES):
            out, _ = self.plif2(out, time_step)
        else:
            out = self.plif2(out)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2,
                 dropout=0.3, init_tau=2.0, output_spiking=True, neuron_model="plif", silif=None):
        super().__init__()
        self.dropout = float(dropout)

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=kernel_size, stride=stride, bias=False)
        # self.norm = nn.BatchNorm2d(out_channels)
        self.norm = nn.GroupNorm(1, out_channels)
        self.spike_neurons = (
            make_spiking_node(
                neuron_model, channels=out_channels,
                init_tau=init_tau,
                surrogate_function=surrogate.ATan(),
                detach_reset=True, silif=silif,
            )
            if output_spiking else nn.GELU()
        )

    def forward(self, x, time_step: int):
        out = self.deconv(x)
        out = self.norm(out)
        if isinstance(self.spike_neurons, SPIKING_NODE_TYPES):
            out, _ = self.spike_neurons(out, time_step)
        else:
            out = self.spike_neurons(out)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class SNNBraTS(nn.Module):
    """
    Forward takes a window x_win: (B, k, 4, H, W) and an absolute starting time t0.
    Enumerates time_step = t0, t0+1, ..., t0+k-1 (no resets inside a sequence).
    Returns logits: (B, out_channels, k, H, W).
    """
    def __init__(self,
                 out_channels: int = 4,
                 selective_scan=None,
                 ssm_d_state: int = 16,
                 ssm_dt_rank="auto",
                 patch_size: int = 4,
                 linear_projection: bool = True,
                 residual_connections: bool = True,
                 dwconv2d_spiking: bool = True,
                 patch_embedding_spiking: bool = False,
                 input_skip: bool = False,
                 output_spiking: bool = True,
                 temporal_mamba=None,
                 spatial_output_spiking: bool = False,
                 hierarchical_encoder: bool = True,
                 stem_depth: int = 1, stem_channels: int = 32,
                 stem_spiking: bool = True, downsampling_spiking: bool = True,
                 encoder_channels=(64, 128, 256),
                 downsample_kernel_size: int = 3,
                 vss_depths=(1, 1, 2),
                 full_resolution_skip: bool = True,
                 final_skip_fusion: str = "concat",
                 decoder_block: str = "standard",
                 resdecoder_spiking1: bool = True, resdecoder_spiking2: bool = True,
                 fue_skip1: bool = False, fue_skip2: bool = False,
                 dwconv2d_kernel_size: int = 3,
                 downsample_strides=(2, 2, 2), neuron_model="plif", silif=None):

        super().__init__()
        if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        if not isinstance(linear_projection, bool):
            raise TypeError("linear_projection must be a boolean")
        if not isinstance(residual_connections, bool):
            raise TypeError("residual_connections must be a boolean")
        if not isinstance(dwconv2d_spiking, bool):
            raise TypeError("dwconv2d_spiking must be a boolean")
        if not isinstance(patch_embedding_spiking, bool):
            raise TypeError("patch_embedding_spiking must be a boolean")
        if not isinstance(output_spiking, bool):
            raise TypeError("output_spiking must be a boolean")
        if not isinstance(spatial_output_spiking, bool):
            raise TypeError("spatial_output_spiking must be a boolean")
        if not isinstance(input_skip, bool):
            raise TypeError("input_skip must be a boolean")
        self.input_skip = input_skip
        self.patch_size = patch_size
        self.linear_projection = linear_projection
        self.residual_connections = residual_connections
        self.dwconv2d_spiking = dwconv2d_spiking
        self.patch_embedding_spiking = patch_embedding_spiking
        self.output_spiking = output_spiking
        self.spatial_output_spiking = spatial_output_spiking
        if temporal_mamba is not None and not isinstance(temporal_mamba, dict):
            raise TypeError("temporal_mamba must be a mapping")
        self.temporal_mamba_config = dict(temporal_mamba or {"enabled": False})
        temporal_enabled = self.temporal_mamba_config.get("enabled", False)
        if not isinstance(temporal_enabled, bool):
            raise TypeError("temporal_mamba.enabled must be a boolean")
        self.temporal_mamba_enabled = temporal_enabled
        silif = silif_options({} if silif is None else silif)
        self.architecture_config = architecture_options({
            "neuron_model": neuron_model,
            "silif": silif,
            "hierarchical_encoder": hierarchical_encoder, "stem_depth": stem_depth,
            "stem_channels": stem_channels, "encoder_channels": encoder_channels,
            "stem_spiking": stem_spiking, "downsampling_spiking": downsampling_spiking,
            "downsample_kernel_size": downsample_kernel_size, "vss_depths": vss_depths,
            "dwconv2d_kernel_size": dwconv2d_kernel_size,
            "downsample_strides": downsample_strides,
            "full_resolution_skip": full_resolution_skip, "final_skip_fusion": final_skip_fusion,
            "decoder_block": decoder_block,
            "resdecoder_spiking1": resdecoder_spiking1, "resdecoder_spiking2": resdecoder_spiking2,
            "fue_skip1": fue_skip1, "fue_skip2": fue_skip2,
        })
        self.hierarchical_encoder = hierarchical_encoder
        self.neuron_model = self.architecture_config["neuron_model"]
        self.dwconv2d_kernel_size = self.architecture_config["dwconv2d_kernel_size"]
        self.downsample_strides = tuple(self.architecture_config["downsample_strides"])
        self.stem_channels = stem_channels
        self.stem_depth = stem_depth
        self.stem_spiking = stem_spiking
        self.downsampling_spiking = downsampling_spiking
        self.encoder_channels = tuple(encoder_channels) if hierarchical_encoder else (32, 64, 128)
        self.vss_depths = tuple(vss_depths) if hierarchical_encoder else (1, 1, 1)
        self.decoder_block = self.architecture_config["decoder_block"]
        self.resdecoder_spiking1 = self.architecture_config["resdecoder_spiking1"]
        self.resdecoder_spiking2 = self.architecture_config["resdecoder_spiking2"]
        self.fue_skip1 = self.architecture_config["fue_skip1"]
        self.fue_skip2 = self.architecture_config["fue_skip2"]
        self.full_resolution_skip = hierarchical_encoder and full_resolution_skip
        stage_strides = self.downsample_strides if hierarchical_encoder else (self.patch_size,) * 3
        self.encoder_scale = math.prod(stage_strides)
        if hierarchical_encoder:
            self.stem = SpikingStem(4, stem_channels, stem_depth, stem_spiking=stem_spiking,
                                    neuron_model=neuron_model, silif=silif)
        # Encoder
        spik_mamba_kwargs = {
            "neuron_model": neuron_model,
            "silif": silif,
            "spikMamba": True,
            "selective_scan": selective_scan,
            "ssm_d_state": ssm_d_state,
            "ssm_dt_rank": ssm_dt_rank,
            "downsample_kernel_size": downsample_kernel_size if hierarchical_encoder else None,
            "downsampling_spiking": downsampling_spiking,
            "dwconv2d_kernel_size": self.dwconv2d_kernel_size,
            "linear_projection": self.linear_projection,
            "residual_connections": self.residual_connections,
            "dwconv2d_spiking": self.dwconv2d_spiking,
            "patch_embedding_spiking": self.patch_embedding_spiking,
            "spatial_output_spiking": self.spatial_output_spiking,
            "output_spiking": output_spiking,
            "temporal_mamba": self.temporal_mamba_config,
        }
        c1, c2, c3 = self.encoder_channels
        self.conv_block1 = ConvBlock(
            stem_channels if hierarchical_encoder else 4, c1,
            patch_size=stage_strides[0], padding=1, dropout=0.1, vss_depth=self.vss_depths[0],
            fue_skip=self.fue_skip1, **spik_mamba_kwargs)
        self.conv_block2 = ConvBlock(
            c1, c2, patch_size=stage_strides[1], padding=1, dropout=0.1, vss_depth=self.vss_depths[1],
            fue_skip=self.fue_skip2, **spik_mamba_kwargs)
        self.conv_block3 = ConvBlock(
            c2, c3, patch_size=stage_strides[2], padding=1, dropout=0.1,
            vss_depth=self.vss_depths[2], **spik_mamba_kwargs)

        # Mirror encoder strides; stride 1 projects channels at the same resolution.
        d1, d2, d3 = (c2, c1, stem_channels) if hierarchical_encoder else (128, 128, 128)
        stride1, stride2, stride3 = reversed(stage_strides)
        self.deconv_block1 = DeconvBlock(c3, d1, stride1, stride1, dropout=0.1, output_spiking=output_spiking, neuron_model=neuron_model, silif=silif)
        self.deconv1_conv = ConvBlock(d1, d1, padding=1, dropout=0.1, output_spiking=output_spiking, neuron_model=neuron_model, silif=silif)
        self.concat1_conv = (
            ResDecoderBlock(d1 + c2, d1, dropout=0.1, output_spiking=output_spiking, neuron_model=neuron_model, silif=silif,
                            spiking1_enabled=self.resdecoder_spiking1,
                            spiking2_enabled=self.resdecoder_spiking2)
            if self.decoder_block == "resdecoder" else
            ConvBlock(d1 + c2, d1, padding=1, dropout=0.1, output_spiking=output_spiking, neuron_model=neuron_model, silif=silif)
        )
        self.deconv_block2 = DeconvBlock(d1, d2, stride2, stride2, dropout=0.1, output_spiking=output_spiking, neuron_model=neuron_model, silif=silif)
        self.deconv2_conv = ConvBlock(d2, d2, padding=1, dropout=0.1, output_spiking=output_spiking, neuron_model=neuron_model, silif=silif)
        self.concat2_conv = (
            ResDecoderBlock(d2 + c1, d2, dropout=0.1, output_spiking=output_spiking, neuron_model=neuron_model, silif=silif,
                            spiking1_enabled=self.resdecoder_spiking1,
                            spiking2_enabled=self.resdecoder_spiking2)
            if self.decoder_block == "resdecoder" else
            ConvBlock(d2 + c1, d2, padding=1, dropout=0.1, output_spiking=output_spiking, neuron_model=neuron_model, silif=silif)
        )
        self.deconv_block3 = DeconvBlock(d2, d3, stride3, stride3, dropout=0.1, output_spiking=output_spiking, neuron_model=neuron_model, silif=silif)
        self.deconv3_conv = ConvBlock(d3, d3, padding=1, dropout=0.1, output_spiking=output_spiking, neuron_model=neuron_model, silif=silif)
        if self.full_resolution_skip:
            self.concat0_conv = ConvBlock(d3 + stem_channels, d3, padding=1,
                                          dropout=0.1, output_spiking=output_spiking, neuron_model=neuron_model, silif=silif)
        self.class_conv = ConvBlock(d3, out_channels, padding=1, dropout=0.0,
                                     normalization=False, spiking=False, activation=False)
        if self.input_skip:
            self.input_proj = nn.Conv2d(4, 8, kernel_size=1, bias=False)
            self.input_fuse = nn.Conv2d(d3 + 8, d3, kernel_size=3, padding=1)
        if neuron_model == "silif":
            active_nodes = sum(isinstance(module, SiLIFNode) for module in self.modules())
            lines = ["SiLIF configuration (initialization; learned values evolve during training):"]
            lines.extend(f"  {key}={value}" for key, value in silif.items())
            lines.append(f"  active_neuron_modules={active_nodes}")
            lines.append("  reset_amplitude=1.0 (independent of v_threshold)")
            print("\n".join(lines), flush=True)


    def _decode(self, x, skip1, skip2, time_step, skip0=None, input_features=None):
        x = self.deconv1_conv(self.deconv_block1(x, time_step), time_step)
        x = self.concat1_conv(torch.cat([skip2, x], dim=1), time_step)
        x = self.deconv2_conv(self.deconv_block2(x, time_step), time_step)
        x = self.concat2_conv(torch.cat([skip1, x], dim=1), time_step)
        x = self.deconv3_conv(self.deconv_block3(x, time_step), time_step)
        if self.full_resolution_skip:
            x = self.concat0_conv(torch.cat([x, skip0], dim=1), time_step)
        if self.input_skip:
            x = self.input_fuse(torch.cat([x, input_features], dim=1))
        return self.class_conv(x, time_step)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if any(key.startswith(prefix + "conv_block1.") for key in state_dict):
            saved_hierarchy = any(key.startswith(prefix + "stem.") for key in state_dict)
            if saved_hierarchy != self.hierarchical_encoder:
                error_msgs.append(
                    "Incompatible architecture checkpoint: set hierarchical_encoder="
                    f"{saved_hierarchy} to reconstruct this model; use a new run for the refactored architecture."
                )
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, x_win: torch.Tensor, t0: int = 0) -> torch.Tensor:
        B, k, C, H, W = x_win.shape
        pad_h = (-H) % self.encoder_scale
        pad_w = (-W) % self.encoder_scale
        logits = []
        if self.temporal_mamba_enabled:
            x_volume = x_win.reshape(B * k, C, H, W)
            if pad_h or pad_w:
                x_volume = F.pad(x_volume, (0, pad_w, 0, pad_h))
            Hp, Wp = x_volume.shape[-2:]
            input_features = self.input_proj(x_volume).reshape(B, k, 8, Hp, Wp) if self.input_skip else None
            x_volume = x_volume.reshape(B, k, C, Hp, Wp)
            skip0 = self.stem.forward_sequence(x_volume, t0) if self.hierarchical_encoder else None
            enc1, skip1 = self.conv_block1.forward_sequence(
                skip0 if skip0 is not None else x_volume, t0, return_skip=True)
            enc2, skip2 = self.conv_block2.forward_sequence(enc1, t0, return_skip=True)
            encoded = self.conv_block3.forward_sequence(enc2, t0)
            for i in range(k):
                x = self._decode(encoded[:, i], skip1[:, i], skip2[:, i], t0 + i,
                                 skip0[:, i] if skip0 is not None else None,
                                 input_features[:, i] if input_features is not None else None)
                logits.append(x[..., :H, :W])
        else:
            for i in range(k):
                time_step = t0 + i
                x = x_win[:, i]
                if pad_h or pad_w:
                    x = F.pad(x, (0, pad_w, 0, pad_h))
                input_features = self.input_proj(x) if self.input_skip else None
                skip0 = self.stem(x, time_step) if self.hierarchical_encoder else None
                enc1, skip1 = self.conv_block1(
                    skip0 if skip0 is not None else x, time_step, return_skip=True)
                enc2, skip2 = self.conv_block2(enc1, time_step, return_skip=True)
                x = self.conv_block3(enc2, time_step)
                x = self._decode(x, skip1, skip2, time_step, skip0, input_features)
                logits.append(x[..., :H, :W])
        return torch.stack(logits, dim=2)

    # TBPTT helper: detach neuron states at window boundary to truncate gradients
    def detach_states(self):
        for m in self.modules():
            # Adjust if your PLIFNode has a different method name
            if hasattr(m, "detach") and callable(m.detach):
                m.detach()


# ───────── helpers (shared) ─────────
def _pad_needed(H: int, W: int, multiple: int):
    return (multiple - (H % multiple)) % multiple, (multiple - (W % multiple)) % multiple

def _pad_rb(x: torch.Tensor, ph: int, pw: int):
    return x if (ph | pw) == 0 else F.pad(x, (0, pw, 0, ph))  # (left,right,top,bottom)

# =========================
# Deep U-Net — 4 pools (×16)
# =========================
class SNNBraTSUNetDeep(nn.Module):
    """
    L1: 32→32 | pool
    L2: 64→64 | pool
    L3: 128→128 | pool
    L4: 256→256 | pool
    Bottom: 512→512
    Up: 512→256 →128 →64 →32 (skip x4,x3,x2,x1), head 1×1
    """
    def __init__(self, out_channels: int = 4, dropout: float = 0.1):
        super().__init__()
        self.out_channels, self.multiple = int(out_channels), 16
        # encoder
        self.enc1a = ConvBlock(4,   32, padding=1, dropout=dropout); self.enc1b = ConvBlock(32,  32, padding=1, dropout=dropout)
        self.enc2a = ConvBlock(32,  64, padding=1, dropout=dropout); self.enc2b = ConvBlock(64,  64, padding=1, dropout=dropout)
        self.enc3a = ConvBlock(64, 128, padding=1, dropout=dropout); self.enc3b = ConvBlock(128,128, padding=1, dropout=dropout)
        self.enc4a = ConvBlock(128,256, padding=1, dropout=dropout); self.enc4b = ConvBlock(256,256, padding=1, dropout=dropout)
        self.pool = nn.MaxPool2d(2,2)
        # bottom
        self.bot_a = ConvBlock(256,512, padding=1, dropout=dropout); self.bot_b = ConvBlock(512,512, padding=1, dropout=dropout)
        # decoder
        self.up4   = DeconvBlock(512,256,dropout=dropout); self.dec4a = ConvBlock(256+256,256,padding=1,dropout=dropout); self.dec4b = ConvBlock(256,256,padding=1,dropout=dropout)
        self.up3   = DeconvBlock(256,128,dropout=dropout); self.dec3a = ConvBlock(128+128,128,padding=1,dropout=dropout); self.dec3b = ConvBlock(128,128,padding=1,dropout=dropout)
        self.up2   = DeconvBlock(128, 64,dropout=dropout); self.dec2a = ConvBlock( 64+ 64, 64,padding=1,dropout=dropout); self.dec2b = ConvBlock( 64, 64,padding=1,dropout=dropout)
        self.up1   = DeconvBlock( 64, 32,dropout=dropout); self.dec1a = ConvBlock( 32+ 32, 32,padding=1,dropout=dropout); self.dec1b = ConvBlock( 32, 32,padding=1,dropout=dropout)
        self.class_conv = ConvBlock(32, self.out_channels, kernel_size=1, padding=0, dropout=0.0, normalization=False, spiking=False, activation=False)

    def forward(self, x_win: torch.Tensor, t0: int = 0) -> torch.Tensor:
        B,k,C,H,W = x_win.shape
        ph,pw = _pad_needed(H,W,self.multiple)
        logits=[]
        for i in range(k):
            t=t0+i; x = x_win[:,i];  x = _pad_rb(x,ph,pw)
            x1=self.enc1a(x,t); x1=self.enc1b(x1,t); p1=self.pool(x1)
            x2=self.enc2a(p1,t); x2=self.enc2b(x2,t); p2=self.pool(x2)
            x3=self.enc3a(p2,t); x3=self.enc3b(x3,t); p3=self.pool(x3)
            x4=self.enc4a(p3,t); x4=self.enc4b(x4,t); p4=self.pool(x4)
            xb=self.bot_a(p4,t); xb=self.bot_b(xb,t)
            y4=self.up4(xb,t); y4=torch.cat([x4,y4],1); y4=self.dec4a(y4,t); y4=self.dec4b(y4,t)
            y3=self.up3(y4,t); y3=torch.cat([x3,y3],1); y3=self.dec3a(y3,t); y3=self.dec3b(y3,t)
            y2=self.up2(y3,t); y2=torch.cat([x2,y2],1); y2=self.dec2a(y2,t); y2=self.dec2b(y2,t)
            y1=self.up1(y2,t); y1=torch.cat([x1,y1],1); y1=self.dec1a(y1,t); y1=self.dec1b(y1,t)
            out=self.class_conv(y1,t); out=out[...,:H,:W]; logits.append(out)
        return torch.stack(logits,2)

    def detach_states(self): 
        for m in self.modules():
            if hasattr(m,"detach") and callable(m.detach): m.detach()


# =========================
# 4L U-Net — 3 pools (×8)
# =========================
class SNNBraTSUNetMedium(nn.Module):
    """
    L1: 32→32 | pool
    L2: 64→64 | pool
    L3: 128→128 | pool
    L4: 256→256 (NO pool)  ← 4 layers, 3 pools total
    Bottom: 512→512
    Up: 512→128 →64 →32 (skips x3,x2,x1). L4 acts as extra capacity at H/8.
    """
    def __init__(self, out_channels: int = 4, dropout: float = 0.1):
        super().__init__()
        self.out_channels, self.multiple = int(out_channels), 8
        # encoder
        self.enc1a = ConvBlock(4,   32, padding=1, dropout=dropout); self.enc1b = ConvBlock(32,  32, padding=1, dropout=dropout)
        self.enc2a = ConvBlock(32,  64, padding=1, dropout=dropout); self.enc2b = ConvBlock(64,  64, padding=1, dropout=dropout)
        self.enc3a = ConvBlock(64, 128, padding=1, dropout=dropout); self.enc3b = ConvBlock(128,128, padding=1, dropout=dropout)
        self.enc4a = ConvBlock(128,256, padding=1, dropout=dropout); self.enc4b = ConvBlock(256,256, padding=1, dropout=dropout)
        self.pool = nn.MaxPool2d(2,2)
        # bottom (no extra pool; operates at H/8)
        self.bot_a = ConvBlock(256,512, padding=1, dropout=dropout); self.bot_b = ConvBlock(512,512, padding=1, dropout=dropout)
        # decoder (3 ups, skips x3,x2,x1)
        self.up3   = DeconvBlock(512,128,dropout=dropout); self.dec3a = ConvBlock(128+128,128,padding=1,dropout=dropout); self.dec3b = ConvBlock(128,128,padding=1,dropout=dropout)
        self.up2   = DeconvBlock(128, 64,dropout=dropout); self.dec2a = ConvBlock( 64+ 64, 64,padding=1,dropout=dropout); self.dec2b = ConvBlock( 64, 64,padding=1,dropout=dropout)
        self.up1   = DeconvBlock( 64, 32,dropout=dropout); self.dec1a = ConvBlock( 32+ 32, 32,padding=1,dropout=dropout); self.dec1b = ConvBlock( 32, 32,padding=1,dropout=dropout)
        self.class_conv = ConvBlock(32, self.out_channels, kernel_size=1, padding=0, dropout=0.0, normalization=False, spiking=False, activation=False)

    def forward(self, x_win: torch.Tensor, t0: int = 0) -> torch.Tensor:
        B,k,C,H,W = x_win.shape
        ph,pw = _pad_needed(H,W,self.multiple)
        logits=[]
        for i in range(k):
            t=t0+i; x=x_win[:,i]; x=_pad_rb(x,ph,pw)
            x1=self.enc1a(x,t); x1=self.enc1b(x1,t); p1=self.pool(x1)
            x2=self.enc2a(p1,t); x2=self.enc2b(x2,t); p2=self.pool(x2)
            x3=self.enc3a(p2,t); x3=self.enc3b(x3,t); p3=self.pool(x3)  # H/8
            x4=self.enc4a(p3,t); x4=self.enc4b(x4,t)                     # H/8 (no pool)
            xb=self.bot_a(x4,t); xb=self.bot_b(xb,t)                      # H/8 bottom
            y3=self.up3(xb,t); y3=torch.cat([x3,y3],1); y3=self.dec3a(y3,t); y3=self.dec3b(y3,t)
            y2=self.up2(y3,t); y2=torch.cat([x2,y2],1); y2=self.dec2a(y2,t); y2=self.dec2b(y2,t)
            y1=self.up1(y2,t); y1=torch.cat([x1,y1],1); y1=self.dec1a(y1,t); y1=self.dec1b(y1,t)
            out=self.class_conv(y1,t); out=out[...,:H,:W]; logits.append(out)
        return torch.stack(logits,2)

    def detach_states(self):
        for m in self.modules():
            if hasattr(m,"detach") and callable(m.detach): m.detach()


# =========================
# 3L U-Net — 2 pools (×4)
# =========================
class SNNBraTSUNetShallow(nn.Module):
    """
    L1: 32→32 | pool
    L2: 64→64 | pool
    L3: 128→128 (NO pool)  ← 3 layers, 2 pools total
    Bottom: 256→512
    Up: 512→64 →32 (skips x2,x1), head 1×1
    """
    def __init__(self, out_channels: int = 4, dropout: float = 0.1):
        super().__init__()
        self.out_channels, self.multiple = int(out_channels), 4
        # encoder
        self.enc1a = ConvBlock(4,   32, padding=1, dropout=dropout); self.enc1b = ConvBlock(32,  32, padding=1, dropout=dropout)
        self.enc2a = ConvBlock(32,  64, padding=1, dropout=dropout); self.enc2b = ConvBlock(64,  64, padding=1, dropout=dropout)
        self.enc3a = ConvBlock(64, 128, padding=1, dropout=dropout); self.enc3b = ConvBlock(128,128, padding=1, dropout=dropout)
        self.pool = nn.MaxPool2d(2,2)
        # bottom (works at H/4)
        self.bot_a = ConvBlock(128,256, padding=1, dropout=dropout); self.bot_b = ConvBlock(256,512, padding=1, dropout=dropout)
        # decoder (2 ups, skips x2,x1)
        self.up2   = DeconvBlock(512, 64,dropout=dropout); self.dec2a = ConvBlock( 64+ 64, 64,padding=1,dropout=dropout); self.dec2b = ConvBlock( 64, 64,padding=1,dropout=dropout)
        self.up1   = DeconvBlock( 64, 32,dropout=dropout); self.dec1a = ConvBlock( 32+ 32, 32,padding=1,dropout=dropout); self.dec1b = ConvBlock( 32, 32,padding=1,dropout=dropout)
        self.class_conv = ConvBlock(32, self.out_channels, kernel_size=1, padding=0, dropout=0.0, normalization=False, spiking=False, activation=False)

    def forward(self, x_win: torch.Tensor, t0: int = 0) -> torch.Tensor:
        B,k,C,H,W = x_win.shape
        ph,pw = _pad_needed(H,W,self.multiple)
        logits=[]
        for i in range(k):
            t=t0+i; x=x_win[:,i]; x=_pad_rb(x,ph,pw)
            x1=self.enc1a(x,t); x1=self.enc1b(x1,t); p1=self.pool(x1)   # H/2
            x2=self.enc2a(p1,t); x2=self.enc2b(x2,t); p2=self.pool(x2)   # H/4
            x3=self.enc3a(p2,t); x3=self.enc3b(x3,t)                     # H/4 (no pool)
            xb=self.bot_a(x3,t); xb=self.bot_b(xb,t)                      # H/4 bottom
            y2=self.up2(xb,t); y2=torch.cat([x2,y2],1); y2=self.dec2a(y2,t); y2=self.dec2b(y2,t)
            y1=self.up1(y2,t); y1=torch.cat([x1,y1],1); y1=self.dec1a(y1,t); y1=self.dec1b(y1,t)
            out=self.class_conv(y1,t); out=out[...,:H,:W]; logits.append(out)
        return torch.stack(logits,2)

    def detach_states(self):
        for m in self.modules():
            if hasattr(m,"detach") and callable(m.detach): m.detach()


def build_model(model_name: str, out_channels: int = 3, patch_size: int = 4,
                linear_projection: bool = True,
                residual_connections: bool = True,
                dwconv2d_spiking: bool = True,
                patch_embedding_spiking: bool = False,
                input_skip: bool = False,
                output_spiking: bool = True,
                temporal_mamba=None,
                spatial_output_spiking: bool = False,
                **architecture) -> nn.Module:
    architecture = normalize_spiking_options(architecture)
    unknown = architecture.keys() - ARCHITECTURE_DEFAULTS.keys()
    if unknown:
        raise TypeError(f"Unknown architecture options: {sorted(unknown)}")
    if architecture.get("neuron_model", "plif") != "plif" and model_name != "orig":
        raise ValueError("neuron_model=silif is supported only for model: orig")
    if model_name != "orig" and architecture_options(architecture)["downsample_strides"] != [2, 2, 2]:
        raise ValueError("Custom downsample_strides is supported only for model: orig")
    if architecture.get("dwconv2d_kernel_size", 3) != 3 and model_name != "orig":
        raise ValueError("dwconv2d_kernel_size is supported only for model: orig")
    if architecture.get("decoder_block", "standard") != "standard" and model_name != "orig":
        raise ValueError("decoder_block=resdecoder is supported only for model: orig")
    if any(architecture.get(key, False) for key in ("fue_skip1", "fue_skip2")) and model_name != "orig":
        raise ValueError("FUE skips are supported only for model: orig")
    if input_skip and model_name != "orig":
        raise ValueError("input_skip=true is supported only for model: orig")
    if temporal_mamba and temporal_mamba.get("enabled", False) and model_name != "orig":
        raise ValueError("temporal_mamba.enabled=true is supported only for model: orig")
    if model_name == "orig":
        return SNNBraTS(
            out_channels=out_channels,
            patch_size=patch_size,
            linear_projection=linear_projection,
            residual_connections=residual_connections,
            dwconv2d_spiking=dwconv2d_spiking,
            patch_embedding_spiking=patch_embedding_spiking,
            input_skip=input_skip,
            output_spiking=output_spiking,
            temporal_mamba=temporal_mamba,
            spatial_output_spiking=spatial_output_spiking,
            **architecture,
        )
    if model_name == "shallow":
        return SNNBraTSUNetShallow(out_channels=out_channels)
    if model_name == "medium":
        return SNNBraTSUNetMedium(out_channels=out_channels)
    if model_name == "deep":
        return SNNBraTSUNetDeep(out_channels=out_channels)
    raise ValueError(f"Unknown model: {model_name}")
