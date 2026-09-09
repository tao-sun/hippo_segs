from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import surrogate
from spike_neurons import PLIFNode
from spikmamba import Spiking2DPatchEmbedding, SpikMambaBlock


def print_model_info(model: nn.Module):
    def readable(number):
        if number >= 1e6:
            return f"{number / 1e6:.2f} M"
        if number >= 1e3:
            return f"{number / 1e3:.2f} K"
        return str(number)

    print("=" * 60)
    print(f"{'Layer (type)':35s} {'Param #':>12s}")
    print("=" * 60)
    total = 0
    for name, module in model.named_modules():
        if list(module.children()) and not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            continue
        parameters = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if parameters:
            print(f"{name:35s} {readable(parameters):>12s}")
            total += parameters
    print("=" * 60)
    print(f"{'Total trainable params:':35s} {readable(total):>12s}")
    print("=" * 60)


class _SpikMambaEncoderAdapter(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, patch_size: int = 4,
                 d_state: int = 16, dt_rank: Union[int, str] = "auto",
                 init_tau: float = 2.0, selective_scan=None,
                 max_time_steps: int = 256, linear_projection: bool = True,
                 residual_connections: bool = True, dwconv2d_spiking: bool = True,
                 patch_embedding_spiking: bool = False):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.patch_embed = Spiking2DPatchEmbedding(
            in_channels=self.in_channels, embed_dim=self.out_channels,
            image_size=(patch_size, patch_size), patch_size=patch_size,
            max_time_steps=max_time_steps,
        )
        self.spik_mamba = SpikMambaBlock(
            dim=self.out_channels, d_state=d_state, dt_rank=dt_rank,
            init_tau=init_tau, selective_scan=selective_scan,
            max_time_steps=max_time_steps, linear_projection=linear_projection,
            patch_embedding_spiking=patch_embedding_spiking,
            residual_connections=residual_connections,
            dwconv2d_spiking=dwconv2d_spiking,
        )
        self.post_norm = nn.GroupNorm(1, self.out_channels)
        self.post_plif = PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=True)

    def forward(self, x, time_step: int):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected BCHW input with {self.in_channels} channels, got {tuple(x.shape)}")
        x = self.patch_embed(x, time_step)
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        tokens = self.spik_mamba(tokens, time_step, spatial_shape=(height, width))
        x = tokens.transpose(1, 2).contiguous().view(batch, channels, height, width)
        x = self.post_norm(x)
        x, _ = self.post_plif(x, time_step)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dropout=0.3, init_tau=2.0, normalization=True, spiking=True,
                 spikMamba=False, selective_scan=None, ssm_d_state=16,
                 ssm_dt_rank="auto", patch_size=4, linear_projection=True,
                 residual_connections=True, dwconv2d_spiking=True,
                 patch_embedding_spiking=False):
        super().__init__()
        self.dropout = float(dropout)
        self.normalization = normalization
        self.spikMamba = bool(spikMamba)
        self.spiking = spiking
        if self.spikMamba:
            self.spik_mamba = _SpikMambaEncoderAdapter(
                in_channels, out_channels, patch_size, ssm_d_state, ssm_dt_rank,
                init_tau, selective_scan, 256, linear_projection,
                residual_connections, dwconv2d_spiking, patch_embedding_spiking,
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.GroupNorm(1, out_channels)
        self.spike_neurons = PLIFNode(
            init_tau=init_tau, surrogate_function=surrogate.ATan(),
            detach_reset=True, no_spiking=not spiking,
        )

    def forward(self, x, time_step: int):
        if self.spikMamba:
            output = self.spik_mamba(x, time_step)
        else:
            output = self.conv(x)
            if self.normalization:
                output = self.norm(output)
            result = self.spike_neurons(output, time_step)
            output = result[0] if self.spiking else result
        return F.dropout(output, p=self.dropout, training=self.training) if self.dropout else output


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, dropout=0.3, init_tau=2.0):
        super().__init__()
        self.dropout = float(dropout)
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=False)
        self.norm = nn.GroupNorm(1, out_channels)
        self.spike_neurons = PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=True)

    def forward(self, x, time_step: int):
        output = self.norm(self.deconv(x))
        output, _ = self.spike_neurons(output, time_step)
        return F.dropout(output, p=self.dropout, training=self.training) if self.dropout else output


class _SNNBraTSVSS(nn.Module):
    def __init__(self, out_channels, stage_channels, stage_depths, patch_size,
                 linear_projection, residual_connections, dwconv2d_spiking,
                 patch_embedding_spiking):
        super().__init__()
        self.stage_channels = tuple(stage_channels)
        self.stage_depths = tuple(stage_depths)
        self.encoder_scale = patch_size ** len(self.stage_channels)
        if len(self.stage_channels) != len(self.stage_depths):
            raise ValueError("stage_channels and stage_depths must have the same length")
        if any(isinstance(depth, bool) or not isinstance(depth, int) or depth < 1 for depth in self.stage_depths):
            raise ValueError("stage_depths must contain positive integers")
        common = dict(spikMamba=True, patch_size=patch_size,
                      linear_projection=linear_projection,
                      residual_connections=residual_connections,
                      dwconv2d_spiking=dwconv2d_spiking,
                      patch_embedding_spiking=patch_embedding_spiking,
                      dropout=0.1)
        stages = []
        input_channels = 4
        for channels, depth in zip(self.stage_channels, self.stage_depths):
            blocks = [ConvBlock(input_channels, channels, **common)]
            repeated = dict(common, patch_size=1)
            blocks.extend(ConvBlock(channels, channels, **repeated) for _ in range(depth - 1))
            stages.append(nn.ModuleList(blocks))
            input_channels = channels
        self.stages = nn.ModuleList(stages)

        skip_channels = list(reversed(self.stage_channels[:-1]))
        self.up_blocks = nn.ModuleList()
        self.concat_blocks = nn.ModuleList()
        decoder_channels = self.stage_channels[-1]
        for channels in skip_channels:
            self.up_blocks.append(DeconvBlock(decoder_channels, channels, patch_size, patch_size, dropout=0.1))
            self.concat_blocks.append(ConvBlock(channels * 2, channels, padding=1, dropout=0.1))
            decoder_channels = channels
        self.up_blocks.append(DeconvBlock(decoder_channels, self.stage_channels[0], patch_size, patch_size, dropout=0.1))
        self.final_conv = ConvBlock(self.stage_channels[0], self.stage_channels[0], padding=1, dropout=0.1)
        self.class_conv = ConvBlock(self.stage_channels[0], out_channels, kernel_size=1, dropout=0, normalization=False, spiking=False)

    def forward(self, x_win, t0=0):
        _, steps, _, height, width = x_win.shape
        pad_h = (-height) % self.encoder_scale
        pad_w = (-width) % self.encoder_scale
        logits = []
        for step in range(steps):
            time_step = t0 + step
            x = F.pad(x_win[:, step], (0, pad_w, 0, pad_h)) if pad_h or pad_w else x_win[:, step]
            skips = []
            for stage in self.stages:
                for block in stage:
                    x = block(x, time_step)
                skips.append(x)
            for index, up_block in enumerate(self.up_blocks[:-1]):
                x = up_block(x, time_step)
                x = self.concat_blocks[index](torch.cat([skips[-2 - index], x], 1), time_step)
            x = self.final_conv(self.up_blocks[-1](x, time_step), time_step)
            logits.append(self.class_conv(x, time_step)[..., :height, :width])
        return torch.stack(logits, 2)

    def detach_states(self):
        for module in self.modules():
            if hasattr(module, "detach") and callable(module.detach):
                module.detach()


class SNNBraTSVSS(_SNNBraTSVSS):
    def __init__(self, out_channels=4, stage_depths=(1, 1, 1), patch_size=4,
                 linear_projection=True, residual_connections=True,
                 dwconv2d_spiking=True, patch_embedding_spiking=False):
        if len(stage_depths) != 3:
            raise ValueError("VSS requires exactly 3 stage depths")
        super().__init__(out_channels, (32, 64, 128), stage_depths, patch_size,
                         linear_projection, residual_connections,
                         dwconv2d_spiking, patch_embedding_spiking)


class SNNBraTSVSSDeep(_SNNBraTSVSS):
    def __init__(self, out_channels=4, stage_depths=(1, 1, 1, 1), patch_size=4,
                 linear_projection=True, residual_connections=True,
                 dwconv2d_spiking=True, patch_embedding_spiking=False):
        if len(stage_depths) != 4:
            raise ValueError("VSS_deep requires exactly 4 stage depths")
        super().__init__(out_channels, (32, 64, 128, 256), stage_depths, patch_size,
                         linear_projection, residual_connections,
                         dwconv2d_spiking, patch_embedding_spiking)
