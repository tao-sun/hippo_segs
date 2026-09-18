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
        out = self.deconv(x)
        out = self.norm(out)
        out, _ = self.spike_neurons(out, time_step)
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
                 patch_embedding_spiking: bool = False):
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
        self.patch_size = patch_size
        self.linear_projection = linear_projection
        self.residual_connections = residual_connections
        self.dwconv2d_spiking = dwconv2d_spiking
        self.patch_embedding_spiking = patch_embedding_spiking
        self.encoder_scale = self.patch_size ** 3
        # Encoder
        spik_mamba_kwargs = {
            "spikMamba": True,
            "selective_scan": selective_scan,
            "ssm_d_state": ssm_d_state,
            "ssm_dt_rank": ssm_dt_rank,
            "patch_size": self.patch_size,
            "linear_projection": self.linear_projection,
            "residual_connections": self.residual_connections,
            "dwconv2d_spiking": self.dwconv2d_spiking,
            "patch_embedding_spiking": self.patch_embedding_spiking,
        }
        self.conv_block1 = ConvBlock(
            4,
            32,
            padding=1,
            dropout=0.1,
            **spik_mamba_kwargs,
        )
        self.conv_block2 = ConvBlock(
            32,
            64,
            padding=1,
            dropout=0.1,
            **spik_mamba_kwargs,
        )
        self.conv_block3 = ConvBlock(
            64,
            128,
            padding=1,
            dropout=0.1,
            **spik_mamba_kwargs,
        )

        # Decoder
        self.deconv_block1 = DeconvBlock(128, 128, self.patch_size, self.patch_size, dropout=0.1)
        self.deconv1_conv = ConvBlock(
            128, 128, padding=1, dropout=0.1
        )
        self.concat1_conv = ConvBlock(
            128 + 64, 128, padding=1, dropout=0.1
        )

        self.deconv_block2 = DeconvBlock(128, 128, self.patch_size, self.patch_size, dropout=0.1)
        self.deconv2_conv = ConvBlock(
            128, 128, padding=1, dropout=0.1
        )
        self.concat2_conv = ConvBlock(
            128 + 32, 128, padding=1, dropout=0.1
        )

        self.deconv_block3 = DeconvBlock(128, 128, self.patch_size, self.patch_size, dropout=0.1)
        self.deconv3_conv = ConvBlock(
            128, 128, padding=1, dropout=0.1
        )

        # Classifier head (non-spiking); classes {0,1,2,3}, with 3 = BraTS 4
        self.class_conv = ConvBlock(
            128,
            out_channels,
            padding=1,
            dropout=0.0,
            normalization=False,
            spiking=False,
        )


    def forward(self, x_win, t0=0):
        _, steps, _, height, width = x_win.shape
        pad_h = (-height) % self.encoder_scale
        pad_w = (-width) % self.encoder_scale
        logits = []

        for i in range(k):
            time_step = t0 + i  # absolute time across the sequence 0..D-1
            x = x_win[:, i, :, :, :]  # (B,4,H,W)
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h))

            skip1 = self.conv_block1(x, time_step)
            skip2 = self.conv_block2(skip1, time_step)
            x = self.conv_block3(skip2, time_step)

            x = self.deconv_block1(x, time_step)
            x = self.deconv1_conv(x, time_step)
            x = torch.cat([skip2, x], dim=1)
            x = self.concat1_conv(x, time_step)

            x = self.deconv_block2(x, time_step)
            x = self.deconv2_conv(x, time_step)
            x = torch.cat([skip1, x], dim=1)
            x = self.concat2_conv(x, time_step)
            
            x = self.deconv_block3(x, time_step)
            x = self.deconv3_conv(x, time_step)

            x = self.class_conv(x, time_step)  # (B,out_channels,H,W)
            x = x[..., :H, :W]
            logits.append(x)

        logits = torch.stack(logits, dim=2)  # (B,out_channels,k,H,W)
        return logits

    # TBPTT helper: detach neuron states at window boundary to truncate gradients
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
