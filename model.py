import torch
import torch.nn as nn
import torch.nn.functional as F

import surrogate
from spike_neurons import PLIFNode


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=0,
                 dropout=0.3, init_tau=2.0,
                 normalization=True, spiking=True):
        super().__init__()
        self.dropout = float(dropout)
        self.normalization = normalization
        self.spiking = spiking

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        # self.norm = nn.BatchNorm2d(out_channels)
        self.norm = nn.GroupNorm(1, out_channels)
        self.spike_neurons = PLIFNode(
            init_tau=init_tau,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            no_spiking=(not spiking)
        )

    def forward(self, x, time_step: int):
        out = self.conv(x)
        if self.normalization:
            out = self.norm(out)
        if self.spiking:
            out, _ = self.spike_neurons(out, time_step)
        else:
            out = self.spike_neurons(out, time_step)
        if self.dropout > 0:
            # IMPORTANT: disable dropout in eval/reference
            out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2,
                 dropout=0.3, init_tau=2.0):
        super().__init__()
        self.dropout = float(dropout)

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=kernel_size, stride=stride, bias=False)
        # self.norm = nn.BatchNorm2d(out_channels)
        self.norm = nn.GroupNorm(1, out_channels)
        self.spike_neurons = PLIFNode(
            init_tau=init_tau,
            surrogate_function=surrogate.ATan(),
            detach_reset=True
        )

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
    def __init__(self, out_channels: int = 4):
        super().__init__()
        # Encoder
        self.conv_block1 = ConvBlock(4, 32, padding=1, dropout=0.1)
        self.conv_block2 = ConvBlock(32, 64, padding=1, dropout=0.1)
        self.conv_block3 = ConvBlock(64, 128, padding=1, dropout=0.1)

        # Decoder
        self.deconv_block1 = DeconvBlock(128, 128, dropout=0.1)
        self.deconv1_conv  = ConvBlock(128, 128, padding=1, dropout=0.1)
        self.concat1_conv  = ConvBlock(128 + 64, 128, padding=1, dropout=0.1)

        self.deconv_block2 = DeconvBlock(128, 128, dropout=0.1)
        self.deconv2_conv  = ConvBlock(128, 128, padding=1, dropout=0.1)
        self.concat2_conv  = ConvBlock(128 + 32, 128, padding=1, dropout=0.1)

        self.deconv_block3 = DeconvBlock(128, 128, dropout=0.1)
        self.deconv3_conv  = ConvBlock(128, 128, padding=1, dropout=0.1)

        # Classifier head (non-spiking); out_channels = 4 classes {0,1,2,3} where 3 corresponds to BraTS label 4
        self.class_conv = ConvBlock(128, out_channels, padding=1, dropout=0.0, normalization=False, spiking=False)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x_win: torch.Tensor, t0: int = 0) -> torch.Tensor:
        B, k, C, H, W = x_win.shape
        logits = []

        for i in range(k):
            time_step = t0 + i  # absolute time across the sequence 0..D-1
            x = x_win[:, i, :, :, :]  # (B,4,H,W)

            x = self.conv_block1(x, time_step)
            pool1 = self.pool(x)

            x = self.conv_block2(pool1, time_step)
            pool2 = self.pool(x)

            x = self.conv_block3(pool2, time_step)
            x = self.pool(x)

            x = self.deconv_block1(x, time_step)
            x = self.deconv1_conv(x, time_step)
            x = torch.cat([pool2, x], dim=1)
            x = self.concat1_conv(x, time_step)

            x = self.deconv_block2(x, time_step)
            x = self.deconv2_conv(x, time_step)
            x = torch.cat([pool1, x], dim=1)
            x = self.concat2_conv(x, time_step)

            x = self.deconv_block3(x, time_step)
            x = self.deconv3_conv(x, time_step)

            x = self.class_conv(x, time_step)  # (B,out_channels,H,W)
            logits.append(x)

        logits = torch.stack(logits, dim=2)  # (B,out_channels,k,H,W)
        return logits

    # TBPTT helper: detach neuron states at window boundary to truncate gradients
    def detach_states(self):
        for m in self.modules():
            # Adjust if your PLIFNode has a different method name
            if hasattr(m, "detach") and callable(m.detach):
                m.detach()
