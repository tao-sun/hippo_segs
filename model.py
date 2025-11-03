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


import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuses your existing ConvBlock and DeconvBlock definitions


import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: This class REUSES your existing ConvBlock and DeconvBlock from model.py
# (the ones that use GroupNorm + PLIFNode). Do NOT redefine them here.
# Just import/keep them in the same module and put this class below them.


class SNNBraTSDeep(nn.Module):
    """
    Deeper variant of your *original* SNNBraTS that robustly accepts ANY H×W.

    Strategy (systematic + size-agnostic):
      1) Compute the number of pooling operations (num_pools). For this deeper
         model we use 5 pools (→ bottleneck 512). Required stride multiple M=2**num_pools=32.
      2) For each window slice, pad on the *right* and *bottom* so (H,W) become
         multiples of M. This guarantees exact halves/doubles through pool/deconv.
      3) Run the network unchanged (no per-level padding/cropping needed).
      4) After the classifier, crop logits back to the original (H,W).

    Input (windowed):
        x_win: (B, k, 4, H, W)
        t0   : absolute start time (no neuron reset inside the window)
    Output:
        logits: (B, out_channels, k, H, W)
    """
    def __init__(self, out_channels: int = 4, dropout: float = 0.1, num_pools: int = 5):
        super().__init__()
        assert num_pools >= 1, "num_pools must be >= 1"
        self.out_channels = int(out_channels)
        self.num_pools = int(num_pools)
        self.multiple = 2 ** self.num_pools  # e.g., 32 for 5 pools

        # ---------------- Encoder (single conv per level) ----------------
        # Channels: 32 → 64 → 128 → 256 → 512 (5 levels before bottom out)
        self.conv_block1 = ConvBlock(4,   32, padding=1, dropout=dropout)
        self.conv_block2 = ConvBlock(32,  64, padding=1, dropout=dropout)
        self.conv_block3 = ConvBlock(64, 128, padding=1, dropout=dropout)
        self.conv_block4 = ConvBlock(128,256, padding=1, dropout=dropout)
        self.conv_block5 = ConvBlock(256,512, padding=1, dropout=dropout)
        self.pool = nn.MaxPool2d(2, 2)

        # ---------------- Decoder (mirror; keep 512 like your original kept 128) ----------------
        self.deconv_block1 = DeconvBlock(512, 512, dropout=dropout)
        self.deconv1_conv  = ConvBlock(512, 512, padding=1, dropout=dropout)
        self.concat1_conv  = ConvBlock(512 + 256, 512, padding=1, dropout=dropout)

        self.deconv_block2 = DeconvBlock(512, 512, dropout=dropout)
        self.deconv2_conv  = ConvBlock(512, 512, padding=1, dropout=dropout)
        self.concat2_conv  = ConvBlock(512 + 128, 512, padding=1, dropout=dropout)

        self.deconv_block3 = DeconvBlock(512, 512, dropout=dropout)
        self.deconv3_conv  = ConvBlock(512, 512, padding=1, dropout=dropout)
        self.concat3_conv  = ConvBlock(512 +  64, 512, padding=1, dropout=dropout)

        self.deconv_block4 = DeconvBlock(512, 512, dropout=dropout)
        self.deconv4_conv  = ConvBlock(512, 512, padding=1, dropout=dropout)
        self.concat4_conv  = ConvBlock(512 +  32, 512, padding=1, dropout=dropout)

        self.deconv_block5 = DeconvBlock(512, 512, dropout=dropout)
        self.deconv5_conv  = ConvBlock(512, 512, padding=1, dropout=dropout)

        # ---------------- Classifier (non-spiking) ----------------
        self.class_conv = ConvBlock(
            512, self.out_channels,
            padding=1, dropout=0.0,
            normalization=False, spiking=False
        )

    # ---------------- internal helpers ----------------
    @staticmethod
    def _pad_right_bottom(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
        """Pad (right,bottom) only to keep alignment semantics of U-Net-style models."""
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h))  # (left,right,top,bottom)

    def _required_padding(self, H: int, W: int) -> tuple[int, int]:
        M = self.multiple
        pad_h = (M - (H % M)) % M
        pad_w = (M - (W % M)) % M
        return pad_h, pad_w

    # ---------------- forward (windowed; size-agnostic) ----------------
    def forward(self, x_win: torch.Tensor, t0: int = 0) -> torch.Tensor:
        B, k, C, H, W = x_win.shape
        pad_h, pad_w = self._required_padding(H, W)  # compute ONCE per window
        Hp, Wp = H + pad_h, W + pad_w

        logits = []
        for i in range(k):
            time_step = t0 + i
            x = x_win[:, i, :, :, :]                    # (B,4,H,W)
            if pad_h or pad_w:
                x = self._pad_right_bottom(x, pad_h, pad_w)  # (B,4,Hp,Wp)

            # -------- Encoder --------
            x1 = self.conv_block1(x, time_step)     # (B,32,Hp,Wp)
            p1 = self.pool(x1)                      # (B,32,Hp/2,Wp/2)

            x2 = self.conv_block2(p1, time_step)    # (B,64,Hp/2,Wp/2)
            p2 = self.pool(x2)                      # (B,64,Hp/4,Wp/4)

            x3 = self.conv_block3(p2, time_step)    # (B,128,Hp/4,Wp/4)
            p3 = self.pool(x3)                      # (B,128,Hp/8,Wp/8)

            x4 = self.conv_block4(p3, time_step)    # (B,256,Hp/8,Wp/8)
            p4 = self.pool(x4)                      # (B,256,Hp/16,Wp/16)

            x5 = self.conv_block5(p4, time_step)    # (B,512,Hp/16,Wp/16)
            p5 = self.pool(x5)                      # (B,512,Hp/32,Wp/32)

            # -------- Decoder --------
            y = self.deconv_block1(p5, time_step)   # (B,512,Hp/16,Wp/16)
            y = self.deconv1_conv(y, time_step)
            y = torch.cat([p4, y], dim=1)           # (B,256+512,Hp/16,Wp/16)
            y = self.concat1_conv(y, time_step)     # (B,512,Hp/16,Wp/16)

            y = self.deconv_block2(y, time_step)    # (B,512,Hp/8,Wp/8)
            y = self.deconv2_conv(y, time_step)
            y = torch.cat([p3, y], dim=1)           # (B,128+512,Hp/8,Wp/8)
            y = self.concat2_conv(y, time_step)     # (B,512,Hp/8,Wp/8)

            y = self.deconv_block3(y, time_step)    # (B,512,Hp/4,Wp/4)
            y = self.deconv3_conv(y, time_step)
            y = torch.cat([p2, y], dim=1)           # (B,64+512,Hp/4,Wp/4)
            y = self.concat3_conv(y, time_step)     # (B,512,Hp/4,Wp/4)

            y = self.deconv_block4(y, time_step)    # (B,512,Hp/2,Wp/2)
            y = self.deconv4_conv(y, time_step)
            y = torch.cat([p1, y], dim=1)           # (B,32+512,Hp/2,Wp/2)
            y = self.concat4_conv(y, time_step)     # (B,512,Hp/2,Wp/2)

            y = self.deconv_block5(y, time_step)    # (B,512,Hp,Wp)
            y = self.deconv5_conv(y, time_step)     # (B,512,Hp,Wp)

            out_t = self.class_conv(y, time_step)   # (B,out_channels,Hp,Wp)

            # crop back to original (H,W)
            out_t = out_t[..., :H, :W]
            logits.append(out_t)

        return torch.stack(logits, dim=2)  # (B,out_channels,k,H,W)

    # TBPTT helper (parity with your original)
    def detach_states(self):
        for m in self.modules():
            if hasattr(m, "detach") and callable(m.detach):
                m.detach()
