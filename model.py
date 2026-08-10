import math

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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=0,
                 dropout=0.3, init_tau=2.0,
                 normalization=True, spiking=True, ss2d=True):
        super().__init__()
        self.dropout = float(dropout)
        self.normalization = normalization
        self.spiking = spiking
        self.ss2d = ss2d

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        
        # if self.ss2d:
            # self.ssm_module = SS2D(
            #         channels=out_channels,
            #         d_state=16,
            #         dt_rank="auto")
            # self.pre_ssm_spike_neurons = PLIFNode(
            #     init_tau=init_tau,
            #     surrogate_function=surrogate.ATan(),
            #     detach_reset=True,
            #     no_spiking=(not spiking)
            # )

            # self.pre_norm = nn.GroupNorm(1, out_channels)

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
        # if self.ss2d:
        #     out = self.ssm_module(out)

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


class SpikMamba2D(nn.Module):
    """Stateful spiking four-direction selective scan for BCHW features."""

    def __init__(self,
                 channels: int,
                 d_state: int = 16,
                 dt_rank="auto",
                 conv_kernel_size: int = 3,
                 init_tau: float = 2.0,
                 selective_scan=None,
                 device=None,
                 dtype=None,
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 dt_init: str = "random",
                 dt_scale: float = 1.0,
                 dt_init_floor: float = 1e-4):
        super().__init__()
        if (
            not isinstance(conv_kernel_size, int)
            or conv_kernel_size <= 0
            or conv_kernel_size % 2 == 0
        ):
            raise ValueError(
                "conv_kernel_size must be a positive odd integer"
            )

        factory_kwargs = {"device": device, "dtype": dtype}
        self.channels = int(channels)
        self.d_state = int(d_state)
        self.dt_rank = (
            math.ceil(self.channels / 16) if dt_rank == "auto" else int(dt_rank)
        )

        self.linear_m = nn.Linear(
            self.channels, self.channels, **factory_kwargs
        )
        self.lif_1 = PLIFNode(
            init_tau=init_tau,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
        )
        self.scan_conv1d = nn.Conv1d(
            4 * self.channels,
            4 * self.channels,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2,
            groups=4 * self.channels,
            device=device,
            dtype=dtype,
        )
        self.lif_2 = PLIFNode(
            init_tau=init_tau,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
        )
        self.lif_ssm = PLIFNode(
            init_tau=init_tau,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
        )
        self.selective_scan = (
            selective_scan if selective_scan is not None else selective_scan_fn
        )

        x_projs = tuple(
            nn.Linear(
                self.channels,
                self.dt_rank + 2 * self.d_state,
                bias=False,
                **factory_kwargs,
            )
            for _ in range(4)
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([projection.weight for projection in x_projs], dim=0)
        )

        dt_projs = tuple(
            self.dt_init(
                self.dt_rank,
                self.channels,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            )
            for _ in range(4)
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([projection.weight for projection in dt_projs], dim=0)
        )
        self.dt_projs_bias = nn.Parameter(
            torch.stack([projection.bias for projection in dt_projs], dim=0)
        )

        self.A_logs = self.A_log_init(
            self.d_state, self.channels, copies=4, merge=True, device=device
        )
        self.Ds = self.D_init(self.channels, copies=4, merge=True, device=device)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"unsupported dt_init={dt_init!r}")

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        base = torch.arange(1, d_state + 1, dtype=torch.float32, device=device)
        A = base.unsqueeze(0).repeat(d_inner, 1)
        A_log = torch.log(A)
        if copies > 1:
            A_log = A_log.unsqueeze(0).repeat(copies, 1, 1)
            if merge:
                A_log = A_log.flatten(0, 1)
        parameter = nn.Parameter(A_log)
        parameter._no_weight_decay = True
        return parameter

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = D.unsqueeze(0).repeat(copies, 1)
            if merge:
                D = D.flatten(0, 1)
        parameter = nn.Parameter(D)
        parameter._no_weight_decay = True
        return parameter

    @staticmethod
    def _cross_scan(x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        length = height * width
        horizontal_vertical = torch.stack(
            [
                x.view(batch, channels, length),
                x.transpose(2, 3).contiguous().view(batch, channels, length),
            ],
            dim=1,
        )
        return torch.cat(
            [horizontal_vertical, torch.flip(horizontal_vertical, dims=[-1])],
            dim=1,
        )

    @staticmethod
    def _align_scan_outputs(out_y: torch.Tensor, height: int, width: int):
        batch, _, channels, length = out_y.shape
        inverse = torch.flip(out_y[:, 2:4], dims=[-1])
        width_height = (
            out_y[:, 1]
            .view(batch, channels, width, height)
            .transpose(2, 3)
            .contiguous()
            .view(batch, channels, length)
        )
        inverse_width_height = (
            inverse[:, 1]
            .view(batch, channels, width, height)
            .transpose(2, 3)
            .contiguous()
            .view(batch, channels, length)
        )
        return out_y[:, 0], inverse[:, 0], width_height, inverse_width_height

    def forward_core(self, x: torch.Tensor):
        batch, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(
                f"SS2D expects {self.channels} channels, got {channels}"
            )
        length = height * width
        directions = 4
        xs = self._cross_scan(x)
        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l", xs, self.x_proj_weight
        )
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dts = torch.einsum(
            "b k r l, k d r -> b k d l", dts, self.dt_projs_weight
        )

        xs_packed = xs.float().view(batch, directions * channels, length)
        dts_packed = dts.contiguous().float().view(
            batch, directions * channels, length
        )
        Bs = Bs.float().view(batch, directions, self.d_state, length)
        Cs = Cs.float().view(batch, directions, self.d_state, length)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(
            directions * channels, self.d_state
        )
        dt_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs_packed,
            dts_packed,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(batch, directions, channels, length)
        return self._align_scan_outputs(out_y, height, width)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"SS2D expects BCHW input, got shape={tuple(x.shape)}"
            )
        batch, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(
                f"SS2D expects {self.channels} channels, got {channels}"
            )
        scans = self.forward_core(x)
        return sum(scans).view(batch, channels, height, width)



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
                 ssm_dt_rank="auto"):
        super().__init__()
        # Encoder
        self.conv_block1 = ConvBlock(4, 32, padding=1, dropout=0.1, ss2d=True)
        self.conv_block2 = ConvBlock(32, 64, padding=1, dropout=0.1)
        self.conv_block3 = ConvBlock(64, 128, padding=1, dropout=0.1)

        # Decoder
        self.deconv_block1 = DeconvBlock(128, 128, dropout=0.1)
        self.deconv1_conv  = ConvBlock(128, 128, padding=1, dropout=0.1, ss2d = False)
        self.concat1_conv  = ConvBlock(128 + 64, 128, padding=1, dropout=0.1, ss2d = False)

        self.deconv_block2 = DeconvBlock(128, 128, dropout=0.1)
        self.deconv2_conv  = ConvBlock(128, 128, padding=1, dropout=0.1, ss2d = False)
        self.concat2_conv  = ConvBlock(128 + 32, 128, padding=1, dropout=0.1, ss2d = False)

        self.deconv_block3 = DeconvBlock(128, 128, dropout=0.1)
        self.deconv3_conv  = ConvBlock(128, 128, padding=1, dropout=0.1, ss2d = False)

        # Classifier head (non-spiking); out_channels = 4 classes {0,1,2,3} where 3 corresponds to BraTS label 4
        self.class_conv = ConvBlock(128, out_channels, padding=1, dropout=0.0, normalization=False, spiking=False, ss2d = False)

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
        self.class_conv = ConvBlock(32, self.out_channels, kernel_size=1, padding=0, dropout=0.0, normalization=False, spiking=False)

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
        self.class_conv = ConvBlock(32, self.out_channels, kernel_size=1, padding=0, dropout=0.0, normalization=False, spiking=False)

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
        self.class_conv = ConvBlock(32, self.out_channels, kernel_size=1, padding=0, dropout=0.0, normalization=False, spiking=False)

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
