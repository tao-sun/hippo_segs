"""Standard continuous Mamba over depth, independently at each spatial site."""
from torch import nn

from snn_nnunet.precision import FP32Module, fp32_forward


class TemporalBottleneckMamba(FP32Module):
    """[B,T,C,H,W] -> X + Mamba(LayerNorm(X)) -> [B,T,C,H,W]."""

    def __init__(self, dim, *, d_state=16, d_conv=4, expand=2, dt_rank='auto'):
        super().__init__()
        # Keep the baseline usable without importing Mamba or its CUDA extension.
        try:
            from mamba_ssm.modules.mamba_simple import Mamba
        except (ImportError, OSError) as error:
            raise ImportError(
                'temporal_mamba=true requires mamba_ssm and its compatible '
                'selective-scan CUDA extension in the training environment'
            ) from error
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv,
                           expand=expand, dt_rank=dt_rank)

    @fp32_forward
    def forward(self, sequence):
        if sequence.ndim != 5 or sequence.shape[2] != self.dim or min(sequence.shape) <= 0:
            raise ValueError(f'expected nonempty [B,T,{self.dim},H,W] bottleneck')
        batch, steps, channels, height, width = sequence.shape
        tokens = sequence.permute(0, 3, 4, 1, 2).reshape(batch * height * width, steps, channels)
        # Fresh Mamba state per patch: inference_params/cached states are unused.
        mixed = tokens + self.mamba(self.norm(tokens))
        return mixed.reshape(batch, height, width, steps, channels).permute(0, 3, 4, 1, 2).contiguous()
