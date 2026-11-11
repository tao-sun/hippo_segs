"""Multi-memory LIF with an input-conditioned global linear attention readout."""
import logging

import torch
from torch import nn
from torch.nn import functional as F

from snn_nnunet.precision import FP32Module, fp32_forward, require_fp32
from snn_nnunet.recurrent_neuron import SelectiveMultiTimescaleLIF
from snn_nnunet.recurrent_config import ATTENTION_MODEL


class MultiMemoryLinearAttention(FP32Module):
    """One-head ELU+1 attention over updated fast/medium/slow BCHW memories.

    Only Bx32x32 sufficient statistics are formed. Spatial token count never
    appears on both matrix axes. The three memory-scale embeddings enter keys
    before the positive feature map. All operations follow the existing FP32
    policy even when called under an outer autocast context.
    """
    key_dim = 32
    value_dim = 32
    epsilon = 1e-6

    def __init__(self, channels):
        super().__init__()
        if type(channels) is not int or channels <= 0:
            raise ValueError('channels must be a positive integer')
        self.channels = channels
        self.query = nn.Conv2d(channels, 32, 1, bias=False)
        self.keys = nn.ModuleList(nn.Conv2d(channels, 32, 1, bias=False) for _ in range(3))
        self.values = nn.ModuleList(nn.Conv2d(channels, 32, 1, bias=False) for _ in range(3))
        self.scale_embeddings = nn.Parameter(torch.zeros(3, 32))
        self.output = nn.Conv2d(32, channels, 1, bias=False)
        for layer in (self.query, *self.keys, *self.values, self.output):
            nn.init.xavier_uniform_(layer.weight)

    @fp32_forward
    def forward(self, features, memories):
        if features.ndim != 4 or features.shape[1] != self.channels or min(features.shape) <= 0:
            raise ValueError(f'expected nonempty [B,{self.channels},H,W] continuous features')
        if len(memories) != 3:
            raise ValueError('Linear Attention requires exactly three updated memories')
        statistics, key_sum = None, None
        for r, (memory, key, value) in enumerate(zip(memories, self.keys, self.values)):
            require_fp32(memory, 'Attention memory')
            if memory.shape != features.shape or memory.device != features.device:
                raise ValueError('Attention memories must match feature shape and device')
            k = key(memory).flatten(2).transpose(1, 2) + self.scale_embeddings[r]
            positive_k = F.elu(k) + 1
            v = value(memory).flatten(2).transpose(1, 2)
            contribution = positive_k.transpose(1, 2) @ v
            reduction = positive_k.sum(dim=1)
            statistics = contribution if statistics is None else statistics + contribution
            key_sum = reduction if key_sum is None else key_sum + reduction
        q = F.elu(self.query(features).flatten(2).transpose(1, 2)) + 1
        numerator = q @ statistics
        denominator = q @ key_sum.unsqueeze(-1)
        attended = numerator / (denominator + self.epsilon)
        b, _, h, w = features.shape
        return self.output(attended.transpose(1, 2).reshape(b, 32, h, w))


class MultiMemoryLinearAttentionLIF(SelectiveMultiTimescaleLIF):
    """Linear attention readout with a normalized memory-to-soma residual."""
    is_linear_attention = True

    def __init__(self, channels, config, **kwargs):
        super().__init__(channels, dict(config, model=ATTENTION_MODEL), **kwargs)
        # No old adaptive or static gates are retained in this variant.
        self.readout_gates = nn.ModuleList()
        self.static_gate_logits = None
        # Preserve initialization of all common downstream model weights at a
        # fixed seed. New readout initialization must not advance their RNG.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices):
            self.attention = MultiMemoryLinearAttention(channels)
        self.output_norm = nn.GroupNorm(1, channels, eps=1e-5, affine=False)
        self.attention_gamma = self.config['attention_gamma']
        self.attention_gamma_mode = self.config['attention_gamma_mode']
        if self.attention_gamma_mode == 'learnable_per_channel':
            # Channels broadcast over a BCHW slice. Every cell owns its parameter.
            self.gamma = nn.Parameter(torch.full(
                (1, channels, 1, 1), self.config['attention_gamma_init'], dtype=torch.float32))
        else:
            # None adds no state_dict keys, preserving old fixed-gamma checkpoints.
            self.register_parameter('gamma', None)
        logging.getLogger(__name__).info(
            '%s initialized: attention_gamma_mode=%s, gamma=%s', type(self).__name__,
            self.attention_gamma_mode,
            self.attention_gamma if self.gamma is None else self.config['attention_gamma_init'])

    @torch.no_grad()
    def project_gamma_(self):
        """Project learned gains after parameter updates, preserving their identity."""
        if self.gamma is not None:
            self.gamma.clamp_(0., 1.)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Clamp a copy so loading old unconstrained weights also works recursively
        # and with assign=True, without modifying the caller's checkpoint tensors.
        key = prefix + 'gamma'
        if self.gamma is not None and isinstance(state_dict.get(key), torch.Tensor):
            state_dict = dict(state_dict)
            state_dict[key] = state_dict[key].clamp(0., 1.)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def _aggregate_memories(self, features, memories):
        return self.attention(features, memories), None

    def _project_memories(self, aggregate):
        # Scale the complete normalized readout immediately before x + residual.
        residual = self.output_norm(super()._project_memories(aggregate))
        gamma = self.attention_gamma if self.gamma is None else self.gamma.clamp(0., 1.)
        return gamma * residual

    def _activity_diagnostics(self, x, projection, current, gates, spikes, voltage):
        # Attention has no local sigmoid gates; report only applicable states.
        stats = {'input/rms': self._rms(x), 'projection/rms': self._rms(projection),
                 'current/rms': self._rms(current), 'soma/voltage_rms': self._rms(voltage),
                 'soma/firing_rate': spikes.detach().float().mean()}
        stats['projection/input_ratio'] = stats['projection/rms'] / stats['input/rms'].clamp_min(1e-8)
        finite = (torch.isfinite(x).all() & torch.isfinite(current).all()
                  & torch.isfinite(spikes).all() & torch.isfinite(voltage).all())
        for name, memory in zip(self.compartment_names, self.memories):
            stats[f'memory/{name}/rms'] = self._rms(memory)
            finite = finite & torch.isfinite(memory).all()
        stats['state/nonfinite'] = (~finite).float()
        largest = torch.stack([stats[f'memory/{name}/rms'] for name in self.compartment_names]).amax()
        stats['state/growth_anomaly'] = (largest > self.config['diagnostics']['growth_ratio_threshold'] * stats['input/rms'].clamp_min(1e-8)).float()
        return stats

    @torch.no_grad()
    def diagnostics(self):
        stats = super().diagnostics()
        if self.gamma is not None:
            gamma = self.gamma.detach().clamp(0., 1.)
            stats.update({'gamma/mean': gamma.mean(), 'gamma/min': gamma.amin(),
                          'gamma/max': gamma.amax()})
            grad = self.gamma.grad
            stats['gradient/gamma/missing'] = gamma.new_tensor(float(grad is None))
            if grad is not None:
                norm = grad.detach().float().norm()
                stats['gradient/gamma/norm'] = norm
                stats['gradient/gamma/nonfinite'] = (~torch.isfinite(norm)).float()
        parameters = list(self.attention.parameters())
        grads = [p.grad for p in parameters if p.grad is not None]
        zero = self.temporal_raw.new_zeros(())
        norm = torch.stack([g.detach().float().square().sum() for g in grads]).sum().sqrt() if grads else zero
        prefix = 'gradient/attention'
        stats[f'{prefix}/norm'] = norm
        stats[f'{prefix}/missing'] = zero + len(parameters)-len(grads)
        stats[f'{prefix}/nonfinite'] = (~torch.isfinite(norm)).float()
        count = self._zero_gradient_samples.get('attention', zero)
        count = torch.where(norm == 0, count+1, zero)
        self._zero_gradient_samples['attention'] = count
        stats[f'{prefix}/consecutive_zero_samples'] = count
        return stats


# Import alias for existing callers and saved experimental code.
SiLIFLinearAttention = MultiMemoryLinearAttentionLIF
