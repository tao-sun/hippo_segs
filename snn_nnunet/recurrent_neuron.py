"""Selective spatial multi-timescale dendrites with the repository's PLIF soma.

The zero-padded signed DWConv has induced infinity norm <= its kernel L1
norm <= 1. Its convex mixture with identity is nonexpansive. Each dendrite
is therefore a contractive EMA for fixed learned lambda in (0,1), bounded
by max(initial state infinity norm, input infinity norm). Somatic resets
never participate in this recurrence. See SELECTIVE_RECURRENT_NEURON.md.
"""
import torch
from torch import nn
from torch.nn import functional as F

from snn_nnunet.precision import FP32Module, fp32_forward
from snn_nnunet.recurrent_config import recurrent_options, initial_temporal_raw, initial_bounded_logit
from snn_nnunet.spike_neurons import PLIFNode


class BoundedDepthwiseConv(nn.Conv2d):
    """Train raw signed weights; normalize the *effective* operator every call."""
    def __init__(self, channels, config):
        super().__init__(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.max_l1 = config['max_kernel_l1_norm']
        with torch.no_grad():
            pattern = self.weight.new_tensor([1, -1, 1, -1, 0, 1, -1, 1, -1]).reshape(1, 1, 3, 3)
            self.weight.copy_(pattern * config['kernel_surround_init'])
            self.weight[:, :, 1, 1] = config['kernel_center_init']

    def effective_weight(self):
        raw = self.weight.float()
        norm = raw.abs().sum((1, 2, 3), keepdim=True)
        # A few ulps of headroom prevent an FP32 reduction rounding above the bound.
        scale = norm / self.max_l1 * (1 + 4*torch.finfo(raw.dtype).eps)
        return raw / scale.clamp_min(1.)

    def forward(self, x):
        return F.conv2d(x, self.effective_weight(), padding=1, groups=self.groups)


class SelectiveMultiTimescaleLIF(FP32Module):
    """One slice per call: (BCHW, absolute step) -> (spikes, soma voltage).

    t=0 begins an independent spatial patch, including a new patient/mirror.
    detach() truncates graphs but keeps values across windows. A changed shape
    or device starts a fresh sequence. Nonconsecutive nonzero steps are errors,
    preventing accidental double updates from overlapping temporal windows.
    Transient memories are nonpersistent buffers, excluded from checkpoints.
    """
    is_spiking_layer = True
    state_names = ('fast_memory', 'medium_memory', 'slow_memory')
    compartment_names = ('fast', 'medium', 'slow')

    def __init__(self, channels, config, *, init_tau=2., surrogate_function=None, detach_reset=True):
        super().__init__()
        self.config = recurrent_options(config)
        if self.config is None:
            raise ValueError('A recurrent_neuron configuration is required')
        if type(channels) is not int or channels <= 0:
            raise ValueError('channels must be a positive integer')
        self.channels = channels
        raw = initial_temporal_raw(self.config['lambda_init'])
        self.temporal_raw = nn.Parameter(raw[:, None].repeat(1, channels))
        spatial = self.config['spatial_mixing']
        self.spatial_enabled = spatial['enabled']
        if self.spatial_enabled:
            eta = spatial['eta_init']
            self.eta_logits = nn.Parameter(torch.full((3, channels), initial_bounded_logit(eta, 'eta_init')))
            self.spatial_kernels = nn.ModuleList([BoundedDepthwiseConv(channels, spatial) for _ in range(3)])
        else:
            self.register_parameter('eta_logits', None)
            self.spatial_kernels = nn.ModuleList()
        readout = self.config['selective_readout']
        self.selective_enabled = readout['enabled']
        if self.selective_enabled:
            self.readout_gates = nn.ModuleList([nn.Conv2d(2*channels, channels, 1) for _ in range(3)])
            for gate in self.readout_gates:
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)
            self.register_parameter('static_gate_logits', None)
        else:
            self.readout_gates = nn.ModuleList()
            gate = readout['static_gate_init']
            self.static_gate_logits = nn.Parameter(torch.full((3, channels), initial_bounded_logit(gate, 'static_gate_init')))
        self.memory_projection = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.xavier_uniform_(self.memory_projection.weight)
        kwargs = {'init_tau': init_tau, 'detach_reset': detach_reset}
        if surrogate_function is not None:
            kwargs['surrogate_function'] = surrogate_function
        self.soma = PLIFNode(**kwargs)
        for name in self.state_names:
            self.register_buffer(name, None, persistent=False)
        self._next_step = None
        self.collect_diagnostics = False
        self._activity = {}
        self._zero_gradient_samples = {}

    @property
    def memories(self):
        return tuple(getattr(self, name) for name in self.state_names)

    @property
    def somatic_membrane(self):
        return getattr(self.soma, 'v', None)

    def temporal_coefficients(self):
        tau = F.softplus(self.temporal_raw.float()).cumsum(0) + 1e-6
        return torch.exp(-tau.reciprocal())

    def mixing_coefficients(self):
        return self.eta_logits.float().sigmoid() if self.eta_logits is not None else None

    def reset(self):
        for name in self.state_names:
            setattr(self, name, None)
        self._next_step = None
        self._activity = {}
        self.soma.neuro_states_init = False
        for name in ('v', 's'):
            if hasattr(self.soma, name):
                delattr(self.soma, name)

    def detach(self):
        for name in self.state_names:
            state = getattr(self, name)
            if state is not None:
                setattr(self, name, state.detach())
        self.soma.detach()

    def readout(self, features, memories):
        if self.selective_enabled:
            return tuple(F.conv2d(torch.cat((features, memory), 1), gate.weight.float(), gate.bias.float()).sigmoid()
                         for gate, memory in zip(self.readout_gates, memories))
        coefficients = self.static_gate_logits.float().sigmoid()
        return tuple(value[None, :, None, None].expand_as(features) for value in coefficients)

    @fp32_forward
    def forward(self, features, time_step):
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(f'expected [B,{self.channels},H,W] features')
        if type(time_step) is not int or time_step < 0:
            raise ValueError('time_step must be a nonnegative integer')
        memory = self.fast_memory
        changed = memory is not None and (memory.shape != features.shape or memory.device != features.device)
        if time_step == 0 or changed:
            self.reset()
        if self._next_step is not None and time_step != self._next_step:
            raise ValueError('recurrent neuron requires consecutive slice offsets; reset for an independent patch')
        with torch.autocast(features.device.type, enabled=False):
            x = features.float()
            if self.fast_memory is None:
                for name in self.state_names:
                    setattr(self, name, torch.zeros_like(x))
                # Preserve PLIF initialization even when a new shape starts at a nonzero offset.
                if time_step != 0:
                    self.soma.init_neuro_states(features)
            lambdas = self.temporal_coefficients()
            etas = self.mixing_coefficients()
            for r, name in enumerate(self.state_names):
                previous = getattr(self, name)
                if self.spatial_enabled:
                    eta = etas[r][None, :, None, None]
                    previous = (1-eta)*previous + eta*self.spatial_kernels[r](previous)
                decay = lambdas[r][None, :, None, None]
                setattr(self, name, decay*previous + (1-decay)*x)
            gates = self.readout(x, self.memories)
            aggregate = sum(gate*memory for gate, memory in zip(gates, self.memories))
            projection = F.conv2d(aggregate, self.memory_projection.weight.float())
            current = x + projection
        # The existing soma receives FP32 current under the same full-precision guard.
        spike, voltage = self.soma(current, time_step)
        self._next_step = time_step + 1
        if self.collect_diagnostics:
            with torch.no_grad():
                self._activity = self._activity_diagnostics(x, projection, current, gates, spike, voltage)
        return spike, voltage

    @staticmethod
    def _rms(value):
        return value.detach().float().square().mean().sqrt()

    @staticmethod
    def _distribution(prefix, value):
        flat = value.detach().float().flatten()
        # Bound quantile cost even for spatially selective gate maps.
        sample = flat[::max(1, flat.numel() // 1024)]
        quantiles = torch.quantile(sample, sample.new_tensor([.1, .5, .9]))
        return {f'{prefix}/mean': flat.mean(), f'{prefix}/min': flat.amin(),
                f'{prefix}/max': flat.amax(), **{f'{prefix}/q{q}': v for q, v in zip((10, 50, 90), quantiles)}}

    def _activity_diagnostics(self, x, projection, current, gates, spikes, voltage):
        stats = {'input/rms': self._rms(x), 'projection/rms': self._rms(projection),
                 'current/rms': self._rms(current), 'soma/voltage_rms': self._rms(voltage), 'soma/firing_rate': spikes.detach().float().mean()}
        stats['projection/input_ratio'] = stats['projection/rms'] / stats['input/rms'].clamp_min(1e-8)
        finite = (torch.isfinite(x).all() & torch.isfinite(current).all()
                  & torch.isfinite(spikes).all() & torch.isfinite(voltage).all())
        for name, memory, gate in zip(self.compartment_names, self.memories, gates):
            stats[f'memory/{name}/rms'] = self._rms(memory)
            stats.update(self._distribution(f'gate/{name}', gate))
            stats[f'gate/{name}/saturated'] = ((gate < .01) | (gate > .99)).float().mean()
            finite = finite & torch.isfinite(memory).all() & torch.isfinite(gate).all()
        stats['state/nonfinite'] = (~finite).float()
        largest = torch.stack([stats[f'memory/{name}/rms'] for name in self.compartment_names]).amax()
        stats['state/growth_anomaly'] = (largest > self.config['diagnostics']['growth_ratio_threshold'] * stats['input/rms'].clamp_min(1e-8)).float()
        return stats

    @torch.no_grad()
    def diagnostics(self):
        stats = dict(self._activity)
        lambdas = self.temporal_coefficients()
        gaps = lambdas[1:] - lambdas[:-1]
        stats['lambda/min_gap'] = gaps.amin()
        stats['lambda/ordered'] = ((gaps > 0).all() & (lambdas > 0).all() & (lambdas < 1).all()).float()
        stats['lambda/nearly_equal'] = (gaps < self.config['diagnostics']['near_lambda_gap']).float().mean()
        for r, name in enumerate(self.compartment_names):
            stats.update(self._distribution(f'lambda/{name}', lambdas[r]))
            if r < 2:
                stats.update(self._distribution(f'lambda/gap_{name}', gaps[r]))
            if self.spatial_enabled:
                stats.update(self._distribution(f'eta/{name}', self.mixing_coefficients()[r]))
                norm = self.spatial_kernels[r].effective_weight().abs().sum((1, 2, 3))
                stats[f'kernel/{name}/l1_max'] = norm.amax()
            if not self.selective_enabled:
                stats.update(self._distribution(f'static_gate/{name}', self.static_gate_logits[r].sigmoid()))
        groups = {'temporal': [self.temporal_raw], 'eta': [self.eta_logits] if self.spatial_enabled else [],
                  'kernels': list(self.spatial_kernels.parameters()),
                  'gates': list(self.readout_gates.parameters()) if self.selective_enabled else [self.static_gate_logits],
                  'projection': list(self.memory_projection.parameters()), 'soma': list(self.soma.parameters())}
        for name, parameters in groups.items():
            if not parameters:
                continue
            grads = [p.grad for p in parameters if p.grad is not None]
            zero = self.temporal_raw.new_zeros(())
            norm = torch.stack([g.detach().float().square().sum() for g in grads]).sum().sqrt() if grads else zero
            stats[f'gradient/{name}/norm'] = norm
            stats[f'gradient/{name}/missing'] = zero + (len(parameters) - len(grads))
            stats[f'gradient/{name}/nonfinite'] = (~torch.isfinite(norm)).float()
            count = self._zero_gradient_samples.get(name, zero)
            count = torch.where(norm == 0, count + 1, zero)
            self._zero_gradient_samples[name] = count
            stats[f'gradient/{name}/consecutive_zero_samples'] = count
        return stats
