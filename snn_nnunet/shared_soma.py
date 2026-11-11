"""Opt-in bidirectional dendritic fusion with one forward-recurrent soma.

Legacy SiLIF.forward and output-fusion operators are intentionally untouched.
Directional recurrence uses the same inherited coefficients/spatial operators,
with independent zero boundaries. Fused memories never enter that recurrence.
"""
import torch
from torch import nn
from torch.nn import functional as F

from snn_nnunet.precision import fp32_forward
from snn_nnunet.recurrent_neuron import SelectiveMultiTimescaleLIF


class SharedSOMALIF(SelectiveMultiTimescaleLIF):
    """[B,Z,C,H,W] -> binary [B,Z,C,H,W]; one independent patch per call."""

    is_shared_soma = True

    def __init__(self, channels, config, **kwargs):
        super().__init__(channels, config, **kwargs)
        if not self.selective_enabled:
            raise ValueError('Shared_SOMA requires selective_readout.enabled=true')
        self.memory_fusions = nn.ModuleList(
            nn.Conv2d(2*channels, channels, 1) for _ in self.state_names)
        with torch.no_grad():
            indices = torch.arange(channels)
            for fusion in self.memory_fusions:
                fusion.weight.zero_()
                fusion.bias.zero_()
                fusion.weight[indices, indices, 0, 0] = .5
                fusion.weight[indices, channels+indices, 0, 0] = .5

    def _scan_memories(self, features, *, reverse=False):
        """Return three anatomically aligned trajectories; never call the soma.

        This isolated recurrence matches the legacy update operations exactly.
        Keeping it here avoids refactoring the numerically protected SiLIF path.
        Reset rebinds tensors without detaching stored trajectories or graphs.
        """
        self.reset()
        trajectories = [[None]*features.shape[1] for _ in self.state_names]
        order = range(features.shape[1]-1, -1, -1) if reverse else range(features.shape[1])
        try:
            for name in self.state_names:
                setattr(self, name, torch.zeros_like(features[:, 0]))
            for z in order:
                x = features[:, z]
                lambdas = self.temporal_coefficients()
                etas = self.mixing_coefficients()
                for r, name in enumerate(self.state_names):
                    previous = getattr(self, name)
                    if self.spatial_enabled:
                        eta = etas[r][None, :, None, None]
                        previous = (1-eta)*previous + eta*self.spatial_kernels[r](previous)
                    decay = lambdas[r][None, :, None, None]
                    memory = decay*previous + (1-decay)*x
                    setattr(self, name, memory)
                    trajectories[r][z] = memory
            return tuple(torch.stack(values, dim=1) for values in trajectories)
        finally:
            self.reset()

    @fp32_forward
    def forward(self, features):
        if features.ndim != 5 or features.shape[2] != self.channels or min(features.shape) <= 0:
            raise ValueError(f'Shared_SOMA expected nonempty [B,Z,{self.channels},H,W] features')
        self.reset()
        snapshot = None
        try:
            forward = self._scan_memories(features)
            backward = self._scan_memories(features, reverse=True)
            outputs = []
            for z in range(features.shape[1]):
                x = features[:, z]
                fused = tuple(fusion(torch.cat((a[:, z], b[:, z]), dim=1))
                              for fusion, a, b in zip(self.memory_fusions, forward, backward))
                gates = self.readout(x, fused)
                aggregate = sum(gate*memory for gate, memory in zip(gates, fused))
                projection = F.conv2d(aggregate, self.memory_projection.weight.float())
                current = x + projection
                # One soma, ascending anatomical order; no reset between slices.
                spike, voltage = self.soma(current, z)
                outputs.append(spike)
                if self.collect_diagnostics and z == features.shape[1]-1:
                    for name, memory in zip(self.state_names, fused):
                        setattr(self, name, memory)
                    with torch.no_grad():
                        snapshot = self._activity_diagnostics(x, projection, current, gates, spike, voltage)
            return torch.stack(outputs, dim=1)
        finally:
            self.reset()
            if snapshot is not None:
                self._activity = snapshot

    @torch.no_grad()
    def diagnostics(self):
        """Keep inherited SiLIF diagnostics and include the new fusion weights."""
        stats = super().diagnostics()
        parameters = list(self.memory_fusions.parameters())
        grads = [p.grad for p in parameters if p.grad is not None]
        zero = self.temporal_raw.new_zeros(())
        norm = torch.stack([g.detach().float().square().sum() for g in grads]).sum().sqrt() if grads else zero
        prefix = 'gradient/memory_fusion'
        stats[f'{prefix}/norm'] = norm
        stats[f'{prefix}/missing'] = zero + len(parameters)-len(grads)
        stats[f'{prefix}/nonfinite'] = (~torch.isfinite(norm)).float()
        count = self._zero_gradient_samples.get('memory_fusion', zero)
        count = torch.where(norm == 0, count+1, zero)
        self._zero_gradient_samples['memory_fusion'] = count
        stats[f'{prefix}/consecutive_zero_samples'] = count
        return stats
