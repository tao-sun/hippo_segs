"""Original SNNBraTS graph from main:model.py, with selectable block neurons."""
import torch
from torch import nn
from torch.nn import functional as F

from snn_nnunet import surrogate
from snn_nnunet.paper_config import PAPER_DEFAULTS, ENCODER_BLOCK_NAMES, paper_options, normalize_bidirectional_fusion, validate_shared_soma_options, neuron_model, bidirectional_blocks
from snn_nnunet.precision import FP32Module, fp32_forward
from snn_nnunet.spike_neurons import PLIFNode
from snn_nnunet.recurrent_neuron import SelectiveMultiTimescaleLIF, make_selective_neuron


def _plif(options, **kwargs):
    options = dict(options)
    alpha = options.pop('surrogate_alpha')
    return PLIFNode(**options, surrogate_function=surrogate.ATan(alpha=alpha), **kwargs)


def _neuron(name, channels, options, *, shared_soma=False):
    model = neuron_model(options, name)
    plif = options['plif']
    if model == 'plif':
        return _plif(plif)
    neuron_type = make_selective_neuron
    if shared_soma:
        from snn_nnunet.shared_soma import SharedSOMALIF
        neuron_type = SharedSOMALIF
    config = dict(options['multi_memory'], model=model,
                  selective_readout=options['selective_readout'])
    return neuron_type(
        channels, config, init_tau=plif['init_tau'],
        detach_reset=plif['detach_reset'],
        surrogate_function=surrogate.ATan(alpha=plif['surrogate_alpha']))


class PaperBlock(nn.Module):
    def __init__(self, name, in_channels, out_channels, options, *, deconv=False, head=False, shared_soma=False):
        super().__init__()
        block = dict(options['head' if head else 'deconv' if deconv else 'conv'])
        self.dropout = block.pop('dropout')
        if head:
            block.pop('normalization')
            block.pop('spiking')
            plif = block.pop('plif')
        layer = nn.ConvTranspose2d if deconv else nn.Conv2d
        setattr(self, 'deconv' if deconv else 'conv', layer(in_channels, out_channels, **block))
        self.is_deconv, self.is_head = deconv, head
        norm = options['normalization']
        self.norm = (nn.Identity() if head else nn.GroupNorm(
            norm['num_groups'], out_channels, eps=norm['eps'], affine=norm['affine']))
        self.spiking = not head and options[
            'encoder_spiking' if name in ENCODER_BLOCK_NAMES else 'decoder_spiking']
        self.spike_neurons = (_plif(plif, no_spiking=True) if head else
                              _neuron(name, out_channels, options['neurons'], shared_soma=shared_soma)
                              if self.spiking else nn.GELU())

    def forward(self, x, time_step):
        return self.activate(self.project(x), time_step)

    def project(self, x):
        """Direction-independent current before the stateful neuron."""
        x = self.deconv(x) if self.is_deconv else self.conv(x)
        return self.norm(x)

    def activate(self, x, time_step, *, apply_dropout=True):
        if not self.is_head and not self.spiking:
            x = self.spike_neurons(x)
        else:
            x = self.spike_neurons(x, time_step)
            if not self.is_head:
                x = x[0]
        return F.dropout(x, self.dropout, self.training) if apply_dropout and self.dropout else x


def _activity_snapshots(module):
    return [(cell, dict(cell._activity)) for cell in module.modules()
            if getattr(cell, 'is_selective_memory', False) and cell.collect_diagnostics]


def _reset_neurons(module, *, preserve_diagnostics=False):
    activity = _activity_snapshots(module) if preserve_diagnostics else ()
    for cell in module.modules():
        if getattr(cell, 'is_selective_memory', False):
            cell.reset()
        elif isinstance(cell, PLIFNode):
            cell.neuro_states_init = False
            for name in ('v', 's'):
                if hasattr(cell, name):
                    delattr(cell, name)
    for cell, snapshot in activity:
        cell._activity = snapshot


class PaperStageFusion(nn.Module):
    """Conv1x1 -> GroupNorm -> PLIF per slice, with no z memory."""
    def __init__(self, channels, plif, normalization=None):
        super().__init__()
        self.conv = nn.Conv2d(2*channels, channels, 1, bias=False)
        norm = PAPER_DEFAULTS['normalization'] if normalization is None else normalization
        self.norm = nn.GroupNorm(norm['num_groups'], channels, eps=norm['eps'], affine=norm['affine'])
        self.neuron = _plif(plif)
        # Retain the existing symmetric convolution initialization. GroupNorm
        # now rescales its currents, so initial fusion is no longer an exact OR.
        # Convolution, affine normalization and PLIF parameters remain learned.
        with torch.no_grad():
            self.conv.weight.zero_()
            indices = torch.arange(channels)
            gain = self.neuron.w.sigmoid().reciprocal() * self.neuron.v_threshold
            self.conv.weight[indices, indices, 0, 0] = gain
            self.conv.weight[indices, indices+channels, 0, 0] = gain

    def forward(self, forward_sequence, backward_sequence):
        outputs = []
        try:
            for z in range(forward_sequence.shape[1]):
                _reset_neurons(self.neuron)
                current = self.conv(torch.cat((forward_sequence[:, z], backward_sequence[:, z]), dim=1))
                values_per_group = (current.shape[1] // self.norm.num_groups
                                    * current.shape[-2] * current.shape[-1])
                if values_per_group < 2:
                    raise ValueError(
                        'Conv+PLIF fusion GroupNorm requires at least two values per group '
                        'per slice; use fewer groups or larger spatial patches')
                outputs.append(self.neuron(self.norm(current), 0)[0])
            return torch.stack(outputs, dim=1)
        finally:
            _reset_neurons(self.neuron)


class PaperORFusion(nn.Module):
    """Binary spike OR with autograd: maximum splits gradients equally at ties."""
    def forward(self, forward_sequence, backward_sequence):
        return torch.maximum(forward_sequence, backward_sequence)


class PaperUNet(FP32Module):
    def __init__(self, out_channels=3, bidirectional=False, bidirectional_fusion='conv_plif', **options):
        super().__init__()
        if type(bidirectional) is not bool:
            raise ValueError('bidirectional must be boolean')
        self.bidirectional = bidirectional
        self.bidirectional_fusion = normalize_bidirectional_fusion(bidirectional_fusion, enabled=bidirectional)
        self.options = paper_options(options)
        opts = self.options
        validate_shared_soma_options(opts, enabled=bidirectional, fusion=self.bidirectional_fusion)
        self.bidirectional_blocks = bidirectional_blocks(opts, enabled=bidirectional)
        shared_soma = bidirectional and self.bidirectional_fusion == 'Shared_SOMA'
        e1, e2, e3 = opts['encoder_channels']
        d1, d2, d3 = opts['decoder_channels']
        for name, incoming, outgoing, up in (
            ('conv_block1', 4, e1, False), ('conv_block2', e1, e2, False),
            ('conv_block3', e2, e3, False), ('deconv_block1', e3, d1, True),
            ('deconv1_conv', d1, d1, False), ('concat1_conv', e2+d1, d1, False),
            ('deconv_block2', d1, d2, True), ('deconv2_conv', d2, d2, False),
            ('concat2_conv', e1+d2, d2, False), ('deconv_block3', d2, d3, True),
            ('deconv3_conv', d3, d3, False)):
            setattr(self, name, PaperBlock(
                name, incoming, outgoing, opts, deconv=up,
                shared_soma=shared_soma and name in self.bidirectional_blocks))
        self.class_conv = PaperBlock('class_conv', d3, out_channels, opts, head=True)
        self.pool = nn.MaxPool2d(**opts['pool'])
        self.multiple = opts['pool']['stride'] ** 3
        self._next_step = None
        if bidirectional and not shared_soma:
            self.bidirectional_fusions = nn.ModuleList(
                (PaperStageFusion(channels, opts['neurons']['plif'], opts['normalization'])
                 if self.bidirectional_fusion == 'conv_plif' else PaperORFusion())
                if f'conv_block{index}' in self.bidirectional_blocks else nn.Identity()
                for index, channels in enumerate((e1, e2, e3), 1))
        self.deep_supervision_enabled = opts['deep_supervision']['enabled']
        self.deep_supervision_resolution = opts['deep_supervision']['resolution']
        self.auxiliary_heads = nn.ModuleList()
        if self.deep_supervision_enabled:
            # Added heads must not change common weights or subsequent RNG draws.
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
            with torch.random.fork_rng(devices=devices):
                self.auxiliary_heads.extend((nn.Conv2d(d2, out_channels, 1),
                                             nn.Conv2d(d1, out_channels, 1)))

    def _segmentation_outputs(self, logits, auxiliary, spatial_shape):
        main = torch.stack(logits, dim=2)
        if auxiliary is None:
            return main
        if self.deep_supervision_resolution == 'native':
            return (main, *(torch.stack(slices, dim=2) for slices in auxiliary))
        h, w = spatial_shape
        padded = (h + (-h) % self.multiple, w + (-w) % self.multiple)
        outputs = [main]
        for slices in auxiliary:
            # Interpolate only H/W; never downsample labels or the slice axis.
            sequence = torch.stack(slices, dim=1)
            batch, steps, channels = sequence.shape[:3]
            resized = F.interpolate(sequence.flatten(0, 1), size=padded,
                                    mode='bilinear', align_corners=False)[..., :h, :w]
            outputs.append(resized.reshape(batch, steps, channels, h, w).transpose(1, 2))
        return tuple(outputs)

    def reset_states(self, *, preserve_diagnostics=False):
        activity = _activity_snapshots(self) if preserve_diagnostics else ()
        for module in self.modules():
            if isinstance(module, SelectiveMultiTimescaleLIF):
                module.reset()
            elif isinstance(module, PLIFNode):
                # PLIF's legacy forward uses a separate initialization flag.
                module.neuro_states_init = False
                for name in ('v', 's'):
                    if hasattr(module, name):
                        delattr(module, name)
        for cell, snapshot in activity:
            cell._activity = snapshot
        self._next_step = None

    def detach_states(self):
        for module in self.modules():
            if isinstance(module, (PLIFNode, SelectiveMultiTimescaleLIF)):
                module.detach()

    def recurrent_states(self):
        """Named live states, including selective memories, soma and readout."""
        result = {}
        for name, module in self.named_modules():
            keys = (module.state_names if isinstance(module, SelectiveMultiTimescaleLIF)
                    else ('v', 's') if isinstance(module, PLIFNode) else ())
            for key in keys:
                value = getattr(module, key, None)
                if isinstance(value, torch.Tensor):
                    result[f'{name}.{key}'] = value
        return result

    def _scan_stage(self, block, currents, *, reverse=False, apply_dropout=True):
        """Local ordinal advances 0..Z-1 in either anatomical direction."""
        _reset_neurons(block)
        outputs = [None] * len(currents)
        order = range(len(currents)-1, -1, -1) if reverse else range(len(currents))
        try:
            for ordinal, z in enumerate(order):
                outputs[z] = block.activate(currents[z], ordinal, apply_dropout=apply_dropout)
            return torch.stack(outputs, dim=1)
        finally:
            # Rebinding/removing live tensors does not detach either scan's graph.
            _reset_neurons(block, preserve_diagnostics=True)

    def _forward_bidirectional(self, x_win, *, return_auxiliary=False):
        h, w = x_win.shape[-2:]
        sequence = F.pad(x_win, (0, (-w) % self.multiple, 0, (-h) % self.multiple))
        stages = []
        for index in range(1, 4):
            block = getattr(self, f'conv_block{index}')
            # Exactly one ConvBlock per actual stage: its input is shared by
            # both directions. Later blocks with directional inputs cannot cache.
            currents = tuple(block.project(sequence[:, z]) for z in range(sequence.shape[1]))
            if f'conv_block{index}' not in self.bidirectional_blocks:
                spikes = self._scan_stage(block, currents)
                sequence = self.pool(spikes.flatten(0, 1)).unflatten(0, spikes.shape[:2])
                stages.append(sequence)
                continue
            if self.bidirectional_fusion == 'Shared_SOMA':
                spikes = block.spike_neurons(torch.stack(currents, dim=1))
                sequence = self.pool(spikes.flatten(0, 1)).unflatten(0, spikes.shape[:2])
                if block.dropout:
                    sequence = F.dropout(sequence, block.dropout, self.training)
                stages.append(sequence)
                continue
            fusion = self.bidirectional_fusions[index-1]
            directional_dropout = self.bidirectional_fusion == 'conv_plif'
            forward = self._scan_stage(block, currents, apply_dropout=directional_dropout)
            cell = block.spike_neurons
            record = getattr(cell, 'collect_diagnostics', False)
            forward_stats = dict(cell._activity) if record else {}
            backward = self._scan_stage(block, currents, reverse=True, apply_dropout=directional_dropout)
            if record:
                cell._activity = {
                    **{f'forward/{key}': value for key, value in forward_stats.items()},
                    **{f'backward/{key}': value for key, value in cell._activity.items()}}
            # Pool each aligned direction before fusion. The same fused output
            # feeds the pooled skip/bottleneck and the next encoder stage.
            forward = self.pool(forward.flatten(0, 1)).unflatten(0, forward.shape[:2])
            backward = self.pool(backward.flatten(0, 1)).unflatten(0, backward.shape[:2])
            sequence = fusion(forward, backward)
            if self.bidirectional_fusion == 'or' and block.dropout:
                # OR itself remains binary. Standard inverted dropout is applied
                # once afterwards, as in the agreed training architecture.
                sequence = F.dropout(sequence, block.dropout, self.training)
            stages.append(sequence)
        logits = []
        auxiliary = ([], []) if return_auxiliary else None
        pool1, pool2, bottleneck = stages
        for z in range(x_win.shape[1]):
            x = self.deconv1_conv(self.deconv_block1(bottleneck[:, z], z), z)
            x = self.concat1_conv(torch.cat((pool2[:, z], x), dim=1), z)
            if auxiliary is not None:
                auxiliary[1].append(self.auxiliary_heads[1](x))
            x = self.deconv2_conv(self.deconv_block2(x, z), z)
            x = self.concat2_conv(torch.cat((pool1[:, z], x), dim=1), z)
            if auxiliary is not None:
                auxiliary[0].append(self.auxiliary_heads[0](x))
            x = self.deconv3_conv(self.deconv_block3(x, z), z)
            logits.append(self.class_conv(x, z)[..., :h, :w])
        return self._segmentation_outputs(logits, auxiliary, (h, w))

    @fp32_forward
    def forward(self, x_win, t0=0, *, return_auxiliary=False):
        if x_win.ndim != 5 or x_win.shape[2] != 4 or min(x_win.shape) <= 0:
            raise ValueError('expected nonempty [B,T,4,H,W] input')
        if type(t0) is not int or t0 < 0:
            raise ValueError('t0 must be a nonnegative integer')
        if type(return_auxiliary) is not bool:
            raise ValueError('return_auxiliary must be boolean')
        if return_auxiliary and not self.deep_supervision_enabled:
            raise ValueError('return_auxiliary requires deep_supervision.enabled')
        if self.bidirectional:
            if t0 != 0:
                raise ValueError('bidirectional mode requires a whole independent patch with t0=0')
            self.reset_states()
            try:
                return self._forward_bidirectional(x_win, return_auxiliary=return_auxiliary)
            finally:
                self.reset_states(preserve_diagnostics=True)
        if t0 == 0:
            self.reset_states()
        elif self._next_step != t0:
            raise ValueError('Expected consecutive windows; start new sequences with t0=0')
        h, w = x_win.shape[-2:]
        ph, pw = (-h) % self.multiple, (-w) % self.multiple
        logits = []
        auxiliary = ([], []) if return_auxiliary else None
        for i in range(x_win.shape[1]):
            t = t0+i
            x = F.pad(x_win[:, i], (0, pw, 0, ph))
            pool1 = self.pool(self.conv_block1(x, t))
            pool2 = self.pool(self.conv_block2(pool1, t))
            x = self.pool(self.conv_block3(pool2, t))
            x = self.deconv1_conv(self.deconv_block1(x, t), t)
            x = self.concat1_conv(torch.cat((pool2, x), dim=1), t)
            if auxiliary is not None:
                auxiliary[1].append(self.auxiliary_heads[1](x))
            x = self.deconv2_conv(self.deconv_block2(x, t), t)
            x = self.concat2_conv(torch.cat((pool1, x), dim=1), t)
            if auxiliary is not None:
                auxiliary[0].append(self.auxiliary_heads[0](x))
            x = self.deconv3_conv(self.deconv_block3(x, t), t)
            logits.append(self.class_conv(x, t)[..., :h, :w])
        self._next_step = t0+x_win.shape[1]
        return self._segmentation_outputs(logits, auxiliary, (h, w))
