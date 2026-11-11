"""Independent PaperUNet operation ledger; optional synthetic activity/CUDA probes.

Default invocation never executes a full-size CPU forward. Counts are forward
convolution MACs (one multiply/accumulate pair = one MAC), not FLOPs or energy.
The model itself is not instrumented or modified by the analytical path.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import torch
from torch import nn

from snn_nnunet.paper_config import BLOCK_NAMES
from snn_nnunet.paper_unet import PaperUNet
from snn_nnunet.recurrent_neuron import SelectiveMultiTimescaleLIF
from snn_nnunet.spike_neurons import PLIFNode


def _product(values):
    return math.prod(int(v) for v in values)


def _output_spatial(layer, spatial):
    if isinstance(layer, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
        return tuple((n-1)*s-2*p+d*(k-1)+op+1 for n, s, p, d, k, op in zip(
            spatial, layer.stride, layer.padding, layer.dilation,
            layer.kernel_size, layer.output_padding))
    return tuple((n+2*p-d*(k-1)-1)//s+1 for n, s, p, d, k in zip(
        spatial, layer.stride, layer.padding, layer.dilation, layer.kernel_size))


def convolution_macs(layer, input_shape):
    """Dense Conv2/3d count; transpose counts only valid emitted products.

    Ordinary convolution uses the standard padded-kernel convention, including
    multiplications against padding zeros. Transpose counts scatter products
    actually landing in its output, rather than output_sites*full_kernel.
    """
    spatial = input_shape[2:]
    output = _output_spatial(layer, spatial)
    channel_pairs = layer.in_channels*layer.out_channels//layer.groups
    if isinstance(layer, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
        valid = [sum(0 <= i*s-p+j*d < out for i in range(n) for j in range(k))
                 for n, s, p, d, k, out in zip(spatial, layer.stride,
                    layer.padding, layer.dilation, layer.kernel_size, output)]
        return input_shape[0]*channel_pairs*_product(valid)
    return input_shape[0]*channel_pairs*_product(output)*_product(layer.kernel_size)


def selective_cost(cell, sites):
    """Continuous operations per actual SSN call times B*Z*H*W sites.

    The functional F.conv2d gates/projection would be missed by child-module
    forward hooks. All three memories are continuous, even if currently zero.
    Scalar multiply/add fields cover the tensor memory mixing/update/readout;
    coefficient functions, normalization and somatic PLIF are separate/excluded.
    """
    channels = cell.channels
    depthwise = (sites*sum(kernel.in_channels*_product(kernel.kernel_size)
                          for kernel in cell.spatial_kernels))
    gates = sites*sum(gate.in_channels*gate.out_channels//gate.groups
                     *_product(gate.kernel_size) for gate in cell.readout_gates)
    projection = sites*cell.memory_projection.in_channels*cell.memory_projection.out_channels
    compartments = len(cell.state_names)
    if getattr(cell, 'is_linear_attention', False):
        from snn_nnunet.profile_linear_attention import linear_attention_cost
        attention = linear_attention_cost(channels, sites)
        attention_macs = sum(value for name, value in attention.items() if name.endswith('_macs'))
        return {
            'spatial_depthwise_macs': depthwise, 'readout_gate_macs': 0,
            'memory_projection_macs': projection,
            'continuous_linear_macs': depthwise+projection+attention_macs,
            'elementwise_multiplications': sites*channels*compartments*(4 if cell.spatial_enabled else 2),
            'elementwise_additions': sites*channels*(compartments*(2 if cell.spatial_enabled else 1)+1),
            'gate_bias_additions': 0, 'gate_sigmoid_elements': 0,
            'ssn_spatial_update_sites': sites, **attention,
        }
    return {
        'spatial_depthwise_macs': depthwise,
        'readout_gate_macs': gates,
        'memory_projection_macs': projection,
        'continuous_linear_macs': depthwise+gates+projection,
        'elementwise_multiplications': sites*channels*compartments*(5 if cell.spatial_enabled else 3),
        'elementwise_additions': sites*channels*(compartments*(3 if cell.spatial_enabled else 2)+1),
        'gate_bias_additions': sites*sum(gate.out_channels for gate in cell.readout_gates
                                         if gate.bias is not None),
        'gate_sigmoid_elements': sites*channels*compartments if cell.selective_enabled else 0,
        'ssn_spatial_update_sites': sites,
    }


def parameter_formula(model):
    """Independent formulas for this fixed graph; checked against parameters()."""
    opts = model.options
    e1, e2, e3 = opts['encoder_channels']
    d1, d2, d3 = opts['decoder_channels']
    out_channels = model.class_conv.conv.out_channels
    topology = [(4, e1, False), (e1, e2, False), (e2, e3, False),
                (e3, d1, True), (d1, d1, False), (e2+d1, d1, False),
                (d1, d2, True), (d2, d2, False), (e1+d2, d2, False),
                (d2, d3, True), (d3, d3, False)]
    total = 0
    for name, (incoming, outgoing, up) in zip(BLOCK_NAMES, topology):
        block = opts['deconv' if up else 'conv']
        total += incoming*outgoing*block['kernel_size']**2
        total += outgoing if block['bias'] else 0
        total += 2*outgoing if opts['normalization']['affine'] else 0
        neuron = getattr(model, name).spike_neurons
        if isinstance(neuron, SelectiveMultiTimescaleLIF):
            # Temporal 3C; spatial eta 3C + DW weights 27C; gate 6C²+3C
            # or static 3C; projection C²; somatic scalar PLIF.
            total += 3*outgoing+outgoing**2+1
            if getattr(neuron, 'is_shared_soma', False):
                total += 6*outgoing**2+3*outgoing
            if neuron.spatial_enabled:
                total += 30*outgoing
            if getattr(neuron, 'is_linear_attention', False):
                total += 256*outgoing+96
                if neuron.gamma is not None:
                    total += outgoing  # Independent LayerScale coefficient per feature channel.
            else:
                total += (6*outgoing**2+3*outgoing if neuron.selective_enabled else 3*outgoing)
        elif isinstance(neuron, PLIFNode):
            total += 1
    total += d3*out_channels*opts['head']['kernel_size']**2
    total += out_channels if opts['head']['bias'] else 0
    total += 1  # Continuous voltage head has a learned PLIF time constant.
    if getattr(model, 'bidirectional', False) and model.bidirectional_fusion == 'conv_plif':
        total += sum(2*c*c+1 + (2*c if opts['normalization']['affine'] else 0)
                     for index, c in enumerate((e1, e2, e3), 1)
                     if f'conv_block{index}' in model.bidirectional_blocks)
    return total


def estimate_cost(model, shape=(81, 128, 128), batch=1):
    """Forward analytical ledger for [B,Z,4,H,W]; BPTT backward not estimated."""
    if len(shape) != 3 or any(type(v) is not int or v < 1 for v in shape) or batch < 1:
        raise ValueError('shape must be positive integer Z,H,W and batch must be positive')
    z, h, w = shape
    h, w = (math.ceil(n/model.multiple)*model.multiple for n in (h, w))
    bi = getattr(model, 'bidirectional', False)
    fusion_method = model.bidirectional_fusion if bi else None
    shared_soma = bi and fusion_method == 'Shared_SOMA'
    layers, ssn = [], []
    plif_sites = 0
    or_comparisons = 0

    def record(name, incoming_shape, *, repeats=1):
        nonlocal plif_sites
        block = getattr(model, name)
        layer = block.deconv if block.is_deconv else block.conv
        out = _output_spatial(layer, incoming_shape[2:])
        macs = convolution_macs(layer, incoming_shape)*z
        if name == 'conv_block1':
            spiking_input = False
        elif name in ('conv_block2', 'conv_block3', 'deconv_block1'):
            spiking_input = model.options['encoder_spiking']
        elif name in ('concat1_conv', 'concat2_conv'):
            spiking_input = model.options['encoder_spiking'] and model.options['decoder_spiking']
        else:
            spiking_input = model.options['decoder_spiking']
        kind = 'spike_eligible' if spiking_input else 'continuous_input'
        neuron_calls = 0 if isinstance(block.spike_neurons, nn.GELU) else z*repeats
        layers.append({'name': name, 'kind': kind, 'macs': macs,
                       'input_bchw': list(incoming_shape), 'output_hw': list(out),
                       'convolution_calls': z, 'neuron_calls': neuron_calls})
        sites = batch*z*_product(out)*repeats
        if not isinstance(block.spike_neurons, nn.GELU):
            plif_sites += sites*layer.out_channels
        if isinstance(block.spike_neurons, SelectiveMultiTimescaleLIF):
            cell = block.spike_neurons
            if getattr(cell, 'is_shared_soma', False):
                cost = selective_cost(cell, sites)
                # A second memory trajectory, but no second readout/projection/soma.
                depthwise = cost['spatial_depthwise_macs']
                fusion = sites*6*cell.channels**2
                cost['spatial_depthwise_macs'] += depthwise
                cost['continuous_linear_macs'] += depthwise+fusion
                cost['memory_fusion_macs'] = fusion
                cost['memory_fusion_bias_additions'] = sites*3*cell.channels
                cost['elementwise_multiplications'] += sites*cell.channels*3*(4 if cell.spatial_enabled else 2)
                cost['elementwise_additions'] += sites*cell.channels*3*(2 if cell.spatial_enabled else 1)
                cost['ssn_spatial_update_sites'] *= 2
                ssn.append({'name': name, 'directional_repeats': 2, 'ssn_calls': 1,
                            'memory_trajectory_calls': 2*z, 'readout_calls': z, 'soma_calls': z, **cost})
            else:
                ssn.append({'name': name, 'directional_repeats': repeats, 'ssn_calls': z*repeats,
                            **selective_cost(cell, sites)})
        return (batch, layer.out_channels, *out)

    spatial = (batch, 4, h, w)
    skips = []
    pool = model.pool
    for index in range(3):
        active = f'conv_block{index+1}' in model.bidirectional_blocks
        spatial = record(f'conv_block{index+1}', spatial, repeats=2 if active and not shared_soma else 1)
        kernel = pool.kernel_size if isinstance(pool.kernel_size, int) else pool.kernel_size[0]
        stride = pool.stride if isinstance(pool.stride, int) else pool.stride[0]
        padding = pool.padding if isinstance(pool.padding, int) else pool.padding[0]
        spatial = (*spatial[:2], *((n+2*padding-kernel)//stride+1 for n in spatial[2:]))
        if active and fusion_method == 'conv_plif':
            fusion = model.bidirectional_fusions[index]
            fusion_input = (batch, spatial[1]*2, *spatial[2:])
            layers.append({'name': f'bidirectional_fusions.{index}',
                           'kind': 'spike_eligible',
                           'macs': convolution_macs(fusion.conv, fusion_input)*z,
                           'input_bchw': list(fusion_input), 'output_hw': list(spatial[2:]),
                           'convolution_calls': z, 'neuron_calls': z})
            plif_sites += batch*z*_product(spatial[1:])
        elif active and not shared_soma:
            or_comparisons += batch*z*_product(spatial[1:])
        skips.append(spatial)
    for index in range(1, 4):
        spatial = record(f'deconv_block{index}', spatial)
        spatial = record(f'deconv{index}_conv', spatial)
        if index < 3:
            skip = skips[2-index]
            spatial = record(f'concat{index}_conv',
                             (batch, spatial[1]+skip[1], *spatial[2:]))
    record('class_conv', spatial)
    continuous = sum(row['macs'] for row in layers if row['kind'] == 'continuous_input')
    eligible = sum(row['macs'] for row in layers if row['kind'] == 'spike_eligible')
    ssn_linear = sum(row['continuous_linear_macs'] for row in ssn)
    parameters = sum(p.numel() for p in model.parameters())
    formula = parameter_formula(model)
    if parameters != formula:
        raise ValueError(f'Parameter formula {formula} differs from actual model {parameters}')
    return {
        'shape_zyx': list(shape), 'padded_shape_zyx': [z, h, w], 'batch': batch,
        'bidirectional': bi, 'bidirectional_fusion': fusion_method,
        'binary_or_comparisons': or_comparisons,
        'actual_parameters': parameters, 'parameter_formula': formula,
        'dense_equivalent_macs': continuous+eligible+ssn_linear,
        'continuous_input_macs': continuous, 'continuous_ssn_linear_macs': ssn_linear,
        'spike_eligible_dense_macs': eligible,
        'theoretical_ac_at_unit_activity': eligible,
        'theoretical_ac_at_10_percent_activity': eligible*.1,
        'ssn_elementwise_multiplications': sum(r['elementwise_multiplications'] for r in ssn),
        'ssn_elementwise_additions': sum(r['elementwise_additions'] for r in ssn),
        'plif_update_elements_including_continuous_head': plif_sites,
        'layers': layers, 'ssn': ssn,
        'scope': 'forward only; batch=2 is training forward, excludes backward/optimizer',
    }


def theoretical_3d_comparator(model, shape=(81, 128, 128), batch=1):
    """Conventional all-continuous 3D CNN matching widths and pooled skips.

    Three isotropic pool stages (2,2,2), 3D transposed upconvs, SAME padded
    convolutions and GroupNorm/ReLU replace neurons; raw 3D head. Pad Z/Y/X
    to a multiple of eight, crop logits. This is NOT an nnU-Net benchmark.
    """
    dims = tuple(math.ceil(n/model.multiple)*model.multiple for n in shape)
    spatial = (batch, 4, *dims)
    layers, skips = [], []
    params = 0

    def record(name, incoming):
        nonlocal params
        block = getattr(model, name)
        old = block.deconv if block.is_deconv else block.conv
        conv = nn.ConvTranspose3d if block.is_deconv else nn.Conv3d
        # Meta device avoids allocating/computing a comparator model.
        layer = conv(old.in_channels, old.out_channels, old.kernel_size[0],
                     stride=old.stride[0], padding=old.padding[0],
                     bias=old.bias is not None, device='meta',
                     **({'output_padding': old.output_padding[0]} if block.is_deconv else {}))
        output = _output_spatial(layer, incoming[2:])
        macs = convolution_macs(layer, incoming)
        params += old.in_channels*old.out_channels*_product(layer.kernel_size)
        params += old.out_channels if old.bias is not None else 0
        if not block.is_head and model.options['normalization']['affine']:
            params += 2*old.out_channels
        layers.append({'name': name, 'macs': macs, 'kind': 'continuous',
                       'input_bczhw': list(incoming), 'output_zyx': list(output)})
        return (batch, old.out_channels, *output)

    scale = model.options['pool']['stride']
    for i in range(1, 4):
        spatial = record(f'conv_block{i}', spatial)
        spatial = (*spatial[:2], *(n//scale for n in spatial[2:]))
        skips.append(spatial)
    for i in range(1, 4):
        spatial = record(f'deconv_block{i}', spatial)
        spatial = record(f'deconv{i}_conv', spatial)
        if i < 3:
            spatial = record(f'concat{i}_conv',
                             (batch, spatial[1]+skips[2-i][1], *spatial[2:]))
    record('class_conv', spatial)
    return {'shape_zyx': list(shape), 'padded_shape_zyx': list(dims), 'batch': batch,
            'conv_kernel_zyx': [model.options['conv']['kernel_size']]*3,
            'head_kernel_zyx': [model.options['head']['kernel_size']]*3,
            'pool_kernel_zyx': [scale]*3,
            'deconv_kernel_zyx': [model.options['deconv']['kernel_size']]*3,
            'parameters_theoretical': params, 'dense_equivalent_macs': sum(r['macs'] for r in layers),
            'spike_eligible_dense_macs': 0, 'layers': layers,
            'scope': 'theoretical continuous 3D CNN forward, not measured nnU-Net'}


def activity_record(layer, x):
    """Density weighted estimate, not an exact boundary/channel weighted AC.

    Binaryness is inspected rather than inferred from the layer name. Nonzero
    training dropout values require scaled arithmetic and are never labeled AC.
    """
    x = x.detach()
    binary = bool(((x == 0) | (x == 1)).all())
    active = int(torch.count_nonzero(x))
    dense = convolution_macs(layer, tuple(x.shape))
    estimate = dense*active/x.numel()
    return {'dense_macs': dense, 'nonzero_inputs': active, 'input_elements': x.numel(),
            'binary_input': binary, 'nonzero_fraction': active/x.numel(),
            'activity_based_ac_estimate': estimate if binary else None,
            'nonzero_weighted_operation_estimate': estimate}


def _fresh_probe(model, device, training):
    # A deep copy of live recurrent states can fail for non-leaf autograd
    # tensors. Reconstruct the exact fixed graph and copy checkpoint weights;
    # source states, user hooks and live graphs are never touched.
    probe = PaperUNet(out_channels=model.class_conv.conv.out_channels,
                      bidirectional=getattr(model, 'bidirectional', False),
                      bidirectional_fusion=model.bidirectional_fusion,
                      **deepcopy(model.options)).to(device).train(training)
    probe.load_state_dict(model.state_dict())
    return probe


def measure_activity(model, inputs, *, training=False):
    """Probe a weight-identical fresh model; remove every hook even on failure.

    Calling this has no effect on the source model's states, modes, weights,
    hooks or autograd graph. Input is user supplied [B,Z,4,H,W]. Continuous
    SSN internals remain analytic, because their F.conv2d bypasses module hooks.
    """
    devices = [inputs.device.index] if inputs.is_cuda else []
    convolutions, neuron_calls, neuron_activity, hooks = {}, {}, {}, []
    with torch.random.fork_rng(devices=devices):
        probe = _fresh_probe(model, inputs.device, training)
        def conv_hook(name):
            def hook(module, args, output):
                record = activity_record(module, args[0])
                previous = convolutions.get(name)
                if previous:
                    for key in ('dense_macs', 'nonzero_inputs', 'input_elements',
                                'nonzero_weighted_operation_estimate'):
                        record[key] += previous[key]
                    record['binary_input'] &= previous['binary_input']
                    record['nonzero_fraction'] = record['nonzero_inputs']/record['input_elements']
                    record['activity_based_ac_estimate'] = (record['nonzero_weighted_operation_estimate']
                                                           if record['binary_input'] else None)
                convolutions[name] = record
            return hook
        def neuron_hook(name):
            def hook(module, args, output):
                neuron_calls[name] = neuron_calls.get(name, 0)+1
                value = (output[0] if isinstance(output, tuple) else output).detach()
                record = neuron_activity.setdefault(name, {
                    'representation': 'continuous_voltage' if getattr(module, 'no_spiking', False) else 'spikes_before_dropout',
                    'elements': 0, 'nonzero': 0, 'sum': 0.0})
                record['elements'] += value.numel()
                record['nonzero'] += int(torch.count_nonzero(value))
                record['sum'] += float(value.sum())
            return hook
        try:
            for name, module in probe.named_modules():
                # Functional SSN internals are already accounted analytically.
                internal = any(word in name for word in ('.spatial_kernels.', '.readout_gates.', '.memory_projection', '.attention.'))
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)) and not internal:
                    hooks.append(module.register_forward_hook(conv_hook(name)))
                if isinstance(module, (PLIFNode, SelectiveMultiTimescaleLIF)):
                    hooks.append(module.register_forward_hook(neuron_hook(name)))
            with torch.no_grad():
                output = probe(inputs, t0=0)
            output_shape = list(output.shape)
        finally:
            for handle in hooks:
                handle.remove()
        for record in neuron_activity.values():
            record['nonzero_fraction'] = record['nonzero']/record['elements']
            record['mean'] = record['sum']/record['elements']
            record['firing_rate'] = (record['mean'] if record['representation'] == 'spikes_before_dropout' else None)
        return {'training': training, 'input_shape_bzchw': list(inputs.shape),
                'output_shape_bczyx': output_shape, 'convolutions': convolutions,
                'neuron_calls': neuron_calls, 'neuron_activity': neuron_activity,
                'scope': 'observed input activity on supplied data; density-weighted AC estimate only'}


def measure_cuda(model, shape, *, batch=1, training=False, warmup=3, repeats=10,
                 input_kind='patch'):
    """Optional FP32 synthetic latency/peak allocated memory; no OOM fallback.

    Training includes forward + mean-square logits loss + backward, no optimizer.
    Inference seconds/volume is reported only when caller identifies a full
    volume. Patch timings are never relabeled as sliding-window volume timings.
    """
    if not torch.cuda.is_available():
        return {'status': 'unmeasured: CUDA unavailable', 'input_kind': input_kind}
    if warmup < 1 or repeats < 1:
        raise ValueError('warmup and repeats must be positive')
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        baseline_allocated = torch.cuda.memory_allocated()
        probe = _fresh_probe(model, torch.device('cuda'), training)
        x = torch.randn(batch, shape[0], 4, shape[1], shape[2], device='cuda')
        def step():
            probe.zero_grad(set_to_none=True)
            if training:
                y = probe(x, t0=0)
                y.square().mean().backward()
            else:
                with torch.no_grad():
                    probe(x, t0=0)
        for _ in range(warmup):
            step()
        probe.zero_grad(set_to_none=True)
        probe.reset_states()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(repeats):
            step()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter()-start)/repeats
        return {'status': 'measured', 'device': torch.cuda.get_device_name(),
                'dtype': 'float32', 'training': training, 'batch': batch,
                'input_kind': input_kind, 'shape_zyx': list(shape),
                'warmup': warmup, 'repeats': repeats,
                'seconds_per_batch': elapsed, 'seconds_per_sample': elapsed/batch,
                'inference_seconds_per_volume': (elapsed/batch if input_kind == 'full-volume' and not training else None),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'baseline_allocated_bytes': baseline_allocated,
                'peak_allocated_above_baseline_bytes': torch.cuda.max_memory_allocated()-baseline_allocated,
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'scope': 'synthetic forward or forward+backward; no optimizer, no sliding-window pipeline'}


def build_variants():
    overrides = {f'conv_block{i}': 'selective_multitimescale_lif' for i in (1, 2, 3)}
    return {'original_plif': PaperUNet(),
            'selective_unidirectional': PaperUNet(neurons={'overrides': overrides}),
            'selective_bidirectional': PaperUNet(bidirectional=True, neurons={'overrides': overrides}),
            'selective_bidirectional_or': PaperUNet(bidirectional=True, bidirectional_fusion='or',
                                                     neurons={'overrides': overrides})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape', nargs=3, type=int, default=(81, 128, 128), metavar=('Z', 'H', 'W'))
    parser.add_argument('--output', type=Path, default=Path('analysis_results/bidirectional_costs/profile.json'))
    parser.add_argument('--activity-smoke', action='store_true', help='synthetic B1,Z3,H8,W8 eval/train probe only')
    parser.add_argument('--measure-cuda', action='store_true')
    parser.add_argument('--input-kind', choices=('patch', 'full-volume'), default='patch')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        variants = build_variants()
        report = {'torch_version': torch.__version__, 'cuda_available': torch.cuda.is_available(),
                  'seed': 42, 'input_kind': args.input_kind,
                  'count_convention': 'forward convolution MACs; one MAC=one multiply+accumulate; ordinary conv counts padding taps, transpose counts valid emitted products',
                  'excluded_from_convolution_macs': ['GroupNorm', 'pool comparisons', 'OR comparisons (separate field)', 'dropout', 'PLIF charge/reset/threshold',
                    'SSN scalar arithmetic (separate fields)', 'coefficient softplus/exp/sigmoid', 'kernel L1 normalization',
                    'gate bias/sigmoid (separate SSN fields)', 'concatenation/padding/data movement', 'backward and optimizer'],
                  'variants': {}}
        for name, model in variants.items():
            result = {'inference_b1': estimate_cost(model, tuple(args.shape), batch=1),
                      'training_forward_b2': estimate_cost(model, tuple(args.shape), batch=2),
                      'gpu_inference': {'status': 'unmeasured: not requested' if torch.cuda.is_available() else 'unmeasured: CUDA unavailable'},
                      'gpu_training': {'status': 'unmeasured: not requested' if torch.cuda.is_available() else 'unmeasured: CUDA unavailable'}}
            if args.activity_smoke:
                inputs = torch.randn(1, 3, 4, 8, 8)
                result['synthetic_tiny_eval_activity'] = measure_activity(model, inputs)
                result['synthetic_tiny_train_activity'] = measure_activity(model, inputs, training=True)
            if args.measure_cuda:
                result['gpu_inference'] = measure_cuda(model, tuple(args.shape), input_kind=args.input_kind,
                    warmup=args.warmup, repeats=args.repeats)
                result['gpu_training'] = measure_cuda(model, tuple(args.shape), batch=2, training=True,
                    input_kind=args.input_kind, warmup=args.warmup, repeats=args.repeats)
            report['variants'][name] = result
        report['theoretical_3d_comparator'] = theoretical_3d_comparator(variants['original_plif'], tuple(args.shape))
        report['theoretical_3d_comparator_training_forward_b2'] = theoretical_3d_comparator(variants['original_plif'], tuple(args.shape), batch=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Wrote {args.output}; full-size counts are analytical; CUDA available: {report["cuda_available"]}')


if __name__ == '__main__':
    main()
