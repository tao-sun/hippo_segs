"""Equivalent three-mode analytical costs and explicitly scoped synthetic timing."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import time

import torch

from snn_nnunet.paper_unet import PaperUNet
from snn_nnunet.profile_paper_unet import estimate_cost, measure_cuda


def state_counts(model):
    """Logical concurrent scan states; retained BPTT trajectories are separate."""
    mode = model.bidirectional_fusion
    rows = []
    for index, channels in enumerate(model.options['encoder_channels'], 1):
        rows.append({'stage': index, 'channels': channels,
                     'directional_dendritic_memory_maps': 6,
                     'directional_soma_voltage_maps': 0 if mode=='Shared_SOMA' else 2,
                     'shared_forward_soma_voltage_maps': 1 if mode=='Shared_SOMA' else 0,
                     'slice_local_fusion_soma_voltage_maps': 1 if mode=='conv_plif' else 0,
                     'soma_cached_spike_maps': 1 if mode=='Shared_SOMA' else 3 if mode=='conv_plif' else 2,
                     'soma_resolution': 'directional before pooling; fusion after pooling' if mode=='conv_plif' else 'before pooling'})
    return {'encoder_stages': rows,
            'decoder_and_head': 'unchanged; one forward-recurrent soma per original decoder/head block',
            'scope': 'logical states across two scans; legacy directional states are reused sequentially; excludes saved activations, fused tensors and autograd graph storage'}


def measure_cpu(model, inputs, warmup, repeats):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup): model(inputs)
        durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(inputs)
            durations.append(time.perf_counter()-start)
    return {'status': 'measured', 'device': 'cpu', 'dtype': 'float32',
            'threads': torch.get_num_threads(), 'shape_bzchw': list(inputs.shape),
            'warmup': warmup, 'repeats': repeats,
            'median_seconds_per_batch': statistics.median(durations),
            'min_seconds_per_batch': min(durations), 'max_seconds_per_batch': max(durations),
            'scope': 'tiny synthetic patch forward only, same input/widths/settings; not full-volume or full-size patch latency'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape', type=int, nargs=3, default=[81,128,128], metavar=('Z','H','W'))
    parser.add_argument('--timing-shape', type=int, nargs=3, default=[5,32,32], metavar=('Z','H','W'))
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--measure-cuda', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('analysis_results/shared_soma/report.json'))
    args = parser.parse_args(argv)
    if args.repeats<1 or args.warmup<0 or any(n<1 for n in args.shape+args.timing_shape):
        parser.error('positive shapes/repeats and nonnegative warmup required')
    if args.measure_cuda and args.warmup<1:
        parser.error('--measure-cuda requires --warmup >= 1')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    overrides = {f'conv_block{i}':'selective_multitimescale_lif' for i in (1,2,3)}
    variants = {}
    for mode in ('or','conv_plif','Shared_SOMA'):
        torch.manual_seed(42)
        variants[mode] = PaperUNet(bidirectional=True,bidirectional_fusion=mode,
                                  neurons={'overrides':overrides}).eval()
    # Align all common learned tensors, not merely the initialization seed.
    common = variants['or'].state_dict()
    for model in variants.values():
        own = model.state_dict()
        own.update({name:value for name,value in common.items() if name in own})
        model.load_state_dict(own,strict=True)
    torch.manual_seed(123)
    inputs = torch.randn(1,args.timing_shape[0],4,*args.timing_shape[1:])
    report = {'seed':42,'torch':str(torch.__version__),'python':platform.python_version(),
              'cpu_platform':platform.platform(),'cuda_available':torch.cuda.is_available(),
              'count_convention':'one MAC = one multiply-accumulate; all counts forward only',
              'excluded_from_macs':['GroupNorm','pooling/OR comparisons','PLIF charge/reset/surrogate',
                  'coefficient functions','kernel normalization','bias and scalar arithmetic (separate fields)',
                  'dropout/data movement','backward/optimizer'],
              'variants':{}}
    for mode,model in variants.items():
        row = {'analytical_inference_b1':estimate_cost(model,tuple(args.shape)),
               'analytical_training_forward_b2':estimate_cost(model,tuple(args.shape),batch=2),
               'logical_states':state_counts(model),
               'cpu_inference':measure_cpu(model,inputs,args.warmup,args.repeats)}
        status = 'unmeasured: CUDA unavailable' if not torch.cuda.is_available() else 'unmeasured: not requested'
        row['gpu_inference'] = {'status':status}
        row['gpu_training'] = {'status':status}
        if args.measure_cuda:
            row['gpu_inference'] = measure_cuda(model,tuple(args.shape),warmup=args.warmup,repeats=args.repeats)
            row['gpu_training'] = measure_cuda(model,tuple(args.shape),batch=2,training=True,warmup=args.warmup,repeats=args.repeats)
        report['variants'][mode] = row
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Wrote {args.output}')
    for mode,row in report['variants'].items():
        costs=row['analytical_inference_b1']
        print(mode, costs['actual_parameters'], f"{costs['dense_equivalent_macs']/1e9:.6f} G MACs",
              f"{row['cpu_inference']['median_seconds_per_batch']*1000:.3f} ms tiny CPU")


if __name__=='__main__':
    main()
