"""Measure inference MACs from a saved SMMLIF log, including functional convs.

Run from the repository root. Writes analysis output only; training is untouched.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from snn_nnunet.network_adapter import SNNConfig, SNNnnUNetAdapter
from snn_nnunet.profile_paper_unet import estimate_cost


class OperationAudit(TorchDispatchMode):
    """Execute original ops; record each ATen convolution exactly once.

    Module hooks supply scope only. Counts are independent of the analytical
    ledger, include functional F.conv2d, and apply no activity/sparsity discount.
    """
    def __init__(self):
        super().__init__()
        self.scopes, self.convolutions, self.non_mac = [], {}, {}

    def enter(self, name):
        def hook(module, args):
            self.scopes.append(name)
        return hook

    def leave(self, module, args, output):
        self.scopes.pop()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        scope = self.scopes[-1] if self.scopes else ''
        op = str(func)
        if op == 'aten.convolution.default':
            x, weight, bias, stride, padding, dilation, transposed, output_padding, groups = args
            kernel = tuple(weight.shape[2:])
            if transposed:
                valid = [sum(0 <= i*s-p+j*d < out for i in range(n) for j in range(k))
                         for n, s, p, d, k, out in zip(x.shape[2:], stride, padding,
                                                     dilation, kernel, result.shape[2:])]
                macs = x.shape[0]*x.shape[1]*result.shape[1]//groups*math.prod(valid)
            else:
                macs = result.numel()*weight.shape[1]*math.prod(kernel)
            if '.spike_neurons' in scope:
                name = scope.split('.spike_neurons')[0]
                kind = ('spatial_depthwise' if groups == result.shape[1] and kernel == (3, 3)
                        else 'readout_gate' if weight.shape[1] == 2*result.shape[1]
                        else 'memory_projection')
                name += '.ssn.' + kind
            else:
                name, kind = scope, 'backbone'
            row = self.convolutions.setdefault(name, dict(kind=kind, calls=0, macs=0,
                input_shape=list(x.shape), output_shape=list(result.shape),
                weight_shape=list(weight.shape), groups=groups, transposed=transposed,
                bias_additions=0))
            assert row['input_shape'] == list(x.shape)
            row['calls'] += 1
            row['macs'] += macs
            row['bias_additions'] += result.numel() if bias is not None else 0
        elif '.spike_neurons' in scope:
            # Output element counts are recorded, not converted indiscriminately
            # to FLOPs. Reductions and movement remain identifiable by op name.
            section = ('soma' if '.soma' in scope else 'kernel_normalization'
                       if '.spatial_kernels.' in scope else 'memory_and_coefficients')
            key = scope.split('.spike_neurons')[0] + '.' + section
            row = self.non_mac.setdefault(key, {})
            stat = row.setdefault(op, dict(calls=0, output_elements=0, input_elements=0))
            values = result if isinstance(result, (tuple, list)) else (result,)
            stat['calls'] += 1
            stat['output_elements'] += sum(v.numel() for v in values if isinstance(v, torch.Tensor))
            stat['input_elements'] += sum(v.numel() for v in args if isinstance(v, torch.Tensor))
        return result


def read_logged_plans(path):
    for line in path.read_text().splitlines():
        if line.strip().startswith("{'dataset_name':"):
            return ast.literal_eval(line.strip())
    raise ValueError('No global plans dictionary found in log')


def volume_costs(summary_path, data_dir, patch, patch_macs):
    import blosc2
    from nnunetv2.inference.sliding_window_prediction import compute_steps_for_sliding_window
    summary = json.loads(summary_path.read_text())
    settings = summary['settings']
    assert settings['patch_size'] == list(patch)
    mirrors = 2**len(settings['mirroring_axes']) if settings['mirroring'] else 1
    rows = []
    for case in summary['cases']:
        path = data_dir / (case['case'] + '.b2nd')
        shape = list(blosc2.open(str(path), mode='r').shape)  # metadata only
        assert shape[0] == 4
        padded = [max(n, p) for n, p in zip(shape[1:], patch)]
        steps = compute_steps_for_sliding_window(padded, patch, settings['tile_step_size'])
        tiles = math.prod(len(s) for s in steps)
        rows.append(dict(case=case['case'], input_shape=shape, steps=steps,
                         tiles=tiles, forwards=tiles*mirrors,
                         macs=patch_macs*tiles*mirrors,
                         gmac=patch_macs*tiles*mirrors/1e9))
    return dict(summary=str(summary_path), settings=settings, mirrors_per_tile=mirrors,
                case_count=len(rows), tile_count_histogram=dict(Counter(r['tiles'] for r in rows)),
                min_gmac=min(r['gmac'] for r in rows), max_gmac=max(r['gmac'] for r in rows),
                mean_gmac=statistics.mean(r['gmac'] for r in rows),
                median_gmac=statistics.median(r['gmac'] for r in rows), cases=rows,
                scope='Model MACs times geometric forward count; excludes tiling, blending, export and postprocessing')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, default=Path('SMMLIF_training_DS_01_29077508.out'))
    parser.add_argument('--checkpoint', type=Path, help='Trusted local checkpoint; load weights strictly')
    parser.add_argument('--saved-plans', type=Path, help='Compare saved plans with log')
    parser.add_argument('--summary', type=Path, help='Saved validation summary for volume geometry')
    parser.add_argument('--data-dir', type=Path, help='Preprocessed .b2nd directory')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path, default=Path('analysis_results/smmlif_29077508/profile.json'))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    plans = read_logged_plans(args.log)
    if args.saved_plans:
        saved = json.loads(args.saved_plans.read_text())
        assert saved['snn_config'] == plans['snn_config'], 'Saved configuration differs from log'
    config = SNNConfig.from_plans(plans)
    assert config.model_name == 'paper_unet'
    patch = tuple(config.validation_patch_size)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(config.deterministic)
    model = SNNnnUNetAdapter(config).to(args.device).eval()
    checkpoint_info = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        assert checkpoint['snn_config'] == plans['snn_config']
        model.load_state_dict(checkpoint['network_weights'], strict=True)
        checkpoint_info = dict(path=str(args.checkpoint), epoch=checkpoint['current_epoch'],
                               mirroring_axes=list(checkpoint['inference_allowed_mirroring_axes']))
        del checkpoint
    ledger = estimate_cost(model.core, patch, batch=1)
    x = torch.randn(1, config.num_input_channels, *patch, device=args.device)
    audit = OperationAudit()
    handles = []
    for name, module in model.named_modules():
        handles.append(module.register_forward_pre_hook(audit.enter(name)))
        handles.append(module.register_forward_hook(audit.leave))
    print(f'Executing eval/no_grad forward {tuple(x.shape)} on {args.device}', flush=True)
    start = time.perf_counter()
    try:
        with torch.no_grad(), audit:
            output = model(x)
    finally:
        for handle in handles:
            handle.remove()
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    observed = sum(row['macs'] for row in audit.convolutions.values())
    assert observed == ledger['dense_equivalent_macs'], (observed, ledger['dense_equivalent_macs'])
    assert output.shape == (1, config.num_output_channels, *patch)
    assert torch.isfinite(output).all()
    assert not any('auxiliary_heads' in name for name in audit.convolutions)
    for row in ledger['ssn']:
        prefix = 'core.' + row['name'] + '.ssn.'
        for field in ('spatial_depthwise', 'readout_gate', 'memory_projection'):
            assert audit.convolutions[prefix+field]['macs'] == row[field+'_macs']
    report = dict(log=str(args.log), log_sha256=hashlib.sha256(args.log.read_bytes()).hexdigest(),
                  torch_version=torch.__version__, device=args.device, threads=args.threads,
                  input_kind='seeded FP32 Gaussian synthetic patch', seed=config.seed,
                  checkpoint=checkpoint_info, logged_snn_config=plans['snn_config'],
                  effective_snn_config=config.to_dict(), input_shape=list(x.shape),
                  core_input_shape=[1, patch[0], 4, patch[1], patch[2]],
                  output_shape=list(output.shape), eval=True, no_grad=True,
                  state_path='t0=0, zero memories; 81 consecutive slices; diagnostics off; no auxiliary heads',
                  convention='Dense convolution MACs; 1 MAC=multiply-accumulate pair; padded ordinary conv taps; transpose valid emitted products; no sparsity discount',
                  total_macs=observed, total_gmac=observed/1e9,
                  actual_convolution_calls=sum(r['calls'] for r in audit.convolutions.values()),
                  instrumented_forward_seconds=elapsed,
                  analytical_equals_observed=True, analytical_ledger=ledger,
                  observed_convolutions=audit.convolutions, observed_non_mac_aten=audit.non_mac,
                  exclusions=['elementwise state updates, soma charge/reset/threshold',
                              'bias additions, nonlinearities, coefficient transforms, kernel normalization',
                              'GroupNorm, max pooling, dropout, concatenation, padding, tensor movement',
                              'pre/postprocessing, Gaussian blending, backward and optimizer'])
    if args.summary:
        if args.data_dir is None:
            parser.error('--summary requires --data-dir')
        report['volume_costs'] = volume_costs(args.summary, args.data_dir, patch, observed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(macs=observed, gmac=observed/1e9,
                         convolution_calls=report['actual_convolution_calls'],
                         analytical_equals_observed=True, output=str(args.output)), indent=2))


if __name__ == '__main__':
    main()
