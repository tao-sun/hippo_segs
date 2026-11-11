"""Paired compute benchmark; never reads a dataset or updates a checkpoint."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import statistics
import time

import torch

from snn_nnunet.deep_supervision import deep_supervision_loss
from snn_nnunet.network_adapter import SNNConfig, SNNnnUNetAdapter
from snn_nnunet.prepare_plans import DEFAULT_SNN_CONFIG
from snn_nnunet.reproducibility import configure_reproducibility
from snn_nnunet.run_model import _config


def benchmark(config, patch, *, batch_size=2, device='cuda', warmup=3, repeats=10, threads=1):
    """Time identical forward/backward/clip work with and without extra heads.

    Uses one common model, unchanged weights and paired RNG seeds. Excludes
    optimizer steps, data loading, augmentation, FPTT and full-volume validation.
    CUDA timings synchronize and report absolute/above-baseline allocated peaks.
    """
    if config.model_name != 'paper_unet' or config.use_fptt:
        raise ValueError('benchmark requires paper_unet with use_fptt=false')
    if (len(patch) != 3 or any(type(n) is not int or n <= 0 for n in patch)
            or type(batch_size) is not int or batch_size <= 0
            or type(repeats) is not int or repeats < 1
            or type(warmup) is not int or warmup < 0
            or type(threads) is not int or threads < 1):
        raise ValueError('positive patch, batch size, repeats and threads; warmup >= 0 required')
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu with a small --patch-size for a smoke test')
    settings = config.to_dict()
    settings['model_kwargs'] = deepcopy(settings['model_kwargs'])
    settings['model_kwargs']['deep_supervision']['enabled'] = True
    settings['k'] = patch[config.temporal_axis]
    geometry = {'configurations': {'3d_fullres': {'patch_size': list(patch)}}}
    settings = SNNConfig.from_plans({'snn_config': settings, **geometry})
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        configure_reproducibility(settings.seed, settings.deterministic)
        torch.manual_seed(settings.seed)
        model = SNNnnUNetAdapter(settings).to(device).train()
        data = torch.randn(batch_size, 4, *patch, device=device)
        labels = torch.randint(0, 4, (batch_size, 1, *patch), device=device)
        target = torch.cat((labels > 0, (labels == 1) | (labels == 3), labels == 3), dim=1).float()
        from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
        loss = DC_and_BCE_loss({}, {'batch_dice': False, 'do_bg': True,
                                   'smooth': 1e-5, 'ddp': False})
        weights = settings.model_kwargs['deep_supervision']['weights']
        timings = {'baseline': [], 'deep_supervision': []}
        peaks = {'baseline': [], 'deep_supervision': []}
        increments = {'baseline': [], 'deep_supervision': []}
        auxiliary_shapes = None
        auxiliary_bytes = None

        def synchronize():
            if device.type == 'cuda':
                torch.cuda.synchronize(device)

        def step(enabled, seed):
            nonlocal auxiliary_shapes, auxiliary_bytes
            # Paired seeds keep shared dropout masks identical; no weight updates.
            torch.manual_seed(seed)
            model.zero_grad(set_to_none=True)
            model.core.reset_states()
            synchronize()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
                allocated = torch.cuda.memory_allocated(device)
            start = time.perf_counter()
            output = model(data, return_auxiliary=enabled)
            objective = (deep_supervision_loss(output, target, loss, weights,
                                                 temporal_axis=settings.temporal_axis,
                                                 spatial_stride=settings.model_kwargs['pool']['stride'])
                         if enabled else loss(output, target))
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12, error_if_nonfinite=True)
            model.detach_states()
            synchronize()
            seconds = time.perf_counter() - start
            if enabled:
                auxiliary_shapes = [list(value.shape) for value in output[1:]]
                auxiliary_bytes = sum(value.numel() * value.element_size() for value in output[1:])
            peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
            return seconds, peak, peak - allocated if peak is not None else None

        for index in range(warmup + repeats):
            order = (False, True) if index % 2 == 0 else (True, False)
            for enabled in order:
                seconds, peak, increment = step(enabled, settings.seed + index)
                if index >= warmup:
                    name = 'deep_supervision' if enabled else 'baseline'
                    timings[name].append(seconds)
                    if peak is not None:
                        peaks[name].append(peak)
                        increments[name].append(increment)
        auxiliary = sum(p.numel() for p in model.core.auxiliary_heads.parameters())
        total = sum(p.numel() for p in model.parameters())
        rows = {name: {'median_seconds': statistics.median(values), 'samples_seconds': values,
                       'peak_cuda_allocated_bytes': max(peaks[name], default=None),
                       'peak_cuda_above_baseline_bytes': max(increments[name], default=None)}
                for name, values in timings.items()}
        return {
            'device': str(device), 'hardware': (torch.cuda.get_device_name(device)
                if device.type == 'cuda' else platform.processor() or platform.machine()),
            'torch': torch.__version__, 'threads': threads, 'patch_size': list(patch),
            'batch_size': batch_size, 'warmup_pairs': warmup, 'measured_pairs': repeats,
            'scope': 'synthetic FP32 forward + Dice/BCE + backward + gradient clipping; '
                     'excludes optimizer, loading, augmentation and volume validation',
            'deterministic': settings.deterministic, 'parameters': {
                'baseline': total - auxiliary, 'with_supervision': total, 'auxiliary_added': auxiliary},
            'additional_auxiliary_logits_bytes': auxiliary_bytes,
            'auxiliary_output_shapes': auxiliary_shapes,
            'deep_supervision_resolution': settings.model_kwargs['deep_supervision']['resolution'],
            'deep_supervision_weights': list(weights),
            'loss': 'DC_and_BCE_loss with native MemoryEfficientSoftDiceLoss',
            'batch_dice': False, 'temporal_window': settings.k,
            'window_policy': 'whole patch; overrides k for this compute benchmark',
            'timings': rows,
            'compute_time_change_percent': 100 * (
                rows['deep_supervision']['median_seconds'] / rows['baseline']['median_seconds'] - 1),
        }
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('unet_config.yaml'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--patch-size', type=int, nargs=3)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    experiment = _config(args.config)
    snn = experiment['snn']
    if snn['variant'] != 'paper_unet':
        parser.error('--config must select paper_unet')
    settings = deepcopy(DEFAULT_SNN_CONFIG)
    settings.update(model_name='paper_unet', model_kwargs=snn['paper_unet'],
                    temporal_axis=snn['temporal_axis'], k=snn['k'], use_fptt=snn['use_fptt'],
                    bidirectional=snn['bidirectional'], bidirectional_fusion=snn['bidirectional_fusion'],
                    seed=snn['seed'], deterministic=snn['deterministic'])
    patch = args.patch_size or snn.get('training_patch_size', [81, 128, 128])
    geometry = {'configurations': {'3d_fullres': {'patch_size': patch}}}
    result = benchmark(SNNConfig.from_plans({'snn_config': settings, **geometry}), tuple(patch),
                       batch_size=args.batch_size, device=args.device,
                       warmup=args.warmup, repeats=args.repeats, threads=args.threads)
    report = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + '\n')
    print(report)


if __name__ == '__main__':
    main()
