"""Continuous LR extensions and periodic full-volume validation of the patch best."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import csv
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch


def _rates(schedule: dict, epoch: int) -> list[float]:
    start, end = schedule['start_epoch'], schedule['end_epoch']
    progress = min(1.0, max(0.0, (epoch - start) / (end - start)))
    return [rate * (1 - progress) ** schedule['exponent'] for rate in schedule['start_lrs']]


def continuation_schedule(checkpoint: dict, epochs: int, previous_report: dict) -> dict:
    """Retain an existing continuation or stretch it from its current LR."""
    current = checkpoint['current_epoch']
    existing = checkpoint.get('resume_lr_schedule') or previous_report.get('lr_schedule')
    if existing and existing['end_epoch'] == epochs:
        return deepcopy(existing)
    if existing:
        rates = _rates(existing, current)
    elif previous_report.get('scheduler_max_steps', 0) > current:
        old_end = previous_report['scheduler_max_steps']
        rates = [group.get('initial_lr', 0.01) * (1 - current / old_end) ** 0.9
                 for group in checkpoint['optimizer_state']['param_groups']]
    else:
        rates = [group['lr'] for group in checkpoint['optimizer_state']['param_groups']]
    return {'start_epoch': current if current < epochs else max(0, epochs - 1),
            'end_epoch': epochs, 'start_lrs': rates, 'exponent': 0.9}


class ContinuationScheduler:
    """Use global epochs while preserving the LR at an extension's start."""
    def __init__(self, optimizer, schedule: dict):
        if schedule['end_epoch'] <= schedule['start_epoch']:
            raise ValueError('The schedule must have a positive duration')
        if len(schedule['start_lrs']) != len(optimizer.param_groups):
            raise ValueError('Schedule and optimizer parameter groups differ')
        if any(not math.isfinite(rate) or rate < 0 for rate in schedule['start_lrs']):
            raise ValueError('Learning rates must be finite and nonnegative')
        self.optimizer = optimizer
        self.schedule = deepcopy(schedule)
        self.max_steps = schedule['end_epoch']
        self._last_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, current_step):
        self._last_lr = _rates(self.schedule, current_step)
        for group, rate in zip(self.optimizer.param_groups, self._last_lr):
            group['lr'] = rate

    def get_last_lr(self):
        return self._last_lr


def _atomic_checkpoint(path: Path, checkpoint: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix='.resume-', suffix='.pth', dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def install_continuation_scheduler(trainer, schedule: dict) -> None:
    trainer.lr_scheduler = ContinuationScheduler(trainer.optimizer, schedule)
    original_save = trainer.save_checkpoint
    def save_with_schedule(filename):
        original_save(filename)
        if getattr(trainer, 'disable_checkpointing', False) or getattr(trainer, 'local_rank', 0) != 0:
            return
        path = Path(filename)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        checkpoint['resume_lr_schedule'] = deepcopy(schedule)
        _atomic_checkpoint(path, checkpoint)
    trainer.save_checkpoint = save_with_schedule


def save_completed_checkpoint(trainer, path: Path) -> None:
    """Save at an already incremented completed-epoch boundary."""
    completed = trainer.current_epoch
    trainer.current_epoch = completed - 1
    try:
        trainer.save_checkpoint(str(path))
    finally:
        trainer.current_epoch = completed


@contextmanager
def _preserve_training_state(trainer):
    modes = [(module, module.training) for module in trainer.network.modules()]
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if trainer.device.type == 'cuda' else None
    try:
        yield
    finally:
        for module, mode in modes:
            module.training = mode
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
            torch.cuda.empty_cache()


@contextmanager
def _temporary_network_weights(network, weights):
    """Swap only model tensors; never restore optimizer, logger or training epoch."""
    from torch._dynamo import OptimizedModule

    module = network._orig_mod if isinstance(network, OptimizedModule) else network
    original = module.state_dict()
    saved = original.copy()
    if hasattr(original, '_metadata'):
        saved._metadata = deepcopy(original._metadata)
    for name, tensor in original.items():
        saved[name] = (tensor.detach().to('cpu', copy=True) if isinstance(tensor, torch.Tensor)
                       else deepcopy(tensor))
    # Live nonpersistent buffers are absent from state_dict, but inference may
    # mutate them or replace them with different shapes.
    transient_buffers = [
        (child, name, child._buffers[name],
         child._buffers[name].detach().to('cpu', copy=True)
         if isinstance(child._buffers[name], torch.Tensor) else None)
        for child in module.modules() for name in child._non_persistent_buffers_set
        if name in child._buffers
    ]
    normalized = {
        (name[7:] if name not in original and name.startswith('module.') else name): tensor
        for name, tensor in weights.items()
    }
    try:
        module.load_state_dict(normalized, strict=True)
        yield
    finally:
        # load_state_dict can partially update weights before reporting a key or
        # shape mismatch. Restore even when loading or inference raises.
        module.load_state_dict(saved, strict=True)
        for child, name, buffer, value in transient_buffers:
            if buffer is not None:
                buffer.copy_(value)
            child._buffers[name] = buffer


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


class PeriodicFullValidation:
    def __init__(self, trainer, every: int, validator=None, patch_size=None):
        if type(every) is not int or every <= 0:
            raise ValueError('Validation interval must be a positive integer')
        self.trainer = trainer
        self.every = every
        self.validator = validator
        if patch_size is None:
            patch_size = getattr(getattr(trainer, 'snn_config', None),
                                 'validation_patch_size', (128, 128, 128))
        if (not isinstance(patch_size, (list, tuple)) or len(patch_size) != 3
                or any(type(value) is not int or value <= 0 for value in patch_size)):
            raise ValueError('Validation patch must contain three positive integers')
        self.patch_size = tuple(patch_size)
        dataset_base = getattr(trainer, 'preprocessed_dataset_folder_base', None)
        patient_wise = (dataset_base is not None and
                        (Path(dataset_base) / 'patient_partition.json').is_file())
        # Patient-wise runs monitor their outer test; legacy runs have no
        # separately defined test cohort and retain full-fold validation.
        self.subset = 'test' if patient_wise else 'val'
        self.role = 'monitoring_test' if patient_wise else 'internal_validation'
        # Keep both the historical current-epoch evaluations and the earlier
        # best-patch internal-validation protocol distinct from test monitoring.
        directory = 'testing_periodic_best_patch' if patient_wise else 'validation_periodic_best_patch'
        self.root = Path(trainer.output_folder) / directory

    def validate_current(self):
        trainer = self.trainer
        epoch = trainer.current_epoch
        if epoch <= 0 or epoch % self.every:
            return
        distributed = getattr(trainer, 'is_ddp', False)
        if not distributed:
            return self._validate_current_rank_zero()

        # All ranks reach the completed epoch before rank zero reads weights.
        torch.distributed.barrier()
        status = [None]
        if torch.distributed.get_rank() == 0:
            try:
                self._validate_current_rank_zero()
            except Exception as error:
                status[0] = f'{type(error).__name__}: {error}'
        # This collective also holds other ranks until validation is complete.
        torch.distributed.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise RuntimeError(f'Periodic full-fold validation failed: {status[0]}')

    def _validate_current_rank_zero(self):
        trainer = self.trainer
        epoch = trainer.current_epoch
        if epoch <= 0 or epoch % self.every:
            return
        best_path = Path(trainer.output_folder) / 'checkpoint_best.pth'
        if not best_path.is_file():
            raise FileNotFoundError(f'Periodic validation requires patch-validation best: {best_path}')
        selected = torch.load(best_path, map_location='cpu', weights_only=False)
        checkpoint_epoch = selected['current_epoch']
        if type(checkpoint_epoch) is not int or not 1 <= checkpoint_epoch <= epoch:
            raise ValueError('Patch-best checkpoint epoch must be within completed training epochs')
        patch_ema = float(selected['_best_ema'])
        destination = self.root / f'epoch_{epoch:04d}'
        summary_path = destination / 'summary.json'
        if summary_path.is_file():
            try:
                saved = json.loads(summary_path.read_text())
            except json.JSONDecodeError:
                saved = {}
            if (saved.get('epoch') == epoch
                    and saved.get('checkpoint_selection') == 'best_patch_ema'
                    and saved.get('checkpoint_epoch') == checkpoint_epoch
                    and saved.get('patch_validation_ema') == patch_ema
                    and saved.get('settings', {}).get('subset') == self.subset
                    and Path(saved.get('checkpoint', '')).is_file()
                    and tuple(saved.get('settings', {}).get('patch_size', ())) == self.patch_size):
                self._update_history()
                return
        # Retain a recovery point for the current optimizer independently of
        # the older patch-selected checkpoint used exclusively for inference.
        save_completed_checkpoint(trainer, Path(trainer.output_folder) / f'checkpoint_epoch_{epoch:04d}.pth')
        destination.mkdir(parents=True, exist_ok=True)
        checkpoint_path = destination / 'checkpoint_best_patch.pth'
        _atomic_checkpoint(checkpoint_path, selected)
        validator = self.validator
        if validator is None:
            from snn_nnunet.custom_full_validation import run_fold_validation
            validator = run_fold_validation
        trainer.print_to_log_file(
            f'Full-volume validation with postprocessing: completed epoch {epoch}; '
            f'evaluating checkpoint_best.pth from completed epoch {checkpoint_epoch} '
            f'(patch EMA pseudo Dice={patch_ema:.6f}); subset={self.subset}; role={self.role}'
        )
        with _preserve_training_state(trainer), torch.no_grad():
            network = trainer.network
            if isinstance(network, torch.nn.parallel.DistributedDataParallel):
                # A rank-zero-only DDP forward would wait for buffer broadcasts.
                trainer.network = network.module
            try:
                with _temporary_network_weights(trainer.network, selected['network_weights']):
                    summary = validator(trainer, output_dir=destination,
                                        patch_size=self.patch_size, subset=self.subset)
            finally:
                trainer.network = network
        summary.setdefault('settings', {})['patch_size'] = list(self.patch_size)
        summary['settings']['subset'] = self.subset
        summary['evaluation_role'] = self.role
        summary['epoch'] = epoch
        summary['checkpoint'] = str(checkpoint_path)
        summary['checkpoint_source'] = str(best_path)
        summary['checkpoint_epoch'] = checkpoint_epoch
        summary['checkpoint_selection'] = 'best_patch_ema'
        summary['patch_validation_ema'] = patch_ema
        _write_json(summary_path, summary)
        self._update_history()
        trainer.print_to_log_file(
            f"Epoch {epoch} full-volume Dice: raw={summary['mean']['raw']['mean']:.6f}; "
            f"postprocessed={summary['mean']['postprocessed']['mean']:.6f}"
        )
        if 'mean_per_patient' in summary:
            trainer.print_to_log_file(
                f"Epoch {epoch} full-volume Dice per patient: "
                f"raw={summary['mean_per_patient']['raw']['mean']:.6f}; "
                f"postprocessed={summary['mean_per_patient']['postprocessed']['mean']:.6f}"
            )

    def _update_history(self):
        summaries = []
        dataset_base = getattr(self.trainer, 'preprocessed_dataset_folder_base', None)
        patient_wise = (dataset_base is not None and
                        (Path(dataset_base) / 'patient_partition.json').is_file())
        for path in sorted(self.root.glob('epoch_*/summary.json')):
            try:
                summary = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if 'epoch' in summary:
                if patient_wise and 'mean_per_patient' not in summary:
                    # Upgrade saved volume scores without reading images or
                    # running inference again. Retain epoch/checkpoint/settings.
                    from snn_nnunet.custom_full_validation import summarize_cases, write_patient_dice_tables
                    summary.update(summarize_cases(summary['cases'], patient_wise=True))
                    _write_json(path, summary)
                    write_patient_dice_tables(summary, path.parent)
                summaries.append(summary)
        if not summaries:
            return
        columns = ['epoch', 'cases', 'raw_WT', 'raw_TC', 'raw_ET', 'raw_mean',
                   'postprocessed_WT', 'postprocessed_TC', 'postprocessed_ET', 'postprocessed_mean', 'checkpoint']
        columns += ['checkpoint_epoch', 'checkpoint_selection', 'patch_validation_ema', 'checkpoint_source',
                    'subset', 'evaluation_role']
        columns += ['patients'] + [f'{stage}_per_{unit}_{region}'
                                  for unit in ('volume', 'patient')
                                  for stage in ('raw', 'postprocessed')
                                  for region in ('WT', 'TC', 'ET', 'mean')]
        destination = self.root / 'dice_history.csv'
        temporary = destination.with_suffix('.csv.tmp')
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for summary in summaries:
                row = {'epoch': summary['epoch'], 'cases': len(summary['cases']), 'checkpoint': summary['checkpoint']}
                row.update({key: summary.get(key, '') for key in
                            ('checkpoint_epoch', 'checkpoint_selection', 'patch_validation_ema',
                             'checkpoint_source', 'evaluation_role')})
                row['subset'] = summary.get('settings', {}).get('subset', self.subset)
                for stage in ('raw', 'postprocessed'):
                    row.update({f'{stage}_{region}': summary['mean'][stage][region]
                                for region in ('WT', 'TC', 'ET', 'mean')})
                    volumes = summary.get('mean_per_volume', summary['mean'])
                    row.update({f'{stage}_per_volume_{region}': volumes[stage][region]
                                for region in ('WT', 'TC', 'ET', 'mean')})
                    if 'mean_per_patient' in summary:
                        row.update({f'{stage}_per_patient_{region}': summary['mean_per_patient'][stage][region]
                                    for region in ('WT', 'TC', 'ET', 'mean')})
                if 'mean_per_patient' in summary:
                    row['patients'] = summary['counts']['patients']
                writer.writerow(row)
        os.replace(temporary, destination)
        # Monitoring test scores never select a checkpoint. The global patch
        # EMA remains the sole criterion for checkpoint_best.pth.
        if self.subset == 'test':
            return
        best = max(summaries, key=lambda s: s['mean']['postprocessed']['mean'])
        best_checkpoint = Path(self.trainer.output_folder) / 'checkpoint_best_postprocessed.pth'
        temporary_checkpoint = best_checkpoint.with_suffix('.pth.tmp')
        shutil.copy2(best['checkpoint'], temporary_checkpoint)
        os.replace(temporary_checkpoint, best_checkpoint)
        _write_json(self.root / 'best_postprocessed.json', {
            'epoch': best['epoch'], 'mean': best['mean'], 'checkpoint': str(best_checkpoint),
            **{key: best[key] for key in ('checkpoint_epoch', 'checkpoint_selection', 'patch_validation_ema',
                                         'checkpoint_source') if key in best},
            **{key: best[key] for key in ('mean_per_volume', 'mean_per_patient', 'counts') if key in best},
        })


def install_periodic_validation(trainer, every: int, validator=None, patch_size=None) -> PeriodicFullValidation:
    controller = PeriodicFullValidation(trainer, every, validator, patch_size)
    existing = getattr(trainer, '_periodic_validation', None)
    if existing is not None:
        existing.every = controller.every
        existing.patch_size = controller.patch_size
        existing.subset = controller.subset
        existing.role = controller.role
        existing.root = controller.root
        if validator is not None:
            existing.validator = validator
        return existing
    trainer._periodic_validation = controller
    original_epoch_end = trainer.on_epoch_end
    def after_epoch():
        original_epoch_end()
        controller.validate_current()
    trainer.on_epoch_end = after_epoch
    return controller
