"""Continuous LR extensions and full-volume validation at completed epoch boundaries."""
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
    mode = trainer.network.training
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if trainer.device.type == 'cuda' else None
    try:
        yield
    finally:
        trainer.network.train(mode)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
            torch.cuda.empty_cache()


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
        self.root = Path(trainer.output_folder) / 'validation_periodic'

    def validate_current(self):
        trainer = self.trainer
        epoch = trainer.current_epoch
        if epoch <= 0 or epoch % self.every:
            return
        destination = self.root / f'epoch_{epoch:04d}'
        summary_path = destination / 'summary.json'
        if summary_path.is_file():
            try:
                saved = json.loads(summary_path.read_text())
            except json.JSONDecodeError:
                saved = {}
            if (saved.get('epoch') == epoch
                    and tuple(saved.get('settings', {}).get('patch_size', ())) == self.patch_size):
                self._update_history()
                return
        checkpoint_path = Path(trainer.output_folder) / f'checkpoint_epoch_{epoch:04d}.pth'
        save_completed_checkpoint(trainer, checkpoint_path)
        validator = self.validator
        if validator is None:
            from snn_nnunet.custom_full_validation import run_fold_validation
            validator = run_fold_validation
        trainer.print_to_log_file(f'Full-volume validation with postprocessing: completed epoch {epoch}')
        with _preserve_training_state(trainer), torch.no_grad():
            summary = validator(trainer, output_dir=destination, patch_size=self.patch_size)
        summary.setdefault('settings', {})['patch_size'] = list(self.patch_size)
        summary['epoch'] = epoch
        summary['checkpoint'] = str(checkpoint_path)
        _write_json(summary_path, summary)
        self._update_history()
        trainer.print_to_log_file(
            f"Epoch {epoch} full-volume Dice: raw={summary['mean']['raw']['mean']:.6f}; "
            f"postprocessed={summary['mean']['postprocessed']['mean']:.6f}"
        )

    def _update_history(self):
        summaries = []
        for path in sorted(self.root.glob('epoch_*/summary.json')):
            try:
                summary = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if 'epoch' in summary:
                summaries.append(summary)
        if not summaries:
            return
        columns = ['epoch', 'cases', 'raw_WT', 'raw_TC', 'raw_ET', 'raw_mean',
                   'postprocessed_WT', 'postprocessed_TC', 'postprocessed_ET', 'postprocessed_mean', 'checkpoint']
        destination = self.root / 'dice_history.csv'
        temporary = destination.with_suffix('.csv.tmp')
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for summary in summaries:
                row = {'epoch': summary['epoch'], 'cases': len(summary['cases']), 'checkpoint': summary['checkpoint']}
                for stage in ('raw', 'postprocessed'):
                    row.update({f'{stage}_{region}': summary['mean'][stage][region]
                                for region in ('WT', 'TC', 'ET', 'mean')})
                writer.writerow(row)
        os.replace(temporary, destination)
        best = max(summaries, key=lambda s: s['mean']['postprocessed']['mean'])
        best_checkpoint = Path(self.trainer.output_folder) / 'checkpoint_best_postprocessed.pth'
        temporary_checkpoint = best_checkpoint.with_suffix('.pth.tmp')
        shutil.copy2(best['checkpoint'], temporary_checkpoint)
        os.replace(temporary_checkpoint, best_checkpoint)
        _write_json(self.root / 'best_postprocessed.json', {
            'epoch': best['epoch'], 'mean': best['mean'], 'checkpoint': str(best_checkpoint),
        })


def install_periodic_validation(trainer, every: int, validator=None, patch_size=None) -> PeriodicFullValidation:
    controller = PeriodicFullValidation(trainer, every, validator, patch_size)
    existing = getattr(trainer, '_periodic_validation', None)
    if existing is not None:
        existing.every = controller.every
        existing.patch_size = controller.patch_size
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
