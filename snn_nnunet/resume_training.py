"""Extend one saved SNN fold using its checkpoint and global polynomial schedule."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import fcntl
import json
import os
import pickle
import warnings
from pathlib import Path
import shutil
import tempfile

import torch
import yaml


def extended_checkpoint(checkpoint: dict, epochs: int) -> dict:
    """Change only the training horizon; retain learned and logged state."""
    if type(epochs) is not int or epochs < checkpoint['current_epoch']:
        raise ValueError('Target epochs must be an integer at least the checkpoint epoch')
    result = deepcopy(checkpoint)
    result['init_args']['plans']['snn_config']['num_epochs'] = epochs
    return result


def load_resume_checkpoint(fold_dir: Path) -> tuple[Path, dict]:
    """A previous final checkpoint must not hide a more recent interrupted run."""
    candidates = []
    paths = [fold_dir / 'checkpoint_final.pth', fold_dir / 'checkpoint_latest.pth',
             *sorted(fold_dir.glob('checkpoint_epoch_*.pth'))]
    for path in paths:
        if path.is_file():
            try:
                saved = torch.load(path, map_location='cpu', weights_only=False)
                required = {'current_epoch', 'network_weights', 'optimizer_state', 'init_args', 'fptt_state'}
                if not isinstance(saved, dict) or not required <= saved.keys():
                    raise ValueError('Missing training state')
            except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as error:
                warnings.warn(f'Skipping incomplete checkpoint {path}: {error}')
                continue
            candidates.append((path, saved))
    if not candidates:
        raise FileNotFoundError(f'No resumable checkpoint in {fold_dir}; refusing a fresh run')
    return max(candidates, key=lambda item: (item[1]['current_epoch'],
                                             bool(item[1].get('resume_lr_schedule')),
                                             item[0].name != 'checkpoint_latest.pth'))


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.resume.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def _validate_resume_plans(saved_plans: dict, checkpoint_plans: dict) -> None:
    """Allow an epoch extension while checking the checkpoint's saved configuration."""
    comparison = deepcopy(saved_plans)
    comparison['snn_config']['num_epochs'] = checkpoint_plans['snn_config'].get('num_epochs', 300)
    original_plans = deepcopy(checkpoint_plans)
    from snn_nnunet.network_adapter import SNNConfig

    # Legacy checkpoints omit newer optional fields; compare their effective values.
    for plans in (comparison, original_plans):
        plans.pop('continue_training', None)
        plans['snn_config'] = SNNConfig.from_plans(plans).to_dict()
    if comparison != original_plans:
        raise ValueError('Checkpoint plans differ from saved model plans')


def _resume(run_dir: Path, epochs: int, fold: int, check_only: bool,
            lr_mode: str | None = None, validation_every: int | None = None) -> None:
    from snn_nnunet.trainer import nnUNetTrainerSNNFPTT

    matches = sorted(run_dir.glob('Dataset*/nnUNetTrainerSNNFPTT__SNNPlans__3d_fullres'))
    if len(matches) != 1:
        raise ValueError(f'Expected one saved SNN model in {run_dir}; found {len(matches)}')
    model_dir = matches[0]
    checkpoint_path, original = load_resume_checkpoint(model_dir / f'fold_{fold}')
    report_path = run_dir / 'resume_config.json'
    previous_report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    lr_mode = lr_mode or ('continuous' if original.get('resume_lr_schedule') or previous_report.get('lr_schedule') else 'global')
    saved_plans = json.loads((model_dir / 'plans.json').read_text())
    validation_every = (previous_report.get('validation_every',
                        saved_plans['snn_config'].get('validation_every', 0))
                        if validation_every is None else validation_every)
    if validation_every < 0:
        raise ValueError('Validation interval must be nonnegative')
    # Extending num_epochs is allowed; changing the architecture or data plans is not.
    _validate_resume_plans(saved_plans, original['init_args']['plans'])
    if original['init_args']['fold'] != fold:
        raise ValueError('Checkpoint fold does not match the requested fold')
    checkpoint = extended_checkpoint(original, epochs)
    if not check_only and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for training; use --check-only for CPU verification')
    device = torch.device('cpu' if check_only else 'cuda')
    plans = deepcopy(checkpoint['init_args']['plans'])
    plans['continue_training'] = True

    with tempfile.TemporaryDirectory(prefix='snn-resume-check-') as temporary:
        # Native trainer initialization writes logs; isolate all check-only outputs.
        os.environ['nnUNet_results'] = temporary if check_only else str(run_dir)
        trainer = nnUNetTrainerSNNFPTT(
            plans=plans, configuration=checkpoint['init_args']['configuration'],
            fold=fold, dataset_json=checkpoint['init_args']['dataset_json'], device=device,
        )
        trainer.load_checkpoint(checkpoint)
        trainer.validation_every = validation_every
        trainer.my_init_kwargs['device'] = device
        schedule = None
        if lr_mode == 'continuous':
            from snn_nnunet.resume_validation import continuation_schedule, install_continuation_scheduler
            schedule = continuation_schedule(original, epochs, previous_report)
            install_continuation_scheduler(trainer, schedule)
        # nnU-Net applies its scheduler at the start of each epoch with this global index.
        trainer.lr_scheduler.step(trainer.current_epoch)
        report = {
            'source_checkpoint': str(checkpoint_path),
            'completed_epochs': trainer.current_epoch,
            'next_epoch': trainer.current_epoch + 1 if trainer.current_epoch < epochs else None,
            'target_epochs': trainer.num_epochs,
            'remaining_epochs': trainer.num_epochs - trainer.current_epoch,
            'next_learning_rates': [group['lr'] for group in trainer.optimizer.param_groups],
            'scheduler': type(trainer.lr_scheduler).__name__,
            'lr_mode': lr_mode,
            'lr_schedule': schedule,
            'validation_every': validation_every,
            'scheduler_max_steps': trainer.lr_scheduler.max_steps,
            'optimizer_states': len(trainer.optimizer.state),
            'check_only': check_only,
        }
        print(json.dumps(report, indent=2), flush=True)
        if check_only:
            return

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        backup = run_dir / 'resume_backups' / f'epoch_{trainer.current_epoch}_to_{epochs}_{stamp}'
        backup.mkdir(parents=True)
        shutil.copytree(model_dir, backup / model_dir.relative_to(run_dir))
        for name in ('experiment.yaml', 'run_config.json', 'resume_config.json'):
            if (run_dir / name).is_file():
                shutil.copy2(run_dir / name, backup / name)
        report['backup'] = str(backup)
        _write_json(run_dir / 'resume_config.json', report)
        config_path = run_dir / 'run_config.json'
        if config_path.is_file():
            config = json.loads(config_path.read_text())
            config['plans'] = deepcopy(trainer.plans_manager.plans)
            config['snn_config'] = deepcopy(config['plans']['snn_config'])
            config.setdefault('arguments', {}).update(epochs=epochs, continue_training=True)
            config['resume'] = report
            _write_json(config_path, config)
        experiment = run_dir / 'experiment.yaml'
        if experiment.is_file():
            config = yaml.safe_load(experiment.read_text())
            config['snn']['epochs'] = epochs
            temporary_yaml = experiment.with_suffix('.yaml.resume.tmp')
            temporary_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
            os.replace(temporary_yaml, experiment)
        print(f'Original results backed up in: {backup}', flush=True)
        # Baseline validation may run before native on_train_start copies these plans.
        _write_json(model_dir / 'plans.json', trainer.plans_manager.plans)
        periodic = None
        if validation_every:
            from snn_nnunet.resume_validation import install_periodic_validation
            periodic = install_periodic_validation(trainer, validation_every)
            periodic.validate_current()
        if trainer.current_epoch < trainer.num_epochs:
            trainer.run_training()
        else:
            # A boundary checkpoint may precede native on_train_end (e.g. failed validation).
            from snn_nnunet.resume_validation import save_completed_checkpoint
            save_completed_checkpoint(trainer, Path(trainer.output_folder) / 'checkpoint_final.pth')
        trainer.perform_actual_validation()
        if periodic is not None:
            periodic.validate_current()
            return

        from snn_nnunet.custom_full_validation import main as validate
        validate([
            '--dataset-id', str(int(saved_plans['dataset_name'][7:10])),
            '--fold', str(fold), '--results-root', str(run_dir),
            '--output-dir', str(model_dir / f'fold_{fold}' / 'validation_128patch_postprocessed'),
            '--checkpoint-name', 'checkpoint_final.pth',
            '--trainer', 'nnUNetTrainerSNNFPTT', '--plans', 'SNNPlans',
            '--configuration', '3d_fullres',
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, required=True, help='Total epochs, including completed epochs')
    parser.add_argument('--fold', type=int, default=0, choices=range(5))
    parser.add_argument('--lr-mode', choices=('global', 'continuous'),
                        help='Global schedule or a continuous extension; saved continuations are restored by default')
    parser.add_argument('--validation-every', type=int,
                        help='Full-volume validation and postprocessing every N completed epochs; 0 disables')
    parser.add_argument('--check-only', action='store_true', help='Load on CPU in a temporary results root; do not train')
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    for name in ('nnUNet_raw', 'nnUNet_preprocessed'):
        if not os.environ.get(name):
            raise ValueError(f'Set {name} before resuming')
    os.environ['nnUNet_results'] = str(run_dir)
    os.environ.setdefault('nnUNet_compile', 'false')
    if args.check_only:
        _resume(run_dir, args.epochs, args.fold, True, args.lr_mode, args.validation_every)
    else:
        with (run_dir / '.resume.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _resume(run_dir, args.epochs, args.fold, False, args.lr_mode, args.validation_every)


if __name__ == '__main__':
    main()
