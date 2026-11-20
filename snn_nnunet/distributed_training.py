"""One nnU-Net training worker per GPU, launched exclusively by Accelerate.

Preparation and final custom validation live in the parent CLI. Workers reuse
nnU-Net's DDP pipeline without calling its multiprocessing launcher again.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import os
from pathlib import Path
import sys


def _redirect_worker_logs():
    """Keep the primary worker on Slurm streams; retain other ranks' diagnostics."""
    rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
    if rank == 0:
        return
    folder = Path(os.environ['nnUNet_results']) / 'worker_logs'
    folder.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get('SLURM_JOB_ID') or f'local_{os.getppid()}'
    prefix = folder / f'job_{job_id}_rank{rank}'
    # Replace descriptors as well as Python output destinations, so native
    # libraries and child workers inherit the same logs. Append across restarts.
    for descriptor, stream, extension in ((1, sys.stdout, 'out'), (2, sys.stderr, 'err')):
        stream.flush()
        with Path(f'{prefix}.{extension}').open('a', encoding='utf-8') as target:
            os.dup2(target.fileno(), descriptor)
        stream.reconfigure(line_buffering=True)


def main(argv=None):
    # Run before imports that can emit initialization messages or tracebacks.
    _redirect_worker_logs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-id', required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--continue-training', action='store_true')
    parser.add_argument('--save-probabilities', action='store_true')
    parser.add_argument('--skip-final-validation', action='store_true')
    args = parser.parse_args(argv)

    import torch
    import torch.distributed as dist

    from snn_nnunet.cli import configure_runtime
    configure_runtime()
    from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint
    from snn_nnunet.cli import CONFIGURATION, PLANS, TRAINER

    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        # Full-fold validation can take hours while the other ranks wait.
        dist.init_process_group('nccl', timeout=timedelta(hours=48))
    try:
        trainer = get_trainer_from_args(
            args.dataset_id, CONFIGURATION, args.fold, TRAINER, PLANS,
            args.continue_training, device=torch.device('cuda', local_rank),
        )
        maybe_load_checkpoint(trainer, args.continue_training, False)
        trainer.run_training()
        if not args.skip_final_validation:
            trainer.perform_actual_validation(args.save_probabilities)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
