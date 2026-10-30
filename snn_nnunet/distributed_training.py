"""One nnU-Net training worker per GPU, launched exclusively by Accelerate.

Preparation and final custom validation live in the parent CLI. Workers reuse
nnU-Net's DDP pipeline without calling its multiprocessing launcher again.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import os

import torch
import torch.distributed as dist


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-id', required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--continue-training', action='store_true')
    parser.add_argument('--save-probabilities', action='store_true')
    args = parser.parse_args(argv)

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
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        trainer = get_trainer_from_args(
            args.dataset_id, CONFIGURATION, args.fold, TRAINER, PLANS,
            args.continue_training, device=torch.device('cuda', local_rank),
        )
        maybe_load_checkpoint(trainer, args.continue_training, False)
        trainer.run_training()
        trainer.perform_actual_validation(args.save_probabilities)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
